"""The v2 convolution boundary must preserve the rounded TP sum and IPC epochs.

Run on eight allocated Hopper GPUs with:
  torchrun --standalone --nproc_per_node=8 -m pytest <this-file> -x -q
Direct execution uses the repository's torchrun harness. These are component
tests, not evidence of full DFlash2 integration or model accuracy.
"""

from datetime import timedelta
import os

import pytest
import torch
import torch.distributed as dist

from sglang.kernels.ops.communication.dflash_tp_fused import DFlashTPState
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="8-gpu-h200")

H = 6144
WORLD = 8


@pytest.fixture(scope="module")
def state():
    # Fail rather than skip: an all-skipped run cannot certify CUDA IPC.
    assert torch.cuda.is_available(), "Eight allocated CUDA GPUs are required"
    assert int(os.environ.get("WORLD_SIZE", "0")) == WORLD, "Launch with torchrun TP8"
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    assert torch.cuda.get_device_capability()[0] == 9, "Hopper component test"
    dist.init_process_group(backend="gloo", timeout=timedelta(minutes=5))
    ipc = DFlashTPState(torch.device("cuda", local_rank), dist.group.WORLD, list(range(WORLD)), rank)
    yield ipc
    # IPC mappings are owned until worker-process exit, matching the production
    # state lifetime; do not free an export while another rank still maps it.
    dist.destroy_process_group()


def contributions(rows, iteration=0, *, cancellation=False):
    """All ranks derive the same independent CPU input/oracle without NCCL sums."""
    if cancellation:
        # FP32 rank-order sum is 1; regrouping the first four terms can produce 2.
        values = torch.tensor([2**24, 1, -(2**24), 1, 0, 0, 0, 0], dtype=torch.bfloat16)
        data = values[:, None, None].expand(WORLD, rows, H).clone()
    else:
        generator = torch.Generator().manual_seed(313 + iteration)
        data = (torch.randn((WORLD, rows, H), generator=generator) * 0.25).bfloat16()
    # All-zero payload words exercise the transport's signed-zero arrival marker.
    data[:, :, :32] = 0
    result = torch.zeros((rows, H), dtype=torch.float32)
    for rank in range(WORLD):
        result.add_(data[rank].float())
    return data, result.bfloat16()


def exact(actual, expected):
    torch.testing.assert_close(actual, expected.to(actual.device), rtol=0, atol=0)


def assert_epoch_delta(state, before, rows, count):
    expected = before.clone()
    expected[:rows] += count
    exact(state.calls, expected)


@pytest.mark.parametrize("rows", [1, 8, 32])
@pytest.mark.parametrize("cancellation", [False, True])
def test_raw_rank_order_without_input_or_residual_mutation(state, rows, cancellation):
    """Rejects normalization, pre-rounded sums, changed rank order and input writes."""
    for iteration in range(4):
        values, expected = contributions(rows, iteration, cancellation=cancellation)
        partial = values[state.rank].cuda()
        original = partial.clone()
        residual = torch.full_like(partial, 3.5)
        residual_before = residual.clone()
        out = torch.full_like(partial, float("nan"))
        calls = state.calls.clone()
        state.all_reduce_raw(partial, out)
        exact(out, expected)
        exact(partial, original)
        exact(residual, residual_before)
        assert_epoch_delta(state, calls, rows, 1)


@pytest.mark.parametrize("rows", [1, 8, 32])
def test_raw_and_arnq_share_epochs_in_eager_and_graphs(state, rows):
    """Catches stale IPC phases and capture-time counters when raw/arnq alternate."""
    values, _ = contributions(rows)
    partial = values[state.rank].cuda()
    first = torch.empty_like(partial)
    last = torch.empty_like(partial)
    normalized = torch.empty_like(partial)
    residual = torch.empty_like(partial)
    gamma = torch.linspace(0.5, 1.5, H, device="cuda", dtype=torch.bfloat16)
    eps = 1e-6

    def chain():
        state.all_reduce_raw(partial, first)
        state.arnq(partial, residual, gamma, eps, normalized, None)
        state.all_reduce_raw(partial, last)

    def verify(expected, original, before, calls):
        exact(first, expected)
        exact(last, expected)
        exact(partial, original)
        added = expected.cuda().float() + before.float()
        exact(residual, added.bfloat16())
        norm = (added * torch.rsqrt(added.square().mean(-1, keepdim=True) + eps)
                * gamma.float()).bfloat16()
        torch.testing.assert_close(normalized, norm, rtol=0.008, atol=0.008)
        assert_epoch_delta(state, calls, rows, 3)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        residual.zero_()
        for _ in range(3):
            chain()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        chain()
    dist.barrier()

    # Alternate graph and eager operations on the same state. Every pass changes
    # both partials and residual while retaining captured storage addresses.
    for iteration in range(6):
        values, expected = contributions(rows, iteration + 10)
        partial.copy_(values[state.rank])
        residual.fill_(0.125 * (iteration + 1))
        original, before, calls = partial.clone(), residual.clone(), state.calls.clone()
        if iteration % 2:
            chain()
        else:
            graph.replay()
        verify(expected, original, before, calls)


def test_changing_active_rows_preserves_inactive_token_epochs(state):
    """Catches counters advanced for inactive CTAs and stale slots on reactivation."""
    for iteration, rows in enumerate((32, 1, 8, 1, 32, 8, 32)):
        values, expected = contributions(rows, iteration + 30)
        partial = values[state.rank].cuda()
        out = torch.empty_like(partial)
        calls = state.calls.clone()
        state.all_reduce_raw(partial, out)
        exact(out, expected)
        assert_epoch_delta(state, calls, rows, 1)


def test_prepushed_gemm_output_ignores_partial_placeholder(state):
    """Uses the real GEMM IPC producer; catches re-pushing/reading a dummy partial."""
    from sglang.kernels.ops.gemm.skinny_gemm import skinny_fp8_pt_gemm

    rows, inner = 8, 2048
    x = torch.ones((rows, inner), device="cuda", dtype=torch.float8_e4m3fn)
    weight = torch.full((H, inner), state.rank + 1, device="cuda", dtype=torch.float8_e4m3fn)
    x_scale = torch.ones(rows, device="cuda")
    weight_scale = torch.ones(H, device="cuda")
    placeholder = torch.full((rows, H), float("nan"), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(placeholder)
    expected = torch.full_like(out, inner * sum(range(1, WORLD + 1)))
    for _ in range(3):
        calls = state.calls.clone()
        skinny_fp8_pt_gemm(x, x_scale, weight, weight_scale, rs=state.rs())
        state.all_reduce_raw(placeholder, out, prepushed=True)
        exact(out, expected)
        assert torch.isnan(placeholder).all()
        assert_epoch_delta(state, calls, rows, 1)


@pytest.mark.parametrize("invalid", ["dtype", "width", "rows", "partial_stride", "out_stride", "cpu"])
def test_invalid_layout_rejected_before_collective(state, invalid):
    """Malformed arguments must fail before any peer can wait on invalid payloads."""
    partial = torch.zeros((8, H), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(partial)
    if invalid == "dtype":
        partial = partial.half()
    elif invalid == "width":
        partial = partial[:, :-1].contiguous()
    elif invalid == "rows":
        partial = torch.empty((33, H), device="cuda", dtype=torch.bfloat16)
        out = torch.empty_like(partial)
    elif invalid == "partial_stride":
        partial = partial.t().contiguous().t()
    elif invalid == "out_stride":
        out = out.t().contiguous().t()
    else:
        partial, out = partial.cpu(), out.cpu()
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        state.all_reduce_raw(partial, out)


@pytest.mark.parametrize("cpu_tensors", ["all", "calls"])
def test_raw_binding_rejects_cpu_device_before_rank_validation(state, cpu_tensors):
    """The binding itself must require CUDA, even when all devices agree.

    An invalid rank is an independent host-side launch guard: a regression that
    accepts CPU tensors fails this test on the wrong exception without launching
    a kernel with CPU pointers or entering a collective.
    """
    from sglang.kernels.ops.communication.dflash_tp_fused import _module, _pdl

    device = "cpu" if cpu_tensors == "all" else state.device
    partial = torch.zeros((8, H), device=device, dtype=torch.bfloat16)
    out = torch.empty_like(partial)
    calls = torch.zeros_like(state.calls, device="cpu")
    error = (
        "Device value .*not in the allowed options"
        if cpu_tensors == "all"
        else "Device mismatch"
    )
    with pytest.raises((ValueError, TypeError, RuntimeError), match=error):
        _module(_pdl()).arnq(
            partial, partial, partial, out, calls, calls, state.peers,
            0.0, -1, 3, 0, 0,
        )


if __name__ == "__main__":
    from sglang.test.kernels.utils import multigpu_pytest_main

    assert torch.cuda.device_count() >= WORLD, "Eight allocated GPUs required; do not skip"
    multigpu_pytest_main(__name__, __file__, num_gpus=(WORLD,), timeout=900)
