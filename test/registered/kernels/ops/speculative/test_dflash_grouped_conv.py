"""GPU component tests; these do not certify a model or the TP MegaKernel.

The reference is the unchanged production _grouped_conv AST, including its
torch.compile decorator, loaded without unrelated distributed-model imports.
It uses the normal CUDA inductor backend. Eager execution rounds BF16 after
each operation whereas inductor can fuse them, so eager is not the GPU oracle.
"""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-small")

ROOT = Path(__file__).resolve().parents[5]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def fused():
    path = ROOT / "python/sglang/kernels/ops/speculative/dflash_grouped_conv.py"
    assert path.exists(), "The fused DFlash2 grouped convolution is not implemented"
    spec = importlib.util.spec_from_file_location("dflash_grouped_conv_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.grouped_conv


@pytest.fixture(scope="module")
def reference():
    path = ROOT / "python/sglang/srt/models/dflash.py"
    tree = ast.parse(path.read_text())
    definition = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_grouped_conv"
    )
    namespace = {
        "torch": torch, "F": F, "get_compiler_backend": lambda: "inductor",
        "_is_npu": False,
    }
    # More than the default eight dtype/layout specializations are intentional.
    # Never count an eager fallback after a cache-limit warning as compiled parity.
    options = {"suppress_errors": False}
    limit_name = (
        "recompile_limit" if hasattr(torch._dynamo.config, "recompile_limit")
        else "cache_size_limit"
    )
    options[limit_name] = 64
    if hasattr(torch._dynamo.config, "fail_on_recompile_limit_hit"):
        options["fail_on_recompile_limit_hit"] = True
    with torch._dynamo.config.patch(options):
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
        yield namespace["_grouped_conv"]


def inputs(rows, width, taps, group, dtype, strided=False):
    generator = torch.Generator(device="cuda").manual_seed(413)
    def random(shape):
        return torch.randn(shape, device="cuda", dtype=dtype, generator=generator) * 0.25
    hidden = random((rows, width))
    if strided:
        # Real output-side view from [M, input/output, K, G] projection.
        delta = random((rows, 2, taps, width // group))[:, 1]
    else:
        delta = random((rows, taps, width // group))
    return hidden, delta, random((taps, width))


def oracle(hidden, delta, base, block_size, group_size):
    """Independent per-request causal stencil accumulated in float64 on CPU."""
    x, d, b = (t.detach().cpu().double() for t in (hidden, delta, base))
    out = torch.zeros_like(x)
    for row in range(x.shape[0]):
        for tap in range(min(row % block_size + 1, b.shape[0])):
            coefficients = b[tap] + d[row, tap].repeat_interleave(group_size)
            out[row] += coefficients * x[row - tap]
    return out.to(device=hidden.device, dtype=hidden.dtype)


def close(actual, expected):
    # One final low-precision rounding, plus float32 accumulation error. Absolute
    # tolerances cover cancellation near zero; not a model-level quality budget.
    rtol, atol = {
        torch.bfloat16: (0.008, 0.002),
        torch.float16: (0.001, 0.00025),
        torch.float32: (2e-6, 2e-7),
    }[actual.dtype]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("rows,width,taps,block,group,strided", [
    (24, 6144, 2, 8, 16, True),
    (32, 5120, 2, 16, 16, False),
    (13, 80, 3, 5, 16, True),
    (16, 48, 4, 8, 8, False),
    (16, 80, 1, 8, 16, True),
    (1024, 6144, 2, 8, 16, True),
    (15, 80, 4, 2, 16, True),
])
def test_matches_compiled_production_and_float64(
    fused, reference, dtype, rows, width, taps, block, group, strided
):
    """Catches missing taps, wrong group indexing, tails and coefficient strides."""
    x, d, b = inputs(rows, width, taps, group, dtype, strided)
    if strided:
        assert not d.is_contiguous()
    torch._dynamo.mark_static(x, 1)
    torch._dynamo.mark_static(d, 1)
    torch._dynamo.mark_static(d, 2)
    expected = reference(x, d, b, block, width // group, group, taps)
    actual = fused(x, d, b, block, group)
    assert actual.shape == x.shape and actual.dtype == dtype
    assert actual.device == x.device
    close(actual, expected)
    close(actual, oracle(x, d, b, block, group))


@pytest.mark.parametrize("rows", [0, 1, 2])
def test_empty_and_rows_shorter_than_taps(fused, rows):
    # Upstream F.pad(blocks[:-tap]) is undefined for some M < tap shapes;
    # the standalone primitive explicitly defines these by the causal stencil.
    x, d, b = inputs(rows, 80, 4, 16, torch.bfloat16, True)
    actual = fused(x, d, b, 5, 16)
    assert actual.shape == x.shape and actual.dtype == x.dtype
    assert actual.device == x.device
    close(actual, oracle(x, d, b, 5, 16))


@pytest.mark.parametrize("layout", ["sliced_taps_and_groups", "broadcast_rows"])
def test_general_coefficient_strides(fused, reference, layout):
    """Catches ignoring tap/group strides or assuming a nonzero row stride."""
    rows, width, taps, block, group = 15, 80, 3, 5, 16
    x, d, b = inputs(rows, width, taps, group, torch.bfloat16)
    if layout == "sliced_taps_and_groups":
        storage = torch.full((rows, 2 * taps, 3 * width // group), 9.0,
                             device=x.device, dtype=x.dtype)
        view = storage[:, ::2, ::3]
        view.copy_(d)
        d = view
        assert d.stride(1) > width // group and d.stride(2) == 3
    else:
        d = d[:1].expand(rows, -1, -1)
        assert d.stride(0) == 0
    torch._dynamo.mark_static(x, 1)
    torch._dynamo.mark_static(d, 1)
    torch._dynamo.mark_static(d, 2)
    actual = fused(x, d, b, block, group)
    close(actual, reference(x, d, b, block, width // group, group, taps))
    close(actual, oracle(x, d, b, block, group))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two allocated CUDA GPUs required")
def test_eager_uses_input_device_and_restores_current_device(fused):
    """Catches launching on the current GPU instead of the tensor's GPU."""
    initial_device = torch.cuda.current_device()
    with torch.cuda.device(0):
        x, d, b = inputs(16, 80, 3, 16, torch.bfloat16, True)
    with torch.cuda.device(1):
        actual = fused(x, d, b, 8, 16)
        assert actual.device == x.device == torch.device("cuda", 0)
        assert torch.cuda.current_device() == 1
        close(actual, oracle(x, d, b, 8, 16))
    assert torch.cuda.current_device() == initial_device


def test_no_cross_request_nonfinite_history(fused):
    """Catches loading/multiplying NaN or Inf history across block boundaries."""
    x = torch.ones((16, 48), device="cuda", dtype=torch.float32)
    d = torch.full((16, 3, 3), 0.5, device="cuda")
    b = torch.full((3, 48), 0.5, device="cuda")
    x[6] = float("inf")
    x[7] = float("nan")
    result = fused(x, d, b, 8, 16)
    expected = torch.tensor([1, 2, 3, 3, 3, 3, 3, 3], device="cuda")
    torch.testing.assert_close(result[8:], expected[:, None].expand(8, 48).float())


def test_graph_replay_reads_new_hidden_and_coefficients(fused):
    """Catches host-value capture, stale scratch and non-capture-safe launches."""
    x, d, b = inputs(24, 6144, 2, 16, torch.bfloat16, True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fused(x, d, b, 8, 16)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fused(x, d, b, 8, 16)
    graph.replay()
    before = captured.clone()
    x.mul_(0.5)
    d.add_(0.125)
    b.sub_(0.0625)
    graph.replay()
    close(captured, oracle(x, d, b, 8, 16))
    assert not torch.equal(before, captured)


@pytest.mark.parametrize("invalid", [
    "block", "group", "group_zero", "taps_zero", "taps_limit", "shape", "rank",
    "dtype", "float64", "device", "hidden_stride", "base_stride",
])
def test_rejects_unsupported_inputs_before_launch(fused, invalid):
    x, d, b = inputs(8, 80, 2, 16, torch.bfloat16)
    block, group = 8, 16
    if invalid == "block":
        block = 0
    elif invalid == "group":
        group = 3
    elif invalid == "group_zero":
        group = 0
    elif invalid == "taps_zero":
        d, b = d[:, :0], b[:0]
    elif invalid == "taps_limit":
        x, d, b = inputs(8, 80, 17, 16, torch.bfloat16)
    elif invalid == "shape":
        d = d[:, :, :-1]
    elif invalid == "dtype":
        b = b.float()
    elif invalid == "rank":
        x = x.unsqueeze(0)
    elif invalid == "float64":
        x, d, b = x.double(), d.double(), b.double()
    elif invalid == "device":
        b = b.cpu()
    elif invalid == "hidden_stride":
        x = x.t().contiguous().t()
    elif invalid == "base_stride":
        b = b.t().contiguous().t()
    with pytest.raises((ValueError, TypeError)):
        fused(x, d, b, block, group)


def test_rejects_cpu_tensors(fused):
    x = torch.zeros((8, 16))
    with pytest.raises(ValueError):
        fused(x, torch.zeros((8, 2, 1)), torch.zeros((2, 16)), 8, 16)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
