import sys
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.kernels.ops.attention import swa_decode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

requires_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="requires SM90 (Hopper)",
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, None, torch.float16])
def test_boundary_preserves_sink_storage_and_dispatches_by_dtype(dtype):
    q = torch.empty((32, 16, 192), dtype=torch.bfloat16)
    k = torch.empty((512, 1, 192), dtype=q.dtype)
    v = torch.empty((512, 1, 128), dtype=q.dtype)
    pt = torch.empty((4, 128), dtype=torch.int32)
    seq = torch.empty(4, dtype=torch.int32)
    sink = None if dtype is None else torch.empty(16, dtype=dtype)
    module = Mock()
    with (
        patch.object(swa_decode, "_mma_module", return_value=module),
        patch.object(swa_decode, "_swa_decode_attn_kernel") as fallback,
    ):
        swa_decode.swa_decode_attention(
            q,
            k,
            v,
            pt,
            seq,
            window_tokens=8,
            window_size=129,
            sm_scale=192**-0.5,
            sinks=sink,
        )
    if dtype == torch.float16:
        module.swa_decode_mma.assert_not_called()
        assert fallback.__getitem__.return_value.call_args.args[5] is sink
        return
    fallback.__getitem__.assert_not_called()
    args = module.swa_decode_mma.call_args.args
    assert len(args) == 15
    assert args[-2:] == (int(sink is not None), int(dtype == torch.bfloat16))
    assert args[5] is (q if sink is None else sink)


def make_case(lengths, *, sink_dtype=torch.bfloat16):
    torch.manual_seed(83)
    width, heads, dim, vdim = 8, 16, 192, 128
    batch = len(lengths)
    rows = batch * width
    storage = torch.randn((rows, 3392), device="cuda", dtype=torch.bfloat16)
    q = storage[:, : heads * dim].view(rows, heads, dim)
    max_length = max(lengths)
    slots = max(144, batch * max_length + 1)
    k = torch.randn((slots, 1, dim), device="cuda", dtype=q.dtype)
    v = torch.randn((slots, 1, vdim), device="cuda", dtype=q.dtype)
    pt = (torch.randperm(slots - 1, device="cuda") + 1).to(torch.int32)
    pt = pt[: batch * max_length].view(batch, max_length)
    seq = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    sink = torch.linspace(-12, 12, heads, device="cuda", dtype=sink_dtype)
    return q, k, v, pt, seq, sink


def reference(q, k, v, pt, lengths, window, sink):
    result = torch.zeros((q.shape[0], 16, 128), device=q.device, dtype=q.dtype)
    for b, length in enumerate(lengths):
        if length == 0:
            continue
        for t in range(8):
            pos = length - 8 + t
            slots = pt[b, max(0, pos - window + 1) : pos + 1].long()
            scores = q[b * 8 + t].float() @ k[slots, 0].float().T * 192**-0.5
            scores = torch.cat((scores, sink.float()[:, None]), dim=1)
            probabilities = scores.softmax(dim=-1)[:, :-1]
            result[b * 8 + t] = (probabilities @ v[slots, 0].float()).to(q.dtype)
    return result


def fa3_reference(q, k, v, pt, seq, window, sink):
    from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache

    return flash_attn_with_kvcache(
        q=q,
        k_cache=k.unsqueeze(1),
        v_cache=v.unsqueeze(1),
        page_table=pt,
        cache_seqlens=seq,
        cu_seqlens_q=torch.arange(
            0, q.shape[0] + 1, 8, device=q.device, dtype=torch.int32
        ),
        max_seqlen_q=8,
        softmax_scale=192**-0.5,
        causal=True,
        window_size=(window - 1, 0),
        sinks=sink,
        ver=3,
    )


@requires_sm90
@pytest.mark.parametrize("sink_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("window", [128, 129])
@pytest.mark.parametrize(
    "lengths", [[8], [129, 4096], [0, 103, 4096], [8, 17, 129, 4096]]
)
def test_sink_storage_matches_math_and_bf16_fa3(sink_dtype, window, lengths):
    q, k, v, pt, seq, sink = make_case(lengths, sink_dtype=sink_dtype)
    actual = swa_decode.swa_decode_attention(
        q,
        k,
        v,
        pt,
        seq,
        window_tokens=8,
        window_size=window,
        sm_scale=192**-0.5,
        sinks=sink,
    )
    expected = reference(q, k, v, pt, lengths, window, sink)
    torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.02)
    assert torch.isfinite(actual).all()
    if lengths[0] == 0:
        assert torch.count_nonzero(actual[:8]) == 0
    elif sink_dtype == torch.bfloat16:
        fa3 = fa3_reference(q, k, v, pt, seq, window, sink)
        torch.testing.assert_close(actual, fa3, atol=0.005, rtol=0.02)


@requires_sm90
def test_graph_replay_reads_updated_bf16_sinks():
    q, k, v, pt, seq, sink = make_case([2048, 1024, 128, 0])

    def invoke():
        return swa_decode.swa_decode_attention(
            q,
            k,
            v,
            pt,
            seq,
            window_tokens=8,
            window_size=129,
            sm_scale=192**-0.5,
            sinks=sink,
        )

    invoke()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = invoke()
    lengths = [17, 512, 0, 2048]
    seq.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
    q.normal_()
    sink.neg_()
    graph.replay()
    torch.cuda.synchronize()
    expected = reference(q, k, v, pt, lengths, 129, sink)
    torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.02)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
