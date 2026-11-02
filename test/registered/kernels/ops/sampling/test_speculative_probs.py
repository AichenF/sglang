"""Real target-probability semantics, including the existing acceptance consumer.

Run on one allocated H200 with pytest. No reference GPU operation is mocked.
The fusion is partial: selection/softmax/FlashInfer renormalization stay original.
"""
import os
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")
V = 152576
FLAG = "SGLANG_DFLASH_VERIFY_PROBS_FUSION"


@pytest.fixture(scope="module", autouse=True)
def cuda_required():
    assert torch.cuda.is_available(), "Actual CUDA execution required; do not accept skipped tests"
    from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig
    from sglang.srt.runtime_context import SpawnRanks, publish, reset_context
    from sglang.srt.server_args import ServerArgs
    publish(ServerArgs(model_path="dummy", tp_size=1, cuda_graph_config=CudaGraphConfig()),
            role="test", ranks=SpawnRanks(world_rank=0))
    yield
    reset_context()


def api():
    from sglang.srt.speculative import dflash_utils as original
    from sglang.kernels.ops.sampling import speculative_probs as candidate
    return original, candidate


def inputs(bs=1, k=50, p=True, need_k=True, ties=False):
    generator = torch.Generator(device="cuda").manual_seed(4107)
    logits = torch.randn(bs * 8, V, generator=generator, device="cuda")
    if ties:
        logits.round_()  # Top-k membership ties must use the original torch.topk.
    temperatures = torch.linspace(0.6, 1.2, bs, device="cuda").reshape(bs, 1)
    info = SimpleNamespace(temperatures=temperatures,
        top_ks=torch.full((bs,), k, dtype=torch.int32, device="cuda"),
        top_ps=torch.full((bs,), 0.95, device="cuda"),
        need_top_k_sampling=need_k, need_top_p_sampling=p)
    return logits, info


def reference(logits, info, bs, maximum, uniform=None, sparse=True):
    original, _ = api()
    previous = os.environ.get(FLAG)
    os.environ[FLAG] = "0"
    try:
        return original.build_speculative_verify_target_probs(
            next_token_logits=logits, sampling_info=info, bs=bs, draft_token_num=8,
            max_top_k=maximum, uniform_top_k_value=uniform, use_sparse_topk=sparse)
    finally:
        if previous is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = previous


def fused(logits, info, bs, maximum, uniform=None, sparse=True):
    original, candidate = api()
    out = candidate.try_fused_speculative_verify_target_probs(
        next_token_logits=logits, sampling_info=info, bs=bs, draft_token_num=8,
        max_top_k=maximum, uniform_top_k_value=uniform, use_sparse_topk=sparse,
        top_k_renorm=original._dflash_top_k_renorm_prob,
        top_p_renorm=original._dflash_top_p_renorm_prob)
    assert out is not None, "Supported workload silently fell back"
    return out


def exact(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32)), (
        "Partial fusion must preserve probability bytes, including support/ties")


@pytest.mark.parametrize("bs,k,p,need_k,sparse,ties,maximum", [
    (1, 1, True, True, True, True, 1),
    (1, 50, True, True, True, False, 50),
    (4, 50, True, True, True, True, 50),
    (4, 128, False, True, True, False, 128),
    (1, 50, True, True, True, False, None),
    (4, V, True, True, True, False, V),
    (1, V, True, True, True, True, None),
    (4, V, True, False, True, False, None),
    (1, 50, True, True, False, True, 50),
])
def test_probability_bytes_and_rng(bs, k, p, need_k, sparse, ties, maximum):
    logits, info = inputs(bs, k, p, need_k, ties)
    snapshots = [x.clone() for x in (logits, info.temperatures, info.top_ks, info.top_ps)]
    expected = reference(logits, info, bs, maximum, k, sparse)
    rng = torch.cuda.get_rng_state()
    actual = fused(logits, info, bs, maximum, k, sparse)
    exact(actual, expected)
    assert torch.equal(torch.cuda.get_rng_state(), rng)
    for actual_input, before in zip((logits, info.temperatures, info.top_ks, info.top_ps), snapshots):
        assert torch.equal(actual_input, before)


def test_heterogeneous_k_p_and_padded_logits():
    logits, info = inputs(4)
    storage = torch.empty((32, V + 16), device="cuda")
    storage[:, :V].copy_(logits)
    logits = storage[:, :V]
    info.top_ks.copy_(torch.tensor([0, 1, 17, 50], device="cuda", dtype=torch.int32))
    info.top_ps.copy_(torch.tensor([1.0, 0.95, 0.5, 0.1], device="cuda"))
    exact(fused(logits, info, 4, 50), reference(logits, info, 4, 50))


def test_upstream_negative_infinity_masks_are_preserved():
    logits, info = inputs()
    logits[:, 100:] = -float("inf")
    actual = fused(logits, info, 1, 50, 50)
    exact(actual, reference(logits, info, 1, 50, 50))
    assert torch.count_nonzero(actual[..., 100:]) == 0


@pytest.mark.parametrize("k", [50, V])
def test_changed_graph_inputs_and_public_dispatch(k, monkeypatch):
    original, candidate = api()
    real = candidate.try_fused_speculative_verify_target_probs
    seen = []

    def observe(**kwargs):
        result = real(**kwargs)
        seen.append(result)
        return result

    monkeypatch.setattr(candidate, "try_fused_speculative_verify_target_probs", observe)
    logits, info = inputs(1, k)
    monkeypatch.setenv(FLAG, "1")
    kwargs = dict(next_token_logits=logits, sampling_info=info, bs=1,
        draft_token_num=8, max_top_k=k, uniform_top_k_value=k)
    for _ in range(3):
        original.build_speculative_verify_target_probs(**kwargs)
    seen.clear()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = original.build_speculative_verify_target_probs(**kwargs)
    assert len(seen) == 1 and seen[0] is actual, "Public builder bypassed the real fusion"
    for iteration in range(4):
        logits.copy_(logits.roll(137, dims=1))
        info.temperatures.fill_(0.5 + iteration * 0.25)
        info.top_ps.fill_(0.7 + iteration * 0.08)
        expected = reference(logits, info, 1, k, k)
        graph.replay()
        exact(actual, expected)
        assert len(seen) == 1, "Graph replay unexpectedly re-entered Python"


def test_unsupported_dtype_preserves_original_fallback(monkeypatch):
    original, candidate = api()
    logits, info = inputs()
    logits = logits.bfloat16()
    expected = reference(logits, info, 1, 50, 50)
    monkeypatch.setenv(FLAG, "1")
    actual = original.build_speculative_verify_target_probs(
        next_token_logits=logits, sampling_info=info, bs=1, draft_token_num=8,
        max_top_k=50, uniform_top_k_value=50)
    exact(actual, expected)
    assert candidate.try_fused_speculative_verify_target_probs(
        next_token_logits=logits, sampling_info=info, bs=1, draft_token_num=8,
        max_top_k=50, uniform_top_k_value=50, use_sparse_topk=True,
        top_k_renorm=original._dflash_top_k_renorm_prob,
        top_p_renorm=original._dflash_top_p_renorm_prob) is None


def test_acceptance_and_bonus_with_fixed_coins(monkeypatch):
    original, _ = api()
    logits, info = inputs(4)
    probs = reference(logits, info, 4, 50, 50)
    candidates = probs.argmax(-1).roll(1, dims=1).contiguous()
    coins = torch.linspace(0.01, 0.99, 32, device="cuda").reshape(4, 8)
    final = torch.tensor([0.1, 0.3, 0.6, 0.9], device="cuda")
    kwargs = dict(candidates=candidates, next_token_logits=logits, sampling_info=info,
        max_top_k=50, uniform_top_k_value=50, threshold_single=1.0,
        threshold_acc=1.0, uniform_samples=coins,
        uniform_samples_for_final_sampling=final)
    monkeypatch.setenv(FLAG, "0")
    expected = tuple(t.clone() for t in original.compute_dflash_sampling_correct_drafts_and_bonus(**kwargs))
    rng = torch.cuda.get_rng_state()
    monkeypatch.setenv(FLAG, "1")
    actual = original.compute_dflash_sampling_correct_drafts_and_bonus(**kwargs)
    for a, b in zip(actual, expected):
        assert torch.equal(a, b)
    assert torch.equal(torch.cuda.get_rng_state(), rng)
