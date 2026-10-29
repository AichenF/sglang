"""DFlash selector's actual CUDA p/q chain verifier with supplied uniforms.

The worker routes through dspark_accept._accept_sampling_core to this public
chain helper with block_verification=False. Its wrapper draws RNG internally;
these tests inject uniforms at the existing public kernel API, without patches.
This is proposal/acceptance component coverage, not KV commits or model accuracy.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.speculative.reject_sampling import chain_speculative_sampling_triton
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-small")

BLOCK, GAMMA, VOCAB = 8, 7, 8
# Only the FP32 probability-builder has a numeric tolerance. Dyadic chain
# fixtures and uniforms away from non-exact CDF boundaries require exact tokens.
PROB_RTOL, PROB_ATOL = 2e-5, 2e-6


@pytest.fixture(scope="module", autouse=True)
def require_real_cuda():
    assert torch.cuda.is_available(), "Actual CUDA required; never skip"
    assert os.environ.get("TRITON_INTERPRET", "0") != "1", "Interpreter is not CUDA evidence"


def cpu_reference(data):
    p, q = data["target"].cpu().double(), data["draft"].cpu().double()
    candidates = data["candidates"].cpu()
    coins, final_coins = data["coins"].cpu().double(), data["final_coins"].cpu().double()
    batch, slots = candidates.shape
    lengths = torch.zeros(batch, dtype=torch.int32)
    output = torch.full((batch, slots), -1, dtype=torch.int32)
    for row in range(batch):
        accepted = 0
        for step in range(slots - 1):
            token = int(candidates[row, step + 1])
            proposal = float(q[row, step, token])
            if not (0 < proposal <= 1 and float(coins[row, step]) < float(p[row, step, token]) / proposal):
                break
            output[row, accepted] = token
            accepted += 1
        weights = p[row, accepted].clone()
        if accepted < slots - 1:
            weights = (weights - q[row, accepted]).clamp_min(0)
        assert weights.sum() > 0, "Fixture must have a nondegenerate final distribution"
        cdf = weights.cumsum(0) / weights.sum()
        bonus = int((cdf <= final_coins[row]).sum())
        assert bonus < p.shape[-1]
        output[row, accepted] = bonus
        lengths[row] = accepted
    return lengths, output


def device_case(data):
    return {key: tensor.cuda() for key, tensor in data.items()}


def make_storage(data):
    batch, slots = data["candidates"].shape
    return {
        "indices": torch.arange(batch * slots, device="cuda", dtype=torch.int64).flip(0).view(batch, slots),
        "predicts": torch.empty(batch * slots, device="cuda", dtype=torch.int32),
        "accepted": torch.empty(batch, slots, device="cuda", dtype=torch.int32),
        "lengths": torch.empty(batch, device="cuda", dtype=torch.int32),
    }


def launch(data, storage):
    # Clear undefined tails on every replay; the actual kernel owns valid slots.
    storage["predicts"].fill_(-1)
    storage["accepted"].fill_(-1)
    storage["lengths"].fill_(-1)
    chain_speculative_sampling_triton(
        predicts=storage["predicts"], accept_index=storage["accepted"],
        accept_token_num=storage["lengths"], candidates=data["candidates"],
        retrive_index=storage["indices"], retrive_next_token=None, retrive_next_sibling=None,
        uniform_samples=data["coins"], uniform_samples_for_final_sampling=data["final_coins"],
        target_probs=data["target"], draft_probs=data["draft"],
        threshold_single=1.0, threshold_acc=1.0, deterministic=True,
        block_verification=False,
    )


def check(data, storage):
    expected_lengths, expected = cpu_reference(data)
    assert storage["lengths"].is_cuda and storage["predicts"].is_cuda
    torch.testing.assert_close(storage["lengths"].cpu(), expected_lengths, rtol=0, atol=0)
    actual = storage["predicts"][storage["indices"]]
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    valid = torch.arange(BLOCK, device="cuda")[None, :] <= storage["lengths"][:, None]
    torch.testing.assert_close(storage["accepted"][valid].long(), storage["indices"][valid], rtol=0, atol=0)
    assert (storage["accepted"][~valid] == -1).all()
    return actual


def each_rejection_case():
    # q has support {1,3}; rejection must sample token5 outside that support.
    p = torch.zeros(BLOCK, BLOCK, VOCAB)
    q = torch.zeros(BLOCK, GAMMA, VOCAB)
    p[..., 1], p[..., 3], p[..., 5] = 0.375, 0.125, 0.5
    q[..., 1], q[..., 3] = 0.75, 0.25
    coins = torch.full((BLOCK, GAMMA), 0.25)
    for row in range(GAMMA):
        coins[row, row] = 0.5  # strict acceptance boundary: p/q == 0.5
    return {"target": p, "draft": q, "candidates": torch.ones(BLOCK, BLOCK, dtype=torch.int64),
            "coins": coins, "final_coins": torch.full((BLOCK,), 0.125)}


@torch.inference_mode()
def test_reject_each_slot_all_accepted_and_zero_q_outside_support():
    data = device_case(each_rejection_case())
    storage = make_storage(data)
    launch(data, storage)
    output = check(data, storage)
    torch.testing.assert_close(storage["lengths"], torch.arange(BLOCK, device="cuda", dtype=torch.int32))
    assert (output[torch.arange(GAMMA, device="cuda"), torch.arange(GAMMA, device="cuda")] == 5).all()
    assert output[-1, -1].item() == 1, "All-accepted bonus must sample pure target p"


@torch.inference_mode()
def test_corrected_rejection_cdf_differs_from_p_and_pure_p_bonus():
    p = torch.tensor([0.125, 0.125, 0.25, 0.5, 0, 0, 0, 0]).repeat(8, BLOCK, 1)
    q = torch.tensor([0.5, 0.5, 0, 0, 0, 0, 0, 0]).repeat(8, GAMMA, 1)
    coins = torch.cat((torch.full((4, GAMMA), 0.9), torch.full((4, GAMMA), 0.1)))
    data = device_case({"target": p, "draft": q, "candidates": torch.zeros(8, BLOCK, dtype=torch.int64),
                        "coins": coins, "final_coins": torch.tensor([0.01, 0.2, 0.4, 0.9] * 2)})
    storage = make_storage(data)
    launch(data, storage)
    actual = check(data, storage)
    assert actual[:4, 0].tolist() == [2, 2, 3, 3]  # normalized positive(p-q)
    assert actual[4:, -1].tolist() == [0, 1, 2, 3]  # pure p after all seven accept


@torch.inference_mode()
def test_near_zero_positive_proposal_probability():
    tiny = 2.0 ** -30
    p, q = torch.zeros(2, BLOCK, VOCAB), torch.zeros(2, GAMMA, VOCAB)
    q[..., 1], q[..., 3], q[..., 4] = tiny, 0.5, 0.5 - tiny
    p[..., 3], p[..., 5] = 0.25, 0.75
    p[0, :, 1], p[1, :, 1] = 2 * tiny, tiny / 4
    p[:, :, 5] -= p[:, :, 1]
    data = device_case({"target": p, "draft": q, "candidates": torch.ones(2, BLOCK, dtype=torch.int64),
                        "coins": torch.full((2, GAMMA), 0.5), "final_coins": torch.full((2,), 0.125)})
    storage = make_storage(data)
    launch(data, storage)
    check(data, storage)
    assert storage["lengths"].tolist() == [7, 0]


def cpu_filtered_probs(logits, temperatures, top_ks, top_ps):
    out = torch.zeros_like(logits, dtype=torch.float64)
    for row in range(logits.shape[0]):
        for step in range(BLOCK):
            values, ids = (logits[row, step].double() / temperatures[row]).sort(descending=True)
            ids, values = ids[:top_ks[row]], values[:top_ks[row]]
            weights = values.softmax(0)
            # Keep the smallest descending prefix reaching top_p (including
            # the token crossing the threshold), after top-k renormalization.
            keep = min(int((weights.cumsum(0) < top_ps[row]).sum()) + 1, len(ids))
            weights[keep:] = 0
            out[row, step, ids] = weights / weights.sum()
    return out


@pytest.mark.parametrize("sparse", [True, False])
@torch.inference_mode()
def test_actual_target_topk_topp_and_greedy_alone_matches_mixed(sparse):
    from sglang.srt.speculative.dflash_utils import build_speculative_verify_target_probs
    from sglang.kernels.ops.speculative.dspark.dspark_accept import accept_greedy_triton

    logits = torch.tensor([0.9, 0.2, -0.3, 2.0, 1.2, -0.7, 0.5, -1.0]).repeat(3, BLOCK, 1)
    temperatures, top_ks, top_ps = [1.0, 0.7, 1.2], [1, 4, 5], [1.0, 0.75, 0.9]
    info = SimpleNamespace(temperatures=torch.tensor(temperatures, device="cuda")[:, None],
                           top_ks=torch.tensor(top_ks, device="cuda"), top_ps=torch.tensor(top_ps, device="cuda"),
                           need_top_k_sampling=True, need_top_p_sampling=True)
    target = build_speculative_verify_target_probs(next_token_logits=logits.cuda().flatten(0, 1),
        sampling_info=info, draft_token_num=BLOCK, bs=3, max_top_k=5, use_sparse_topk=sparse)
    expected = cpu_filtered_probs(logits, temperatures, top_ks, top_ps)
    torch.testing.assert_close(target.cpu().double(), expected, rtol=PROB_RTOL, atol=PROB_ATOL)
    q = torch.zeros(3, GAMMA, VOCAB)
    q[1:, :, 3], q[1:, :, 4] = 0.6, 0.4
    candidates = torch.full((3, BLOCK), 3, dtype=torch.int64)
    candidates[0, 4] = 1  # one-hot proposal disagrees with greedy target at slot4
    q[0].scatter_(1, candidates[0, 1:, None], 1.0)
    candidates[1:, 2::2] = 4
    data = device_case({"draft": q, "candidates": candidates, "coins": torch.full((3, GAMMA), 0.37),
                        "final_coins": torch.tensor([0.19, 0.43, 0.81])})
    data["target"] = target
    storage = make_storage(data)
    launch(data, storage)
    mixed = check(data, storage)
    alone = {key: value[:1].contiguous() for key, value in data.items()}
    solo = make_storage(alone)
    launch(alone, solo)
    isolated = check(alone, solo)
    torch.testing.assert_close(isolated[0], mixed[0], rtol=0, atol=0)
    greedy_len, greedy_bonus, _ = accept_greedy_triton(candidates=alone["candidates"],
        target_logits=logits[:1].cuda().flatten(0, 1), verify_num_draft_tokens=BLOCK)
    assert greedy_len.item() == solo["lengths"].item() == 3
    assert greedy_bonus.item() == isolated[0, 3].item() == 3


@torch.inference_mode()
def test_chain_graph_replay_changes_probabilities_tokens_uniforms_and_indices():
    base = device_case(each_rejection_case())
    data = {key: value.clone() for key, value in base.items()}
    storage = make_storage(data)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            launch(data, storage)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(data, storage)
    for iteration in range(4):
        data["target"].copy_(base["target"].roll(iteration + 1, -1))
        data["draft"].copy_(base["draft"].roll(iteration + 1, -1))
        data["candidates"].copy_((base["candidates"] + iteration + 1) % VOCAB)
        data["coins"].copy_(base["coins"].roll(iteration, 0))
        data["final_coins"].fill_(0.137 + iteration * 0.1)
        storage["indices"].copy_(storage["indices"].roll(1, 0))
        graph.replay()
        check(data, storage)
