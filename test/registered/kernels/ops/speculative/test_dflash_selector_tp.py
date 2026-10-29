"""Real TP8 head/selector/Philox graph tests with a synthetic padded vocabulary.

torchrun --standalone --nproc_per_node=8 -m pytest <this-file> -x -q -s
Copy test_dflash_selector_cuda.py alongside this file (shared independent oracle).
This exercises production ParallelLMHead, candidate collectives and worker
sampler. It does not instantiate a backbone, quantized head, or customer model.
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from sglang.srt.distributed import parallel_state as ps
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import publish_build_topology
from test_dflash_selector_cuda import cpu_scores, cpu_walk

register_cuda_ci(est_time=300, stage="base-b-kernel-unit", runner_config="8-gpu-h200")

WORLD, BLOCK, HIDDEN, VOCAB, RANK = 8, 8, 32, 509, 8
# Unique FP32 head logits are exactly representable (one-hot x rational weight).
# Learned selector reductions/exp use the same predeclared FP32 gate as CUDA tests.
Q_RTOL, Q_ATOL = 3e-5, 2e-6


@pytest.fixture(scope="module", autouse=True)
def topology():
    from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig
    from sglang.srt.runtime_context import reset_context
    from sglang.srt.utils import get_compiler_backend
    from torch._dynamo.utils import counters

    assert torch.cuda.is_available(), "TP8 CUDA required, never skip"
    assert int(os.environ.get("WORLD_SIZE", "0")) == WORLD, "Launch with torchrun TP8"
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    assert torch.cuda.get_device_capability()[0] == 9, "Hopper required"
    assert get_compiler_backend() == "inductor"
    for key in ("TORCHDYNAMO_DISABLE", "TORCH_COMPILE_DISABLE"):
        assert os.environ.get(key, "0").lower() in ("", "0", "false"), key
    torch._dynamo.reset()
    counters.clear()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    options = {"suppress_errors": False}
    limit = "recompile_limit" if hasattr(torch._dynamo.config, "recompile_limit") else "cache_size_limit"
    options[limit] = 64
    if hasattr(torch._dynamo.config, "fail_on_recompile_limit_hit"):
        options["fail_on_recompile_limit_hit"] = True
    ps.init_distributed_environment(world_size=WORLD, rank=rank, local_rank=local_rank,
                                    distributed_init_method="env://")
    publish_build_topology(tp_size=WORLD, world_rank=rank, speculative_algorithm="DFLASH",
                           speculative_num_draft_tokens=BLOCK, speculative_eagle_topk=1,
                           cuda_graph_config=CudaGraphConfig())
    ps.initialize_model_parallel()
    try:
        with torch._dynamo.config.patch(options):
            yield ps.get_tp_group()
    finally:
        ps.destroy_model_parallel()
        ps.destroy_distributed_environment()
        reset_context()
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def full_weight(*, ties=False):
    token = torch.arange(VOCAB)[:, None]
    channel = torch.arange(HIDDEN)[None, :]
    # Permutations scatter winners across physical shards. Triples deliberately
    # straddle the global K=4 cutoff in the separate tie test.
    if ties:
        return (((token // 3 * 73 + channel * 41) % 173).float() - 86) / 128
    return (((token * 73 + channel * 41) % VOCAB).float() - 254) / 256


def make_head(weights):
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

    head = ParallelLMHead(VOCAB, HIDDEN, params_dtype=torch.float32, padding_size=64).cuda()
    shard = head.shard_indices
    assert head.org_vocab_size_padded == 512
    assert head.num_embeddings_per_partition == 64
    with torch.no_grad():
        # If padding leaks into logits it overwhelms every legitimate candidate.
        head.weight.fill_(4096)
        head.weight[:shard.num_org_elements].copy_(
            weights[shard.org_vocab_start_index:shard.org_vocab_end_index])
    return head


def assert_rank_agreement(tensor):
    gathered = [torch.empty_like(tensor) for _ in range(WORLD)]
    dist.all_gather(gathered, tensor.contiguous(), group=ps.get_tp_group().device_group)
    for other in gathered:
        torch.testing.assert_close(other, tensor, rtol=0, atol=0)


def make_draft(head, top_k):
    from sglang.srt.models.dflash import CandidateSelector, DFlash2DraftModel
    from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config

    # Deliberately construct only the production selector/head submodules. No
    # fake compute_candidates, fake collective, or hand-written GPU sampler.
    draft = DFlash2DraftModel.__new__(DFlash2DraftModel)
    torch.nn.Module.__init__(draft)
    draft.lm_head = head
    draft.candidate_selector = CandidateSelector(hidden_size=HIDDEN, vocab_size=VOCAB,
                                                 state_rank=RANK, top_k=top_k).cuda().eval()
    draft.draft_config = parse_dflash_draft_config(draft_hf_config={
        "num_hidden_layers": 1,
        "dflash_config": {"block_size": BLOCK, "selector_rank": RANK,
                          "selector_top_k": top_k, "output_multiplier": 1.3,
                          "final_logit_softcapping": 2.0},
    })
    generator = torch.Generator(device="cuda").manual_seed(415)
    with torch.no_grad():
        for parameter in draft.candidate_selector.parameters():
            parameter.normal_(0, 0.4, generator=generator)
            assert_rank_agreement(parameter)
    return draft


def fill_hidden(hidden, iteration):
    hidden.zero_()
    channel = (torch.arange(hidden.shape[0], device="cuda") * 3 + iteration * 7) % HIDDEN
    hidden.scatter_(1, channel[:, None], 1.0)


def check_sampler(draft, sampler, hidden, input_ids, weights, batch, top_k):
    from torch._dynamo.utils import counters

    assert counters["stats"]["unique_graphs"] > 0, "Scorer never compiled"
    assert any(value > 0 for value in counters["inductor"].values()), "Inductor never reached"
    assert not any(counters["graph_break"].values()), dict(counters["graph_break"])
    candidates = sampler.candidate_out[:batch]
    tokens = sampler.out[:batch * (BLOCK - 1)].view(batch, BLOCK - 1)
    q = sampler.q_out[:batch]
    pred_hidden = hidden.view(batch, BLOCK, HIDDEN)[:, 1:]
    # Full, unsharded CPU head oracle; no use of local/gathered top-K results.
    logits = pred_hidden.cpu().double() @ weights.double().T
    expected_vals, expected_ids = logits.topk(top_k, dim=-1)
    torch.testing.assert_close(candidates.cpu(), expected_ids, rtol=0, atol=0)
    assert ((candidates >= 0) & (candidates < VOCAB)).all()
    assert torch.unique(candidates // 64).numel() > 1, "Fixture winners must cross TP shards"
    # Match the documented unary transform order independently on the CPU.
    unary = (expected_vals.float() * 1.3 / 2.0).tanh() * 2.0
    inputs = {"candidate_ids": expected_ids, "unary_logits": unary,
              "hidden_states": pred_hidden, "anchor_token_ids": input_ids.view(batch, BLOCK)[:, 0],
              "uniforms": sampler.uniforms[:batch], "temperatures": sampler.temperatures[:batch],
              "greedy_mask": sampler.greedy_mask[:batch]}
    scores = cpu_scores(draft.candidate_selector, inputs)
    expected_tokens, expected_q = cpu_walk(inputs, scores)
    torch.testing.assert_close(tokens.cpu(), expected_tokens, rtol=0, atol=0)
    torch.testing.assert_close(q.cpu(), expected_q, rtol=Q_RTOL, atol=Q_ATOL)
    torch.testing.assert_close(q.sum(-1), torch.ones_like(q[..., 0]), rtol=0, atol=2e-6)
    for value in (candidates, tokens, q, sampler.uniforms[:batch]):
        assert_rank_agreement(value)
    # A B=1 capture in capacity4 must not overwrite inactive proposal buffers.
    if batch < 4:
        assert sampler.q_out[batch:].isnan().all()
        assert (sampler.candidate_out[batch:] == -1).all()


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("top_k", [4, 16])
@torch.inference_mode()
def test_real_tp_head_and_worker_sampler_changed_graphs(topology, batch, top_k):
    from sglang.srt.speculative.dflash_worker_v2 import _SelectorDraftSampler

    weights = full_weight()
    draft = make_draft(make_head(weights), top_k)
    sampler = _SelectorDraftSampler(draft_model=draft, block_size=BLOCK, max_bs=4,
                                   device="cuda", sampling_enabled=True)
    sampler.q_out.fill_(float("nan"))
    sampler.candidate_out.fill_(-1)
    hidden = torch.empty(batch * BLOCK, HIDDEN, device="cuda")
    input_ids = torch.zeros(batch * BLOCK, device="cuda", dtype=torch.long)
    fill_hidden(hidden, 0)
    input_ids.view(batch, BLOCK)[:, 0] = torch.arange(batch, device="cuda") * 31 + 5
    info = SimpleNamespace(temperatures=torch.linspace(0.5, 1.5, batch, device="cuda")[:, None],
                           top_ks=torch.full((batch, 1), top_k, device="cuda", dtype=torch.int32),
                           is_all_greedy=False)
    sampler.stage_sampling_params(bs=batch, sampling_info=info)
    # Identical state immediately before any worker-owned RNG draws on all ranks.
    torch.cuda.manual_seed(1991)
    sampler(hidden, input_ids)
    check_sampler(draft, sampler, hidden, input_ids, weights, batch, top_k)
    with ps.graph_capture() as capture:
        for _ in range(3):
            sampler(hidden, input_ids)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture.stream):
            sampler(hidden, input_ids)
    torch.cuda.synchronize()
    previous_uniforms = sampler.uniforms[:batch].clone()
    for iteration in range(4):
        fill_hidden(hidden, iteration + 1)
        input_ids.view(batch, BLOCK)[:, 0].add_(17).remainder_(VOCAB)
        info.temperatures.add_(0.1)
        info.top_ks.fill_(top_k)
        info.top_ks[iteration % batch] = 1
        sampler.stage_sampling_params(bs=batch, sampling_info=info)
        graph.replay()
        check_sampler(draft, sampler, hidden, input_ids, weights, batch, top_k)
        assert not torch.equal(previous_uniforms, sampler.uniforms[:batch]), "Captured Philox did not advance"
        previous_uniforms = sampler.uniforms[:batch].clone()
    # Switch back to the uncaptured production sampler on the same stable buffers.
    sampler(hidden, input_ids)
    check_sampler(draft, sampler, hidden, input_ids, weights, batch, top_k)


@torch.inference_mode()
def test_real_tp_global_cutoff_ties_and_padding(topology):
    from sglang.srt.models.dflash import candidate_topk

    weights = full_weight(ties=True)
    head = make_head(weights)
    hidden = torch.empty(7, HIDDEN, device="cuda")
    fill_hidden(hidden, 3)
    ids, values, _ = candidate_topk(hidden, head, 4)
    logits = hidden.cpu().double() @ weights.double().T
    expected_values = logits.topk(4, dim=-1).values
    torch.testing.assert_close(values.cpu().double(), expected_values, rtol=0, atol=0)
    # torch.topk does not promise a global token-ID tie order. Require valid,
    # distinct IDs, complete above-cutoff membership, and exact rank agreement.
    assert ((ids >= 0) & (ids < VOCAB)).all()
    cpu_ids = ids.cpu()
    torch.testing.assert_close(logits.gather(1, cpu_ids), expected_values, rtol=0, atol=0)
    for row in range(hidden.shape[0]):
        assert cpu_ids[row].unique().numel() == 4
        above = set(torch.nonzero(logits[row] > expected_values[row, -1]).flatten().tolist())
        assert above.issubset(set(cpu_ids[row].tolist()))
    assert_rank_agreement(ids)
    assert_rank_agreement(values)


if __name__ == "__main__":
    from sglang.test.kernels.utils import multigpu_pytest_main

    assert torch.cuda.device_count() >= WORLD, "Eight allocated GPUs required"
    multigpu_pytest_main(__name__, __file__, num_gpus=(WORLD,), timeout=1800)
