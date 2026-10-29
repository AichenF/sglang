"""Real CUDA selector component checks; no CPU fallback or model-accuracy claim.

Run: python -m pytest test/registered/kernels/ops/speculative/test_dflash_selector_cuda.py -x -q
The independent CPU oracle uses full learned codebooks and every conditional q
row. Fixed uniforms separate arithmetic correctness from RNG consumption.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.models.dflash import CandidateSelector
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-small")

BLOCK, HIDDEN, VOCAB, RANK = 8, 32, 97, 8
# FP32 synthetic parameters isolate scoring/walking from BF16 model GEMMs.
# These bounds cover FP32 reduction/exp error, not changed proposal semantics.
SCORE_RTOL, SCORE_ATOL = 2e-5, 2e-6
Q_RTOL, Q_ATOL = 3e-5, 2e-6


@pytest.fixture(scope="module", autouse=True)
def require_cuda():
    assert torch.cuda.is_available(), "CUDA required; an all-skipped run is not evidence"
    from torch._dynamo.utils import counters
    from sglang.srt.utils import get_compiler_backend

    assert get_compiler_backend() == "inductor", "Production scorer must use Inductor"
    for name in ("TORCHDYNAMO_DISABLE", "TORCH_COMPILE_DISABLE"):
        assert os.environ.get(name, "0").lower() in ("", "0", "false"), name
    torch._dynamo.reset()
    counters.clear()
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    options = {"suppress_errors": False}
    limit = "recompile_limit" if hasattr(torch._dynamo.config, "recompile_limit") else "cache_size_limit"
    options[limit] = 64
    if hasattr(torch._dynamo.config, "fail_on_recompile_limit_hit"):
        options["fail_on_recompile_limit_hit"] = True
    with torch._dynamo.config.patch(options):
        yield
    torch.backends.cuda.matmul.allow_tf32 = old


def make_selector(top_k):
    model = CandidateSelector(hidden_size=HIDDEN, vocab_size=VOCAB,
                              state_rank=RANK, top_k=top_k).cuda().eval()
    generator = torch.Generator(device="cuda").manual_seed(414)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.4, generator=generator)
    return model


def make_inputs(top_k, batch=4):
    generator = torch.Generator(device="cuda").manual_seed(822)
    slots = BLOCK - 1
    return {
        "candidate_ids": (torch.arange(batch * slots * top_k, device="cuda")
                          .view(batch, slots, top_k) * 7 % VOCAB).long(),
        "unary_logits": torch.randn(batch, slots, top_k, device="cuda", generator=generator) * 0.2,
        "hidden_states": torch.randn(batch, slots, HIDDEN, device="cuda", generator=generator) * 0.3,
        "anchor_token_ids": torch.arange(batch, device="cuda", dtype=torch.long) * 11 + 2,
        "uniforms": torch.linspace(0.037, 0.963, batch * slots, device="cuda").view(batch, slots),
        "temperatures": torch.linspace(0.4, 1.7, batch, device="cuda"),
        "greedy_mask": torch.arange(batch, device="cuda") % 2 == 0,
    }


def cpu_scores(selector, inputs):
    """Independent explicit predecessor/anchor indexing, accumulated in float64."""
    ids = inputs["candidate_ids"].cpu()
    h = inputs["hidden_states"].cpu().double()
    projected = h @ selector.hidden_projection.weight.detach().cpu().double().T
    predecessor = selector.predecessor_codebook.detach().cpu().double()
    successor = selector.successor_codebook.detach().cpu().double()
    unary = inputs["unary_logits"].cpu().double()
    anchor = inputs["anchor_token_ids"].cpu()
    batch, slots, top_k = ids.shape
    scores = torch.empty(batch, slots, top_k, top_k, dtype=torch.float64)
    for row in range(batch):
        for slot in range(slots):
            for previous in range(top_k):
                pred = anchor[row] if slot == 0 else ids[row, slot - 1, previous]
                for candidate in range(top_k):
                    scores[row, slot, previous, candidate] = unary[row, slot, candidate] + (
                        predecessor[pred] * projected[row, slot] * successor[ids[row, slot, candidate]]
                    ).sum()
    return scores


def cpu_walk(inputs, scores):
    """Return the selected token and ALL K probabilities for each realized edge."""
    ids = inputs["candidate_ids"].cpu()
    uniforms, temperatures = inputs["uniforms"].cpu().double(), inputs["temperatures"].cpu().double()
    greedy = inputs["greedy_mask"].cpu()
    scores = scores.cpu().double()
    batch, slots, top_k = ids.shape
    tokens, q = torch.empty(batch, slots, dtype=torch.long), torch.zeros(batch, slots, top_k)
    for row in range(batch):
        previous = 0
        for slot in range(slots):
            edge = scores[row, slot, previous]
            if greedy[row]:
                choice = int(edge.argmax())
                q[row, slot, choice] = 1
            else:
                probabilities = torch.softmax(edge / temperatures[row], dim=0)
                choice = min(int((uniforms[row, slot] >= probabilities.cumsum(0)).sum()), top_k - 1)
                q[row, slot] = probabilities.float()
            tokens[row, slot] = ids[row, slot, choice]
            previous = choice
    return tokens, q


def run_selector(selector, inputs):
    scores = selector.build_lattice(**{key: inputs[key] for key in (
        "candidate_ids", "unary_logits", "hidden_states", "anchor_token_ids")})
    tokens, q = selector.sample_path(scores=scores, **{key: inputs[key] for key in (
        "candidate_ids", "uniforms", "temperatures", "greedy_mask")})
    return scores, tokens, q


def check(selector, inputs, output):
    from torch._dynamo.utils import counters

    # Observe the real production compiled callable; no AST/backend replacement.
    # A unique Dynamo graph alone could be an eager backend, so also require
    # actual Inductor work/cache activity and reject every graph break.
    assert counters["stats"]["unique_graphs"] > 0, "Scorer never compiled"
    assert any(value > 0 for value in counters["inductor"].values()), "Inductor was never reached"
    assert not any(counters["graph_break"].values()), dict(counters["graph_break"])
    scores, tokens, q = output
    assert scores.is_cuda and tokens.is_cuda and q.is_cuda
    torch.testing.assert_close(scores.cpu().double(), cpu_scores(selector, inputs),
                               rtol=SCORE_RTOL, atol=SCORE_ATOL)
    # Use the actual scored lattice for walk parity, isolating the FP32 GEMM
    # rounding from inverse-CDF boundary decisions. Scores were checked above.
    expected_tokens, expected_q = cpu_walk(inputs, scores)
    torch.testing.assert_close(tokens.cpu(), expected_tokens, rtol=0, atol=0)
    torch.testing.assert_close(q.cpu(), expected_q, rtol=Q_RTOL, atol=Q_ATOL)
    torch.testing.assert_close(q.sum(-1), torch.ones_like(q[..., 0]), rtol=0, atol=2e-6)
    assert torch.isfinite(q).all() and (q >= 0).all()


@pytest.mark.parametrize("top_k", [4, 16])
@pytest.mark.parametrize("mode", ["greedy", "sample", "mixed"])
@torch.inference_mode()
def test_learned_lattice_and_complete_proposal_q(top_k, mode):
    selector, inputs = make_selector(top_k), make_inputs(top_k)
    if mode != "mixed":
        inputs["greedy_mask"].fill_(mode == "greedy")
    output = run_selector(selector, inputs)
    check(selector, inputs, output)
    # Standalone rows must agree with the mixed batch, including greedy q.
    for row in range(4):
        one = {key: value[row:row + 1] for key, value in inputs.items()}
        isolated = run_selector(selector, one)
        torch.testing.assert_close(isolated[1][0], output[1][row], rtol=0, atol=0)
        torch.testing.assert_close(isolated[2][0], output[2][row], rtol=Q_RTOL, atol=Q_ATOL)


@torch.inference_mode()
def test_greedy_first_tie_and_first_nan_and_sample_cdf_boundaries():
    selector, inputs = make_selector(4), make_inputs(4)
    scores = torch.zeros(4, BLOCK - 1, 4, 4, device="cuda")
    inputs["greedy_mask"].copy_(torch.tensor([True, True, False, False], device="cuda"))
    scores[1, :, :, 1] = float("nan")
    scores[1, :, :, 3] = float("nan")
    inputs["uniforms"][2].fill_(0.25)  # equality advances to the next candidate
    inputs["uniforms"][3].fill_(0.999)
    tokens, q = selector.sample_path(scores=scores, **{key: inputs[key] for key in (
        "candidate_ids", "uniforms", "temperatures", "greedy_mask")})
    expected_tokens, expected_q = cpu_walk(inputs, scores)
    torch.testing.assert_close(tokens.cpu(), expected_tokens, rtol=0, atol=0)
    torch.testing.assert_close(q.cpu(), expected_q, rtol=0, atol=0)


@torch.inference_mode()
def test_worker_stages_zero_temperature_greedy_and_mixed_proposal_q():
    from sglang.srt.speculative.dflash_worker_v2 import _SelectorDraftSampler

    selector, inputs = make_selector(4), make_inputs(4)
    sampler = _SelectorDraftSampler(draft_model=SimpleNamespace(candidate_selector=selector),
                                   block_size=BLOCK, max_bs=4, device="cuda", sampling_enabled=True)
    info = SimpleNamespace(temperatures=torch.tensor([[0.0], [0.7], [0.0], [1.2]], device="cuda"),
                           top_ks=torch.tensor([[1], [4], [1], [4]], device="cuda"), is_all_greedy=False)
    sampler.stage_sampling_params(bs=4, sampling_info=info)
    torch.testing.assert_close(sampler.temperatures, torch.tensor([1e-5, 0.7, 1e-5, 1.2], device="cuda"))
    inputs["temperatures"], inputs["greedy_mask"] = sampler.temperatures, sampler.greedy_mask
    check(selector, inputs, run_selector(selector, inputs))


@torch.inference_mode()
def test_changed_inputs_graph_replay_and_request_isolation():
    selector, inputs = make_selector(4), make_inputs(4)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run_selector(selector, inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run_selector(selector, inputs)
    snapshots = []
    for iteration in range(4):
        inputs["hidden_states"].add_(0.07)
        inputs["unary_logits"].mul_(-0.8)
        inputs["candidate_ids"].add_(3).remainder_(VOCAB)
        inputs["anchor_token_ids"].add_(7).remainder_(VOCAB)
        inputs["uniforms"].add_(0.137).remainder_(1)
        inputs["temperatures"].add_(0.11)
        inputs["greedy_mask"].logical_not_()
        graph.replay()
        check(selector, inputs, output)
        snapshots.append(output[2].clone())
    assert not torch.equal(snapshots[0], snapshots[-1]), "Graph retained stale q"
    previous = tuple(value.clone() for value in output)
    inputs["hidden_states"][0].add_(1)
    inputs["anchor_token_ids"][0].add_(1).remainder_(VOCAB)
    graph.replay()
    check(selector, inputs, output)
    for actual, before in zip(output, previous):
        torch.testing.assert_close(actual[1:], before[1:], rtol=0, atol=0)
