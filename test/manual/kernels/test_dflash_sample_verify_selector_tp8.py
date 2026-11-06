"""8-GPU test of the selector (sparse q) mode of the vocab-parallel DFlash sampling verify.

    torchrun --nproc-per-node 8 test/manual/kernels/test_dflash_sample_verify_selector_tp8.py

Reference: the worker's selector accept (dense q scattered from the sparse q, then dspark_accept.accept_sampling) with
the same coins (same CUDA seed). Accept lengths and bonus tokens must match; greedy rows must equal torch.argmax.
"""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
ws = int(os.environ["WORLD_SIZE"])
assert ws == 8, "run with torchrun --nproc-per-node 8"
torch.cuda.set_device(rank)
dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
cpu = dist.new_group(backend="gloo")
dev = torch.device("cuda", rank)

from sglang.kernels.ops.speculative.dspark.dspark_accept import accept_sampling  # noqa: E402
from sglang.kernels.ops.speculative.sample_verify import SEL_K, SampleVerifyState  # noqa: E402
from sglang.srt.sampling.sampling_params import TOP_K_ALL  # noqa: E402

Vs = 19072
V = Vs * ws
G = 8
GAMMA = G - 1
vs0 = rank * Vs
TRIALS = int(os.environ.get("SVS_TRIALS", "600"))

st = SampleVerifyState(dev, cpu, list(range(ws)), rank, local_vocab=Vs, vocab_start=vs0, vocab=V, seed=1234)
st.enable_selector(GAMMA)


def log(*a):
    if rank == 0:
        print(*a, flush=True)


def make_logits(n, gen, kinds):
    rows = []
    for k in kinds:
        if k == "flat":
            x = torch.randn(V, device=dev, generator=gen)
        elif k == "ties":
            x = torch.round(torch.randn(V, device=dev, generator=gen) * 2.0)
            top = torch.randperm(V, device=dev, generator=gen)[:8]
            x[top] += torch.round(torch.rand(8, device=dev, generator=gen) * 6.0 + 6.0)
        else:
            sig = {"peak1": 2.0, "peak5": 2.5, "peak50": 1.5}[k]
            npk = {"peak1": 1, "peak5": 5, "peak50": 50}[k]
            x = torch.randn(V, device=dev, generator=gen) * sig
            top = torch.randperm(V, device=dev, generator=gen)[:npk]
            x[top] += torch.rand(npk, device=dev, generator=gen) * 10.0 + 8.0
        rows.append(x)
    return torch.stack(rows).to(torch.bfloat16)


def sampling_info_for(params):
    return SimpleNamespace(
        temperatures=torch.tensor([p[1] for p in params], dtype=torch.float32, device=dev).view(-1, 1),
        top_ps=torch.tensor([p[2] for p in params], dtype=torch.float32, device=dev),
        top_ks=torch.tensor([1 if p[0] == 0 else TOP_K_ALL for p in params], dtype=torch.int32, device=dev),
        is_all_greedy=all(p[0] == 0 for p in params),
        need_top_k_sampling=any(p[0] == 0 for p in params),
        need_top_p_sampling=any(p[2] != 1.0 for p in params),
    )


def make_case(trial, gen):
    """Inputs identical on every rank. Candidates: mostly the row's top tokens (some far outside top-p), q a random
    distribution over them (some zeros), the draft token drawn from q like the selector does."""
    bs = [4, 4, 1, 2, 3, 8, 4, 6][trial % 8]
    n = bs * G
    kinds = ["peak1", "peak5", "peak50", "flat", "ties"]
    kind = [kinds[int(torch.randint(0, len(kinds), (1,), device=dev, generator=gen))] for _ in range(n)]
    full = make_logits(n, gen, kind)
    params = []
    for b in range(bs):
        greedy = (trial % 7 == 3 and b == 0) or trial % 23 == 5
        T = [1.0, 1.0, 0.6, 1.5, 0.8][int(torch.randint(0, 5, (1,), device=dev, generator=gen))]
        tp = [0.95, 0.95, 1.0, 0.7, 0.3, 0.999][int(torch.randint(0, 6, (1,), device=dev, generator=gen))]
        params.append((0, 1.0, 1.0) if greedy else (1, T, tp))
    ids = torch.empty((bs, GAMMA, SEL_K), dtype=torch.int64, device=dev)
    q = torch.empty((bs, GAMMA, SEL_K), dtype=torch.float32, device=dev)
    cand = torch.empty(n, dtype=torch.int64, device=dev)
    fl = full.float()
    for b in range(bs):
        cand[b * G] = int(torch.randint(0, V, (1,), device=dev, generator=gen))
        for j in range(GAMMA):
            r = b * G + j
            n_top = int(torch.randint(4, SEL_K + 1, (1,), device=dev, generator=gen))
            top = torch.topk(fl[r], 64).indices[torch.randperm(64, device=dev, generator=gen)[:n_top]]
            pool = torch.cat([top, torch.randint(0, V, (64,), device=dev, generator=gen)])
            uniq = []
            for t in pool.tolist():
                if t not in uniq:
                    uniq.append(t)
                if len(uniq) == SEL_K:
                    break
            ids[b, j] = torch.tensor(uniq, dtype=torch.int64, device=dev)
            w = torch.softmax(torch.randn(SEL_K, device=dev, generator=gen) * 2.0, dim=0)
            nz = int(torch.randint(0, 5, (1,), device=dev, generator=gen))
            if nz:
                w[torch.randperm(SEL_K, device=dev, generator=gen)[:nz]] = 0.0
            w = w / w.sum()
            q[b, j] = w
            if params[b][0] == 0:
                k = int(torch.argmax(w))
            else:
                k = int(torch.multinomial(w, 1, generator=gen))
            cand[r + 1] = ids[b, j, k]
    return bs, n, full, params, ids, q, cand


def reference(full, params, ids, q, cand, bs, seed):
    tks = [1 if p[0] == 0 else TOP_K_ALL for p in params]
    info = sampling_info_for(params)
    draft_input = SimpleNamespace(max_top_k=max(tks), uniform_top_k_value=tks[0] if len(set(tks)) == 1 else None)
    dp = torch.zeros((bs, GAMMA, V), dtype=torch.float32, device=dev).scatter_(-1, ids, q)
    torch.cuda.manual_seed(seed)
    acc, bonus, _ = accept_sampling(
        candidates=cand.view(bs, G), target_logits=full.float(), draft_probs=dp, sampling_info=info,
        draft_input=draft_input, gamma=GAMMA, verify_num_draft_tokens=G, cutoff_verify_lens=None,
    )
    return acc.tolist(), bonus.tolist()


def run_kernel(shard, cand, params, ids, q, bs, seed, out=None):
    st.stage(params)
    st.stage_selector(ids, q)
    torch.cuda.manual_seed(seed)
    st.draw_selector_coins(bs)
    return st.run(shard, cand, G, out=out, selector=True)


def decode(pred, cand, bs):
    p, c = pred.view(bs, G).tolist(), cand.view(bs, G).tolist()
    acc = []
    for b in range(bs):
        a = 0
        while a < GAMMA and p[b][a] == c[b][a + 1]:
            a += 1
        acc.append(a)
    return acc, [p[b][acc[b]] for b in range(bs)]


def same_on_all_ranks(t):
    g = [torch.empty_like(t.cpu()) for _ in range(ws)]
    dist.all_gather(g, t.cpu(), group=cpu)
    return all(torch.equal(g[0], x) for x in g)


gen = torch.Generator(device=dev).manual_seed(2025)
c = dict(req=0, acc_bad=0, bon_bad=0, greedy_bad=0, rank_bad=0)
for trial in range(TRIALS):
    bs, _, full, params, ids, q, cand = make_case(trial, gen)
    pred = run_kernel(full[:, vs0 : vs0 + Vs].contiguous(), cand, params, ids, q, bs, 1000 + trial).clone()
    racc, rbon = reference(full, params, ids, q, cand, bs, 1000 + trial)
    kacc, kbon = decode(pred, cand, bs)
    c["rank_bad"] += int(not same_on_all_ranks(pred))
    for b in range(bs):
        if params[b][0] == 0:
            c["greedy_bad"] += int(not torch.equal(pred[b * G : (b + 1) * G], full[b * G : (b + 1) * G].float().argmax(-1)))
            continue
        c["req"] += 1
        c["acc_bad"] += int(kacc[b] != racc[b])
        c["bon_bad"] += int(kacc[b] == racc[b] and kbon[b] != rbon[b])
        if (kacc[b], kbon[b]) != (racc[b], rbon[b]):
            log(f"  mismatch trial {trial} b {b} {params[b]}: kernel {kacc[b]}/{kbon[b]}, reference {racc[b]}/{rbon[b]}")
log(f"[eager] {TRIALS} trials: {c}")
bad = c["acc_bad"] + c["bon_bad"] + c["greedy_bad"] + c["rank_bad"]

# graph: the worker's layout (static state buffers refreshed before each replay) against eager and the reference
gen = torch.Generator(device=dev).manual_seed(99)
cases = [cs for cs in (make_case(8 * k, gen) for k in range(6)) if cs[0] == 4]
logits_buf = torch.empty((32, Vs), dtype=torch.bfloat16, device=dev)
cand_buf = torch.empty(32, dtype=torch.int64, device=dev)
out_buf = torch.empty(32, dtype=torch.int64, device=dev)
_, _, full0, params0, ids0, q0, cand0 = cases[0]
logits_buf.copy_(full0[:, vs0 : vs0 + Vs])
cand_buf.copy_(cand0)
run_kernel(logits_buf, cand_buf, params0, ids0, q0, 4, 0, out=out_buf)
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    st.run(logits_buf, cand_buf, G, out=out_buf, selector=True)
gbad = 0
for k, (_, _, full, params, ids, q, cand) in enumerate(cases * 3):
    st.stage(params)
    st.stage_selector(ids, q)
    torch.cuda.manual_seed(5000 + k)
    st.draw_selector_coins(4)
    logits_buf.copy_(full[:, vs0 : vs0 + Vs])
    cand_buf.copy_(cand)
    graph.replay()
    g_out = out_buf.clone()
    e_out = st.run(logits_buf, cand_buf, G, out=torch.empty_like(out_buf), selector=True).clone()
    racc, rbon = reference(full, params, ids, q, cand, 4, 5000 + k)
    kacc, kbon = decode(g_out, cand, 4)
    ok = torch.equal(g_out, e_out) and same_on_all_ranks(g_out)
    ok = ok and all(params[b][0] == 0 or (kacc[b], kbon[b]) == (racc[b], rbon[b]) for b in range(4))
    gbad += int(not ok)
log(f"[graph] {len(cases) * 3} replays, mismatches {gbad}")

t = torch.tensor([bad + gbad], device=dev)
dist.all_reduce(t)
log(f"ALL {'OK' if int(t) == 0 else 'NOT OK'}")
dist.destroy_process_group()
