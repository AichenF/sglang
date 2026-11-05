"""8-GPU test of the selector (sparse q) mode of the vocab-parallel DFlash sampling verify.

    torchrun --nproc-per-node 8 test/manual/kernels/test_dflash_sample_verify_selector_tp8.py

The kernel gets the candidate ids / q of each draft position and the uniforms the reference draws (torch.rand of
shapes (bs, G - 1) and (bs,)); the reference is the selector path of the DFlash worker: dense q scattered from the
sparse q, then dspark_accept.accept_sampling with the same uniforms. Every accept length and bonus token must match;
a mismatch is reported with its margin in fp64 (accept test |coin * q - p| / p, bonus distance to a CDF boundary).

Environment:
    SVS_MODE    comma-separated sections (default "eager,graph")
    SVS_TRIALS  trials of the eager section (default 600)
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
from sglang.kernels.ops.speculative.sample_verify import (  # noqa: E402
    SEL_K,
    SampleVerifyState,
)
from sglang.srt.sampling.sampling_params import TOP_K_ALL  # noqa: E402

MODES = set(os.environ.get("SVS_MODE", "eager,graph").split(","))
Vs = 19072  # a 152576-entry vocabulary over 8 ranks
V = Vs * ws
G = 8
GAMMA = G - 1
vs0 = rank * Vs


def log(*a):
    if rank == 0:
        print(*a, flush=True)


st = SampleVerifyState(dev, cpu, list(range(ws)), rank, local_vocab=Vs, vocab_start=vs0, vocab=V, seed=1234)
st.enable_selector(GAMMA)


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


def draft_input_for(params):
    """max_top_k / uniform_top_k_value as DFlashDraftInputV2 derives them from the requests' top_k."""
    tks = [1 if p[0] == 0 else TOP_K_ALL for p in params]
    return SimpleNamespace(max_top_k=max(max(tks), 1), uniform_top_k_value=tks[0] if len(set(tks)) == 1 else None)


def reference(full, params, ids, q, cand, U, UF, bs):
    """The worker's selector accept (_selector_sampling_accept) with the given uniforms."""
    dp = torch.zeros((bs, GAMMA, V), dtype=torch.float32, device=dev)
    dp.scatter_(-1, ids, q)
    acc, bonus, _ = accept_sampling(
        candidates=cand.view(bs, G),
        target_logits=full.float(),
        draft_probs=dp,
        sampling_info=sampling_info_for(params),
        draft_input=draft_input_for(params),
        gamma=GAMMA,
        verify_num_draft_tokens=G,
        cutoff_verify_lens=None,
        uniform_samples=U,
        uniform_samples_final=UF,
    )
    return acc.long(), bonus.long()


def decode_predict(pred, cand, bs):
    p = pred.view(bs, G)
    c = cand.view(bs, G)
    acc, bon = [], []
    for b in range(bs):
        a = 0
        while a < GAMMA and int(p[b, a]) == int(c[b, a + 1]):
            a += 1
        acc.append(a)
        bon.append(int(p[b, a]))
    return acc, bon


def exact_p(row_bf16, T, top_p):
    q = torch.softmax(row_bf16.double() / T, dim=-1)
    if top_p < 1.0:
        qs, _ = torch.sort(q, descending=True)
        cs = torch.cumsum(qs, 0)
        k = int(torch.searchsorted(cs, torch.tensor([top_p], dtype=torch.float64, device=dev)))
        keep = q >= qs[min(k, qs.numel() - 1)]
    else:
        keep = q > 0
    t = torch.where(keep, q, torch.zeros_like(q))
    return t / t.sum()


def margin(full, params, ids, q, cand, U, UF, b, ka, ra):
    """fp64 margins of the decision that differs: the accept test at the first differing row, else the bonus CDF."""
    T, top_p = params[b][1], params[b][2]
    j = min(ka, ra)
    p = exact_p(full[b * G + j], T, top_p)
    x = int(cand[b * G + j + 1])
    if ka != ra and j < GAMMA:
        qx = float(q[b, j][ids[b, j] == x].sum())
        px = float(p[x])
        return f"accept row {j}: coin*q={float(U[b, j]) * qx:.9g} p={px:.9g} rel {abs(float(U[b, j]) * qx - px) / max(px, 1e-30):.3g}"
    dq = torch.zeros(V, dtype=torch.float64, device=dev)
    if j < GAMMA:
        dq.scatter_(0, ids[b, j], q[b, j].double())
        w = torch.clamp(p - dq, min=0.0)
    else:
        w = p
    cw = torch.cumsum(w, 0)
    u = float(UF[b]) * float(cw[-1])
    idx = int(torch.searchsorted(cw, torch.tensor([u], dtype=torch.float64, device=dev), right=True))
    lo = float(cw[idx - 1]) if idx > 0 else 0.0
    hi = float(cw[min(idx, V - 1)])
    return f"bonus: u={u:.9g} in [{lo:.9g}, {hi:.9g}) of total {float(cw[-1]):.9g}, rel distance {min(u - lo, hi - u) / float(cw[-1]):.3g}"


def same_on_all_ranks(t):
    tc = t.detach().contiguous().cpu()
    if tc.dtype == torch.bfloat16:
        tc = tc.view(torch.int16).to(torch.int32)
    g = [torch.empty_like(tc) for _ in range(ws)]
    dist.all_gather(g, tc, group=cpu)
    return all(torch.equal(g[0], x) for x in g)


bad_total = 0

if "eager" in MODES:
    gen = torch.Generator(device=dev)
    gen.manual_seed(2025)
    n_trials = int(os.environ.get("SVS_TRIALS", "600"))
    c = dict(req=0, rows_rejected=0, acc_bad=0, bon_bad=0, greedy_bad=0, rank_bad=0, bonus_eq_cand=0)
    for trial in range(n_trials):
        bs, n, full, params, ids, q, cand = make_case(trial, gen)
        torch.cuda.manual_seed(1000 + trial)  # same CUDA generator state on every rank, as in the server
        U = torch.rand((bs, GAMMA), dtype=torch.float32, device=dev)
        UF = torch.rand((bs,), dtype=torch.float32, device=dev)
        assert same_on_all_ranks(full) and same_on_all_ranks(cand) and same_on_all_ranks(U), "inputs differ"
        shard = full[:, vs0 : vs0 + Vs].contiguous()
        st.stage(params)
        pred = st.run(shard, cand, G, sel_ids=ids, sel_q=q, level_coins=U, final_coins=UF).clone()
        torch.cuda.synchronize()
        racc, rbon = reference(full, params, ids, q, cand, U, UF, bs)
        kacc, kbon = decode_predict(pred, cand, bs)
        c["rank_bad"] += 0 if same_on_all_ranks(pred) else 1
        for b in range(bs):
            if params[b][0] == 0:
                am = torch.argmax(full[b * G : (b + 1) * G].float(), dim=-1)
                c["greedy_bad"] += 0 if torch.equal(pred[b * G : (b + 1) * G], am) else 1
                continue
            c["req"] += 1
            c["rows_rejected"] += int(kacc[b] < GAMMA)
            a_bad = kacc[b] != int(racc[b])
            b_bad = not a_bad and kbon[b] != int(rbon[b])
            c["acc_bad"] += int(a_bad)
            c["bon_bad"] += int(b_bad)
            if a_bad or b_bad:
                log(f"  mismatch trial {trial} b {b} params {params[b]}: accept kernel {kacc[b]} ref {int(racc[b])}, "
                    f"bonus kernel {kbon[b]} ref {int(rbon[b])}; "
                    + margin(full, params, ids, q, cand, U, UF, b, kacc[b], int(racc[b])))
        if trial % 100 == 99:
            log(f"[eager] {trial + 1} trials: {c}")
    fails = int(c["acc_bad"] + c["bon_bad"] + c["greedy_bad"] + c["rank_bad"])
    log(f"[eager] {n_trials} trials, {c['req']} sampled requests ({c['rows_rejected']} with a rejection): accept-length "
        f"mismatches {c['acc_bad']}, bonus mismatches {c['bon_bad']}, greedy != argmax {c['greedy_bad']}, "
        f"ranks disagree {c['rank_bad']}")
    t = torch.tensor([fails], device=dev)
    dist.all_reduce(t)
    log(f"[eager] RESULT {'OK' if int(t) == 0 else 'MISMATCHES (see margins above)'}")
    bad_total += int(t)

if "graph" in MODES:
    # the worker's layout: state buffers (ids, q, coins) at fixed addresses, refreshed before each replay
    gen = torch.Generator(device=dev)
    gen.manual_seed(99)
    bs, n = 4, 32
    cases = [make_case(8 * k, gen) for k in range(6)]
    cases = [cs for cs in cases if cs[0] == bs]
    logits_buf = torch.empty((n, Vs), dtype=torch.bfloat16, device=dev)
    cand_buf = torch.empty(n, dtype=torch.int64, device=dev)
    out_buf = torch.empty(n, dtype=torch.int64, device=dev)
    _, _, full0, params0, ids0, q0, cand0 = cases[0]
    st.stage(params0)
    st.stage_selector(ids0, q0)
    st.draw_selector_coins(bs)
    logits_buf.copy_(full0[:, vs0 : vs0 + Vs])
    cand_buf.copy_(cand0)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        st.run(logits_buf, cand_buf, G, out=out_buf, selector=True)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        st.run(logits_buf, cand_buf, G, out=out_buf, selector=True)
    gbad = 0
    for k, (_, _, full, params, ids, q, cand) in enumerate(cases * 3):
        torch.cuda.manual_seed(5000 + k)
        st.stage(params)
        st.stage_selector(ids, q)
        coins = st.draw_selector_coins(bs).clone()
        logits_buf.copy_(full[:, vs0 : vs0 + Vs])
        cand_buf.copy_(cand)
        graph.replay()
        torch.cuda.synchronize()
        g_out = out_buf.clone()
        g = st.sel_gamma
        e_out = st.run(
            logits_buf, cand_buf, G, out=torch.empty_like(out_buf), sel_ids=ids, sel_q=q,
            level_coins=coins[: bs * g].view(bs, g), final_coins=coins[32 * g : 32 * g + bs],
        ).clone()
        racc, rbon = reference(full, params, ids, q, cand, coins[: bs * g].view(bs, g), coins[32 * g : 32 * g + bs], bs)
        kacc, kbon = decode_predict(g_out, cand, bs)
        same = torch.equal(g_out, e_out)
        dec = all(params[b][0] == 0 or (kacc[b] == int(racc[b]) and kbon[b] == int(rbon[b])) for b in range(bs))
        gbad += int(not same) + int(not dec) + int(not same_on_all_ranks(g_out))
    t = torch.tensor([gbad], device=dev)
    dist.all_reduce(t)
    log(f"[graph] {len(cases) * 3} replays: graph == eager and == reference: {'OK' if int(t) == 0 else f'FAIL ({int(t)})'}")
    bad_total += int(t)

dist.barrier()
log(f"ALL {'OK' if bad_total == 0 else 'NOT OK'}")
dist.destroy_process_group()
