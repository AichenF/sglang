"""8-GPU test of the vocab-parallel DFlash sampling verify (csrc/distributed/sample_verify.cuh).

    torchrun --nproc-per-node 8 test/manual/kernels/test_dflash_sample_verify_tp8.py

Environment:
    SV_MODE    comma-separated sections (default "eager,stat,graph,fast"; also "time", "stamps", "san")
    SV_TRIALS  trials of the eager section (default 150)
    SV_STAT_N  kernel calls of the statistics section (default 20000)
    SV_FAST_N  graph replays of the fast-path section (default 150)

Sections:
    eager   identical uniforms: the kernel against the reference DFlash path (full fp32 logits ->
            build_speculative_verify_target_probs -> tree_speculative_sampling_target_only) and against an exact
            fp64 top-p + chain-rejection reference; mixed greedy / sampled requests, batch 1..8, several
            temperatures / top-p values and peaked, flat and heavily tied rows. Greedy requests must equal
            torch.argmax. Pass: no accept-length mismatch, bonus mismatches only at the fp32 CDF boundary,
            kernel distribution within 1e-3 total variation of the exact one, ranks agree.
    stat    in-kernel Philox uniforms: accept-length and bonus-token histograms against the exact distributions
            (chi-square), the reference sampling kernel with torch.rand uniforms for comparison.
    graph   CUDA graph replays with changing logits / candidates / uniforms / parameters: bitwise equal to eager
            calls, accept decisions equal to the reference, ranks agree; Philox graph replay == eager call.
    fast    the worker's capture layout: the in-graph greedy argmax on the main stream and the kernel with
            skip_greedy on a forked stream. All-greedy batches: argmax == torch.argmax bitwise, the kernel leaves
            its output and call counters untouched; mixed / sampled batches: kernel output bitwise == eager call,
            accept decisions == reference, greedy rows == torch.argmax; ranks agree.
    time    graph-replayed kernel time against the in-graph greedy argmax and the eager reference path; the
            fast-path layout (argmax + forked kernel) against the argmax alone (greedy) and the kernel alone (sampled).
    stamps  per-phase %globaltimer stamps of the kernel.
    san     a few Philox calls only (for compute-sanitizer runs).
"""

import math
import os
import time
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

from sglang.kernels.ops.speculative.sample_verify import (  # noqa: E402
    DBG,
    SampleVerifyState,
    unsupported_reason,
)
from sglang.srt.sampling.sampling_params import TOP_K_ALL  # noqa: E402
from sglang.srt.speculative.dflash_utils import (  # noqa: E402
    build_speculative_verify_target_probs,
    compute_dflash_sampling_correct_drafts_and_bonus,
)

MODES = set(os.environ.get("SV_MODE", "eager,stat,graph,fast").split(","))
Vs = 19072  # a 152576-entry vocabulary over 8 ranks
V = Vs * ws
G = 8
vs0 = rank * Vs
assert unsupported_reason(Vs, dev) is None, unsupported_reason(Vs, dev)


def log(*a):
    if rank == 0:
        print(*a, flush=True)


st = SampleVerifyState(dev, cpu, list(range(ws)), rank, local_vocab=Vs, vocab_start=vs0, vocab=V, seed=1234)


def key16(xb):
    """bf16 tensor -> int32 monotonic 16-bit keys (-0 -> +0), as in the kernel."""
    raw = xb.view(torch.int16).to(torch.int32) & 0xFFFF
    raw = torch.where(raw == 0x8000, torch.zeros_like(raw), raw)
    return torch.where((raw & 0x8000) != 0, (~raw) & 0xFFFF, raw | 0x8000)


def make_logits(n, gen, kind):
    """[n, V] bf16, identical on every rank (same generator state)."""
    rows = []
    for r in range(n):
        k = kind[r]
        if k == "flat":
            x = torch.randn(V, device=dev, generator=gen)
        elif k == "ties":  # coarse values -> large tie groups at every threshold
            x = torch.round(torch.randn(V, device=dev, generator=gen) * 2.0)
            # unique indices: index_put with duplicates is nondeterministic (ranks would hold different rows)
            top = torch.randperm(V, device=dev, generator=gen)[:8]
            x[top] += torch.round(torch.rand(8, device=dev, generator=gen) * 6.0 + 6.0)
        else:  # peaked: noise + a few boosted tokens
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


def reference(full_bf16, cand, params, coins, bs):
    si = sampling_info_for(params)
    logits = full_bf16.float()
    tprobs = build_speculative_verify_target_probs(
        next_token_logits=logits, sampling_info=si, draft_token_num=G, bs=bs
    )
    acc, bonus = compute_dflash_sampling_correct_drafts_and_bonus(
        candidates=cand.view(bs, G),
        next_token_logits=logits,
        sampling_info=si,
        uniform_samples=coins[: bs * G].view(bs, G).contiguous(),
        uniform_samples_for_final_sampling=coins[bs * G : bs * G + bs].contiguous(),
        threshold_single=1.0,
        threshold_acc=1.0,
    )
    return tprobs, acc.long(), bonus.long()


def exact64(full_bf16, cand, params, coins, bs):
    """fp64 reference: q = softmax(x / T); kept = {q >= v*}, v* = largest value with sum_{q >= v*} q >= top_p (ties
    kept; top_p >= 1 keeps all); tp = q / sum(kept); chain accept with the coins; bonus = inverse CDF (vocab order,
    first cumulative > u) of tp without the rejected candidate. Returns per-request accept / bonus and per-row
    (kept mask, tp)."""
    acc, bon, rows = [], [], {}
    for b in range(bs):
        if params[b][0] == 0:
            acc.append(None)
            bon.append(None)
            continue
        T, top_p = params[b][1], params[b][2]
        tps = []
        for j in range(G):
            q = torch.softmax(full_bf16[b * G + j].double() / T, dim=-1)
            if top_p < 1.0:
                qs, _ = torch.sort(q, descending=True)
                cs = torch.cumsum(qs, 0)
                k = int(torch.searchsorted(cs, torch.tensor([top_p], dtype=torch.float64, device=dev)))
                keep = q >= qs[min(k, qs.numel() - 1)]
            else:
                keep = q > 0
            t = torch.where(keep, q, torch.zeros_like(q))
            t = t / t.sum()
            tps.append(t)
            rows[b * G + j] = (keep, t)
        a = 0
        while a < G - 1:
            c = int(cand[b * G + a + 1])
            if float(tps[a][c]) > 0 and float(coins[b * G + a]) < float(tps[a][c]):
                a += 1
            else:
                break
        w = tps[a].clone()
        if a < G - 1:
            w[int(cand[b * G + a + 1])] = 0.0
        cw = torch.cumsum(w, 0)
        u = float(coins[bs * G + b]) * float(cw[-1])
        idx = int(torch.searchsorted(cw, torch.tensor([u], dtype=torch.float64, device=dev), right=True))
        acc.append(a)
        bon.append(min(idx, w.numel() - 1))
    return acc, bon, rows


def decode_predict(pred, cand, bs):
    """greedy-verify layout -> (accept length, bonus) per request"""
    p = pred.view(bs, G)
    c = cand.view(bs, G)
    acc = torch.zeros(bs, dtype=torch.long, device=dev)
    bon = torch.zeros(bs, dtype=torch.long, device=dev)
    for b in range(bs):
        a = 0
        while a < G - 1 and int(p[b, a]) == int(c[b, a + 1]):
            a += 1
        acc[b] = a
        bon[b] = p[b, a]
    return acc, bon


def same_on_all_ranks(t):
    # gloo on host copies: no NCCL kernels between the kernels under test
    tc = t.detach().contiguous().cpu()
    if tc.dtype == torch.bfloat16:
        tc = tc.view(torch.int16).to(torch.int32)
    g = [torch.empty_like(tc) for _ in range(ws)]
    dist.all_gather(g, tc, group=cpu)
    return all(torch.equal(g[0], x) for x in g)


def make_case(trial, gen):
    bs = [4, 4, 4, 1, 2, 3, 8, 4][trial % 8]
    n = bs * G
    kinds = ["peak1", "peak5", "peak50", "flat", "ties"]
    kind = [kinds[int(torch.randint(0, len(kinds), (1,), device=dev, generator=gen))] for _ in range(n)]
    full = make_logits(n, gen, kind)
    params = []
    for b in range(bs):
        greedy = (trial % 7 == 3 and b == 0) or trial % 23 == 5
        T = [1.0, 1.0, 0.6, 1.5][int(torch.randint(0, 4, (1,), device=dev, generator=gen))]
        tp = [0.95, 0.95, 1.0, 0.7, 0.3, 0.999][int(torch.randint(0, 6, (1,), device=dev, generator=gen))]
        params.append((0, 1.0, 1.0) if greedy else (1, T, tp))
    # candidates: the argmax / a sample of the row's softmax / a random token, so chains get accepted and rejected
    cand = torch.empty(n, dtype=torch.int64, device=dev)
    fl = full.float()
    for b in range(bs):
        cand[b * G] = int(torch.randint(0, V, (1,), device=dev, generator=gen))
        for j in range(G - 1):
            r = b * G + j
            u = float(torch.rand(1, device=dev, generator=gen))
            if u < 0.45:
                c = int(torch.argmax(fl[r]))
            elif u < 0.8:
                c = int(torch.multinomial(torch.softmax(fl[r] / params[b][1], dim=-1), 1, generator=gen))
            else:
                c = int(torch.randint(0, V, (1,), device=dev, generator=gen))
            cand[r + 1] = c
    coins = torch.rand(n + bs, device=dev, generator=gen)
    return bs, n, full, params, cand, coins


bad_total = 0

# ---------------- eager trials with identical uniforms ----------------
if "eager" in MODES:
    gen = torch.Generator(device=dev)
    gen.manual_seed(2024)
    n_trials = int(os.environ.get("SV_TRIALS", "150"))
    c = dict(req=0, acc_ref=0, bon_ref=0, acc_exact=0, bon_exact=0, greedy_bad=0, rank_bad=0, rows=0,
             set_exact=0, set_ref_exact=0, tv=0.0, tv_ref=0.0, tp_rel=0.0)
    dbg = torch.zeros(64 * DBG, dtype=torch.float32, device=dev)
    for trial in range(n_trials):
        bs, n, full, params, cand, coins = make_case(trial, gen)
        assert same_on_all_ranks(full) and same_on_all_ranks(cand) and same_on_all_ranks(coins), "inputs differ"
        shard = full[:, vs0 : vs0 + Vs].contiguous()
        st.stage(params)
        pred = st.run(shard, cand, G, coins=coins, dbg=dbg).clone()
        torch.cuda.synchronize()
        d = dbg.view(64, DBG)[:n].clone()  # per row: M, Z, kept mass, key*, p_target(candidate), ...
        tprobs, racc, rbon = reference(full, cand, params, coins, bs)
        kacc, kbon = decode_predict(pred, cand, bs)
        eacc, ebon, erows = exact64(full, cand, params, coins, bs)
        c["rank_bad"] += 0 if same_on_all_ranks(pred) else 1
        for b in range(bs):
            if params[b][0] == 0:
                am = torch.argmax(full[b * G : (b + 1) * G].float(), dim=-1)
                c["greedy_bad"] += 0 if torch.equal(pred[b * G : (b + 1) * G], am) else 1
                continue
            c["req"] += 1
            c["acc_ref"] += int(kacc[b]) != int(racc[b])
            c["bon_ref"] += int(kacc[b]) == int(racc[b]) and int(kbon[b]) != int(rbon[b])
            c["acc_exact"] += int(kacc[b]) != eacc[b]
            c["bon_exact"] += int(kacc[b]) == eacc[b] and int(kbon[b]) != ebon[b]
            if int(kacc[b]) != int(racc[b]) or int(kbon[b]) != int(rbon[b]):
                log(f"  mismatch trial {trial} b {b} params {params[b]}: accept kernel {int(kacc[b])} ref "
                    f"{int(racc[b])} exact {eacc[b]}, bonus kernel {int(kbon[b])} ref {int(rbon[b])} exact {ebon[b]}")
            T = params[b][1]
            for j in range(G):
                r = b * G + j
                keep, t = erows[r]
                q32 = torch.softmax(full[r].float() / T, dim=-1)
                kset = (key16(full[r]) >= int(d[r, 3])) & (q32 > 0)
                c["rows"] += 1
                c["set_exact"] += 0 if torch.equal(kset, keep) else 1
                c["set_ref_exact"] += 0 if torch.equal(tprobs[b, j] > 0, keep) else 1
                kd = torch.where(kset, q32.double(), torch.zeros_like(t))
                c["tv"] = max(c["tv"], 0.5 * float((kd / kd.sum() - t).abs().sum()))
                c["tv_ref"] = max(c["tv_ref"], 0.5 * float((tprobs[b, j].double() - t).abs().sum()))
                if j < G - 1:
                    te = float(t[int(cand[r + 1])])
                    if te > 0:
                        c["tp_rel"] = max(c["tp_rel"], abs(float(d[r, 4]) - te) / te)
        if trial % 50 == 49:
            log(f"[eager] {trial + 1} trials: {c}")
    req = max(c["req"], 1)
    fails = int(c["acc_ref"] != 0 or c["acc_exact"] != 0 or c["greedy_bad"] != 0 or c["rank_bad"] != 0)
    fails += int(c["bon_ref"] > max(2, 0.01 * req) or c["bon_exact"] > max(2, 0.01 * req))
    fails += int(c["tv"] > 1e-3)
    log(f"[eager] {n_trials} trials, {c['req']} sampled requests, {c['rows']} rows: accept-length mismatches vs "
        f"reference {c['acc_ref']} / vs exact {c['acc_exact']}; bonus mismatches vs reference {c['bon_ref']} / vs exact "
        f"{c['bon_exact']}; greedy requests != argmax {c['greedy_bad']}; ranks disagree {c['rank_bad']}")
    log(f"[eager] kept set != exact: kernel {c['set_exact']} rows, reference {c['set_ref_exact']} rows; max TV to exact: "
        f"kernel {c['tv']:.3g}, reference {c['tv_ref']:.3g}; candidate p_target max rel. err {c['tp_rel']:.3g}")
    t = torch.tensor([fails], device=dev)
    dist.all_reduce(t)
    log(f"[eager] RESULT {'OK' if int(t) == 0 else 'FAIL'}")
    bad_total += int(t)


def chi2(counts, expected):
    """chi-square statistic, dof and Wilson-Hilferty tail p-value; bins with expected < 5 pooled"""
    cnt, e = counts.double(), expected.double()
    big = e >= 5
    cs = torch.cat([cnt[big], cnt[~big].sum().view(1)])
    es = torch.cat([e[big], e[~big].sum().view(1)])
    keep = es > 0
    cs, es = cs[keep], es[keep]
    x2 = float(((cs - es) ** 2 / es).sum())
    dof = max(int(keep.sum()) - 1, 1)
    return x2, dof, chi2_tail(x2, dof)


def chi2_tail(x2, dof):
    z = ((x2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    return 0.5 * math.erfc(z / math.sqrt(2))


# ---------------- in-kernel Philox uniforms vs the exact distributions ----------------
if "stat" in MODES:
    from sglang.kernels.ops.speculative.sampling import tree_speculative_sampling_target_only
    from sglang.srt.speculative.dflash_utils import _get_or_create_chain_verify_buffers

    gen = torch.Generator(device=dev)
    gen.manual_seed(77)
    bs, n = 4, 32
    full = make_logits(n, gen, ["peak5"] * 8 + ["peak50"] * 8 + ["flat"] * 8 + ["ties"] * 8)
    params = [(1, 1.0, 0.95), (1, 0.7, 0.9), (1, 1.0, 1.0), (1, 1.3, 0.95)]
    tprobs = build_speculative_verify_target_probs(
        next_token_logits=full.float(), sampling_info=sampling_info_for(params), draft_token_num=G, bs=bs
    )
    # candidates with a moderate target probability at every level (1st-3rd most likely): all lengths occur
    cand = torch.empty(n, dtype=torch.int64, device=dev)
    for b in range(bs):
        cand[b * G] = 11
        for j in range(G - 1):
            cand[b * G + j + 1] = torch.argsort(tprobs[b, j], descending=True)[j % 3]
    exp_len = torch.zeros(bs, G, dtype=torch.float64)
    exp_tok = {}
    for b in range(bs):
        surv = 1.0
        for k in range(G):
            if k < G - 1:
                pk = float(tprobs[b, k, int(cand[b * G + k + 1])])
                exp_len[b, k] = surv * (1.0 - pk)
                dk = tprobs[b, k].double().clone()
                dk[int(cand[b * G + k + 1])] = 0.0
                if float(dk.sum()) > 0 and exp_len[b, k] > 0:
                    exp_tok[(b, k)] = (dk / dk.sum()).cpu()
                surv *= pk
            else:
                exp_len[b, k] = surv
                exp_tok[(b, k)] = tprobs[b, k].double().cpu()
    shard = full[:, vs0 : vs0 + Vs].contiguous()
    st.stage(params)
    K = int(os.environ.get("SV_STAT_N", "20000"))
    outs = torch.empty(K, n, dtype=torch.int64, device=dev)
    for i in range(K):
        st.run(shard, cand, G, out=outs[i])
    torch.cuda.synchronize()
    ok_rank = same_on_all_ranks(outs)
    # the reference sampling kernel with torch.rand uniforms on the same target probabilities
    ref_acc = torch.empty(K, bs, dtype=torch.long, device=dev)
    ref_bon = torch.empty(K, bs, dtype=torch.long, device=dev)
    rg = torch.Generator(device=dev)
    rg.manual_seed(5)
    for i in range(K):
        ri, rnt, rns, pr, ai, atn = _get_or_create_chain_verify_buffers(bs=bs, draft_token_num=G, device=dev)
        tree_speculative_sampling_target_only(
            predicts=pr, accept_index=ai, accept_token_num=atn, candidates=cand.view(bs, G), retrive_index=ri,
            retrive_next_token=rnt, retrive_next_sibling=rns,
            uniform_samples=torch.rand(bs, G, device=dev, generator=rg),
            uniform_samples_for_final_sampling=torch.rand(bs, device=dev, generator=rg), target_probs=tprobs,
            draft_probs=torch.zeros_like(tprobs), threshold_single=1.0, threshold_acc=1.0, deterministic=True,
        )
        ref_acc[i] = atn.long()
        ref_bon[i] = pr[ai[torch.arange(bs, device=dev), atn.long()].long()].long()
    torch.cuda.synchronize()
    o = outs.cpu()
    match = (o.view(K, bs, G)[:, :, : G - 1] == cand.cpu().view(bs, G)[:, 1:].unsqueeze(0)).long()
    kacc_all = torch.cumprod(match, dim=2).sum(dim=2)  # [K, bs]
    kbon_all = o.view(K, bs, G).gather(2, kacc_all.unsqueeze(2)).squeeze(2)
    pmin = 1.0
    for name, acc_s, bon_s in (("kernel", kacc_all, kbon_all), ("reference", ref_acc.cpu(), ref_bon.cpu())):
        for b in range(bs):
            acc_b, bon_b = acc_s[:, b], bon_s[:, b]
            hist = torch.bincount(acc_b, minlength=G)[:G]
            x2, dof, pv = chi2(hist, exp_len[b] * K)
            x2t, doft = 0.0, 0
            for k in range(G):
                sel = acc_b == k
                m = int(sel.sum())
                if m < 50 or (b, k) not in exp_tok:
                    continue
                e = exp_tok[(b, k)] * m
                top = torch.argsort(e, descending=True)[:200]
                cnt = torch.bincount(bon_b[sel], minlength=V).double()
                xx, dd, _ = chi2(torch.cat([cnt[top], (m - cnt[top].sum()).view(1)]),
                                 torch.cat([e[top], (m - e[top].sum()).view(1)]))
                x2t += xx
                doft += dd
            pt = chi2_tail(x2t, doft) if doft else 1.0
            if name == "kernel":
                pmin = min(pmin, pv, pt)
            log(f"[stat] {name} req{b} {params[b]}: accept-length chi2 {x2:.1f}/{dof} p={pv:.3f}; bonus chi2 "
                f"{x2t:.1f}/{doft} p={pt:.3f}; mean accept {float(acc_b.double().mean()):.4f} exp "
                f"{float((exp_len[b] * torch.arange(G, dtype=torch.float64)).sum()):.4f}")
    ok = ok_rank and pmin > 1e-3
    log(f"[stat] RESULT {'OK' if ok else 'FAIL'} (ranks agree {ok_rank}, min kernel p-value {pmin:.4f}, N={K})")
    bad_total += 0 if ok else 1

# ---------------- a few Philox calls (compute-sanitizer runs) ----------------
if "san" in MODES:
    gen = torch.Generator(device=dev)
    gen.manual_seed(99)
    sbad = 0
    for it in range(4):
        _, _, full, params, cand, _ = make_case(it * 8, gen)
        st.stage(params)
        o = st.run(full[:, vs0 : vs0 + Vs].contiguous(), cand, G).clone()
        torch.cuda.synchronize()
        sbad += 0 if same_on_all_ranks(o) else 1
        # skip_greedy: the same batch (runs unless all-greedy) and its all-greedy version (returns at once)
        o2 = torch.full_like(o, -7)
        st.run(full[:, vs0 : vs0 + Vs].contiguous(), cand, G, out=o2, skip_greedy=True)
        st.stage_all_greedy(len(params))
        calls0 = st.calls.clone()
        o3 = torch.full_like(o, -7)
        st.run(full[:, vs0 : vs0 + Vs].contiguous(), cand, G, out=o3, skip_greedy=True)
        torch.cuda.synchronize()
        sbad += 0 if same_on_all_ranks(o2) and bool((o3 == -7).all()) and torch.equal(calls0, st.calls) else 1
    log(f"[san] RESULT {'OK' if sbad == 0 else 'FAIL'} (Philox calls, ranks agree; skip_greedy early return)")
    bad_total += sbad

# ---------------- CUDA graph replays ----------------
if "graph" in MODES:
    gen = torch.Generator(device=dev)
    gen.manual_seed(31337)
    bs, n = 4, 32
    s_logits = torch.zeros(n, Vs, dtype=torch.bfloat16, device=dev)
    s_cand = torch.zeros(n, dtype=torch.int64, device=dev)
    s_coins = torch.zeros(n + bs, dtype=torch.float32, device=dev)
    s_out = torch.zeros(n, dtype=torch.int64, device=dev)
    st.stage([(1, 1.0, 0.95)] * bs)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        st.run(s_logits, s_cand, G, out=s_out, coins=s_coins)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr, stream=s):
            st.run(s_logits, s_cand, G, out=s_out, coins=s_coins)
    torch.cuda.synchronize()
    gbad = bon_diff = sampled = 0
    for it in range(150):
        _, _, full, params, cand, coins = make_case(8 * it, gen)  # batch-4 cases
        s_logits.copy_(full[:, vs0 : vs0 + Vs])
        s_cand.copy_(cand)
        s_coins.copy_(coins)
        st.stage(params)
        saved = st.calls.clone()
        gr.replay()
        torch.cuda.synchronize()
        g_out = s_out.clone()
        st.calls.copy_(saved)
        e_out = st.run(s_logits, s_cand, G, out=torch.zeros_like(s_out), coins=s_coins).clone()
        if it % 3 == 1:  # extra eager calls on the shared counters between replays
            st.run(s_logits, s_cand, G, coins=s_coins)
        torch.cuda.synchronize()
        _, racc, rbon = reference(full, cand, params, coins, bs)
        kacc, kbon = decode_predict(g_out, cand, bs)
        ok = torch.equal(g_out, e_out) and same_on_all_ranks(g_out)
        for b in range(bs):
            if params[b][0] == 0:
                ok &= torch.equal(g_out[b * G : (b + 1) * G], torch.argmax(full[b * G : (b + 1) * G].float(), dim=-1))
            else:
                ok &= int(kacc[b]) == int(racc[b])
                bon_diff += int(int(kbon[b]) != int(rbon[b]))
                sampled += 1
        gbad += 0 if ok else 1
    # Philox path: graph replay == eager call with the same counters
    with torch.cuda.stream(s):
        gr2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr2, stream=s):
            st.run(s_logits, s_cand, G, out=s_out)
    torch.cuda.synchronize()
    pbad = 0
    e_out = torch.zeros(n, dtype=torch.int64, device=dev)
    for it in range(40):
        _, _, full, params, cand, _ = make_case(8 * it, gen)
        s_logits.copy_(full[:, vs0 : vs0 + Vs])
        s_cand.copy_(cand)
        st.stage(params)
        saved = st.calls.clone()
        gr2.replay()
        torch.cuda.synchronize()
        st.calls.copy_(saved)
        st.run(s_logits, s_cand, G, out=e_out)
        torch.cuda.synchronize()
        pbad += 0 if (torch.equal(e_out, s_out) and same_on_all_ranks(s_out)) else 1
    t = torch.tensor([gbad, pbad], device=dev)
    dist.all_reduce(t)
    log(f"[graph] bonus differences vs the reference (fp32 CDF boundary): {bon_diff} of {sampled} sampled requests")
    log(f"[graph] RESULT {'OK' if int(t.sum()) == 0 else 'FAIL'}: 150 replays (bitwise == eager, accept == reference, "
        f"ranks agree) {int(t[0])} bad; Philox graph replay vs eager {int(t[1])} bad of 40 (sums over ranks)")
    bad_total += int(t.sum())

# ---------------- fast path: the worker's capture layout (argmax on the main stream, kernel on a fork) ----------------
def fork_layout(tps, shard, cand, head_out, out, coins=None):
    """argmax on the current stream; the sampling verify with skip_greedy on a forked stream; join"""
    main = torch.cuda.current_stream()
    ev = torch.cuda.Event()
    ev.record(main)
    tps.argmax(shard, head_out, vs0)
    fork_stream.wait_event(ev)
    with torch.cuda.stream(fork_stream):
        st.run(shard, cand, G, out=out, coins=coins, skip_greedy=True)
    main.wait_stream(fork_stream)


fork_stream = torch.cuda.Stream()

if "fast" in MODES:
    from sglang.kernels.ops.communication.dflash_tp_fused import DFlashTPState

    tps = DFlashTPState(dev, cpu, list(range(ws)), rank)
    gen = torch.Generator(device=dev)
    gen.manual_seed(5150)
    bs, n = 4, 32
    s_logits = torch.zeros(n, Vs, dtype=torch.bfloat16, device=dev)
    s_cand = torch.zeros(n, dtype=torch.int64, device=dev)
    s_coins = torch.zeros(n + bs, dtype=torch.float32, device=dev)
    s_out = torch.zeros(n, dtype=torch.int64, device=dev)
    s_head = torch.zeros(32, dtype=torch.int64, device=dev)
    st.stage_all_greedy(bs)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fork_layout(tps, s_logits, s_cand, s_head, s_out, s_coins)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr, stream=s):
            fork_layout(tps, s_logits, s_cand, s_head, s_out, s_coins)
    torch.cuda.synchronize()
    cnt = dict(greedy=0, mixed=0, sampled=0)
    fbad = 0
    n_fast = int(os.environ.get("SV_FAST_N", "150"))
    for it in range(n_fast):
        _, _, full, params, cand, coins = make_case(8 * it, gen)  # batch-4 cases
        kind = it % 3  # 0 all greedy, 1 mixed, 2 as drawn (mostly all sampled)
        if kind == 0:
            params = [(0, 1.0, 1.0)] * bs
        elif kind == 1:
            params = [(0, 1.0, 1.0) if b % 2 == 0 else (p if p[0] else (1, 1.0, 0.95)) for b, p in enumerate(params)]
        s_logits.copy_(full[:, vs0 : vs0 + Vs])
        s_cand.copy_(cand)
        s_coins.copy_(coins)
        s_out.fill_(-7)
        s_head.fill_(-7)
        st.stage(params)
        saved = st.calls.clone()
        gr.replay()
        torch.cuda.synchronize()
        g_out, g_head = s_out.clone(), s_head[:n].clone()
        am = torch.argmax(full.float(), dim=-1)
        ok = torch.equal(g_head, am) and same_on_all_ranks(g_out)
        if all(p[0] == 0 for p in params):
            cnt["greedy"] += 1
            ok &= bool((g_out == -7).all()) and torch.equal(saved, st.calls)
        else:
            cnt["mixed" if any(p[0] == 0 for p in params) else "sampled"] += 1
            st.calls.copy_(saved)
            e_out = st.run(s_logits, s_cand, G, out=torch.zeros_like(s_out), coins=s_coins).clone()
            torch.cuda.synchronize()
            ok &= torch.equal(g_out, e_out)
            _, racc, _ = reference(full, cand, params, coins, bs)
            kacc, _ = decode_predict(g_out, cand, bs)
            for b in range(bs):
                if params[b][0] == 0:
                    ok &= torch.equal(g_out[b * G : (b + 1) * G], am[b * G : (b + 1) * G])
                else:
                    ok &= int(kacc[b]) == int(racc[b])
        fbad += 0 if ok else 1
    t = torch.tensor([fbad], device=dev)
    dist.all_reduce(t)
    log(f"[fast] RESULT {'OK' if int(t) == 0 else 'FAIL'}: {n_fast} replays of the fork layout ({cnt['greedy']} all-greedy, "
        f"{cnt['mixed']} mixed, {cnt['sampled']} sampled batches): argmax == torch.argmax, all-greedy -> kernel output / "
        f"counters untouched, otherwise kernel == eager and accept == reference; {int(t)} bad (sum over ranks)")
    bad_total += int(t)


def graph_time(fn, calls=20, replays=10):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr, stream=s):
            for _ in range(calls):
                fn()
    torch.cuda.synchronize()
    for _ in range(5):
        gr.replay()
    torch.cuda.synchronize()
    dist.barrier()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(replays):
        gr.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / (calls * replays)


# ---------------- timing ----------------
if "time" in MODES:
    from sglang.kernels.ops.communication.dflash_tp_fused import DFlashTPState

    gen = torch.Generator(device=dev)
    gen.manual_seed(9)
    tps = DFlashTPState(dev, cpu, list(range(ws)), rank)
    for bs in (4, 1, 8):
        n = bs * G
        _, _, full, _, cand, _ = make_case({4: 0, 1: 3, 8: 6}[bs], gen)
        shard = full[:, vs0 : vs0 + Vs].contiguous()
        res = {}
        for label, prm in (("sampled p0.95", [(1, 1.0, 0.95)] * bs), ("sampled p1.0", [(1, 1.0, 1.0)] * bs),
                           ("greedy", [(0, 1.0, 1.0)] * bs)):
            st.stage(prm)
            res[label] = graph_time(lambda: st.run(shard, cand, G))
        if n <= 32:
            o = torch.zeros(32, dtype=torch.int64, device=dev)
            res["in-graph argmax"] = graph_time(lambda: tps.argmax(shard, o, vs0))
            so = torch.zeros(n, dtype=torch.int64, device=dev)
            st.stage([(0, 1.0, 1.0)] * bs)
            res["fork layout greedy"] = graph_time(lambda: fork_layout(tps, shard, cand, o, so))
            st.stage([(1, 1.0, 0.95)] * bs)
            res["fork layout sampled p0.95"] = graph_time(lambda: fork_layout(tps, shard, cand, o, so))
        si = sampling_info_for([(1, 1.0, 0.95)] * bs)

        def ref_once():  # the eager path: all-gather + fp32 + probabilities + sampling kernel + two broadcasts
            g = [torch.empty_like(shard) for _ in range(ws)]
            dist.all_gather(g, shard)
            a, bo = compute_dflash_sampling_correct_drafts_and_bonus(
                candidates=cand.view(bs, G), next_token_logits=torch.cat(g, dim=1).float(), sampling_info=si,
                threshold_single=1.0, threshold_acc=1.0,
            )
            dist.broadcast(a, src=0)
            dist.broadcast(bo, src=0)

        for _ in range(5):
            ref_once()
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(50):
            ref_once()
        torch.cuda.synchronize()
        res["eager reference (wall)"] = (time.perf_counter() - t0) * 1e6 / 50
        log(f"[time] bs {bs} (n {n}): " + ", ".join(f"{k} {v:.1f} us" for k, v in res.items()))

# ---------------- per-phase stamps ----------------
if "stamps" in MODES:
    gen = torch.Generator(device=dev)
    gen.manual_seed(4242)
    names = ["load+count+argmax", "masses", "r1 push", "r1 poll", "scan d1", "r2 poll", "d2+SK", "flags",
             "pre-scan", "final scan", "r3"]
    for kname in ("peak5", "flat", "ties", "peak50"):
        bs, n = 4, 32
        full = make_logits(n, gen, [kname] * n)
        shard = full[:, vs0 : vs0 + Vs].contiguous()
        cand = torch.roll(torch.argmax(full.float(), dim=-1), 1)
        for prm in ([(1, 1.0, 0.95)] * bs, [(1, 1.0, 1.0)] * bs):
            st.stage(prm)
            stamps = torch.zeros(64 * 16, dtype=torch.int64, device=dev)
            acc = torch.zeros(12, dtype=torch.float64)
            cnt = torch.zeros(12, dtype=torch.float64)
            span = []
            # back-to-back calls in one graph (ranks stay in lockstep through the exchanges); stamps of the last call
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                st.run(shard, cand, G, stamps=stamps)
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr, stream=s):
                    for _ in range(12):
                        st.run(shard, cand, G, stamps=stamps)
            torch.cuda.synchronize()
            for it in range(30):
                stamps.zero_()
                gr.replay()
                torch.cuda.synchronize()
                if it < 5:
                    continue
                sm = stamps.view(64, 16)[:n].cpu().double()
                t0 = sm[:, 0].min()
                for k in range(1, 12):
                    v = sm[:, k][sm[:, k] >= t0]  # rows that reached phase k in the last call
                    if v.numel():
                        acc[k] += float((v - t0).mean())
                        cnt[k] += 1
                span.append(float(sm.max() - t0) / 1000)
            line = ", ".join(f"{names[k - 1]} {acc[k] / cnt[k] / 1000:.1f}" for k in range(1, 12) if cnt[k] > 0)
            log(f"[stamps] {kname} top_p {prm[0][2]}: span {sum(span) / len(span):.1f} us; phase ends (us): {line}")

t = torch.tensor([bad_total], device=dev)
dist.all_reduce(t)
log(f"SV_TEST {'OK' if int(t) == 0 else 'FAIL'} ({int(t)})")
dist.destroy_process_group()
