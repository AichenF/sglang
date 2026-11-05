"""1-GPU test of the MiMo-V2 per-token-slot attention kernels on a KV pool allocated at page_size > 1.

    python3 test/manual/kernels/test_mimo_attention_token_tables.py [--page 64] [--seqlens 65543,65601,70000,65536]
    python3 test/manual/kernels/test_mimo_attention_token_tables.py --page 16 --seqlens 65543,129,70000,4097

At page_size > 1 the FA backend builds token-level target-verify tables (FlashAttentionMetadata.page_table_1 /
swa_page_table_1) next to FA3's block tables, and the kernels that address the KV pool per token slot read those over a
token-major view of the pool. The test builds the same K/V content (per request and position) twice: in a page-1 pool
(every token an independent random slot) and in a page-P pool (each request owns random whole pages, plus a
page-granular full -> SWA map), and runs every kernel on both.

Sections:
    tables   build_trtllm_mha_page_table(page P, page_table_1=, swa_page_table_1=) == the original builder at page P
             (block tables) and at page 1 (token tables), full and SWA, live prefixes and untouched tails.
    kernels  page-P layout + token tables vs page-1 layout, every output bitwise: fa_block (full-attention layers:
             attention output, o_proj partial, the stored K/V rows), fa_verify, swa_decode (window 128 + sinks),
             attn_block.run and run_mega (SWA layers, window 128 + sinks; run_mega also its qkv rows), bidir_decode
             (DFlash draft layers: window 1023 on both sides + sinks). fa_block / attn_block also against their unfused
             chain (rope_kv -> fa_verify / swa_decode -> o_proj) within one bf16 ulp. The fused blocks run in
             MODE_FULL_LOCAL (one GPU, every o_proj row local).

Prints PASS / FAIL and exits non-zero on a failure.
"""

import argparse
import sys

import torch

B, W = 4, 8  # requests, verify tokens per request
M, NH, D, DV, DIM = B * W, 16, 192, 128, 6144  # rows, q heads per rank, qk / v head dims, hidden size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlens", default="65543,65601,70000,65536")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--page", type=int, default=64)
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.manual_seed(0)
    PAGE = a.page
    seqlens = [int(x) for x in a.seqlens.split(",")]
    assert len(seqlens) == B
    S = ((max(seqlens) + 64 + PAGE - 1) // PAGE) * PAGE  # table width (tokens), a multiple of PAGE
    NPG = S // PAGE
    nslots = B * S + 64 * PAGE  # pool slots (a multiple of PAGE); slot 0 = padding
    RPI = torch.tensor([3, 1, 6, 4], dtype=torch.int64, device=dev)  # request pool rows (non-trivial)
    seq_t = torch.tensor(seqlens, dtype=torch.int32, device=dev)
    fails = []

    def check(name, ok, extra=""):
        print(f"  {name:62s} {'OK' if ok else 'MISMATCH'} {extra}", flush=True)
        if not ok:
            fails.append(name)

    # -------------------------------------------------------------------------------------------- layouts
    def layout_p1(seed):
        """page-1 allocation: every token an independent random slot."""
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(nslots - 1, generator=g)[: B * S].to(torch.int64) + 1
        return perm.view(B, S)

    def layout_pg(seed):
        """page-P allocation: each request owns random whole pages; token i -> page[i // P] * P + i % P."""
        g = torch.Generator().manual_seed(seed)
        pages = torch.randperm(nslots // PAGE - 1, generator=g)[: B * NPG].to(torch.int64) + 1
        pages = pages.view(B, NPG)
        off = torch.arange(S, dtype=torch.int64)
        return pages[:, off // PAGE] * PAGE + off % PAGE

    def f2s_p1(seed):
        g = torch.Generator().manual_seed(seed)
        m = torch.randperm(nslots - 1, generator=g).to(torch.int64) + 1
        return torch.cat([torch.zeros(1, dtype=torch.int64), m])  # slot 0 -> 0

    def f2s_pg(seed):
        g = torch.Generator().manual_seed(seed)
        m = torch.randperm(nslots // PAGE - 1, generator=g).to(torch.int64) + 1
        m = torch.cat([torch.zeros(1, dtype=torch.int64), m])
        s = torch.arange(nslots, dtype=torch.int64)
        return m[s // PAGE] * PAGE + s % PAGE  # page-granular full -> SWA map

    tok = {"p1": layout_p1(11), "pg": layout_pg(12)}  # [B, S] full slots per (request, position)
    f2s = {"p1": f2s_p1(13), "pg": f2s_pg(14)}
    for k in tok:
        assert tok[k].unique().numel() == B * S

    def r2t_of(t):
        r2t = torch.zeros(8, S, dtype=torch.int32, device=dev)
        r2t[RPI] = t.to(torch.int32).to(dev)
        return r2t

    # -------------------------------------------------------------------------------------------- tables
    print("== tables", flush=True)
    from sglang.kernels.ops.kvcache.trtllm_mha_page_table import build_trtllm_mha_page_table

    tabs = {}
    for k in ("p1", "pg"):
        r2t = r2t_of(tok[k])
        f2s_d = f2s[k].to(dev)
        if k == "pg":
            # block tables + token tables in one pass
            pt = torch.full((B, NPG), -7, dtype=torch.int32, device=dev)
            spt = torch.full((B, NPG), -7, dtype=torch.int32, device=dev)
            t1 = torch.full((B, S), -7, dtype=torch.int32, device=dev)
            st1 = torch.full((B, S), -7, dtype=torch.int32, device=dev)
            build_trtllm_mha_page_table(r2t, RPI, seq_t, pt, PAGE, spt, f2s_d, page_table_1=t1, swa_page_table_1=st1)
            # references: the original builder at page P and at page 1
            rpt, rspt = torch.full_like(pt, -7), torch.full_like(spt, -7)
            build_trtllm_mha_page_table(r2t, RPI, seq_t, rpt, PAGE, rspt, f2s_d)
            rt1, rst1 = torch.full_like(t1, -7), torch.full_like(st1, -7)
            build_trtllm_mha_page_table(r2t, RPI, seq_t, rt1, 1, rst1, f2s_d)
            torch.cuda.synchronize()
            check(f"block table == original builder (page {PAGE}), incl. tails", torch.equal(pt, rpt))
            check(f"SWA block table == original builder (page {PAGE})", torch.equal(spt, rspt))
            check("token table == original builder (page 1), incl. tails", torch.equal(t1, rt1))
            check("SWA token table == original builder (page 1)", torch.equal(st1, rst1))
            live = all(torch.equal(t1[b, : seqlens[b]].cpu().long(), tok[k][b, : seqlens[b]]) for b in range(B))
            check("token table live prefix == req_to_token rows", live)
            slive = all(
                torch.equal(st1[b, : seqlens[b]].cpu().long(), f2s[k][tok[k][b, : seqlens[b]]]) for b in range(B)
            )
            check("SWA token table live prefix == full_to_swa[req_to_token]", slive)
            check("token table tails untouched", all(bool((t1[b, seqlens[b] :] == -7).all()) for b in range(B)))
            np_live = [(s + PAGE - 1) // PAGE for s in seqlens]
            blk = all(
                torch.equal(pt[b, : np_live[b]].cpu().long(), tok[k][b, : np_live[b] * PAGE : PAGE] // PAGE)
                for b in range(B)
            )
            check("block table live prefix == req_to_token[::page] // page", blk)
        else:
            t1 = torch.zeros(B, S, dtype=torch.int32, device=dev)
            st1 = torch.zeros(B, S, dtype=torch.int32, device=dev)
            build_trtllm_mha_page_table(r2t, RPI, seq_t, t1, 1, st1, f2s_d)  # page 1: block table == token table
        tabs[k] = (t1, st1)

    # -------------------------------------------------------------------------------------------- content
    g = torch.Generator(device=dev)
    g.manual_seed(1)
    kc_content = (torch.randn(B, S, D, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    vc_content = torch.randn(B, S, DV, generator=g, device=dev).to(torch.bfloat16)
    dk_content = (torch.randn(B, S, 128, generator=g, device=dev) * 0.5).to(torch.bfloat16)  # draft layers: 128 / 128
    dv_content = torch.randn(B, S, 128, generator=g, device=dev).to(torch.bfloat16)

    def slots_of(k, swa):
        return f2s[k][tok[k]] if swa else tok[k]

    def fill(nrow, content, slots):
        c = torch.zeros(nslots, 1, nrow, dtype=torch.bfloat16, device=dev)
        c[slots.to(dev).view(-1)] = content.reshape(-1, 1, nrow)
        return c

    def caches(k, swa=False, draft=False):
        s = slots_of(k, swa)
        if draft:
            return fill(128, dk_content, s), fill(128, dv_content, s)
        return fill(D, kc_content, s), fill(DV, vc_content, s)

    def gather_rows(cache, k, swa=False):
        s = slots_of(k, swa)
        return torch.stack([cache[s[b, seqlens[b] - W : seqlens[b]].to(dev), 0] for b in range(B)])

    def new_loc(k, swa=False):
        s = slots_of(k, swa)
        return torch.stack([s[b, seqlens[b] - W + i] for b in range(B) for i in range(W)]).long().to(dev)

    positions = torch.cat([torch.arange(s - W, s, device=dev) for s in seqlens]).long()
    inv = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32, device=dev) / 64))
    fr = torch.outer(torch.arange(S + W, dtype=torch.float32, device=dev), inv)
    cos_sin = torch.cat([fr.cos(), fr.sin()], -1).to(torch.bfloat16).contiguous()
    sm_scale = D**-0.5
    sinks = (torch.randn(NH, generator=g, device=dev) * 0.5).float().contiguous()
    Wo = (torch.randn(DIM, NH * DV, generator=g, device=dev) * 0.02).to(torch.bfloat16)

    def new_qkv(it):
        gg = torch.Generator(device=dev)
        gg.manual_seed(777 + 31 * it)
        return (torch.randn(M, 3392, generator=gg, device=dev) * 0.5).to(torch.bfloat16)

    def ulp_close(ref, out):  # <= 1 bf16 ulp of max |ref|
        d = (ref.float() - out.float()).abs().max().item()
        return d <= ref.float().abs().max().item() * 2**-7, f"max diff {d:.3e}"

    # -------------------------------------------------------------------------------------------- fa_block
    print("== fa_block (full-attention layers, MODE_FULL_LOCAL)", flush=True)
    from sglang.kernels.ops.attention import mimo_rope_kv as RK
    from sglang.kernels.ops.attention.fa_verify import fa_verify_attention
    from sglang.srt.layers.moe.mimo_fused_moe import fa_block as FB

    E = FB.ext()
    wmap = E.make_wmap(Wo)
    work = torch.zeros(4, dtype=torch.int32, device=dev)
    peers = torch.zeros(8, dtype=torch.int64)
    sc = FB.Scratch(dev)
    for it in range(a.iters):
        qkv = new_qkv(it)
        out = {}
        for k in ("p1", "pg"):
            kc, vc = caches(k)
            part = torch.zeros(M, DIM, dtype=torch.bfloat16, device=dev)
            work[2] += 1
            FB.run(qkv, cos_sin, positions, new_loc(k), kc, vc, tabs[k][0], seq_t, sc, part, wmap, peers, work, 0,
                   sm_scale, E.MODE_FULL_LOCAL, pdl=True)
            torch.cuda.synchronize()
            out[k] = (sc.attn_out.clone(), part, gather_rows(kc, k), gather_rows(vc, k))
            if k == "pg":  # vs the unfused chain: rope_kv -> fa_verify (token tables) -> fp32 o_proj
                kc2, vc2 = caches(k)
                q2 = qkv.clone()
                RK.run(q2, cos_sin, positions, new_loc(k), kc2, vc2)
                att = fa_verify_attention(q2[:, : NH * D].view(M, NH, D), kc2, vc2, tabs[k][0], seq_t,
                                          sm_scale=sm_scale).reshape(M, NH * DV)
                torch.cuda.synchronize()
                check(f"it{it} fa_block attn vs unfused chain (<= 1 bf16 ulp)", *ulp_close(att, out[k][0]))
                check(f"it{it} fa_block K/V rows == rope_kv's",
                      torch.equal(gather_rows(kc2, k), out[k][2]) and torch.equal(gather_rows(vc2, k), out[k][3]))
                dp = (out[k][1].float() - out[k][0].float() @ Wo.float().t()).abs().max().item()
                check(f"it{it} fa_block partial vs fp32 o_proj", dp < 0.02, f"max diff {dp:.3e}")
        for n, i in (("attn_out", 0), ("o_proj partial", 1), ("stored K rows", 2), ("stored V rows", 3)):
            check(f"it{it} fa_block {n}: page {PAGE} == page 1", torch.equal(out["p1"][i], out["pg"][i]))

    # -------------------------------------------------------------------------------------------- fa_verify
    print("== fa_verify (per-layer full-attention kernel)", flush=True)
    for it in range(a.iters):
        q = new_qkv(50 + it)[:, : NH * D].view(M, NH, D)
        o = {}
        for k in ("p1", "pg"):
            kc, vc = caches(k)
            o[k] = fa_verify_attention(q, kc, vc, tabs[k][0], seq_t, sm_scale=sm_scale).clone()
        torch.cuda.synchronize()
        check(f"it{it} fa_verify out: page {PAGE} == page 1", torch.equal(o["p1"], o["pg"]))

    # -------------------------------------------------------------------------------------------- swa_decode
    print("== swa_decode (per-layer SWA kernel, window 128, sinks)", flush=True)
    from sglang.kernels.ops.attention.swa_decode import swa_decode_attention

    for it in range(a.iters):
        q = new_qkv(80 + it)[:, : NH * D].view(M, NH, D)
        o = {}
        for k in ("p1", "pg"):
            kc, vc = caches(k, swa=True)
            o[k] = swa_decode_attention(q, kc, vc, tabs[k][1], seq_t, window_tokens=W, window_size=128,
                                        sm_scale=sm_scale, sinks=sinks).clone()
        torch.cuda.synchronize()
        check(f"it{it} swa_decode out: page {PAGE} == page 1", torch.equal(o["p1"], o["pg"]))

    # -------------------------------------------------------------------------------------------- attn_block
    print("== attn_block.run / run_mega (SWA layers, window 128, sinks, MODE_FULL_LOCAL)", flush=True)
    from sglang.srt.layers.moe.mimo_fused_moe import attn_block as AB

    EA = AB.ext()
    asc = AB.Scratch(dev)
    awmap = EA.make_wmap(Wo)
    qw = (torch.randn(3392, DIM, generator=g, device=dev) * 0.05).to(torch.float8_e4m3fn).contiguous()
    qsw = (torch.rand(27, 48, generator=g, device=dev) * 0.02 + 0.01).float().contiguous()
    qwmap = EA.make_qwmap(qw)
    mode = EA.MODE_FULL_LOCAL | EA.MODE_NO_PUSH  # one GPU: every o_proj row local, no reduce-scatter push
    for it in range(a.iters):
        qkv = new_qkv(120 + it)
        xq = torch.randn(M, DIM, generator=g, device=dev).to(torch.float8_e4m3fn).contiguous()
        xs = (torch.rand(48, M, generator=g, device=dev) * 0.1 + 0.01).float().contiguous().t()  # column-major
        for mega in (False, True):
            out = {}
            for k in ("p1", "pg"):
                kc, vc = caches(k, swa=True)
                part = torch.zeros(M, DIM, dtype=torch.bfloat16, device=dev)
                work[2] += 1
                if not mega:
                    AB.run(qkv, cos_sin, positions, new_loc(k, swa=True), kc, vc, tabs[k][1], seq_t, sinks, asc, part,
                           awmap, peers, work, 0, sm_scale, 128, mode, pdl=True)
                else:
                    AB.run_mega(xq, xs, qwmap, qsw, cos_sin, positions, new_loc(k, swa=True), kc, vc, tabs[k][1],
                                seq_t, sinks, asc, part, awmap, peers, work, 0, sm_scale, 128, mode, pdl=True, qpre=16)
                torch.cuda.synchronize()
                out[k] = (asc.attn_out.clone(), part, gather_rows(kc, k, swa=True), gather_rows(vc, k, swa=True),
                          asc.qout.clone() if mega else None)
                if k == "pg" and not mega:  # vs the unfused chain: rope_kv -> swa_decode (token tables)
                    kc2, vc2 = caches(k, swa=True)
                    q2 = qkv.clone()
                    RK.run(q2, cos_sin, positions, new_loc(k, swa=True), kc2, vc2)
                    att = swa_decode_attention(q2[:, : NH * D].view(M, NH, D), kc2, vc2, tabs[k][1], seq_t,
                                               window_tokens=W, window_size=128, sm_scale=sm_scale,
                                               sinks=sinks).reshape(M, NH * DV)
                    torch.cuda.synchronize()
                    # within one bf16 ulp, not bitwise, in this standalone harness at any page size (page 1 gives
                    # the same bits); the page comparison below is bitwise
                    check(f"it{it} attn_block attn vs swa_decode chain (<= 1 bf16 ulp)", *ulp_close(att, out[k][0]))
                    check(f"it{it} attn_block K/V rows == rope_kv's",
                          torch.equal(gather_rows(kc2, k, swa=True), out[k][2])
                          and torch.equal(gather_rows(vc2, k, swa=True), out[k][3]))
            nm = "run_mega" if mega else "run"
            for n, i in (("attn_out", 0), ("o_proj partial", 1), ("stored K rows", 2), ("stored V rows", 3)):
                check(f"it{it} attn_block.{nm} {n}: page {PAGE} == page 1", torch.equal(out["p1"][i], out["pg"][i]))
            if mega:
                check(f"it{it} attn_block.run_mega qkv rows: page {PAGE} == page 1",
                      torch.equal(out["p1"][4], out["pg"][4]))

    # -------------------------------------------------------------------------------------------- bidir_decode
    print("== bidir_decode (DFlash draft layers: window 1023 on both sides, sinks)", flush=True)
    from sglang.kernels.ops.attention.bidir_decode import bidir_decode_attention

    for it in range(a.iters):
        gq = torch.Generator(device=dev)
        gq.manual_seed(200 + it)
        qd = (torch.randn(M, 16, 128, generator=gq, device=dev) * 0.5).to(torch.bfloat16)
        o = {}
        for k in ("p1", "pg"):
            kc, vc = caches(k, swa=True, draft=True)
            o[k] = bidir_decode_attention(qd, kc, vc, tabs[k][1], seq_t, block_tokens=W, sm_scale=128**-0.5,
                                          window_size=(1023, 1023), sinks=sinks).clone()
        torch.cuda.synchronize()
        check(f"it{it} bidir_decode out: page {PAGE} == page 1", torch.equal(o["p1"], o["pg"]))

    print("PASS" if not fails else f"FAIL {fails}", flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
