"""The DFlash decode step's replay prep as ONE launch per CUDA graph (SGLANG_STEP_PREP_FUSED).

Every decode step runs two eager prep sequences in front of its two CUDA graphs:

  draft graph:  prepare_block -> fill_from copy (input_ids, positions, out_cache_loc, req_pool_indices, seq_lens) ->
                FA metadata (cache_seqlens, cu_seqlens_k) -> page table (create_trtllm_mha_kv_indices)
  verify graph: fill_from copy -> SWA write-loc refill -> FA metadata -> page table + SWA page table

Each launch is ~1.1-1.7 us plus ~1.2 us of launch gap although the work is tiny. While a recording window is open
(``begin`` .. ``flush``), the hooked helpers (``_prepare_dflash_draft_block_unchecked``, the registry's grouped copy,
``fill_swa_write_loc``, ``normal_decode_set_metadata``, ``build_trtllm_mha_page_table``) only RECORD their exact
arguments. ``flush`` checks that the recorded sequence is the known pattern (and the aliasing between one op's
outputs and the next op's inputs) and runs it as one Triton kernel that computes every output from the original
inputs (the forwarding replaces the intermediate reads); anything unexpected is replayed op by op with the original
functions, in order (exact fallback). The outputs are bitwise identical to the eager sequence (integer copies /
gathers only); SGLANG_STEP_PREP_CHECK=1 re-runs the eager ops after every fused launch and compares (host syncs).

The caller guarantees that no unrecorded GPU work between ``begin`` and ``flush`` reads a recorded op's outputs
(the decode graph runner records exactly the full-path ``load_batch`` of a padding-free replay; the DFlash worker
additionally defers its prepare_block into the draft runner's window).
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_recording = False
_replaying = False
_ops: List[Tuple[str, dict]] = []
_stats = {"fused": 0, "fallback": 0, "check_bad": 0, "check_n": 0}
_warned = set()

CH = int(os.environ.get("SGLANG_STEP_PREP_CH", "2048"))  # page-table tokens per program (the original builder: 4096)


def enabled() -> bool:
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_STEP_PREP_FUSED.get())


def is_recording() -> bool:
    return _recording and not _replaying


def record(op: str, **kw) -> None:
    _ops.append((op, kw))


def begin() -> None:
    global _recording
    if _ops:
        flush("begin")
    _recording = True


def pending() -> bool:
    return bool(_ops)


_gen = 0


def bump() -> None:
    """Every full-path load_batch and every graph replay (any runner) bumps the generation: a pre-planned replay may
    skip its input re-copy only if nothing touched the shared static buffers since its plan."""
    global _gen
    _gen += 1


def generation() -> int:
    return _gen


def prepare_gathered(*, bonus_buf, bonus_index, **kw) -> None:
    """prepare_block with bonus_tokens = bonus_buf[bonus_index] (the overlap scheduler's deferred FutureMap gather)."""
    if is_recording():
        record("prepare", bonus_tokens=bonus_buf, bonus_index=bonus_index, **kw)
        return
    from sglang.kernels.ops.speculative.dflash import _prepare_dflash_draft_block_unchecked

    _prepare_dflash_draft_block_unchecked(bonus_tokens=bonus_buf[bonus_index], **kw)


def tokens(dst: torch.Tensor, col0: torch.Tensor, rest: torch.Tensor) -> None:
    """dst[:, 0] = col0; dst[:, 1:] = rest (the verify candidates [bonus, draft tokens]) -- recorded inside a window."""
    if is_recording():
        record("tokens", dst=dst, col0=col0, rest=rest)
        return
    dst[:, 0].copy_(col0)
    dst[:, 1:].copy_(rest)


def _warn_once(key, msg, *args):
    if key not in _warned:
        _warned.add(key)
        logger.warning("step_prep: " + msg, *args)


# ----------------------------------------------------------------------------------------------------------------
# fused kernel
# ----------------------------------------------------------------------------------------------------------------
@triton.jit
def _step_prep_kernel(
    # prepare_block (HAS_PREP); HAS_BIDX: bonus = bonus_ptr[bidx_ptr[b]]
    bonus_ptr, bonus_stride, bidx_ptr, prefix_ptr, prefix_stride, prep_rpi_ptr, prep_rpi_stride, prep_r2t_ptr, prep_r2t_stride,
    blk_out_ptr, blk_stride, pos_out_ptr, pos_stride, loc_out_ptr, loc_stride, mask_token_id, r2t_width,
    # tokens op (HAS_TOK): tk_dst[b, 0] = tk_c0[b], tk_dst[b, 1:] = tk_rest[b, :]
    tk_dst, tk_dst_stride, tk_c0, tk_c0_stride, tk_rest, tk_rest_stride,
    # three token-axis copies (source kind per slot: 0 = read src, 1 = prep block ids, 2 = prep positions, 3 = prep cache
    # loc, 4 = the tokens op's output)
    t0_src, t0_dst, t1_src, t1_dst, t2_src, t2_dst,
    # two bs-axis copies: req_pool_indices, seq_lens
    rpi_src, rpi_src_stride, rpi_dst, seq_src, seq_src_stride, seq_dst,
    # SWA write loc (HAS_WLOC): dst[i] = map[loc[i]] (i < T), 0 (T <= i < total); loc kind: 0 = read wl_loc,
    # 1/2/3 = the prep's block ids / positions / cache loc, 10 + s = token copy slot s (its static value)
    wl_dst, wl_loc, wl_map, wl_map_len, wl_total,
    # FA metadata
    cs_ptr, cu_ptr, delta,
    # page table (+ SWA page table)
    r2t_ptr, r2t_stride, pt_ptr, pt_stride, spt_ptr, f2s_ptr, f2s_numel,
    B, T, NBLK,
    W: tl.constexpr, WP: tl.constexpr, BB: tl.constexpr, ZB: tl.constexpr, CHT: tl.constexpr,
    K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr, LOC_KIND: tl.constexpr,
    HAS_PREP: tl.constexpr, HAS_WLOC: tl.constexpr, HAS_SWA_PT: tl.constexpr, HAS_TOK: tl.constexpr,
    HAS_BIDX: tl.constexpr,
):
    b = tl.program_id(0)
    j = tl.program_id(1)
    # the static (post-copy) per-request values every later op reads
    rpi = tl.load(rpi_src + b * rpi_src_stride).to(rpi_dst.dtype.element_ty)
    seq = tl.load(seq_src + b * seq_src_stride).to(seq_dst.dtype.element_ty)
    cache_len = (seq + delta).to(tl.int32)

    if j == NBLK:  # the per-request copies / prepare / SWA write loc (an extra program: off the page-table chunks' path)
        tl.store(rpi_dst + b, rpi)
        tl.store(seq_dst + b, seq)
        tl.store(cs_ptr + b, cache_len)
        cols = tl.arange(0, WP)
        cm = cols < W
        ti = b * W + cols
        if HAS_PREP:
            prefix = tl.load(prefix_ptr + b * prefix_stride)
            prpi = tl.load(prep_rpi_ptr + b * prep_rpi_stride)
            if HAS_BIDX:
                bonus = tl.load(bonus_ptr + tl.load(bidx_ptr + b) * bonus_stride)
            else:
                bonus = tl.load(bonus_ptr + b * bonus_stride)
            lpos = prefix.to(tl.int64) + cols
            valid = cm & (lpos < r2t_width)
            slot_ids = tl.load(prep_r2t_ptr + prpi * prep_r2t_stride + lpos, mask=valid, other=0)
            ids = tl.full((WP,), mask_token_id, tl.int64)
            ids = tl.where(cols == 0, bonus.to(tl.int64), ids)
            tl.store(blk_out_ptr + b * blk_stride + cols, ids, mask=cm)
            tl.store(pos_out_ptr + b * pos_stride + cols, lpos, mask=cm)
            locv = slot_ids.to(tl.int64)
            tl.store(loc_out_ptr + b * loc_stride + cols, locv, mask=cm)
        if HAS_TOK:
            c0 = tl.load(tk_c0 + b * tk_c0_stride)
            rv = tl.load(tk_rest + b * tk_rest_stride + cols - 1, mask=cm & (cols > 0), other=0)
            tkv = tl.where(cols == 0, c0.to(tk_dst.dtype.element_ty), rv.to(tk_dst.dtype.element_ty))
            tl.store(tk_dst + b * tk_dst_stride + cols, tkv, mask=cm)
        # token copy slot 0
        if K0 == 1:
            v0 = ids
        elif K0 == 2:
            v0 = lpos
        elif K0 == 3:
            v0 = locv
        elif K0 == 4:
            v0 = tkv
        else:
            v0 = tl.load(t0_src + ti, mask=cm, other=0)
        v0 = v0.to(t0_dst.dtype.element_ty)
        tl.store(t0_dst + ti, v0, mask=cm)
        if K1 == 1:
            v1 = ids
        elif K1 == 2:
            v1 = lpos
        elif K1 == 3:
            v1 = locv
        elif K1 == 4:
            v1 = tkv
        else:
            v1 = tl.load(t1_src + ti, mask=cm, other=0)
        v1 = v1.to(t1_dst.dtype.element_ty)
        tl.store(t1_dst + ti, v1, mask=cm)
        if K2 == 1:
            v2 = ids
        elif K2 == 2:
            v2 = lpos
        elif K2 == 3:
            v2 = locv
        elif K2 == 4:
            v2 = tkv
        else:
            v2 = tl.load(t2_src + ti, mask=cm, other=0)
        v2 = v2.to(t2_dst.dtype.element_ty)
        tl.store(t2_dst + ti, v2, mask=cm)
        if HAS_WLOC:
            if LOC_KIND == 10:
                lv = v0
            elif LOC_KIND == 11:
                lv = v1
            elif LOC_KIND == 12:
                lv = v2
            elif LOC_KIND == 1:
                lv = ids
            elif LOC_KIND == 2:
                lv = lpos
            elif LOC_KIND == 3:
                lv = locv
            elif LOC_KIND == 4:
                lv = tkv
            else:
                lv = tl.load(wl_loc + ti, mask=cm, other=0)
            lv = lv.to(tl.int64)
            lv = tl.where(lv < 0, lv + wl_map_len, lv)  # torch negative-index semantics (the -1 sentinel row)
            mv = tl.load(wl_map + lv, mask=cm, other=0)
            tl.store(wl_dst + ti, mv.to(wl_dst.dtype.element_ty), mask=cm)
        if b == 0:
            bb = tl.arange(0, BB)
            bm = bb < B
            sq = tl.load(seq_src + bb * seq_src_stride, mask=bm, other=0).to(seq_dst.dtype.element_ty)
            cl = tl.where(bm, (sq + delta).to(tl.int32), 0)
            inc = tl.cumsum(cl, axis=0)
            tl.store(cu_ptr + bb, inc - cl, mask=bm)
            tl.store(cu_ptr + B, tl.sum(cl, axis=0))
            if HAS_WLOC:
                for z in range(T, wl_total, ZB):
                    zo = z + tl.arange(0, ZB)
                    tl.store(wl_dst + zo, tl.zeros((ZB,), wl_dst.dtype.element_ty), mask=zo < wl_total)

    # page table chunk j of request b (create_trtllm_mha_kv_indices_triton, PAGE_SIZE 1)
    if (j < NBLK) & (j * CHT < cache_len):
        page_idx = tl.arange(0, CHT) + j * CHT
        mask = page_idx < cache_len
        slot = tl.load(r2t_ptr + rpi.to(tl.int64) * r2t_stride + page_idx.to(tl.int64), mask=mask)
        out_off = b.to(tl.int64) * pt_stride + page_idx
        tl.store(pt_ptr + out_off, slot.to(tl.int32), mask=mask)
        if HAS_SWA_PT:
            swa_index = tl.minimum(tl.maximum(slot, 0), f2s_numel - 1)
            swa_slot = tl.load(f2s_ptr + swa_index.to(tl.int64), mask=mask)
            tl.store(spt_ptr + out_off, swa_slot.to(tl.int32), mask=mask)


# ----------------------------------------------------------------------------------------------------------------
# pattern matching
# ----------------------------------------------------------------------------------------------------------------
def _same(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> bool:
    """b is the same memory as a (same start, same element count / dtype, both dense)."""
    if a is None or b is None:
        return False
    return (
        a.data_ptr() == b.data_ptr()
        and a.numel() == b.numel()
        and a.dtype == b.dtype
        and a.is_contiguous()
        and b.is_contiguous()
    )


def _is_int(t: torch.Tensor) -> bool:
    return t.dtype in (torch.int32, torch.int64)


def _plan(ops):
    """Return the kernel arguments for a recognised op sequence, or (None, reason)."""
    names = [o[0] for o in ops]
    i = 0
    prep = None
    tko = None
    if names and names[0] == "prepare":
        prep = ops[0][1]
        i = 1
    elif names and names[0] == "tokens":
        tko = ops[0][1]
        i = 1
    if names[i:] not in (["copy", "wloc", "metadata", "pt"], ["copy", "metadata", "pt"]):
        return None, f"sequence {names}"
    copy = ops[i][1]
    wloc = ops[i + 1][1] if names[i + 1] == "wloc" else None
    meta = ops[-2][1]
    pt = ops[-1][1]

    if meta["skip_page_table"] is not True or meta["page_size"] != 1 or pt["page_size"] != 1:
        return None, "metadata/page-table mode"
    cs = meta["cache_seqlens_int32"]
    cu = meta["cu_seqlens_k"]
    B = int(cs.shape[0])
    if B <= 0 or B > 256 or cs.dtype != torch.int32 or not cs.is_contiguous():
        return None, "cache_seqlens"
    if cu.dtype != torch.int32 or not cu.is_contiguous() or cu.numel() < B + 1:
        return None, "cu_seqlens_k"
    if not _same(cs, pt["cache_seqlens"]):
        return None, "page table cache_seqlens is not the metadata output"
    dsts, srcs = copy["dsts"], copy["srcs"]
    if len(dsts) != 5:
        return None, f"{len(dsts)} copies"
    for d, s in zip(dsts, srcs):
        if d.dim() != 1 or s.dim() != 1 or d.shape != s.shape or not _is_int(d) or not _is_int(s):
            return None, "copy shapes / dtypes"
        if not d.is_contiguous():
            return None, "copy dst layout"
    tok = [k for k, d in enumerate(dsts) if d.numel() != B]
    bsf = [k for k, d in enumerate(dsts) if d.numel() == B]
    if len(tok) != 3 or len(bsf) != 2:
        return None, "copy field split"
    T = int(dsts[tok[0]].numel())
    if any(int(dsts[k].numel()) != T for k in tok) or T % B != 0 or T // B > 64:
        return None, "token count"
    W = T // B
    for k in tok:
        if srcs[k].stride(0) != 1:
            return None, "token src layout"
    # bs fields: req_pool_indices (read by the page table) and seq_lens (read by the metadata)
    k_rpi = [k for k in bsf if _same(dsts[k], pt["req_pool_indices"])]
    k_seq = [k for k in bsf if _same(dsts[k], meta["seq_lens"])]
    if len(k_rpi) != 1 or len(k_seq) != 1 or k_rpi[0] == k_seq[0]:
        return None, "rpi / seq_lens aliasing"
    k_rpi, k_seq = k_rpi[0], k_seq[0]
    # page table
    r2t = pt["req_to_token"]
    ptab = pt["page_table"]
    if r2t.dim() != 2 or r2t.stride(1) != 1 or r2t.dtype != torch.int32:
        return None, "req_to_token"
    if ptab.dim() != 2 or ptab.shape[0] != B or ptab.stride(1) != 1 or ptab.dtype != torch.int32:
        return None, "page table"
    spt, f2s = pt["swa_page_table"], pt["full_to_swa"]
    has_swa_pt = f2s is not None
    if has_swa_pt:
        if spt is None or spt.shape != ptab.shape or spt.stride() != ptab.stride() or spt.dtype != torch.int32:
            return None, "swa page table"
        if f2s.dim() != 1 or not f2s.is_contiguous() or not _is_int(f2s):
            return None, "full_to_swa"
    # SWA write loc: the loc it reads is resolved below (after the prepare forwarding is known)
    if wloc is not None:
        wd, wl, wm = wloc["dst"], wloc["loc"], wloc["mapping"]
        if wd.dim() != 1 or not wd.is_contiguous() or wm.dim() != 1 or not wm.is_contiguous() or not _is_int(wm):
            return None, "wloc buffers"
        if wl.dim() != 1 or not wl.is_contiguous() or not _is_int(wl) or int(wl.numel()) != T or int(wd.numel()) < T:
            return None, "wloc sizes"
    # tokens op forwarding
    kinds = [0, 0, 0]
    if tko is not None:
        td, c0, rest = tko["dst"], tko["col0"], tko["rest"]
        if td.dim() != 2 or td.shape[0] != B or td.shape[1] != W or not td.is_contiguous() or not _is_int(td):
            return None, "tokens dst"
        if c0.dim() != 1 or c0.numel() != B or not _is_int(c0):
            return None, "tokens col0"
        if rest.dim() != 2 or rest.shape[0] != B or rest.shape[1] != W - 1 or rest.stride(1) != 1 or not _is_int(rest):
            return None, "tokens rest"
        for n, k in enumerate(tok):
            if _same(srcs[k], td.view(-1)):
                kinds[n] = 4
        if kinds.count(4) != 1:
            return None, "tokens -> copy aliasing"
    # prepare_block forwarding
    if prep is not None:
        bo, po, lo = prep["block_ids_out"], prep["positions_out"], prep["cache_loc_out"]
        for t2 in (bo, po, lo):
            if t2.dim() != 2 or t2.shape[0] != B or t2.shape[1] != W or t2.stride(1) != 1 or t2.dtype != torch.int64:
                return None, "prepare outputs"
        if not (bo.is_contiguous() and po.is_contiguous() and lo.is_contiguous()):
            return None, "prepare output layout"
        for n, k in enumerate(tok):
            for kind, out in ((1, bo), (2, po), (3, lo)):
                if _same(srcs[k], out.view(-1)):
                    kinds[n] = kind
        if sorted(kinds) != [1, 2, 3]:
            return None, "prepare -> copy aliasing"
        pr2t = prep["req_to_token"]
        if pr2t.dim() != 2 or pr2t.stride(1) != 1:
            return None, "prepare req_to_token"
        for t1 in (prep["prefix_lens"], prep["req_pool_indices"]):
            if t1.dim() != 1 or t1.numel() != B:
                return None, "prepare inputs"
        bi = prep.get("bonus_index")
        if bi is None:
            if prep["bonus_tokens"].dim() != 1 or prep["bonus_tokens"].numel() != B:
                return None, "prepare bonus"
        elif bi.dim() != 1 or bi.numel() != B or not _is_int(bi) or bi.stride(0) != 1 or prep["bonus_tokens"].dim() != 1:
            return None, "prepare bonus index"
    loc_kind = 0
    if wloc is not None:
        wl = wloc["loc"]
        hit = [n for n, k in enumerate(tok) if _same(dsts[k], wl)]
        if len(hit) == 1:
            loc_kind = 10 + hit[0]
        elif tko is not None and _same(wl, tko["dst"].view(-1)):
            loc_kind = 4
        elif prep is not None:
            for kind, out in ((1, prep["block_ids_out"]), (2, prep["positions_out"]), (3, prep["cache_loc_out"])):
                if _same(wl, out.view(-1)):
                    loc_kind = kind
    # every input the kernel reads DIRECTLY must not be written by any recorded op (only the forwarded edges above
    # may carry a recorded output into a later op)
    writes = [d for d in dsts]
    writes += [meta["cache_seqlens_int32"], meta["cu_seqlens_k"][: B + 1], ptab[:B]]
    if has_swa_pt:
        writes.append(pt["swa_page_table"][:B])
    if wloc is not None:
        writes.append(wloc["dst"])
    if prep is not None:
        writes += [prep["block_ids_out"], prep["positions_out"], prep["cache_loc_out"]]
    if tko is not None:
        writes.append(tko["dst"])
    reads = [srcs[k] for n, k in enumerate(tok) if kinds[n] == 0]
    reads += [srcs[k_rpi], srcs[k_seq], r2t]
    if has_swa_pt:
        reads.append(f2s)
    if wloc is not None:
        reads.append(wloc["mapping"])
        if loc_kind == 0:
            reads.append(wloc["loc"])
    if prep is not None:
        reads += [prep["bonus_tokens"], prep["prefix_lens"], prep["req_pool_indices"], prep["req_to_token"]]
        if prep.get("bonus_index") is not None:
            reads.append(prep["bonus_index"])
    if tko is not None:
        reads += [tko["col0"], tko["rest"]]
    rr = [_span(t) for t in reads]
    for w in writes:
        a0, a1 = _span(w)
        for b0, b1 in rr:
            if a0 < b1 and b0 < a1:
                return None, "a directly read input overlaps a recorded output"
    return dict(
        B=B, T=T, W=W, tok=tok, kinds=kinds, k_rpi=k_rpi, k_seq=k_seq, loc_kind=loc_kind, prep=prep, tko=tko, copy=copy,
        wloc=wloc, meta=meta, pt=pt, has_swa_pt=has_swa_pt,
    ), None


def _span(t: torch.Tensor):
    if t.numel() == 0:
        return (0, 0)
    first = t.data_ptr()
    last = first + (sum((d - 1) * st for d, st in zip(t.shape, t.stride())) + 1) * t.element_size()
    return (first, last)


def _launch(p) -> None:
    B, T, W = p["B"], p["T"], p["W"]
    dsts, srcs = p["copy"]["dsts"], p["copy"]["srcs"]
    tok = p["tok"]
    pt = p["pt"]
    meta = p["meta"]
    prep = p["prep"]
    wloc = p["wloc"]
    ptab = pt["page_table"]
    r2t = pt["req_to_token"]
    dummy = meta["cache_seqlens_int32"]
    if prep is not None:
        pr2t = prep["req_to_token"]
        bidx = prep.get("bonus_index")
        prep_args = (
            prep["bonus_tokens"], prep["bonus_tokens"].stride(0), bidx if bidx is not None else dummy,
            prep["prefix_lens"], prep["prefix_lens"].stride(0),
            prep["req_pool_indices"], prep["req_pool_indices"].stride(0),
            pr2t, pr2t.stride(0),
            prep["block_ids_out"], prep["block_ids_out"].stride(0),
            prep["positions_out"], prep["positions_out"].stride(0),
            prep["cache_loc_out"], prep["cache_loc_out"].stride(0),
            int(prep["mask_token_id"]), int(pr2t.shape[1]),
        )
    else:
        prep_args = (dummy, 0, dummy, dummy, 0, dummy, 0, dummy, 0, dummy, 0, dummy, 0, dummy, 0, 0, 0)
    if wloc is not None:
        wl_args = (wloc["dst"], wloc["loc"], wloc["mapping"], int(wloc["mapping"].shape[0]), int(wloc["dst"].shape[0]))
    else:
        wl_args = (dummy, dummy, dummy, 0, 0)
    if p["has_swa_pt"]:
        f2s = pt["full_to_swa"]
        sw_args = (pt["swa_page_table"], f2s, int(f2s.numel()))
    else:
        sw_args = (dummy, dummy, 0)
    tko = p["tko"]
    if tko is not None:
        tk_args = (tko["dst"], tko["dst"].stride(0), tko["col0"], tko["col0"].stride(0), tko["rest"], tko["rest"].stride(0))
    else:
        tk_args = (dummy, 0, dummy, 0, dummy, 0)
    rs, ss = srcs[p["k_rpi"]], srcs[p["k_seq"]]
    nblk = max(1, triton.cdiv(int(ptab.shape[1]), CH))
    _step_prep_kernel[(B, nblk + 1)](
        *prep_args,
        *tk_args,
        srcs[tok[0]], dsts[tok[0]], srcs[tok[1]], dsts[tok[1]], srcs[tok[2]], dsts[tok[2]],
        rs, rs.stride(0), dsts[p["k_rpi"]], ss, ss.stride(0), dsts[p["k_seq"]],
        *wl_args,
        meta["cache_seqlens_int32"], meta["cu_seqlens_k"], int(meta["seq_len_delta"]),
        r2t, r2t.stride(0), ptab, ptab.stride(0), *sw_args,
        B, T, nblk,
        W=W, WP=triton.next_power_of_2(W), BB=max(16, triton.next_power_of_2(B + 1)), ZB=1024, CHT=CH,
        K0=p["kinds"][0], K1=p["kinds"][1], K2=p["kinds"][2], LOC_KIND=p["loc_kind"],
        HAS_PREP=prep is not None, HAS_WLOC=wloc is not None, HAS_SWA_PT=p["has_swa_pt"], HAS_TOK=tko is not None,
        HAS_BIDX=prep is not None and prep.get("bonus_index") is not None,
        num_warps=4,
    )


# ----------------------------------------------------------------------------------------------------------------
# eager replay (fallback / check)
# ----------------------------------------------------------------------------------------------------------------
def _replay(ops) -> None:
    global _replaying
    _replaying = True
    try:
        for name, kw in ops:
            if name == "prepare":
                from sglang.kernels.ops.speculative.dflash import _prepare_dflash_draft_block_unchecked

                kw = dict(kw)
                bi = kw.pop("bonus_index", None)
                if bi is not None:
                    kw["bonus_tokens"] = kw["bonus_tokens"][bi]
                _prepare_dflash_draft_block_unchecked(**kw)
            elif name == "copy":
                kw["fn"](kw["dsts"], kw["srcs"])
            elif name == "tokens":
                kw["dst"][:, 0].copy_(kw["col0"])
                kw["dst"][:, 1:].copy_(kw["rest"])
            elif name == "wloc":
                from sglang.kernels.ops.attention.swa_write_loc import fill_swa_write_loc

                fill_swa_write_loc(kw["dst"], kw["loc"], kw["mapping"])
            elif name == "metadata":
                from sglang.kernels.ops.attention.metadata import normal_decode_set_metadata

                normal_decode_set_metadata(**kw)
            elif name == "pt":
                from sglang.kernels.ops.kvcache.trtllm_mha_page_table import build_trtllm_mha_page_table

                build_trtllm_mha_page_table(**kw)
            else:
                raise RuntimeError(f"step_prep: unknown op {name}")
    finally:
        _replaying = False


def _outputs(p):
    """(name, tensor view) of every region the fused kernel writes (for the check)."""
    B, T = p["B"], p["T"]
    out = []
    if p["prep"] is not None:
        for n in ("block_ids_out", "positions_out", "cache_loc_out"):
            out.append((n, p["prep"][n]))
    if p["tko"] is not None:
        out.append(("tokens", p["tko"]["dst"]))
    for k, d in enumerate(p["copy"]["dsts"]):
        out.append((f"copy{k}", d))
    if p["wloc"] is not None:
        out.append(("wloc", p["wloc"]["dst"]))
    out.append(("cache_seqlens", p["meta"]["cache_seqlens_int32"]))
    out.append(("cu_seqlens_k", p["meta"]["cu_seqlens_k"][: B + 1]))
    pt = p["pt"]
    cs = p["meta"]["cache_seqlens_int32"]
    out.append(("page_table", pt["page_table"][:B]))
    if p["has_swa_pt"]:
        out.append(("swa_page_table", pt["swa_page_table"][:B]))
    return out


def flush(where: str = "") -> None:
    """Run the recorded ops (fused when recognised) and close the recording window."""
    global _recording
    _recording = False
    if not _ops:
        return
    ops = list(_ops)
    _ops.clear()
    plan, reason = _plan(ops)
    if plan is None:
        _stats["fallback"] += 1
        _warn_once(("fb", reason), "unrecognised prep sequence at %s (%s): eager replay", where, reason)
        _replay(ops)
        return
    _launch(plan)
    _stats["fused"] += 1
    if _check():
        _run_check(plan, ops, where)


def _check() -> bool:
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_STEP_PREP_CHECK.get())


def _run_check(plan, ops, where) -> None:
    torch.cuda.synchronize()
    outs = _outputs(plan)
    # page-table rows: only the live prefix is defined (the tails keep stale values in both paths)
    fused = [(n, t.clone()) for n, t in outs]
    _replay(ops)
    torch.cuda.synchronize()
    bad = []
    for (n, a), (_, t) in zip(fused, outs):
        if not torch.equal(a, t):
            bad.append((n, int((a != t).sum().item())))
    _stats["check_n"] += 1
    if bad:
        _stats["check_bad"] += 1
    n = _stats["check_n"]
    if bad or n <= 3 or n % 500 == 0:
        try:
            from sglang.srt.runtime_context import get_parallel

            r = get_parallel().tp_rank
        except Exception:
            r = -1
        logger.info(
            "STEP_PREP_CHECK rank%d #%d (%s, B=%d T=%d prep=%d tokens=%d wloc=%d swa_pt=%d): %s (bad launches %d, fused %d, fallback %d)",
            r, n, where, plan["B"], plan["T"], plan["prep"] is not None, plan["tko"] is not None, plan["wloc"] is not None,
            plan["has_swa_pt"],
            "MISMATCH " + str(bad) if bad else "bitwise OK", _stats["check_bad"], _stats["fused"], _stats["fallback"],
        )


def stats():
    return dict(_stats)
