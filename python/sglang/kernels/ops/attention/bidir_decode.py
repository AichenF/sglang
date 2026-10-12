"""Bidirectional (non-causal) paged attention for the DFlash draft propose block.

Semantics match ``flash_attn_with_kvcache(..., causal=False, window_size=(L, R), sinks=...)``
on the same page table and ``cache_seqlens``: query row ``t`` of request ``b`` sits at position
``cache_seqlens[b] - W + t`` (FA's bottom-right alignment) and attends keys ``[0, cache_seqlens[b])``
through ``page_table[b]``, restricted to ``[pos - L, pos + R]`` when a window side is >= 0. A per-head
sink is an extra (unscaled) logit in the softmax denominator that adds no value (as in swa_decode.py).

Flash-decoding structure: the key range is split across programs so the grid
covers the machine, and every query head of a request shares one gathered KV
tile -- the 16:1 GQA sharing FA3 cannot exploit at this size.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl

_NEG = tl.constexpr(-1.0e30)

_HEADS_PER_PROGRAM = 8
_BLOCK_N = 64
_NUM_WARPS = 4
_NUM_STAGES = 3
# NSPLIT is chosen to put the grid near this many programs.
_TARGET_GRID = 128
_MAX_NSPLIT = 64

# The draft block is 6-8 tokens; the cap rejects anything that is not one.
_MAX_BLOCK_TOKENS = 32
_WINDOWED_OK = os.environ.get("SGLANG_BIDIR_DECODE_WINDOWED", "1") == "1"
# PDL launches of the decode + combine kernels (combine launch hidden under the decode kernel).
_PDL = os.environ.get("SGLANG_BIDIR_DECODE_PDL", "1") == "1"


@triton.jit
def _bidir_decode_kernel(
    Q, K, V, PT, SEQLENS, SINKS, OUT, PART_O, PART_LSE,
    stride_qm, stride_qh, stride_ks, stride_vs, stride_ptb,
    stride_om, stride_oh,
    stride_pob, stride_poh, stride_pos,
    sm_scale, WL, WR,         # window left / right (-1 = unbounded)
    HAS_WIN: tl.constexpr, HAS_SINK: tl.constexpr,
    W: tl.constexpr,          # query rows per request (draft block size)
    HP: tl.constexpr,         # query heads per program
    D: tl.constexpr, DV: tl.constexpr,
    BLOCK_R: tl.constexpr,    # next pow2 >= W * HP, min 16 for tl.dot
    BLOCK_N: tl.constexpr,    # keys per iteration
    NSPLIT: tl.constexpr,     # kv splits; 1 = single-kernel path
    HQN: tl.constexpr,        # total query heads (for the PART_LSE index)
    USE_PDL: tl.constexpr = False,
):
    if USE_PDL:
        # PDL: wait for the producer of q / the new KV rows, then let the combine kernel launch right away
        # (it waits for this grid's completion itself).
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    b = tl.program_id(0)
    hg = tl.program_id(1)
    sp = tl.program_id(2)

    seqlen = tl.load(SEQLENS + b)

    rows = tl.arange(0, BLOCK_R)
    rvalid = rows < W * HP
    t = rows // HP                      # draft-block token index
    h = hg * HP + rows % HP             # query head
    m_idx = b * W + t                   # row in the flattened [M, HQ, D] q

    d = tl.arange(0, D)
    q = tl.load(
        Q + m_idx[:, None] * stride_qm + h[:, None] * stride_qh + d[None, :],
        mask=rvalid[:, None], other=0.0,
    )

    nblk = tl.cdiv(seqlen, BLOCK_N)
    qpos = seqlen - W + t               # FA bottom-right alignment of the block rows
    blo = 0
    if HAS_WIN:
        if WL >= 0:
            blo = tl.maximum(seqlen - W - WL, 0) // BLOCK_N

    m_i = tl.full((BLOCK_R,), _NEG, tl.float32)
    l_i = tl.zeros((BLOCK_R,), tl.float32)
    acc = tl.zeros((BLOCK_R, DV), tl.float32)
    dv = tl.arange(0, DV)

    # Strided, not contiguous: a contiguous split leaves late programs empty
    # whenever NSPLIT does not divide nblk.
    for bi in tl.range(blo + sp, nblk, NSPLIT):
        start = bi * BLOCK_N
        j = start + tl.arange(0, BLOCK_N)
        kvalid = j < seqlen
        slot = tl.load(PT + b * stride_ptb + j, mask=kvalid, other=0)
        k = tl.load(K + slot[:, None] * stride_ks + d[None, :],
                    mask=kvalid[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * sm_scale

        keep = rvalid[:, None] & kvalid[None, :]
        if HAS_WIN:
            if WL >= 0:
                keep = keep & (j[None, :] >= (qpos - WL)[:, None])
            if WR >= 0:
                keep = keep & (j[None, :] <= (qpos + WR)[:, None])
        # Masked entries must sit strictly below the -1e30 running max, so a
        # fully masked tile contributes exactly zero rather than ones.
        s = tl.where(keep, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + slot[:, None] * stride_vs + dv[None, :],
                    mask=kvalid[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    if NSPLIT == 1:
        # A padded CUDA-graph slot has cache_seqlens == 0, so the loop never
        # runs and l_i stays 0; emit zeros rather than 0/0 = NaN, which would
        # otherwise flow into o_proj and the all-reduce.
        if HAS_SINK:
            sink = tl.load(SINKS + h, mask=rvalid, other=float("-inf")).to(tl.float32)
            m2 = tl.maximum(m_i, sink)
            sc = tl.exp(m_i - m2)
            acc = acc * sc[:, None]
            l_i = l_i * sc + tl.where(l_i > 0, tl.exp(sink - m2), 0.0)
        o = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
        tl.store(
            OUT + m_idx[:, None] * stride_om + h[:, None] * stride_oh + dv[None, :],
            o.to(OUT.dtype.element_ty), mask=rvalid[:, None],
        )
    else:
        # Store the *normalised* partial output plus its log-sum-exp; the
        # combine is then a plain lse-weighted average and needs no separate
        # running max / denominator.
        #   o_s = acc_s / l_s ,  lse_s = m_s + log l_s
        #   o   = sum_s w_s o_s / sum_s w_s ,  w_s = exp(lse_s - max_s lse)
        po = PART_O + m_idx[:, None] * stride_pob + h[:, None] * stride_poh \
            + sp * stride_pos + dv[None, :]
        # A split that received no blocks has l_i == 0; give it lse = -inf so
        # it drops out of the combine instead of producing a NaN.
        l_safe = tl.where(l_i > 0, l_i, 1.0)
        tl.store(po, acc / l_safe[:, None], mask=rvalid[:, None])
        lse = tl.where(l_i > 0, m_i + tl.log(l_safe), float("-inf"))
        tl.store(PART_LSE + m_idx * (HQN * NSPLIT) + h * NSPLIT + sp, lse,
                 mask=rvalid)


@triton.jit
def _bidir_combine_kernel(
    PART_O, PART_LSE, SINKS, OUT,
    stride_pob, stride_poh, stride_pos, stride_om, stride_oh,
    NSPLIT: tl.constexpr, DV: tl.constexpr, HQN: tl.constexpr, HAS_SINK: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    m = tl.program_id(0)
    h = tl.program_id(1)
    sp = tl.arange(0, NSPLIT)
    lse = tl.load(PART_LSE + m * (HQN * NSPLIT) + h * NSPLIT + sp)
    mx = tl.max(lse)
    # All splits empty (cache_seqlens == 0 on a padded slot): mx is -inf and
    # exp(-inf - -inf) is NaN. Pin the frame and let the zero denominator
    # guard below produce a zero row instead.
    empty = mx == float("-inf")
    if HAS_SINK:
        sink = tl.load(SINKS + h).to(tl.float32)
        mx = tl.maximum(mx, sink)
    mx = tl.where(mx > float("-inf"), mx, 0.0)
    w = tl.exp(lse - mx)
    dv = tl.arange(0, DV)
    po = tl.load(PART_O + m * stride_pob + h * stride_poh
                 + sp[:, None] * stride_pos + dv[None, :])
    den = tl.sum(w)
    if HAS_SINK:
        den += tl.where(empty, 0.0, tl.exp(sink - mx))
    o = tl.sum(po * w[:, None], 0) / tl.where(den > 0, den, 1.0)
    tl.store(OUT + m * stride_om + h * stride_oh + dv,
             o.to(OUT.dtype.element_ty))


def _pick_nsplit(bs: int, head_groups: int) -> int:
    """Split the key range just enough to cover the machine, rounded down to a
    power of two (the combine kernel indexes splits with tl.arange)."""
    want = max(1, _TARGET_GRID // max(1, bs * head_groups))
    ns = 1
    while ns * 2 <= min(want, _MAX_NSPLIT):
        ns *= 2
    return ns


def can_use_bidir_decode(
    *,
    q: torch.Tensor,
    forward_mode,
    num_kv_heads: int,
    num_q_heads: int,
    head_dim: int,
    v_head_dim: int,
    window_size: tuple,
    causal: bool,
    page_size: int,
    softcap: float,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    kwargs: dict,
) -> bool:
    """Shape gate for the DFlash draft propose block.

    Deliberately narrow: it must select the five draft layers and nothing else.
    The target's own non-SWA layers are causal and carry asymmetric head dims
    (192 / 128), so ``not causal`` plus ``head_dim == v_head_dim`` already
    separates them; the rest is the contract the kernel implements.
    """
    # The draft propose runs as TARGET_VERIFY (see DFlashWorkerV2), with a
    # uniform block of query rows per request.
    if not forward_mode.is_target_verify():
        return False
    if causal:
        return False
    if max_seqlen_q is None or not 1 <= max_seqlen_q <= _MAX_BLOCK_TOKENS:
        return False
    if softcap not in (0, 0.0):
        return False
    if num_kv_heads != 1 or num_q_heads % _HEADS_PER_PROGRAM != 0:
        return False
    if page_size != 1:
        return False
    if head_dim != v_head_dim or head_dim & (head_dim - 1):
        return False
    if q.dtype != torch.bfloat16:
        return False
    # Per-head sinks are supported (MiMo-V2.6 draft); FP8 KV descaling is a
    # different contract.
    if any(k in kwargs for k in ("q_descale", "k_descale", "v_descale",
                                 "score_mod", "rel_bias")):
        return False
    if window_size[0] < -1 or window_size[1] < -1:
        return False
    # SGLANG_BIDIR_DECODE_WINDOWED=0 keeps windowed / sink calls (MiMo-V2.6 draft) on FA3 (A/B knob).
    if (window_size[0] >= 0 or window_size[1] >= 0 or "sinks" in kwargs) and not _WINDOWED_OK:
        return False
    # Uniform rows per request (the draft-block graph shape); ragged falls back.
    return q.shape[0] == cache_seqlens.shape[0] * max_seqlen_q


def bidir_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    block_tokens: int,
    sm_scale: float,
    window_size: tuple = (-1, -1),
    sinks: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    heads_per_program: int = _HEADS_PER_PROGRAM,
    block_n: Optional[int] = None,
    nsplit: Optional[int] = None,
    num_warps: int = _NUM_WARPS,
    num_stages: int = _NUM_STAGES,
) -> torch.Tensor:
    """q: [M, HQ, D]; k_cache/v_cache: [slots, ..., D]; page_table: [bs, pages]
    (page_size 1); cache_seqlens: [bs] int32.  Non-causal; window_size = (left, right) as in FA
    (-1 = unbounded side); sinks: optional fp32/bf16 [HQ] per-head sink logits."""
    # The kernel loads the head dim as one contiguous vector; row and head
    # strides are explicit, so a row-strided q (the QKV slice) is fine.
    if q.stride(-1) != 1:
        q = q.contiguous()
    m, hq, d = q.shape
    bs = cache_seqlens.shape[0]
    dv = v_cache.shape[-1]
    if out is None:
        out = torch.empty((m, hq, dv), device=q.device, dtype=q.dtype)
    out_view = out.view(m, hq, dv)

    k2 = k_cache.view(k_cache.shape[0], -1)
    v2 = v_cache.view(v_cache.shape[0], -1)
    head_groups = hq // heads_per_program
    if nsplit is None:
        nsplit = _pick_nsplit(bs, head_groups)

    if nsplit == 1:
        part_o = part_lse = q            # unused dummy pointers
        spo = sph = sps = 0
    else:
        # Allocated per call on purpose. Under CUDA-graph capture the caching
        # allocator serves this from the capturing graph private pool and bakes
        # the pointer into the replay; a module-level cache would hand a buffer
        # owned by one graph pool to another graph and alias freed memory.
        part_o = torch.empty((m, hq, nsplit, dv), device=q.device, dtype=torch.float32)
        part_lse = torch.empty((m, hq, nsplit), device=q.device, dtype=torch.float32)
        spo, sph, sps = part_o.stride(0), part_o.stride(1), part_o.stride(2)

    wl, wr = int(window_size[0]), int(window_size[1])
    has_win = wl >= 0 or wr >= 0
    if block_n is None:
        # windowed draft (V2.6, ~1k keys per request): 32-key tiles measured fastest on H200
        # (7.6 vs 8.9 us at B4 x 8 rows x 16 heads); full-context keeps the original 64
        block_n = 32 if has_win else _BLOCK_N
    sink_arg = sinks if sinks is not None else q
    _bidir_decode_kernel[(bs, head_groups, nsplit)](
        q, k2, v2, page_table, cache_seqlens, sink_arg, out_view, part_o, part_lse,
        q.stride(0), q.stride(1), k2.stride(0), v2.stride(0), page_table.stride(0),
        out_view.stride(0), out_view.stride(1),
        spo, sph, sps,
        sm_scale, wl, wr, HAS_WIN=has_win, HAS_SINK=sinks is not None,
        W=block_tokens, HP=heads_per_program,
        D=d, DV=dv,
        BLOCK_R=max(16, triton.next_power_of_2(block_tokens * heads_per_program)),
        BLOCK_N=block_n, NSPLIT=nsplit, HQN=hq, USE_PDL=_PDL,
        num_warps=num_warps, num_stages=num_stages,
        **({"launch_pdl": True} if _PDL else {}),
    )
    if nsplit > 1:
        _bidir_combine_kernel[(m, hq)](
            part_o, part_lse, sink_arg, out_view,
            part_o.stride(0), part_o.stride(1), part_o.stride(2),
            out_view.stride(0), out_view.stride(1),
            NSPLIT=nsplit, DV=dv, HQN=hq, HAS_SINK=sinks is not None, USE_PDL=_PDL, num_warps=4,
            **({"launch_pdl": True} if _PDL else {}),
        )
    return out
