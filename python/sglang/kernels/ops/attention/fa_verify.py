"""Split-KV verify-decode attention for the MiMo-V2.x full-attention layers (TP8 rank shape).

Semantics match ``flash_attn_with_kvcache(q, k_cache, v_cache, page_table=..., cache_seqlens=...,
cu_seqlens_q=..., max_seqlen_q=8, causal=True)`` for a uniform verify block of 8 tokens per request:
query t of request b sits at position ``cache_seqlens[b] - 8 + t`` and attends keys ``[0, pos]``.
One KV head, 16 query heads, head dims 192 (QK) / 128 (V), page_size 1, bf16.

See csrc/attention/fa_verify_decode.cuh: one CTA owns all 128 query rows of a request (KV read once),
the key range is split over ~num_SMs / bs CTAs, and a PDL-launched combine merges the partials.
"""
from __future__ import annotations

import os

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

_W = 8
_HQ = 16
_D = 192
_DV = 128
_STAGES = int(os.environ.get("SGLANG_FA_VERIFY_STAGES", "4"))
_PDL = os.environ.get("SGLANG_FA_VERIFY_PDL", "1") == "1"
_NSPLIT_ENV = int(os.environ.get("SGLANG_FA_VERIFY_NSPLIT", "0"))
_INTERLEAVE = int(os.environ.get("SGLANG_FA_VERIFY_INTERLEAVE", "1"))


@cache_once
def _module():
    return load_jit(
        "fa_verify_decode",
        cuda_files=["attention/fa_verify_decode.cuh"],
        cuda_wrappers=[("fa_verify_decode", "fa_verify_decode")],
        extra_cuda_cflags=["--use_fast_math"],
    )


@cache_once
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


_NO_ROPE = {}


def _no_rope(device):
    key = str(device)
    t = _NO_ROPE.get(key)
    if t is None:
        t = torch.empty((0,), dtype=torch.int64, device=device)
        _NO_ROPE[key] = t
    return t


def pick_nsplit(bs: int, device: torch.device) -> int:
    if _NSPLIT_ENV > 0:
        return _NSPLIT_ENV
    return max(1, _num_sms(device.index if device.index is not None else torch.cuda.current_device()) // max(1, bs))


def can_use_fa_verify(*, q, forward_mode, num_kv_heads, num_q_heads, head_dim, v_head_dim, window_size, causal,
                      page_size, softcap, cache_seqlens, max_seqlen_q, kwargs) -> bool:
    if os.environ.get("SGLANG_FA_VERIFY", "1") != "1":
        return False
    if not forward_mode.is_target_verify():
        return False
    if max_seqlen_q != _W or not causal:
        return False
    if num_kv_heads != 1 or num_q_heads != _HQ or head_dim != _D or v_head_dim != _DV:
        return False
    if page_size != 1 or softcap not in (0, 0.0):
        return False
    if window_size[0] >= 0 or window_size[1] >= 0:
        return False
    if q.dtype != torch.bfloat16 or q.stride(-1) != 1:
        return False
    # Plain bf16 attention only: no sinks / descales / score mods / block-scale factors.
    if kwargs:
        return False
    return q.shape[0] == cache_seqlens.shape[0] * _W


def fa_verify_attention(q, k_cache, v_cache, page_table, cache_seqlens, *, sm_scale: float, out=None,
                        nsplit: int | None = None, stages: int | None = None, use_pdl: bool | None = None,
                        interleave: int | None = None, rope=None):
    """rope = (cos_sin [max_pos, 64] bf16, positions [M] int32/int64): RoPE + KV-cache store of the new tokens are
    fused (q must then be the UN-rotated q slice of the contiguous fused qkv rows [M, 16*192 + 192 + 128], and the
    cache slots of the new tokens are the page-table entries of positions cache_seqlens - 8 .. cache_seqlens - 1)."""
    """q: [M, 16, 192] (row/head strided view ok); k_cache [slots, ..., 192]; v_cache [slots, ..., 128];
    page_table [bs, pages] int32 (page_size 1); cache_seqlens [bs] int32 (includes the 8 verify tokens)."""
    m, hq, d = q.shape
    if page_table.stride(1) != 1:
        page_table = page_table.contiguous()
    bs = cache_seqlens.shape[0]
    if out is None:
        out = torch.empty((m, hq, _DV), device=q.device, dtype=q.dtype)
    out_view = out.view(m, hq, _DV)
    k2 = k_cache.view(k_cache.shape[0], -1)
    v2 = v_cache.view(v_cache.shape[0], -1)
    ns = min(nsplit if nsplit is not None else pick_nsplit(bs, q.device), 128)
    # Per call on purpose (CUDA-graph private pool, see bidir_decode.py).
    part_o = torch.empty((bs * _W * _HQ * ns * _DV,), device=q.device, dtype=torch.float32)
    part_lse = torch.empty((bs * _W * _HQ * ns,), device=q.device, dtype=torch.float32)
    _module().fa_verify_decode(
        q, k2, v2, page_table, cache_seqlens, part_o, part_lse, out_view,
        q.stride(0), q.stride(1), page_table.stride(0), out_view.stride(0), out_view.stride(1),
        ns, float(sm_scale), stages if stages is not None else _STAGES,
        int(_PDL if use_pdl is None else use_pdl),
        int(_INTERLEAVE if interleave is None else interleave),
        rope[0] if rope is not None else _no_rope(q.device),
        rope[1] if rope is not None else _no_rope(q.device),
    )
    return out
