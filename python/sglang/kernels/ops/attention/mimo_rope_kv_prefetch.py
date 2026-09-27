"""Exact fused RoPE/KV with read-only O-weight L2 prefetch on the legal M32 path."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.mimo_rope_kv import _fma_bf16, _mul_bf16, _neg_bf16


@triton.jit
def _rope_prefetch(
    QKV,
    CS,
    POS,
    LOC,
    KC,
    VC,
    WEIGHT,
    KS: tl.constexpr,
    VS: tl.constexpr,
    ROWS: tl.constexpr,
    CHUNK_BYTES: tl.constexpr,
):
    token = tl.program_id(0)
    # A scalar Triton value can be replicated across lanes. Explicit physical
    # thread predication guarantees exactly one bulk request per CTA.
    tl.inline_asm_elementwise(
        "{ .reg .u32 tid; .reg .pred leader; "
        "mov.u32 tid, %tid.x; setp.eq.u32 leader, tid, 0; "
        "@leader cp.async.bulk.prefetch.L2.global [$1], $2; mov.u32 $0, 0; }",
        constraints="=r,l,r",
        args=[WEIGHT + token * (CHUNK_BYTES // 2), CHUNK_BYTES],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    pos = tl.load(POS + token)
    r = tl.arange(0, 512)
    head, dim = r // 32, r % 32
    c = tl.load(CS + pos * 64 + dim)
    s = tl.load(CS + pos * 64 + 32 + dim)
    offset = token * 3392 + head * 192 + dim
    x, y = tl.load(QKV + offset), tl.load(QKV + offset + 32)
    lo = _fma_bf16(x, c, _neg_bf16(_mul_bf16(y, s)))
    hi = _fma_bf16(y, c, _mul_bf16(x, s))
    tl.store(QKV + offset, lo)
    tl.store(QKV + offset + 32, hi)
    dimk = tl.arange(0, 32)
    xk = tl.load(QKV + token * 3392 + 3072 + dimk)
    yk = tl.load(QKV + token * 3392 + 3104 + dimk)
    ck = tl.load(CS + pos * 64 + dimk)
    sk = tl.load(CS + pos * 64 + 32 + dimk)
    kl = _fma_bf16(xk, ck, _neg_bf16(_mul_bf16(yk, sk)))
    kh = _fma_bf16(yk, ck, _mul_bf16(xk, sk))
    slot = tl.load(LOC + token)
    tl.device_assert((slot >= 0) & (slot < ROWS), "KV cache slot out of bounds")
    tl.store(QKV + token * 3392 + 3072 + dimk, kl)
    tl.store(QKV + token * 3392 + 3104 + dimk, kh)
    tl.store(KC + slot * KS + dimk, kl, slot != 0)
    tl.store(KC + slot * KS + 32 + dimk, kh, slot != 0)
    d = tl.arange(0, 128)
    old = tl.load(QKV + token * 3392 + 3136 + d)
    tl.store(KC + slot * KS + 64 + d, old, slot != 0)
    value = tl.load(QKV + token * 3392 + 3264 + d)
    tl.store(VC + slot * VS + d, value, slot != 0)


def run(qkv, cos_sin, positions, locations, k_cache, v_cache, weight):
    assert (
        tuple(qkv.shape) == (32, 3392)
        and qkv.dtype == torch.bfloat16
        and qkv.is_contiguous()
    )
    assert (
        cos_sin.ndim == 2
        and cos_sin.shape[1] == 64
        and cos_sin.dtype == torch.bfloat16
        and cos_sin.is_contiguous()
    )
    assert (
        positions.shape == locations.shape == (32,)
        and positions.is_contiguous()
        and locations.is_contiguous()
    )
    assert positions.dtype in (torch.int32, torch.int64) and locations.dtype in (
        torch.int32,
        torch.int64,
    )
    assert k_cache.shape[1:] == (1, 192) and v_cache.shape[1:] == (1, 128)
    assert (
        k_cache.shape[0] == v_cache.shape[0]
        and k_cache.stride(-1) == v_cache.stride(-1) == 1
    )
    assert k_cache.dtype == v_cache.dtype == torch.bfloat16
    assert (
        tuple(weight.shape) == (6144, 2048)
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
    )
    assert weight.data_ptr() % 16 == 0
    assert (
        all(
            t.device == qkv.device
            for t in (cos_sin, positions, locations, k_cache, v_cache, weight)
        )
        and qkv.is_cuda
    )
    chunk = weight.numel() * weight.element_size() // 32
    assert (
        chunk == 786432
        and 32 * chunk == weight.numel() * weight.element_size()
        and chunk % 16 == 0
    )
    _rope_prefetch[(32,)](
        qkv,
        cos_sin,
        positions,
        locations,
        k_cache,
        v_cache,
        weight,
        k_cache.stride(0),
        v_cache.stride(0),
        k_cache.shape[0],
        chunk,
        num_warps=4,
        enable_fp_fusion=False,
        debug=True,
    )
