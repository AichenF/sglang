"""Experimental M32 BF16 NeoX64 RoPE + asymmetric192/128 NHD KV store."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mul_bf16(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.bf16 $0, $1, $2;",
        constraints="=h,h,h",
        args=[a, b],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _neg_bf16(x):
    return (
        (x.to(tl.uint16, bitcast=True) ^ 0x8000)
        .to(tl.uint16)
        .to(tl.bfloat16, bitcast=True)
    )


@triton.jit
def _fma_bf16(a, b, c):
    return tl.inline_asm_elementwise(
        "fma.rn.bf16 $0, $1, $2, $3;",
        constraints="=h,h,h,h",
        args=[a.to(tl.bfloat16), b.to(tl.bfloat16), c.to(tl.bfloat16)],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rope_kv(
    QKV,
    CS,
    POS,
    LOC,
    KC,
    VC,
    KS: tl.constexpr,
    VS: tl.constexpr,
    HEADS: tl.constexpr,
    ROWS: tl.constexpr,
):
    token = tl.program_id(0)
    group = tl.program_id(1)
    pos = tl.load(POS + token)
    r = tl.arange(0, HEADS * 32)
    head = group * HEADS + r // 32
    dim = r % 32
    c = tl.load(CS + pos * 64 + dim)
    s = tl.load(CS + pos * 64 + 32 + dim)
    offset = token * 3392 + head * 192 + dim
    x = tl.load(QKV + offset)
    y = tl.load(QKV + offset + 32)
    # The pinned CUDA13 fallback contracts the left product into BF16 FMA;
    # only the right product rounds before the final operation.
    lo = _fma_bf16(x, c, _neg_bf16(_mul_bf16(y, s)))
    hi = _fma_bf16(y, c, _mul_bf16(x, s))
    tl.store(QKV + offset, lo)
    tl.store(QKV + offset + 32, hi)
    if group == 0:
        dimk = tl.arange(0, 32)
        xk = tl.load(QKV + token * 3392 + 3072 + dimk)
        yk = tl.load(QKV + token * 3392 + 3072 + 32 + dimk)
        ck = tl.load(CS + pos * 64 + dimk)
        sk = tl.load(CS + pos * 64 + 32 + dimk)
        kl = _fma_bf16(xk, ck, _neg_bf16(_mul_bf16(yk, sk)))
        kh = _fma_bf16(yk, ck, _mul_bf16(xk, sk))
        slot = tl.load(LOC + token)
        tl.device_assert((slot >= 0) & (slot < ROWS), "KV cache slot out of bounds")
        # A single lane owns both members of each pair, so no other warp
        # reads either element after an in-place rotation has overwritten it.
        tl.store(QKV + token * 3392 + 3072 + dimk, kl)
        tl.store(QKV + token * 3392 + 3104 + dimk, kh)
        tl.store(KC + slot * KS + dimk, kl, slot != 0)
        tl.store(KC + slot * KS + 32 + dimk, kh, slot != 0)
        d = tl.arange(0, 128)
        old = tl.load(QKV + token * 3392 + 3136 + d)
        tl.store(KC + slot * KS + 64 + d, old, slot != 0)
        value = tl.load(QKV + token * 3392 + 3264 + d)
        tl.store(VC + slot * VS + d, value, slot != 0)


def run(qkv, cos_sin, positions, locations, k_cache, v_cache, heads_per_cta=16):
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
    assert k_cache.dtype == v_cache.dtype == torch.bfloat16 and heads_per_cta in (
        4,
        8,
        16,
    )
    assert (
        all(
            t.device == qkv.device
            for t in (cos_sin, positions, locations, k_cache, v_cache)
        )
        and qkv.is_cuda
    )
    _rope_kv[(32, 16 // heads_per_cta)](
        qkv,
        cos_sin,
        positions,
        locations,
        k_cache,
        v_cache,
        k_cache.stride(0),
        v_cache.stride(0),
        heads_per_cta,
        k_cache.shape[0],
        num_warps=4,
        enable_fp_fusion=False,
        debug=True,
    )
