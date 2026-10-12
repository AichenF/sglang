"""goal11-glue: pack up to 5 deferred DFlash aux captures ([M, H] bf16 row-major sources) into the packed
[M, K * H] aux buffer slots in one launch (replaces one copy kernel per capture)."""

from typing import List, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _aux_pack_kernel(
    dst_ptr,
    dst_stride,
    s0, s1, s2, s3, s4,
    st0, st1, st2, st3, st4,
    o0, o1, o2, o3, o4,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0)
    k = tl.program_id(1)
    c = tl.program_id(2)
    offs = c * BLOCK + tl.arange(0, BLOCK)
    mask = offs < H
    if k == 0:
        v = tl.load(s0 + m * st0 + offs, mask=mask)
        start = o0
    elif k == 1:
        v = tl.load(s1 + m * st1 + offs, mask=mask)
        start = o1
    elif k == 2:
        v = tl.load(s2 + m * st2 + offs, mask=mask)
        start = o2
    elif k == 3:
        v = tl.load(s3 + m * st3 + offs, mask=mask)
        start = o3
    else:
        v = tl.load(s4 + m * st4 + offs, mask=mask)
        start = o4
    tl.store(dst_ptr + m * dst_stride + start + offs, v, mask=mask)


def aux_pack(buf: torch.Tensor, pending: List[Tuple[int, torch.Tensor]]) -> None:
    """buf [M, K * H] (row stride free, unit column stride); pending: (column start, source [>= M, H] row-major)."""
    n = len(pending)
    assert 1 <= n <= 5 and buf.dim() == 2 and buf.stride(1) == 1
    M = int(buf.shape[0])
    H = int(pending[0][1].shape[-1])
    srcs = [p[1] for p in pending] + [pending[0][1]] * (5 - n)
    starts = [int(p[0]) for p in pending] + [0] * (5 - n)
    for s in srcs:
        assert s.dim() == 2 and s.stride(1) == 1 and s.shape[0] >= M and int(s.shape[1]) == H and s.dtype == buf.dtype
    BLOCK = 1024
    grid = (M, n, triton.cdiv(H, BLOCK))
    _aux_pack_kernel[grid](
        buf,
        buf.stride(0),
        *srcs,
        *[int(s.stride(0)) for s in srcs],
        *starts,
        H=H,
        BLOCK=BLOCK,
        num_warps=4,
    )
