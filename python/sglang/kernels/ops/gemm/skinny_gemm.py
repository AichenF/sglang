"""Skinny (M <= 32) FP8 GEMM for Hopper decode (csrc/gemm/skinny_gemm_sm90.cuh).

Swap-AB cluster split-K kernel: one cluster of S CTAs per 128-row block of the weight, each CTA streams a K-slice
(16 KB TMA units, two consumer warpgroups), rank 0 reduces the partials over DSMEM in rank order (deterministic).
With PDL the weight TMA loads of the first stages are issued before griddepcontrol.wait.

  skinny_fp8_block_gemm(x, x_scale, w, w_scale)  DeepGEMM block scales (w 128x128, x per token per 128-K group,
                                                  x_scale column-major [M, K/128] with a column stride <= 32)
  skinny_fp8_pt_gemm(x, x_scale, w, w_scale)      per-token activation scale [M] x per-tensor ([1]) or per-row ([N])
                                                  weight scale
"""

import functools
import os
from typing import Optional

import torch

MAX_M = 32

# test / tuning knobs
STAMPS = None   # int64 cuda tensor [>= grid*16]: per-CTA globaltimer stamps (separate stamp build)
USE_PDL = None  # None = arch default
DEPTH = int(os.environ.get("SGLANG_SKINNY_DEPTH", "4"))  # weight units in flight per CTA (<= kStages = 6)
CLUSTER = 0     # 0 = automatic
WARM = 0        # unused
PRELOAD = int(os.environ.get("SGLANG_SKINNY_PRELOAD", "0"))  # weight units issued before griddepcontrol.wait


@functools.lru_cache(maxsize=None)
def _module(mode: int, use_pdl: bool, stamp: bool = False):
    from sglang.kernels.jit.utils import cuda_stubs_dir, load_jit, make_cpp_args

    args = make_cpp_args(mode, use_pdl)
    return load_jit(
        "skinny_gemm_sm90" + ("_stamp" if stamp else ""),
        *args,
        cuda_files=["gemm/skinny_gemm_sm90.cuh"],
        cuda_wrappers=[("run", f"SkinnyGemmKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"] + (["-DSKINNY_STAMP=1"] if stamp else []),
        extra_ldflags=[f"-L{cuda_stubs_dir()}", "-lcuda"],
    )


@functools.lru_cache(maxsize=None)
def _num_sms(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


@functools.lru_cache(maxsize=None)
def _arch_pdl() -> bool:
    from sglang.kernels.jit.utils import is_arch_support_pdl

    return is_arch_support_pdl()


_NO_PEERS = None


def _run(mode, x, x_scale, w, w_scale, out, preload=None, depth=None, l2pre=0, rs=None, smem_pad=0, next_w=None):
    global _NO_PEERS
    M = x.shape[0]
    N = w.shape[0]
    if rs is not None:
        # reduce-scatter epilogue: rs = (peers host int64 [8], calls int32 cuda [32], rank); no output tensor
        peers, calls, rank = rs
        out = x  # unused placeholder
        rs_args = (peers, calls.data_ptr(), int(rank))
    else:
        if _NO_PEERS is None:
            _NO_PEERS = torch.zeros(8, dtype=torch.int64)
        if out is None:
            out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
        rs_args = (_NO_PEERS, 0, 0)
    idx = x.device.index if x.device.index is not None else torch.cuda.current_device()
    pdl = _arch_pdl() if USE_PDL is None else USE_PDL
    stamp = STAMPS is not None
    d = DEPTH if depth is None else max(1, min(int(depth), 6))
    _module(mode, pdl, stamp).run(
        x, w, x_scale, w_scale, out, _num_sms(idx), CLUSTER, d, WARM, PRELOAD if preload is None else min(int(preload), d),
        STAMPS.data_ptr() if stamp else 0, int(l2pre), *rs_args, int(smem_pad),
        next_w.data_ptr() if next_w is not None else 0,
        (next_w.numel() * next_w.element_size()) // 16 * 16 if next_w is not None else 0,
    )
    return None if rs is not None else out


def skinny_fp8_block_gemm(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    preload: Optional[int] = None,
    depth: Optional[int] = None,
    l2pre: int = 0,
    smem_pad: int = 0,
    next_w: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """x [M, K] e4m3, x_scale [M, K/128] fp32 column-major (x_scale.t() has contiguous rows of stride <= 32),
    w [N, K] e4m3, w_scale [ceil(N/128), K/128] fp32 -> [M, N] bf16. l2pre: weight units L2-prefetched before the PDL
    wait beyond the preloaded ones (-1 = all the rest)."""
    return _run(0, x, x_scale.t(), w, w_scale, out, preload, depth, l2pre, None, smem_pad, next_w)


def skinny_fp8_pt_gemm(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    preload: Optional[int] = None,
    depth: Optional[int] = None,
    l2pre: int = 0,
    rs=None,
) -> Optional[torch.Tensor]:
    """x [M, K] e4m3 with per-token scale x_scale [M] (or [M, 1]), w [N, K] e4m3 with w_scale [1] or [N] -> bf16.
    preload: weight units issued before griddepcontrol.wait (default SGLANG_SKINNY_PRELOAD); useful when the previous
    kernel triggers its dependents early and leaves HBM idle (e.g. the NVLink-bound TP reduce). depth: units in flight
    (default SGLANG_SKINNY_DEPTH, <= 6). l2pre: further units L2-prefetched before the wait (-1 = the whole rest).
    rs: (peers, calls, rank) of a DFlashTPState -> push the partial into the tp_arnq reduce-scatter slots instead of
    returning it (N must be 6144); follow with DFlashTPState.arnq(..., prepushed=True)."""
    return _run(1, x, x_scale.reshape(-1), w, w_scale.reshape(-1), out, preload, depth, l2pre, rs)


def skinny_supported(M: int, N: int, K: int) -> bool:
    return 1 <= M <= MAX_M and K % 128 == 0 and N % 16 == 0
