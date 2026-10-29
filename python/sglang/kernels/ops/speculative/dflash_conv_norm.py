"""DFlash BF16 output convolution followed by residual addition and RMSNorm.

This standalone boundary primitive does not enable TP or persistent model paths.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _conv_add_norm_kernel(
    hidden, delta, base, residual, gamma, output,
    EPS: tl.constexpr, TAPS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr, DELTA_ROW: tl.constexpr,
    DELTA_TAP: tl.constexpr, DELTA_GROUP: tl.constexpr,
    WIDTH: tl.constexpr, TILE: tl.constexpr,
):
    row = tl.program_id(0)
    channel = tl.arange(0, TILE)
    valid = channel < WIDTH
    group = channel // GROUP_SIZE
    position = row % BLOCK_SIZE
    conv = tl.full((TILE,), 0, tl.float32)
    for tap in tl.static_range(TAPS):
        history = tl.load(
            hidden + (row - tap) * WIDTH + channel,
            mask=valid & (position >= tap), other=0,
        ).to(tl.float32)
        static = tl.load(base + tap * WIDTH + channel, mask=valid, other=0).to(tl.float32)
        dynamic = tl.load(
            delta + row * DELTA_ROW + tap * DELTA_TAP + group * DELTA_GROUP,
            mask=valid, other=0,
        ).to(tl.float32)
        conv += (static + dynamic) * history

    # The unfused convolution writes BF16 before RMSNorm reads it. Keep this
    # explicit rounding even though the following addition uses FP32 registers.
    rounded_conv = conv.to(tl.bfloat16).to(tl.float32)
    old_residual = tl.load(residual + row * WIDTH + channel, mask=valid, other=0).to(tl.float32)
    summed = rounded_conv + old_residual
    variance = tl.sum(tl.where(valid, summed * summed, 0), axis=0) / WIDTH
    inv_rms = tl.rsqrt(variance + EPS)
    weight = tl.load(gamma + channel, mask=valid, other=0).to(tl.float32)
    normalized = summed * inv_rms * weight
    # Store rounded residual separately; normalization retains the FP32 sum.
    tl.store(residual + row * WIDTH + channel, summed, mask=valid)
    tl.store(output + row * WIDTH + channel, normalized, mask=valid)


def grouped_conv_add_rmsnorm(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    block_size: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(new_normalized_output, residual)`` with residual updated in place.

    CUDA BF16 only: hidden/residual are contiguous [M,6144], 1 <= M <= 32;
    base is contiguous [K,6144], 1 <= K <= 16; gamma is contiguous [6144].
    Delta is [M,K,6144/group_size] with arbitrary nonnegative strides, including
    output-side views of [M,2,K,G]. Blocks start at row zero and never share
    history. A final partial block is allowed.

    Hidden, delta, base and gamma are read-only. Residual must not share storage
    with any of them, including disjoint views of one allocation. This prevents
    cross-CTA history corruption. Validation reads metadata only and is capture
    safe. This inference primitive does not provide autograd.
    """
    tensors = (hidden_states, delta, base, residual, gamma)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("all convolution/norm inputs must be tensors")
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if type(group_size) is not int or group_size <= 0 or 6144 % group_size:
        raise ValueError("group_size must divide 6144")
    if isinstance(eps, bool) or not isinstance(eps, (float, int)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a finite positive number")
    if any(value.layout != torch.strided for value in tensors):
        raise ValueError("all convolution/norm inputs must be strided tensors")
    if not hidden_states.is_cuda or any(value.device != hidden_states.device for value in tensors):
        raise ValueError("all convolution/norm inputs must share a CUDA device")
    if any(value.dtype != torch.bfloat16 for value in tensors):
        raise TypeError("convolution/norm supports BF16 only")
    if hidden_states.ndim != 2 or hidden_states.shape[1] != 6144:
        raise ValueError("hidden_states must have shape [M,6144]")
    rows = hidden_states.shape[0]
    if not 1 <= rows <= 32:
        raise ValueError("convolution/norm supports 1 through 32 rows")
    if base.ndim != 2 or base.shape[1] != 6144 or not 1 <= base.shape[0] <= 16:
        raise ValueError("base must have shape [K,6144] with 1 <= K <= 16")
    if delta.shape != (rows, base.shape[0], 6144 // group_size):
        raise ValueError("delta shape must match rows, taps and groups")
    if residual.shape != hidden_states.shape or gamma.shape != (6144,):
        raise ValueError("residual or gamma shape does not match hidden_states")
    if not all(value.is_contiguous() for value in (hidden_states, base, residual, gamma)):
        raise ValueError("hidden_states, base, residual and gamma must be contiguous")
    residual_storage = residual.untyped_storage()
    residual_begin = residual_storage.data_ptr()
    residual_end = residual_begin + residual_storage.nbytes()
    for value in (hidden_states, delta, base, gamma):
        storage = value.untyped_storage()
        begin = storage.data_ptr()
        end = begin + storage.nbytes()
        # DLPack can expose an offset view as a new storage whose starting
        # pointer differs while its allocation still overlaps residual.
        if begin < residual_end and residual_begin < end:
            raise ValueError("residual storage must not overlap or alias read inputs")

    output = torch.empty_like(hidden_states)
    with torch.cuda.device(hidden_states.device):
        _conv_add_norm_kernel[(rows,)](
            hidden_states, delta, base, residual, gamma, output,
            float(eps), base.shape[0], block_size, group_size, *delta.stride(),
            6144, triton.next_power_of_2(6144), num_warps=8,
            enable_fp_fusion=False,
        )
    return output, residual
