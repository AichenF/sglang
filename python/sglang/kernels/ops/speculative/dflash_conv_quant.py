"""Grouped block convolution followed by per-token FP8 quantization, for the fused TP draft path.

The convolution matches DFlashGroupedConv (fp32 accumulate, BF16 output); the quantization
matches the per-token FP8 quant of the unfused linears as built for sm_90 (fast math):
scale = amax * (1 / 448), q = saturate(x * rcp.approx.ftz(scale)), zero rows -> scale 0.
"""

import torch
import triton
import triton.language as tl

FP8_MAX = 448.0


@triton.jit
def _rcp_approx(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _conv_quant_kernel(
    hidden,
    delta,
    base,
    out_q,
    out_scale,
    WIDTH: tl.constexpr,
    TAPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DELTA_ROW_STRIDE: tl.constexpr,
    DELTA_TAP_STRIDE: tl.constexpr,
    DELTA_GROUP_STRIDE: tl.constexpr,
    PADDED_WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    channel = tl.arange(0, PADDED_WIDTH)
    valid = channel < WIDTH
    group = channel // GROUP_SIZE
    position = row % BLOCK_SIZE
    acc = tl.zeros((PADDED_WIDTH,), tl.float32)
    for tap in tl.static_range(TAPS):
        history = tl.load(
            hidden + (row - tap) * WIDTH + channel,
            mask=valid & (position >= tap),
            other=0,
        ).to(tl.float32)
        static = tl.load(base + tap * WIDTH + channel, mask=valid, other=0).to(tl.float32)
        dynamic = tl.load(
            delta + row * DELTA_ROW_STRIDE + tap * DELTA_TAP_STRIDE + group * DELTA_GROUP_STRIDE,
            mask=valid,
            other=0,
        ).to(tl.float32)
        acc += (static + dynamic) * history
    y = acc.to(tl.bfloat16).to(tl.float32)
    amax = tl.max(tl.where(valid, tl.abs(y), 0.0), axis=0)
    scale = amax * (1.0 / 448.0)
    inv = tl.where(scale != 0.0, _rcp_approx(scale), 0.0)
    q = tl.minimum(tl.maximum(y * inv, -448.0), 448.0)
    tl.store(out_q + row * WIDTH + channel, q.to(tl.float8e4nv), mask=valid)
    tl.store(out_scale + row, scale)


def grouped_conv_quant(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    group_size: int,
    out_q: torch.Tensor,
    out_scale: torch.Tensor,
) -> None:
    """hidden_states [M, H] bf16 contiguous, delta [M, K, H / group_size] (may be strided),
    base [K, H] contiguous -> out_q [M, H] e4m3, out_scale [M] fp32."""
    rows, width = hidden_states.shape
    taps = base.shape[0]
    if rows == 0:
        return
    _conv_quant_kernel[(rows,)](
        hidden_states,
        delta,
        base,
        out_q,
        out_scale,
        WIDTH=width,
        TAPS=taps,
        BLOCK_SIZE=block_size,
        GROUP_SIZE=group_size,
        DELTA_ROW_STRIDE=delta.stride(0),
        DELTA_TAP_STRIDE=delta.stride(1),
        DELTA_GROUP_STRIDE=delta.stride(2),
        PADDED_WIDTH=triton.next_power_of_2(width),
        num_warps=8,
    )
