"""Fused per-block grouped convolution for DFlash2 inference.

This primitive does not enable the TP fused backbone or persistent MegaKernel.
Its arithmetic follows the compiled pointwise reference: promote inputs to
float32, accumulate there, and round only when writing the output.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_conv_kernel(
    hidden,
    delta,
    base,
    output,
    WIDTH: tl.constexpr,
    TAPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DELTA_ROW_STRIDE: tl.constexpr,
    DELTA_TAP_STRIDE: tl.constexpr,
    DELTA_GROUP_STRIDE: tl.constexpr,
    CHANNEL_TILE: tl.constexpr,
):
    tiles_per_row = tl.cdiv(WIDTH, CHANNEL_TILE)
    tile = tl.program_id(0)
    row = tile // tiles_per_row
    channel = (tile % tiles_per_row) * CHANNEL_TILE + tl.arange(0, CHANNEL_TILE)
    valid_channel = channel < WIDTH
    group = channel // GROUP_SIZE
    position = row % BLOCK_SIZE
    accumulator = tl.full((CHANNEL_TILE,), 0, tl.float32)
    for tap in tl.static_range(TAPS):
        # Mask the load itself: multiplying a previous request's NaN/Inf by
        # zero after loading it would still contaminate this request.
        history = tl.load(
            hidden + (row - tap) * WIDTH + channel,
            mask=valid_channel & (position >= tap),
            other=0,
        ).to(tl.float32)
        static = tl.load(
            base + tap * WIDTH + channel, mask=valid_channel, other=0
        ).to(tl.float32)
        dynamic = tl.load(
            delta
            + row * DELTA_ROW_STRIDE
            + tap * DELTA_TAP_STRIDE
            + group * DELTA_GROUP_STRIDE,
            mask=valid_channel,
            other=0,
        ).to(tl.float32)
        accumulator += (static + dynamic) * history
    tl.store(output + row * WIDTH + channel, accumulator, mask=valid_channel)


def grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    """Apply a grouped causal stencil independently to each draft block.

    ``hidden_states`` is contiguous [M, H], ``base`` is contiguous [K, H],
    and ``delta`` is [M, K, H / group_size]. Delta may be strided, including
    either side of the model's shared [M, 2, K, H / group_size] projection.
    All tensors must share a CUDA device and BF16, FP16 or FP32 dtype.

    K is bounded to 1..16 to bound static kernel expansion. K may exceed the
    block size or row count: unavailable history contributes zero. M may be
    zero, and the last block may be partial. Blocks always start at row zero.
    Returns a new tensor with the input shape, dtype and device. This is an
    inference primitive and does not provide autograd.
    """
    tensors = (hidden_states, delta, base)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("hidden_states, delta and base must be tensors")
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if type(group_size) is not int or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    if any(tensor.layout != torch.strided for tensor in tensors):
        raise ValueError("grouped_conv requires dense strided tensors")
    if hidden_states.ndim != 2 or delta.ndim != 3 or base.ndim != 2:
        raise ValueError("expected hidden_states [M,H], delta [M,K,G], base [K,H]")
    if not hidden_states.is_cuda or any(
        tensor.device != hidden_states.device for tensor in tensors
    ):
        raise ValueError("all grouped_conv tensors must share a CUDA device")
    if hidden_states.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("grouped_conv supports only BF16, FP16 and FP32")
    if any(tensor.dtype != hidden_states.dtype for tensor in tensors):
        raise TypeError("all grouped_conv tensors must have the same dtype")
    if not hidden_states.is_contiguous() or not base.is_contiguous():
        raise ValueError("hidden_states and base must be contiguous")

    rows, width = hidden_states.shape
    taps = base.shape[0]
    if width <= 0 or width % group_size:
        raise ValueError("group_size must divide positive hidden width")
    if not 1 <= taps <= 16:
        raise ValueError("grouped_conv supports 1 through 16 taps")
    if base.shape[1] != width or delta.shape != (rows, taps, width // group_size):
        raise ValueError("base or delta shape does not match hidden_states and groups")

    output = torch.empty_like(hidden_states)
    if rows == 0:
        return output

    channel_tile = 256
    # Select the tensor's device even when another GPU is current. No host read
    # of tensor values or synchronization occurs, including during graph capture.
    with torch.cuda.device(hidden_states.device):
        _grouped_conv_kernel[(rows * triton.cdiv(width, channel_tile),)](
            hidden_states,
            delta,
            base,
            output,
            width,
            taps,
            block_size,
            group_size,
            *delta.stride(),
            channel_tile,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output
