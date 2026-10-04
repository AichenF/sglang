"""Guarded MiMo target-verify RoPE and BF16 KV-cache fusion."""

import torch

from sglang.srt.environ import envs


def select_mimo_rope_kv(attention, qkv, positions, forward_batch):
    """Return compatible cache arguments without modifying any tensor."""
    if torch.compiler.is_compiling():
        return None
    if qkv.shape != (32, 3392) or qkv.dtype != torch.bfloat16:
        return None
    if not qkv.is_cuda or not qkv.is_contiguous():
        return None
    if torch.cuda.get_device_capability(qkv.device) != (9, 0):
        return None
    if not forward_batch.forward_mode.is_target_verify():
        return None
    if (
        attention.num_heads,
        attention.num_kv_heads,
        attention.head_dim,
        attention.v_head_dim,
    ) != (16, 1, 192, 128):
        return None
    rope = attention.rotary_emb
    if not rope.use_fallback_kernel or not rope.is_neox_style or rope.rotary_dim != 64:
        return None
    cos_sin = rope.cos_sin_cache
    if (
        cos_sin.ndim != 2
        or cos_sin.shape[1] != 64
        or cos_sin.dtype != torch.bfloat16
        or not cos_sin.is_contiguous()
        or cos_sin.device != qkv.device
    ):
        return None
    if (
        positions.shape != (32,)
        or positions.dtype not in (torch.int32, torch.int64)
        or not positions.is_contiguous()
        or positions.device != qkv.device
    ):
        return None

    # Keep backend imports lazy: the feature is opt-in and CUDA-specific.
    from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend()
    if type(backend) is not FlashAttentionBackend:
        return None
    if backend.use_mla or backend.fa_skip_kv_cache or backend.kv_cache_is_mxfp8:
        return None
    pool = backend.token_to_kv_pool
    locations = forward_batch.out_cache_loc
    if type(pool) is SWAKVPool:
        _, sliding = pool.layers_mapping[attention.attn.layer_id]
        physical = pool.swa_kv_pool if sliding else pool.full_kv_pool
        if sliding:
            locations = getattr(backend.forward_metadata, "swa_out_cache_loc", None)
    else:
        physical = pool
    if type(physical) is not MHATokenToKVPool:
        return None
    if physical.dtype != torch.bfloat16 or physical.store_dtype != torch.bfloat16:
        return None
    if physical.use_hnd or physical.is_quantized_kv_cache:
        return None
    if physical.kv_cache_layout == "vectorized_5d":
        return None
    if (
        locations is None
        or locations.shape != (32,)
        or locations.dtype not in (torch.int32, torch.int64)
        or not locations.is_contiguous()
        or locations.device != qkv.device
    ):
        return None
    keys = pool.get_key_buffer(attention.attn.layer_id)
    values = pool.get_value_buffer(attention.attn.layer_id)
    if (
        keys.ndim != 3
        or values.ndim != 3
        or keys.shape[1:] != (1, 192)
        or values.shape[1:] != (1, 128)
        or keys.shape[0] != values.shape[0]
    ):
        return None
    if any(
        tensor.dtype != torch.bfloat16
        or tensor.device != qkv.device
        or not tensor.is_contiguous()
        for tensor in (keys, values)
    ):
        return None
    return cos_sin, locations, keys, values


def try_fused_mimo_rope_kv(attention, qkv, positions, forward_batch):
    """Rotate Q/K and write KV; false means the original path must run."""
    if not envs.SGLANG_OPT_MIMO_ROPE_KV.get():
        return False
    selected = select_mimo_rope_kv(attention, qkv, positions, forward_batch)
    if selected is None:
        return False
    from sglang.kernels.ops.attention.mimo_rope_kv import run

    cos_sin, locations, keys, values = selected
    run(qkv, cos_sin, positions, locations, keys, values)
    return True
