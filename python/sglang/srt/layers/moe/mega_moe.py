# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Mega-MoE forward path and expert-weight prep shared by Deepseek V2/V4."""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.dsv4.moe import (
    mega_moe_pre_dispatch,
    mega_moe_pre_dispatch_sm90,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.dp_attention import get_dp_global_num_tokens
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.models.deepseek_common.utils import _device_sm
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from deep_gemm import SymmBuffer

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE


_MEGA_MOE_SYMM_BUFFER: dict = {}
_MEGA_MOE_DG_ENV_APPLIED = False


@dataclass(frozen=True)
class _MegaMoeArchConfig:
    name: str
    deep_gemm_entry: str
    run_recipe: tuple[int, int, int]
    scale_recipe: tuple[int, int]
    pre_dispatch_group_size: int
    fp4_weight_packed: bool
    uses_raw_fp32_scales: bool
    use_dp_max_tokens: bool
    fold_routed_scaling_in_pre_dispatch: bool


_SM90_FP8_CONFIG = _MegaMoeArchConfig(
    name="sm90_fp8",
    deep_gemm_entry="fp8_mega_moe",
    run_recipe=(128, 128, 128),
    scale_recipe=(128, 128),
    pre_dispatch_group_size=128,
    fp4_weight_packed=False,
    uses_raw_fp32_scales=True,
    use_dp_max_tokens=True,
    fold_routed_scaling_in_pre_dispatch=True,
)
_SM100_FP8_FP4_CONFIG = _MegaMoeArchConfig(
    name="sm100_fp8_fp4",
    deep_gemm_entry="fp8_fp4_mega_moe",
    run_recipe=(1, 1, 32),
    scale_recipe=(1, 32),
    pre_dispatch_group_size=32,
    fp4_weight_packed=True,
    uses_raw_fp32_scales=False,
    use_dp_max_tokens=False,
    fold_routed_scaling_in_pre_dispatch=False,
)
_SM90_NVFP4_CONFIG = _MegaMoeArchConfig(
    name="sm90_nvfp4",
    deep_gemm_entry="nvfp4_mega_moe",
    run_recipe=(128, 128, 128),
    scale_recipe=(1, 16),
    pre_dispatch_group_size=128,
    fp4_weight_packed=True,
    uses_raw_fp32_scales=False,
    use_dp_max_tokens=True,
    fold_routed_scaling_in_pre_dispatch=True,
)
_MEGA_MOE_ARCH_CONFIGS = {
    config.name: config
    for config in (_SM90_FP8_CONFIG, _SM100_FP8_FP4_CONFIG, _SM90_NVFP4_CONFIG)
}


def _select_mega_moe_arch_config(
    w13: torch.Tensor, w2: torch.Tensor
) -> Optional[_MegaMoeArchConfig]:
    if (
        _device_sm == 90
        and w13.dtype == torch.float8_e4m3fn
        and w2.dtype == torch.float8_e4m3fn
    ):
        return _SM90_FP8_CONFIG
    if _device_sm == 90 and w13.dtype == torch.uint8 and w2.dtype == torch.uint8:
        return _SM90_NVFP4_CONFIG
    if (
        _device_sm is not None
        and _device_sm >= 100
        and w13.dtype == torch.int8
        and w2.dtype == torch.int8
    ):
        return _SM100_FP8_FP4_CONFIG
    return None


def _get_built_mega_moe_arch_config(experts) -> Optional[_MegaMoeArchConfig]:
    return _MEGA_MOE_ARCH_CONFIGS.get(getattr(experts, "_mega_moe_arch", None))


def _apply_mega_moe_dg_env() -> None:
    """Forward sglang's FP4/MXF4 opt-in flags to DeepGEMM via env vars.

    DeepGEMM reads `DG_USE_FP4_ACTS` (and `DG_USE_MXF4_KIND`) at host-function
    call time — both `get_symm_buffer_for_mega_moe` and `fp8_fp4_mega_moe`.
    Forwarding once at first use is sufficient (these are static config
    flags, not per-request state) and matches the `setdefault` pattern so
    explicit `DG_USE_*` overrides from outside still win.
    """
    global _MEGA_MOE_DG_ENV_APPLIED
    if _MEGA_MOE_DG_ENV_APPLIED:
        return
    if envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS.get():
        os.environ.setdefault("DG_USE_FP4_ACTS", "1")
    if envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND.get():
        os.environ.setdefault("DG_USE_MXF4_KIND", "1")
    _MEGA_MOE_DG_ENV_APPLIED = True


def _get_mega_moe_symm_buffer(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> SymmBuffer:
    import deep_gemm

    _apply_mega_moe_dg_env()

    key = (
        id(group),
        num_max_tokens_per_rank,
        num_experts,
        num_topk,
        hidden,
        intermediate_hidden,
    )
    buf = _MEGA_MOE_SYMM_BUFFER.get(key)
    if buf is None:
        buf = deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            use_fp8_dispatch=True,
            activation="swiglu",
        )
        _MEGA_MOE_SYMM_BUFFER[key] = buf
    return buf


def _ensure_mega_moe_symm_buffer(moe: "DeepseekV2MoE") -> SymmBuffer:
    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    return _get_mega_moe_symm_buffer(
        get_moe_ep_group().device_group,
        num_experts=moe.experts.num_experts,
        num_max_tokens_per_rank=(
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
        ),
        num_topk=moe.config.num_experts_per_tok + moe.num_fused_shared_experts,
        hidden=moe.config.hidden_size,
        intermediate_hidden=moe.config.moe_intermediate_size,
    )


def _get_dp_global_num_tokens_or_none() -> Optional[list[int]]:
    try:
        return get_dp_global_num_tokens()
    except AttributeError:
        return None


def _deep_gemm_supports_mega_moe_config(config: _MegaMoeArchConfig) -> bool:
    try:
        import deep_gemm
    except ImportError:
        return False
    return hasattr(deep_gemm, config.deep_gemm_entry)


def _get_effective_num_tokens(config: _MegaMoeArchConfig, num_tokens: int) -> int:
    if not config.use_dp_max_tokens:
        return num_tokens
    global_num_tokens = _get_dp_global_num_tokens_or_none()
    if not global_num_tokens:
        effective_num_tokens = num_tokens
    else:
        effective_num_tokens = max(max(global_num_tokens), num_tokens)
    if (
        0 < effective_num_tokens < config.pre_dispatch_group_size
        and (
            _get_disaggregation_mode_or_none() == "prefill"
            or config.name == _SM90_NVFP4_CONFIG.name
        )
    ):
        effective_num_tokens = config.pre_dispatch_group_size

    # SM90 NVFP4 MegaMoE corrupts exact power-of-two token tiles on H20.
    # Pad one dummy token and slice back after the kernel.
    if (
        config.name == _SM90_NVFP4_CONFIG.name
        and effective_num_tokens >= config.pre_dispatch_group_size
        and effective_num_tokens & (effective_num_tokens - 1) == 0
    ):
        effective_num_tokens += 1
    return effective_num_tokens



def _get_disaggregation_mode_or_none() -> Optional[str]:
    try:
        return getattr(get_global_server_args(), "disaggregation_mode", None)
    except ValueError:
        return None


def should_use_mega_moe(moe: "DeepseekV2MoE", hidden_states: torch.Tensor) -> bool:
    if not get_moe_a2a_backend().is_megamoe():
        return False
    if not getattr(moe.experts, "_mega_moe_weights_built", False):
        return False

    config = _get_built_mega_moe_arch_config(moe.experts)
    if config is None or not _deep_gemm_supports_mega_moe_config(config):
        return False
    is_capture_mode = get_is_capture_mode()
    max_tokens_per_rank = _get_effective_num_tokens(config, hidden_states.shape[0])
    if is_capture_mode:
        return True

    cap = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    return max_tokens_per_rank <= cap


def forward_mega_moe(
    moe: "DeepseekV2MoE",
    hidden_states: torch.Tensor,
    forward_batch: Optional["ForwardBatch"] = None,
    input_ids_global: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]

    sbo_overlap_flag = (
        moe.alt_stream is not None
        and moe.num_fused_shared_experts == 0
        and num_tokens > 0
        and get_is_capture_mode()
    )

    if sbo_overlap_flag:
        current_stream = torch.cuda.current_stream()
        moe.alt_stream.wait_stream(current_stream)
        shared_output = moe._forward_shared_experts(hidden_states)
        mega_stream_ctx = torch.cuda.stream(moe.alt_stream)
    else:
        shared_output = moe._forward_shared_experts(hidden_states)
        mega_stream_ctx = nullcontext()

    with mega_stream_ctx:
        y = _run_mega_routed(
            moe, hidden_states, forward_batch, input_ids_global, num_tokens
        )

    if sbo_overlap_flag:
        current_stream.wait_stream(moe.alt_stream)

    if shared_output is not None:
        y.add_(shared_output)
    return y


def _run_mega_routed(
    moe: "DeepseekV2MoE",
    hidden_states: torch.Tensor,
    forward_batch: Optional["ForwardBatch"],
    input_ids_global: Optional[torch.Tensor],
    num_tokens: int,
) -> torch.Tensor:
    import deep_gemm

    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    config = _get_built_mega_moe_arch_config(moe.experts)
    assert config is not None, "MegaMoE weights must be built before forward"

    hidden_size = moe.config.hidden_size
    effective_num_tokens = _get_effective_num_tokens(config, num_tokens)
    if config.use_dp_max_tokens and effective_num_tokens == 0:
        _ensure_mega_moe_symm_buffer(moe)
        return hidden_states.new_empty((0, hidden_size))

    if num_tokens > 0:
        router_logits = moe.gate(hidden_states, forward_batch=forward_batch)
        topk_kwargs = {"input_ids": input_ids_global} if moe.is_hash else {}
        with get_global_expert_distribution_recorder().with_current_layer(
            moe.layer_id
        ):
            topk_output = moe.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=(
                    forward_batch.num_token_non_padded
                    if forward_batch is not None
                    else None
                ),
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=moe.layer_id,
                ),
                **topk_kwargs,
            )
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights
    else:
        topk_ids = None
        topk_weights = None

    moe_ep_group = get_moe_ep_group()
    ep_group = moe_ep_group.device_group
    num_experts = moe.experts.num_experts
    top_k = moe.config.num_experts_per_tok + moe.num_fused_shared_experts
    intermediate_size = moe.config.moe_intermediate_size
    if (
        num_tokens > 0
        and os.environ.get("SGLANG_MEGAMOE_DEBUG_TOPK", "0") == "1"
        and not getattr(moe, "_mega_moe_debug_topk_printed", False)
    ):
        moe._mega_moe_debug_topk_printed = True
        try:
            topk_min = int(topk_ids.min().item())
            topk_max = int(topk_ids.max().item())
            weights_min = float(topk_weights.min().item())
            weights_max = float(topk_weights.max().item())
            sample_rows = min(2, topk_ids.shape[0])
            topk_sample = topk_ids[:sample_rows].detach().cpu().tolist()
            weight_sample = topk_weights[:sample_rows].detach().cpu().tolist()
            print(
                "[MegaMoE debug] "
                f"layer={moe.layer_id} arch={config.name} "
                f"ep_rank={moe_ep_group.rank_in_group}/{moe_ep_group.world_size} "
                f"num_experts={num_experts} "
                f"num_local_experts={moe.experts.num_local_experts} "
                f"num_fused_shared={moe.num_fused_shared_experts} top_k={top_k} "
                f"topk_shape={tuple(topk_ids.shape)} topk_min={topk_min} "
                f"topk_max={topk_max} weights_min={weights_min:.6g} "
                f"weights_max={weights_max:.6g} topk_sample={topk_sample} "
                f"weight_sample={weight_sample}",
                flush=True,
            )
        except Exception as exc:
            print(f"[MegaMoE debug] failed to inspect topk: {exc}", flush=True)

    num_max_tokens_per_rank = (
        envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    )
    assert effective_num_tokens <= num_max_tokens_per_rank, (
        f"mega MoE: effective_num_tokens={effective_num_tokens} exceeds cap "
        f"SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="
        f"{num_max_tokens_per_rank}; raise the env var or shrink "
        f"cuda_graph_max_bs / chunked_prefill_size accordingly"
    )
    buf = _get_mega_moe_symm_buffer(
        ep_group,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_topk=top_k,
        hidden=hidden_size,
        intermediate_hidden=intermediate_size,
    )
    dispatch_num_tokens = effective_num_tokens if config.use_dp_max_tokens else num_tokens
    dispatch_hidden_states = (
        hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
    )
    pad_dispatch_inputs = dispatch_num_tokens > num_tokens
    if pad_dispatch_inputs:
        dispatch_hidden_states = hidden_states.new_zeros(
            (dispatch_num_tokens, hidden_size)
        )
        if num_tokens > 0:
            dispatch_hidden_states[:num_tokens].copy_(hidden_states)

    dummy_expert_ids = None
    if config.fold_routed_scaling_in_pre_dispatch:
        num_experts_per_rank = num_experts // moe_ep_group.world_size
        dummy_expert_base = moe_ep_group.rank_in_group * num_experts_per_rank
        dummy_expert_ids = torch.arange(
            dummy_expert_base,
            dummy_expert_base + top_k,
            device=hidden_states.device,
            dtype=torch.int32,
        )

    if num_tokens > 0:
        topk_ids_in = topk_ids.to(torch.int32).contiguous()
        topk_weights_in = topk_weights.to(torch.float32).contiguous()
        if dummy_expert_ids is not None:
            invalid_topk = topk_ids_in < 0
            dummy_topk_ids = dummy_expert_ids.view(1, top_k).expand_as(topk_ids_in)
            topk_ids_in = torch.where(invalid_topk, dummy_topk_ids, topk_ids_in)
            topk_weights_in = topk_weights_in.masked_fill(invalid_topk, 0.0)
    else:
        topk_ids_in = hidden_states.new_empty((0, top_k), dtype=torch.int32)
        topk_weights_in = hidden_states.new_empty((0, top_k), dtype=torch.float32)
    if pad_dispatch_inputs:
        padded_topk_ids = hidden_states.new_full(
            (dispatch_num_tokens, top_k), -1, dtype=torch.int32
        )
        padded_topk_weights = hidden_states.new_zeros(
            (dispatch_num_tokens, top_k), dtype=torch.float32
        )
        if dummy_expert_ids is not None:
            padded_topk_ids[num_tokens:, :].copy_(dummy_expert_ids)
        if num_tokens > 0:
            padded_topk_ids[:num_tokens].copy_(topk_ids_in)
            padded_topk_weights[:num_tokens].copy_(topk_weights_in)
        topk_ids_in = padded_topk_ids
        topk_weights_in = padded_topk_weights


    fused_routed_scaling = False

    if config.fold_routed_scaling_in_pre_dispatch:
        if moe.experts.should_fuse_routed_scaling_factor_in_topk:
            scale = 1.0
        else:
            scale = float(moe.routed_scaling_factor)
            fused_routed_scaling = True
        mega_moe_pre_dispatch_sm90(
            dispatch_hidden_states,
            topk_ids_in,
            topk_weights_in,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            routed_scaling_factor=scale,
            quant_group_size=config.pre_dispatch_group_size,
        )
    elif envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS.get():
        # FP4 path goes through DeepGEMM's mega_moe_pre_dispatch which
        # handles the E2M1 packing variant. The jit implementation
        # only emits FP8.
        deep_gemm.mega_moe_pre_dispatch(
            dispatch_hidden_states,
            topk_ids_in,
            topk_weights_in,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            num_tokens=dispatch_num_tokens,
            group_size=config.pre_dispatch_group_size,
            use_fp4_acts=True,
        )
    else:
        mega_moe_pre_dispatch(
            dispatch_hidden_states,
            topk_ids_in,
            topk_weights_in,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            quant_group_size=config.pre_dispatch_group_size,
        )

    y_num_tokens = (
        effective_num_tokens if config.use_dp_max_tokens else max(num_tokens, 1)
    )
    y = torch.empty(
        (y_num_tokens, hidden_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    swiglu_limit = getattr(moe.config, "swiglu_limit", None)
    mega_moe_kwargs = {
        "recipe": config.run_recipe,
        "activation": "swiglu",
        "activation_clamp": swiglu_limit,
        "fast_math": True,
    }
    if config.name == _SM90_NVFP4_CONFIG.name:
        mega_moe_kwargs.update(
            {
                "l1_global_scales": getattr(moe.experts, "mega_l1_global_scales", None),
                "l2_global_scales": getattr(moe.experts, "mega_l2_global_scales", None),
            }
        )
    getattr(deep_gemm, config.deep_gemm_entry)(
        y, moe.experts.mega_l1_weights, moe.experts.mega_l2_weights, buf, **mega_moe_kwargs
    )
    y = y[:num_tokens]

    if (
        not moe.experts.should_fuse_routed_scaling_factor_in_topk
        and not fused_routed_scaling
    ):
        y.mul_(moe.routed_scaling_factor)
    return y


def _interleave_l1_weight_only(weight: torch.Tensor, gran: int = 8) -> torch.Tensor:
    num_groups, n, *rest = weight.shape
    half = n // 2
    gate = weight[:, :half].reshape(num_groups, half // gran, gran, *rest)
    up = weight[:, half:].reshape(num_groups, half // gran, gran, *rest)
    return torch.empty_like(weight).copy_(
        torch.stack([gate, up], dim=2).reshape(num_groups, n, *rest)
    )


def _interleave_l1_tensor_inplace(tensor: torch.Tensor, gran: int = 8) -> torch.Tensor:
    num_groups, n, *rest = tensor.shape
    half = n // 2
    assert half % gran == 0, f"invalid gated L1 shape for interleave: {tensor.shape}"
    num_blocks = half // gran
    blocks = tensor.reshape(num_groups, num_blocks * 2, gran, *rest)
    old_to_new = [2 * i for i in range(num_blocks)] + [
        2 * i + 1 for i in range(num_blocks)
    ]
    visited = [False] * (num_blocks * 2)
    for start in range(num_blocks * 2):
        if visited[start]:
            continue
        target = old_to_new[start]
        if target == start:
            visited[start] = True
            continue
        tmp = blocks[:, start, ...].clone()
        current = start
        while True:
            visited[current] = True
            target = old_to_new[current]
            if target == start:
                blocks[:, target, ...].copy_(tmp)
                visited[target] = True
                break
            next_tmp = blocks[:, target, ...].clone()
            blocks[:, target, ...].copy_(tmp)
            tmp = next_tmp
            current = target
    return tensor


def _modelopt_nvfp4_to_deepgemm_packed_inplace(
    tensor: torch.Tensor, chunk_n: int = 128
) -> torch.Tensor:
    """Convert ModelOpt FP4x2 byte order to DeepGEMM SM90 NVFP4 order.

    ModelOpt/PyTorch FP4x2 stores each byte as low=K[2*i], high=K[2*i+1].
    DeepGEMM's SM90 NVFP4 loader consumes the Marlin-style packing used by
    `deep_gemm.quantize_to_nvfp4`: for each logical K[0..7] chunk, the four
    bytes store high=K[0..3] and low=K[4..7].
    """
    assert tensor.dtype == torch.uint8, tensor.dtype
    assert tensor.dim() == 3, tensor.shape
    assert tensor.shape[-1] % 4 == 0, tensor.shape

    chunk_n = max(1, chunk_n)
    for start in range(0, tensor.shape[1], chunk_n):
        end = min(start + chunk_n, tensor.shape[1])
        chunk = tensor[:, start:end, :]
        blocks = chunk.view(*chunk.shape[:-1], -1, 4)
        r0 = blocks[..., 0].clone()
        r1 = blocks[..., 1].clone()
        r2 = blocks[..., 2].clone()
        r3 = blocks[..., 3].clone()

        blocks[..., 0].copy_(((r0 & 0x0F) << 4) | (r2 & 0x0F))
        blocks[..., 1].copy_(((r0 >> 4) << 4) | (r2 >> 4))
        blocks[..., 2].copy_(((r1 & 0x0F) << 4) | (r3 & 0x0F))
        blocks[..., 3].copy_(((r1 >> 4) << 4) | (r3 >> 4))
    return tensor


def _modelopt_nvfp4_scale_to_ue4m3(
    weight_scale: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    gated_l1: bool,
    round_to_nearest: bool = False,
) -> torch.Tensor:
    from deep_gemm.quantization_nvfp4 import (
        UE4M3_MAX_DENORM,
        UE4M3_MAX_FINITE,
        UE4M3_MIN_DENORM,
        UE4M3_MIN_NORMAL,
        fp32_to_ue4m3_ceil,
        ue4m3_to_fp32,
    )

    def fp32_to_ue4m3_nearest(x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32).clamp(min=UE4M3_MIN_DENORM, max=UE4M3_MAX_FINITE)
        denorm_code = torch.round(x / UE4M3_MIN_DENORM).to(torch.int32).clamp(1, 7)
        x_norm = torch.clamp(x, min=UE4M3_MIN_NORMAL)
        exp_unbiased = torch.floor(torch.log2(x_norm))
        exp_bits = (exp_unbiased + 7).to(torch.int32)
        base = torch.exp2(exp_unbiased)
        mant = torch.round((x_norm / base - 1.0) * 8.0).to(torch.int32)
        overflow = mant > 7
        exp_bits = torch.where(overflow, exp_bits + 1, exp_bits)
        mant = torch.where(overflow, torch.zeros_like(mant), mant).clamp(0, 7)
        normal_code = (exp_bits * 8 + mant).clamp(8, 0x7E)
        code = torch.where(x <= UE4M3_MAX_DENORM, denorm_code, normal_code)
        return code.to(torch.uint8)

    global_scale = global_scale.to(torch.float32)
    encode_scale = fp32_to_ue4m3_nearest if round_to_nearest else fp32_to_ue4m3_ceil
    encoded = torch.empty(
        weight_scale.shape, dtype=torch.uint8, device=weight_scale.device
    )
    chunk_n = int(os.environ.get("DG_SM90_NVFP4_SCALE_ENCODE_CHUNK_N", "128"))
    chunk_n = max(1, chunk_n)
    n = weight_scale.shape[1]
    half_n = n // 2

    def encode_range(start: int, end: int, tensor_scale: torch.Tensor) -> None:
        chunk = weight_scale[:, start:end, :]
        if weight_scale.dtype == torch.uint8:
            folded = ue4m3_to_fp32(chunk)
        else:
            folded = chunk.to(torch.float32)
        folded.mul_(tensor_scale.reshape(-1, 1, 1))
        encoded[:, start:end, :] = encode_scale(folded)

    if gated_l1 and global_scale.dim() == 2 and global_scale.shape[1] >= 2:
        for start in range(0, half_n, chunk_n):
            encode_range(start, min(start + chunk_n, half_n), global_scale[:, 0])
        for start in range(half_n, n, chunk_n):
            encode_range(start, min(start + chunk_n, n), global_scale[:, 1])
    else:
        if global_scale.dim() == 2:
            global_scale = global_scale[:, 0]
        for start in range(0, n, chunk_n):
            encode_range(start, min(start + chunk_n, n), global_scale)
    return encoded.contiguous()


def _modelopt_nvfp4_global_scale_1d(
    global_scale: torch.Tensor,
    *,
    gated_l1: bool,
    device: torch.device,
) -> torch.Tensor:
    global_scale = global_scale.to(device=device, dtype=torch.float32)
    if global_scale.dim() == 0:
        global_scale = global_scale.reshape(1)
    if global_scale.dim() == 1:
        return global_scale.contiguous()
    if global_scale.dim() != 2:
        raise AssertionError(
            f"unexpected ModelOpt NVFP4 global scale shape: {tuple(global_scale.shape)}"
        )
    if global_scale.shape[1] == 1:
        return global_scale[:, 0].contiguous()
    if gated_l1:
        gate_scale = global_scale[:, 0]
        up_scale = global_scale[:, 1]
        torch.testing.assert_close(
            gate_scale,
            up_scale,
            msg="SM90 NVFP4 MegaMoE expects equal gate/up weight_scale_2 per expert",
        )
        return gate_scale.contiguous()
    return global_scale[:, 0].contiguous()


def _modelopt_nvfp4_prefold_scale(
    weight_scale: torch.Tensor, *, device: torch.device
) -> torch.Tensor:
    max_scale = weight_scale.to(torch.float32).amax(dim=(1, 2)).clamp_min(1e-30)
    safety = float(os.environ.get("SGLANG_MEGAMOE_NVFP4_PREFOLD_SAFETY", "0.95"))
    prefold = (448.0 * safety / (6.0 * max_scale)).clamp(max=1.0)
    return prefold.to(device=device, dtype=torch.float32).contiguous()


def _requantize_modelopt_nvfp4_for_deepgemm(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    gated_l1: bool,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from deep_gemm.quantization_nvfp4 import quantize_to_nvfp4

    assert weight.dtype == torch.uint8, weight.dtype
    assert weight.dim() == 3, weight.shape
    assert weight_scale.shape == (
        weight.shape[0],
        weight.shape[1],
        weight.shape[2] * 2 // group_size,
    ), (
        f"NVFP4 scale shape mismatch: weight={tuple(weight.shape)}, "
        f"scale={tuple(weight_scale.shape)}, group_size={group_size}"
    )

    global_scale = global_scale.to(device=weight.device, dtype=torch.float32)
    if global_scale.dim() == 2 and global_scale.shape[1] == 1:
        global_scale = global_scale[:, 0]

    fp4_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=weight.device,
        dtype=torch.float32,
    )
    packed_out = torch.empty_like(weight)
    scale_out = torch.empty(
        weight_scale.shape, dtype=torch.uint8, device=weight_scale.device
    )
    chunk_n = int(os.environ.get("SGLANG_MEGAMOE_NVFP4_REQUANT_CHUNK_N", "128"))
    chunk_n = max(1, chunk_n)
    n = weight.shape[1]
    half_n = n // 2

    def tensor_scale_for_range(start: int, end: int) -> torch.Tensor:
        if gated_l1 and global_scale.dim() == 2 and global_scale.shape[1] >= 2:
            tensor_scale = torch.empty(
                (weight.shape[0], end - start),
                dtype=torch.float32,
                device=weight.device,
            )
            first_end = min(end, half_n)
            if start < first_end:
                tensor_scale[:, : first_end - start] = global_scale[:, 0].reshape(
                    -1, 1
                )
            if end > half_n:
                offset = max(0, half_n - start)
                tensor_scale[:, offset:] = global_scale[:, 1].reshape(-1, 1)
            return tensor_scale

        tensor_scale = global_scale[:, 0] if global_scale.dim() == 2 else global_scale
        return tensor_scale.reshape(-1, 1)

    for start in range(0, n, chunk_n):
        end = min(start + chunk_n, n)
        packed = weight[:, start:end, :]
        nibbles = torch.empty(
            (*packed.shape[:-1], packed.shape[-1] * 2),
            dtype=torch.uint8,
            device=packed.device,
        )
        nibbles[..., 0::2] = packed & 0x0F
        nibbles[..., 1::2] = packed >> 4

        dequant = fp4_values[(nibbles & 0x07).long()]
        dequant = torch.where((nibbles & 0x08) == 0, dequant, -dequant)
        block_scale = weight_scale[:, start:end, :].to(torch.float32)
        dequant.mul_(block_scale.repeat_interleave(group_size, dim=-1))
        dequant.mul_(tensor_scale_for_range(start, end).unsqueeze(-1))

        requant_packed, requant_scale = quantize_to_nvfp4(
            dequant, group_size=group_size
        )
        packed_out[:, start:end, :].copy_(requant_packed)
        scale_out[:, start:end, :].copy_(requant_scale)

    return packed_out.contiguous(), scale_out.contiguous()


def _build_sm90_nvfp4_mega_moe_weights(
    experts,
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> None:
    from deep_gemm.quantization_nvfp4 import (
        nvfp4_fuse_packed_with_scale_tile_major,
        nvfp4_scale_to_tile_major,
    )

    quant_config = getattr(experts, "quant_config", None)
    group_size = getattr(quant_config, "group_size", 16)
    assert (
        group_size == 16
    ), f"SM90 NVFP4 MegaMoE expects group_size=16, got {group_size}"
    assert hasattr(experts, "w13_weight_scale") and hasattr(
        experts, "w2_weight_scale"
    ), "SM90 NVFP4 MegaMoE expects ModelOpt w*_weight_scale tensors"
    assert hasattr(experts, "w13_weight_scale_2") and hasattr(
        experts, "w2_weight_scale_2"
    ), "SM90 NVFP4 MegaMoE expects ModelOpt w*_weight_scale_2 tensors"

    num_groups, n1, half_k1 = w13.shape
    _, n2, half_k2 = w2.shape
    k1 = half_k1 * 2
    k2 = half_k2 * 2
    expected_w13_scale_shape = (num_groups, n1, k1 // group_size)
    expected_w2_scale_shape = (num_groups, n2, k2 // group_size)
    assert tuple(experts.w13_weight_scale.shape) == expected_w13_scale_shape, (
        f"w13 NVFP4 scale shape mismatch: got {tuple(experts.w13_weight_scale.shape)}, "
        f"expected {expected_w13_scale_shape}"
    )
    assert tuple(experts.w2_weight_scale.shape) == expected_w2_scale_shape, (
        f"w2 NVFP4 scale shape mismatch: got {tuple(experts.w2_weight_scale.shape)}, "
        f"expected {expected_w2_scale_shape}"
    )

    gated_l1 = getattr(
        getattr(experts, "moe_runner_config", None), "is_gated", False
    )
    requantize = os.environ.get("SGLANG_MEGAMOE_NVFP4_REQUANTIZE", "0") == "1"
    if requantize:
        from deep_gemm.mega import transform_nvfp4_weights_for_mega_moe_sm90

        block_n = int(os.environ.get("DG_SM90_NVFP4_BLOCK_N", "128"))
        block_k = 128
        fused_env = os.environ.get("DG_SM90_NVFP4_FUSED_B_SCALE")
        fused_b_scale = None if fused_env is None else fused_env != "0"
        w13_requant, w13_scale_ue4m3 = _requantize_modelopt_nvfp4_for_deepgemm(
            w13,
            experts.w13_weight_scale.data,
            experts.w13_weight_scale_2.data,
            gated_l1=gated_l1,
            group_size=group_size,
        )
        w2_requant, w2_scale_ue4m3 = _requantize_modelopt_nvfp4_for_deepgemm(
            w2,
            experts.w2_weight_scale.data,
            experts.w2_weight_scale_2.data,
            gated_l1=False,
            group_size=group_size,
        )
        if os.environ.get("SGLANG_MEGAMOE_NVFP4_FP8_SHADOW", "0") == "1":
            from deep_gemm.mega import (
                materialize_nvfp4_fp8_shadow_for_mega_moe_sm90,
            )

            experts.mega_l1_weights, experts.mega_l2_weights = (
                materialize_nvfp4_fp8_shadow_for_mega_moe_sm90(
                    (w13_requant, w13_scale_ue4m3),
                    (w2_requant, w2_scale_ue4m3),
                    group_size=group_size,
                )
            )
            experts.mega_l1_global_scales = None
            experts.mega_l2_global_scales = None
            experts._mega_moe_arch = _SM90_FP8_CONFIG.name
            return

        experts.mega_l1_weights, experts.mega_l2_weights = (
            transform_nvfp4_weights_for_mega_moe_sm90(
                (w13_requant, w13_scale_ue4m3),
                (w2_requant, w2_scale_ue4m3),
                block_n=block_n,
                block_k=block_k,
                group_size=group_size,
                fused_b_scale=fused_b_scale,
            )
        )
        experts.mega_l1_global_scales = None
        experts.mega_l2_global_scales = None
        return

    l1_global_scales = _modelopt_nvfp4_global_scale_1d(
        experts.w13_weight_scale_2.data,
        gated_l1=gated_l1,
        device=w13.device,
    )
    l2_global_scales = _modelopt_nvfp4_global_scale_1d(
        experts.w2_weight_scale_2.data,
        gated_l1=False,
        device=w2.device,
    )
    w13_prefold_scale = _modelopt_nvfp4_prefold_scale(
        experts.w13_weight_scale.data, device=w13.device
    )
    w2_prefold_scale = _modelopt_nvfp4_prefold_scale(
        experts.w2_weight_scale.data, device=w2.device
    )
    w13_scale_ue4m3 = _modelopt_nvfp4_scale_to_ue4m3(
        experts.w13_weight_scale.data,
        w13_prefold_scale,
        gated_l1=gated_l1,
        round_to_nearest=True,
    )
    w2_scale_ue4m3 = _modelopt_nvfp4_scale_to_ue4m3(
        experts.w2_weight_scale.data,
        w2_prefold_scale,
        gated_l1=False,
        round_to_nearest=True,
    )
    l1_global_scales = l1_global_scales / w13_prefold_scale
    l2_global_scales = l2_global_scales / w2_prefold_scale

    block_n = int(os.environ.get("DG_SM90_NVFP4_BLOCK_N", "128"))
    block_k = 128
    fused_env = os.environ.get("DG_SM90_NVFP4_FUSED_B_SCALE")
    fused_b_scale = True if fused_env is None else fused_env != "0"

    _modelopt_nvfp4_to_deepgemm_packed_inplace(w13)
    _modelopt_nvfp4_to_deepgemm_packed_inplace(w2)

    w13_interleaved = _interleave_l1_tensor_inplace(w13)
    w13_scale_interleaved = _interleave_l1_tensor_inplace(w13_scale_ue4m3)
    del w13_scale_ue4m3

    w13_scale_tile = nvfp4_scale_to_tile_major(
        w13_scale_interleaved,
        block_n=block_n,
        block_k=block_k,
        group_size=group_size,
    )
    del w13_scale_interleaved
    w2_scale_tile = nvfp4_scale_to_tile_major(
        w2_scale_ue4m3,
        block_n=block_n,
        block_k=block_k,
        group_size=group_size,
    )

    if fused_b_scale:
        w13_packed = nvfp4_fuse_packed_with_scale_tile_major(
            w13_interleaved,
            w13_scale_tile,
            block_k=block_k,
        )
        w2_packed = nvfp4_fuse_packed_with_scale_tile_major(
            w2.contiguous(),
            w2_scale_tile,
            block_k=block_k,
        )
    else:
        w13_packed = w13_interleaved
        w2_packed = w2 if w2.is_contiguous() else w2.contiguous()

    experts.mega_l1_weights = (w13_packed, w13_scale_tile)
    experts.mega_l2_weights = (w2_packed, w2_scale_tile)
    experts.mega_l1_global_scales = l1_global_scales
    experts.mega_l2_global_scales = l2_global_scales


def build_mega_moe_experts_weights(experts) -> bool:
    from deep_gemm import (
        transform_sf_into_required_layout,
        transform_weights_for_mega_moe,
    )
    from deep_gemm.mega import _interleave_l1_weights, _transpose_sf_for_utccp

    if getattr(experts, "_mega_moe_weights_built", False):
        return _get_built_mega_moe_arch_config(experts) is not None

    w13 = experts.w13_weight.data
    w2 = experts.w2_weight.data
    config = _select_mega_moe_arch_config(w13, w2)
    if config is None:
        return False

    if config.name == _SM90_NVFP4_CONFIG.name:
        _build_sm90_nvfp4_mega_moe_weights(experts, w13, w2)
        if getattr(experts, "_mega_moe_arch", None) is None:
            experts._mega_moe_arch = config.name
        experts._mega_moe_weights_built = True
        return True

    w13_sf_fp32 = experts.w13_weight_scale_inv.data
    w2_sf_fp32 = experts.w2_weight_scale_inv.data

    num_groups, n1, half_k1 = w13.shape
    _, n2, half_k2 = w2.shape

    # FP4 weights are packed as int8 and have last dim K//2; FP8 weights use K.
    k_factor = 2 if config.fp4_weight_packed else 1
    k1 = half_k1 * k_factor
    k2 = half_k2 * k_factor

    scale_group_mn, scale_group_k = config.scale_recipe
    assert k1 % scale_group_k == 0 and k2 % scale_group_k == 0, (
        f"invalid mega-moe K/group_size: k1={k1}, k2={k2}, "
        f"group_k={scale_group_k}"
    )
    expected_n_groups_1 = (n1 + scale_group_mn - 1) // scale_group_mn
    expected_n_groups_2 = (n2 + scale_group_mn - 1) // scale_group_mn
    expected_k_groups_1 = k1 // scale_group_k
    expected_k_groups_2 = k2 // scale_group_k
    assert w13_sf_fp32.shape[1] == expected_n_groups_1, (
        f"w13 scale N groups mismatch: got {w13_sf_fp32.shape[1]}, "
        f"expected {expected_n_groups_1} (n1={n1}, group_mn={scale_group_mn})"
    )
    assert w2_sf_fp32.shape[1] == expected_n_groups_2, (
        f"w2 scale N groups mismatch: got {w2_sf_fp32.shape[1]}, "
        f"expected {expected_n_groups_2} (n2={n2}, group_mn={scale_group_mn})"
    )
    assert w13_sf_fp32.shape[2] == expected_k_groups_1, (
        f"w13 scale K groups mismatch: got {w13_sf_fp32.shape[2]}, "
        f"expected {expected_k_groups_1} (k1={k1}, group_k={scale_group_k})"
    )
    assert w2_sf_fp32.shape[2] == expected_k_groups_2, (
        f"w2 scale K groups mismatch: got {w2_sf_fp32.shape[2]}, "
        f"expected {expected_k_groups_2} (k2={k2}, group_k={scale_group_k})"
    )

    fix_mega_moe_memory = envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get()
    if fix_mega_moe_memory and config.name == _SM90_FP8_CONFIG.name:
        # SM90 shares both fp8 weights and block-(128, 128) FP32 scales with the
        # DeepEP grouped-GEMM path. SM90 has no UTCCP scale transpose, and its
        # scale tensors stay in checkpoint layout.
        w13_interleaved = _interleave_l1_weight_only(w13)
        experts.w13_weight.data = w13_interleaved
        experts.mega_l1_weights = (
            experts.w13_weight.data,
            experts.w13_weight_scale_inv.data,
        )
        experts.mega_l2_weights = (
            experts.w2_weight.data,
            experts.w2_weight_scale_inv.data,
        )
    else:
        w13_sf = transform_sf_into_required_layout(
            w13_sf_fp32,
            mn=n1,
            k=k1,
            recipe=config.scale_recipe,
            num_groups=num_groups,
            disable_ue8m0_cast=config.uses_raw_fp32_scales,
        )
        w2_sf = transform_sf_into_required_layout(
            w2_sf_fp32,
            mn=n2,
            k=k2,
            recipe=config.scale_recipe,
            num_groups=num_groups,
            disable_ue8m0_cast=config.uses_raw_fp32_scales,
        )

        if fix_mega_moe_memory and config.name == _SM100_FP8_FP4_CONFIG.name:
            # Build the interleaved L1 weight + scale once; share the weight buffer
            # between `w13_weight.data` (normal deep-ep path) and `mega_l1_weights[0]`
            # (mega moe path). Mega moe additionally needs a UTCCP-transposed scale;
            # the deep-ep path consumes the non-transposed interleaved scale and a
            # swizzle-aware activation kernel. L2 weight is untouched by the mega
            # transform, so the existing `w2_weight.data` is shared directly.
            w13_interleaved, w13_sf_interleaved = _interleave_l1_weights(
                (w13, w13_sf)
            )
            w13_sf_utccp = _transpose_sf_for_utccp(w13_sf_interleaved)
            w2_sf_utccp = _transpose_sf_for_utccp(w2_sf)

            experts.w13_weight.data = w13_interleaved
            experts.w13_weight_scale_inv.data = w13_sf_interleaved
            experts.w2_weight_scale_inv.data = w2_sf
            experts.w13_weight_scale_inv.format_ue8m0 = True
            experts.w2_weight_scale_inv.format_ue8m0 = True

            experts.mega_l1_weights = (experts.w13_weight.data, w13_sf_utccp)
            experts.mega_l2_weights = (experts.w2_weight.data, w2_sf_utccp)
        else:
            transform_fn = transform_weights_for_mega_moe
            if config.name == _SM90_FP8_CONFIG.name:
                from deep_gemm import transform_weights_for_mega_moe_sm90

                transform_fn = transform_weights_for_mega_moe_sm90

            l1_pair, l2_pair = transform_fn((w13, w13_sf), (w2, w2_sf))
            experts.mega_l1_weights = l1_pair
            experts.mega_l2_weights = l2_pair

    experts._mega_moe_arch = config.name
    experts._mega_moe_weights_built = True
    return True
