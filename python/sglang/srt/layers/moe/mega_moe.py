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
_MEGA_MOE_ARCH_CONFIGS = {
    config.name: config
    for config in (_SM90_FP8_CONFIG, _SM100_FP8_FP4_CONFIG)
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
        and _get_disaggregation_mode_or_none() == "prefill"
    ):
        return config.pre_dispatch_group_size
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
    dispatch_hidden_states = hidden_states
    pad_dispatch_inputs = dispatch_num_tokens > num_tokens
    if pad_dispatch_inputs:
        dispatch_hidden_states = hidden_states.new_zeros(
            (dispatch_num_tokens, hidden_size)
        )
        if num_tokens > 0:
            dispatch_hidden_states[:num_tokens].copy_(hidden_states)

    if num_tokens > 0:
        topk_ids_in = topk_ids.to(torch.int32)
        topk_weights_in = topk_weights.to(torch.float32)
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
        if config.fold_routed_scaling_in_pre_dispatch:
            num_experts_per_rank = num_experts // moe_ep_group.world_size
            dummy_expert_base = moe_ep_group.rank_in_group * num_experts_per_rank
            dummy_expert_ids = torch.arange(
                dummy_expert_base,
                dummy_expert_base + top_k,
                device=hidden_states.device,
                dtype=torch.int32,
            )
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
    getattr(deep_gemm, config.deep_gemm_entry)(
        y,
        moe.experts.mega_l1_weights,
        moe.experts.mega_l2_weights,
        buf,
        recipe=config.run_recipe,
        activation="swiglu",
        activation_clamp=swiglu_limit,
        fast_math=True,
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


def build_mega_moe_experts_weights(experts) -> bool:
    from deep_gemm import (
        transform_sf_into_required_layout,
        transform_weights_for_mega_moe,
    )
    from deep_gemm.mega import _interleave_l1_weights, _transpose_sf_for_utccp

    if getattr(experts, "_mega_moe_weights_built", False):
        return _get_built_mega_moe_arch_config(experts) is not None

    w13 = experts.w13_weight.data
    w13_sf_fp32 = experts.w13_weight_scale_inv.data
    w2 = experts.w2_weight.data
    w2_sf_fp32 = experts.w2_weight_scale_inv.data
    config = _select_mega_moe_arch_config(w13, w2)
    if config is None:
        return False

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
