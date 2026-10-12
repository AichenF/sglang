"""goal11-dense: fused TP8 decode path for MiMo-V2's dense-MLP decoder layer (layer 0), M <= 32 tokens.

Unfused (stage-boundary) path after the attention's o_proj partial: custom all-reduce (2-shot) -> flashinfer
fused_add_rmsnorm (post_attention_layernorm) -> per-token-group-128 fp8 quant -> gate_up (DeepGEMM) -> SiluAndMul ->
group quant -> down (DeepGEMM) -> [next layer's prepare] custom all-reduce -> fused_add_rmsnorm (next input_layernorm)
-> group quant -> next qkv_proj. Here:
  * all-reduce + residual add + RMSNorm + group-128 quant = ONE kernel (DFlashTPState.arnq_g128: reduce-scatter +
    all-gather over CUDA-IPC Lamport pushes; bit-exact with custom AR + flashinfer + SGLang group quant), twice;
  * SiluAndMul + group quant = ONE kernel (silu_mul_group_quant, bit-exact);
  * the second one produces the NEXT layer's normalized fp8 qkv_proj input and residual and hands them over with the
    fused MoE kernel's norm-next protocol (NormNextHandoff / take_norm_next_handoff), so layer 1 skips its prepare.

Knob: SGLANG_MIMO_DENSE_TP (default 1).
"""

import logging
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

H = 6144
TP = 8
MAX_M = 32


def _plain_rmsnorm(norm) -> Optional[str]:
    from sglang.srt.layers.layernorm import RMSNorm

    if (
        type(norm) is not RMSNorm
        or getattr(norm, "variance_size_override", None) is not None
        or getattr(norm, "cast_x_before_out_mul", False)
        or getattr(norm, "fp32_residual", False)
        or getattr(norm, "override_orig_dtype", None) is not None
    ):
        return f"{type(norm).__name__} is not a plain RMSNorm"
    if norm.weight.dtype != torch.bfloat16 or norm.weight.numel() != H or not norm.weight.data.is_contiguous():
        return f"norm weight {norm.weight.dtype} {norm.weight.numel()}"
    try:
        target = norm.dispatch_forward()
    except Exception as e:  # pragma: no cover
        return f"dispatch_forward failed: {e}"
    if target != norm.forward_cuda:
        return f"norm dispatches to {getattr(target, '__name__', target)} (need forward_cuda -> flashinfer)"
    return None


def _deepgemm_block_linear(linear, n: int, k: int) -> Optional[str]:
    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    from sglang.srt.layers.quantization.fp8_utils import deepgemm_w8a8_block_fp8_linear_with_fallback

    qm = getattr(linear, "quant_method", None)
    if (
        not isinstance(qm, Fp8LinearMethod)
        or not qm.block_quant
        or getattr(qm, "use_mxfp8", False)
        or getattr(qm, "block_fp8_as_mxfp8", False)
        or qm.use_marlin
    ):
        return f"{type(qm).__name__} (need block-quant Fp8LinearMethod)"
    if list(qm.weight_block_size) != [128, 128] or qm.w8a8_block_fp8_linear is not deepgemm_w8a8_block_fp8_linear_with_fallback:
        return "need [128, 128] blocks + the DeepGEMM linear"
    if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0 or not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
        return "DeepGEMM disabled or UE8M0 scales"
    w = getattr(linear, "weight", None)
    if w is None or w.dtype != torch.float8_e4m3fn or tuple(w.shape) != (n, k):
        return f"weight {None if w is None else (w.dtype, tuple(w.shape))} (need {(n, k)})"
    if getattr(linear, "bias", None) is not None or getattr(linear, "skip_bias_add", False):
        return "bias"
    return None


class MiMoDenseTP:
    """Fused decode path of one dense MiMoV2DecoderLayer. Construct through ``try_create`` (collective on the first call)."""

    def __init__(self, layer, state):
        from sglang.srt.layers.layer_boundary import PLAIN_ADD
        from sglang.srt.layers.moe.mimo_fused_moe.layer import NormNextHandoff

        self.layer = layer
        self.state = state
        self.nxt = layer._fmoe_next_layer
        self.inter = layer.mlp.down_proj.weight.shape[1]
        self.ffn_plan = layer.ffn_boundary.plan
        self.PLAIN_ADD = PLAIN_ADD
        self.Handoff = NormNextHandoff
        dev = layer.post_attention_layernorm.weight.device
        # the MLP GEMMs follow the NVLink-bound fused TP kernels (which trigger their dependents right after their own
        # wait): the skinny block-scale GEMM streams its whole smem ring of weights before griddepcontrol.wait
        self.skinny = None
        if envs.SGLANG_MIMO_DENSE_TP_SKINNY.get():
            from sglang.kernels.ops.gemm.skinny_gemm import skinny_fp8_block_gemm, skinny_supported

            mlp = layer.mlp
            if skinny_supported(MAX_M, mlp.gate_up_proj.weight.shape[0], H) and skinny_supported(
                MAX_M, H, mlp.down_proj.weight.shape[1]
            ):
                self.skinny = skinny_fp8_block_gemm
        # the layer's "FFN output" handed to the residual stream is all zeros: the residual stream already holds
        # residual + attention + MLP (written in place by the second arnq); a consumer that does not take the norm-next
        # handoff adds zeros to it and still gets the right layer output.
        self.zeros = torch.zeros(MAX_M, H, dtype=torch.bfloat16, device=dev)
        # goal11-post: skinny preload / depth of the gate_up and down GEMMs (env "gu_preload,gu_depth,dn_preload,dn_depth";
        # default 6,6,6,6). The preload TMAs land on the SMs of the NVLink-bound arnq that precedes them.
        import os

        pd = [int(v) for v in os.environ.get("SGLANG_MIMO_DENSE_TP_PRELOAD", "6,6,6,6").split(",")]
        # + L2 prefetch of the rest of each CTA's weight slice before the PDL wait ("gu,dn" units, -1 = all; 0 = off)
        l2 = [int(v) for v in os.environ.get("SGLANG_MIMO_DENSE_TP_L2PRE", "0,0").split(",")]
        # + extra dynamic smem (bytes) so a GEMM CTA cannot share an SM with an arnq CTA ("gu,dn"; 0 = off)
        sp = [int(v) for v in os.environ.get("SGLANG_MIMO_DENSE_TP_SMEM_PAD", "0,0").split(",")]
        self.gu_pd, self.dn_pd = (pd[0], pd[1], l2[0], sp[0]), (pd[2], pd[3], l2[1], sp[1])
        # the gate_up GEMM L2-prefetches the down_proj weights from its tail (SGLANG_MIMO_DENSE_TP_NEXT_PF, default 0)
        self.next_pf_w = layer.mlp.down_proj.weight if os.environ.get("SGLANG_MIMO_DENSE_TP_NEXT_PF", "0") == "1" else None

    def _stamp_bufs(self):
        """goal11-post diagnostics (SGLANG_MIMO_DENSE_TP_STAMPS=1): globaltimer stamps of the two arnq calls (separate
        buffers, captured into the graph); dump_stamps() prints them."""
        import os

        if os.environ.get("SGLANG_MIMO_DENSE_TP_STAMPS", "0") != "1":
            return None
        if getattr(self, "_sbufs", None) is None:
            dev = self.zeros.device
            self._sbufs = (torch.zeros(32 * 16, dtype=torch.int64, device=dev), torch.zeros(32 * 16, dtype=torch.int64, device=dev))
            _STAMP_REG.append(self)
        return self._sbufs

    @staticmethod
    def check(layer) -> Optional[str]:
        from sglang.srt.layers.layer_boundary import BatchVariant, SumGroup
        from sglang.srt.runtime_context import get_parallel

        if not envs.SGLANG_MIMO_DENSE_TP.get():
            return "disabled by SGLANG_MIMO_DENSE_TP=0"
        par = get_parallel()
        if par.tp_size != TP or par.moe_ep_size != 1 or par.attn_dp_size != 1 or par.attn_tp_size != par.tp_size:
            return f"tp={par.tp_size} ep={par.moe_ep_size} attn_dp={par.attn_dp_size} (need plain TP{TP})"
        if layer.is_layer_sparse or int(layer.config.hidden_size) != H:
            return "not a dense H6144 layer"
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            return "not sm90"
        mlp = layer.mlp
        if getattr(mlp, "is_replicated", False) or type(getattr(mlp, "act_fn", None)).__name__ != "SiluAndMul":
            return "MLP not TP-sharded SiluAndMul"
        inter = mlp.down_proj.weight.shape[1]
        if inter not in (1024, 2048):
            return f"per-rank intermediate {inter}"
        why = _deepgemm_block_linear(mlp.gate_up_proj, 2 * inter, H) or _deepgemm_block_linear(mlp.down_proj, H, inter)
        if why:
            return f"MLP linear: {why}"
        if getattr(mlp.down_proj, "reduce_results", False):
            return "down_proj reduces its own output"
        why = _plain_rmsnorm(layer.post_attention_layernorm)
        if why:
            return f"post_attention_layernorm: {why}"
        nxt = getattr(layer, "_fmoe_next_layer", None)
        if nxt is None:
            return "no next layer"
        why = _plain_rmsnorm(nxt.input_layernorm)
        if why:
            return f"next input_layernorm: {why}"
        qkv = nxt.self_attn.qkv_proj
        why = _deepgemm_block_linear(qkv, qkv.weight.shape[0], H)
        if why or qkv.weight.shape[0] % 64 or getattr(qkv, "gather_output", False):
            return f"next qkv_proj: {why}"
        eps = float(layer.config.layernorm_epsilon)
        if float(layer.post_attention_layernorm.variance_epsilon) != eps or float(nxt.input_layernorm.variance_epsilon) != eps:
            return "eps mismatch"
        attn_path = layer.attn_boundary.plan.paths.get(BatchVariant.ORDINARY)
        if (
            attn_path is None
            or attn_path.output.group is not SumGroup.ATTN_TP
            or not attn_path.output.always_partial
            or attn_path.output.transform is not None
            or not attn_path.output.update.is_plain_add
        ):
            return "attention output is not a declared attention-TP partial sum with a plain add"
        ffn = layer.ffn_boundary
        decl = ffn.declaration
        if (
            not decl.update.is_plain_add
            or decl.output_transform is not None
            or ffn.plan.fusions is not None
            or ffn.norm is not layer.post_attention_layernorm
        ):
            return "FFN boundary is not a plain residual add + post_attention_layernorm"
        return None

    @classmethod
    def try_create(cls, layer) -> Optional["MiMoDenseTP"]:
        """Collective over the TP group (the IPC state): every rank calls it at the same point."""
        from sglang.kernels.ops.communication.dflash_tp_fused import get_state
        from sglang.srt.distributed import get_tp_group

        try:
            why = cls.check(layer)
        except Exception as e:  # identical on every rank
            why = f"check failed: {e}"
        group = get_tp_group()
        if why is None:
            if torch.cuda.is_current_stream_capturing() and get_state.__globals__["_STATE"] is None:
                why = "first use under CUDA-graph capture"
        if why is None:
            state = get_state(group)
            if state is None:
                why = "IPC state unavailable"
        if why is not None:
            if group.rank_in_group == 0:
                logger.info(f"MiMo dense TP fused path disabled for layer {layer.layer_id}: {why}")
            return None
        if group.rank_in_group == 0:
            logger.info(
                f"MiMo dense TP fused path enabled for layer {layer.layer_id} (AR+add+RMSNorm+group quant x2, "
                f"SiLU-mul+quant, norm-next handoff to layer {layer._fmoe_next_layer.layer_id})"
            )
        return cls(layer, state)

    def applicable(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor], forward_batch) -> bool:
        from sglang.srt.layers.layer_boundary import BatchVariant

        if residual is None or not isinstance(hidden_states, torch.Tensor):
            return False
        m = hidden_states.shape[0]
        if not (1 <= m <= MAX_M and m % 4 == 0):
            return False
        if hidden_states.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
            return False
        if tuple(hidden_states.shape) != (m, H) or tuple(residual.shape) != (m, H):
            return False
        if not residual.is_contiguous():
            return False
        mode = forward_batch.forward_mode
        if not (mode.is_decode() or mode.is_target_verify()):
            return False
        if self.ffn_plan.variant_for(forward_batch) is not BatchVariant.ORDINARY:
            return False
        if getattr(forward_batch, "can_run_tbo", False) or torch.compiler.is_compiling():
            return False
        return True

    def _gemm(self, linear, xq: torch.Tensor, xs_t: torch.Tensor, pd=(6, 6, 0, 0), next_w=None) -> torch.Tensor:
        """Block-scale fp8 GEMM of a pre-quantized [M, K] input with column-major group scales xs_t [K/128, M]."""
        if self.skinny is not None:
            return self.skinny(
                xq, xs_t.t(), linear.weight, linear.weight_scale_inv, preload=pd[0], depth=pd[1], l2pre=pd[2],
                smem_pad=pd[3], next_w=next_w,
            )
        return linear.quant_method.apply(linear, (xq, xs_t.t()))

    def forward(self, partial: torch.Tensor, residual: torch.Tensor, forward_batch) -> torch.Tensor:
        from sglang.kernels.ops.communication.dflash_tp_fused import silu_mul_group_quant
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

        layer, st = self.layer, self.state
        mlp = layer.mlp
        M = partial.shape[0]
        dev = partial.device
        if not partial.is_contiguous():
            partial = partial.contiguous()
        # (1) attention all-reduce + residual add (in place) + post_attention_layernorm + group quant
        ln = layer.post_attention_layernorm
        xq = torch.empty(M, H, dtype=torch.float8_e4m3fn, device=dev)
        xs = torch.empty(H // 128, M, dtype=torch.float32, device=dev)
        sb = self._stamp_bufs()
        if sb is not None:
            old_st, st.stamps = st.stamps, sb[0]
        st.arnq_g128(partial, residual, ln.weight, ln.variance_epsilon, xq, xs)
        if sb is not None:
            st.stamps = old_st
        gu = self._gemm(mlp.gate_up_proj, xq, xs, self.gu_pd, next_w=self.next_pf_w)
        # (2) SiluAndMul + group quant
        I = self.inter
        aq = torch.empty(M, I, dtype=torch.float8_e4m3fn, device=dev)
        asc = torch.empty(I // 128, M, dtype=torch.float32, device=dev)
        silu_mul_group_quant(gu, aq, asc)
        d_part = self._gemm(mlp.down_proj, aq, asc, self.dn_pd)
        # (3) MLP all-reduce + residual add (in place) + NEXT layer's input_layernorm + group quant
        nln = self.nxt.input_layernorm
        xq2 = torch.empty(M, H, dtype=torch.float8_e4m3fn, device=dev)
        xs2 = torch.empty(H // 128, M, dtype=torch.float32, device=dev)
        if sb is not None:
            old_st, st.stamps = st.stamps, sb[1]
        st.arnq_g128(d_part, residual, nln.weight, nln.variance_epsilon, xq2, xs2)
        if sb is not None:
            st.stamps = old_st
        stream = residual_batch.stream_of(forward_batch)
        stream.write(residual)
        out = stream.record(self.zeros[:M], self.PLAIN_ADD)
        out._fmoe_norm_next = self.Handoff(self.nxt.layer_id, residual, xq2, xs2.t())
        return out


_EMBED_NQ_CACHE = {}
_STAMP_REG = []
_STAMP_N = [0]


def dump_stamps(M: int) -> None:
    """Print the arnq stamps (us, relative to the first token CTA's stamp 0 of arnq 1) every 64th call (host sync)."""
    if not _STAMP_REG:
        return
    _STAMP_N[0] += 1
    if _STAMP_N[0] % 64 != 0:
        return
    for obj in _STAMP_REG:
        a, b = (x.view(32, 16)[:M].cpu() for x in obj._sbufs)
        t0 = int(a[:, 0].min())
        def ph(s, k):
            return ((s[:, k] - t0).float() / 1000.0)
        msg = []
        for name, s in (("arnq1", a), ("arnq2", b)):
            cols = {k: ph(s, k) for k in (9, 0, 1, 8, 2, 3, 5)}
            msg.append(name + " " + " ".join(f"s{k} {float(v.median()):.2f}[{float(v.min()):.2f},{float(v.max()):.2f}]" for k, v in cols.items()))
        logger.info("DENSE_TP_STAMPS layer0 M=%d: %s", M, " | ".join(msg))


def embed_norm_next(model, input_ids: torch.Tensor, forward_batch) -> Optional[torch.Tensor]:
    """Step start of MiMoV2Model at M <= 32 on TP8: vocab-parallel embedding (owner push over CUDA IPC) + layer 0's
    input_layernorm + group-128 fp8 quant in ONE kernel (DFlashTPState.embed_nq, bit-exact with the zero-padded
    gather + all-reduce, sgl_kernel rmsnorm and SGLang's group quant). Returns the embedding rows carrying a norm-next
    handoff for layer 0 (which then skips its prepare and feeds qkv_proj the fp8 pair), or None."""
    key = id(model)
    ent = _EMBED_NQ_CACHE.get(key)
    if ent is None:
        ent = False
        why = None
        try:
            from sglang.srt.layers.moe.mimo_fused_moe.layer import NormNextHandoff

            layer0 = model.layers[model.start_layer]
            emb = model.embed_tokens
            if not envs.SGLANG_MIMO_EMBED_NQ.get():
                why = "disabled by SGLANG_MIMO_EMBED_NQ=0"
            elif model.start_layer != 0 or not hasattr(emb, "_fused_embed_state"):
                why = "not the first PP stage / no fused embedding"
            else:
                why = _plain_rmsnorm(layer0.input_layernorm)
                if why is None:
                    qkv = layer0.self_attn.qkv_proj
                    why = _deepgemm_block_linear(qkv, qkv.weight.shape[0], H)
                    if why is None and (qkv.weight.shape[0] % 64 or getattr(qkv, "gather_output", False)):
                        why = "qkv shape"
            if why is None:
                ent = (layer0, emb, NormNextHandoff)
        except Exception as e:
            why = f"check failed: {e}"
        _EMBED_NQ_CACHE[key] = ent
        if why is not None and torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info(f"MiMo fused embedding + layer-0 input norm/quant disabled: {why}")
    if ent is False:
        return None
    layer0, emb, Handoff = ent
    M = input_ids.shape[0]
    if M % 4 != 0:
        return None
    mode = forward_batch.forward_mode
    if not (mode.is_decode() or mode.is_target_verify()) or getattr(forward_batch, "can_run_tbo", False):
        return None
    st = emb._fused_embed_state(input_ids)
    if st is None:
        return None
    dev = input_ids.device
    out = torch.empty(M, H, dtype=emb.weight.dtype, device=dev)
    xq = torch.empty(M, H, dtype=torch.float8_e4m3fn, device=dev)
    xs = torch.empty(H // 128, M, dtype=torch.float32, device=dev)
    ln = layer0.input_layernorm
    st.embed_nq(input_ids, emb.weight, out, emb._fused_embed_ranges, ln.weight, ln.variance_epsilon, xq, xs)
    out._fmoe_norm_next = Handoff(layer0.layer_id, out, xq, xs.t())
    return out
