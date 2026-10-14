"""Fused TP8 decode path for the DFlash draft (hidden 6144, fp8 per-channel draft weights, M <= 32).

Per draft layer the unfused path launches ~20 kernels: RMSNorm, per-token quant, qkv GEMM, q/k norm, RoPE, value
scale, KV store, attention, per-token quant, o_proj GEMM, custom all-reduce (2-shot), fused add+RMSNorm, per-token
quant, gate_up GEMM, act_and_mul, per-token quant, down GEMM, custom all-reduce, fused add+RMSNorm. Here the two
"projection -> TP all-reduce -> residual add -> RMSNorm -> per-token FP8 quant" chains are ONE kernel each
(``DFlashTPState.arnq``: reduce-scatter + all-gather over CUDA-IPC Lamport pushes, the RMS / amax statistics ride
along), act_and_mul + quant is one kernel, and the GEMMs take the pre-quantized activations directly. The
attention part (DFlashAttention.attend) is unchanged. Draft-only numerics (the target verifies every token).

Knob: SGLANG_DFLASH_TP_FUSED (default 1).
"""

import logging
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

H = 6144
TP = 8
MAX_M = 32
INTER = 2048  # per-rank intermediate size the silu kernel is built for

_state = None
_state_failed = False


def _channelwise_fp8(linear) -> bool:
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

    qm = getattr(linear, "quant_method", None)
    if not isinstance(qm, Fp8LinearMethod) or qm.block_quant:
        return False
    w = getattr(linear, "weight", None)
    s = getattr(linear, "weight_scale", None)
    if w is None or s is None or w.dtype != torch.float8_e4m3fn or w.dim() != 2:
        return False
    # Fp8LinearMethod keeps the [K, N] (column-major) view of the per-channel quantized weight
    return s.numel() == w.shape[1] and w.t().is_contiguous()


class DFlashTPFast:
    def __init__(self, model, state):
        self.model = model
        self.state = state
        self.layers = list(model.layers)
        from sgl_kernel import fp8_scaled_mm

        self.mm = fp8_scaled_mm
        self.skinny = None
        # preload the whole smem ring (6 units) before griddepcontrol.wait: the preceding fused TP kernel is
        # NVLink-bound and triggers its dependents right after its own wait
        self.preload = int(envs.SGLANG_DFLASH_TP_SKINNY_PRELOAD.get())
        self.depth = int(envs.SGLANG_DFLASH_TP_SKINNY_DEPTH.get())
        if envs.SGLANG_DFLASH_TP_SKINNY.get():
            from sglang.kernels.ops.gemm.skinny_gemm import skinny_fp8_pt_gemm

            self.skinny = skinny_fp8_pt_gemm
        # the 5 layers at M = 32 as one persistent kernel (built at the first eager M = 32 forward)
        self.mega = None
        self.mega_state = "off" if not envs.SGLANG_DFLASH_MEGA.get() else "pending"

    @staticmethod
    def try_create(model) -> Optional["DFlashTPFast"]:
        global _state, _state_failed
        if not envs.SGLANG_DFLASH_TP_FUSED.get() or _state_failed:
            return None
        try:
            from sglang.srt.distributed import get_tp_group
            from sglang.srt.layers.layernorm import RMSNorm
            from sglang.srt.models.dflash import DFlashAttention, DFlashDecoderLayer

            group = get_tp_group()
            if group.world_size != TP or int(model.config.hidden_size) != H:
                return None
            if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
                return None
            for layer in model.layers:
                if type(layer) is not DFlashDecoderLayer or type(layer.self_attn) is not DFlashAttention:
                    return None
                if layer.attention_conv is not None or layer.mlp_conv is not None:
                    return None
                if type(layer.input_layernorm) is not RMSNorm or type(layer.post_attention_layernorm) is not RMSNorm:
                    return None
                a, m = layer.self_attn, layer.mlp
                if not (_channelwise_fp8(a.qkv_proj) and _channelwise_fp8(a.o_proj)):
                    return None
                if not (_channelwise_fp8(m.gate_up_proj) and _channelwise_fp8(m.down_proj)):
                    return None
                if m.gate_up_proj.weight.shape[1] != 2 * INTER or m.down_proj.weight.shape[0] != INTER:
                    return None
                if getattr(a.o_proj, "bias", None) is not None or getattr(m.down_proj, "bias", None) is not None:
                    return None
            if type(model.norm) is not RMSNorm:
                return None
            from sglang.kernels.ops.communication.dflash_tp_fused import get_state

            if torch.cuda.is_current_stream_capturing() and get_state.__globals__["_STATE"] is None:
                _state_failed = True
                logger.warning("DFLASH TP fused path: first use under CUDA-graph capture, disabled.")
                return None
            _state = get_state(group)
            if _state is None:
                _state_failed = True
                return None
            if group.rank_in_group == 0:
                logger.info("DFLASH draft TP fused path enabled (arnq reduce-scatter/all-gather + norm + fp8 quant).")
            return DFlashTPFast(model, _state)
        except Exception as e:  # never fatal; identical on every rank (same checks / same failure)
            _state_failed = True
            logger.warning(f"DFLASH TP fused path disabled: {e}")
            return None

    @staticmethod
    def applicable(hidden_states: torch.Tensor) -> bool:
        return (
            hidden_states.dim() == 2
            and 1 <= hidden_states.shape[0] <= MAX_M
            and hidden_states.shape[1] == H
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
        )

    def _attend(self, attn, qkv, positions, forward_batch):
        """DFlashAttention.attend with the q/k RMSNorm + RoPE + value scale + KV-cache store as ONE kernel
        (kernels/ops/speculative/dflash_qknr_kv.py) also when the draft KV lives in an SWA pool;
        falls back to attend() when the layout is not the plain bf16 one."""
        args, why = self._qknr_args(attn, qkv, positions, forward_batch)
        if not getattr(self, "_qknr_logged", False) and self.state.rank == 0:
            self._qknr_logged = True
            logger.info(
                "DFLASH TP fused path: fused q/k-norm + RoPE + KV store "
                + ("ON" if args is not None else f"OFF (fallback: {why})")
            )
        if args is None:
            return attn.attend(qkv, positions, qkv, forward_batch)
        from sglang.kernels.ops.speculative.dflash_qknr_kv import dflash_qk_norm_rope_kv_store

        loc, k_cache, v_cache = args
        dflash_qk_norm_rope_kv_store(
            qkv,
            positions,
            attn.q_norm.weight,
            attn.k_norm.weight,
            attn.rotary_emb.cos_sin_cache,
            loc,
            k_cache,
            v_cache,
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            int(attn.rotary_emb.rotary_dim),
            attn.q_norm.variance_epsilon,
            attn.v_scale,
        )
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        if attn.attention_sink_bias is None:
            out = attn.attn(q, k, v, forward_batch, save_kv_cache=False)
        else:
            out = attn.attn(q, k, v, forward_batch, save_kv_cache=False, sinks=attn.attention_sink_bias)
        return attn.apply_attention_output(out, qkv)

    def _qknr_args(self, attn, qkv, positions, forward_batch):
        """-> ((loc, k_cache, v_cache), "") or (None, reason)."""
        if not envs.SGLANG_DFLASH_FUSED_QKNR_KV.get():
            return None, "knob off"
        try:
            from sglang.kernels.ops.speculative.dflash_qknr_kv import can_use_dflash_qknr_kv
            from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
            from sglang.srt.layers.layernorm import RMSNorm

            rope = attn.rotary_emb
            if type(attn.q_norm) is not RMSNorm or type(attn.k_norm) is not RMSNorm:
                return None, "q/k norm type"
            if attn.q_norm.variance_epsilon != attn.k_norm.variance_epsilon:
                return None, "q/k eps"
            if not getattr(rope, "is_neox_style", False) or getattr(rope, "cos_sin_cache", None) is None:
                return None, f"rope {type(rope).__name__} neox={getattr(rope, 'is_neox_style', None)}"
            from sglang.srt.model_executor.forward_context import get_attn_backend, get_token_to_kv_pool

            backend = get_attn_backend()
            if not isinstance(backend, FlashAttentionBackend):
                return None, f"backend {type(backend).__name__}"
            if backend.use_mla or getattr(backend, "fa_skip_kv_cache", False) or getattr(backend, "kv_cache_is_mxfp8", False):
                return None, "backend mode (mla / skip kv / mxfp8)"
            pool = get_token_to_kv_pool()
            layer_id = attn.attn.layer_id
            if getattr(backend, "use_sliding_window_kv_pool", False):
                md = backend.forward_metadata
                loc = getattr(md, "swa_out_cache_loc", None) if md is not None else None
                if loc is None:
                    return None, "swa pool without swa_out_cache_loc"
                mapping = getattr(pool, "layers_mapping", None)
                if mapping is None or not mapping[layer_id][1]:
                    return None, f"swa pool layer {layer_id} not swa ({type(pool).__name__})"
            else:
                loc = forward_batch.out_cache_loc
            k_cache = pool.get_key_buffer(layer_id)
            v_cache = pool.get_value_buffer(layer_id)
            if getattr(pool, "is_quantized_kv_cache", False) or k_cache.dtype != qkv.dtype:
                return None, f"kv dtype {k_cache.dtype}"
            if not can_use_dflash_qknr_kv(
                qkv, positions, rope.cos_sin_cache, loc, k_cache, v_cache,
                attn.num_heads, attn.num_kv_heads, attn.head_dim, int(rope.rotary_dim),
            ):
                return None, (
                    f"shape check: qkv {tuple(qkv.shape)}/{qkv.stride()} pos {tuple(positions.shape)} cs "
                    f"{tuple(rope.cos_sin_cache.shape)}/{rope.cos_sin_cache.stride()} rd {int(rope.rotary_dim)} loc "
                    f"{None if loc is None else tuple(loc.shape)} k {tuple(k_cache.shape)}/{k_cache.stride()} "
                    f"v {tuple(v_cache.shape)}/{v_cache.stride()} h {attn.num_heads}/{attn.num_kv_heads}/{attn.head_dim}"
                )
            return (loc, k_cache, v_cache), ""
        except Exception as e:
            return None, f"exception {type(e).__name__}: {e}"

    def _gemm(self, linear, xq, xs):
        if self.skinny is not None:
            # skinny GEMM launched with PDL: its first weight tiles stream while the previous (NVLink-bound) kernel
            # runs -- the fused TP kernels trigger their dependents right after their own griddepcontrol.wait
            return self.skinny(
                xq, xs, linear.weight.t(), linear.weight_scale, preload=self.preload, depth=self.depth
            )
        return self.mm(xq, linear.weight, xs.view(-1, 1), linear.weight_scale, torch.bfloat16, None)

    # ---------------------------------------------------------------- draft megakernel
    def _mega_layer_args(self, forward_batch, layer_idx: int):
        from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
        if not isinstance(backend, FlashAttentionBackend):
            return None
        return backend.dflash_mega_attn_args(self.layers[layer_idx].self_attn.attn, forward_batch)

    def _mega_args(self, forward_batch):
        """-> (args, "") for the M = 32 megakernel or (None, reason). Identical on every rank (same metadata)."""
        try:
            a = self._mega_layer_args(forward_batch, 0)
        except Exception as e:  # noqa: BLE001
            return None, f"exception {type(e).__name__}: {e}"
        if a is None:
            return None, "attention path is not the bidirectional window kernel"
        if a["max_seqlen_q"] != 8 or tuple(a["window_size"]) != (1023, 1023):
            return None, f"block {a['max_seqlen_q']} / window {a['window_size']}"
        pt, sl, loc = a["page_table"], a["cache_seqlens"], a["loc"]
        if sl.numel() != 4 or sl.dtype != torch.int32 or not sl.is_contiguous():
            return None, f"cache_seqlens {tuple(sl.shape)} {sl.dtype}"
        if pt.dim() != 2 or pt.shape[0] != 4 or pt.dtype != torch.int32 or pt.stride(1) != 1:
            return None, f"page_table {tuple(pt.shape)} {pt.dtype} {pt.stride()}"
        if loc is None or loc.numel() != 32 or loc.dtype not in (torch.int32, torch.int64) or not loc.is_contiguous():
            return None, "loc"
        return a, ""

    def _mega_build(self, forward_batch):
        from sglang.kernels.ops.speculative.dflash_mega import DFlashMegaState

        L = []
        keep = []
        for i, layer in enumerate(self.layers):
            attn, mlp = layer.self_attn, layer.mlp
            a = self._mega_layer_args(forward_batch, i)
            if a is None:
                raise RuntimeError(f"layer {i}: attention args unavailable")
            rope = attn.rotary_emb
            if int(rope.rotary_dim) != 64 or not getattr(rope, "is_neox_style", False) or attn.head_dim != 128:
                raise RuntimeError("rope / head layout")
            if attn.num_heads != 16 or attn.num_kv_heads != 1 or attn.attn.layer_id != i:
                raise RuntimeError("head counts / layer ids")
            d = {}
            for k, lin in (("qkv", attn.qkv_proj), ("o", attn.o_proj), ("gu", mlp.gate_up_proj), ("dn", mlp.down_proj)):
                w = lin.weight.t()
                sc = lin.weight_scale.reshape(-1).to(torch.float32).contiguous()
                keep.append(sc)
                d["w_" + k] = w
                d["s_" + k] = sc
            d["ln_in"] = layer.input_layernorm.weight
            d["ln_post"] = layer.post_attention_layernorm.weight
            d["q_norm"] = attn.q_norm.weight
            d["k_norm"] = attn.k_norm.weight
            d["sinks"] = attn.attention_sink_bias
            d["k_cache"] = a["key_cache"]
            d["v_cache"] = a["value_cache"]
            L.append(d)
        eps = float(self.layers[0].input_layernorm.variance_epsilon)
        for layer in self.layers:
            for n in (layer.input_layernorm, layer.post_attention_layernorm):
                if float(n.variance_epsilon) != eps:
                    raise RuntimeError("eps differs between norms")
        if float(self.model.norm.variance_epsilon) != eps:
            raise RuntimeError("final norm eps")
        a0 = self.layers[0].self_attn
        state = DFlashMegaState(
            self.state, L, self.model.norm.weight, a0.rotary_emb.cos_sin_cache, eps,
            float(a0.q_norm.variance_epsilon), float(a0.v_scale if a0.v_scale is not None else 1.0), float(a0.scaling),
        )
        state._keep = keep
        if state.max_clusters() < 15:
            raise RuntimeError(f"only {state.max_clusters()} co-resident 8-CTA clusters")
        return state

    def _mega_agree(self, ok: bool) -> bool:
        import torch.distributed as dist
        from sglang.srt.distributed import get_tp_group

        t = torch.tensor([1 if ok else 0], dtype=torch.int32)
        dist.all_reduce(t, op=dist.ReduceOp.MIN, group=get_tp_group().cpu_group)
        return bool(t.item())

    def _mega_try(self, positions, hidden_states, forward_batch):
        """The draft megakernel's output for this forward, or None (fused-TP path)."""
        if self.mega_state in ("off", "failed") or hidden_states.shape[0] != 32:
            return None
        args, why = self._mega_args(forward_batch)
        if args is None:
            if not getattr(self, "_mega_why_logged", False) and self.state.rank == 0:
                self._mega_why_logged = True
                logger.info(f"DFLASH draft megakernel not used at M=32: {why}")
            return None
        if self.mega is None:
            if torch.cuda.is_current_stream_capturing():
                return None
            ok, err = True, ""
            try:
                self.mega = self._mega_build(forward_batch)
            except Exception as e:  # noqa: BLE001
                ok, err = False, f"{type(e).__name__}: {e}"
            if not self._mega_agree(ok):
                self.mega_state, self.mega = "failed", None
                logger.warning(f"DFLASH draft megakernel disabled (build): {err or 'another rank failed'}")
                return None
            # the self-check needs real metadata (CUDA-graph warmup batches are dummies: tiny lengths, slot-0 writes)
            real = bool((args["cache_seqlens"] >= 16).all().item()) and bool((args["loc"] != 0).all().item())
            if envs.SGLANG_DFLASH_MEGA_CHECK.get() and self._mega_agree(real):
                ok, msg = self._mega_check(positions, hidden_states, forward_batch, args)
                if not self._mega_agree(ok):
                    self.mega_state, self.mega = "failed", None
                    logger.warning(f"DFLASH draft megakernel disabled (self-check): {msg}")
                    return None
                if self.state.rank == 0:
                    logger.info(f"DFLASH draft megakernel self-check: {msg}")
            elif self.state.rank == 0:
                logger.info("DFLASH draft megakernel: no self-check (warmup / dummy metadata)")
            self.mega_state = "on"
            if self.state.rank == 0:
                logger.info("DFLASH draft megakernel enabled (5 layers at M=32, one persistent kernel)")
        out = torch.empty_like(hidden_states)
        self.mega.run(hidden_states, out, positions, args["cache_seqlens"], args["page_table"], args["loc"])
        return out

    def _mega_check(self, positions, hidden_states, forward_batch, args):
        """Once, eager: the megakernel vs the fused-TP path on copies of this forward's input (both write the same block
        rows to the draft KV cache). Also checks that the block rows are the last 8 page-table entries of each request
        (the megakernel takes the new keys from qkv instead of re-reading the cache)."""
        try:
            pt = args["page_table"].cpu()
            sl = args["cache_seqlens"].cpu()
            loc = args["loc"].cpu().to(torch.int64)
            for b in range(4):
                n = int(sl[b])
                if n < 8 or not torch.equal(pt[b, n - 8:n].to(torch.int64), loc[8 * b:8 * b + 8]):
                    return False, f"block rows of request {b} are not the last page-table entries"
            h_ref = hidden_states.clone()
            o_ref = self._forward_fused(positions, h_ref, forward_batch)
            h_m = hidden_states.clone()
            o_m = torch.empty_like(o_ref)
            self.mega.run(h_m, o_m, positions, args["cache_seqlens"], args["page_table"], args["loc"])
            torch.cuda.synchronize()
            if not bool(torch.isfinite(o_m).all()):
                return False, "non-finite output"
            d = (o_m.float() - o_ref.float())
            rel = (d.norm() / o_ref.float().norm().clamp(min=1e-20)).item()
            mx = (d.abs().max() / o_ref.float().abs().max().clamp(min=1e-20)).item()
            dr = ((h_m.float() - h_ref.float()).norm() / h_ref.float().norm().clamp(min=1e-20)).item()
            return rel < 0.3, f"out rel-L2 {rel:.3e} max-rel {mx:.3e} residual rel-L2 {dr:.3e}"
        except Exception as e:  # noqa: BLE001
            return False, f"exception {type(e).__name__}: {e}"

    def forward(self, positions, hidden_states, forward_batch) -> torch.Tensor:
        out = self._mega_try(positions, hidden_states, forward_batch)
        if out is not None:
            return out
        return self._forward_fused(positions, hidden_states, forward_batch)

    def _forward_fused(self, positions, hidden_states, forward_batch) -> torch.Tensor:
        from sglang.kernels.ops.communication.dflash_tp_fused import rmsnorm_quant, silu_mul_quant

        st = self.state
        M = hidden_states.shape[0]
        dev = hidden_states.device
        layers = self.layers
        xq = torch.empty(M, H, dtype=torch.float8_e4m3fn, device=dev)
        xs = torch.empty(M, dtype=torch.float32, device=dev)
        ln0 = layers[0].input_layernorm
        rmsnorm_quant(hidden_states, ln0.weight, ln0.variance_epsilon, st.rank, xq, xs)
        # the unfused path also accumulates into the embedding buffer in place (residual = hidden_states)
        residual = hidden_states
        out = None
        for i, layer in enumerate(layers):
            attn, mlp = layer.self_attn, layer.mlp
            qkv = self._gemm(attn.qkv_proj, xq, xs)
            a = self._attend(attn, qkv, positions, forward_batch)
            o_part = attn.o_proj.quant_method.apply(attn.o_proj, a)
            xq = torch.empty(M, H, dtype=torch.float8_e4m3fn, device=dev)
            xs = torch.empty(M, dtype=torch.float32, device=dev)
            ln = layer.post_attention_layernorm
            st.arnq(o_part, residual, ln.weight, ln.variance_epsilon, xq, xs)
            gu = self._gemm(mlp.gate_up_proj, xq, xs)
            aq = torch.empty(M, INTER, dtype=torch.float8_e4m3fn, device=dev)
            asc = torch.empty(M, dtype=torch.float32, device=dev)
            silu_mul_quant(gu, aq, asc)
            d_part = self._gemm(mlp.down_proj, aq, asc)
            if i + 1 < len(layers):
                xq = torch.empty(M, H, dtype=torch.float8_e4m3fn, device=dev)
                xs = torch.empty(M, dtype=torch.float32, device=dev)
                ln = layers[i + 1].input_layernorm
                st.arnq(d_part, residual, ln.weight, ln.variance_epsilon, xq, xs)
            else:
                out = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
                norm = self.model.norm
                st.arnq(d_part, residual, norm.weight, norm.variance_epsilon, out, None)
        return out
