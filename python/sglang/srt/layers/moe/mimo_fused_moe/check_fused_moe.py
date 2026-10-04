#!/usr/bin/env python3
"""Multi-process check of the fused MoE kernel on REAL checkpoint weights, one process per TP rank
(the production setting: CUDA-IPC all-reduce buffers, Humming's own weight transform), without the server.

  torchrun --nproc_per_node 8 check_fused_moe.py --model /path/to/checkpoint --layer 1 --M 24 32 48 64

Per rank: loads its TP slice of one MoE layer (gate/up rows, down columns) from the safetensors shards, runs Humming's
transform exactly as SGLang's prepare_humming_moe_layer does (fp8 g128 activations), then for each M:
  * fused kernel (bf16 out, residual_out) with random bf16 hidden/residual (identical on all ranks),
  * fp32 reference: the kernel's own normed x + routing (checked against the SGLang-semantics oracle), fc1/fc2 with the
    effective dequantized weights (the torch re-implementation of Humming's transform, bit-checked against Humming),
    summed over ranks with an fp32 NCCL all-reduce,
  * timing: median wall time per launch over all ranks (barrier-aligned loop, every rank launches back-to-back).
"""

import argparse
import json
import os
import time

import fused_moe_wgmma as F
import torch
import torch.distributed as dist
from safetensors import safe_open

DIM, INTER, NEXP, TOPK = F.DIM, F.INTER, F.NEXP, F.TOPK


def load_layer_slice(model_dir, layer, rank, tp, device):
    """This rank's TP slice of one MoE layer: (gate_packed, gate_scale, up_packed, up_scale, down_packed, down_scale,
    router_w fp32, bias fp32, norm_w bf16)."""
    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))[
        "weight_map"
    ]
    inter = 2 * INTER * tp // 2  # full intermediate 2048
    rows = slice(rank * INTER, (rank + 1) * INTER)
    kcols = slice(rank * (INTER // 2), (rank + 1) * (INTER // 2))
    scols = slice(rank * (INTER // 32), (rank + 1) * (INTER // 32))
    handles = {}

    def get(name, sl=None):
        shard = index[name]
        if shard not in handles:
            handles[shard] = safe_open(
                os.path.join(model_dir, shard), framework="pt", device="cpu"
            )
        t = handles[shard].get_slice(name)
        return (t[sl] if sl is not None else t[:]).clone()

    gp, gs, up, us, dp, ds = [], [], [], [], [], []
    for e in range(NEXP):
        p = f"model.layers.{layer}.mlp.experts.{e}."
        gp.append(get(p + "gate_proj.weight", rows))
        gs.append(get(p + "gate_proj.weight_scale", rows))
        up.append(get(p + "up_proj.weight", rows))
        us.append(get(p + "up_proj.weight_scale", rows))
        dp.append(get(p + "down_proj.weight", (slice(None), kcols)))
        ds.append(get(p + "down_proj.weight_scale", (slice(None), scols)))
    st = lambda l: torch.stack(l).to(device)
    pre = f"model.layers.{layer}."
    router_w = get(pre + "mlp.gate.weight").float().to(device).contiguous()
    bias = get(pre + "mlp.gate.e_score_correction_bias").float().to(device).contiguous()
    norm_w = (
        get(pre + "post_attention_layernorm.weight")
        .to(torch.bfloat16)
        .to(device)
        .contiguous()
    )
    return st(gp), st(gs), st(up), st(us), st(dp), st(ds), router_w, bias, norm_w


def humming_transform(packed, scale, sublayer):
    """Humming's transform as SGLang's prepare_humming_moe_layer runs it (fp8 g128 input schema)."""
    from humming import dtypes
    from humming.layer import HummingMethod
    from humming.schema import HummingInputSchema, HummingWeightSchema

    E, N, K2 = packed.shape
    layer = torch.nn.Module()
    setattr(
        layer,
        f"{sublayer}_weight",
        torch.nn.Parameter(packed.contiguous().view(torch.int32), requires_grad=False),
    )
    setattr(
        layer,
        f"{sublayer}_weight_scale",
        torch.nn.Parameter(
            scale.contiguous().view(torch.float8_e8m0fnu), requires_grad=False
        ),
    )
    wschema = HummingWeightSchema(
        b_dtype=dtypes.float4e2m1,
        bs_dtype=dtypes.float8e8m0,
        weight_scale_group_size=32,
    )
    ischema = HummingInputSchema.from_config(
        {
            "a_dtype": "float8e4m3",
            "input_scale_group_size": 128,
            "quant_method": "humming",
        }
    )
    HummingMethod.prepare_layer_meta(
        layer=layer,
        shape_n=N,
        shape_k=2 * K2,
        pad_n_to_multiple=256,
        pad_k_to_multiple=128,
        input_schema=ischema,
        weight_schema=wschema,
        has_bias=False,
        num_experts=E,
        torch_dtype=torch.bfloat16,
        sublayer_name=sublayer,
    )
    HummingMethod.transform_humming_layer(layer, sublayer_name=sublayer)
    return layer


def ipc_share(nbytes, rank, world, device):
    """cudaMalloc + IPC handle exchange -> (local ptr object, [ptr per rank])."""
    from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

    lib = CudaRTLibrary()
    ptr = lib.cudaMalloc(nbytes)
    lib.cudaMemset(ptr, 0, nbytes)
    handle = lib.cudaIpcGetMemHandle(ptr)
    objs = [None] * world
    dist.all_gather_object(objs, bytes(handle))
    ptrs = []
    for r, h in enumerate(objs):
        ptrs.append(
            ptr.value
            if r == rank
            else lib.cudaIpcOpenMemHandle(type(handle).from_buffer_copy(h)).value
        )
    return ptr, ptrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default="/raid/data/aichenf_scratch/models/MiMo-V2.5-Pro-FP4-DFlash"
    )
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--M", type=int, nargs="*", default=[24, 32, 48, 64])
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument(
        "--no-humming",
        action="store_true",
        help="use the torch re-implementation of the transform instead of humming",
    )
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local}")
    torch.cuda.set_device(device)
    say = lambda *a: print(f"[rank {rank}]", *a, flush=True) if rank == 0 else None

    t0 = time.time()
    gp, gs, up, us, dp, ds, router_w, bias, norm_w = load_layer_slice(
        args.model, args.layer, rank, world, device
    )
    say(
        f"loaded layer {args.layer} slice in {time.time() - t0:.1f}s: gate {tuple(gp.shape)} down {tuple(dp.shape)}"
    )

    # effective weights + (reference) kernel layout from the torch re-implementation; the kernel weights from humming
    w13_ref, wg_eff, wu_eff = F.prep_fc1(gp, gs, up, us)
    w2_ref, wd_eff = F.prep_fc2(dp, ds)
    if args.no_humming:
        w = {**w13_ref, **w2_ref}
    else:
        l13 = humming_transform(
            torch.cat([gp, up], dim=1), torch.cat([gs, us], dim=1), "w13"
        )
        l2 = humming_transform(dp, ds, "w2")
        layer = torch.nn.Module()
        for n in ("w13_weight", "w13_weight_scale", "w13_weight_scale_2"):
            setattr(layer, n, getattr(l13, n))
        for n in ("w2_weight", "w2_weight_scale", "w2_weight_scale_2"):
            setattr(layer, n, getattr(l2, n))
        w = F.humming_layer_weights(layer)
        same = all(
            torch.equal(
                w[k].view(torch.uint8).reshape(-1),
                w13_ref.get(k, w2_ref.get(k)).view(torch.uint8).reshape(-1),
            )
            if w[k].dtype != torch.float32
            else torch.equal(
                w[k].reshape(-1), w13_ref.get(k, w2_ref.get(k)).reshape(-1)
            )
            for k in w
        )
        say(f"humming transform == torch re-implementation on the real weights: {same}")
        assert same, "layout mismatch"
    del gp, gs, up, us, dp, ds
    torch.cuda.empty_cache()

    F.ext()
    # Match the router precision selected by the compiled production kernel.
    # Convert once, outside every measured launch and any future Graph replay.
    router_w_bf16 = router_w.to(torch.bfloat16) if F.ext().ROUTER_BF16 else None
    router_reference_w = (
        router_w_bf16.float() if router_w_bf16 is not None else router_w
    )
    say(f"router reference: {'bf16-rounded' if router_w_bf16 is not None else 'fp32'}")
    state = F.RankState(device, alloc_comm=False)
    _rs, rs_ptrs = ipc_share(F.RankState.RS_BYTES, rank, world, device)
    _ag, ag_ptrs = ipc_share(F.RankState.AG_BYTES, rank, world, device)
    state.link_ptrs(rs_ptrs, ag_ptrs)
    torch.cuda.synchronize()
    dist.barrier()

    all_ok = True
    for M in args.M:
        gen = torch.Generator(device=device)
        gen.manual_seed(1234 + M)  # identical hidden/residual on every rank
        hidden = (torch.randn(M, DIM, device=device, generator=gen) * 1.0).to(
            torch.bfloat16
        )
        residual = (torch.randn(M, DIM, device=device, generator=gen) * 1.0).to(
            torch.bfloat16
        )
        inp = dict(
            hidden=hidden,
            residual=residual,
            norm_w=norm_w,
            router_w=router_w,
            bias=bias,
        )
        if router_w_bf16 is not None:
            inp["router_w_bf16"] = router_w_bf16
        out = torch.zeros(M, DIM, dtype=torch.bfloat16, device=device)
        ro = torch.zeros(M, DIM, dtype=torch.bfloat16, device=device)
        dist.barrier()
        F.run_fused_full(state, inp, w, out, ro, rank, world)
        torch.cuda.synchronize()
        # prologue checks vs the SGLang-semantics oracles
        ro_ref, xn_ref = F.rmsnorm_reference(hidden, residual, norm_w)
        xn_k = state.xn_buf[: M * DIM].reshape(M, DIM)
        _, ids_ref, w_ref = F.routing_reference(xn_k, router_reference_w, bias)
        ids_k, wk = state.kernel_topk(M)
        sets_ok = all(
            set(ids_k[t].tolist()) == set(ids_ref[t].tolist()) for t in range(M)
        )
        uni = state.kernel_union().tolist()
        U = len(uni)
        # fp32 reference of this rank's partial from the kernel's own routing / quantized x, summed over ranks
        union = torch.tensor(uni, dtype=torch.int32, device=device)
        gate_w = F.gate_w_from_routing(ids_k, wk, uni)
        xq = state.xq_buf[: M * DIM].reshape(M, DIM).view(torch.float8_e4m3fn)
        xs = state.xs_buf[: M * (DIM // F.A_GROUP)].reshape(M, DIM // F.A_GROUP)
        ref = F.reference_full(xq, xs, wg_eff, wu_eff, wd_eff, union, gate_w)
        dist.all_reduce(ref)
        diff = (out.float() - ref).abs()
        tol = 0.05 * ref.abs() + 0.02 * ref.abs().max()
        viol = int((diff > tol).sum())
        finite = all(bool(torch.isfinite(t).all()) for t in (out, ro, ref, wk))
        ok = finite and bool(torch.equal(ro, ro_ref)) and sets_ok and viol == 0
        # the same routing / union on every rank
        unis = [None] * world
        dist.all_gather_object(unis, uni)
        ok &= all(u == uni for u in unis)
        all_ok &= ok
        # timing
        for _ in range(5):
            F.run_fused_full(state, inp, w, out, ro, rank, world)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(args.rounds):
            F.run_fused_full(state, inp, w, out, ro, rank, world)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / args.rounds * 1e6
        ts = [None] * world
        dist.all_gather_object(ts, us)
        say(
            f"M={M:3d} U={U:3d}: residual_out exact={bool(torch.equal(ro, ro_ref))} top8 sets ok={sets_ok} union identical on ranks={all(u == uni for u in unis)} "
            f"out vs fp32 ref: max_err={diff.max().item():.3e} ref_max={ref.abs().max().item():.3e} viol={viol}/{diff.numel()} ok={ok} | "
            f"fused us/launch (min..max over ranks) {min(ts):.1f}..{max(ts):.1f}"
        )
    verdict = torch.tensor(int(all_ok), dtype=torch.int32, device=device)
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
    all_ok = bool(verdict.item())
    say("ALL OK" if all_ok else "SOME CHECKS FAILED")
    dist.barrier()
    dist.destroy_process_group()
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
