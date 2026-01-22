#!/usr/bin/env python3
"""
Test script: compare SGLang SVG2 Triton kernels vs Sparse-VideoGen ("SVG") kernels.

What we compare:
1) KMeans pieces (Euclidean):
   - nearest-centroid assignment (Triton)
   - centroid update (sorted-ids chunk kernel)
2) SAP permutation:
   - permute + inverse permute (Triton)
3) Dynamic block mask:
   - SGLang identify_dynamic_mask vs Sparse-VideoGen identify_dynamic_map
4) Sparse attention (permuted domain):
   - SGLang block_sparse_attention (split-k + indirect access, Triton)
   - Sparse-VideoGen dynamic_block_sparse_fwd_triton (direct block iteration, Triton)

Run:
  python3 sglang/test_sglang_vs_sparsevideogen_kernels.py --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Tuple

import torch


def _add_repo_paths() -> None:
    """Make local repo modules importable without installation."""
    this_file = os.path.abspath(__file__)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(this_file), ".."))

    # Prefer local sources
    sglang_python = os.path.join(repo_root, "sglang", "python")
    sparse_videogen_root = os.path.join(repo_root, "Sparse-VideoGen")

    for p in [repo_root, sglang_python, sparse_videogen_root]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _mock_optional_deps_for_sparse_videogen() -> None:
    """
    Sparse-VideoGen's `svg/kmeans_utils.py` unconditionally imports RAPIDS cuVS:
      `from cuvs.cluster.kmeans import KMeansParams, fit`

    That dependency is only needed for the optional cuVS-based KMeans path and
    is not required for the Triton/PyTorch kernels we compare here.

    To keep this test runnable on environments without RAPIDS, we provide a
    minimal stub module that satisfies the import. If code tries to *use* cuVS,
    it will raise with a helpful error.
    """
    if "cuvs" in sys.modules:
        return

    import types

    cuvs_mod = types.ModuleType("cuvs")
    cluster_mod = types.ModuleType("cuvs.cluster")
    kmeans_mod = types.ModuleType("cuvs.cluster.kmeans")

    class _KMeansParams:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "cuvs (RAPIDS) is not installed. This test stubs it only to import "
                "Sparse-VideoGen kernels. If you need cuVS KMeans, install RAPIDS cuVS."
            )

    def _fit(*args, **kwargs):  # pragma: no cover
        raise RuntimeError(
            "cuvs (RAPIDS) is not installed. This test stubs it only to import "
            "Sparse-VideoGen kernels. If you need cuVS KMeans, install RAPIDS cuVS."
        )

    kmeans_mod.KMeansParams = _KMeansParams
    kmeans_mod.fit = _fit

    # Register modules
    sys.modules["cuvs"] = cuvs_mod
    sys.modules["cuvs.cluster"] = cluster_mod
    sys.modules["cuvs.cluster.kmeans"] = kmeans_mod


def _require_cuda(device: str) -> None:
    if device != "cuda":
        raise ValueError("This test is intended for CUDA. Use --device cuda.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")


def _sync_if_cuda(t: torch.Tensor) -> None:
    if t.is_cuda:
        torch.cuda.synchronize()


@dataclass
class Tols:
    rtol: float
    atol: float


def _assert_allclose(name: str, a: torch.Tensor, b: torch.Tensor, tols: Tols) -> None:
    if a.shape != b.shape:
        raise AssertionError(f"{name}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.dtype != b.dtype:
        # allow dtype mismatch only if both are floating and we can compare in fp32
        if not (a.is_floating_point() and b.is_floating_point()):
            raise AssertionError(f"{name}: dtype mismatch {a.dtype} vs {b.dtype}")
        a_cmp = a.float()
        b_cmp = b.float()
    else:
        a_cmp = a
        b_cmp = b
    torch.testing.assert_close(a_cmp, b_cmp, rtol=tols.rtol, atol=tols.atol)


def _assert_equal(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    if a.shape != b.shape:
        raise AssertionError(f"{name}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    if not torch.equal(a, b):
        # Provide a small diagnostic
        neq = (a != b).flatten()
        mismatch = int(neq.sum().item())
        total = a.numel()
        raise AssertionError(f"{name}: not equal ({mismatch}/{total} mismatched)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--kmeans_N", type=int, default=1024)
    parser.add_argument("--kmeans_K", type=int, default=64)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--S", type=int, default=512)
    parser.add_argument("--Kq", type=int, default=32)
    parser.add_argument("--Kk", type=int, default=48)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--min_kc_ratio", type=float, default=0.1)
    parser.add_argument("--split_k", type=int, default=4)
    args = parser.parse_args()

    _add_repo_paths()
    _mock_optional_deps_for_sparse_videogen()
    _require_cuda(args.device)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device(args.device)

    # Imports (after sys.path setup)
    from sglang.multimodal_gen.runtime.layers.attention.backends import (  # noqa: E402
        svg2_sparse_attn as sgl_svg2,
    )

    from svg.kmeans_utils import (  # noqa: E402
        euclid_assign_triton as svg_euclid_assign,
        identify_dynamic_map as svg_identify_dynamic_map,
        triton_centroid_update_sorted_euclid as svg_centroid_update_sorted,
        dynamic_block_sparse_fwd_triton as svg_sparse_attn_triton,
    )
    from svg.kernels.triton.permute import (  # noqa: E402
        permute_tensor_by_labels_triton as svg_permute_triton,
        apply_inverse_permutation_triton as svg_inv_permute_triton,
    )

    print("=== [1] KMeans: Euclid assign kernel ===")
    Bk = args.B * args.H  # test batched-over-(B*H) like common usage
    N = args.kmeans_N
    K = args.kmeans_K
    D = args.D
    x = torch.randn((Bk, N, D), device=device, dtype=dtype)
    centroids = torch.randn((Bk, K, D), device=device, dtype=dtype)
    x_sq = (x.float() ** 2).sum(dim=-1)  # [Bk, N] float32

    _sync_if_cuda(x)
    sgl_labels = sgl_svg2._svg_euclid_assign(x, centroids, x_sq)
    _sync_if_cuda(x)
    svg_labels = svg_euclid_assign(x, centroids, x_sq)
    _sync_if_cuda(x)
    _assert_equal("kmeans.assign.labels", sgl_labels, svg_labels)
    print("PASS: kmeans assign labels match exactly")

    print("=== [2] KMeans: centroid update kernel (sorted ids) ===")
    _sync_if_cuda(x)
    sgl_new_centroids, sgl_counts = sgl_svg2._svg_centroid_update_sorted(x, sgl_labels, centroids)
    _sync_if_cuda(x)
    svg_new_centroids, svg_counts = svg_centroid_update_sorted(x, svg_labels, centroids)
    _sync_if_cuda(x)
    _assert_equal("kmeans.update.counts", sgl_counts.to(torch.int64), svg_counts.to(torch.int64))
    _assert_allclose("kmeans.update.centroids", sgl_new_centroids, svg_new_centroids, Tols(rtol=1e-3, atol=1e-3))
    print("PASS: centroid update counts match, centroids close")

    print("=== [3] SAP permute/inverse kernels ===")
    B, H, S = args.B, args.H, args.S
    x4 = torch.randn((B, H, S, D), device=device, dtype=dtype)
    labels4 = torch.randint(0, args.Kq, (B * H, S), device=device, dtype=torch.int64)

    # SGLang permute (returns indices [BH, S])
    x4_sgl_perm, sgl_sorted = sgl_svg2.permute_by_labels(x4, labels=labels4)
    # Sparse-VideoGen permute (returns indices [BH, S]); compare values and indices
    x4_svg_perm, svg_sorted = svg_permute_triton(x4, labels4, dim=2)

    _assert_equal("permute.sorted_indices", sgl_sorted.to(torch.int32), svg_sorted.to(torch.int32))
    _assert_allclose("permute.output", x4_sgl_perm, x4_svg_perm, Tols(rtol=0.0, atol=0.0))

    # Inverse permute: SGLang wants [BH,S], Sparse helper wants [B,H,S] per docs
    x4_sgl_inv = sgl_svg2.inverse_permute(x4_sgl_perm, sgl_sorted)
    x4_svg_inv = svg_inv_permute_triton(x4_svg_perm, svg_sorted.reshape(B, H, S), dim=2)
    _assert_allclose("inverse_permute.output", x4_sgl_inv, x4_svg_inv, Tols(rtol=0.0, atol=0.0))
    _assert_allclose("inverse_permute.reconstruct", x4_sgl_inv, x4, Tols(rtol=0.0, atol=0.0))
    print("PASS: permute + inverse-permute match and reconstruct exactly")

    print("=== [4] Dynamic block mask generation ===")
    Kq, Kk = args.Kq, args.Kk
    q_centroids = torch.randn((B, H, Kq, D), device=device, dtype=dtype)
    k_centroids = torch.randn((B, H, Kk, D), device=device, dtype=dtype)
    q_sizes = torch.randint(0, 32, (B, H, Kq), device=device, dtype=torch.int64)
    k_sizes = torch.randint(0, 32, (B, H, Kk), device=device, dtype=torch.int64)

    sgl_mask = sgl_svg2.identify_dynamic_mask(
        q_centroids, k_centroids, q_sizes, k_sizes, top_p=args.top_p, min_kc_ratio=args.min_kc_ratio
    )
    svg_mask = svg_identify_dynamic_map(
        q_centroids, k_centroids, q_sizes, k_sizes, p=args.top_p, min_kc_ratio=args.min_kc_ratio
    )
    _assert_equal("dynamic_mask", sgl_mask, svg_mask)
    print("PASS: dynamic block mask matches exactly")

    print("=== [5] Sparse attention output (permuted domain) ===")
    # Build a consistent test case:
    # - random q/k/v
    # - derive q/k labels
    # - compute cluster sizes via bincount
    # - permute (SGLang + Sparse agree on order because we reuse sorted indices)
    q = torch.randn((B, H, S, D), device=device, dtype=dtype)
    k = torch.randn((B, H, S, D), device=device, dtype=dtype)
    v = torch.randn((B, H, S, D), device=device, dtype=dtype)

    q_labels = torch.randint(0, Kq, (B * H, S), device=device, dtype=torch.int64)
    k_labels = torch.randint(0, Kk, (B * H, S), device=device, dtype=torch.int64)

    # cluster sizes [B,H,K]
    q_sizes2 = torch.zeros((B * H, Kq), device=device, dtype=torch.int64)
    k_sizes2 = torch.zeros((B * H, Kk), device=device, dtype=torch.int64)
    ones = torch.ones((B * H, S), device=device, dtype=torch.int64)
    q_sizes2.scatter_add_(1, q_labels, ones)
    k_sizes2.scatter_add_(1, k_labels, ones)
    q_sizes2 = q_sizes2.reshape(B, H, Kq)
    k_sizes2 = k_sizes2.reshape(B, H, Kk)

    # mask based on centroids (use random centroids, independent of q/k here; OK for kernel equivalence)
    q_c2 = torch.randn((B, H, Kq, D), device=device, dtype=dtype)
    k_c2 = torch.randn((B, H, Kk, D), device=device, dtype=dtype)
    block_mask = sgl_svg2.identify_dynamic_mask(
        q_c2, k_c2, q_sizes2, k_sizes2, top_p=args.top_p, min_kc_ratio=args.min_kc_ratio
    )

    # Permute with shared sorted indices
    q_perm, q_sorted = sgl_svg2.permute_by_labels(q, labels=q_labels)
    k_perm, k_sorted = sgl_svg2.permute_by_labels(k, labels=k_labels)
    v_perm, _ = sgl_svg2.permute_by_labels(v, sorted_indices=k_sorted)

    _sync_if_cuda(q_perm)
    out_sgl = sgl_svg2.block_sparse_attention(
        q_perm, k_perm, v_perm, block_mask, q_sizes2, k_sizes2, split_k=args.split_k
    )
    _sync_if_cuda(q_perm)
    out_svg = svg_sparse_attn_triton(q_perm, k_perm, v_perm, block_mask, q_sizes2, k_sizes2)
    _sync_if_cuda(q_perm)

    # Different accumulation order (split-k vs direct) => allow a bit more tolerance
    _assert_allclose("sparse_attn.out", out_sgl, out_svg, Tols(rtol=2e-2, atol=2e-2))
    print("PASS: sparse attention outputs are close (rtol=2e-2, atol=2e-2)")

    print("\nALL TESTS PASSED ✅")


if __name__ == "__main__":
    main()


