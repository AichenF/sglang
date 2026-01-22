#!/usr/bin/env python3
"""
Comparison Test: SGLang SVG2 vs Sparse-VideoGen SAP Attention

This script compares the outputs of:
1. SGLang's svg2_attention_forward (Triton-based)
2. Sparse-VideoGen's SAP attention (FlashInfer-based)

Usage:
    python test_svg2_vs_svg_comparison.py
"""

import math
import sys
import torch
import torch.nn.functional as F
from typing import Tuple, Optional

# ============================================================================
# Helper Functions
# ============================================================================

def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_stats(name: str, tensor: torch.Tensor):
    """Print statistics about a tensor."""
    print(f"  {name}:")
    print(f"    shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    print(f"    min={tensor.min().item():.6f}, max={tensor.max().item():.6f}")
    print(f"    mean={tensor.float().mean().item():.6f}, std={tensor.float().std().item():.6f}")
    if torch.isnan(tensor).any():
        print(f"    ⚠️  Contains NaN!")
    if torch.isinf(tensor).any():
        print(f"    ⚠️  Contains Inf!")


def compare_tensors(name: str, t1: torch.Tensor, t2: torch.Tensor, rtol: float = 1e-2, atol: float = 1e-2):
    """Compare two tensors and print differences."""
    if t1.shape != t2.shape:
        print(f"  {name}: Shape mismatch! {t1.shape} vs {t2.shape}")
        return False, float('inf')
    
    diff = (t1.float() - t2.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # Relative difference
    rel_diff = diff / (t1.float().abs() + 1e-6)
    max_rel_diff = rel_diff.max().item()
    
    close = torch.allclose(t1, t2, rtol=rtol, atol=atol)
    status = "✓ PASS" if close else "✗ FAIL"
    
    print(f"  {name}: {status}")
    print(f"    max_abs_diff={max_diff:.6f}, mean_abs_diff={mean_diff:.6f}")
    print(f"    max_rel_diff={max_rel_diff:.6f}")
    
    return close, max_diff


# ============================================================================
# Dense Attention Reference
# ============================================================================

def dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Standard dense attention.
    
    Args:
        q, k, v: [B, S, H, D]
    
    Returns:
        output: [B, S, H, D]
    """
    B, S, H, D = q.shape
    scale = 1.0 / math.sqrt(D)
    
    # [B, S, H, D] -> [B, H, S, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, v)
    
    # [B, H, S, D] -> [B, S, H, D]
    return out.transpose(1, 2)


# ============================================================================
# Import Functions
# ============================================================================

def import_sglang_svg2():
    """Import SGLang's SVG2 attention forward."""
    try:
        from sglang.multimodal_gen.runtime.layers.attention.backends.svg2_sparse_attn import (
            svg2_attention_forward,
            triton_kmeans,
            permute_by_labels,
            inverse_permute,
            identify_dynamic_mask,
            block_sparse_attention,
        )
        return {
            'svg2_attention_forward': svg2_attention_forward,
            'triton_kmeans': triton_kmeans,
            'permute_by_labels': permute_by_labels,
            'inverse_permute': inverse_permute,
            'identify_dynamic_mask': identify_dynamic_mask,
            'block_sparse_attention': block_sparse_attention,
        }
    except ImportError as e:
        print(f"Failed to import SGLang SVG2: {e}")
        return None


def import_svg_sap():
    """Import Sparse-VideoGen's SAP attention."""
    try:
        sys.path.insert(0, "/home/scratch.aichenf_wwfo/Sparse-VideoGen")
        
        from svg.kmeans_utils import (
            batch_kmeans_Euclid,
            identify_dynamic_map,
            dynamic_block_sparse_fwd_flashinfer,
        )
        
        # Permutation functions are in a separate module
        from svg.kernels.triton.permute import (
            permute_tensor_by_labels_triton,
            apply_inverse_permutation_triton,
        )
        
        return {
            'batch_kmeans_Euclid': batch_kmeans_Euclid,
            'identify_dynamic_map': identify_dynamic_map,
            'dynamic_block_sparse_fwd_flashinfer': dynamic_block_sparse_fwd_flashinfer,
            'permute_tensor_by_labels_triton': permute_tensor_by_labels_triton,
            'apply_inverse_permutation_triton': apply_inverse_permutation_triton,
        }
    except ImportError as e:
        print(f"Failed to import Sparse-VideoGen SAP: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# Test: K-Means Comparison
# ============================================================================

def test_kmeans_comparison(sglang_funcs: dict, svg_funcs: dict, device: str = "cuda"):
    """Compare K-Means clustering outputs."""
    print_header("Test: K-Means Clustering Comparison")
    
    torch.manual_seed(42)
    
    # Test parameters
    B, N, D = 2, 1024, 128
    K = 16
    max_iters = 10
    
    x = torch.randn(B, N, D, device=device, dtype=torch.float16)
    
    # SGLang's Triton K-Means
    print("\n[SGLang Triton K-Means]")
    sglang_labels, sglang_centroids, sglang_sizes = sglang_funcs['triton_kmeans'](
        x, K, max_iters=max_iters
    )
    print_stats("labels", sglang_labels)
    print_stats("centroids", sglang_centroids)
    print_stats("sizes", sglang_sizes)
    
    # SVG's batch_kmeans_Euclid  
    print("\n[SVG batch_kmeans_Euclid]")
    svg_labels, svg_centroids, svg_sizes, svg_iters = svg_funcs['batch_kmeans_Euclid'](
        x, n_clusters=K, max_iters=max_iters
    )
    print_stats("labels", svg_labels)
    print_stats("centroids", svg_centroids)
    print_stats("sizes", svg_sizes)
    print(f"    iterations used: {svg_iters}")
    
    # Check sizes sum to N
    sglang_sum = sglang_sizes.sum(dim=-1).tolist()
    svg_sum = svg_sizes.sum(dim=-1).tolist()
    print(f"\n[Cluster Size Sum Check]")
    print(f"  SGLang sizes sum: {sglang_sum}")
    print(f"  SVG sizes sum: {svg_sum}")
    print(f"  Expected: [{N}] * {B}")
    
    # Note: Cluster assignments may differ due to different initialization/algorithm
    # We mainly check that both produce valid outputs
    return True


# ============================================================================
# Test: Full Attention Comparison
# ============================================================================

def test_full_attention_comparison(sglang_funcs: dict, svg_funcs: dict, device: str = "cuda"):
    """Compare full SVG2/SAP attention outputs."""
    print_header("Test: Full Sparse Attention Comparison")
    
    torch.manual_seed(42)
    
    # Test parameters (small for testing)
    B, S, H, D = 1, 512, 8, 64
    num_q_clusters = 8
    num_k_clusters = 8
    top_p = 0.9
    kmeans_iters = 10
    
    # Generate input [B, S, H, D] for SGLang
    q_bshd = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
    k_bshd = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
    v_bshd = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
    
    # Dense attention reference
    print("\n[Dense Attention Reference]")
    dense_out = dense_attention(q_bshd, k_bshd, v_bshd)
    print_stats("dense_output", dense_out)
    
    # SGLang SVG2
    print("\n[SGLang SVG2 Attention]")
    try:
        sglang_out, q_cent, k_cent, profile = sglang_funcs['svg2_attention_forward'](
            q_bshd, k_bshd, v_bshd,
            num_q_clusters=num_q_clusters,
            num_k_clusters=num_k_clusters,
            top_p=top_p,
            kmeans_iters=kmeans_iters,
            enable_profiling=True,
        )
        print_stats("sglang_output", sglang_out)
        if profile:
            print(f"  Profile: {profile}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        sglang_out = None
    
    # SVG SAP - requires [B, H, S, D] format
    print("\n[SVG SAP Attention (via component functions)]")
    try:
        # Convert to [B, H, S, D] for SVG
        q_bhsd = q_bshd.transpose(1, 2).contiguous()
        k_bhsd = k_bshd.transpose(1, 2).contiguous()
        v_bhsd = v_bshd.transpose(1, 2).contiguous()
        
        # Flatten batch and head
        q_flat = q_bhsd.reshape(B * H, S, D)
        k_flat = k_bhsd.reshape(B * H, S, D)
        
        # K-Means
        q_labels, q_centroids, q_sizes, _ = svg_funcs['batch_kmeans_Euclid'](
            q_flat, n_clusters=num_q_clusters, max_iters=kmeans_iters
        )
        k_labels, k_centroids, k_sizes, _ = svg_funcs['batch_kmeans_Euclid'](
            k_flat, n_clusters=num_k_clusters, max_iters=kmeans_iters
        )
        
        # Reshape for identify_dynamic_map
        q_sizes_reshaped = q_sizes.view(B, H, num_q_clusters)
        k_sizes_reshaped = k_sizes.view(B, H, num_k_clusters)
        q_centroids_reshaped = q_centroids.view(B, H, num_q_clusters, D)
        k_centroids_reshaped = k_centroids.view(B, H, num_k_clusters, D)
        
        # Dynamic mask
        dynamic_map = svg_funcs['identify_dynamic_map'](
            q_centroids_reshaped, k_centroids_reshaped,
            q_sizes_reshaped, k_sizes_reshaped,
            p=top_p, min_kc_ratio=0.0
        )
        print_stats("dynamic_map", dynamic_map.float())
        
        # Permutation
        q_perm, q_sorted = svg_funcs['permute_tensor_by_labels_triton'](q_bhsd, q_labels, dim=2)
        k_perm, k_sorted = svg_funcs['permute_tensor_by_labels_triton'](k_bhsd, k_labels, dim=2)
        v_perm, _ = svg_funcs['permute_tensor_by_labels_triton'](v_bhsd, k_labels, dim=2, sorted_indices=k_sorted)
        
        # Sparse attention
        out_perm = svg_funcs['dynamic_block_sparse_fwd_flashinfer'](
            q_perm, k_perm, v_perm, dynamic_map, q_sizes_reshaped, k_sizes_reshaped, is_cpu=False
        )
        
        # Inverse permutation
        svg_out_bhsd = svg_funcs['apply_inverse_permutation_triton'](out_perm, q_sorted, dim=2)
        svg_out = svg_out_bhsd.transpose(1, 2).contiguous()  # Back to [B, S, H, D]
        
        print_stats("svg_output", svg_out)
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        svg_out = None
    
    # Compare outputs
    print_header("Comparison Results")
    
    if sglang_out is not None:
        compare_tensors("SGLang vs Dense", sglang_out, dense_out, rtol=0.5, atol=0.5)
    
    if svg_out is not None:
        compare_tensors("SVG vs Dense", svg_out, dense_out, rtol=0.5, atol=0.5)
    
    if sglang_out is not None and svg_out is not None:
        compare_tensors("SGLang vs SVG", sglang_out, svg_out, rtol=0.1, atol=0.1)
    
    return True


# ============================================================================
# Test: Shared-intermediate comparison (pinpoint semantic mismatches)
# ============================================================================

def test_shared_intermediates(svg_funcs: dict, sglang_funcs: dict, device: str = "cuda"):
    """
    Use Sparse-VideoGen to produce intermediates (labels/centroids/sizes/mask/permutation),
    then run BOTH kernels using the SAME intermediates:
      - SGLang: block_sparse_attention + inverse_permute
      - SVG: dynamic_block_sparse_fwd_flashinfer + apply_inverse_permutation_triton

    If these match, SGLang is not "wrong"—differences come from KMeans/mask differences.
    If these DO NOT match, we likely have a real semantic mismatch (mask/sizes/permutation)
    between SGLang and Sparse-VideoGen.
    """
    print_header("Test: Shared Intermediates (SVG→SGLang) Kernel Equivalence")

    torch.manual_seed(123)

    # Keep it small-ish but non-trivial
    B, S, H, D = 1, 512, 8, 64
    Kq, Kk = 8, 8
    top_p = 0.9
    min_kc_ratio = 0.0
    kmeans_iters = 10

    # Input in [B, S, H, D]
    q_bshd = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
    k_bshd = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
    v_bshd = torch.randn(B, S, H, D, device=device, dtype=torch.float16)

    # Convert to [B, H, S, D] (the format both block-sparse kernels use)
    q = q_bshd.transpose(1, 2).contiguous()
    k = k_bshd.transpose(1, 2).contiguous()
    v = v_bshd.transpose(1, 2).contiguous()

    # Flatten [B,H] for KMeans
    q_flat = q.reshape(B * H, S, D)
    k_flat = k.reshape(B * H, S, D)

    print("\n[1) SVG KMeans → centroids/sizes/labels]")
    q_labels, q_centroids, q_sizes, _ = svg_funcs["batch_kmeans_Euclid"](
        q_flat, n_clusters=Kq, max_iters=kmeans_iters
    )
    k_labels, k_centroids, k_sizes, _ = svg_funcs["batch_kmeans_Euclid"](
        k_flat, n_clusters=Kk, max_iters=kmeans_iters
    )

    q_sizes_bh = q_sizes.view(B, H, Kq).to(torch.int32)
    k_sizes_bh = k_sizes.view(B, H, Kk).to(torch.int32)
    q_cent_bh = q_centroids.view(B, H, Kq, D)
    k_cent_bh = k_centroids.view(B, H, Kk, D)

    # Sanity: sizes sum to S
    print(f"  q_sizes sum: {q_sizes_bh.sum(dim=-1).unique().tolist()} (expect {S})")
    print(f"  k_sizes sum: {k_sizes_bh.sum(dim=-1).unique().tolist()} (expect {S})")

    print("\n[2) Mask generation: SVG identify_dynamic_map vs SGLang identify_dynamic_mask (same centroids)]")
    mask_svg = svg_funcs["identify_dynamic_map"](
        q_cent_bh, k_cent_bh, q_sizes_bh, k_sizes_bh, p=top_p, min_kc_ratio=min_kc_ratio
    ).to(torch.bool)
    mask_sgl = sglang_funcs["identify_dynamic_mask"](
        q_cent_bh, k_cent_bh, q_sizes_bh, k_sizes_bh,
        top_p=top_p, min_kc_ratio=min_kc_ratio, max_k_clusters_per_q=None
    ).to(torch.bool)

    # Compare masks (they may differ; this is informative)
    same_mask = (mask_svg == mask_sgl).float().mean().item()
    print(f"  mask_svg density: {mask_svg.float().mean().item():.4f}")
    print(f"  mask_sgl density: {mask_sgl.float().mean().item():.4f}")
    print(f"  mask equality ratio: {same_mask:.4f}")

    print("\n[3) SVG permutation (labels → permuted Q/K/V + indices)]")
    q_perm, q_sorted = svg_funcs["permute_tensor_by_labels_triton"](q, q_labels, dim=2)
    k_perm, k_sorted = svg_funcs["permute_tensor_by_labels_triton"](k, k_labels, dim=2)
    v_perm, _ = svg_funcs["permute_tensor_by_labels_triton"](v, k_labels, dim=2, sorted_indices=k_sorted)

    print("\n[4) Run BOTH block-sparse kernels with the SAME intermediates]")

    # (A) Use SVG mask
    print("\n  [4A) Using SVG mask]")
    out_perm_svg = svg_funcs["dynamic_block_sparse_fwd_flashinfer"](
        q_perm, k_perm, v_perm, mask_svg, q_sizes_bh, k_sizes_bh, is_cpu=False
    )
    out_perm_sgl = sglang_funcs["block_sparse_attention"](
        q_perm, k_perm, v_perm, mask_svg, q_sizes_bh, k_sizes_bh
    )
    compare_tensors("Permuted: SGLang vs SVG (SVG mask)", out_perm_sgl, out_perm_svg, rtol=1e-2, atol=1e-2)

    # Inverse permute both ways (use same indices)
    out_svg_restored = svg_funcs["apply_inverse_permutation_triton"](out_perm_svg, q_sorted, dim=2)
    out_sgl_restored = sglang_funcs["inverse_permute"](out_perm_sgl, q_sorted)
    compare_tensors("Restored: SGLang vs SVG (SVG mask)", out_sgl_restored, out_svg_restored, rtol=1e-2, atol=1e-2)

    # (B) Use SGLang mask
    print("\n  [4B) Using SGLang mask]")
    out_perm_svg2 = svg_funcs["dynamic_block_sparse_fwd_flashinfer"](
        q_perm, k_perm, v_perm, mask_sgl, q_sizes_bh, k_sizes_bh, is_cpu=False
    )
    out_perm_sgl2 = sglang_funcs["block_sparse_attention"](
        q_perm, k_perm, v_perm, mask_sgl, q_sizes_bh, k_sizes_bh
    )
    compare_tensors("Permuted: SGLang vs SVG (SGL mask)", out_perm_sgl2, out_perm_svg2, rtol=1e-2, atol=1e-2)

    out_svg2_restored = svg_funcs["apply_inverse_permutation_triton"](out_perm_svg2, q_sorted, dim=2)
    out_sgl2_restored = sglang_funcs["inverse_permute"](out_perm_sgl2, q_sorted)
    compare_tensors("Restored: SGLang vs SVG (SGL mask)", out_sgl2_restored, out_svg2_restored, rtol=1e-2, atol=1e-2)

    print("\n[5) Compare end outputs back in [B, S, H, D] layout (optional)]")
    out_svg_bshd = out_svg_restored.transpose(1, 2).contiguous()
    out_sgl_bshd = out_sgl_restored.transpose(1, 2).contiguous()
    compare_tensors("Final (B,S,H,D): SGLang vs SVG (SVG mask)", out_sgl_bshd, out_svg_bshd, rtol=1e-2, atol=1e-2)

    return True


# ============================================================================
# Test: Block Sparse Attention Only
# ============================================================================

def test_block_sparse_attention_only(sglang_funcs: dict, svg_funcs: dict, device: str = "cuda"):
    """Compare block sparse attention kernels with identical masks."""
    print_header("Test: Block Sparse Attention Kernel Comparison")
    
    torch.manual_seed(42)
    
    # Parameters
    B, H, S, D = 1, 4, 256, 64
    Kq, Kk = 4, 4
    block_size = S // Kq  # 64
    
    # Generate [B, H, S, D] input
    q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
    
    # Uniform cluster sizes (sum to S=256)
    q_cluster_sizes = torch.full((B, H, Kq), block_size, dtype=torch.int32, device=device)
    k_cluster_sizes = torch.full((B, H, Kk), block_size, dtype=torch.int32, device=device)
    
    # All-ones mask (dense)
    block_mask = torch.ones(B, H, Kq, Kk, dtype=torch.bool, device=device)
    
    print("\n[SGLang Block Sparse Attention]")
    try:
        sglang_out = sglang_funcs['block_sparse_attention'](
            q, k, v, block_mask, q_cluster_sizes, k_cluster_sizes
        )
        print_stats("sglang_output", sglang_out)
    except Exception as e:
        print(f"  ✗ Error: {e}")
        sglang_out = None
    
    print("\n[SVG dynamic_block_sparse_fwd_flashinfer]")
    try:
        svg_out = svg_funcs['dynamic_block_sparse_fwd_flashinfer'](
            q, k, v, block_mask, q_cluster_sizes, k_cluster_sizes, is_cpu=False
        )
        print_stats("svg_output", svg_out)
    except Exception as e:
        print(f"  ✗ Error: {e}")
        svg_out = None
    
    print("\n[Dense Attention Reference]")
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    dense_out = torch.matmul(attn, v.float()).to(torch.float16)
    print_stats("dense_output", dense_out)
    
    # Compare
    print_header("Block Sparse Comparison Results")
    
    if sglang_out is not None:
        compare_tensors("SGLang Block Sparse vs Dense", sglang_out, dense_out)
    
    if svg_out is not None:
        compare_tensors("SVG Block Sparse vs Dense", svg_out, dense_out)
    
    if sglang_out is not None and svg_out is not None:
        compare_tensors("SGLang vs SVG Block Sparse", sglang_out, svg_out)
    
    return True


# ============================================================================
# Main
# ============================================================================

def main():
    print_header("SVG2 (SGLang) vs SAP (Sparse-VideoGen) Comparison Test")
    
    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        return
    
    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    
    # Import functions
    print("\nImporting SGLang SVG2...")
    sglang_funcs = import_sglang_svg2()
    if sglang_funcs is None:
        print("Cannot import SGLang SVG2. Exiting.")
        return
    print("  ✓ SGLang SVG2 imported successfully")
    
    print("\nImporting Sparse-VideoGen SAP...")
    svg_funcs = import_svg_sap()
    if svg_funcs is None:
        print("Cannot import Sparse-VideoGen SAP. Some tests will be skipped.")
    else:
        print("  ✓ Sparse-VideoGen SAP imported successfully")
    
    # Run tests
    all_passed = True
    
    if svg_funcs is not None:
        try:
            test_kmeans_comparison(sglang_funcs, svg_funcs, device)
        except Exception as e:
            print(f"K-Means test failed: {e}")
            import traceback
            traceback.print_exc()
    
    if svg_funcs is not None:
        try:
            test_block_sparse_attention_only(sglang_funcs, svg_funcs, device)
        except Exception as e:
            print(f"Block sparse attention test failed: {e}")
            import traceback
            traceback.print_exc()
    
    if svg_funcs is not None:
        try:
            test_shared_intermediates(svg_funcs, sglang_funcs, device)
        except Exception as e:
            print(f"Shared-intermediate test failed: {e}")
            import traceback
            traceback.print_exc()

    if svg_funcs is not None:
        try:
            test_full_attention_comparison(sglang_funcs, svg_funcs, device)
        except Exception as e:
            print(f"Full attention test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print_header("Summary")
    print("All tests completed. Check individual test results above.")


if __name__ == "__main__":
    main()

