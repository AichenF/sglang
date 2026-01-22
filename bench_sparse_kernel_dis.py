#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark: 测试 block sparse attention kernel 在相同稀疏度、不同块分布下的性能差异

目的：验证稀疏块分布的均衡性对 kernel 性能的影响
- 均匀分布：每个 Q 簇对应相近数量的 K 簇
- 不均匀分布：某些 Q 簇对应很多 K 簇，某些很少
- 极端不均匀：少数 Q 簇对应大量 K 簇，多数 Q 簇几乎不对应
"""

import math
import time
import torch
import triton

# 导入 kernel 和相关函数
import sys
sys.path.insert(0, '/home/xutingz/workspace/fac/sglang/python')

from sglang.multimodal_gen.runtime.layers.attention.backends.svg2_sparse_attn import (
    _split_k_compute_kernel,
    _split_k_reduce_kernel,
    _kv_index_expansion_kernel,
    block_sparse_attention,
)


def make_uniform_cluster_sizes(total_tokens: int, num_clusters: int, device):
    """均匀分配 cluster sizes"""
    base = total_tokens // num_clusters
    rem = total_tokens % num_clusters
    sizes = torch.full((num_clusters,), base, device=device, dtype=torch.int32)
    if rem > 0:
        sizes[:rem] += 1
    return sizes


def create_uniform_mask(B, H, QC, KC, target_density, device):
    """
    创建均匀分布的 block mask
    每个 Q 簇对应相近数量的 K 簇
    """
    k_per_q = max(1, int(KC * target_density))
    
    mask = torch.zeros(B, H, QC, KC, device=device, dtype=torch.bool)
    for b in range(B):
        for h in range(H):
            for q in range(QC):
                start = (q * k_per_q) % KC
                indices = [(start + i) % KC for i in range(k_per_q)]
                mask[b, h, q, indices] = True
    
    return mask


def create_skewed_mask(B, H, QC, KC, target_density, skew_factor=0.7, device="cuda"):
    """
    创建倾斜分布的 block mask
    前 skew_factor 比例的 Q 簇获得更多的 K 簇
    """
    total_blocks = int(QC * KC * target_density)
    
    mask = torch.zeros(B, H, QC, KC, device=device, dtype=torch.bool)
    
    for b in range(B):
        for h in range(H):
            hot_q_count = max(1, int(QC * (1 - skew_factor)))
            hot_blocks = int(total_blocks * 0.8)
            cold_blocks = total_blocks - hot_blocks
            
            k_per_hot_q = hot_blocks // hot_q_count
            for q in range(hot_q_count):
                k_indices = torch.randperm(KC, device=device)[:k_per_hot_q]
                mask[b, h, q, k_indices] = True
            
            cold_q_count = QC - hot_q_count
            if cold_q_count > 0:
                k_per_cold_q = max(1, cold_blocks // cold_q_count)
                for q in range(hot_q_count, QC):
                    k_indices = torch.randperm(KC, device=device)[:k_per_cold_q]
                    mask[b, h, q, k_indices] = True
    
    return mask


def create_extreme_skewed_mask(B, H, QC, KC, target_density, device="cuda"):
    """
    创建极端不均匀分布的 block mask
    10% 的 Q 簇对应 90% 的块
    """
    total_blocks = int(QC * KC * target_density)
    
    mask = torch.zeros(B, H, QC, KC, device=device, dtype=torch.bool)
    
    for b in range(B):
        for h in range(H):
            hot_q_count = max(1, int(QC * 0.1))
            hot_blocks = int(total_blocks * 0.9)
            cold_blocks = total_blocks - hot_blocks
            
            k_per_hot_q = min(KC, hot_blocks // hot_q_count)
            for q in range(hot_q_count):
                k_indices = torch.randperm(KC, device=device)[:k_per_hot_q]
                mask[b, h, q, k_indices] = True
            
            cold_q_count = QC - hot_q_count
            if cold_q_count > 0:
                k_per_cold_q = max(1, cold_blocks // cold_q_count)
                for q in range(hot_q_count, QC):
                    k_indices = torch.randperm(KC, device=device)[:min(k_per_cold_q, KC)]
                    mask[b, h, q, k_indices] = True
    
    return mask


def create_random_mask(B, H, QC, KC, target_density, device="cuda"):
    """创建随机分布的 block mask"""
    mask = torch.rand(B, H, QC, KC, device=device) < target_density
    for b in range(B):
        for h in range(H):
            for q in range(QC):
                if not mask[b, h, q].any():
                    mask[b, h, q, 0] = True
    return mask


def benchmark_kernel_only(
    q, k, v, 
    block_mask, 
    q_cluster_sizes, 
    k_cluster_sizes,
    split_k=4,
    warmup=5,
    repeat=20
):
    """只 benchmark kernel 部分"""
    B, H, S, D = q.shape
    QC = q_cluster_sizes.shape[-1]
    KC = k_cluster_sizes.shape[-1]
    device = q.device
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(D)
    
    # PLANNING PHASE
    flat_mask = block_mask.reshape(-1, KC)
    sparse_mask = flat_mask.float().to_sparse_csr()
    q_blk_indptr = sparse_mask.crow_indices().int()
    k_blk_indices = sparse_mask.col_indices().int()
    
    qc_offsets = torch.zeros((B, H, QC + 1), device=device, dtype=torch.int32)
    qc_offsets[..., 1:] = torch.cumsum(q_cluster_sizes, dim=-1)
    kc_offsets = torch.zeros((B, H, KC + 1), device=device, dtype=torch.int32)
    kc_offsets[..., 1:] = torch.cumsum(k_cluster_sizes, dim=-1)
    
    active_counts = q_blk_indptr[1:] - q_blk_indptr[:-1]
    q_global_ids = torch.repeat_interleave(
        torch.arange(B*H*QC, device=device, dtype=torch.int32), active_counts
    )
    batch_ids = q_global_ids // (H * QC)
    head_ids = (q_global_ids // QC) % H
    flat_kc_offsets = kc_offsets.reshape(B*H, KC+1)
    bh_ids = batch_ids * H + head_ids
    
    active_k_starts = flat_kc_offsets[bh_ids, k_blk_indices]
    active_k_ends = flat_kc_offsets[bh_ids, k_blk_indices + 1]
    active_k_lengths = active_k_ends - active_k_starts
    global_k_offsets = bh_ids * S + active_k_starts
    
    NNZ = active_k_lengths.numel()
    total_active_k_tokens = int(active_k_lengths.sum().item())
    
    kv_indices = torch.empty(total_active_k_tokens, device=device, dtype=torch.int64)
    write_offsets = torch.zeros(NNZ + 1, device=device, dtype=torch.int32)
    if NNZ > 0:
        write_offsets[1:] = torch.cumsum(active_k_lengths, dim=0)
        max_len = int(active_k_lengths.max().item())
        _kv_index_expansion_kernel[(NNZ,)](
            global_k_offsets, active_k_lengths, write_offsets, kv_indices,
            triton.next_power_of_2(max_len)
        )
    
    flat_q_sizes = q_cluster_sizes.reshape(-1)
    tiles_per_q_blk = (flat_q_sizes + BLOCK_M - 1) // BLOCK_M
    total_q_tiles = int(tiles_per_q_blk.sum().item())
    
    task_to_q_map = torch.repeat_interleave(
        torch.arange(B*H*QC, device=device, dtype=torch.int32), tiles_per_q_blk
    )
    cum_tiles = torch.zeros(B*H*QC + 1, device=device, dtype=torch.int32)
    cum_tiles[1:] = torch.cumsum(tiles_per_q_blk, dim=0)
    task_start_indices = cum_tiles[task_to_q_map]
    task_local_idx = torch.arange(total_q_tiles, device=device, dtype=torch.int32) - task_start_indices
    offset_in_cluster = task_local_idx * BLOCK_M
    
    flat_q_starts = qc_offsets[..., :-1].reshape(-1)
    q_cluster_base = flat_q_starts[task_to_q_map]
    t_batch = task_to_q_map // (H * QC)
    t_head = (task_to_q_map // QC) % H
    
    task_q_global_base = (t_batch * H + t_head) * S + q_cluster_base + offset_in_cluster
    current_q_sizes = flat_q_sizes[task_to_q_map]
    task_q_lens = torch.clamp(current_q_sizes - offset_in_cluster, max=BLOCK_M)
    
    q_cluster_start_block = q_blk_indptr[task_to_q_map].long()
    q_cluster_end_block = q_blk_indptr[task_to_q_map + 1].long()
    task_k_token_starts = write_offsets[q_cluster_start_block]
    task_k_token_ends = write_offsets[q_cluster_end_block]
    
    tmp_acc = torch.empty((total_q_tiles, split_k, BLOCK_M, D), device=device, dtype=torch.float32)
    tmp_m = torch.empty((total_q_tiles, split_k, BLOCK_M), device=device, dtype=torch.float32)
    tmp_l = torch.empty((total_q_tiles, split_k, BLOCK_M), device=device, dtype=torch.float32)
    out = torch.empty_like(q)
    
    grid_compute = (total_q_tiles * split_k,)
    grid_reduce = (total_q_tiles,)
    
    # WARMUP
    for _ in range(warmup):
        _split_k_compute_kernel[grid_compute](
            q, k, v, tmp_acc, tmp_m, tmp_l, kv_indices,
            task_k_token_starts, task_k_token_ends, task_q_global_base, task_q_lens,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            tmp_acc.stride(0), tmp_acc.stride(1), tmp_acc.stride(2), tmp_acc.stride(3),
            tmp_m.stride(0), tmp_m.stride(1),
            1.0 / math.sqrt(D), SPLIT_K=split_k, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D
        )
        _split_k_reduce_kernel[grid_reduce](
            out, tmp_acc, tmp_m, tmp_l, task_q_global_base, task_q_lens,
            out.stride(2), out.stride(3),
            tmp_acc.stride(0), tmp_acc.stride(1), tmp_acc.stride(2), tmp_acc.stride(3),
            tmp_m.stride(0), tmp_m.stride(1), SPLIT_K=split_k, BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D
        )
    torch.cuda.synchronize()
    
    # BENCHMARK
    compute_times, reduce_times, total_times = [], [], []
    
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _split_k_compute_kernel[grid_compute](
            q, k, v, tmp_acc, tmp_m, tmp_l, kv_indices,
            task_k_token_starts, task_k_token_ends, task_q_global_base, task_q_lens,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            tmp_acc.stride(0), tmp_acc.stride(1), tmp_acc.stride(2), tmp_acc.stride(3),
            tmp_m.stride(0), tmp_m.stride(1),
            1.0 / math.sqrt(D), SPLIT_K=split_k, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D
        )
        torch.cuda.synchronize()
        compute_times.append((time.perf_counter() - start) * 1000)
    
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _split_k_reduce_kernel[grid_reduce](
            out, tmp_acc, tmp_m, tmp_l, task_q_global_base, task_q_lens,
            out.stride(2), out.stride(3),
            tmp_acc.stride(0), tmp_acc.stride(1), tmp_acc.stride(2), tmp_acc.stride(3),
            tmp_m.stride(0), tmp_m.stride(1), SPLIT_K=split_k, BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D
        )
        torch.cuda.synchronize()
        reduce_times.append((time.perf_counter() - start) * 1000)
    
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _split_k_compute_kernel[grid_compute](
            q, k, v, tmp_acc, tmp_m, tmp_l, kv_indices,
            task_k_token_starts, task_k_token_ends, task_q_global_base, task_q_lens,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            tmp_acc.stride(0), tmp_acc.stride(1), tmp_acc.stride(2), tmp_acc.stride(3),
            tmp_m.stride(0), tmp_m.stride(1),
            1.0 / math.sqrt(D), SPLIT_K=split_k, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D
        )
        _split_k_reduce_kernel[grid_reduce](
            out, tmp_acc, tmp_m, tmp_l, task_q_global_base, task_q_lens,
            out.stride(2), out.stride(3),
            tmp_acc.stride(0), tmp_acc.stride(1), tmp_acc.stride(2), tmp_acc.stride(3),
            tmp_m.stride(0), tmp_m.stride(1), SPLIT_K=split_k, BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D
        )
        torch.cuda.synchronize()
        total_times.append((time.perf_counter() - start) * 1000)
    
    return {
        'total_q_tiles': total_q_tiles,
        'total_active_k_tokens': total_active_k_tokens,
        'grid_compute': grid_compute[0],
        'grid_reduce': grid_reduce[0],
        'compute_mean': sum(compute_times) / len(compute_times),
        'compute_std': (sum((t - sum(compute_times)/len(compute_times))**2 for t in compute_times) / len(compute_times)) ** 0.5,
        'reduce_mean': sum(reduce_times) / len(reduce_times),
        'reduce_std': (sum((t - sum(reduce_times)/len(reduce_times))**2 for t in reduce_times) / len(reduce_times)) ** 0.5,
        'total_mean': sum(total_times) / len(total_times),
        'total_std': (sum((t - sum(total_times)/len(total_times))**2 for t in total_times) / len(total_times)) ** 0.5,
    }


def compute_mask_statistics(block_mask):
    """计算 mask 的分布统计信息"""
    B, H, QC, KC = block_mask.shape
    k_per_q = block_mask.sum(dim=-1).float()
    
    mean_k = k_per_q.mean().item()
    std_k = k_per_q.std().item()
    
    k_flat = k_per_q.flatten()
    sorted_k = torch.sort(k_flat)[0]
    n = sorted_k.numel()
    cumsum = torch.cumsum(sorted_k, dim=0)
    gini = (2 * torch.sum((torch.arange(1, n+1, device=block_mask.device).float()) * sorted_k) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])
    
    return {
        'density': block_mask.float().mean().item(),
        'mean_k_per_q': mean_k,
        'std_k_per_q': std_k,
        'min_k_per_q': k_per_q.min().item(),
        'max_k_per_q': k_per_q.max().item(),
        'cv': std_k / mean_k if mean_k > 0 else 0,
        'gini': gini.item(),
    }


def run_benchmark(B=1, H=40, S=75600, D=128, QC=300, KC=1000, target_density=0.3, split_k=4, warmup=5, repeat=20, device="cuda"):
    print("=" * 80)
    print(f"Block Sparse Attention Kernel Benchmark")
    print(f"Config: B={B}, H={H}, S={S}, D={D}, QC={QC}, KC={KC}")
    print(f"Target Density: {target_density*100:.1f}%, Split-K: {split_k}")
    print("=" * 80)
    
    torch.manual_seed(42)
    
    q = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
    
    q_cluster_sizes = torch.stack([make_uniform_cluster_sizes(S, QC, device) for _ in range(B * H)]).view(B, H, QC)
    k_cluster_sizes = torch.stack([make_uniform_cluster_sizes(S, KC, device) for _ in range(B * H)]).view(B, H, KC)
    
    masks = {
        'uniform': create_uniform_mask(B, H, QC, KC, target_density, device),
        'random': create_random_mask(B, H, QC, KC, target_density, device),
        'skewed_0.5': create_skewed_mask(B, H, QC, KC, target_density, skew_factor=0.5, device=device),
        'skewed_0.7': create_skewed_mask(B, H, QC, KC, target_density, skew_factor=0.7, device=device),
        'extreme': create_extreme_skewed_mask(B, H, QC, KC, target_density, device),
    }
    
    results = {}
    for name, mask in masks.items():
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")
        
        mask_stats = compute_mask_statistics(mask)
        print(f"  Actual Density: {mask_stats['density']*100:.2f}%")
        print(f"  K per Q: mean={mask_stats['mean_k_per_q']:.1f}, std={mask_stats['std_k_per_q']:.1f}, min={mask_stats['min_k_per_q']:.0f}, max={mask_stats['max_k_per_q']:.0f}")
        print(f"  CV: {mask_stats['cv']:.3f}, Gini: {mask_stats['gini']:.3f}")
        
        stats = benchmark_kernel_only(q, k, v, mask, q_cluster_sizes, k_cluster_sizes, split_k=split_k, warmup=warmup, repeat=repeat)
        
        print(f"  Tiles: {stats['total_q_tiles']}, Active K: {stats['total_active_k_tokens']:,}")
        print(f"  Compute: {stats['compute_mean']:.2f}±{stats['compute_std']:.2f}ms, Reduce: {stats['reduce_mean']:.2f}±{stats['reduce_std']:.2f}ms")
        print(f"  Total: {stats['total_mean']:.2f}±{stats['total_std']:.2f}ms")
        
        results[name] = {**mask_stats, **stats}
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Distribution':<15} {'Density':>8} {'CV':>8} {'Gini':>8} {'Compute':>10} {'Total':>10}")
    print("-" * 65)
    
    baseline = results['uniform']['total_mean']
    for name, r in results.items():
        ratio = r['total_mean'] / baseline if baseline > 0 else 0
        print(f"{name:<15} {r['density']*100:>7.2f}% {r['cv']:>8.3f} {r['gini']:>8.3f} {r['compute_mean']:>8.2f}ms {r['total_mean']:>8.2f}ms ({ratio:.2f}x)")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true", help="Use small config")
    parser.add_argument("--density", type=float, default=0.3)
    parser.add_argument("--split_k", type=int, default=4)
    args = parser.parse_args()
    
    if args.small:
        run_benchmark(B=1, H=8, S=4096, D=128, QC=32, KC=64, target_density=args.density, split_k=args.split_k, warmup=3, repeat=10)
    else:
        run_benchmark(target_density=args.density, split_k=args.split_k)
