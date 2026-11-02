// Mechanically copied from pinned FlashInfer sampling.cuh. See README.txt.
// Original Apache-2.0 license is retained in native_sampling_LICENSE.txt.
#pragma once
namespace flashinfer { namespace sampling {
template <uint32_t BLOCK_THREADS, BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE,
          typename DType>
__device__ __forceinline__ void FinishTopPRenormProb(
    DType* probs, DType* renormed_prob, float* top_p_arr, float top_p_val, uint32_t d) {
  const uint32_t bx = 0, tx = threadIdx.x;
  const uint32_t row_idx = bx;
  float p = top_p_arr == nullptr ? top_p_val : top_p_arr[bx];

  extern __shared__ __align__(alignof(RenormTempStorage<BLOCK_THREADS, REDUCE_ALGO>))
      uint8_t smem_renorm[];
  auto& temp_storage =
      reinterpret_cast<RenormTempStorage<BLOCK_THREADS, REDUCE_ALGO>&>(smem_renorm);
  vec_t<float, VEC_SIZE> probs_vec;

  // Fast-path: when p >= 1.0 (e.g., p == 1.0), perform simple sum and normalization
  if (p >= 1.0f) {
    // Stage A: per-thread float accumulation over assigned lanes (vectorized)
    float thread_sum = 0.0f;
    const uint32_t num_iters = ceil_div(d, BLOCK_THREADS * VEC_SIZE);
    for (uint32_t i = 0; i < num_iters; ++i) {
      probs_vec.fill(0.0f);
      const uint32_t base_idx = (i * BLOCK_THREADS + tx) * VEC_SIZE;
      if (base_idx < d) {
        probs_vec.cast_load(probs + row_idx * d + base_idx);
      }
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        const uint32_t idx = base_idx + j;
        if (idx < d) thread_sum += probs_vec[j];
      }
    }

    // Block reduce (float)
    float row_sum =
        BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
            .Sum(thread_sum);
    // Broadcast via shared
    if (tx == 0) temp_storage.row_sum = row_sum;
    __syncthreads();
    row_sum = temp_storage.row_sum;

    // Guard against zero sum
    const float denom = (row_sum <= 1e-8f) ? 1.0f : row_sum;
    const float normalizer = math::ptx_rcp(denom);

    // Stage B: normalize and store
    for (uint32_t i = 0; i < num_iters; ++i) {
      probs_vec.fill(0.0f);
      const uint32_t base_idx = (i * BLOCK_THREADS + tx) * VEC_SIZE;
      if (base_idx < d) {
        probs_vec.cast_load(probs + row_idx * d + base_idx);
      }
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        const uint32_t idx = base_idx + j;
        float v = probs_vec[j];
        probs_vec[j] = (idx < d) ? (v * normalizer) : 0.0f;
      }
      if (base_idx < d) {
        probs_vec.cast_store(renormed_prob + row_idx * d + base_idx);
      }
    }
    return;  // Exit after fast-path processing
  }

  // Original Top-P renormalization logic
  temp_storage.max_val = 0;
  float max_val = GetMaxValue<VEC_SIZE, BLOCK_THREADS, REDUCE_ALGORITHM,
                              RenormTempStorage<BLOCK_THREADS, REDUCE_ALGORITHM>>(probs, row_idx, d,
                                                                                  temp_storage);

  float low = 0, high = max_val;
  float min_gt_low, max_le_high;
  float sum_low = 1;
  // f(x) = sum(probs[probs > x]), f(x) is non-increasing
  // min_gt_low = min{p \in probs | p > low}, max_le_high = max{p \in probs | p <= high}
  // loop invariant:
  // - f(low) >= p, f(high) < p
  // - f(low) > f(min_gt_low) >= f(max_le_high) == f(high)
  // stopping condition
  // - f(low) >= p, f(min_gt_low) == f(max_le_high) == f(high) < p
  do {
    // Use IEEE-754 arithmetic (no FTZ) to prevent -use_fast_math from flushing
    // subnormal pivots to zero, which would stall the search (#769, #774).
    float pivot_0 = ieee_div(ieee_add(high, ieee_mul(2.f, low)), 3.f);
    float pivot_1 = ieee_div(ieee_add(ieee_mul(2.f, high), low), 3.f);

    float aggregate_gt_pivot_0 = 0, aggregate_gt_pivot_1 = 0;
    min_gt_low = high;
    max_le_high = low;
    float threadlocal_aggregate_gt_pivot_0 = 0;
    float threadlocal_aggregate_gt_pivot_1 = 0;
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(0);
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.cast_load(probs + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      }

      float probs_gt_pivot_0[VEC_SIZE], probs_gt_pivot_1[VEC_SIZE];
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        probs_gt_pivot_0[j] = (probs_vec[j] > pivot_0) ? probs_vec[j] : 0;
        probs_gt_pivot_1[j] = (probs_vec[j] > pivot_1) ? probs_vec[j] : 0;

        if (probs_vec[j] > low && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d) {
          min_gt_low = min(min_gt_low, probs_vec[j]);
        }
        if (probs_vec[j] <= high && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d) {
          max_le_high = max(max_le_high, probs_vec[j]);
        }
        threadlocal_aggregate_gt_pivot_0 += probs_gt_pivot_0[j];
        threadlocal_aggregate_gt_pivot_1 += probs_gt_pivot_1[j];
      }
    }
    aggregate_gt_pivot_0 =
        BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
            .Sum(threadlocal_aggregate_gt_pivot_0);
    __syncthreads();
    aggregate_gt_pivot_1 =
        BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
            .Sum(threadlocal_aggregate_gt_pivot_1);
    __syncthreads();

    min_gt_low = BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                     .Reduce(min_gt_low, MinReduceOp{});
    __syncthreads();
    max_le_high =
        BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
            .Reduce(max_le_high, MaxReduceOp{});
    if (tx == 0) {
      temp_storage.block_aggregate.values[0] = aggregate_gt_pivot_0;
      temp_storage.block_aggregate.values[1] = aggregate_gt_pivot_1;
      temp_storage.min_val = min_gt_low;
      temp_storage.max_val = max_le_high;
    }
    __syncthreads();
    aggregate_gt_pivot_0 = temp_storage.block_aggregate.values[0];
    aggregate_gt_pivot_1 = temp_storage.block_aggregate.values[1];
    min_gt_low = temp_storage.min_val;
    max_le_high = temp_storage.max_val;

    if (aggregate_gt_pivot_1 >= p) {
      low = pivot_1;
      sum_low = aggregate_gt_pivot_1;
    } else if (aggregate_gt_pivot_0 >= p) {
      low = pivot_0;
      high = min(pivot_1, max_le_high);
      sum_low = aggregate_gt_pivot_0;
    } else {
      high = min(pivot_0, max_le_high);
    }
  } while (min_gt_low < max_le_high && nextafterf(min_gt_low, max_le_high) < max_le_high);

  float normalizer = math::ptx_rcp(max(sum_low, 1e-8));

  // normalize
#pragma unroll 2
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    probs_vec.fill(0);
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      probs_vec.cast_load(probs + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      probs_vec[j] = (probs_vec[j] > low) ? probs_vec[j] * normalizer : 0;
    }
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      probs_vec.cast_store(renormed_prob + row_idx * d + i * BLOCK_THREADS * VEC_SIZE +
                           tx * VEC_SIZE);
    }
  }
}

}}
