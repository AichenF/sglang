// Experimental only: original torch.topk + single small-K finish.
#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>
#include <tvm/ffi/container/tensor.h>
#include <flashinfer/sampling.cuh>
#include <cmath>
#include <limits>
#include "native_topp_clone.cuh"

namespace sglang {
namespace smallk_finish {
// Per-row reduction order matches native Torch PersistentSoftmax.cuh:
// lane + it*32, sequential local iterations, XOR 16/8/4/2/1.
// Native WARP_BATCH=2 rows have independent scalar dependency chains.
template<int K, typename KS>
__global__ __launch_bounds__(1024) void kernel(
    const float* values, const int64_t* indices, const KS* ks,
    float* ps, float* target, float* soft_debug, float* renorm_debug,
    bool use_p, bool skip_mask, bool zero_output) {
  using namespace flashinfer::sampling;
  constexpr int ITERS = K == 50 ? 2 : 4;
  constexpr int VEC = K == 50 ? 2 : 4;
  const int row = blockIdx.x, tx = threadIdx.x;
  __shared__ __align__(16) float prob[K];
  __shared__ __align__(16) float renorm[K];
  if (zero_output) {
    for (int col = tx; col < 152576; col += 1024)
      target[row * 152576 + col] = 0.f;
  }
  if (tx < 32) {
    const int64_t raw_k = static_cast<int64_t>(ks[row]);
    const int64_t actual_k = raw_k < 1 ? 1 : (raw_k > 152576 ? 152576 : raw_k);
    float v[ITERS];
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
      int col = tx + it * 32;
      v[it] = col < K && (skip_mask || col < actual_k)
          ? values[row*K+col] : -std::numeric_limits<float>::infinity();
    }
    float maximum = v[0];
#pragma unroll
    for (int it = 0; it < ITERS; ++it)
      maximum = maximum > v[it] ? maximum : v[it];
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      float other = __shfl_xor_sync(0xffffffff, maximum, offset, 32);
      maximum = maximum < other ? other : maximum;
    }
    float sum = 0.f;
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
      v[it] = std::exp(v[it] - maximum);
      sum += v[it];
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
      sum += __shfl_xor_sync(0xffffffff, sum, offset, 32);
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
      int col = tx + it * 32;
      if (col < K)
        prob[col] = sum == 0 ? std::numeric_limits<float>::quiet_NaN() : v[it] / sum;
    }
  }
  // All 1024 threads participate in the exact native CUB renormalization.
  __syncthreads();
  if (use_p)
    FinishTopPRenormProb<1024, REDUCE_ALGO, VEC, float>(prob, renorm, ps+row, 0.f, K);
  __syncthreads();
  if (tx < K) {
    float result = use_p ? renorm[tx] : prob[tx];
    soft_debug[row*K+tx] = prob[tx];
    renorm_debug[row*K+tx] = result;
    target[row*152576+indices[row*K+tx]] = result;
  }
}
}  // namespace smallk_finish

template<int K, typename KS>
struct SmallKFinish {
  static void run(tvm::ffi::TensorView values, tvm::ffi::TensorView indices,
                  tvm::ffi::TensorView ks, tvm::ffi::TensorView ps,
                  tvm::ffi::TensorView target, tvm::ffi::TensorView soft,
                  tvm::ffi::TensorView renorm, bool use_p, bool skip_mask,
                  bool zero_output) {
    using namespace host;
    const int64_t m = values.size(0);
    RuntimeCheck(m >= 8 && m <= 32 && m % 8 == 0);
    auto device = SymbolicDevice{};
    TensorMatcher({m,K}).with_strides({K,1}).with_dtype<float>()
        .with_device<kDLCUDA>(device).verify(values).verify(soft).verify(renorm);
    TensorMatcher({m,K}).with_strides({K,1}).with_dtype<int64_t>()
        .with_device<kDLCUDA>(device).verify(indices);
    TensorMatcher({m}).with_strides({1}).with_dtype<KS>()
        .template with_device<kDLCUDA>(device).verify(ks);
    TensorMatcher({m}).with_strides({1}).with_dtype<float>()
        .with_device<kDLCUDA>(device).verify(ps);
    TensorMatcher({m,152576}).with_strides({152576,1}).with_dtype<float>()
        .with_device<kDLCUDA>(device).verify(target);
    tvm::ffi::CUDADeviceGuard guard(values.device().device_id);
    constexpr int SHARED = sizeof(flashinfer::sampling::RenormTempStorage<1024,flashinfer::sampling::REDUCE_ALGO>);
    LaunchKernel(dim3(m),dim3(1024),values.device(),SHARED)(smallk_finish::kernel<K,KS>,
        static_cast<const float*>(values.data_ptr()), static_cast<const int64_t*>(indices.data_ptr()),
        static_cast<const KS*>(ks.data_ptr()),static_cast<float*>(ps.data_ptr()),
        static_cast<float*>(target.data_ptr()), static_cast<float*>(soft.data_ptr()),
        static_cast<float*>(renorm.data_ptr()), use_p,skip_mask,zero_output);
  }
};
}  // namespace sglang
