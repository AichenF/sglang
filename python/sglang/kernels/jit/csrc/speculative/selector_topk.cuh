// SPDX-License-Identifier: Apache-2.0
// DFlash2 candidate selector: the local top-K of each shard-logits row + the packed candidate row, in one kernel.
//
// out[r] = [float(v_0) .. float(v_{K-1}) | float(i_0 + id_offset) .. float(i_{K-1} + id_offset)] (fp32, 2K values) where
// (v_j, i_j) are the K largest bf16 logits of source row src(r) = (r / group) * gstride + goff + r % group, ordered by value
// descending, ties by index ascending, and the boundary ties taken by smallest index -- exactly
// pack_candidates(flashinfer.top_k(logits, K, sorted=True, deterministic=True, tie_break=TopKTieBreak.SMALL)).
// Values are compared on the monotonic 16-bit key of the bf16 bits (as the radix top-k does).
//
// One CTA (1024 threads) per output row: element e = j * 8192 + t * 8 + v (16-byte loads), two 8-bit radix histogram
// passes find the K-th key T, keys > T are all taken, keys == T are ranked in index order with one block scan of
// per-chunk tie counts, the K winners are sorted by one warp (bitonic on (key, ~index)).
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>

#include <cstdint>

namespace sglang {
namespace selector_topk {

constexpr int THREADS = 1024;
constexpr int WARPS = THREADS / 32;
constexpr int CHUNK = THREADS * 8;

__device__ __forceinline__ uint32_t key16(uint32_t bits) {
  bits &= 0xFFFFu;
  return (bits & 0x8000u) ? (~bits & 0xFFFFu) : (bits | 0x8000u);
}

template <int K, int NCH>
__global__ void __launch_bounds__(THREADS) topk_pack_kernel(
    const __nv_bfloat16* __restrict__ logits, int64_t row_stride, int n, int group, int gstride, int goff, int id_offset,
    float* __restrict__ out) {
  static_assert(K <= 32 && NCH <= 4, "selector_topk: K <= 32, n <= 4 * 8192");
  __shared__ uint32_t hist[256];
  __shared__ uint32_t sel[2];
  __shared__ unsigned long long wsum[WARPS];
  __shared__ unsigned long long cand[32];
  __shared__ unsigned long long fcand[THREADS];
  __shared__ uint32_t ncand;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int r = blockIdx.x;
  const int src = (r / group) * gstride + goff + r % group;
  const __nv_bfloat16* row = logits + static_cast<int64_t>(src) * row_stride;
  const bool vec = ((reinterpret_cast<uintptr_t>(row) & 15u) == 0);

  // ---- load: 8 consecutive elements per thread per chunk; keys packed 2 per word ----
  uint32_t kw[NCH][4];
#pragma unroll
  for (int j = 0; j < NCH; ++j) {
    const int e0 = j * CHUNK + t * 8;
    if (vec && e0 + 8 <= n) {
      const uint4 u = __ldg(reinterpret_cast<const uint4*>(row + e0));
      kw[j][0] = u.x; kw[j][1] = u.y; kw[j][2] = u.z; kw[j][3] = u.w;
    } else {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        uint32_t lo = 0u, hi = 0u;  // key 0 marks an element past n (never selected: a valid key is >= 1 unless -NaN)
        const int e = e0 + 2 * q;
        if (e < n) lo = __bfloat16_as_ushort(row[e]);
        if (e + 1 < n) hi = __bfloat16_as_ushort(row[e + 1]);
        kw[j][q] = lo | (hi << 16);
      }
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int e = e0 + 2 * q;
      const uint32_t lo = (e < n) ? key16(kw[j][q]) : 0u;
      const uint32_t hi = (e + 1 < n) ? key16(kw[j][q] >> 16) : 0u;
      kw[j][q] = lo | (hi << 16);
    }
  }
  // valid element <=> its index < n; key 0 is reserved for padding, so the histograms skip e >= n explicitly
  auto valid = [&](int j, int v) { return j * CHUNK + t * 8 + v < n; };
  auto keyof = [&](int j, int v) { return (kw[j][v >> 1] >> (16 * (v & 1))) & 0xFFFFu; };

  // ---- fast path: L = the K-th largest per-thread maximum key is a lower bound of the K-th largest key (the K largest
  // maxima are K distinct elements >= L), so the top K are among the keys >= L -- usually a few dozen. They are
  // collected and each is ranked by counting the larger (key, ~index) words: exact, and already in output order. ----
  {
    uint32_t mx = 0u;
#pragma unroll
    for (int j = 0; j < NCH; ++j)
#pragma unroll
      for (int v = 0; v < 8; ++v)
        if (valid(j, v)) mx = max(mx, keyof(j, v));
    // radix select of the K-th largest of the THREADS maxima (one key per thread: no atomic contention to speak of)
    if (t < 256) hist[t] = 0u;
    __syncthreads();
    atomicAdd(&hist[mx >> 8], 1u);
    __syncthreads();
    if (warp == 0) {
      uint32_t c[8], sm = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        c[i] = hist[255 - 8 * lane - i];
        sm += c[i];
      }
      uint32_t inc = sm;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const uint32_t y = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += y;
      }
      const uint32_t ex = inc - sm;
      if (ex < static_cast<uint32_t>(K) && inc >= static_cast<uint32_t>(K)) {
        uint32_t run = ex;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          if (run < static_cast<uint32_t>(K) && run + c[i] >= static_cast<uint32_t>(K)) {
            sel[0] = 255 - 8 * lane - i;
            sel[1] = run;
          }
          run += c[i];
        }
      }
    }
    __syncthreads();
    const uint32_t hb = sel[0], ab = sel[1];
    __syncthreads();
    if (t < 256) hist[t] = 0u;
    __syncthreads();
    if ((mx >> 8) == hb) atomicAdd(&hist[mx & 0xFFu], 1u);
    __syncthreads();
    if (warp == 0) {
      uint32_t c[8], sm = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        c[i] = hist[255 - 8 * lane - i];
        sm += c[i];
      }
      uint32_t inc = sm;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const uint32_t y = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += y;
      }
      const uint32_t ex = inc - sm + ab;
      inc += ab;
      if (ex < static_cast<uint32_t>(K) && inc >= static_cast<uint32_t>(K)) {
        uint32_t run = ex;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          if (run < static_cast<uint32_t>(K) && run + c[i] >= static_cast<uint32_t>(K)) sel[0] = (hb << 8) | (255 - 8 * lane - i);
          run += c[i];
        }
      }
    }
    if (t == 0) ncand = 0u;
    __syncthreads();
    const uint32_t L = sel[0];
    // collect the keys >= L (at most THREADS of them, else the exact histogram path below)
#pragma unroll
    for (int j = 0; j < NCH; ++j)
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        if (valid(j, v) && keyof(j, v) >= L) {
          const uint32_t slot = atomicAdd(&ncand, 1u);
          if (slot < static_cast<uint32_t>(THREADS)) {
            const uint32_t idx = static_cast<uint32_t>(j * CHUNK + t * 8 + v);
            fcand[slot] = (static_cast<unsigned long long>(keyof(j, v)) << 32) | (0xFFFFFFFFull - idx);
          }
        }
      }
    __syncthreads();
    const uint32_t C = ncand;
    if (C <= static_cast<uint32_t>(THREADS)) {
      if (t < static_cast<int>(C)) {
        const unsigned long long x = fcand[t];
        uint32_t rank = 0;
        for (uint32_t i = 0; i < C; ++i) rank += fcand[i] > x ? 1u : 0u;
        if (rank < static_cast<uint32_t>(K)) {
          const uint32_t idx = 0xFFFFFFFFu - static_cast<uint32_t>(x & 0xFFFFFFFFull);
          const uint32_t k = static_cast<uint32_t>(x >> 32);
          const uint32_t bits = (k & 0x8000u) ? (k & 0x7FFFu) : (~k & 0xFFFFu);
          float* o = out + static_cast<int64_t>(r) * (2 * K);
          o[rank] = __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(bits)));
          o[K + rank] = static_cast<float>(static_cast<int>(idx) + id_offset);
        }
      }
      return;  // uniform: C is shared
    }
    __syncthreads();
  }
  // ---- radix pass 1: high byte ----
  if (t < 256) hist[t] = 0u;
  __syncthreads();
#pragma unroll
  for (int j = 0; j < NCH; ++j)
#pragma unroll
    for (int v = 0; v < 8; ++v)
      if (valid(j, v)) atomicAdd(&hist[keyof(j, v) >> 8], 1u);
  __syncthreads();
  if (warp == 0) {
    // lane l owns bins 255 - 8l .. 248 - 8l (descending); find the bin where the running count from the top reaches K
    uint32_t c[8], s = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      c[i] = hist[255 - 8 * lane - i];
      s += c[i];
    }
    uint32_t incl = s;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const uint32_t y = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += y;
    }
    const uint32_t excl = incl - s;
    if (excl < static_cast<uint32_t>(K) && incl >= static_cast<uint32_t>(K)) {
      uint32_t run = excl;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        if (run < static_cast<uint32_t>(K) && run + c[i] >= static_cast<uint32_t>(K)) {
          sel[0] = 255 - 8 * lane - i;  // high byte of the K-th key
          sel[1] = run;                  // keys with a larger high byte
        }
        run += c[i];
      }
    }
  }
  __syncthreads();
  const uint32_t b1 = sel[0], above1 = sel[1];
  __syncthreads();
  // ---- radix pass 2: low byte inside bin b1 ----
  if (t < 256) hist[t] = 0u;
  __syncthreads();
#pragma unroll
  for (int j = 0; j < NCH; ++j)
#pragma unroll
    for (int v = 0; v < 8; ++v)
      if (valid(j, v) && (keyof(j, v) >> 8) == b1) atomicAdd(&hist[keyof(j, v) & 0xFFu], 1u);
  __syncthreads();
  if (warp == 0) {
    uint32_t c[8], s = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      c[i] = hist[255 - 8 * lane - i];
      s += c[i];
    }
    uint32_t incl = s;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const uint32_t y = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += y;
    }
    const uint32_t excl = incl - s + above1;
    incl += above1;
    if (excl < static_cast<uint32_t>(K) && incl >= static_cast<uint32_t>(K)) {
      uint32_t run = excl;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        if (run < static_cast<uint32_t>(K) && run + c[i] >= static_cast<uint32_t>(K)) {
          sel[0] = (b1 << 8) | (255 - 8 * lane - i);  // the K-th key T
          sel[1] = run;                                 // keys > T
        }
        run += c[i];
      }
    }
  }
  if (t == 0) ncand = 0u;
  __syncthreads();
  const uint32_t T = sel[0], ngt = sel[1];
  const uint32_t need = static_cast<uint32_t>(K) - ngt;  // >= 1 ties taken, smallest indices first
  // ---- ties at T ranked in index order: per-chunk counts (<= 8 each) packed in 16-bit fields, one block scan ----
  unsigned long long mine = 0ull;
#pragma unroll
  for (int j = 0; j < NCH; ++j) {
    uint32_t cnt = 0;
#pragma unroll
    for (int v = 0; v < 8; ++v) cnt += (valid(j, v) && keyof(j, v) == T) ? 1u : 0u;
    mine |= static_cast<unsigned long long>(cnt) << (16 * j);
  }
  unsigned long long incl = mine;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const unsigned long long y = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += y;
  }
  if (lane == 31) wsum[warp] = incl;
  __syncthreads();
  if (warp == 0) {
    unsigned long long w = wsum[lane];
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const unsigned long long y = __shfl_up_sync(0xffffffffu, w, o);
      if (lane >= o) w += y;
    }
    wsum[lane] = w;  // inclusive over warps
  }
  __syncthreads();
  const unsigned long long total = wsum[WARPS - 1];
  const unsigned long long before = (warp ? wsum[warp - 1] : 0ull) + (incl - mine);
  uint32_t chunk_base = 0;  // ties in the chunks before j (index order: chunk, then thread, then lane element)
#pragma unroll
  for (int j = 0; j < NCH; ++j) {
    uint32_t rank = chunk_base + static_cast<uint32_t>((before >> (16 * j)) & 0xFFFFull);
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      if (!valid(j, v)) continue;
      const uint32_t k = keyof(j, v);
      bool take = k > T;
      if (k == T) {
        take = rank < need;
        ++rank;
      }
      if (take) {
        const uint32_t idx = static_cast<uint32_t>(j * CHUNK + t * 8 + v);
        const uint32_t slot = atomicAdd(&ncand, 1u);
        cand[slot] = (static_cast<unsigned long long>(k) << 32) | static_cast<unsigned long long>(0xFFFFFFFFu - idx);
      }
    }
    chunk_base += static_cast<uint32_t>((total >> (16 * j)) & 0xFFFFull);
  }
  __syncthreads();
  // ---- sort the K winners: (key desc, index asc) == descending (key << 32 | ~index) ----
  if (warp == 0) {
    unsigned long long x = lane < K ? cand[lane] : 0ull;
#pragma unroll
    for (int size = 2; size <= 32; size <<= 1) {
#pragma unroll
      for (int stride = size >> 1; stride > 0; stride >>= 1) {
        const unsigned long long y = __shfl_xor_sync(0xffffffffu, x, stride);
        const bool up = ((lane & size) == 0);          // descending blocks
        const bool lower = ((lane & stride) == 0);
        const bool keep_max = (lower == up);
        x = keep_max ? (x > y ? x : y) : (x < y ? x : y);
      }
    }
    if (lane < K) {
      const uint32_t idx = 0xFFFFFFFFu - static_cast<uint32_t>(x & 0xFFFFFFFFull);
      const uint32_t k = static_cast<uint32_t>(x >> 32);
      const uint32_t bits = (k & 0x8000u) ? (k & 0x7FFFu) : (~k & 0xFFFFu);
      float* o = out + static_cast<int64_t>(r) * (2 * K);
      o[lane] = __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(bits)));
      o[K + lane] = static_cast<float>(static_cast<int>(idx) + id_offset);
    }
  }
}

template <int K, int NCH>
void launch(const tvm::ffi::TensorView& logits, const tvm::ffi::TensorView& out, int n, int rows, int group, int gstride,
            int goff, int id_offset) {
  using namespace host;
  LaunchKernel(dim3(rows), dim3(THREADS), logits.device())(
      topk_pack_kernel<K, NCH>, static_cast<const __nv_bfloat16*>(logits.data_ptr()), static_cast<int64_t>(logits.stride(0)),
      n, group, gstride, goff, id_offset, static_cast<float*>(out.data_ptr()));
}

}  // namespace selector_topk

struct SelectorTopK {
  // logits bf16 [src_rows, >= n] (row stride in elements, unit column stride); out fp32 [rows, 2K] contiguous.
  static void topk_pack(tvm::ffi::TensorView logits, tvm::ffi::TensorView out, int64_t n, int64_t k, int64_t group,
                        int64_t gstride, int64_t goff, int64_t id_offset) {
    using namespace host;
    using namespace selector_topk;
    const int rows = static_cast<int>(out.size(0));
    RuntimeCheck(logits.ndim() == 2 && logits.stride(1) == 1, "selector_topk: logits [rows, n] with unit column stride");
    RuntimeCheck(out.ndim() == 2 && out.size(1) == 2 * k && out.stride(1) == 1 && out.stride(0) == 2 * k,
                 "selector_topk: out [rows, 2k] contiguous");
    RuntimeCheck(n >= k && n <= 4 * CHUNK && n <= logits.size(1), "selector_topk: n");
    RuntimeCheck(group >= 1, "selector_topk: group");
    if (rows == 0) return;
    const int nch = static_cast<int>((n + CHUNK - 1) / CHUNK);
    const int ni = static_cast<int>(n), g = static_cast<int>(group), gs = static_cast<int>(gstride), go = static_cast<int>(goff),
              off = static_cast<int>(id_offset);
    RuntimeCheck(k == 16, "selector_topk: k == 16");
    switch (nch) {
      case 1: launch<16, 1>(logits, out, ni, rows, g, gs, go, off); break;
      case 2: launch<16, 2>(logits, out, ni, rows, g, gs, go, off); break;
      case 3: launch<16, 3>(logits, out, ni, rows, g, gs, go, off); break;
      default: launch<16, 4>(logits, out, ni, rows, g, gs, go, off); break;
    }
  }
};

}  // namespace sglang
