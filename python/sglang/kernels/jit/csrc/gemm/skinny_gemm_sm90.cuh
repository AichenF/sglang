// Skinny (M <= 32) FP8 GEMM for Hopper decode: out[M, N] (bf16) = x[M, K] (e4m3) . w[N, K]^T (e4m3).
//
// Why: at M = 32 the decode GEMMs are weight streams (qkv_proj 20.8 MB, draft projections 12.6-25.2 MB per rank).
// DeepGEMM's 1D2D kernel picks BLOCK 32x32 for M = 32 and walks the 48 K-blocks of a tile serially (wgmma wait +
// promotion per K-block, ~0.3 us each) on 106 of 132 SMs -> ~13 GB/s per SM, 10-12 us for a 4.6 us weight stream.
// CUTLASS fp8_scaled_mm (draft) tiles N by 64 and uses 36/64/96 CTAs.
//
// Design (swap-AB, cluster split-K):
//  * the weight is the wgmma A operand (64 rows per m64 wgmma, 2 per 128-row block), the <= 32 tokens are the
//    wgmma N dimension (m64n32k32 e4m3, A and B K-major from 128B-swizzled smem written by TMA);
//  * one thread-block cluster of S CTAs per 128-row block; CTA rank r streams K-blocks [r*KB/S, (r+1)*KB/S)
//    (16 KB weight units). Two consumer warpgroups take alternate units (a warpgroup's wgmma -> wait ->
//    promote chain is latency-bound at N = 32; two of them keep up with the HBM rate);
//  * reduction without global round trips: warpgroup 2 hands its fp32 accumulators to warpgroup 1 through smem,
//    then rank 0 of the cluster sums the S CTA partials over DSMEM in rank order -> deterministic;
//  * weights do not depend on the previous kernel: with PDL the producer issues the TMA loads of its first
//    `depth` units BEFORE griddepcontrol.wait, the activation tiles after it;
//  * scale modes: kMode 0 = DeepGEMM block scales (w: [ceil(N/128), K/128] fp32, x: per token per 128-K group,
//    column-major [K/128][sx_ld]), promoted per K-block exactly like DeepGEMM (acc += partial * (sw * sx));
//    kMode 1 = per-row weight scale (sw_stride 0 = per-tensor) x per-token activation scale, applied once in
//    the epilogue (the wgmma chain accumulates the whole K-slice); kMode 2 = stream only (diagnostic).
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <map>
#include <tuple>
#include <type_traits>

#ifdef SKINNY_STAMP
#define SKINNY_NS skinny_v_stamp
#else
#define SKINNY_NS skinny_v_plain
#endif

namespace sglang {
inline namespace SKINNY_NS {
namespace skinny_gemm {

constexpr int BN = 128;              // weight rows per row block
constexpr int BK = 128;              // K (bytes = elements) per unit
constexpr int MT = 32;               // tokens = wgmma N
constexpr int kStages = 6;
constexpr int kMaxCluster = 7;
constexpr int kWBytes = BN * BK;     // 16384
constexpr int kXBytes = MT * BK;     // 4096
constexpr int kConsumerWGs = 2;
constexpr int kThreads = 128 * (1 + kConsumerWGs);  // warpgroup 0: producer (warp 0 lane 0), 1..2: consumers
constexpr int kFragFloats = 16;      // m64n32 fp32 accumulator registers per thread
constexpr int kPartFloats = 2 * 128 * kFragFloats;  // one 128 x 32 fp32 partial in fragment order (16 KB)
// Reduce-scatter over the cluster: the 128 rows are 8 groups of 16 (one per consumer warp); rank r finalizes groups
// [8r/S, 8(r+1)/S) and receives each from the S-1 other ranks (2 KB per group and peer). Max over S <= 7: 24 KB.
constexpr int kGroupFloats = 16 * MT;          // 512
constexpr int kRecvBytes = 24 * 1024;
constexpr int kOutStage = 8 * MT * 16 * 2;     // per-warp [32 tok][16 rows] bf16 staging, 8 warps (8 KB)
constexpr int kSmemBytes =
    1024 + kStages * (kWBytes + kXBytes) + kRecvBytes + kOutStage + kStages * MT * 4 + (2 * kStages + 1) * 8 + 64;

struct Params {
  const float* sw;            // mode 0: [NB][KB]; mode 1: [N] (sw_stride 1) or [1] (sw_stride 0)
  const float* sx;            // mode 0: [KB][sx_ld] (column-major per-token-group scales); mode 1: [M]
  __nv_bfloat16* out;         // [M][out_ld]
  int64_t out_ld;
  int M, N, K, NB, KB, S;
  int sx_ld;
  int sw_stride;
  int depth;                  // max weight units in flight per CTA (<= kStages)
  int warm;                   // unused (kept for the ABI)
  int preload;                // weight units issued before griddepcontrol.wait (<= depth)
  unsigned long long* stamps; // SKINNY_STAMP builds: [grid][16] globaltimer ns
  int l2pre;                  // further weight units L2-prefetched before griddepcontrol.wait (-1 = all the rest)
  // reduce-scatter epilogue (dflash_tp_fused.cuh tp_arnq "prepushed" mode): instead of storing out[t][n], push the
  // bf16 partial into the owner rank's RS slot (owner = n / 768) of the CUDA-IPC receive buffers, Lamport-marked
  // (+0.0 -> -0.0). rs_calls = the per-token call counters (phase = calls[t] & 1), nullptr = plain store.
  const int* rs_calls;
  uint8_t* rs_peer[8];
  int rs_rank;
  // goal11-post: L2 prefetch of the NEXT kernel's weights (nx_bytes from nx_ptr, split over the CTAs), issued by the
  // producer right after its last weight TMA so HBM keeps streaming through this GEMM's tail (0 = off)
  const uint8_t* nx_ptr;
  int64_t nx_bytes;
};
// tp_arnq receive-buffer layout (keep in sync with distributed/dflash_tp_fused.cuh)
constexpr int RS_R = 8, RS_SL = 768, RS_MAXT = 32;
constexpr int64_t RS_SLOT_BYTES = RS_SL * 2;

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
#ifdef SKINNY_STAMP
#define STAMP(k) do { if (p.stamps) p.stamps[blockIdx.x * 16 + (k)] = gtimer(); } while (0)
#define CLK(v) long long v = clock64()
#define CSTAMP(k, v) do { if (p.stamps) p.stamps[blockIdx.x * 16 + (k)] = (v); } while (0)
#else
#define STAMP(k) do {} while (0)
#define CLK(v) do {} while (0)
#define CSTAMP(k, v) do {} while (0)
#endif

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra DONE_%=;\n"
      "bra WAIT_%=;\n"
      "DONE_%=:\n"
      "}\n" ::"r"(smem_u32(b)),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void tma_prefetch_l2_2d(const CUtensorMap* map, int c0, int c1) {
  asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" ::"l"(map), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void tma_load_2d(
    const CUtensorMap* map, void* dst, uint64_t* bar, int c0, int c1, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(smem_u32(dst)),
      "l"(map),
      "r"(smem_u32(bar)),
      "r"(c0),
      "r"(c1),
      "l"(policy)
      : "memory");
}
__device__ __forceinline__ void bulk_load(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
          smem_u32(dst)),
      "l"(src),
      "r"(bytes),
      "r"(smem_u32(bar))
      : "memory");
}
__device__ __forceinline__ void bulk_s2s_cluster(uint32_t dst_cluster, const void* src, uint32_t bytes, uint32_t mbar_cluster) {
  asm volatile(
      "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst_cluster),
      "r"(smem_u32(src)),
      "r"(bytes),
      "r"(mbar_cluster)
      : "memory");
}
__device__ __forceinline__ void cluster_arrive_relaxed_wait() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\nbarrier.cluster.wait.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void st_async_v4(uint32_t addr_cluster, float4 v, uint32_t mbar_cluster) {
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(addr_cluster),
      "f"(v.x),
      "f"(v.y),
      "f"(v.z),
      "f"(v.w),
      "r"(mbar_cluster)
      : "memory");
}
__device__ __forceinline__ uint64_t make_desc(const void* p) {
  // K-major, 128B swizzle, 8-row x 128 B atoms (SBO 1024 B); LBO unused for swizzled K-major.
  const uint64_t a = smem_u32(p);
  return ((a & 0x3FFFFull) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}
__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}
template <int N>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
}
__device__ __forceinline__ void wgmma_m64n32k32_e4m3(float* d, uint64_t da, uint64_t db, int scale_d) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %18, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, p, 1, 1;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ void named_bar_sync(int id, int n) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ uint32_t cluster_id_x() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(r));
  return r;
}
__device__ __forceinline__ uint32_t mapa(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
__device__ __forceinline__ float4 ld_dsmem_v4(uint32_t addr) {
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "r"(addr)
               : "memory");
  return v;
}

// Accumulator fragment (m64nN, per warpgroup thread): register r -> row (within 64) and token column.
__device__ __forceinline__ int frag_row(int ctid, int r) {
  return (ctid >> 5) * 16 + ((ctid & 31) >> 2) + ((r >> 1) & 1) * 8;
}
__device__ __forceinline__ int frag_col(int ctid, int r) {
  return (r >> 2) * 8 + (ctid & 3) * 2 + (r & 1);
}

template <int kMode>
__device__ __forceinline__ void epilogue_store(
    const Params& p, const float (&v)[2][kFragFloats], int rb, int ctid, __nv_bfloat16* sOut) {
  // v: final fp32 sums of this 128-row block (fragment order) -> bf16 out rows [rb*128, rb*128+128).
#pragma unroll
  for (int h = 0; h < 2; ++h) {
#pragma unroll
    for (int r = 0; r < kFragFloats; ++r) {
      const int row = h * 64 + frag_row(ctid, r);
      const int col = frag_col(ctid, r);
      float x = v[h][r];
      if constexpr (kMode == 1) {
        const int grow = rb * BN + row;
        const float swv = (grow < p.N) ? __ldg(p.sw + static_cast<int64_t>(grow) * p.sw_stride) : 0.f;
        const float sxv = (col < p.M) ? __ldg(p.sx + col) : 0.f;
        x = x * swv * sxv;
      }
      sOut[col * BN + row] = __float2bfloat16_rn(x);
    }
  }
  named_bar_sync(2, 128);
  // 32 tokens x 16 chunks of 8 bf16
#pragma unroll
  for (int i = 0; i < (MT * BN / 8) / 128; ++i) {
    const int idx = i * 128 + ctid;
    const int tok = idx >> 4;
    const int ch = idx & 15;
    const int grow = rb * BN + ch * 8;
    if (tok < p.M && grow < p.N) {
      *reinterpret_cast<uint4*>(p.out + static_cast<int64_t>(tok) * p.out_ld + grow) =
          *reinterpret_cast<const uint4*>(sOut + tok * BN + ch * 8);
    }
  }
}

template <int kMode, bool kUsePDL>
__global__ void __launch_bounds__(kThreads, 1) skinny_gemm_kernel(
    const __grid_constant__ CUtensorMap tmW, const __grid_constant__ CUtensorMap tmX, const __grid_constant__ Params p) {
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint8_t* sW = smem;
  uint8_t* sX = sW + kStages * kWBytes;
  float* sRecv = reinterpret_cast<float*>(sX + kStages * kXBytes);
  __nv_bfloat16* sOutAll = reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<uint8_t*>(sRecv) + kRecvBytes);
  float* sSX = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(sOutAll) + kOutStage);
  uint64_t* full = reinterpret_cast<uint64_t*>(sSX + kStages * MT);
  uint64_t* empty = full + kStages;
  uint64_t* red_bar = empty + kStages;

  const int rank = static_cast<int>(cluster_rank());
  const int S = p.S;
  const int rb = static_cast<int>(cluster_id_x());
  const int kb_lo = (rank * p.KB) / S;
  const int kb_hi = ((rank + 1) * p.KB) / S;
  const int n = kb_hi - kb_lo;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  // row groups finalized by this rank
  const int g_lo = (rank * 8) / S;
  const int g_hi = ((rank + 1) * 8) / S;

  if (tid == 0) STAMP(0);
  if (tid == 0) {
    for (int s = 0; s < kStages; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], 8);
    }
    mbar_init(red_bar, 1);
    if (S > 1) mbar_expect_tx(red_bar, static_cast<uint32_t>((g_hi - g_lo) * (S - 1) * kGroupFloats * 4));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmW) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tmX) : "memory");
  }
  __syncthreads();
  // every rank's reduction barrier must be initialized before any peer stores into it: arrive now (after the
  // init), wait only right before the first remote store / at exit (split cluster barrier, off the start path)
  if (S > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

  if (warp == 0) {
    if (lane == 0) {
      uint64_t pol_w, pol_x;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol_w));
      asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol_x));
      const uint32_t sx_bytes = (kMode == 0) ? static_cast<uint32_t>(p.sx_ld) * 4u : 0u;
      const uint32_t stage_tx = kWBytes + kXBytes + sx_bytes;
      const int depth = p.depth;
      const int pre = n < depth ? n : depth;
      // weights first: independent of the previous kernel (p.preload of them before griddepcontrol.wait; under the
      // fused MoE kernel's TP tail an early weight stream measurably slows that tail -> knob)
      const int early = pre < p.preload ? pre : p.preload;
      for (int i = 0; i < early; ++i) {
        mbar_expect_tx(&full[i], stage_tx);
        tma_load_2d(&tmW, sW + i * kWBytes, &full[i], (kb_lo + i) * BK, rb * BN, pol_w);
      }
      if (p.l2pre != 0) {
        // the units that do not fit the smem ring: pull them into L2 while the previous kernel (NVLink-bound TP
        // reduce) leaves HBM idle; their TMA loads after the wait then hit L2
        const int hi = p.l2pre < 0 ? n : (early + p.l2pre < n ? early + p.l2pre : n);
        for (int i = (early > pre ? early : pre); i < hi; ++i) tma_prefetch_l2_2d(&tmW, (kb_lo + i) * BK, rb * BN);
      }
      STAMP(1);
      device::PDLWaitPrimary<kUsePDL>();
      STAMP(2);
      for (int i = early; i < pre; ++i) {
        mbar_expect_tx(&full[i], stage_tx);
        tma_load_2d(&tmW, sW + i * kWBytes, &full[i], (kb_lo + i) * BK, rb * BN, pol_w);
      }
      for (int i = 0; i < pre; ++i) {
        const int kb = kb_lo + i;
        tma_load_2d(&tmX, sX + i * kXBytes, &full[i], kb * BK, 0, pol_x);
        if constexpr (kMode == 0) bulk_load(sSX + i * MT, p.sx + static_cast<int64_t>(kb) * p.sx_ld, sx_bytes, &full[i]);
      }
      for (int i = pre; i < n; ++i) {
        const int q = i - depth;  // its release frees a slot (stage i % kStages was released by unit i - kStages <= q)
        mbar_wait(&empty[q % kStages], (q / kStages) & 1);
        const int s = i % kStages;
        const int kb = kb_lo + i;
        mbar_expect_tx(&full[s], stage_tx);
        tma_load_2d(&tmW, sW + s * kWBytes, &full[s], kb * BK, rb * BN, pol_w);
        tma_load_2d(&tmX, sX + s * kXBytes, &full[s], kb * BK, 0, pol_x);
        if constexpr (kMode == 0) bulk_load(sSX + s * MT, p.sx + static_cast<int64_t>(kb) * p.sx_ld, sx_bytes, &full[s]);
      }
      if (p.nx_bytes > 0) {
        const int64_t per = ((p.nx_bytes + gridDim.x - 1) / gridDim.x + 15) / 16 * 16;
        const int64_t beg = per * blockIdx.x;
        const int64_t end = beg + per < p.nx_bytes ? beg + per : p.nx_bytes;
        for (int64_t o = beg; o < end; o += 32768) {
          const uint32_t sz = static_cast<uint32_t>(end - o < 32768 ? end - o : 32768);
          asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p.nx_ptr + o), "r"(sz) : "memory");
        }
      }
    }
  } else if (warp >= 4) {
    // consumer warpgroup wg owns weight rows [wg*64, wg*64+64) of every unit
    const int wg = (warp >> 2) - 1;
    const int ctid = tid & 127;
    const int wq = ctid >> 5;                // warp within the warpgroup
    const int g = wg * 4 + wq;               // row group (16 rows)
    const int frow = wq * 16 + (lane >> 2);  // + i*8 (within the warpgroup's 64 rows)
    const int fcol = (lane & 3) * 2;         // + n8*8 + j
    device::PDLWaitPrimary<kUsePDL>();
    const bool owner = (g >= g_lo && g < g_hi);
    float srow[2] = {1.f, 1.f};
    float scol[8] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    if constexpr (kMode == 1) {
      if (owner) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int grow = rb * BN + wg * 64 + frow + i * 8;
          srow[i] = (grow < p.N) ? __ldg(p.sw + static_cast<int64_t>(grow) * p.sw_stride) : 0.f;
        }
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const int col = (q >> 1) * 8 + fcol + (q & 1);
          scol[q] = (col < p.M) ? __ldg(p.sx + col) : 0.f;
        }
      }
    }
    float acc[kFragFloats];
    float tmp[kFragFloats];
#pragma unroll
    for (int r = 0; r < kFragFloats; ++r) acc[r] = 0.f;
    if constexpr (kMode == 1) {
      // per-channel mode: one accumulator chain over the whole K-slice; one unit's wgmma group stays in flight while
      // waiting for the next unit (wait_group 1), a unit's slot is released once its group retired. (Four independent
      // accumulator chains measured no faster: at N = 32 the SS wgmma is smem-bound, ~0.25 us per 16 KB unit.)
      for (int i = 0; i < n; ++i) {
        const int s = i % kStages;
        mbar_wait(&full[s], (i / kStages) & 1);
        if (ctid == 0 && wg == 0 && i == 0) STAMP(3);
        if (ctid == 0 && wg == 0 && i > 0 && i < 7) STAMP(7 + i);
        const uint64_t da = make_desc(sW + s * kWBytes) + wg * 512;
        const uint64_t db = make_desc(sX + s * kXBytes);
        wgmma_fence();
#pragma unroll
        for (int k = 0; k < 4; ++k) wgmma_m64n32k32_e4m3(acc, da + k * 2, db + k * 2, (k > 0) || (i > 0));
        wgmma_commit();
        if (i > 0) {
          wgmma_wait<1>();
          __syncwarp();
          if (lane == 0) mbar_arrive(&empty[(i - 1) % kStages]);
        }
      }
      wgmma_wait<0>();
      __syncwarp();
      if (lane == 0 && n > 0) mbar_arrive(&empty[(n - 1) % kStages]);
    }
    for (int i = 0; kMode != 1 && i < n; ++i) {
      const int s = i % kStages;
      const int kb = kb_lo + i;
      float swv = 0.f;
      if constexpr (kMode == 0) swv = __ldg(p.sw + static_cast<int64_t>(rb) * p.KB + kb);
      mbar_wait(&full[s], (i / kStages) & 1);
      if (ctid == 0 && wg == 0 && i == 0) STAMP(3);
      if constexpr (kMode != 2) {
        const uint64_t da = make_desc(sW + s * kWBytes) + wg * 512;
        const uint64_t db = make_desc(sX + s * kXBytes);
        wgmma_fence();
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          if constexpr (kMode == 0) {
            wgmma_m64n32k32_e4m3(tmp, da + k * 2, db + k * 2, k > 0);
          } else {
            wgmma_m64n32k32_e4m3(acc, da + k * 2, db + k * 2, (k > 0) || (i > 0));
          }
        }
        wgmma_commit();
        wgmma_wait<0>();
      }
      if constexpr (kMode == 0) {
        float sxv[8];
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const int col = (q >> 1) * 8 + fcol + (q & 1);
          sxv[q] = (col < p.M) ? sSX[s * MT + col] : 0.f;
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty[s]);
#pragma unroll
        for (int r = 0; r < kFragFloats; ++r) acc[r] += tmp[r] * (swv * sxv[(r >> 2) * 2 + (r & 1)]);
      } else {
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty[s]);
      }
    }
    if (ctid == 0) STAMP(4 + wg);
    device::PDLTriggerSecondary<kUsePDL>();
    if constexpr (kMode != 2) {
      if (!owner) {
        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
        // send this warp's 16-row partial to the owner rank: recv[(g - g_lo(o)) * (S-1) + slot(rank)][lane][16]
        int o = 0;
        while (((o + 1) * 8) / S <= g) ++o;
        const int og_lo = (o * 8) / S;
        const int slot = rank < o ? rank : rank - 1;
        const uint32_t off = static_cast<uint32_t>((((g - og_lo) * (S - 1) + slot) * 32 + lane) * kFragFloats * 4);
        const uint32_t dst = mapa(smem_u32(sRecv), static_cast<uint32_t>(o)) + off;
        const uint32_t bar = mapa(smem_u32(red_bar), static_cast<uint32_t>(o));
#pragma unroll
        for (int r = 0; r < kFragFloats; r += 4)
          st_async_v4(dst + r * 4, make_float4(acc[r], acc[r + 1], acc[r + 2], acc[r + 3]), bar);
      } else {
        if (S > 1) {
          mbar_wait(red_bar, 0);
          if (lane == 0 && wq == 0 && wg == 0) STAMP(6);
          // sum in rank order: rank 0 .. S-1 (own partial at position `rank`)
          float tot[kFragFloats];
#pragma unroll
          for (int r = 0; r < kFragFloats; ++r) tot[r] = 0.f;
#pragma unroll 1
          for (int src = 0; src < S; ++src) {
            if (src == rank) {
#pragma unroll
              for (int r = 0; r < kFragFloats; ++r) tot[r] += acc[r];
            } else {
              const int slot = src < rank ? src : src - 1;
              const float* rp = sRecv + (((g - g_lo) * (S - 1) + slot) * 32 + lane) * kFragFloats;
#pragma unroll
              for (int r = 0; r < kFragFloats; r += 4) {
                const float4 v = *reinterpret_cast<const float4*>(rp + r);
                tot[r] += v.x;
                tot[r + 1] += v.y;
                tot[r + 2] += v.z;
                tot[r + 3] += v.w;
              }
            }
          }
#pragma unroll
          for (int r = 0; r < kFragFloats; ++r) acc[r] = tot[r];
        }
        // ---- epilogue (warp-local): (scale) -> bf16 -> [token][16 rows] staging -> 32-byte row segments ----
        __nv_bfloat16* sOut = sOutAll + (wg * 4 + wq) * (MT * 16);
#pragma unroll
        for (int r = 0; r < kFragFloats; ++r) {
          const int row = (lane >> 2) + ((r >> 1) & 1) * 8;          // within the group's 16 rows
          const int col = (r >> 2) * 8 + fcol + (r & 1);
          float x = acc[r];
          if constexpr (kMode == 1) x = x * srow[(r >> 1) & 1] * scol[(r >> 2) * 2 + (r & 1)];
          sOut[col * 16 + row] = __float2bfloat16_rn(x);
        }
        __syncwarp();
        const int grow = rb * BN + wg * 64 + wq * 16;
        if (lane < p.M && grow < p.N) {
          const uint4* src = reinterpret_cast<const uint4*>(sOut + lane * 16);
          if (p.rs_calls != nullptr) {
            const int phase = p.rs_calls[lane] & 1;
            const int q = grow / RS_SL;
            uint8_t* dst = p.rs_peer[q] + ((static_cast<int64_t>(phase) * RS_MAXT + lane) * RS_R + p.rs_rank) * RS_SLOT_BYTES +
                           static_cast<int64_t>(grow - q * RS_SL) * 2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              uint4 v = src[h];
              v.x = v.x == 0u ? 0x80000000u : v.x;
              v.y = v.y == 0u ? 0x80000000u : v.y;
              v.z = v.z == 0u ? 0x80000000u : v.z;
              v.w = v.w == 0u ? 0x80000000u : v.w;
              asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(dst + h * 16), "r"(v.x), "r"(v.y), "r"(v.z),
                           "r"(v.w)
                           : "memory");
            }
          } else {
            uint4* dstp = reinterpret_cast<uint4*>(p.out + static_cast<int64_t>(lane) * p.out_ld + grow);
            dstp[0] = src[0];
            dstp[1] = src[1];
          }
        }
        if (lane == 0 && wq == 0 && wg == 0) STAMP(7);
        if (S > 1) asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
      }
    } else {
      if (S > 1) asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    }
  }
  if (warp < 4 && S > 1) {
    __syncwarp();
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
  }
  if (tid == 0) STAMP(15);
}

inline CUtensorMap make_map_2d(const void* base, uint64_t inner, uint64_t outer, uint64_t row_bytes, uint32_t box_inner, uint32_t box_outer) {
  CUtensorMap map;
  uint64_t dim[2] = {inner, outer};
  uint64_t stride[1] = {row_bytes};
  uint32_t box[2] = {box_inner, box_outer};
  uint32_t elem_stride[2] = {1, 1};
  const CUresult rc = cuTensorMapEncodeTiled(
      &map,
      CU_TENSOR_MAP_DATA_TYPE_UINT8,
      2,
      const_cast<void*>(base),
      dim,
      stride,
      box,
      elem_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  host::RuntimeCheck(rc == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed");
  return map;
}

// Largest cluster size S <= 8 (and <= KB) whose NB clusters are all co-resident, with NB * S <= num_sms.
template <typename Kern>
inline int pick_cluster(Kern kernel, int NB, int KB, int num_sms, int forced) {
  if (forced > 0) return forced;
  static std::map<std::tuple<int, int, int>, int> cache;
  const auto key = std::make_tuple(NB, KB, num_sms);
  if (auto it = cache.find(key); it != cache.end()) return it->second;
  int best = 1;
  for (int s = 1; s <= kMaxCluster && s <= KB; ++s) {
    if (NB * s > num_sms) break;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(NB * s);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = kSmemBytes;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = static_cast<unsigned>(s);
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    int clusters = 0;
    if (cudaOccupancyMaxActiveClusters(&clusters, kernel, &cfg) != cudaSuccess) {
      (void)cudaGetLastError();
      break;
    }
    if (clusters >= NB) best = s;
  }
  cache[key] = best;
  return best;
}

}  // namespace skinny_gemm

template <int kMode, bool kUsePDL>
struct SkinnyGemmKernel {
  // x: [M, K] e4m3 (row stride multiple of 16 B), w: [N, K] e4m3 contiguous rows, out: [M, >= N] bf16.
  // mode 0: sx [K/128, sx_ld] fp32 viewed 2-D (column-major scales: x row t, group g at sx[g * sx_ld + t]),
  //         sw [ceil(N/128), K/128] fp32 contiguous.
  // mode 1: sx [M] fp32, sw [N] (per-row) or [1] (per-tensor) fp32.
  // cluster: 0 = automatic (largest co-resident cluster size, NB * S <= num_sms).
  static void run(
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView w,
      tvm::ffi::TensorView sx,
      tvm::ffi::TensorView sw,
      tvm::ffi::TensorView out,
      int64_t num_sms,
      int64_t cluster,
      int64_t depth,
      int64_t warm,
      int64_t preload,
      int64_t stamps_addr,
      int64_t l2pre,
      tvm::ffi::TensorView rs_peers,
      int64_t rs_calls_addr,
      int64_t rs_rank,
      int64_t smem_pad,
      int64_t nx_addr,
      int64_t nx_bytes) {
    using namespace host;
    using namespace skinny_gemm;
    RuntimeCheck(depth >= 1 && depth <= kStages);
    RuntimeCheck(x.ndim() == 2 && w.ndim() == 2 && out.ndim() == 2);
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(x.size(1));
    const int N = static_cast<int>(w.size(0));
    RuntimeCheck(M >= 1 && M <= MT, "skinny_gemm: M must be in [1, 32]");
    RuntimeCheck(w.size(1) == K && K % BK == 0 && N % 16 == 0);
    const bool rs = rs_calls_addr != 0;
    RuntimeCheck(x.dtype().bits == 8 && w.dtype().bits == 8 && (rs || out.dtype().bits == 16));
    RuntimeCheck(x.stride(1) == 1 && w.stride(1) == 1);
    RuntimeCheck((x.stride(0) % 16) == 0 && w.stride(0) == K);
    if (!rs) {
      RuntimeCheck(out.stride(1) == 1 && (out.stride(0) % 8) == 0);
      RuntimeCheck(out.size(0) >= M && out.size(1) >= N);
      RuntimeCheck(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0);
    } else {
      RuntimeCheck(kMode == 1 && N == RS_R * RS_SL && rs_peers.numel() == RS_R && rs_rank >= 0 && rs_rank < RS_R,
                   "skinny_gemm: reduce-scatter epilogue needs mode 1 and N = 6144 (TP8 slices of 768)");
    }
    RuntimeCheck(sx.dtype().bits == 32 && sw.dtype().bits == 32);
    const int NB = (N + BN - 1) / BN;
    const int KB = K / BK;
    int sx_ld = 0, sw_stride = 0;
    if constexpr (kMode == 0) {
      RuntimeCheck(sx.ndim() == 2 && sx.size(0) == KB && sx.stride(1) == 1 && sx.size(1) >= M);
      sx_ld = static_cast<int>(sx.stride(0));
      RuntimeCheck(sx_ld % 4 == 0 && sx_ld <= MT && reinterpret_cast<uintptr_t>(sx.data_ptr()) % 16 == 0);
      RuntimeCheck(sw.ndim() == 2 && sw.size(0) == NB && sw.size(1) == KB && sw.stride(0) == KB && sw.stride(1) == 1);
    } else {
      RuntimeCheck(sx.numel() >= M && sx.stride(0) == 1);
      RuntimeCheck(sw.numel() == N || sw.numel() == 1);
      sw_stride = sw.numel() == 1 ? 0 : 1;
    }
    auto kernel = skinny_gemm_kernel<kMode, kUsePDL>;
    static bool attr_set = false;
    if (!attr_set) {
      // goal11-post: allow up to the 227-KB maximum (smem_pad steers placement: a padded CTA cannot share an SM with a
      // co-resident NVLink-bound TP kernel CTA)
      RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024));
      RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
      attr_set = true;
    }
    const int S = pick_cluster(kernel, NB, KB, static_cast<int>(num_sms), static_cast<int>(cluster));
    RuntimeCheck(S >= 1 && S <= kMaxCluster && S <= KB);
    const CUtensorMap mw = make_map_2d(w.data_ptr(), K, N, static_cast<uint64_t>(K), BK, BN);
    const CUtensorMap mx = make_map_2d(x.data_ptr(), K, M, static_cast<uint64_t>(x.stride(0)), BK, MT);
    const auto params = Params{
        .sw = static_cast<const float*>(sw.data_ptr()),
        .sx = static_cast<const float*>(sx.data_ptr()),
        .out = static_cast<__nv_bfloat16*>(out.data_ptr()),
        .out_ld = out.stride(0),
        .M = M,
        .N = N,
        .K = K,
        .NB = NB,
        .KB = KB,
        .S = S,
        .sx_ld = sx_ld,
        .sw_stride = sw_stride,
        .depth = static_cast<int>(depth),
        .warm = static_cast<int>(warm),
        .preload = static_cast<int>(preload),
        .stamps = reinterpret_cast<unsigned long long*>(stamps_addr),
        .l2pre = static_cast<int>(l2pre),
        .rs_calls = reinterpret_cast<const int*>(rs_calls_addr),
        .rs_peer = {},
        .rs_rank = static_cast<int>(rs_rank),
        .nx_ptr = reinterpret_cast<const uint8_t*>(nx_addr),
        .nx_bytes = nx_bytes,
    };
    RuntimeCheck(nx_bytes >= 0 && nx_bytes % 16 == 0 && nx_addr % 16 == 0);
    Params prm = params;
    if (rs) {
      const int64_t* pp = static_cast<const int64_t*>(rs_peers.data_ptr());
      for (int i = 0; i < RS_R; ++i) prm.rs_peer[i] = reinterpret_cast<uint8_t*>(pp[i]);
    }
    RuntimeCheck(smem_pad >= 0 && kSmemBytes + smem_pad <= 227 * 1024);
    LaunchKernel(dim3(NB * S), dim3(kThreads), x.device(), kSmemBytes + static_cast<int>(smem_pad))
        .enable_pdl(kUsePDL)
        .enable_cluster(dim3(S, 1, 1))(kernel, mw, mx, prm);
  }
};

}  // inline namespace SKINNY_NS
}  // namespace sglang
