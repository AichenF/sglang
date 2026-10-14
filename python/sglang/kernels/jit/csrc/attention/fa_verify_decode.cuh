// Split-KV verify-decode attention for sm_90a: the full-attention layers of MiMo-V2.x at TP8
// (1 KV head per rank, 16 query heads, verify block of W = 8 tokens, head dims 192 (QK) / 128 (V),
// page_size 1, causal bottom-right, bf16 KV).
//
// All W x 16 = 128 query rows of a request share one KV head, so one CTA owns a request's full
// 128-row block and every KV byte is read from HBM exactly once. The key range of each request is
// split over `nsplit` CTAs (grid = nsplit x bs, ~one wave on 132 SMs); a small combine kernel
// (PDL-launched) merges the normalized partials by their log-sum-exp.
//
// CTA = 3 warpgroups:
//   WG0      producer: gathers 64-key stages (K 64x192, V 64x128) through the page table with 16-B
//            cp.async into 128B-swizzled smem, STAGES deep; completion via cp.async.mbarrier.arrive.
//   WG1, WG2 consumers: rows 0-63 / 64-127 (token t = row / 16, head = row % 16). Q lives in
//            registers as the wgmma A operand; S = Q K^T is m64n64k16 x 12 (K-major B), the online
//            softmax runs in fp32 in the log2 domain, O += P V is m64n128k16 x 4 with P from registers
//            (MN-major B: V rows are [key][dv]).
// FA3 at this shape keeps ~2 stages of KV in flight per SM (pipeline-bound); this kernel keeps up to
// STAGES-1 stages (160 KB) in flight.
#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <cuda.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <mutex>

namespace sglang {
namespace fa_verify_detail {

constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void cp_async_mbar_arrive(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void named_sync(int id, int n) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void named_arrive(int id, int n) {
  asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
}
template <int N>
__device__ __forceinline__ void fence_f(float (&d)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(d[i])::"memory");
}
template <int N>
__device__ __forceinline__ void fence_u(uint32_t (&d)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+r"(d[i])::"memory");
}
// 128B-swizzled smem matrix descriptor (start, LBO, SBO in bytes).
__device__ __forceinline__ uint64_t desc_sw128(uint32_t addr, uint32_t lbo, uint32_t sbo) {
  uint64_t d = 0;
  d |= (uint64_t)((addr & 0x3FFFF) >> 4);
  d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
  d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
  d |= (uint64_t)1 << 62;
  return d;
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  uint32_t u;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(u) : "f"(hi), "f"(lo));
  return u;
}

// ---- wgmma wrappers --------------------------------------------------------------------------------
// D[64 x 64] (+)= A[64 x 16] (registers, bf16) * B[16 x 64] (smem desc, K-major), fp32 accumulate
__device__ __forceinline__ void wgmma_m64n64k16_rs_kmaj(float (&d)[32], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\n"
      "setp.ne.b32 p, %37, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{"
      "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31"
      "}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}
// D[64 x 128] (+)= A[64 x 16] (registers, bf16) * B[16 x 128] (smem desc, MN-major), fp32 accumulate
__device__ __forceinline__ void wgmma_m64n128k16_rs_mnmaj(float (&d)[64], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\n"
      "setp.ne.b32 p, %69, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{"
      "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63"
      "}, {%64, %65, %66, %67}, %68, p, 1, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}
// -----------------------------------------------------------------------------------------------------

constexpr int W = 8;          // verify tokens per request
constexpr int HQ = 16;        // query heads per rank (one KV head)
constexpr int ROWS = W * HQ;  // 128 query rows per request
constexpr int D = 192;        // QK head dim
constexpr int DV = 128;       // V head dim
constexpr int BN = 64;        // keys per stage
constexpr int KCH = D / 8;    // 24 x 16 B chunks per K row
constexpr int VCH = DV / 8;   // 16 x 16 B chunks per V row
constexpr int CH = KCH + VCH; // 40 chunks per key
constexpr int KBLK = BN * 128;            // one 64-element column block of a stage (64 rows x 128 B)
constexpr int KSTAGE = (D / 64) * KBLK;   // 24 KB
constexpr int VSTAGE = (DV / 64) * KBLK;  // 16 KB
constexpr int STAGE_BYTES = KSTAGE + VSTAGE;
constexpr int NTHREADS = 384;
constexpr int PRODUCER_REGS = 72;       // 128 x 72 + 256 x 216 = 64 K
constexpr int CONSUMER_REGS = 216;

template <int STAGES>
constexpr int smem_bytes() {
  return 1024 /*align slack*/ + STAGES * STAGE_BYTES + 2 * STAGES * 8 + 8;
}

// part_o:   [bs * ROWS, nsplit, DV] fp32, normalized by the split's own denominator
// part_lse: [bs * ROWS, nsplit]     fp32, log2-domain LSE of the split (-inf: empty split)
//
// The kernel signature still carries the template parameters and arguments of retired variants (kTMA: tensor-map row
// gather through tmk / tmv; kDiag: stream-only diagnostics; kRope: RoPE + KV-cache store folded in, through `rope`).
// Only kTMA = false, kDiag = 0, kRope = false is instantiated; the unused parameters are kept so the kernel symbol and
// its generated code stay unchanged.
struct RopeArgs {
  const __nv_bfloat16* knew = nullptr;  // qkv + 3072: un-rotated K rows of the new tokens (row stride q_stride_m)
  const __nv_bfloat16* vnew = nullptr;  // qkv + 3264: V rows of the new tokens
  const __nv_bfloat16* cos_sin = nullptr;  // [max_pos, 64] (cos 0..31 | sin 32..63)
  const void* pos = nullptr;            // [M] int32 / int64 positions
  int pos64 = 0;
};
template <int STAGES, bool kPDL, bool kTMA = false, int kDiag = 0, bool kRope = false>
__global__ __launch_bounds__(NTHREADS, 1) void fa_verify_kernel(
    const __grid_constant__ CUtensorMap tmk, const __grid_constant__ CUtensorMap tmv,
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V, const int* __restrict__ PT,
    const int* __restrict__ SEQLENS, float* __restrict__ part_o, float* __restrict__ part_lse,
    int q_stride_m, int q_stride_h, int k_stride, int v_stride, int pt_stride, int nsplit,
    float scale_log2, int interleave, const RopeArgs rope) {
  static_assert(!kTMA && kDiag == 0 && !kRope, "retired variant");
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const uint32_t sbase = smem_u32(smem);
  const uint32_t bar_full = sbase + STAGES * STAGE_BYTES;
  const uint32_t bar_empty = bar_full + STAGES * 8;
  const uint32_t bar_prep = bar_empty + STAGES * 8;  // unused (retired kRope variant); still initialized below

  const int sp = blockIdx.x, b = blockIdx.y;
  const int tid = threadIdx.x;
  const int wg = tid >> 7;

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(bar_full + s * 8, 128);   // 32 arrivals per producer warp (cp.async.mbarrier.arrive.noinc)
      mbar_init(bar_empty + s * 8, 256);  // one arrive per consumer thread
    }
    mbar_init(bar_prep, 256);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  const int seqlen = SEQLENS[b];
  const int ntiles = (seqlen + BN - 1) / BN;
  // Tiles of this CTA: interleaved (tile = sp + i * nsplit; the nsplit CTAs of a request sweep the KV
  // together, so concurrent reads stay inside a ~nsplit x 40 KB window -> DRAM page locality) or
  // contiguous chunks [t0, t1).
  const int t0 = interleave ? sp : (int)(((long long)sp * ntiles) / nsplit);
  const int t1 = interleave ? sp : (int)(((long long)(sp + 1) * ntiles) / nsplit);
  const int ntile_cta = interleave ? max(0, (ntiles - sp + nsplit - 1) / nsplit) : t1 - t0;
  const int tstep = interleave ? nsplit : 1;
  auto tile_of = [&](int i) { return t0 + i * tstep; };
  // Keys at or after this index belong to the current verify block (written by the preceding kernel).
  const int new_key0 = seqlen - W;

  if (wg == 0) {
    // ------------------------------------------------------------------ producer
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(PRODUCER_REGS));
    const int warp = tid >> 5, lane = tid & 31;
    const int* pt_row = PT + (size_t)b * pt_stride;
    bool waited = !kPDL;
    // Page-table entries are prefetched 4 tiles ahead in a register ring: the slot load is a
    // dependent global load in front of every stage's copies and would otherwise cap the issue rate.
    auto pt_load = [&](int tile) -> int {
      const int kidx = min(tile * BN + warp * 16 + (lane & 15), seqlen - 1);
      return tile < ntiles ? __ldg(pt_row + kidx) : 0;
    };
    auto produce = [&](int tile, int it, int slot) {
      const int s = it % STAGES;
      const uint32_t ph = (uint32_t)(it / STAGES) & 1u;
      if (!waited && (tile + 1) * BN > new_key0) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        waited = true;
      }
      mbar_wait(bar_empty + s * 8, ph ^ 1u);
      const uint32_t kst = sbase + s * STAGE_BYTES + warp * 16 * 128;
      const uint32_t vst = kst + KSTAGE;
      // Byte offsets of this warp's 16 keys (lane & 15 holds key lane & 15); < 4 GB for any pool size here.
      const uint32_t koff = (uint32_t)slot * (uint32_t)(k_stride * 2);
      const uint32_t voff = (uint32_t)slot * (uint32_t)(v_stride * 2);
      // Lane roles are fixed so all smem offsets except the stage base are per-lane constants:
      //  K chunks 0-15 of keys 2k, 2k+1 (lanes 0-15 / 16-31), 8 instructions;
      //  K chunks 16-23 of keys 4k .. 4k+3 (lanes 0-7, 8-15, 16-23, 24-31), 4 instructions;
      //  V chunks 0-15 of keys 2k, 2k+1, 8 instructions.
      // Consecutive lanes copy consecutive 16-B chunks of a row: every 32-B sector is fully used.
      const int ca = lane & 15, cb = 16 + (lane & 7);
      const int ha = lane >> 4, hb = lane >> 3;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int key = 2 * k + ha;
        const uint32_t ko = __shfl_sync(0xffffffffu, koff, key);
        const uint32_t vo = __shfl_sync(0xffffffffu, voff, key);
        const uint32_t sw = (uint32_t)(((ca & 7) ^ (key & 7)) << 4);
        cp_async16(kst + (ca >> 3) * KBLK + key * 128 + sw, reinterpret_cast<const char*>(K) + ko + ca * 16);
        cp_async16(vst + (ca >> 3) * KBLK + key * 128 + sw, reinterpret_cast<const char*>(V) + vo + ca * 16);
      }
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int key = 4 * k + hb;
        const uint32_t ko = __shfl_sync(0xffffffffu, koff, key);
        const uint32_t sw = (uint32_t)(((cb & 7) ^ (key & 7)) << 4);
        cp_async16(kst + (cb >> 3) * KBLK + key * 128 + sw, reinterpret_cast<const char*>(K) + ko + cb * 16);
      }
      cp_async_mbar_arrive(bar_full + s * 8);
    };
    // Unrolled by the prefetch distance so every ring register is reloaded in place: a rotating ring
    // (p0 = p1; ...) makes each move wait on the load issued one iteration earlier.
    const int n = ntile_cta;
    auto pt_load_i = [&](int i) { return i < n ? pt_load(tile_of(i)) : 0; };
    int p0 = pt_load_i(0), p1 = pt_load_i(1), p2 = pt_load_i(2), p3 = pt_load_i(3);
    for (int i = 0; i < n; i += 4) {
      produce(tile_of(i), i, p0);
      p0 = pt_load_i(i + 4);
      if (i + 1 < n) {
        produce(tile_of(i + 1), i + 1, p1);
        p1 = pt_load_i(i + 5);
      }
      if (i + 2 < n) {
        produce(tile_of(i + 2), i + 2, p2);
        p2 = pt_load_i(i + 6);
      }
      if (i + 3 < n) {
        produce(tile_of(i + 3), i + 3, p3);
        p3 = pt_load_i(i + 7);
      }
    }
    if (!waited) asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("cp.async.wait_all;" ::: "memory");  // do not exit with copies in flight
    return;
  }

  // -------------------------------------------------------------------- consumers
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(CONSUMER_REGS));
  const int cw = wg - 1;                 // 0 / 1: rows 0-63 / 64-127
  const int warp = (tid >> 5) & 3;       // warp within the warpgroup
  const int lane = tid & 31;
  const int g = lane >> 2, c4 = lane & 3;
  const int tok = cw * 4 + warp;         // this warp's 16 rows are token `tok`, heads 0..15
  const int h0 = g, h1 = g + 8;

  if constexpr (kPDL) asm volatile("griddepcontrol.wait;" ::: "memory");

  // Q fragments (A operand of S = Q K^T, m64k16 layout): rows (tok, h0) / (tok, h1).
  uint32_t qf[D / 16][4];
  {
    const __nv_bfloat16* q0 = Q + (size_t)(b * W + tok) * q_stride_m + (size_t)h0 * q_stride_h;
    const __nv_bfloat16* q1 = Q + (size_t)(b * W + tok) * q_stride_m + (size_t)h1 * q_stride_h;
#pragma unroll
    for (int kk = 0; kk < D / 16; ++kk) {
      const int col = kk * 16 + c4 * 2;
      qf[kk][0] = *reinterpret_cast<const uint32_t*>(q0 + col);
      qf[kk][1] = *reinterpret_cast<const uint32_t*>(q1 + col);
      qf[kk][2] = *reinterpret_cast<const uint32_t*>(q0 + col + 8);
      qf[kk][3] = *reinterpret_cast<const uint32_t*>(q1 + col + 8);
    }
  }

  float o[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) o[i] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const int klim = new_key0 + tok;  // last key this token may see
  int n = ntile_cta;

  // FA3-style intra-warpgroup overlap. Iteration i: issue S(i) = Q K(i) and O += P(i-1) V(i-1) back to
  // back, wait for S(i) only, run the softmax of S(i) while PV(i-1) is on the tensor core, then wait for
  // PV(i-1) and hand stage i-1 back. S is a fresh accumulator every iteration (no loop-carried wgmma
  // registers), P(i-1) is a plain bf16 A fragment, O is rescaled before PV(i-1) is issued.
  auto stage_of = [&](int it, uint32_t& ph) {
    ph = (uint32_t)(it / STAGES) & 1u;
    return it % STAGES;
  };
  auto issue_s = [&](float (&sacc)[32], int s) {
    const uint32_t kst = sbase + s * STAGE_BYTES;
    fence_f(sacc);
    wg_fence();
#pragma unroll
    for (int kk = 0; kk < D / 16; ++kk) {
      const uint64_t dk = desc_sw128(kst + (kk >> 2) * KBLK + (kk & 3) * 32, 16, 1024);
      wgmma_m64n64k16_rs_kmaj(sacc, qf[kk], dk, kk > 0 ? 1 : 0);
    }
    wg_commit();
    fence_f(sacc);
  };
  auto issue_pv = [&](uint32_t (&pf)[4][4], int s) {
    const uint32_t vst = sbase + s * STAGE_BYTES + KSTAGE;
    fence_f(o);
#pragma unroll
    for (int j = 0; j < 4; ++j) fence_u(pf[j]);
    wg_fence();
#pragma unroll
    for (int j = 0; j < BN / 16; ++j) {
      const uint64_t dv = desc_sw128(vst + j * 2048, KBLK, 1024);
      wgmma_m64n128k16_rs_mnmaj(o, pf[j], dv, 1);
    }
    wg_commit();
    fence_f(o);
#pragma unroll
    for (int j = 0; j < 4; ++j) fence_u(pf[j]);
  };
  // softmax of one S tile: updates m/l, returns the O rescale factors and the bf16 P fragment.
  auto softmax = [&](float (&sacc)[32], int tile, uint32_t (&pf)[4][4], float& al0, float& al1) {
    const int key_base = tile * BN;
    if (key_base + BN - 1 > new_key0) {
#pragma unroll
      for (int nb = 0; nb < 8; ++nb) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int key = key_base + nb * 8 + c4 * 2 + e;
          if (key > klim) {
            sacc[nb * 4 + e] = -INFINITY;
            sacc[nb * 4 + 2 + e] = -INFINITY;
          }
        }
      }
    }
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int nb = 0; nb < 8; ++nb) {
      mx0 = fmaxf(mx0, fmaxf(sacc[nb * 4 + 0], sacc[nb * 4 + 1]));
      mx1 = fmaxf(mx1, fmaxf(sacc[nb * 4 + 2], sacc[nb * 4 + 3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    const float mn0 = fmaxf(m0, mx0 * scale_log2);
    const float mn1 = fmaxf(m1, mx1 * scale_log2);
    // A row that has seen no visible key yet keeps an exact zero state.
    const float mu0 = (mn0 == -INFINITY) ? 0.f : mn0;
    const float mu1 = (mn1 == -INFINITY) ? 0.f : mn1;
    al0 = exp2f(m0 - mu0);
    al1 = exp2f(m1 - mu1);
    m0 = mn0;
    m1 = mn1;
    float ps0 = 0.f, ps1 = 0.f;
#pragma unroll
    for (int nb = 0; nb < 8; ++nb) {
      const float p0 = exp2f(fmaf(sacc[nb * 4 + 0], scale_log2, -mu0));
      const float p1 = exp2f(fmaf(sacc[nb * 4 + 1], scale_log2, -mu0));
      const float p2 = exp2f(fmaf(sacc[nb * 4 + 2], scale_log2, -mu1));
      const float p3 = exp2f(fmaf(sacc[nb * 4 + 3], scale_log2, -mu1));
      ps0 += p0 + p1;
      ps1 += p2 + p3;
      // k16 chunk j = nb / 2: regs {row g, keys 2c..}, {row g+8, ..}, {row g, keys 8+2c..}, {row g+8, 8+..}
      pf[nb >> 1][(nb & 1) * 2 + 0] = pack_bf16(p0, p1);
      pf[nb >> 1][(nb & 1) * 2 + 1] = pack_bf16(p2, p3);
    }
    l0 = l0 * al0 + ps0;
    l1 = l1 * al1 + ps1;
  };
  auto rescale_o = [&](float al0, float al1) {
    // Once the running max settles (most tiles of a long context) the rescale is a no-op: skip it.
    if (!__all_sync(0xffffffffu, al0 == 1.f && al1 == 1.f)) {
#pragma unroll
      for (int nb = 0; nb < 16; ++nb) {
        o[nb * 4 + 0] *= al0;
        o[nb * 4 + 1] *= al0;
        o[nb * 4 + 2] *= al1;
        o[nb * 4 + 3] *= al1;
      }
    }
  };

  // Ping-pong between the two consumer warpgroups (FA3 warp-scheduler barriers): a warpgroup issues its
  // GEMMs only in its turn, so one warpgroup's GEMMs run on the tensor core while the other does its
  // softmax instead of both doing the same phase at once. Named barrier 1 + cw = "cw may issue".
  const int bar_me = 1 + cw, bar_other = 1 + (cw ^ 1);
  if (cw == 1 && n > 0) named_arrive(1, 256);  // warpgroup 0 issues first

  uint32_t pf[4][4];
  float al0 = 1.f, al1 = 1.f;
  int s_prev = 0;
  if (n > 0) {
    uint32_t ph;
    const int s = stage_of(0, ph);
    mbar_wait(bar_full + s * 8, ph);
    float sacc[32];
    named_sync(bar_me, 256);
    issue_s(sacc, s);
    named_arrive(bar_other, 256);
    wg_wait<0>();
    fence_f(sacc);
    softmax(sacc, tile_of(0), pf, al0, al1);
    s_prev = s;
  }
  for (int i = 1; i < n; ++i) {
    uint32_t ph;
    const int s = stage_of(i, ph);
    mbar_wait(bar_full + s * 8, ph);
    float sacc[32];
    named_sync(bar_me, 256);
    issue_s(sacc, s);
    rescale_o(al0, al1);
    issue_pv(pf, s_prev);
    named_arrive(bar_other, 256);
    wg_wait<1>();
    fence_f(sacc);
    uint32_t pn[4][4];
    softmax(sacc, tile_of(i), pn, al0, al1);
    wg_wait<0>();
    fence_f(o);
#pragma unroll
    for (int j = 0; j < 4; ++j) fence_u(pf[j]);
    mbar_arrive(bar_empty + s_prev * 8);
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) pf[j][e] = pn[j][e];
    s_prev = s;
  }
  if (n > 0) {
    named_sync(bar_me, 256);
    rescale_o(al0, al1);
    issue_pv(pf, s_prev);
    if (cw == 0) named_arrive(bar_other, 256);  // warpgroup 1's last turn needs no successor
    wg_wait<0>();
    fence_f(o);
    mbar_arrive(bar_empty + s_prev * 8);
  }

  if constexpr (kPDL) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  // ---- epilogue: normalized partial O + log2 LSE of this split, [b][row][split][dv]
  l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
  l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
  const int r0 = tok * HQ + h0, r1 = tok * HQ + h1;
  const size_t row0 = (size_t)b * ROWS + r0, row1 = (size_t)b * ROWS + r1;
  if (c4 == 0) {
    part_lse[row0 * nsplit + sp] = (l0 > 0.f) ? m0 + __log2f(l0) : -INFINITY;
    part_lse[row1 * nsplit + sp] = (l1 > 0.f) ? m1 + __log2f(l1) : -INFINITY;
  }
  // A CTA with no tile writes zeros (o == 0, inv == 0): the combine loads every partial unconditionally.
  const float inv0 = (l0 > 0.f) ? 1.f / l0 : 0.f;
  const float inv1 = (l1 > 0.f) ? 1.f / l1 : 0.f;
  float* po0 = part_o + (row0 * nsplit + sp) * DV;
  float* po1 = part_o + (row1 * nsplit + sp) * DV;
#pragma unroll
  for (int nb = 0; nb < 16; ++nb) {
    const int col = nb * 8 + c4 * 2;
    *reinterpret_cast<float2*>(po0 + col) = make_float2(o[nb * 4 + 0] * inv0, o[nb * 4 + 1] * inv0);
    *reinterpret_cast<float2*>(po1 + col) = make_float2(o[nb * 4 + 2] * inv1, o[nb * 4 + 3] * inv1);
  }
}

// Combine: CTA = 8 warps = 2 output rows; the 4 warps of a row split the nsplit partials (s = w, w+4, ...),
// lanes hold 4 dv each, and the 4 warp sums meet in smem. All partial loads of a warp are independent, so a row
// costs ~2 L2 round trips instead of nsplit dependent ones.
constexpr int CMB_WARPS_PER_ROW = 4;
template <bool kPDL>
__global__ __launch_bounds__(256) void fa_verify_combine_kernel(
    const float* __restrict__ part_o, const float* __restrict__ part_lse, __nv_bfloat16* __restrict__ OUT,
    int nsplit, int out_stride_m, int out_stride_h, int total_rows) {
  if constexpr (kPDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    // Let a PDL-launched consumer of the attention output (e.g. a fused o_proj kernel) start its weight prefetch
    // now; it must griddepcontrol.wait before reading OUT.
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
  __shared__ float4 s_acc[2][CMB_WARPS_PER_ROW][32];
  __shared__ float s_den[2][CMB_WARPS_PER_ROW];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int rl = warp / CMB_WARPS_PER_ROW, wr = warp % CMB_WARPS_PER_ROW;
  const int row = blockIdx.x * 2 + rl;
  const bool valid = row < total_rows;
  const float* lse = part_lse + (size_t)(valid ? row : 0) * nsplit;
  const float* po = part_o + (size_t)(valid ? row : 0) * nsplit * DV + lane * 4;
  // The first batch of partials (independent of the LSEs) is loaded together with the LSEs: one L2 round trip.
  // Lane l holds the LSEs of splits l, l + 32, l + 64, l + 96 (nsplit <= 128); weights are broadcast by shuffle.
  constexpr int U = 9;
  float4 v[U];
#pragma unroll
  for (int u = 0; u < U; ++u) {
    const int s = wr + u * CMB_WARPS_PER_ROW;
    v[u] = (valid && s < nsplit) ? *reinterpret_cast<const float4*>(po + (size_t)s * DV) : make_float4(0.f, 0.f, 0.f, 0.f);
  }
  float lv[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) lv[k] = (valid && lane + 32 * k < nsplit) ? lse[lane + 32 * k] : -INFINITY;
  float mx = fmaxf(fmaxf(lv[0], lv[1]), fmaxf(lv[2], lv[3]));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  const float mref = (mx == -INFINITY) ? 0.f : mx;
  float wv[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) wv[k] = (lv[k] == -INFINITY) ? 0.f : exp2f(lv[k] - mref);
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  float den = 0.f;
  for (int s0 = wr;; s0 += CMB_WARPS_PER_ROW * U) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int s = s0 + u * CMB_WARPS_PER_ROW;
      const float w = __shfl_sync(0xffffffffu, wv[(s >> 5) & 3], s & 31) * (s < nsplit ? 1.f : 0.f);
      den += w;
      acc.x = fmaf(w, v[u].x, acc.x);
      acc.y = fmaf(w, v[u].y, acc.y);
      acc.z = fmaf(w, v[u].z, acc.z);
      acc.w = fmaf(w, v[u].w, acc.w);
    }
    const int s1 = s0 + CMB_WARPS_PER_ROW * U;
    if (s1 >= nsplit) break;
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int s = s1 + u * CMB_WARPS_PER_ROW;
      v[u] = (valid && s < nsplit) ? *reinterpret_cast<const float4*>(po + (size_t)s * DV) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
  }
  s_acc[rl][wr][lane] = acc;
  if (lane == 0) s_den[rl][wr] = den;
  __syncthreads();
  if (wr != 0 || !valid) return;
#pragma unroll
  for (int k = 1; k < CMB_WARPS_PER_ROW; ++k) {
    const float4 a = s_acc[rl][k][lane];
    acc.x += a.x;
    acc.y += a.y;
    acc.z += a.z;
    acc.w += a.w;
    den += s_den[rl][k];
  }
  const int b = row / ROWS, r = row % ROWS;
  const int tok = r / HQ, h = r % HQ;
  __nv_bfloat16* dst = OUT + (size_t)(b * W + tok) * out_stride_m + (size_t)h * out_stride_h + lane * 4;
  const float inv = den > 0.f ? 1.f / den : 0.f;
  uint2 pk;
  pk.x = pack_bf16(acc.x * inv, acc.y * inv);
  pk.y = pack_bf16(acc.z * inv, acc.w * inv);
  *reinterpret_cast<uint2*>(dst) = pk;
}

template <int STAGES, bool kPDL>
void launch(const void* q, const void* k, const void* v, const int* pt, const int* sl, float* po, float* plse, void* out,
            int bs, int nsplit, int q_stride_m, int q_stride_h, int k_stride, int v_stride, int pt_stride,
            int out_stride_m, int out_stride_h, float scale_log2, int interleave, DLDevice device) {
  auto kern = fa_verify_kernel<STAGES, kPDL>;
  constexpr int kSmem = smem_bytes<STAGES>();
  static std::once_flag once;
  std::call_once(once, [&] {
    host::RuntimeDeviceCheck(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  });
  const CUtensorMap tmk{}, tmv{};  // unused kernel arguments (see RopeArgs)
  const RopeArgs rope{};
  host::LaunchKernel(dim3(nsplit, bs), dim3(NTHREADS), device, kSmem)
      .enable_pdl(kPDL)(kern, tmk, tmv, (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, (const __nv_bfloat16*)v, pt,
                        sl, po, plse, q_stride_m, q_stride_h, k_stride, v_stride, pt_stride, nsplit, scale_log2,
                        interleave, rope);
  const int rows = bs * ROWS;
  host::LaunchKernel(dim3((rows + 1) / 2), dim3(256), device)
      .enable_pdl(kPDL)(fa_verify_combine_kernel<kPDL>, (const float*)po, (const float*)plse, (__nv_bfloat16*)out,
                        nsplit, out_stride_m, out_stride_h, rows);
}

}  // namespace fa_verify_detail

// q [M, 16, 192] (row/head strides explicit, last dim contiguous); k [slots, 192] / v [slots, 128] 2-D views;
// page_table [bs, pages] int32; cache_seqlens [bs] int32; part_o [bs*128*nsplit*128] fp32;
// part_lse [bs*128*nsplit] fp32; out [M, 16, 128] view (row/head strides explicit); stages 4 or 5.
inline void fa_verify_decode(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView k,
    tvm::ffi::TensorView v,
    tvm::ffi::TensorView page_table,
    tvm::ffi::TensorView cache_seqlens,
    tvm::ffi::TensorView part_o,
    tvm::ffi::TensorView part_lse,
    tvm::ffi::TensorView out,
    int64_t q_stride_m,
    int64_t q_stride_h,
    int64_t pt_stride_b,
    int64_t out_stride_m,
    int64_t out_stride_h,
    int64_t nsplit,
    double sm_scale,
    int64_t stages,
    int64_t use_pdl,
    int64_t interleave) {
  using namespace fa_verify_detail;
  const int bs = (int)page_table.shape()[0];
  const int pt_stride = (int)pt_stride_b;
  const int k_stride = (int)k.shape()[1];
  const int v_stride = (int)v.shape()[1];
  const float sl2 = (float)(sm_scale * (double)LOG2E);
  const DLDevice device = q.device();
#define FA_VERIFY_CASE(st, pdl)                                                                                    \
  if (stages == st && (use_pdl != 0) == pdl) {                                                                    \
    launch<st, pdl>(q.data_ptr(), k.data_ptr(), v.data_ptr(), (const int*)page_table.data_ptr(),                  \
                    (const int*)cache_seqlens.data_ptr(), (float*)part_o.data_ptr(), (float*)part_lse.data_ptr(), \
                    out.data_ptr(), bs, (int)nsplit, (int)q_stride_m, (int)q_stride_h, k_stride, v_stride,         \
                    pt_stride, (int)out_stride_m, (int)out_stride_h, sl2, (int)interleave, device);               \
    return;                                                                                                        \
  }
  FA_VERIFY_CASE(4, false) FA_VERIFY_CASE(4, true)
  FA_VERIFY_CASE(5, false) FA_VERIFY_CASE(5, true)
#undef FA_VERIFY_CASE
  TVM_FFI_ICHECK(false) << "fa_verify_decode: unsupported stages";
}

}  // namespace sglang
