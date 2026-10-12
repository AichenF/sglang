// DFlash draft megakernel (goal11-draft): the 5 decoder layers of the MiMo-V2.6 DFlash draft at TP8 for M = 32 rows
// (4 requests x 8 block tokens) in ONE persistent kernel, plus the layer-0 input RMSNorm/quant and the final norm.
//
// Per layer (per rank): x (fp8, per-token scale) -> qkv GEMM (2304 x 6144 fp8, per-channel scale) -> q/k RMSNorm + RoPE
// (64 of 128 dims, NeoX) + value scale + KV-cache store -> bidirectional window-1024 attention with per-head sinks (16 q
// heads, 1 kv head) -> per-token fp8 quant -> o_proj (6144 x 2048) -> TP8 all-reduce + residual + RMSNorm + per-token
// fp8 quant (dense's tp_arnq protocol, same IPC buffers / call counters) -> gate_up (4096 x 6144) -> silu*up + quant ->
// down (6144 x 2048) -> all-reduce + residual + RMSNorm + quant. The numerics (quant formulas, rounding points, the
// all-reduce rank order, flashinfer's norm order) are those of the unfused fused-TP path (models/dflash_tp_fast.py);
// only GEMM / attention summation orders differ.
//
// Grid: 15 clusters x 8 CTAs (the most 8-CTA clusters an H200 keeps co-resident at 1 CTA/SM), 320 threads:
//   warps 0-7 (two warpgroups) = the phase program; warp 8 = weight producer (TMA into a per-CTA ring of 16-KB slabs,
//   running ahead across phase and layer boundaries); warp 9 = KV gather (attention clusters: cp.async of the window keys
//   of the next layer as soon as the KV region is free).
// Clusters 0-7: attention (request c/2, query heads 8*(c%2)..+7; rank = key split, combine over DSMEM).
// Clusters 8-11: all-reduce CTAs (one token each). Clusters 12-14: GEMM only. Every CTA takes GEMM tiles (host plan).
// GEMMs: swap-AB wgmma m64n32k32 (weights = A, 64-row tiles, two tiles per 16-KB ring slab), split-K over the cluster
//   (8-way for K = 6144, 2 x 4-way for K = 2048), partials reduce-scattered over DSMEM (st.async), summed in rank order.
// Cross-CTA ordering: monotonic uint32 counters in global memory (targets derived from a per-launch epoch), acquire
//   polls; cross-rank: Lamport-marked IPC pushes (tp_arnq).
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <cstring>

#ifndef DM_PACE
#define DM_PACE 0
#endif
#ifndef DM_LL_QKV
#define DM_LL_QKV 0
#endif
#ifndef DM_SC_FENCE
#define DM_SC_FENCE 1
#endif
#ifndef DM_STAMPS
#define DM_STAMPS 0
#endif
#ifndef DM_THREADS
#define DM_THREADS 320
#endif
#ifndef DM_REG_LO
#define DM_REG_LO 112
#endif
#ifndef DM_REG_HI
#define DM_REG_HI 200
#endif
#ifndef DM_NO_ATT
#define DM_NO_ATT 0
#endif

namespace sglang {
namespace dflash_mega {

// ------------------------------------------------------------------ configuration
constexpr int R = 8;
constexpr int H = 6144;
constexpr int SL = H / R;  // 768
constexpr int MT = 32;
constexpr int NREQ = 4;
constexpr int BLK = 8;
constexpr int NL = 5;
constexpr int NHQ = 16;
constexpr int HD = 128;
constexpr int QKV_N = (NHQ + 2) * HD;  // 2304
constexpr int INTER = 2048;
constexpr int WIN = 1023;  // window (left == right) of the bidirectional sliding window
constexpr int CS = 8;
constexpr int NCL = 15;
constexpr int GRID = NCL * CS;  // 120
constexpr int NCT = 256;        // compute threads (warps 0-7)
constexpr int THREADS = DM_THREADS;  // 384: warpgroup 2 = producer + gather (+ 2 idle warps), registers moved by setmaxnreg
constexpr int NATT = 8;   // clusters 0..7: attention
constexpr int AR0 = 8;    // clusters 8..11: all-reduce token CTAs
constexpr int SLAB = 16384;
constexpr int TILEB = 8192;
constexpr int XU = 4096;
constexpr int MAXT_PH = 5;
constexpr int MAX_SLOT = 16;
constexpr int KVU = 16;
constexpr int KV_MAXU = 9;
constexpr int KV_MAXK = KVU * KV_MAXU;  // 144
constexpr int SMEM_BYTES = 232448;
constexpr float FP8_MAX = 448.f;
enum { PH_QKV = 0, PH_O = 1, PH_GU = 2, PH_DN = 3 };
enum { C_LAUNCH = 0, C_X = 1, C_ATT = 2, C_ACT = 3, C_QKV = 8, C_N = 64 };

// smem layout (bytes from the 1024-aligned dynamic base)
constexpr int MISC_BYTES = 2048;
constexpr int XREG = 24576;                  // X operand (up to 6 units)
constexpr int GREG_ATT = XREG + 4 * TILEB;   // attention clusters: <= 4 tiles per phase -> 56 KB
constexpr int GREG_FULL = XREG + 5 * TILEB;  // others: <= 5 tiles -> 64 KB
constexpr int KV_BYTES = 2 * KV_MAXK * HD * 2;  // 73728
constexpr int COMB_STRIDE = 132;             // floats per (src, head): 128 o + lse + pad
constexpr int COMB_SRC_BYTES = 8 * (HD * 4 + 4);  // 4128 bytes per source split
constexpr int QREG = 34 * 1024;              // Q tile inside the GEMM region (after COMB)

// IPC receive-buffer layout of csrc/distributed/dflash_tp_fused.cuh (keep in sync)
constexpr int64_t RS_SLOT = SL * 2;
constexpr int64_t RS_BYTES = 2LL * MT * R * RS_SLOT;
constexpr int64_t ST_BYTES = 2LL * MT * R * 8;
constexpr int64_t AG_SLOT = SL * 2 + 16;
constexpr int64_t RS_OFF = 0;
constexpr int64_t AG_OFF = RS_OFF + RS_BYTES + ST_BYTES;

struct LayerP {
  const float* sw[4];             // per-channel weight scales: qkv [2304], o [6144], gate_up [4096], down [6144]
  const __nv_bfloat16* ln_in;     // input_layernorm [H]
  const __nv_bfloat16* ln_post;   // post_attention_layernorm [H]
  const __nv_bfloat16* qn;        // q_norm [128]
  const __nv_bfloat16* kn;        // k_norm [128]
  const void* sinks;              // [16] bf16 / fp32 (nullptr: none)
  __nv_bfloat16* kc;              // k cache [slots][1][128]
  __nv_bfloat16* vc;
};

struct Plan {
  int16_t tile[4][NCL][2][MAXT_PH];  // [phase][cluster][k-group][i]: global 64-row tile ids
  int16_t n[4][NCL][2];
  int16_t nslot[NCL];
};

struct Params {
  LayerP L[NL];
  const CUtensorMap* maps;  // global memory: [l * 4 + phase] weight maps, then [NL * 4 + {0, 1}] x maps
  const __nv_bfloat16* final_norm;
  __nv_bfloat16* hidden;    // [MT][H] residual stream (in place)
  __nv_bfloat16* out;       // [MT][H] final normalized hidden
  const void* positions;    // [MT]
  const int* seqlens;       // [NREQ]
  const int* page_table;    // [NREQ][pt_stride]
  const void* loc;          // [MT] cache slots of the block rows
  const void* cos_sin;      // [max_pos][64] (cos | sin)
  int64_t pt_stride;
  int64_t cs_stride;
  int64_t kv_stride;        // elements between consecutive cache slots
  int pos64, loc64, cs_f32, sink_f32;
  float eps, qk_eps, v_scale, sm_scale;
  uint8_t* xq;              // [2][MT][H] e4m3
  float* xs;                // [2][MT]
  __nv_bfloat16* qkv;       // [2][MT][QKV_N] (layer-parity double buffer; DM_LL_QKV: Lamport-marked words)
  __nv_bfloat16* att;       // [MT][2048]
  __nv_bfloat16* act;       // [MT][INTER]
  unsigned* amax;           // [2 parity][2 kind][NL][MT]
  unsigned* cnt;            // [C_N]
  int* calls;               // [MT] tp_arnq per-token call counters
  uint8_t* peer[R];
  int rank;
  int nl;                   // layers run (5; fewer only in tests -- the epoch counters are padded at the end)
  int look;                 // weight units L2-prefetched ahead of the ring
  unsigned long long* stamps;  // optional [GRID][64]
  Plan plan;
};

#define DEV __device__ __forceinline__

// ------------------------------------------------------------------ PTX helpers
DEV uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
DEV unsigned long long gtime() {
  unsigned long long v;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(v));
  return v;
}
DEV void mbar_init(uint64_t* b, int cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(cnt) : "memory");
}
DEV void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(b)), "r"(bytes) : "memory");
}
DEV void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(b)) : "memory");
}
DEV void mbar_arrive_remote(uint32_t addr_cluster) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(addr_cluster) : "memory");
}
DEV void mbar_wait(uint64_t* b, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred P1;\nWAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra DONE_%=;\nbra WAIT_%=;\nDONE_%=:\n}\n" ::"r"(smem_u32(b)),
      "r"(parity)
      : "memory");
}
DEV void mbar_wait_cluster(uint64_t* b, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred P1;\nWAIT_%=:\n"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra DONE_%=;\nbra WAIT_%=;\nDONE_%=:\n}\n" ::"r"(smem_u32(b)),
      "r"(parity)
      : "memory");
}
DEV uint32_t mapa(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
DEV uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
DEV uint32_t cluster_id_x() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(r));
  return r;
}
DEV void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
DEV void st_async_v2(uint32_t addr, float a, float b, uint32_t bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];" ::"r"(addr),
               "f"(a), "f"(b), "r"(bar)
               : "memory");
}
DEV void st_async_v4(uint32_t addr, float a, float b, float c, float d, uint32_t bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(addr),
               "f"(a), "f"(b), "f"(c), "f"(d), "r"(bar)
               : "memory");
}
// position of token t inside a 32-float row of the reduce-scatter receive buffers: the 8 tokens a wgmma fragment thread
// holds for one row ((t & 7) >> 1 == lane & 3) are contiguous -> two 16-byte st.async per row and thread
DEV int tpos(int t) { return ((t & 7) >> 1) * 8 + (t >> 3) * 2 + (t & 1); }
DEV void st_async_b32(uint32_t addr, float a, uint32_t bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(addr),
               "r"(__float_as_uint(a)), "r"(bar)
               : "memory");
}
DEV void tma2d(const CUtensorMap* map, void* dst, uint64_t* bar, int c0, int c1, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "l"(pol)
      : "memory");
}
DEV void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src) : "memory");
}
DEV void cp_async_arrive_noinc(uint64_t* bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(smem_u32(bar)) : "memory");
}
DEV void fence_proxy_async_shared() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
DEV void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
DEV void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
DEV unsigned ld_acq(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
// release-add: orders this thread's (and, after a CTA barrier, the CTA's) prior writes before the increment
DEV void red_release(unsigned* p, unsigned v) {
#if DM_SC_FENCE
  __threadfence();
  atomicAdd(p, v);
#else
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
#endif
}
DEV void wait_ge(const unsigned* p, unsigned target) {
  while (static_cast<int>(ld_acq(p) - target) < 0) {
  }
}
DEV uint4 ldcg_v4(const void* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
DEV unsigned ldcg_u32(const void* p) {
  unsigned v;
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
DEV void st_v4(void* p, uint4 v) {
  asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
DEV void st_v2(void* p, uint2 v) {
  asm volatile("st.global.v2.u32 [%0], {%1, %2};" ::"l"(p), "r"(v.x), "r"(v.y) : "memory");
}
DEV uint4 ld_vol_v4(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
DEV uint2 ld_vol_v2(const void* p) {
  uint2 v;
  asm volatile("ld.volatile.global.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p) : "memory");
  return v;
}
DEV uint32_t nz(uint32_t w) { return w == 0u ? 0x80000000u : w; }  // +0.0 -> -0.0 (Lamport arrival marker)
DEV uint32_t unz(uint32_t w) { return w == 0x80000000u ? 0u : w; }
// poll a Lamport-marked 16-byte vector until every word has arrived, then drop the markers
DEV uint4 poll_ll_v4(const void* p) {
  uint4 v;
  do {
    v = ld_vol_v4(p);
  } while (v.x == 0u || v.y == 0u || v.z == 0u || v.w == 0u);
  return make_uint4(unz(v.x), unz(v.y), unz(v.z), unz(v.w));
}
DEV float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
DEV float bf_hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }
DEV uint32_t pack_bf2(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}
DEV float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
DEV uint16_t pack_e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}
DEV uint64_t make_desc(const void* p) {
  // K-major, 128B swizzle, 8-row x 128 B atoms (SBO 1024 B)
  const uint64_t a = smem_u32(p);
  return ((a & 0x3FFFFull) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}
DEV void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
DEV void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
DEV void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
}
DEV void wgmma_m64n32k32_e4m3(float* d, uint64_t da, uint64_t db) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, p, 1, 1;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "l"(da), "l"(db));
}
DEV void ldsm_x4(uint32_t addr, uint32_t* r) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
DEV void ldsm_x4_t(uint32_t addr, uint32_t* r) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
DEV void mma_bf16(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// 16-byte chunk swizzle of a 256-byte row (128-B halves, XOR with the row's low 3 bits)
DEV int swz(int ch, int row) { return (ch & 8) | ((ch ^ row) & 7); }

// ------------------------------------------------------------------ shared state
struct Smem {
  uint64_t* full;     // [MAX_SLOT]
  uint64_t* empty;    // [MAX_SLOT]
  uint64_t* xbar;
  uint64_t* recvbar;
  uint64_t* kvfull;
  uint64_t* kvempty;
  uint64_t* combbar;
  uint64_t* combok;
  int* calls0;        // [MT]
  float* sx;          // [MT]
  float* sinv;        // [MT]
  unsigned* amax;     // [MT]
  float* red;         // [64]
  int* slot;          // [KV_MAXK]
  int* misc;          // [16]: 0 E, 1 seqlen (attention request), ...
  long long* pos;     // [BLK] positions of the attention request's rows
  long long* loc;     // [BLK]
  uint8_t* ring;
  uint8_t* greg;
  uint8_t* kv;
  int nslot;
};

DEV void stamp(const Params& p, int k) {
#if DM_STAMPS
  if (p.stamps) p.stamps[blockIdx.x * 128 + k] = gtime();
#endif
}
constexpr int STAMP_L = 2;  // layer with the detailed stamps (slots 40..63)
DEV void dstamp(const Params& p, int l, int k) {
#if DM_STAMPS
  if (p.stamps && l == STAMP_L) p.stamps[blockIdx.x * 128 + k] = gtime();
#endif
}

// ------------------------------------------------------------------ producer (warp 8, lane 0)
// The CTA's weight units (16-KB slabs = up to two 64-row tiles x 128 K) in consumption order: layer, phase, slab, unit.
struct UIt {
  int l, ph, sl, u, nt, nu, ks;
  const int16_t* tl;
};
DEV bool uit_settle(const Params& p, int c, int rank, UIt& it) {
  while (it.l < p.nl) {
    const bool s8 = (it.ph == PH_QKV || it.ph == PH_GU);
    const int g = s8 ? 0 : (rank >> 2);
    it.nu = s8 ? 6 : 4;
    it.ks = s8 ? rank : (rank & 3);
    it.nt = p.plan.n[it.ph][c][g];
    it.tl = p.plan.tile[it.ph][c][g];
    if (2 * it.sl < it.nt) return true;
    it.sl = 0;
    it.u = 0;
    if (++it.ph == 4) {
      it.ph = 0;
      ++it.l;
    }
  }
  return false;
}
DEV bool uit_next(const Params& p, int c, int rank, UIt& it) {
  if (++it.u < it.nu) return true;
  it.u = 0;
  ++it.sl;
  return uit_settle(p, c, rank, it);
}
DEV void tma_prefetch_l2(const CUtensorMap* map, int c0, int c1) {
  asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" ::"l"(reinterpret_cast<uint64_t>(map)),
               "r"(c0), "r"(c1)
               : "memory");
}
DEV bool mbar_test(uint64_t* b, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n.reg .pred P1;\nmbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2;\nselp.u32 %0, 1, 0, P1;\n}\n"
      : "=r"(ok)
      : "r"(smem_u32(b)), "r"(parity)
      : "memory");
  return ok != 0;
}
// pf: L2 prefetch only; else TMA into dst (ring slot) completing on bar
DEV void uit_issue(const Params& p, const UIt& it, bool pf, uint8_t* dst, uint64_t* bar, uint64_t pol) {
  const CUtensorMap* map = p.maps + it.l * 4 + it.ph;
  const int t0 = it.tl[2 * it.sl];
  const int t1 = (2 * it.sl + 1 < it.nt) ? it.tl[2 * it.sl + 1] : -1;
  const int kc = (it.ks * it.nu + it.u) * 128;
#pragma unroll
  for (int w = 0; w < 2; ++w) {
    const int t = w ? t1 : t0;
    if (t < 0) continue;
    if (it.ph == PH_GU) {
      if (pf) {
        tma_prefetch_l2(map, kc, t * 32);
        tma_prefetch_l2(map, kc, INTER + t * 32);
      } else {
        tma2d(map, dst + w * TILEB, bar, kc, t * 32, pol);
        tma2d(map, dst + w * TILEB + 4096, bar, kc, INTER + t * 32, pol);
      }
    } else {
      if (pf) tma_prefetch_l2(map, kc, t * 64);
      else tma2d(map, dst + w * TILEB, bar, kc, t * 64, pol);
    }
  }
}

DEV void producer(const Params& p, Smem& s, int c, int rank) {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  const uint32_t nslot = static_cast<uint32_t>(s.nslot);
  const uint32_t look = static_cast<uint32_t>(p.look);
  UIt ld{0, 0, 0, 0, 0, 0, 0, nullptr};
  bool vld = uit_settle(p, c, rank, ld);
  UIt pf = ld;
  bool vpf = vld;
  uint32_t pfi = 0, pu = 0, slot = 0, wpar = 1;  // wpar: parity of the empty phase to wait for (from the 2nd lap)
  const uint32_t pace = static_cast<uint32_t>(DM_PACE);
  uint32_t qslot = 0, qpar = 0;  // the unit issued `pace` units ago
  while (vld) {
    if (pace > 0 && pu >= pace) {
      // pacing: at most `pace` units in flight per CTA (a whole-grid ring refill at a phase end floods the memory
      // queues exactly while the latency-bound handoffs run)
      mbar_wait(&s.full[qslot], qpar);
      if (++qslot == nslot) {
        qslot = 0;
        qpar ^= 1u;
      }
    }
    if (pu >= nslot) {
      const uint32_t par = wpar;
      // the ring is full: pull the next units into L2 while the consumers drain it, then block (suspending
      // try_wait; a tight test_wait spin steals the shared-memory pipe from the consumers' wgmma)
      while (vpf && pfi < pu + look && !mbar_test(&s.empty[slot], par)) {
        if (pfi >= pu) uit_issue(p, pf, true, nullptr, nullptr, 0);
        vpf = uit_next(p, c, rank, pf);
        ++pfi;
      }
      mbar_wait(&s.empty[slot], par);
    }
    if (ld.sl == 0 && ld.u == 0 && ld.ph != PH_QKV) dstamp(p, ld.l, 60 + ld.ph);
    const int t1 = (2 * ld.sl + 1 < ld.nt) ? 1 : 0;
    mbar_expect_tx(&s.full[slot], (1 + t1) * TILEB);
    uit_issue(p, ld, false, s.ring + slot * SLAB, &s.full[slot], pol);
    vld = uit_next(p, c, rank, ld);
    ++pu;
    if (++slot == nslot) {
      slot = 0;
      wpar ^= 1u;
    }
    if (pfi < pu) {
      pf = ld;
      vpf = vld;
      pfi = pu;
    }
  }
}

// ------------------------------------------------------------------ KV gather (warp 9, attention clusters)
struct Split {
  int seqlen, kbase, nu, ntot, nold;
};
DEV Split split_of(int seqlen, int rank) {
  Split sp;
  sp.seqlen = seqlen;
  const int L0 = max(0, seqlen - (BLK + WIN));
  const int nk = max(0, seqlen - L0);
  const int U = (nk + KVU - 1) / KVU;
  const int u0 = rank * U / CS, u1 = (rank + 1) * U / CS;
  sp.nu = u1 - u0;
  sp.kbase = L0 + KVU * u0;
  sp.ntot = max(0, min(KVU * sp.nu, seqlen - sp.kbase));
  sp.nold = max(0, min(sp.ntot, seqlen - BLK - sp.kbase));
  return sp;
}

DEV void gather(const Params& p, Smem& s, int b, int rank, int lane) {
  const Split sp = split_of(s.misc[1], rank);
  for (int i = lane; i < sp.nold; i += 32) s.slot[i] = __ldg(p.page_table + b * p.pt_stride + sp.kbase + i);
  __syncwarp();
  uint8_t* sK = s.kv;
  uint8_t* sV = s.kv + KV_MAXK * HD * 2;
  for (int l = 0; l < p.nl; ++l) {
    if (l > 0) mbar_wait(s.kvempty, (l - 1) & 1);
    const __nv_bfloat16* K = p.L[l].kc;
    const __nv_bfloat16* V = p.L[l].vc;
#pragma unroll 4
    for (int idx = lane; idx < sp.nold * 32; idx += 32) {
      const int key = idx >> 5, part = idx & 31, kv = part >> 4, ch = part & 15;
      const int64_t sl = s.slot[key];
      const __nv_bfloat16* src = (kv ? V : K) + sl * p.kv_stride + ch * 8;
      uint8_t* dst = (kv ? sV : sK) + key * 256 + swz(ch, key) * 16;
      cp_async16(dst, src);
    }
    // rows past the split's keys (inside its tiles) must be finite: zero them
    for (int idx = lane; idx < (KVU * sp.nu - sp.ntot) * 32; idx += 32) {
      const int key = sp.ntot + (idx >> 5), part = idx & 31, kv = part >> 4, ch = part & 15;
      *reinterpret_cast<uint4*>((kv ? sV : sK) + key * 256 + swz(ch, key) * 16) = make_uint4(0u, 0u, 0u, 0u);
    }
    cp_async_arrive_noinc(s.kvfull);
    __syncwarp();
    if (lane == 0) mbar_arrive(s.kvfull);
  }
}

// ------------------------------------------------------------------ all-reduce (tp_arnq, token CTA, 256 threads)
// kar: index of the all-reduce in this launch (0..2*NL-1); mode 0: fp8 + scale -> x buffer, mode 1: bf16 -> out.
DEV void allreduce(const Params& p, Smem& s, int t, int kar, int ctid, unsigned E) {
  const int l = kar >> 1;
  const int rank = p.rank;
  const int lane = ctid & 31, warp = ctid >> 5;
  uint8_t* const mine = p.peer[rank];
  const int phase = (s.calls0[t] + kar) & 1;
  float* s_v = reinterpret_cast<float*>(s.greg);
  __nv_bfloat16* s_h = reinterpret_cast<__nv_bfloat16*>(s.greg + H * 4);
  const __nv_bfloat16* gamma = (kar & 1) ? (l + 1 < p.nl ? p.L[l + 1].ln_in : p.final_norm) : p.L[l].ln_post;
  const bool final_out = (kar == 2 * p.nl - 1);
  __nv_bfloat16* const rrow = p.hidden + static_cast<int64_t>(t) * H;
  const int col = ctid * 4;
  constexpr int ACT = SL / 4;  // 192
  // ---- owner reduce of the 8 pushed partial slices, in rank order ----
  uint2 hb = make_uint2(0u, 0u);
  if (ctid < ACT) {
    uint2 w[R];
    uint8_t* slot0 = mine + RS_OFF + (static_cast<int64_t>(phase) * MT + t) * R * RS_SLOT + col * 2;
#pragma unroll
    for (int q = 0; q < R; ++q) w[q] = ld_vol_v2(slot0 + q * RS_SLOT);
    bool done;
    do {
      done = true;
#pragma unroll
      for (int q = 0; q < R; ++q) {
        if (w[q].x == 0u || w[q].y == 0u) {
          w[q] = ld_vol_v2(slot0 + q * RS_SLOT);
          done = false;
        }
      }
    } while (!done);
#pragma unroll
    for (int q = 0; q < R; ++q) st_v2(slot0 + q * RS_SLOT, make_uint2(0u, 0u));
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int q = 0; q < R; ++q) {
      acc[0] += bf_lo(w[q].x); acc[1] += bf_hi(w[q].x); acc[2] += bf_lo(w[q].y); acc[3] += bf_hi(w[q].y);
    }
    hb = make_uint2(pack_bf2(acc[0], acc[1]), pack_bf2(acc[2], acc[3]));
    *reinterpret_cast<uint2*>(s_h + rank * SL + col) = hb;
  }
  // ---- all-gather push of the owned slice ----
  {
    const uint32_t o0 = __shfl_down_sync(0xffffffffu, hb.x, 1);
    const uint32_t o1 = __shfl_down_sync(0xffffffffu, hb.y, 1);
    if (ctid < ACT && (ctid & 1) == 0) {
      const uint4 d = make_uint4(nz(hb.x), nz(hb.y), nz(o0), nz(o1));
      const int v = ctid >> 1;
#pragma unroll 1
      for (int di = 0; di < R - 1; ++di) {
        const int q = di + (di >= rank);
        st_v4(p.peer[q] + AG_OFF + ((static_cast<int64_t>(phase) * MT + t) * R + rank) * AG_SLOT + v * 16, d);
      }
    }
  }
  // ---- receive the other slices (concurrent polls) ----
  {
    constexpr int NV = (R - 1) * (SL / 8);  // 672
    constexpr int PER = (NV + NCT - 1) / NCT;
    uint4 d[PER];
    uint8_t* sl[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const int k = j * NCT + ctid;
      sl[j] = nullptr;
      d[j] = make_uint4(1u, 1u, 1u, 1u);
      if (k < NV) {
        const int di = k / (SL / 8);
        const int v = k - di * (SL / 8);
        const int q = di + (di >= rank);
        sl[j] = mine + AG_OFF + ((static_cast<int64_t>(phase) * MT + t) * R + q) * AG_SLOT + v * 16;
        d[j] = ld_vol_v4(sl[j]);
      }
    }
    bool done;
    do {
      done = true;
#pragma unroll
      for (int j = 0; j < PER; ++j) {
        if (d[j].x == 0u || d[j].y == 0u || d[j].z == 0u || d[j].w == 0u) {
          d[j] = ld_vol_v4(sl[j]);
          done = false;
        }
      }
    } while (!done);
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const int k = j * NCT + ctid;
      if (k < NV) {
        st_v4(sl[j], make_uint4(0u, 0u, 0u, 0u));
        const int di = k / (SL / 8);
        const int v = k - di * (SL / 8);
        const int q = di + (di >= rank);
        uint4 w = d[j];
        w.x = w.x == 0x80000000u ? 0u : w.x;
        w.y = w.y == 0x80000000u ? 0u : w.y;
        w.z = w.z == 0x80000000u ? 0u : w.z;
        w.w = w.w == 0x80000000u ? 0u : w.w;
        *reinterpret_cast<uint4*>(s_h + q * SL + v * 8) = w;
      }
    }
  }
  bar_sync(1, NCT);
  // ---- x = f32(h) + f32(r) (unrounded) -> s_v; residual_out = bf16(x) ----
#pragma unroll
  for (int cc = 0; cc < 3; ++cc) {
    const int e = cc * NCT * 8 + ctid * 8;
    const uint4 hv = *reinterpret_cast<const uint4*>(s_h + e);
    const uint4 rv = *reinterpret_cast<const uint4*>(rrow + e);
    const uint32_t hw[4] = {hv.x, hv.y, hv.z, hv.w};
    const uint32_t rw[4] = {rv.x, rv.y, rv.z, rv.w};
    float v[8];
    uint32_t o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[2 * j] = bf_lo(hw[j]) + bf_lo(rw[j]);
      v[2 * j + 1] = bf_hi(hw[j]) + bf_hi(rw[j]);
      o[j] = pack_bf2(v[2 * j], v[2 * j + 1]);
    }
    *reinterpret_cast<float4*>(s_v + e) = make_float4(v[0], v[1], v[2], v[3]);
    *reinterpret_cast<float4*>(s_v + e + 4) = make_float4(v[4], v[5], v[6], v[7]);
    *reinterpret_cast<uint4*>(rrow + e) = make_uint4(o[0], o[1], o[2], o[3]);
  }
  bar_sync(1, NCT);
  // ---- sum of squares in flashinfer's order (64 virtual threads, 2 warps) ----
  if (ctid < 64) {
    float acc = 0.f;
#pragma unroll
    for (int v = 0; v < 12; ++v) {
      const float4* vp = reinterpret_cast<const float4*>(s_v + 512 * v + 8 * ctid);
      const float4 a = vp[0], b = vp[1];
      acc = fmaf(a.x, a.x, acc); acc = fmaf(a.y, a.y, acc); acc = fmaf(a.z, a.z, acc); acc = fmaf(a.w, a.w, acc);
      acc = fmaf(b.x, b.x, acc); acc = fmaf(b.y, b.y, acc); acc = fmaf(b.z, b.z, acc); acc = fmaf(b.w, b.w, acc);
    }
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, off));
    if (lane == 0) s.red[warp] = acc;
  }
  bar_sync(1, NCT);
  const float rstd = rsqrtf(__fadd_rn(__fdiv_rn(__fadd_rn(s.red[0], s.red[1]), static_cast<float>(H)), p.eps));
  float tmax = 0.f;
  uint32_t yb[3][4];
#pragma unroll
  for (int cc = 0; cc < 3; ++cc) {
    const int e = cc * NCT * 8 + ctid * 8;
    const float4 a = *reinterpret_cast<const float4*>(s_v + e);
    const float4 b = *reinterpret_cast<const float4*>(s_v + e + 4);
    const float xv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    const uint4 gv = *reinterpret_cast<const uint4*>(gamma + e);
    const uint32_t gw[4] = {gv.x, gv.y, gv.z, gv.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float n0 = __fmul_rn(__fmul_rn(xv[2 * j], rstd), __fadd_rn(bf_lo(gw[j]), 0.0f));
      const float n1 = __fmul_rn(__fmul_rn(xv[2 * j + 1], rstd), __fadd_rn(bf_hi(gw[j]), 0.0f));
      yb[cc][j] = pack_bf2(n0, n1);
      tmax = fmaxf(tmax, fmaxf(fabsf(bf_lo(yb[cc][j])), fabsf(bf_hi(yb[cc][j]))));
    }
  }
  if (final_out) {
#pragma unroll
    for (int cc = 0; cc < 3; ++cc) {
      const int e = cc * NCT * 8 + ctid * 8;
      *reinterpret_cast<uint4*>(p.out + static_cast<int64_t>(t) * H + e) = make_uint4(yb[cc][0], yb[cc][1], yb[cc][2], yb[cc][3]);
    }
  } else {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, o));
    if (lane == 0) s.red[8 + warp] = tmax;
    bar_sync(1, NCT);
    float am = 0.f;
#pragma unroll
    for (int w = 0; w < NCT / 32; ++w) am = fmaxf(am, s.red[8 + w]);
    const float scale = fmaxf(am, 1e-10f) / FP8_MAX;
    const float inv = 1.f / scale;
    const int buf = (kar + 1) & 1;  // x event kar + 1
    uint8_t* xrow = p.xq + (static_cast<int64_t>(buf) * MT + t) * H;
#pragma unroll
    for (int cc = 0; cc < 3; ++cc) {
      const int e = cc * NCT * 8 + ctid * 8;
      uint32_t q[2];
#pragma unroll
      for (int i = 0; i < 2; ++i)
        q[i] = static_cast<uint32_t>(pack_e4m3x2(fminf(fmaxf(bf_lo(yb[cc][2 * i]) * inv, -FP8_MAX), FP8_MAX),
                                                 fminf(fmaxf(bf_hi(yb[cc][2 * i]) * inv, -FP8_MAX), FP8_MAX))) |
               (static_cast<uint32_t>(pack_e4m3x2(fminf(fmaxf(bf_lo(yb[cc][2 * i + 1]) * inv, -FP8_MAX), FP8_MAX),
                                                  fminf(fmaxf(bf_hi(yb[cc][2 * i + 1]) * inv, -FP8_MAX), FP8_MAX)))
                << 16);
      *reinterpret_cast<uint2*>(xrow + e) = make_uint2(q[0], q[1]);
    }
    if (ctid == 0) p.xs[buf * MT + t] = scale;
  }
  bar_sync(1, NCT);
  if (ctid == 0) red_release(p.cnt + C_X, 1u);
}

// ---- layer-0 input RMSNorm + per-token quant (rmsnorm_quant numerics), token CTA ----
DEV void phase0(const Params& p, Smem& s, int t, int ctid) {
  constexpr int PER = H / NCT;  // 24
  const int lane = ctid & 31, warp = ctid >> 5;
  const __nv_bfloat16* row = p.hidden + static_cast<int64_t>(t) * H;
  const __nv_bfloat16* gamma = p.L[0].ln_in;
  float v[PER], gm[PER];
  float ssq = 0.f;
#pragma unroll
  for (int cc = 0; cc < PER; cc += 8) {
    const int e = (cc / 8) * NCT * 8 + ctid * 8;
    const uint4 xv = *reinterpret_cast<const uint4*>(row + e);
    const uint4 gv = *reinterpret_cast<const uint4*>(gamma + e);
    const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
    const uint32_t gw[4] = {gv.x, gv.y, gv.z, gv.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[cc + 2 * j] = bf_lo(xw[j]);
      v[cc + 2 * j + 1] = bf_hi(xw[j]);
      gm[cc + 2 * j] = bf_lo(gw[j]);
      gm[cc + 2 * j + 1] = bf_hi(gw[j]);
    }
  }
#pragma unroll
  for (int i = 0; i < PER; ++i) ssq += v[i] * v[i];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffffu, ssq, o);
  if (lane == 0) s.red[warp] = ssq;
  bar_sync(1, NCT);
  float tot = 0.f;
#pragma unroll
  for (int w = 0; w < NCT / 32; ++w) tot += s.red[w];
  const float rstd = rsqrtf(tot / static_cast<float>(H) + p.eps);
  float amx = 0.f;
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    v[i] = bf16r(v[i] * rstd * gm[i]);
    amx = fmaxf(amx, fabsf(v[i]));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amx = fmaxf(amx, __shfl_xor_sync(0xffffffffu, amx, o));
  if (lane == 0) s.red[8 + warp] = amx;
  bar_sync(1, NCT);
  float m = 0.f;
#pragma unroll
  for (int w = 0; w < NCT / 32; ++w) m = fmaxf(m, s.red[8 + w]);
  const float sc = fmaxf(m, 1e-10f) / FP8_MAX;
  const float inv = 1.f / sc;
  uint8_t* xrow = p.xq + static_cast<int64_t>(t) * H;  // buffer 0
#pragma unroll
  for (int cc = 0; cc < PER; cc += 8) {
    const int e = (cc / 8) * NCT * 8 + ctid * 8;
    uint32_t w[2] = {0u, 0u};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const __nv_fp8_e4m3 q(fminf(fmaxf(v[cc + j] * inv, -FP8_MAX), FP8_MAX));
      w[j >> 2] |= static_cast<uint32_t>(*reinterpret_cast<const uint8_t*>(&q)) << (8 * (j & 3));
    }
    *reinterpret_cast<uint2*>(xrow + e) = make_uint2(w[0], w[1]);
  }
  if (ctid == 0) p.xs[t] = sc;
  bar_sync(1, NCT);
  if (ctid == 0) red_release(p.cnt + C_X, 1u);
}

// ------------------------------------------------------------------ GEMM phase
template <int PH>
DEV void gemm(const Params& p, Smem& s, int l, int c, int rank, int ctid, unsigned E, int& cslot, uint32_t& cpar, int& gph,
               int& xph) {
  constexpr bool S8 = (PH == PH_QKV || PH == PH_GU);
  constexpr int NU = S8 ? 6 : 4;
  constexpr int NS = S8 ? 8 : 4;
  constexpr int ROWS = S8 ? 8 : 16;
  const int g = S8 ? 0 : (rank >> 2);
  const int src = S8 ? rank : (rank & 3);
  const int nt = p.plan.n[PH][c][g];
  const int16_t* tl = p.plan.tile[PH][c][g];
  const int wg = ctid >> 7, ww = (ctid >> 5) & 3, lane = ctid & 31;
  uint8_t* X = s.greg;
  float* recv = reinterpret_cast<float*>(s.greg + XREG);
  const float* sw = p.L[l].sw[PH];
  if (ctid == 0) mbar_expect_tx(s.recvbar, static_cast<uint32_t>(nt) * 8192u);
  if (ctid < MT) s.amax[ctid] = 0u;
  // the per-channel weight scales of the rows this rank finalizes -> s.red (free during GEMM phases): tile k at k * 16
  // (qkv: 8 rows, gate_up: 4 gate + 4 up, o / down: 16 rows); loaded now, consumed after the receive wait
  if (ctid >= 64 && ctid < 64 + nt * 16) {
    const int k = (ctid - 64) >> 4, j = (ctid - 64) & 15;
    float v = 0.f;
    if constexpr (PH == PH_QKV) {
      if (j < 8) v = __ldg(sw + tl[k] * 64 + rank * 8 + j);
    } else if constexpr (PH == PH_GU) {
      if (j < 8) v = __ldg(sw + (j < 4 ? 0 : INTER) + tl[k] * 32 + rank * 4 + (j & 3));
    } else {
      v = __ldg(sw + tl[k] * 64 + (rank & 3) * 16 + j);
    }
    s.red[k * 16 + j] = v;
  }
  // ---- X operand ----
  if constexpr (S8) {
    const int e = 2 * l + (PH == PH_GU ? 1 : 0);
    const unsigned tgt = (E * 11u + static_cast<unsigned>(e) + 1u) * 32u;
    if (ctid < 32) {
      if (ctid == 0) wait_ge(p.cnt + C_X, tgt);
      __syncwarp();
      if (ctid == 0) {
        uint64_t pol;
        asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
        fence_proxy_async_global();
        fence_proxy_async_shared();
        mbar_expect_tx(s.xbar, NU * XU);
        const CUtensorMap* xm = p.maps + NL * 4 + (e & 1);
#pragma unroll 1
        for (int u = 0; u < NU; ++u) tma2d(xm, X + u * XU, s.xbar, (src * NU + u) * 128, 0, pol);
      }
      s.sx[ctid] = __uint_as_float(ld_acq(reinterpret_cast<const unsigned*>(p.xs + (e & 1) * MT + ctid)));
    }
    mbar_wait(s.xbar, xph & 1);
    ++xph;
    if (ctid == 0) dstamp(p, l, 40 + PH * 4);
  } else {
    const unsigned tgt = (E * 5u + static_cast<unsigned>(l) + 1u) * (PH == PH_O ? 64u : static_cast<unsigned>(GRID));
    if (ctid < 32) {
      if (ctid == 0) wait_ge(p.cnt + (PH == PH_O ? C_ATT : C_ACT), tgt);
      __syncwarp();
      const unsigned* am = p.amax + ((((E & 1u) * 2u + (PH == PH_DN ? 1u : 0u)) * NL + l) * MT);
      const float a = __uint_as_float(ld_acq(am + ctid));
      const float sc = (PH == PH_O) ? a / FP8_MAX : fmaxf(a, 1e-10f) / FP8_MAX;
      s.sx[ctid] = sc;
      s.sinv[ctid] = sc == 0.f ? 0.f : 1.f / sc;
    }
    bar_sync(1, NCT);
    const __nv_bfloat16* sp = (PH == PH_O ? p.att : p.act) + src * 512;
#pragma unroll 2
    for (int i = ctid; i < MT * 32; i += NCT) {
      const int t = i >> 5, ch = i & 31, u = ch >> 3, cc = ch & 7;
      const float inv = s.sinv[t];
      const uint4 a0 = ldcg_v4(sp + t * 2048 + ch * 16);
      const uint4 a1 = ldcg_v4(sp + t * 2048 + ch * 16 + 8);
      const uint32_t w[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
      uint32_t q[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint16_t lo = pack_e4m3x2(fminf(fmaxf(bf_lo(w[2 * j]) * inv, -FP8_MAX), FP8_MAX),
                                        fminf(fmaxf(bf_hi(w[2 * j]) * inv, -FP8_MAX), FP8_MAX));
        const uint16_t hi = pack_e4m3x2(fminf(fmaxf(bf_lo(w[2 * j + 1]) * inv, -FP8_MAX), FP8_MAX),
                                        fminf(fmaxf(bf_hi(w[2 * j + 1]) * inv, -FP8_MAX), FP8_MAX));
        q[j] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
      }
      *reinterpret_cast<uint4*>(X + u * XU + t * 128 + ((cc ^ (t & 7)) * 16)) = make_uint4(q[0], q[1], q[2], q[3]);
    }
    fence_proxy_async_shared();
    bar_sync(1, NCT);
    if (ctid == 0) dstamp(p, l, 40 + PH * 4);
  }
  // ---- K loop over the slabs (two 64-row tiles per 16-KB ring slab, one per warpgroup) ----
  const int nslab = (nt + 1) >> 1;
#pragma unroll 1
  for (int sl = 0; sl < nslab; ++sl) {
    const int k = 2 * sl + wg;
    const bool have = k < nt;
    float acc[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) acc[r] = 0.f;
    int prev = -1;
#pragma unroll 1
    for (int u = 0; u < NU; ++u) {
      // ring position kept incrementally (a runtime modulo here sits on the wgmma issue path: +140 cycles/unit)
      const int slot = cslot;
      const uint32_t par = cpar;
      if (++cslot == s.nslot) {
        cslot = 0;
        cpar ^= 1u;
      }
      mbar_wait(&s.full[slot], par);
      // issued by both warpgroups even when this slab has no tile for this one (it then multiplies the unused half
      // of the slab and the result is dropped): a predicated wgmma makes ptxas serialize every wgmma (C7518)
      {
        const uint64_t da = make_desc(s.ring + slot * SLAB + wg * TILEB);
        const uint64_t db = make_desc(X + u * XU);
        wgmma_fence();
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) wgmma_m64n32k32_e4m3(acc, da + 2 * kk, db + 2 * kk);
        wgmma_commit();
      }
      if (prev >= 0) {
        wgmma_wait<1>();
        __syncwarp();
        if (lane == 0) mbar_arrive(&s.empty[prev]);
      }
      prev = slot;
    }
    wgmma_wait<0>();
    __syncwarp();
    if (lane == 0) mbar_arrive(&s.empty[prev]);
    if (have) {
      const uint32_t rbase = smem_u32(recv);
      const uint32_t rbar = smem_u32(s.recvbar);
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int rho = 16 * ww + (lane >> 2) + 8 * hf;
        int owner, lrow;
        if constexpr (PH == PH_QKV) {
          owner = rho >> 3;
          lrow = rho & 7;
        } else if constexpr (PH == PH_GU) {
          owner = (rho & 31) >> 2;
          lrow = ((rho >> 5) << 2) | (rho & 3);
        } else {
          owner = (g << 2) | (rho >> 4);
          lrow = rho & 15;
        }
        // registers {0,1,4,5,8,9,12,13} (+2 for the second row half) = tokens 2q, 2q+1, 2q+8, .. -> positions q*8 + 0..7
        const uint32_t off = static_cast<uint32_t>((((k * NS + src) * ROWS + lrow) * 32 + (lane & 3) * 8) * 4);
        const uint32_t dst = mapa(rbase + off, owner), bar = mapa(rbar, owner);
        const int r0 = 2 * hf;
        st_async_v4(dst, acc[r0], acc[r0 + 1], acc[r0 + 4], acc[r0 + 5], bar);
        st_async_v4(dst + 16, acc[r0 + 8], acc[r0 + 9], acc[r0 + 12], acc[r0 + 13], bar);
      }
    }
  }
  // ---- finalize the rows this rank owns ----
  if (ctid == 0) dstamp(p, l, 41 + PH * 4);
  mbar_wait(s.recvbar, gph & 1);
  ++gph;
  bar_sync(1, NCT);
  if (ctid == 0) dstamp(p, l, 42 + PH * 4);
  if constexpr (PH == PH_QKV) {
    for (int i = ctid; i < nt * MT; i += NCT) {
      const int k = i >> 5, t = i & 31;
      float v[8];
#pragma unroll
      for (int rr = 0; rr < 8; ++rr) v[rr] = 0.f;
#pragma unroll
      for (int q = 0; q < 8; ++q)
#pragma unroll
        for (int rr = 0; rr < 8; ++rr) v[rr] += recv[((k * 8 + q) * 8 + rr) * 32 + tpos(t)];
      const int n0 = tl[k] * 64 + rank * 8;
      const float sxv = s.sx[t];
      uint32_t w[4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        w[j] = pack_bf2(v[2 * j] * s.red[k * 16 + 2 * j] * sxv, v[2 * j + 1] * s.red[k * 16 + 2 * j + 1] * sxv);
#if DM_LL_QKV
      // data = flag: Lamport-marked words into this layer's parity buffer; the other parity (read by the previous layer,
      // written by the next one -- by this same thread) is cleared now
      const unsigned gl = E * 5u + static_cast<unsigned>(l);
      st_v4(p.qkv + (static_cast<int64_t>(gl & 1u) * MT + t) * QKV_N + n0, make_uint4(nz(w[0]), nz(w[1]), nz(w[2]), nz(w[3])));
      st_v4(p.qkv + (static_cast<int64_t>((gl + 1u) & 1u) * MT + t) * QKV_N + n0, make_uint4(0u, 0u, 0u, 0u));
#else
      *reinterpret_cast<uint4*>(p.qkv + t * QKV_N + n0) = make_uint4(w[0], w[1], w[2], w[3]);
#endif
    }
#if !DM_LL_QKV
    if (ctid == 0) dstamp(p, l, 120);
    bar_sync(1, NCT);
    if (ctid == 0) {
      dstamp(p, l, 121);
      for (int k = 0; k < nt; ++k) red_release(p.cnt + C_QKV + tl[k], 1u);
      dstamp(p, l, 122);
    }
#endif
  } else if constexpr (PH == PH_GU) {
    for (int i = ctid; i < nt * MT; i += NCT) {
      const int k = i >> 5, t = i & 31;
      float v[8];
#pragma unroll
      for (int rr = 0; rr < 8; ++rr) v[rr] = 0.f;
#pragma unroll
      for (int q = 0; q < 8; ++q)
#pragma unroll
        for (int rr = 0; rr < 8; ++rr) v[rr] += recv[((k * 8 + q) * 8 + rr) * 32 + tpos(t)];
      const int cg = tl[k] * 32 + rank * 4;
      const float sxv = s.sx[t];
      float a[4];
      float amx = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float gg = bf16r(v[j] * s.red[k * 16 + j] * sxv);
        const float uu = bf16r(v[4 + j] * s.red[k * 16 + 4 + j] * sxv);
        a[j] = bf16r(gg / (1.f + __expf(-gg)) * uu);
        amx = fmaxf(amx, fabsf(a[j]));
      }
      *reinterpret_cast<uint2*>(p.act + t * INTER + cg) = make_uint2(pack_bf2(a[0], a[1]), pack_bf2(a[2], a[3]));
      atomicMax(s.amax + t, __float_as_uint(amx));
    }
    bar_sync(1, NCT);
    if (ctid < 32) {
      unsigned* am = p.amax + ((((E & 1u) * 2u + 1u) * NL + l) * MT);
      const unsigned v = s.amax[ctid];
      if (v) atomicMax(am + ctid, v);
      __syncwarp();
      if (ctid == 0) {
        __threadfence();  // the warp's amax atomics before the release
        red_release(p.cnt + C_ACT, 1u);
      }
    }
  } else {
    const int kar = 2 * l + (PH == PH_DN ? 1 : 0);
    for (int i = ctid; i < nt * MT * 2; i += NCT) {
      const int k = i >> 6, t = (i >> 1) & 31, h = i & 1;
      float v[8];
#pragma unroll
      for (int rr = 0; rr < 8; ++rr) v[rr] = 0.f;
#pragma unroll
      for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int rr = 0; rr < 8; ++rr) v[rr] += recv[((k * 4 + q) * 16 + 8 * h + rr) * 32 + tpos(t)];
      const int n0 = tl[k] * 64 + (rank & 3) * 16 + 8 * h;
      const float sxv = s.sx[t];
      uint32_t w[4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        w[j] = nz(pack_bf2(v[2 * j] * s.red[k * 16 + 8 * h + 2 * j] * sxv, v[2 * j + 1] * s.red[k * 16 + 8 * h + 2 * j + 1] * sxv));
      const int q = n0 / SL;
      const int phase = (s.calls0[t] + kar) & 1;
      st_v4(p.peer[q] + RS_OFF + ((static_cast<int64_t>(phase) * MT + t) * R + p.rank) * RS_SLOT + (n0 - q * SL) * 2,
            make_uint4(w[0], w[1], w[2], w[3]));
    }
  }
}

// ------------------------------------------------------------------ attention (clusters 0..7)
constexpr int CSREG = 50 * 1024;  // inside the GEMM region: cos|sin rows of the request's 8 positions (fp32 [8][64])
                                  // + q/k norm weights (bf16 [2][128])

// q/k RMSNorm + NeoX RoPE (first 64 dims) of a 32-dim quarter (qq) of a head row. xv: the quarter's raw bf16 values,
// w: the norm weights (smem, bf16 [128]), cs: cos|sin of the row's position (smem, fp32 [64]). Numerics of
// dflash_qknr_kv (normalized values rounded to bf16 before the rotation, rotation in fp32, rounded once).
DEV void qk_norm_rope(const uint4* xv, const __nv_bfloat16* w, const float* cs, int qq, float eps, uint32_t* out) {
  float x[32];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t vw[4] = {xv[i].x, xv[i].y, xv[i].z, xv[i].w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      x[i * 8 + 2 * j] = bf_lo(vw[j]);
      x[i * 8 + 2 * j + 1] = bf_hi(vw[j]);
    }
  }
  float ss = 0.f;
#pragma unroll
  for (int d = 0; d < 32; ++d) ss += x[d] * x[d];
  ss += __shfl_xor_sync(0xffffffffu, ss, 1);
  ss += __shfl_xor_sync(0xffffffffu, ss, 2);
  const float inv = 1.f / sqrtf(ss / static_cast<float>(HD) + eps);
  float xn[32];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint4 gv = *reinterpret_cast<const uint4*>(w + qq * 32 + i * 8);
    const uint32_t gw[4] = {gv.x, gv.y, gv.z, gv.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      xn[i * 8 + 2 * j] = bf16r(x[i * 8 + 2 * j] * inv * bf_lo(gw[j]));
      xn[i * 8 + 2 * j + 1] = bf16r(x[i * 8 + 2 * j + 1] * inv * bf_hi(gw[j]));
    }
  }
  float oth[32];
#pragma unroll
  for (int d = 0; d < 32; ++d) oth[d] = __shfl_xor_sync(0xffffffffu, xn[d], 1);
  if (qq < 2) {
#pragma unroll
    for (int d = 0; d < 32; ++d) {
      const float c = cs[d], sn = cs[32 + d];
      // quarter 0 holds x1 (dims d), quarter 1 holds x2 (dims 32 + d)
      xn[d] = (qq == 0) ? bf16r(xn[d] * c - oth[d] * sn) : bf16r(xn[d] * c + oth[d] * sn);
    }
  }
#pragma unroll
  for (int j = 0; j < 16; ++j) out[j] = pack_bf2(xn[2 * j], xn[2 * j + 1]);
}

DEV void attention(const Params& p, Smem& s, int l, int c, int rank, int ctid, unsigned E) {
  const int b = c >> 1, hg = c & 1;
  const int warp = ctid >> 5, lane = ctid & 31;
  const LayerP& LP = p.L[l];
  const Split sp = split_of(s.misc[1], rank);
  uint8_t* sK = s.kv;
  uint8_t* sV = s.kv + KV_MAXK * HD * 2;
  float* comb = reinterpret_cast<float*>(s.greg);
  uint8_t* sQ = s.greg + QREG;
  float* scs = reinterpret_cast<float*>(s.greg + CSREG);
  __nv_bfloat16* sqn = reinterpret_cast<__nv_bfloat16*>(s.greg + CSREG + 2048);
  __nv_bfloat16* skn = sqn + HD;
  if (ctid == 0) mbar_expect_tx(s.combbar, CS * COMB_SRC_BYTES);
  // my GEMM region (aliased by the combine buffer) is free: tell every cluster peer
  if (ctid < CS) mbar_arrive_remote(mapa(smem_u32(s.combok), ctid));
  // stage what does not depend on this layer's qkv while the tiles are produced
  for (int i = ctid; i < BLK * 64; i += NCT) {
    const long long pos = s.pos[i >> 6];
    scs[i] = p.cs_f32 ? __ldg(reinterpret_cast<const float*>(p.cos_sin) + pos * p.cs_stride + (i & 63))
                      : __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(p.cos_sin)[pos * p.cs_stride + (i & 63)]);
  }
  if (ctid < 16) {
    reinterpret_cast<uint4*>(sqn)[ctid] = __ldg(reinterpret_cast<const uint4*>(LP.qn) + ctid);
  } else if (ctid < 32) {
    reinterpret_cast<uint4*>(skn)[ctid - 16] = __ldg(reinterpret_cast<const uint4*>(LP.kn) + (ctid - 16));
  } else if (ctid < 40) {
    const int h = 8 * hg + ctid - 32;
    float v = -INFINITY;
    if (LP.sinks) v = p.sink_f32 ? __ldg(reinterpret_cast<const float*>(LP.sinks) + h)
                                 : __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(LP.sinks)[h]);
    s.red[32 + ctid - 32] = v;
  }
#if !DM_LL_QKV
  // wait for this head half's q tiles + the k / v tiles
  if (ctid >= 64 && ctid < 84) {
    const int i = ctid - 64;
    const int tile = i < 16 ? 16 * hg + i : 16 + i;
    wait_ge(p.cnt + C_QKV + tile, (E * 5u + static_cast<unsigned>(l) + 1u) * 8u);
  }
#endif
  bar_sync(1, NCT);
  if (ctid == 0) dstamp(p, l, 56);
  // q rows (token-major: row = t * 8 + h) and the block's new k / v rows (warps 0 / 1) -> normalized + rotated
  {
    const int row = ctid >> 2, qq = ctid & 3;
    const int t = row >> 3, h = row & 7;
    const bool store = (hg == 0 && rank == 0);  // writes the block rows to the KV cache
    const int first_new = sp.seqlen - BLK - sp.kbase;  // local index of block row 0
    const bool need = (first_new + BLK > sp.nold) && (first_new < sp.ntot);
    const bool kvw = (warp < 2) && (need || store);
    const int tk = lane >> 2;
    uint4 xq[4], xk[4];
#if DM_LL_QKV
    if (ctid == 0) dstamp(p, l, 123);
    const __nv_bfloat16* qbuf = p.qkv + static_cast<int64_t>((E * 5u + static_cast<unsigned>(l)) & 1u) * MT * QKV_N;
#pragma unroll
    for (int i = 0; i < 4; ++i) xq[i] = poll_ll_v4(qbuf + (BLK * b + t) * QKV_N + (8 * hg + h) * HD + qq * 32 + i * 8);
    if (kvw) {
#pragma unroll
      for (int i = 0; i < 4; ++i)
        xk[i] = poll_ll_v4(qbuf + (BLK * b + tk) * QKV_N + (warp == 0 ? NHQ : NHQ + 1) * HD + qq * 32 + i * 8);
    }
#else
    const __nv_bfloat16* qsrc = p.qkv + (BLK * b + t) * QKV_N + (8 * hg + h) * HD + qq * 32;
#pragma unroll
    for (int i = 0; i < 4; ++i) xq[i] = ldcg_v4(qsrc + i * 8);
    if (kvw) {
      const __nv_bfloat16* ksrc = p.qkv + (BLK * b + tk) * QKV_N + (warp == 0 ? NHQ : NHQ + 1) * HD + qq * 32;
#pragma unroll
      for (int i = 0; i < 4; ++i) xk[i] = ldcg_v4(ksrc + i * 8);
    }
#endif
    if (ctid == 0) {
      asm volatile("" ::"r"(xq[0].x), "r"(xq[3].w));
      dstamp(p, l, 124);
    }
    uint32_t o[16];
    qk_norm_rope(xq, sqn, scs + t * 64, qq, p.qk_eps, o);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      *reinterpret_cast<uint4*>(sQ + row * 256 + swz(4 * qq + i, row) * 16) = make_uint4(o[4 * i], o[4 * i + 1], o[4 * i + 2], o[4 * i + 3]);
    if (kvw) {
      uint8_t* sdst;
      __nv_bfloat16* cdst;
      if (warp == 0) {
        qk_norm_rope(xk, skn, scs + tk * 64, qq, p.qk_eps, o);
        sdst = sK;
        cdst = LP.kc;
      } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint32_t vw[4] = {xk[i].x, xk[i].y, xk[i].z, xk[i].w};
#pragma unroll
          for (int j = 0; j < 4; ++j) o[4 * i + j] = pack_bf2(bf_lo(vw[j]) * p.v_scale, bf_hi(vw[j]) * p.v_scale);
        }
        sdst = sV;
        cdst = LP.vc;
      }
      const int i = first_new + tk;
      if (i >= 0 && i < sp.ntot) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
          *reinterpret_cast<uint4*>(sdst + i * 256 + swz(4 * qq + j, i) * 16) = make_uint4(o[4 * j], o[4 * j + 1], o[4 * j + 2], o[4 * j + 3]);
      }
      const long long slot = s.loc[tk];
      if (store && slot != 0) {
        uint4* dst = reinterpret_cast<uint4*>(cdst + slot * p.kv_stride + qq * 32);
#pragma unroll
        for (int j = 0; j < 4; ++j) dst[j] = make_uint4(o[4 * j], o[4 * j + 1], o[4 * j + 2], o[4 * j + 3]);
      }
    }
  }
  if (ctid == 0) dstamp(p, l, 57);
  mbar_wait(s.kvfull, l & 1);
  bar_sync(1, NCT);
  if (ctid == 0) dstamp(p, l, 58);
  // ---- online softmax over 16-key units: warp w -> 16 query rows (tokens 2(w&3), 2(w&3)+1 x 8 heads); the two
  // warpgroups split the split's key units (WG1 takes the last half, merged into WG0 through the freed KV tile) ----
  {
    const int rt = warp & 3, half = warp >> 2;
    const int hsplit = (sp.nu + 1) >> 1;
    const bool ksplit = (sp.nu - hsplit) >= 4;  // the merge buffer is the KV tile of WG1's units (>= 4 x 4 KB each)
    const int ub = half == 0 ? 0 : hsplit;
    const int ue = half == 0 ? (ksplit ? hsplit : sp.nu) : (ksplit ? sp.nu : hsplit);
    const uint32_t qb = smem_u32(sQ), kb = smem_u32(sK), vb = smem_u32(sV);
    uint32_t aq[8][4];
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      const int row = 16 * rt + (lane & 15);
      const int ch = 2 * kk + (lane >> 4);
      ldsm_x4(qb + row * 256 + swz(ch, row) * 16, aq[kk]);
    }
    float O[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) O[j][0] = O[j][1] = O[j][2] = O[j][3] = 0.f;
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
    const int lo0 = sp.seqlen - (BLK + WIN) + 2 * rt;
    const float scale = p.sm_scale;
#pragma unroll 1
    for (int u = ub; u < ue; ++u) {
      float sc[2][4];
#pragma unroll
      for (int jt = 0; jt < 2; ++jt) sc[jt][0] = sc[jt][1] = sc[jt][2] = sc[jt][3] = 0.f;
      {
        const int key = 16 * u + (lane & 7) + 8 * (lane >> 4);
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
          const int ch = 2 * kk + ((lane >> 3) & 1);
          uint32_t bb[4];
          ldsm_x4(kb + key * 256 + swz(ch, key) * 16, bb);
          mma_bf16(sc[0], aq[kk], bb[0], bb[1]);
          mma_bf16(sc[1], aq[kk], bb[2], bb[3]);
        }
      }
      float mu0 = m0, mu1 = m1;
#pragma unroll
      for (int jt = 0; jt < 2; ++jt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int key = sp.kbase + 16 * u + 8 * jt + (lane & 3) * 2 + (e & 1);
          const bool ok = (key < sp.seqlen) && (key >= lo0 + (e >> 1));
          const float v = ok ? sc[jt][e] * scale : -INFINITY;
          sc[jt][e] = v;
          if (e < 2) mu0 = fmaxf(mu0, v); else mu1 = fmaxf(mu1, v);
        }
      }
      mu0 = fmaxf(mu0, __shfl_xor_sync(0xffffffffu, mu0, 1));
      mu0 = fmaxf(mu0, __shfl_xor_sync(0xffffffffu, mu0, 2));
      mu1 = fmaxf(mu1, __shfl_xor_sync(0xffffffffu, mu1, 1));
      mu1 = fmaxf(mu1, __shfl_xor_sync(0xffffffffu, mu1, 2));
      const float a0 = __expf(m0 - mu0), a1 = __expf(m1 - mu1);
      m0 = mu0;
      m1 = mu1;
      l0 *= a0;
      l1 *= a1;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        O[j][0] *= a0; O[j][1] *= a0; O[j][2] *= a1; O[j][3] *= a1;
      }
#pragma unroll
      for (int jt = 0; jt < 2; ++jt) {
        sc[jt][0] = __expf(sc[jt][0] - m0);
        sc[jt][1] = __expf(sc[jt][1] - m0);
        sc[jt][2] = __expf(sc[jt][2] - m1);
        sc[jt][3] = __expf(sc[jt][3] - m1);
        l0 += sc[jt][0] + sc[jt][1];
        l1 += sc[jt][2] + sc[jt][3];
      }
      uint32_t a[4];
      a[0] = pack_bf2(sc[0][0], sc[0][1]);
      a[1] = pack_bf2(sc[0][2], sc[0][3]);
      a[2] = pack_bf2(sc[1][0], sc[1][1]);
      a[3] = pack_bf2(sc[1][2], sc[1][3]);
      {
        const int key = 16 * u + (lane & 7) + 8 * ((lane >> 3) & 1);
#pragma unroll
        for (int dd = 0; dd < 8; ++dd) {
          const int ch = 2 * dd + (lane >> 4);
          uint32_t bb[4];
          ldsm_x4_t(vb + key * 256 + swz(ch, key) * 16, bb);
          mma_bf16(O[2 * dd], a, bb[0], bb[1]);
          mma_bf16(O[2 * dd + 1], a, bb[2], bb[3]);
        }
      }
    }
    // WG1 -> WG0 merge (same lane -> (row, dims) mapping in both warpgroups; [chunk][thread] float4 layout)
    {
      float4* oK = reinterpret_cast<float4*>(sK + hsplit * KVU * 256);
      float4* oV = reinterpret_cast<float4*>(sV + hsplit * KVU * 256);
      float4* ml = reinterpret_cast<float4*>(sQ);  // Q is in registers (aq) by now
      const int tw = ctid & 127;
      if (half == 1 && ksplit) {
        bar_sync(3, 128);  // every WG1 warp is done reading the K / V rows the merge buffer overwrites
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          oK[i * 128 + tw] = make_float4(O[i][0], O[i][1], O[i][2], O[i][3]);
          oV[i * 128 + tw] = make_float4(O[8 + i][0], O[8 + i][1], O[8 + i][2], O[8 + i][3]);
        }
        ml[tw] = make_float4(m0, m1, l0, l1);
      }
      bar_sync(1, NCT);
      if (half == 0 && ksplit) {
        const float4 o = ml[tw];
        const float M0 = fmaxf(m0, o.x), M1 = fmaxf(m1, o.y);
        const float a0 = __expf(m0 - M0), b0 = __expf(o.x - M0), a1 = __expf(m1 - M1), b1 = __expf(o.y - M1);
        l0 = l0 * a0 + o.z * b0;
        l1 = l1 * a1 + o.w * b1;
        m0 = M0;
        m1 = M1;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float4 k4 = oK[i * 128 + tw], v4 = oV[i * 128 + tw];
          O[i][0] = O[i][0] * a0 + k4.x * b0; O[i][1] = O[i][1] * a0 + k4.y * b0;
          O[i][2] = O[i][2] * a1 + k4.z * b1; O[i][3] = O[i][3] * a1 + k4.w * b1;
          O[8 + i][0] = O[8 + i][0] * a0 + v4.x * b0; O[8 + i][1] = O[8 + i][1] * a0 + v4.y * b0;
          O[8 + i][2] = O[8 + i][2] * a1 + v4.z * b1; O[8 + i][3] = O[8 + i][3] * a1 + v4.w * b1;
        }
      }
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    // normalized partial + lse -> the combine buffers of the two token ranks (2rt, 2rt+1)
    const float r0 = l0 > 0.f ? 1.f / l0 : 0.f, r1 = l1 > 0.f ? 1.f / l1 : 0.f;
    const float lse0 = l0 > 0.f ? m0 + __logf(l0) : -INFINITY;
    const float lse1 = l1 > 0.f ? m1 + __logf(l1) : -INFINITY;
    const int g = lane >> 2;
    if (ctid == 0) dstamp(p, l, 59);
    if (half == 0) {
    mbar_wait_cluster(s.combok, l & 1);
    const uint32_t cbase = smem_u32(comb), cbar = smem_u32(s.combbar);
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const int dst = 2 * rt + hf;
      const uint32_t rb = mapa(cbase, dst) + ((rank * 8 + g) * COMB_STRIDE) * 4;
      const uint32_t rbar = mapa(cbar, dst);
      const float rr = hf ? r1 : r0;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int d = 8 * j + (lane & 3) * 2;
        st_async_v2(rb + d * 4, O[j][2 * hf] * rr, O[j][2 * hf + 1] * rr, rbar);
      }
      if ((lane & 3) == 0) st_async_b32(rb + HD * 4, hf ? lse1 : lse0, rbar);
    }
    }
  }
  // KV region no longer read: the gather warp may fetch the next layer's keys
  bar_sync(1, NCT);
  if (ctid == 0) mbar_arrive(s.kvempty);
  // ---- combine the 8 splits for token `rank` of request b (heads 8hg..8hg+7) ----
  mbar_wait(s.combbar, l & 1);
  if (ctid == 0) dstamp(p, l, 60);
  {
    const int h = ctid >> 5, d4 = ctid & 31;
    float lse[CS];
    float mx = -INFINITY;
#pragma unroll
    for (int q = 0; q < CS; ++q) {
      lse[q] = comb[(q * 8 + h) * COMB_STRIDE + HD];
      mx = fmaxf(mx, lse[q]);
    }
    const bool empty = (mx == -INFINITY);
    const float sink = s.red[32 + h];
    const bool has_sink = LP.sinks != nullptr;
    if (has_sink) mx = fmaxf(mx, sink);
    mx = (mx > -INFINITY) ? mx : 0.f;
    float den = 0.f;
    float o[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int q = 0; q < CS; ++q) {
      const float w = __expf(lse[q] - mx);
      den += w;
      const float4 v = *reinterpret_cast<const float4*>(comb + (q * 8 + h) * COMB_STRIDE + 4 * d4);
      o[0] += v.x * w; o[1] += v.y * w; o[2] += v.z * w; o[3] += v.w * w;
    }
    if (has_sink && !empty) den += __expf(sink - mx);
    const float rden = 1.f / (den > 0.f ? den : 1.f);
    const uint32_t w0 = pack_bf2(o[0] * rden, o[1] * rden), w1 = pack_bf2(o[2] * rden, o[3] * rden);
    const int row = BLK * b + rank;
    *reinterpret_cast<uint2*>(p.att + row * 2048 + hg * 1024 + h * HD + 4 * d4) = make_uint2(w0, w1);
    float amx = fmaxf(fmaxf(fabsf(bf_lo(w0)), fabsf(bf_hi(w0))), fmaxf(fabsf(bf_lo(w1)), fabsf(bf_hi(w1))));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) amx = fmaxf(amx, __shfl_xor_sync(0xffffffffu, amx, off));
    if (lane == 0) s.red[16 + warp] = amx;
  }
  bar_sync(1, NCT);
  if (ctid == 0) {
    float amx = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) amx = fmaxf(amx, s.red[16 + w]);
    unsigned* am = p.amax + ((((E & 1u) * 2u + 0u) * NL + l) * MT);
    if (amx > 0.f) atomicMax(am + BLK * b + rank, __float_as_uint(amx));
    red_release(p.cnt + C_ATT, 1u);
  }
}

// ------------------------------------------------------------------ the kernel
template <bool kUsePDL>
__global__ void __launch_bounds__(THREADS, 1) dflash_mega_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rank = static_cast<int>(cluster_rank());
  const int c = static_cast<int>(cluster_id_x());
  const bool is_att = c < NATT;
  const bool is_ar = c >= AR0 && c < AR0 + 4;
  Smem s;
  {
    uint8_t* m = smem;
    s.full = reinterpret_cast<uint64_t*>(m);
    s.empty = s.full + MAX_SLOT;
    s.xbar = s.empty + MAX_SLOT;
    s.recvbar = s.xbar + 1;
    s.kvfull = s.xbar + 2;
    s.kvempty = s.xbar + 3;
    s.combbar = s.xbar + 4;
    s.combok = s.xbar + 5;
    s.calls0 = reinterpret_cast<int*>(m + 320);
    s.sx = reinterpret_cast<float*>(m + 448);
    s.sinv = reinterpret_cast<float*>(m + 576);
    s.amax = reinterpret_cast<unsigned*>(m + 704);
    s.red = reinterpret_cast<float*>(m + 832);
    s.slot = reinterpret_cast<int*>(m + 1088);
    s.misc = reinterpret_cast<int*>(m + 1664);
    s.pos = reinterpret_cast<long long*>(m + 1728);
    s.loc = reinterpret_cast<long long*>(m + 1792);
    s.nslot = p.plan.nslot[c];
    s.ring = m + MISC_BYTES;
    s.greg = s.ring + s.nslot * SLAB;
    s.kv = s.greg + GREG_ATT;
  }
  if (tid == 0) {
    if ((smem_u32(smem) & 1023u) != 0u) __trap();
    for (int i = 0; i < MAX_SLOT; ++i) {
      mbar_init(&s.full[i], 1);
      mbar_init(&s.empty[i], 8);
    }
    mbar_init(s.xbar, 1);
    mbar_init(s.recvbar, 1);
    mbar_init(s.kvfull, 33);
    mbar_init(s.kvempty, 1);
    mbar_init(s.combbar, 1);
    mbar_init(s.combok, CS);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  cluster_sync_all();
  if constexpr (THREADS == 384) {
    if (warp >= 8) {
      asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(DM_REG_LO) : "memory");
    } else {
      asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(DM_REG_HI) : "memory");
    }
  }
  if (warp >= 10) {
  } else if (warp == 8) {
    // weights do not depend on the previous kernel: start streaming before griddepcontrol.wait
    if (lane == 0) producer(p, s, c, rank);
  } else {
    device::PDLWaitPrimary<kUsePDL>();
    if (warp == 9) {
      if (is_att && lane < 8) {
        s.misc[1] = __ldg(p.seqlens + (c >> 1));
      }
      __syncwarp();
      if (is_att) gather(p, s, c >> 1, rank, lane);
    } else {
      const int ctid = tid;
      if (ctid < MT) s.calls0[ctid] = p.calls[ctid];
      if (ctid == 0) s.misc[0] = static_cast<int>(ld_acq(p.cnt + C_LAUNCH));
      if (is_att && ctid >= 32 && ctid < 32 + BLK) {
        const int i = ctid - 32, row = BLK * (c >> 1) + i;
        s.pos[i] = p.pos64 ? reinterpret_cast<const long long*>(p.positions)[row]
                           : static_cast<long long>(reinterpret_cast<const int*>(p.positions)[row]);
        s.loc[i] = p.loc64 ? reinterpret_cast<const long long*>(p.loc)[row]
                           : static_cast<long long>(reinterpret_cast<const int*>(p.loc)[row]);
      }
      if (is_att && ctid == 64) s.misc[1] = __ldg(p.seqlens + (c >> 1));
      bar_sync(1, NCT);
      const unsigned E = static_cast<unsigned>(s.misc[0]);
      if (c == 0 && rank == 0) {
        // zero the other parity's amax slots for the next launch
        unsigned* z = p.amax + ((E & 1u) ^ 1u) * 2u * NL * MT;
        for (int i = ctid; i < 2 * NL * MT; i += NCT) z[i] = 0u;
      }
      if (ctid == 0) stamp(p, 0);
      int cslot = 0;
      uint32_t cpar = 0;
      int gph = 0, xph = 0;
      const int t_ar = is_ar ? (c - AR0) * CS + rank : -1;
      if (is_ar) phase0(p, s, t_ar, ctid);
      for (int l = 0; l < p.nl; ++l) {
        gemm<PH_QKV>(p, s, l, c, rank, ctid, E, cslot, cpar, gph, xph);
        if (ctid == 0) stamp(p, 1 + 8 * l);
#if !DM_NO_ATT
        if (is_att) attention(p, s, l, c, rank, ctid, E);
#endif
        if (ctid == 0) stamp(p, 2 + 8 * l);
        gemm<PH_O>(p, s, l, c, rank, ctid, E, cslot, cpar, gph, xph);
        if (ctid == 0) stamp(p, 3 + 8 * l);
        if (is_ar) allreduce(p, s, t_ar, 2 * l, ctid, E);
        if (ctid == 0) stamp(p, 4 + 8 * l);
        gemm<PH_GU>(p, s, l, c, rank, ctid, E, cslot, cpar, gph, xph);
        if (ctid == 0) stamp(p, 5 + 8 * l);
        gemm<PH_DN>(p, s, l, c, rank, ctid, E, cslot, cpar, gph, xph);
        if (ctid == 0) stamp(p, 6 + 8 * l);
        if (is_ar) allreduce(p, s, t_ar, 2 * l + 1, ctid, E);
        if (ctid == 0) stamp(p, 7 + 8 * l);
      }
      if (is_ar && ctid == 0) p.calls[t_ar] = s.calls0[t_ar] + 2 * p.nl;
      if (c == 0 && rank == 0 && ctid == 0) {
        if (p.nl < NL) {  // keep the per-launch counter increments of a full 5-layer launch
          const unsigned d = static_cast<unsigned>(NL - p.nl);
          atomicAdd(p.cnt + C_X, 2u * d * 32u);
          atomicAdd(p.cnt + C_ATT, d * 64u);
          atomicAdd(p.cnt + C_ACT, d * static_cast<unsigned>(GRID));
          for (int i = 0; i < 36; ++i) atomicAdd(p.cnt + C_QKV + i, d * 8u);
        }
        atomicAdd(p.cnt + C_LAUNCH, 1u);
      }
      device::PDLTriggerSecondary<kUsePDL>();
    }
  }
  __syncwarp();
  cluster_sync_all();
}

inline CUtensorMap make_map_2d(const void* base, uint64_t inner, uint64_t outer, uint64_t row_bytes, uint32_t box_inner,
                               uint32_t box_outer) {
  CUtensorMap map;
  uint64_t dim[2] = {inner, outer};
  uint64_t stride[1] = {row_bytes};
  uint32_t box[2] = {box_inner, box_outer};
  uint32_t elem_stride[2] = {1, 1};
  const CUresult rc = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dim, stride, box,
                                             elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                             CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  host::RuntimeCheck(rc == CUDA_SUCCESS, "dflash_mega: cuTensorMapEncodeTiled failed");
  return map;
}

}  // namespace dflash_mega

template <bool kUsePDL>
struct DFlashMega {
  static int64_t params_bytes() { return static_cast<int64_t>(sizeof(dflash_mega::Params)); }

  // maps: cuda uint8 [(NL*4 + 2) * 128]; wptr: host int64 [NL*4] fp8 weight bases ([N][K] contiguous rows) in the order
  // (qkv, o, gate_up, down) per layer; xq: cuda uint8 [2][32][H] x buffers.
  static void build_maps(tvm::ffi::TensorView maps, tvm::ffi::TensorView wptr, tvm::ffi::TensorView xq) {
    using namespace host;
    using namespace dflash_mega;
    RuntimeCheck(maps.numel() >= (NL * 4 + 2) * 128 && wptr.numel() == NL * 4 && xq.numel() >= 2 * MT * H);
    const int64_t* wp = static_cast<const int64_t*>(wptr.data_ptr());
    CUtensorMap h[NL * 4 + 2];
    const uint64_t N[4] = {QKV_N, H, 2 * INTER, H};
    const uint64_t K[4] = {H, INTER, H, INTER};
    for (int l = 0; l < NL; ++l)
      for (int ph = 0; ph < 4; ++ph)
        h[l * 4 + ph] = make_map_2d(reinterpret_cast<const void*>(wp[l * 4 + ph]), K[ph], N[ph], K[ph], 128,
                                    ph == PH_GU ? 32 : 64);
    for (int b = 0; b < 2; ++b)
      h[NL * 4 + b] = make_map_2d(static_cast<const uint8_t*>(xq.data_ptr()) + static_cast<int64_t>(b) * MT * H, H, MT, H, 128, MT);
    RuntimeDeviceCheck(cudaMemcpy(maps.data_ptr(), h, sizeof(h), cudaMemcpyHostToDevice));
  }

  // blob: host uint8 [sizeof(Params)]; ptrs: host int64 (layout below); ivals: host int64; fvals: host float32;
  // plan: host int16 [4*NCL*2*MAXT_PH + 4*NCL*2 + NCL]
  static void fill(tvm::ffi::TensorView blob, tvm::ffi::TensorView ptrs, tvm::ffi::TensorView ivals,
                   tvm::ffi::TensorView fvals, tvm::ffi::TensorView plan) {
    using namespace host;
    using namespace dflash_mega;
    RuntimeCheck(blob.numel() >= static_cast<int64_t>(sizeof(Params)));
    constexpr int NPL = 11;
    RuntimeCheck(ptrs.numel() == NL * NPL + 12 + R, "dflash_mega: pointer table size");
    RuntimeCheck(ivals.numel() == 7 && fvals.numel() == 4);
    RuntimeCheck(plan.numel() == 4 * NCL * 2 * MAXT_PH + 4 * NCL * 2 + NCL);
    const int64_t* pp = static_cast<const int64_t*>(ptrs.data_ptr());
    const int64_t* iv = static_cast<const int64_t*>(ivals.data_ptr());
    const float* fv = static_cast<const float*>(fvals.data_ptr());
    Params prm;
    std::memset(&prm, 0, sizeof(prm));
    for (int l = 0; l < NL; ++l) {
      const int64_t* q = pp + l * NPL;
      for (int ph = 0; ph < 4; ++ph) prm.L[l].sw[ph] = reinterpret_cast<const float*>(q[ph]);
      prm.L[l].ln_in = reinterpret_cast<const __nv_bfloat16*>(q[4]);
      prm.L[l].ln_post = reinterpret_cast<const __nv_bfloat16*>(q[5]);
      prm.L[l].qn = reinterpret_cast<const __nv_bfloat16*>(q[6]);
      prm.L[l].kn = reinterpret_cast<const __nv_bfloat16*>(q[7]);
      prm.L[l].sinks = reinterpret_cast<const void*>(q[8]);
      prm.L[l].kc = reinterpret_cast<__nv_bfloat16*>(q[9]);
      prm.L[l].vc = reinterpret_cast<__nv_bfloat16*>(q[10]);
    }
    const int64_t* q = pp + NL * NPL;
    prm.maps = reinterpret_cast<const CUtensorMap*>(q[0]);
    prm.final_norm = reinterpret_cast<const __nv_bfloat16*>(q[1]);
    prm.cos_sin = reinterpret_cast<const void*>(q[2]);
    prm.xq = reinterpret_cast<uint8_t*>(q[3]);
    prm.xs = reinterpret_cast<float*>(q[4]);
    prm.qkv = reinterpret_cast<__nv_bfloat16*>(q[5]);
    prm.att = reinterpret_cast<__nv_bfloat16*>(q[6]);
    prm.act = reinterpret_cast<__nv_bfloat16*>(q[7]);
    prm.amax = reinterpret_cast<unsigned*>(q[8]);
    prm.cnt = reinterpret_cast<unsigned*>(q[9]);
    prm.calls = reinterpret_cast<int*>(q[10]);
    prm.stamps = reinterpret_cast<unsigned long long*>(q[11]);
    for (int i = 0; i < R; ++i) prm.peer[i] = reinterpret_cast<uint8_t*>(q[12 + i]);
    prm.rank = static_cast<int>(iv[0]);
    prm.cs_stride = iv[1];
    prm.cs_f32 = static_cast<int>(iv[2]);
    prm.sink_f32 = static_cast<int>(iv[3]);
    prm.kv_stride = iv[4];
    prm.nl = static_cast<int>(iv[5]);
    prm.look = static_cast<int>(iv[6]);
    RuntimeCheck(prm.nl >= 1 && prm.nl <= NL);
    prm.eps = fv[0];
    prm.qk_eps = fv[1];
    prm.v_scale = fv[2];
    prm.sm_scale = fv[3];
    const int16_t* pl = static_cast<const int16_t*>(plan.data_ptr());
    std::memcpy(prm.plan.tile, pl, sizeof(prm.plan.tile));
    std::memcpy(prm.plan.n, pl + 4 * NCL * 2 * MAXT_PH, sizeof(prm.plan.n));
    std::memcpy(prm.plan.nslot, pl + 4 * NCL * 2 * MAXT_PH + 4 * NCL * 2, sizeof(prm.plan.nslot));
    for (int cc = 0; cc < NCL; ++cc) {
      const int ns = prm.plan.nslot[cc];
      const int greg = cc < NATT ? GREG_ATT : GREG_FULL;
      const int kv = cc < NATT ? KV_BYTES : 0;
      RuntimeCheck(ns >= 2 && ns <= MAX_SLOT && MISC_BYTES + ns * SLAB + greg + kv <= SMEM_BYTES,
                   "dflash_mega: ring does not fit");
      for (int ph = 0; ph < 4; ++ph)
        for (int g = 0; g < 2; ++g) {
          const int n = prm.plan.n[ph][cc][g];
          RuntimeCheck(n >= 0 && n <= MAXT_PH && (cc >= NATT || n <= 4), "dflash_mega: plan tile count");
        }
    }
    std::memcpy(blob.data_ptr(), &prm, sizeof(prm));
  }

  static void run(tvm::ffi::TensorView blob, tvm::ffi::TensorView hidden, tvm::ffi::TensorView out,
                  tvm::ffi::TensorView positions, tvm::ffi::TensorView seqlens, tvm::ffi::TensorView page_table,
                  tvm::ffi::TensorView loc) {
    using namespace host;
    using namespace dflash_mega;
    RuntimeCheck(blob.numel() >= static_cast<int64_t>(sizeof(Params)));
    RuntimeCheck(hidden.ndim() == 2 && hidden.size(0) == MT && hidden.size(1) == H && hidden.stride(0) == H);
    RuntimeCheck(out.ndim() == 2 && out.size(0) == MT && out.size(1) == H && out.stride(0) == H);
    RuntimeCheck(positions.numel() == MT && loc.numel() == MT && seqlens.numel() == NREQ);
    RuntimeCheck(seqlens.dtype().bits == 32 && page_table.dtype().bits == 32 && page_table.ndim() == 2 &&
                 page_table.size(0) == NREQ && page_table.stride(1) == 1);
    Params prm;
    std::memcpy(&prm, blob.data_ptr(), sizeof(prm));
    prm.hidden = static_cast<__nv_bfloat16*>(hidden.data_ptr());
    prm.out = static_cast<__nv_bfloat16*>(out.data_ptr());
    prm.positions = positions.data_ptr();
    prm.pos64 = positions.dtype().bits == 64 ? 1 : 0;
    prm.seqlens = static_cast<const int*>(seqlens.data_ptr());
    prm.page_table = static_cast<const int*>(page_table.data_ptr());
    prm.pt_stride = page_table.stride(0);
    prm.loc = loc.data_ptr();
    prm.loc64 = loc.dtype().bits == 64 ? 1 : 0;
    auto kernel = dflash_mega_kernel<kUsePDL>;
    static bool attr_set = false;
    if (!attr_set) {
      RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
      attr_set = true;
    }
    LaunchKernel(dim3(GRID), dim3(THREADS), hidden.device(), SMEM_BYTES)
        .enable_cluster(dim3(CS, 1, 1))
        .enable_pdl(kUsePDL)(kernel, prm);
  }

  // max co-resident 8-CTA clusters of this kernel (must be >= 15)
  static int64_t max_clusters(tvm::ffi::TensorView any) {
    using namespace host;
    using namespace dflash_mega;
    auto kernel = dflash_mega_kernel<kUsePDL>;
    RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(GRID);
    cfg.blockDim = dim3(THREADS);
    cfg.dynamicSmemBytes = SMEM_BYTES;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = CS;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    int n = 0;
    RuntimeDeviceCheck(cudaOccupancyMaxActiveClusters(&n, kernel, &cfg));
    (void)any;
    return n;
  }
};

}  // namespace sglang
