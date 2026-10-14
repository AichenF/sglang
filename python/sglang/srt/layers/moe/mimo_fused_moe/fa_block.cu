// goal11 "fa2": the full-attention layers' tail of MiMo-V2.x at TP8, M = 32 (4 requests x 8 verify tokens), sm_90a, in ONE kernel:
//   RoPE (Q in registers; the 8 new K/V rows built from the qkv rows straight into the KV stage that holds them, and stored to the cache)
//   + split-KV verify attention (fa_verify_decode.cuh's CTA: cp.async gather producer warpgroup, 2 wgmma consumer warpgroups)
//   + the split combine (in-kernel, fa_verify_combine_kernel's arithmetic: bitwise the same attention output)
//   + o_proj (attn_block.cu's cluster-pair K-split o_proj: W staged into this CTA's smem -- 5 k-tiles during the attention, the rest into
//     the freed KV ring -- the partner's fp32 partial over DSMEM)
//   + the push of the o_proj partial into the fused MoE kernel's entry reduce-scatter buffers (blk's FMOE_BLK_PREPUSH protocol).
// Grid = 132 CTAs = 66 cluster pairs, 1 CTA / SM. CTA c: attention split sp = c % 33 of request b = c / 33; c < 128 also combines rows
// [4c, 4c + 4) of the 512 (request, token, head) rows and computes o_proj columns [96 (c / 2) + 48 h, + 48) (h = cluster rank) of all 32 tokens.
// Cross-CTA handoffs (one flag per CTA on its own 128-B line, value = epoch = fused MoE work[2] + 1, never reset):
//   attention partial stored -> flags_att[c]; combined rows stored -> flags_cmb[c].
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstring>

namespace fab {

constexpr int W = 8, HQ = 16, ROWS = W * HQ, D = 192, DV = 128, BN = 64;   // verify tokens, heads, rows / request, dims, keys / stage
constexpr int NREQ = 4, M = W * NREQ, DIM = 6144, KO = HQ * DV, QKVW = HQ * D + D + DV;   // 32 tokens, hidden, o_proj K 2048, qkv row 3392
constexpr int QOFF_K = HQ * D;                        // 3072: the new K (then V) within a qkv row
constexpr int NSPLIT = 33, NCTA = NREQ * NSPLIT;      // 132 CTAs
constexpr int KBLK = BN * 128, KSTAGE = (D / 64) * KBLK, VSTAGE = (DV / 64) * KBLK, STAGE_BYTES = KSTAGE + VSTAGE;   // 24 + 16 KB
#ifndef FAB_STAGES
#define FAB_STAGES 4
#endif
constexpr int STAGES = FAB_STAGES;
constexpr int NTHREADS = 384;
#ifndef FAB_PREGS
#define FAB_PREGS 88
#endif
constexpr int PRODUCER_REGS = FAB_PREGS, CONSUMER_REGS = FAB_PREGS == 72 ? 216 : 208;   // 128 x 88 + 256 x 208 = 64 K (72 / 216: fa_verify's)
// o_proj (attn_block.cu geometry): 64 cluster pairs x 96 columns, CTA h = K half h (heads 8h .. 8h + 7), 48 columns finalized per CTA
constexpr int CL = 2, PCOLS = 96, COLS = PCOLS / CL, NOP = CL * (DIM / PCOLS);   // 128 o_proj CTAs
constexpr int KT = 64, NKT = KO / KT / CL;                                     // 16 k-tiles of 64 per CTA
constexpr int WT_BYTES = PCOLS * KT * 2, BT_BYTES = M * KT * 2;                // 12 KB W k-tile (96 rows x 128 B), 4 KB attention k-tile
#ifndef FAB_NA
#define FAB_NA 6
#endif
constexpr int NA = FAB_NA;                                                     // attention k-tile ring slots
#ifndef NACC
#define NACC 4
#endif
constexpr int CROWS = 4;                                                       // combine rows per CTA (128 x 4 = 512)
constexpr int FLAG_STRIDE = 32;
// smem (from a 1024-aligned base): barriers | W k-tiles 0..15 | A ring | partner partial ; the KV ring sits at the TOP and overlaps W k-tiles
// >= NW_EARLY and the A ring / partner partial (those are written only after this CTA's attention is done)
constexpr int OFF_BAR = 0;
constexpr int PTR = 16;                // page-table ring: the 64 entries of one tile per slot (TMA)
constexpr int OFF_W = 1024;
constexpr int SMEM_TOP = 231424;
constexpr int OFF_KV = SMEM_TOP - STAGES * STAGE_BYTES;
constexpr int OFF_A = OFF_W + NKT * WT_BYTES;
constexpr int OFF_X = OFF_A + NA * BT_BYTES;
// combine CTAs' stash for the 8 new keys' contribution to their 4 rows (filled by bulk copies at kernel start, read after the attention):
// Q of (token, 4 heads) | K | V rows of the request's 8 new tokens | their cos_sin rows | 4 x 8 scores
constexpr int MS_Q = 0, MS_KV = MS_Q + 4 * D * 2, MS_CS = MS_KV + W * (D + DV) * 2, MS_SC = MS_CS + W * 128, MS_BYTES = MS_SC + 4 * W * 4;
constexpr int MS_TX = MS_SC;   // bytes delivered by the bulk copies
constexpr int NW_EARLY = (OFF_KV - OFF_W - PTR * BN * 4 - MS_BYTES) / WT_BYTES;   // W k-tiles below the KV ring next to the rings / stash
constexpr int OFF_PT = OFF_W + NW_EARLY * WT_BYTES;   // page-table ring (attention phase only)
constexpr int OFF_MS = OFF_PT + PTR * BN * 4;         // stash (until the combine's mini part); W k-tile NW_EARLY lands over both after it
static_assert(OFF_MS + MS_BYTES <= OFF_KV && OFF_MS + MS_BYTES <= OFF_W + (NW_EARLY + 1) * WT_BYTES && OFF_MS % 128 == 0, "page-table ring / stash");

#ifndef FAB_KV_EVF
#define FAB_KV_EVF 1       // 1: KV gathers carry an L2 evict_first policy (lab: -1 us; the KV stream is read once)
#endif
#ifndef FAB_DISCARD
#define FAB_DISCARD 1      // 1: the combine discards (L2, no write-back) the 8.6 MB of split partials it consumed: dead data that otherwise
#endif                     // gets written back to HBM during the next kernel (the fused MoE's FC1 ramp)
#ifndef FAB_WD
#define FAB_WD 2           // W k-tiles in flight per CTA after its attention (pacing: unpaced, the early finishers' W stream delayed the
                           // late CTAs' KV by ~3 us; 1: same, 4: +1.2 us)
#endif
constexpr int SMEM_ALLOC = SMEM_TOP + 1024;
static_assert(OFF_X + COLS * M * 4 <= SMEM_TOP && OFF_KV % 1024 == 0 && OFF_A % 1024 == 0 && SMEM_ALLOC <= 232448 && NW_EARLY >= 0, "fab smem");
static_assert(NOP <= NCTA && NOP * CROWS == NREQ * ROWS, "fab geometry");
// fused MoE entry receive buffer (fused_moe_wgmma.cu, FMOE_TP_OWNER_REDUCE): IN_RS at message 0 of every rank's prologue buffer,
// [src 8][owner-local token 4][slice 8][128], message m of slice k = elements 768 k + 6 m .. + 5 of the token's partial row.
constexpr int TP_NDEV = 8, TP_KS = DIM / TP_NDEV, TP_RS_ROW = TP_KS / 6, TP_TOK_PER_RANK = M / TP_NDEV;
static_assert(TP_KS == 768 && TP_RS_ROW == 128 && COLS % 12 == 0 && TP_KS % COLS == 0, "rs geometry");
__device__ __forceinline__ size_t tp_rs_idx(int src, int t, int k, int m) {
    return ((size_t)((src * TP_TOK_PER_RANK + t / TP_NDEV) * TP_NDEV + k) * TP_RS_ROW + m);
}
constexpr int MODE_POS64 = 1, MODE_LOC64 = 2, MODE_FULL_LOCAL = 4, MODE_NO_PUSH = 8, MODE_ATT_ONLY = 16;
constexpr int MODE_DBG_NOPOLL = 32, MODE_DBG_NOW = 64;   // diagnostics (wrong results): no combine-flag polls / no W loads
constexpr int MODE_DBG_QLATE_SHIFT = 16;                 // diagnostics (goal12 P3 bound): mode bits 16..25 = Q delay in 16-ns units -- the consumer
                                                         // warpgroups wait until (CTA entry + delay) after griddepcontrol.wait (KV producer unaffected)
constexpr float LOG2E = 1.4426950408889634f;

struct Params {
    CUtensorMap tm_pt;                 // page table [NREQ][pt_width] int32: box {64, 1} (one tile's entries), OOB -> 0
    CUtensorMap tm_w;                  // o_proj weight [6144][2048] bf16: box {64, 96}, SWIZZLE_128B
    CUtensorMap tm_a;                  // combined attention output [32][2048] bf16: box {64, 32}, SWIZZLE_128B
    const __nv_bfloat16* qkv;          // [32][3392] un-rotated qkv rows (the previous kernel's output)
    const __nv_bfloat16* cos_sin;      // [max_pos][64] (cos 0..31 | sin 32..63)
    const void* positions;             // [32] int32 / int64
    const void* loc;                   // [32] KV write slots int32 / int64 (slot 0 = padding, not written: mimo_rope_kv's rule)
    __nv_bfloat16* kc;                 // key cache rows (192 elements, stride ks)
    __nv_bfloat16* vc;                 // value cache rows (128 elements, stride vs)
    long long ks, vs;
    const int* pt;                     // [NREQ][pt_stride] page table (page size 1)
    const int* seqlens;                // [NREQ] (include the 8 verify tokens)
    float* part_o;                     // [NREQ * ROWS][NSPLIT][DV] fp32 normalized split partials
    float* part_lse;                   // [NREQ * ROWS][NSPLIT] fp32 log2-domain LSE (-inf: empty split)
    __nv_bfloat16* attn_out;           // [32][2048] combined attention output (scratch; tm_a points at it)
    unsigned* flags_att;               // [NCTA] attention partial stored (epoch), FLAG_STRIDE apart
    unsigned* flags_cmb;               // [NOP] combined rows stored (epoch)
    __nv_bfloat16* partial;            // [32][6144] o_proj partial: rows of the tokens this rank owns (all rows with MODE_FULL_LOCAL)
    uint4* peer[TP_NDEV];              // fused MoE prologue IPC buffers (IN_RS at message 0)
    const int* work;                   // fused MoE work counters: epoch = work[2] + 1
    unsigned long long* dbg;           // optional [NCTA][32] globaltimer stamps
    int pt_stride, pt_width, my_rank, mode;
    float scale_log2;                  // sm_scale * log2(e)
};

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(cnt) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
    asm volatile("{\n.reg .pred p;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra WAIT_%=;\n}\n" ::"r"(bar), "r"(parity) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory"); }
__device__ __forceinline__ void cp_async_mbar_arrive(uint32_t bar) { asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(bar) : "memory"); }
__device__ __forceinline__ void mbar_expect(uint32_t bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory"); }
__device__ __forceinline__ void cp_async16_pol(uint32_t dst, const void* src, uint64_t pol) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "l"(pol) : "memory");
}
__device__ __forceinline__ uint64_t policy_evict_first() { uint64_t p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p; }
__device__ __forceinline__ void named_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void named_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar) {
    asm volatile("{ .reg .b64 policy;\n createpolicy.fractional.L2::evict_first.b64 policy, 1.0;\n"
                 " cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], policy;\n}"
                 :: "r"(dst), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_load_2d_nohint(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(dst), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ unsigned ld_acquire_gpu(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_release_gpu(unsigned* p, unsigned v) { asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void st_ll(uint4* p, uint4 v) { asm volatile("st.relaxed.sys.global.v4.b32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ uint32_t lds32(uint32_t a) { uint32_t v; asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v; }
__device__ __forceinline__ void sts128(uint32_t a, uint4 v) { asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
__device__ __forceinline__ void sts32f(uint32_t a, float v) { asm volatile("st.shared.f32 [%0], %1;" :: "r"(a), "f"(v) : "memory"); }
__device__ __forceinline__ float lds32f(uint32_t a) { float v; asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(a) : "memory"); return v; }
__device__ __forceinline__ float2 bf2f(uint32_t u) { return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u)); }
__device__ __forceinline__ uint4 lds128(uint32_t a) { uint4 v; asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory"); return v; }
__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory"); return t; }
__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void griddep_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ unsigned cluster_rank() { unsigned r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ uint32_t mapa(uint32_t a, uint32_t rank) { uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank)); return r; }
__device__ __forceinline__ void cluster_sync_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;\nbarrier.cluster.wait.aligned;" ::: "memory"); }
__device__ __forceinline__ void st_async_v2(uint32_t addr, float a, float b, uint32_t bar) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];" :: "r"(addr), "f"(a), "f"(b), "r"(bar) : "memory");
}
#define BSTAMP(k) do { if (P.dbg) P.dbg[blockIdx.x * 32 + (k)] = gtimer(); } while (0)

// bf16x2 RoPE pieces with the rounding of the mimo_rope_kv Triton kernel: lo = fma.rn(x, c, -mul.rn(y, s)), hi = fma.rn(y, c, mul.rn(x, s))
__device__ __forceinline__ uint32_t bf2_mul(uint32_t a, uint32_t b) { uint32_t r; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ uint32_t bf2_fma(uint32_t a, uint32_t b, uint32_t c) { uint32_t r; asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
__device__ __forceinline__ uint32_t rope_lo(uint32_t x, uint32_t y, uint32_t c, uint32_t s) { return bf2_fma(x, c, bf2_mul(y, s) ^ 0x80008000u); }
__device__ __forceinline__ uint32_t rope_hi(uint32_t x, uint32_t y, uint32_t c, uint32_t s) { return bf2_fma(y, c, bf2_mul(x, s)); }
__device__ __forceinline__ uint4 rope_lo4(uint4 x, uint4 y, uint4 c, uint4 s) {
    return make_uint4(rope_lo(x.x, y.x, c.x, s.x), rope_lo(x.y, y.y, c.y, s.y), rope_lo(x.z, y.z, c.z, s.z), rope_lo(x.w, y.w, c.w, s.w));
}
__device__ __forceinline__ uint4 rope_hi4(uint4 x, uint4 y, uint4 c, uint4 s) {
    return make_uint4(rope_hi(x.x, y.x, c.x, s.x), rope_hi(x.y, y.y, c.y, s.y), rope_hi(x.z, y.z, c.z, s.z), rope_hi(x.w, y.w, c.w, s.w));
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void fence_f(float (&d)[N]) {
#pragma unroll
    for (int i = 0; i < N; ++i) asm volatile("" : "+f"(d[i])::"memory");
}
template <int N> __device__ __forceinline__ void fence_u(uint32_t (&d)[N]) {
#pragma unroll
    for (int i = 0; i < N; ++i) asm volatile("" : "+r"(d[i])::"memory");
}
// 128B-swizzled smem matrix descriptor (start, LBO, SBO in bytes)
__device__ __forceinline__ uint64_t desc_sw128(uint32_t addr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 62;
    return d;
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) { uint32_t u; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(u) : "f"(hi), "f"(lo)); return u; }

// ---- fa_verify_decode.cuh's wgmma wrappers (claude_tmp/goal11/attn/kern/gen_wgmma.py)
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
// o_proj: D[64 W rows][32 tokens] += W[64 x 16] (smem) * A^T (smem), both K-major SW128 (attn_block.cu)
__device__ __forceinline__ void wgmma_m64n32k16(float* d, uint64_t da, uint64_t db, int scale_d) {
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %18, 0;\n"
                 " wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, p, 1, 1, 0, 0;\n}"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                   "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
                 : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ uint64_t desc_sw128_k(uint32_t addr) {   // K-major SW128, 8-row atoms (SBO 1024)
    return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ long long ld_idx(const void* p, int i, bool i64) {
    return i64 ? reinterpret_cast<const long long*>(p)[i] : (long long)reinterpret_cast<const int*>(p)[i];
}

__global__ void __launch_bounds__(NTHREADS, 1) fa_block_kernel(const __grid_constant__ Params P) {
    extern __shared__ uint8_t smem_raw[];
    uint8_t* smem = smem_raw + ((1024u - (sa(smem_raw) & 1023u)) & 1023u);
    const uint32_t sbase = sa(smem);
    // barriers
    const uint32_t kv_full = sbase + OFF_BAR;                 // [STAGES] (128 producer arrivals)
    const uint32_t kv_empty = kv_full + STAGES * 8;           // [STAGES] (256 consumer arrivals)
    const uint32_t wfull = kv_empty + STAGES * 8;             // [NKT] W k-tile landed (each used once)
    const uint32_t afull = wfull + NKT * 8;                   // [NA] attention k-tile landed
    const uint32_t aempty = afull + NA * 8;                   // [NA] attention k-tile consumed (8 consumer warps)
    const uint32_t att_done = aempty + NA * 8;                // the consumers' last KV-ring read retired (8 warps)
    const uint32_t xfull = att_done + 8;                      // the partner's 48 x 32 fp32 partial of my columns landed (st.async)
    const uint32_t bgo = xfull + 8;                           // combined rows visible: the rest of the attention k-tiles may be issued
    const uint32_t ptbar = bgo + 8;                     // [PTR] page-table slot landed (TMA)
    const uint32_t ms_full = ptbar + PTR * 8;                 // the stash landed (bulk copies)
    const uint32_t ms_dead = ms_full + 8;                     // the stash was read for the last time (8 consumer warps): W k-tile NW_EARLY may land
    const uint32_t cmbrel = ms_dead + 8;                      // no users (initialized only)
    const int tid = threadIdx.x, wg = tid >> 7;
    const int c = blockIdx.x;
    const int b = c / NSPLIT, sp = c - b * NSPLIT;
    const bool opj = c < NOP && !(P.mode & MODE_ATT_ONLY);   // this CTA combines rows and computes o_proj columns
    const int h = (int)cluster_rank();
    if (tid == 0) {
        BSTAMP(0);
        if (P.dbg) { unsigned sm_; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_)); P.dbg[blockIdx.x * 32 + 13] = sm_; }
#pragma unroll
        for (int s = 0; s < STAGES; ++s) { mbar_init(kv_full + s * 8, 128); mbar_init(kv_empty + s * 8, 256); }
        for (int i = 0; i < NKT; ++i) mbar_init(wfull + i * 8, 1);
        for (int i = 0; i < NA; ++i) { mbar_init(afull + i * 8, 1); mbar_init(aempty + i * 8, 8); }
        mbar_init(att_done, 8); mbar_init(xfull, 1); mbar_init(bgo, 1);
        for (int i = 0; i < PTR; ++i) mbar_init(ptbar + i * 8, 1);
        mbar_init(ms_full, 1); mbar_init(ms_dead, 8); mbar_init(cmbrel, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        if (opj) mbar_expect(xfull, COLS * M * 4);   // armed before the partner can see the barrier (cluster sync below)
    }
    cluster_sync_relaxed();
    griddep_launch();   // the fused MoE may launch now: it griddepcontrol.waits before it reads anything this kernel writes

    const int seqlen = P.seqlens[b];
    // Tiles [0, nr) hold at least one OLD key (keys < new_key0), interleaved over the 33 CTAs of the request (tile = sp + i * NSPLIT: they
    // sweep the KV together -> DRAM page locality); keys >= new_key0 are masked. The 8 new keys (the verify tokens) are never gathered: every
    // combine CTA computes their contribution to its own 4 rows from a stash (RoPE'd K, V of the qkv rows) as a 34th split, and stores them to
    // the cache. (Gathering them needs the rows in the cache first: a cross-thread fence that waited ~3 us behind the SM's in-flight loads; a
    // "mini tile" in one split per request made those 4 CTAs the stragglers in situ, +2.6..3.9 us.)
    const int new_key0 = seqlen - W;
    const int nr = new_key0 > 0 ? (new_key0 + BN - 1) / BN : 0;
    const int ntile_cta = max(0, (nr - sp + NSPLIT - 1) / NSPLIT);
    auto tile_of = [&](int i) { return sp + i * NSPLIT; };

    if (wg == 0) {
        // ============================================================== producer warpgroup
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(PRODUCER_REGS));
        const int warp = tid >> 5, lane = tid & 31;
        const uint64_t kvpol = FAB_KV_EVF ? policy_evict_first() : 0ull;
        auto produce = [&](int it, int slot) {
            const int s = it % STAGES;
            const uint32_t ph = (uint32_t)(it / STAGES) & 1u;
            mbar_wait(kv_empty + s * 8, ph ^ 1u);
            const uint32_t kst = sbase + OFF_KV + s * STAGE_BYTES + warp * 16 * 128;
            const uint32_t vst = kst + KSTAGE;
            const uint32_t koff = (uint32_t)slot * (uint32_t)(P.ks * 2);
            const uint32_t voff = (uint32_t)slot * (uint32_t)(P.vs * 2);
            {
                auto cpk = [&](uint32_t d, const void* g) { if (FAB_KV_EVF) cp_async16_pol(d, g, kvpol); else cp_async16(d, g); };
                // fixed lane roles (fa_verify): K chunks 0-15 of keys 2k, 2k+1; K chunks 16-23 of keys 4k..4k+3; V chunks 0-15 of keys 2k, 2k+1
                const int ca = lane & 15, cb = 16 + (lane & 7);
                const int ha = lane >> 4, hb = lane >> 3;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const int key = 2 * k + ha;
                    const uint32_t ko = __shfl_sync(0xffffffffu, koff, key);
                    const uint32_t vo = __shfl_sync(0xffffffffu, voff, key);
                    const uint32_t sw = (uint32_t)(((ca & 7) ^ (key & 7)) << 4);
                    cpk(kst + (ca >> 3) * KBLK + key * 128 + sw, reinterpret_cast<const char*>(P.kc) + ko + ca * 16);
                    cpk(vst + (ca >> 3) * KBLK + key * 128 + sw, reinterpret_cast<const char*>(P.vc) + vo + ca * 16);
                }
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int key = 4 * k + hb;
                    const uint32_t ko = __shfl_sync(0xffffffffu, koff, key);
                    const uint32_t sw = (uint32_t)(((cb & 7) ^ (key & 7)) << 4);
                    cpk(kst + (cb >> 3) * KBLK + key * 128 + sw, reinterpret_cast<const char*>(P.kc) + ko + cb * 16);
                }
                cp_async_mbar_arrive(kv_full + s * 8);
                return;
            }
        };
        // ONE loop body; the page-table entries of the regular tiles arrive by TMA into a PTR-slot smem ring, PT_AHEAD tiles ahead of use.
        // No slot-free handshake: the 4 producer warps are never more than STAGES positions apart (each stage needs all 4 warps' arrivals
        // before the consumers release it), so a slot refilled PT_AHEAD + STAGES <= PTR positions later is no longer read by anyone.
        constexpr int PT_AHEAD = 8;
        static_assert(PT_AHEAD + STAGES <= PTR, "page-table ring reuse distance");
        const int n = ntile_cta;
        auto pt_issue = [&](int q) {
            if (q < n) {
                const int sl = q % PTR;
                mbar_expect(ptbar + sl * 8, BN * 4);
                tma_load_2d_nohint(sbase + OFF_PT + sl * BN * 4, &P.tm_pt, tile_of(q) * BN, b, ptbar + sl * 8);
            }
        };
        if (warp == 0 && lane == 0) {
            asm volatile("prefetch.tensormap [%0];" :: "l"(reinterpret_cast<uint64_t>(&P.tm_pt)) : "memory");
            for (int q = 0; q < PT_AHEAD; ++q) pt_issue(q);
        }
#pragma unroll 1
        for (int q = 0; q < n; ++q) {
            const int sl = q % PTR;
            mbar_wait(ptbar + sl * 8, (q / PTR) & 1);
            int slot;
            const int tl = tile_of(q);
            const int kl = min(tl * BN + warp * 16 + (lane & 15), new_key0 - 1) - tl * BN;
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(slot) : "r"(sbase + OFF_PT + sl * BN * 4 + kl * 4) : "memory");
            produce(q, slot);
            if (warp == 0 && lane == 0) pt_issue(q + PT_AHEAD);
        }
        asm volatile("cp.async.wait_all;" ::: "memory");
        if (opj) {
            if (warp == 0 && lane == 0) {
                // W k-tiles NW_EARLY.. into the KV ring once the consumers' last read of it retired
                BSTAMP(1);
                const int q = c / CL;
                asm volatile("prefetch.tensormap [%0];" :: "l"(reinterpret_cast<uint64_t>(&P.tm_w)) : "memory");
                if (!(P.mode & MODE_DBG_NOW)) {
                    for (int kt = 0; kt < NW_EARLY; ++kt) {
                        mbar_expect(wfull + kt * 8, WT_BYTES);
                        tma_load_2d(sbase + OFF_W + kt * WT_BYTES, &P.tm_w, h * (KO / CL) + kt * KT, q * PCOLS, wfull + kt * 8);
                    }
                }
                mbar_wait(att_done, 0);
                BSTAMP(2);
                // issue order: k-tiles NW_EARLY + 1 .. 15 (paced), then k-tile NW_EARLY once the stash under it is dead
                const int kt0 = NW_EARLY;
                auto seq = [&](int j) { return j < NW_EARLY ? j : (j + 1 < NKT ? j + 1 : NW_EARLY); };   // j-th issued k-tile
                for (int j = kt0; j < NKT && !(P.mode & MODE_DBG_NOW); ++j) {
                    const int kt = seq(j);
                    if (kt == NW_EARLY) mbar_wait(ms_dead, 0);
                    if (FAB_WD < NKT && j - FAB_WD >= kt0) mbar_wait(wfull + seq(j - FAB_WD) * 8, 0);
                    mbar_expect(wfull + kt * 8, WT_BYTES);
                    tma_load_2d(sbase + OFF_W + kt * WT_BYTES, &P.tm_w, h * (KO / CL) + kt * KT, q * PCOLS, wfull + kt * 8);
                }
            } else if (warp == 1 && lane == 0) {
                // attention k-tiles NA.. as ring slots free up (k-tiles 0..NA-1 are issued by consumer thread 0 right after the flags)
                asm volatile("prefetch.tensormap [%0];" :: "l"(reinterpret_cast<uint64_t>(&P.tm_a)) : "memory");
                mbar_wait(bgo, 0);
                asm volatile("fence.proxy.async.global;" ::: "memory");
                for (int kt = NA; kt < NKT; ++kt) {
                    const int s = kt % NA;
                    mbar_wait(aempty + s * 8, ((kt / NA) - 1) & 1);
                    mbar_expect(afull + s * 8, BT_BYTES);
                    tma_load_2d_nohint(sbase + OFF_A + s * BT_BYTES, &P.tm_a, h * (KO / CL) + kt * KT, 0, afull + s * 8);
                }
            }
        }
        __syncwarp();
    } else {
        // ============================================================== consumer warpgroups
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(CONSUMER_REGS));
        const int cw = wg - 1;                 // 0 / 1: rows 0-63 / 64-127
        const int ct = tid - 128;              // 0..255
        const int warp = (tid >> 5) & 3;       // warp within the warpgroup
        const int lane = tid & 31;
        const int g = lane >> 2, c4 = lane & 3;
        const int tok = cw * 4 + warp;         // this warp's 16 rows are token `tok`, heads 0..15
        const int h0 = g, h1 = g + 8;
        const unsigned long long qlate = (unsigned long long)((P.mode >> MODE_DBG_QLATE_SHIFT) & 0x3ff) * 16ull;
        const unsigned long long t_ent = qlate ? gtimer() : 0ull;
        // Inputs written by EARLIER kernels (positions, cos_sin, loc) are loaded before griddepcontrol.wait -- while this SM has no KV in
        // flight yet -- so that after the wait only the qkv rows remain: one load level (loads queue behind the producer's copies, ~4-5 us
        // per round trip once ~160 KB per SM are in flight).
        const bool pos64 = P.mode & MODE_POS64;
        const long long pos = ld_idx(P.positions, b * W + tok, pos64);
        const __nv_bfloat16* csr = P.cos_sin + pos * 64;
        uint32_t qcv[2][4], qsv[2][4];
#pragma unroll
        for (int kk = 0; kk < 2; ++kk)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int d = kk * 16 + c4 * 2 + ((e & 2) ? 8 : 0);
                qcv[kk][e] = *reinterpret_cast<const uint32_t*>(csr + d);
                qsv[kk][e] = *reinterpret_cast<const uint32_t*>(csr + 32 + d);
            }
        // combine CTAs: the stash for the new keys' contribution to rows [4c, 4c + 4) = request b2, token tokc, heads 4 hgc .. 4 hgc + 3
        // (bulk copies: off the LSU queue, no registers) -- cos_sin rows of the request's 8 new tokens now, the qkv rows after the wait
        const int b2 = c / 32, tokc = (c % 32) / 4, hgc = c % 4;
        long long slotc = 0;
        // one copy per consumer warp (lane 0, warp k = token k): a single thread issuing all of them (8 dependent position loads + 17 bulk
        // ops) delayed its warp's Q -- and so the warpgroup's first QK -- by ~1.8 us in situ. The tx count may go negative before the expect.
        if (opj && lane == 0) {
            if (tok == 0) mbar_expect(ms_full, MS_TX);
            const long long pk = ld_idx(P.positions, b2 * W + tok, pos64);
            bulk_g2s(sbase + OFF_MS + MS_CS + tok * 128, P.cos_sin + pk * 64, 128, ms_full);
        }
        if (opj && hgc == 0 && ct < 40) slotc = ld_idx(P.loc, b2 * W + tokc, P.mode & MODE_LOC64);
        griddep_wait();
        if (qlate) { while (gtimer() < t_ent + qlate) {} }
        const unsigned epoch = (unsigned)__ldcg(P.work + 2) + 1u;
        if (ct == 0) BSTAMP(3);
        uint32_t qf[D / 16][4];
        const __nv_bfloat16* q0 = P.qkv + (size_t)(b * W + tok) * QKVW + (size_t)h0 * D;
        const __nv_bfloat16* q1 = P.qkv + (size_t)(b * W + tok) * QKVW + (size_t)h1 * D;
#pragma unroll
        for (int kk = 0; kk < D / 16; ++kk) {
            const int col = kk * 16 + c4 * 2;
            qf[kk][0] = *reinterpret_cast<const uint32_t*>(q0 + col);
            qf[kk][1] = *reinterpret_cast<const uint32_t*>(q1 + col);
            qf[kk][2] = *reinterpret_cast<const uint32_t*>(q0 + col + 8);
            qf[kk][3] = *reinterpret_cast<const uint32_t*>(q1 + col + 8);
        }
        if (opj && lane == 0) {
            bulk_g2s(sbase + OFF_MS + MS_KV + tok * (D + DV) * 2, P.qkv + (size_t)(b2 * W + tok) * QKVW + QOFF_K, (D + DV) * 2, ms_full);
            if (tok == 7) bulk_g2s(sbase + OFF_MS + MS_Q, P.qkv + (size_t)(b2 * W + tokc) * QKVW + hgc * 4 * D, 4 * D * 2, ms_full);
        }
        // Q RoPE on dims 0..63 (NeoX halves d <-> d + 32 = fragment kk <-> kk + 2, same lane slots)
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const uint32_t x = qf[kk][e], y = qf[kk + 2][e];
                qf[kk][e] = rope_lo(x, y, qcv[kk][e], qsv[kk][e]);
                qf[kk + 2][e] = rope_hi(x, y, qcv[kk][e], qsv[kk][e]);
            }
        }

        float o[64];
#pragma unroll
        for (int i = 0; i < 64; ++i) o[i] = 0.f;
        float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
        const int n = ntile_cta;
        const float scale_log2 = P.scale_log2;
        auto stage_of = [&](int it, uint32_t& ph) { ph = (uint32_t)(it / STAGES) & 1u; return it % STAGES; };
        auto issue_s = [&](float (&sacc)[32], int s) {
            const uint32_t kst = sbase + OFF_KV + s * STAGE_BYTES;
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
            const uint32_t vst = sbase + OFF_KV + s * STAGE_BYTES + KSTAGE;
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
        // keys >= new_key0 (the new keys: their contribution comes from the combine's 34th split) are masked
        auto softmax = [&](float (&sacc)[32], int p, uint32_t (&pf)[4][4], float& al0, float& al1) {
            const int key_base = tile_of(p) * BN;
            const int lim = new_key0 - 1;
            if (key_base + BN - 1 > lim) {
#pragma unroll
                for (int nb = 0; nb < 8; ++nb) {
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int key = key_base + nb * 8 + c4 * 2 + e;
                        if (key > lim) {
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
                pf[nb >> 1][(nb & 1) * 2 + 0] = pack_bf16(p0, p1);
                pf[nb >> 1][(nb & 1) * 2 + 1] = pack_bf16(p2, p3);
            }
            l0 = l0 * al0 + ps0;
            l1 = l1 * al1 + ps1;
        };
        auto rescale_o = [&](float al0, float al1) {
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
        // FA3 ping-pong between the two consumer warpgroups (named barriers 1 / 2)
        const int bar_me = 1 + cw, bar_other = 1 + (cw ^ 1);
        if (cw == 1 && n > 0) named_arrive(1, 256);
        uint32_t pf[4][4];
        float al0 = 1.f, al1 = 1.f;
        int s_prev = 0;
        if (n > 0) {
            uint32_t ph;
            const int s = stage_of(0, ph);
            mbar_wait(kv_full + s * 8, ph);
            if (ct == 0) BSTAMP(4);
            float sacc[32];
            named_sync(bar_me, 256);
            issue_s(sacc, s);
            named_arrive(bar_other, 256);
            wg_wait<0>();
            fence_f(sacc);
            softmax(sacc, 0, pf, al0, al1);
            s_prev = s;
        }
        for (int i = 1; i < n; ++i) {
            uint32_t ph;
            const int s = stage_of(i, ph);
            mbar_wait(kv_full + s * 8, ph);
            float sacc[32];
            named_sync(bar_me, 256);
            issue_s(sacc, s);
            rescale_o(al0, al1);
            issue_pv(pf, s_prev);
            named_arrive(bar_other, 256);
            wg_wait<1>();
            fence_f(sacc);
            uint32_t pn[4][4];
            softmax(sacc, i, pn, al0, al1);
            wg_wait<0>();
            fence_f(o);
#pragma unroll
            for (int j = 0; j < 4; ++j) fence_u(pf[j]);
            mbar_arrive(kv_empty + s_prev * 8);
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
            if (cw == 0) named_arrive(bar_other, 256);
            wg_wait<0>();
            fence_f(o);
            mbar_arrive(kv_empty + s_prev * 8);
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(att_done);   // this warp's last KV-ring read retired: W k-tiles may land there
        // ---- normalized partial O + log2 LSE of this split, [b][row][split][dv]
        l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
        l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
        {
            const int r0 = tok * HQ + h0, r1 = tok * HQ + h1;
            const size_t row0 = (size_t)b * ROWS + r0, row1 = (size_t)b * ROWS + r1;
            if (c4 == 0) {
                P.part_lse[row0 * NSPLIT + sp] = (l0 > 0.f) ? m0 + __log2f(l0) : -INFINITY;
                P.part_lse[row1 * NSPLIT + sp] = (l1 > 0.f) ? m1 + __log2f(l1) : -INFINITY;
            }
            const float inv0 = (l0 > 0.f) ? 1.f / l0 : 0.f;
            const float inv1 = (l1 > 0.f) ? 1.f / l1 : 0.f;
            float* po0 = P.part_o + (row0 * NSPLIT + sp) * DV;
            float* po1 = P.part_o + (row1 * NSPLIT + sp) * DV;
#pragma unroll
            for (int nb = 0; nb < 16; ++nb) {
                const int col = nb * 8 + c4 * 2;
                *reinterpret_cast<float2*>(po0 + col) = make_float2(o[nb * 4 + 0] * inv0, o[nb * 4 + 1] * inv0);
                *reinterpret_cast<float2*>(po1 + col) = make_float2(o[nb * 4 + 2] * inv1, o[nb * 4 + 3] * inv1);
            }
        }
        named_sync(3, 256);
        if (ct == 0) {
            BSTAMP(5);
            st_release_gpu(P.flags_att + c * FLAG_STRIDE, epoch);   // cumulative over the 256 threads' partial stores (barrier above)
            BSTAMP(18);
        }
        if (opj) {
            // ---- the 8 new keys' contribution to rows [4c, 4c + 4) (token tokc, heads 4 hgc ..): a 34th split computed from the stash --
            // RoPE in place (mimo_rope_kv's bf16 rounding), scores (fp32), softmax in the log2 domain, PV in fp32 -- while the other splits
            // finish. CTAs with hgc 0 also store token tokc's K / V row to the cache for the next steps (slot 0 = padding: not written).
            float lse_n;
            float2 o_n;
            {
                const uint32_t ms = sbase + OFF_MS;
                mbar_wait(ms_full, 0);
                if (ct < 48) {   // items 0..15: Q head ct / 4, NeoX chunk pair ct % 4 | 16..47: K row (ct - 16) / 4, pair ct % 4
                    const bool isq = ct < 16;
                    const int r = isq ? ct >> 2 : (ct - 16) >> 2, cp = ct & 3;
                    const uint32_t rowa = isq ? ms + MS_Q + r * D * 2 : ms + MS_KV + r * (D + DV) * 2;
                    const uint32_t csa = ms + MS_CS + (isq ? tokc : r) * 128;
                    const uint4 x = lds128(rowa + cp * 16), y = lds128(rowa + (cp + 4) * 16);
                    const uint4 cv = lds128(csa + cp * 16), sv = lds128(csa + 64 + cp * 16);
                    sts128(rowa + cp * 16, rope_lo4(x, y, cv, sv));
                    sts128(rowa + (cp + 4) * 16, rope_hi4(x, y, cv, sv));
                }
                named_sync(3, 256);
                if (hgc == 0 && ct < 40 && slotc != 0) {
                    const uint4 v = lds128(ms + MS_KV + tokc * (D + DV) * 2 + ct * 16);   // K chunks 0..23 | V chunks 0..15
                    if (ct < 24) *reinterpret_cast<uint4*>(reinterpret_cast<char*>(P.kc) + slotc * (P.ks * 2) + ct * 16) = v;
                    else *reinterpret_cast<uint4*>(reinterpret_cast<char*>(P.vc) + slotc * (P.vs * 2) + (ct - 24) * 16) = v;
                }
                {   // scores: thread (row lr, key k, part) sums dims [24 part, + 24); the 8 parts are 8 consecutive lanes
                    const int lr = ct >> 6, k = (ct >> 3) & 7, part = ct & 7;
                    const uint32_t qa = ms + MS_Q + lr * D * 2 + part * 48, ka = ms + MS_KV + k * (D + DV) * 2 + part * 48;
                    float acc = 0.f;
#pragma unroll
                    for (int i = 0; i < 12; ++i) {
                        const float2 qv = bf2f(lds32(qa + i * 4)), kv = bf2f(lds32(ka + i * 4));
                        acc = fmaf(qv.x, kv.x, acc);
                        acc = fmaf(qv.y, kv.y, acc);
                    }
                    acc += __shfl_xor_sync(0xffffffffu, acc, 1);
                    acc += __shfl_xor_sync(0xffffffffu, acc, 2);
                    acc += __shfl_xor_sync(0xffffffffu, acc, 4);
                    if (part == 0) sts32f(ms + MS_SC + (lr * W + k) * 4, k <= tokc ? acc * P.scale_log2 : -INFINITY);
                }
                named_sync(3, 256);
                {
                    const int lr = ct >> 6, dvp = (ct & 63) * 2;
                    float sc[W];
                    float m = -INFINITY;
#pragma unroll
                    for (int k = 0; k < W; ++k) { sc[k] = lds32f(ms + MS_SC + (lr * W + k) * 4); m = fmaxf(m, sc[k]); }   // key 0 always visible
                    float l = 0.f;
                    float2 o2 = make_float2(0.f, 0.f);
#pragma unroll
                    for (int k = 0; k < W; ++k) {
                        const float pk = k <= tokc ? exp2f(sc[k] - m) : 0.f;
                        const float2 vv = bf2f(lds32(ms + MS_KV + k * (D + DV) * 2 + D * 2 + dvp * 2));
                        l += pk;
                        o2.x = fmaf(pk, vv.x, o2.x);
                        o2.y = fmaf(pk, vv.y, o2.y);
                    }
                    const float il = 1.f / l;
                    o_n = make_float2(o2.x * il, o2.y * il);
                    lse_n = m + __log2f(l);
                }
                __syncwarp();
                if (lane == 0) mbar_arrive(ms_dead);   // last stash read of this warp: W k-tile NW_EARLY may land over it
            }
            // ---- combine rows [4c, 4c + 4) (request b2): fa_verify_combine_kernel's arithmetic over the 33 splits (split s -> accumulator s % 4
            // in s order, then ((a0 + a1) + a2) + a3), plus the new keys' split
            {
                if (ct < NSPLIT && !(P.mode & MODE_DBG_NOPOLL)) {
                    const unsigned* fl = P.flags_att + (b2 * NSPLIT + ct) * FLAG_STRIDE;
                    while (ld_acquire_gpu(fl) != epoch) __nanosleep(32);
                }
                named_sync(3, 256);
                if (ct == 0) BSTAMP(6);
                const int row = c * CROWS + (ct >> 6);   // global row b2 * 128 + r
                const int dv = (ct & 63) * 2;
                const float* lse = P.part_lse + (size_t)row * NSPLIT;
                const float* po = P.part_o + (size_t)row * NSPLIT * DV + dv;
                float lv[NSPLIT];
                float2 v[NSPLIT];
#pragma unroll
                for (int s = 0; s < NSPLIT; ++s) {
                    lv[s] = __ldcg(lse + s);
                    v[s] = __ldcg(reinterpret_cast<const float2*>(po + (size_t)s * DV));
                }
                float mx = lse_n;
#pragma unroll
                for (int s = 0; s < NSPLIT; ++s) mx = fmaxf(mx, lv[s]);
                const float mref = (mx == -INFINITY) ? 0.f : mx;
                float2 acc[4];
                float den[4];
#pragma unroll
                for (int k = 0; k < 4; ++k) { acc[k] = make_float2(0.f, 0.f); den[k] = 0.f; }
#pragma unroll
                for (int s = 0; s < NSPLIT; ++s) {
                    const float w = (lv[s] == -INFINITY) ? 0.f : exp2f(lv[s] - mref);
                    den[s & 3] += w;
                    acc[s & 3].x = fmaf(w, v[s].x, acc[s & 3].x);
                    acc[s & 3].y = fmaf(w, v[s].y, acc[s & 3].y);
                }
                float2 a = acc[0];
                float dn = den[0];
#pragma unroll
                for (int k = 1; k < 4; ++k) { a.x += acc[k].x; a.y += acc[k].y; dn += den[k]; }
                {
                    const float w = exp2f(lse_n - mref);
                    dn += w;
                    a.x = fmaf(w, o_n.x, a.x);
                    a.y = fmaf(w, o_n.y, a.y);
                }
                const float inv = dn > 0.f ? 1.f / dn : 0.f;
                if (ct == 0) BSTAMP(16);
                if (FAB_DISCARD) {
                    // rows [4c, 4c + 4) x 33 splits x 128 fp32 = 528 whole 128-B lines read by this CTA only (and only now: the values are in
                    // registers) -> drop them from L2 without write-back
                    named_sync(3, 256);
                    const char* base = reinterpret_cast<const char*>(P.part_o + (size_t)c * CROWS * NSPLIT * DV);
                    for (int l = ct; l < CROWS * NSPLIT * DV * 4 / 128; l += 256)
                        asm volatile("discard.global.L2 [%0], 128;" :: "l"(base + (size_t)l * 128) : "memory");
                }
                const int r = row % ROWS, t = r / HQ, hh = r % HQ;
                *reinterpret_cast<uint32_t*>(P.attn_out + (size_t)(b2 * W + t) * KO + hh * DV + dv) = pack_bf16(a.x * inv, a.y * inv);
                named_sync(3, 256);
                if (ct == 0) {
                    BSTAMP(7);
                    st_release_gpu(P.flags_cmb + c * FLAG_STRIDE, epoch);
                    BSTAMP(17);
                }
            }
            // ---- o_proj: D[96 (+32) W rows][32 tokens] += W[rows][k] * A[tok][k] over this CTA's K half; warpgroup cw takes rows 64 cw ..
            const int q = c / CL;
            if (ct < 64 && !(P.mode & MODE_DBG_NOPOLL)) {
                // the combined rows of heads 8h .. 8h + 7 (all 4 requests, all 8 tokens): combine CTAs 4j + 2h + e
                const int cc = 4 * (ct >> 1) + 2 * h + (ct & 1);
                const unsigned* fl = P.flags_cmb + cc * FLAG_STRIDE;
                while (ld_acquire_gpu(fl) != epoch) __nanosleep(32);
            }
            named_sync(3, 256);
            if (ct == 0) {
                asm volatile("fence.acq_rel.gpu;" ::: "memory");
                BSTAMP(8);
                asm volatile("fence.proxy.async.global;" ::: "memory");   // generic-proxy stores (released) -> async-proxy TMA reads
                for (int kt = 0; kt < NA; ++kt) {
                    mbar_expect(afull + kt * 8, BT_BYTES);
                    tma_load_2d_nohint(sbase + OFF_A + kt * BT_BYTES, &P.tm_a, h * (KO / CL) + kt * KT, 0, afull + kt * 8);
                }
                mbar_arrive(bgo);
            }
            float d[NACC][16];
#pragma unroll
            for (int a = 0; a < NACC; ++a)
#pragma unroll
                for (int i = 0; i < 16; ++i) d[a][i] = 0.f;
            const uint64_t dA0 = desc_sw128_k(sbase + OFF_W + cw * 64 * 128), dB0 = desc_sw128_k(sbase + OFF_A);
#pragma unroll
            for (int kt = 0; kt < NKT; ++kt) {
                const int s = kt % NA;
                if (!(P.mode & MODE_DBG_NOW)) mbar_wait(wfull + kt * 8, 0);
                mbar_wait(afull + s * 8, (kt / NA) & 1);
                if (kt == 0 && ct == 0) BSTAMP(9);
                wg_fence();
#pragma unroll
                for (int j = 0; j < KT / 16; ++j)
                    wgmma_m64n32k16(d[j % NACC], dA0 + (uint64_t)((kt * WT_BYTES + 32 * j) >> 4), dB0 + (uint64_t)((s * BT_BYTES + 32 * j) >> 4),
                                    (kt * (KT / 16) + j) >= NACC);
                wg_commit();
                wg_wait<1>();
                if (kt >= 1 && lane == 0) mbar_arrive(aempty + ((kt - 1) % NA) * 8);
            }
            wg_wait<0>();
#pragma unroll
            for (int a = 1; a < NACC; ++a)
#pragma unroll
                for (int i = 0; i < 16; ++i) d[0][i] += d[a][i];
            if (ct == 0) BSTAMP(10);
            // pair exchange: the rows the PARTNER finalizes go to its OFF_X by st.async (complete_tx on its xfull); my own rows stay in registers
            const int row = 64 * cw + 16 * warp + g;   // rows row, row + 8 of d[0][4i + 0/1], d[0][4i + 2/3]; tokens 8i + 2 c4 (+1)
            const int prow0 = COLS * (h ^ 1);
            const uint32_t rx = mapa(sbase + OFF_X, (uint32_t)(h ^ 1)), rbar = mapa(xfull, (uint32_t)(h ^ 1));
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int r = row + 8 * hh;
                if (r >= prow0 && r < prow0 + COLS) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int tk = 8 * i + 2 * c4;
                        st_async_v2(rx + ((r - prow0) * M + tk) * 4, d[0][4 * i + 2 * hh], d[0][4 * i + 2 * hh + 1], rbar);
                    }
                }
            }
            mbar_wait(xfull, 0);   // the partner's 48 x 32 partial of my columns
            if (ct == 0) BSTAMP(11);
            named_sync(3, 256);    // both warpgroups' MMAs retired: W k-tile 0 is dead -> bf16 staging of my 48 columns there
            __nv_bfloat16* outs = reinterpret_cast<__nv_bfloat16*>(smem + OFF_W);   // [32 tok][48]
            const float* xs = reinterpret_cast<const float*>(smem + OFF_X);
            const int orow0 = COLS * h;
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int r = row + 8 * hh;
                if (r >= orow0 && r < orow0 + COLS) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int tk = 8 * i + 2 * c4;
                        const float2 pv = *reinterpret_cast<const float2*>(xs + (r - orow0) * M + tk);
                        // fixed order: K half 0 + K half 1 (the same value on both CTAs of the pair is not needed: each finalizes its own columns)
                        const float v0 = h ? pv.x + d[0][4 * i + 2 * hh] : d[0][4 * i + 2 * hh] + pv.x;
                        const float v1 = h ? pv.y + d[0][4 * i + 2 * hh + 1] : d[0][4 * i + 2 * hh + 1] + pv.y;
                        outs[tk * COLS + r - orow0] = __float2bfloat16_rn(v0);
                        outs[(tk + 1) * COLS + r - orow0] = __float2bfloat16_rn(v1);
                    }
                }
            }
            named_sync(3, 256);   // outs complete
            // message (token tk, 6-column group g6): thread tk * 8 + g6; a warp = 4 tokens x 8 groups (whole 32-B sectors per store instruction)
            {
                const int c0 = q * PCOLS + COLS * h;
                const int tk = ct >> 3, g6 = ct & 7;
                const uint32_t* wv = reinterpret_cast<const uint32_t*>(outs + tk * COLS + 6 * g6);
                const uint32_t w0 = wv[0], w1 = wv[1], w2 = wv[2];
                const int owner = tk % TP_NDEV;
                if (owner == P.my_rank || (P.mode & MODE_FULL_LOCAL)) {
                    uint32_t* dst = reinterpret_cast<uint32_t*>(P.partial + (size_t)tk * DIM + c0 + 6 * g6);
                    dst[0] = w0; dst[1] = w1; dst[2] = w2;
                }
                if (owner != P.my_rank && !(P.mode & (MODE_NO_PUSH | MODE_FULL_LOCAL))) {
                    const int k = c0 / TP_KS, m = (c0 % TP_KS) / 6 + g6;
                    st_ll(P.peer[owner] + tp_rs_idx(P.my_rank, tk, k, m), make_uint4(w0, w1, w2, epoch));
                }
            }
            if (ct == 0) BSTAMP(12);
        }
    }
    cluster_sync_relaxed();   // no CTA exits while its partner's st.async into it may be in flight
}

// ------------------------------------------------------------------------------------------------------------------ host
typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                                  const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeTiledFn encode_tiled_fn() {
    static EncodeTiledFn fn = nullptr;
    if (!fn) {
        void* p = nullptr;
        cudaDriverEntryPointQueryResult q;
        cudaError_t err = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q);
        TORCH_CHECK(err == cudaSuccess && q == cudaDriverEntryPointSuccess && p, "cuTensorMapEncodeTiled not available");
        fn = reinterpret_cast<EncodeTiledFn>(p);
    }
    return fn;
}
// bf16 [rows][2048] (row stride elems) -> 128-byte CUtensorMap (CPU uint8 tensor), box {64 k, box_rows}, SWIZZLE_128B
static torch::Tensor make_map(const void* base, int64_t rows, int64_t row_stride_elems, int box_rows) {
    auto out = torch::zeros({(int64_t)sizeof(CUtensorMap)}, torch::dtype(torch::kUInt8));
    const cuuint64_t dims[2] = {(cuuint64_t)KO, (cuuint64_t)rows};
    const cuuint64_t strides[1] = {(cuuint64_t)row_stride_elems * 2};
    const cuuint32_t box[2] = {KT, (cuuint32_t)box_rows}, estr[2] = {1, 1};
    CUresult r = encode_tiled_fn()(reinterpret_cast<CUtensorMap*>(out.data_ptr()), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), dims, strides, box,
                                   estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
    return out;
}
// page table [NREQ][S] int32 (row stride pt_stride elements) -> map with box {64, 1}, no swizzle, OOB elements read as 0
static void make_ptmap(CUtensorMap* out, const torch::Tensor& pt) {
    const cuuint64_t dims[2] = {(cuuint64_t)pt.size(1), (cuuint64_t)pt.size(0)};
    const cuuint64_t strides[1] = {(cuuint64_t)pt.stride(0) * 4};
    const cuuint32_t box[2] = {BN, 1}, estr[2] = {1, 1};
    CUresult r = encode_tiled_fn()(out, CU_TENSOR_MAP_DATA_TYPE_INT32, 2, pt.data_ptr(), dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                   CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (page table) failed: ", (int)r);
}
torch::Tensor make_amap(torch::Tensor a) {
    TORCH_CHECK(a.is_cuda() && a.scalar_type() == torch::kBFloat16 && a.numel() == (int64_t)M * KO && a.is_contiguous() &&
                reinterpret_cast<uint64_t>(a.data_ptr()) % 16 == 0, "attention output must be a contiguous bf16 [32][2048]");
    return make_map(a.data_ptr(), M, KO, M);
}
torch::Tensor make_wmap(torch::Tensor w) {
    TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kBFloat16 && w.dim() == 2 && w.size(0) == DIM && w.size(1) == KO && w.stride(1) == 1 &&
                w.stride(0) % 8 == 0 && reinterpret_cast<uint64_t>(w.data_ptr()) % 16 == 0, "o_proj weight must be bf16 [6144][2048], 16-B aligned rows");
    return make_map(w.data_ptr(), DIM, w.stride(0), PCOLS);
}

// qkv [32][3392] bf16 (un-rotated) | cos_sin [P][64] bf16 | positions / loc [32] i32/i64 | kc [rows][..192] / vc [rows][..128] bf16 |
// pt [4][S] i32 | seqlens [4] i32 | part_o f32 >= 512*33*128 | part_lse f32 >= 512*33 | attn_out bf16 [32][2048] (amap built over it) |
// flags_att / flags_cmb i32 >= 132*32 / 128*32 | partial bf16 [32][6144] | wmap / amap CPU u8[128] | peers CPU i64[8] | work i32 (fused MoE
// counters, epoch = work[2] + 1) | mode | pdl | dbg i64 [132*32] or empty
void fa_block(torch::Tensor qkv, torch::Tensor cos_sin, torch::Tensor positions, torch::Tensor loc, torch::Tensor kc, torch::Tensor vc,
              torch::Tensor pt, torch::Tensor seqlens, torch::Tensor part_o, torch::Tensor part_lse, torch::Tensor attn_out, torch::Tensor flags_att,
              torch::Tensor flags_cmb, torch::Tensor partial, torch::Tensor wmap, torch::Tensor amap, torch::Tensor peers, torch::Tensor work,
              int64_t my_rank, double sm_scale, int64_t mode, int64_t pdl, torch::Tensor dbg) {
    TORCH_CHECK(qkv.is_cuda() && qkv.scalar_type() == torch::kBFloat16 && qkv.size(0) == M && qkv.size(1) == QKVW && qkv.is_contiguous(), "qkv [32][3392] bf16");
    TORCH_CHECK(cos_sin.scalar_type() == torch::kBFloat16 && cos_sin.size(1) == 64 && cos_sin.is_contiguous(), "cos_sin [P][64] bf16");
    TORCH_CHECK(positions.numel() == M && loc.numel() == M && positions.is_contiguous() && loc.is_contiguous(), "positions / loc [32]");
    TORCH_CHECK(kc.scalar_type() == torch::kBFloat16 && vc.scalar_type() == torch::kBFloat16 && kc.size(0) == vc.size(0), "kv cache bf16");
    TORCH_CHECK(kc.stride(0) % 8 == 0 && vc.stride(0) % 8 == 0 && kc.numel() / kc.size(0) >= D && vc.numel() / vc.size(0) >= DV, "kv cache rows");
    TORCH_CHECK((kc.size(0) * kc.stride(0) * 2) < (1ll << 32) && (vc.size(0) * vc.stride(0) * 2) < (1ll << 32), "kv cache > 4 GB (32-bit offsets)");
    TORCH_CHECK(pt.scalar_type() == torch::kInt32 && pt.dim() == 2 && pt.size(0) == NREQ && pt.stride(1) == 1 && pt.stride(0) % 4 == 0 &&
                reinterpret_cast<uint64_t>(pt.data_ptr()) % 16 == 0, "page table [4][S] int32, 16-B aligned rows");
    TORCH_CHECK(seqlens.scalar_type() == torch::kInt32 && seqlens.numel() == NREQ && seqlens.is_contiguous(), "seqlens [4] int32");
    TORCH_CHECK(part_o.scalar_type() == torch::kFloat32 && part_o.numel() >= (int64_t)NREQ * ROWS * NSPLIT * DV, "part_o");
    TORCH_CHECK(part_lse.scalar_type() == torch::kFloat32 && part_lse.numel() >= (int64_t)NREQ * ROWS * NSPLIT, "part_lse");
    TORCH_CHECK(attn_out.scalar_type() == torch::kBFloat16 && attn_out.numel() == (int64_t)M * KO && attn_out.is_contiguous(), "attn_out [32][2048]");
    TORCH_CHECK(flags_att.numel() >= (int64_t)NCTA * FLAG_STRIDE && flags_cmb.numel() >= (int64_t)NOP * FLAG_STRIDE && flags_att.element_size() == 4 &&
                flags_cmb.element_size() == 4, "flags");
    TORCH_CHECK(partial.numel() >= (int64_t)M * DIM && partial.scalar_type() == torch::kBFloat16 && partial.is_contiguous(), "partial");
    TORCH_CHECK(wmap.device().is_cpu() && wmap.numel() == sizeof(CUtensorMap) && amap.device().is_cpu() && amap.numel() == sizeof(CUtensorMap), "maps");
    TORCH_CHECK(peers.device().is_cpu() && peers.scalar_type() == torch::kInt64 && peers.numel() == TP_NDEV, "peers CPU int64[8]");
    TORCH_CHECK(work.scalar_type() == torch::kInt32 && work.numel() >= 4, "work");
    Params P{};
    std::memcpy(&P.tm_w, wmap.data_ptr(), sizeof(CUtensorMap));
    std::memcpy(&P.tm_a, amap.data_ptr(), sizeof(CUtensorMap));
    P.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
    P.cos_sin = reinterpret_cast<const __nv_bfloat16*>(cos_sin.data_ptr());
    P.positions = positions.data_ptr(); P.loc = loc.data_ptr();
    P.kc = reinterpret_cast<__nv_bfloat16*>(kc.data_ptr()); P.vc = reinterpret_cast<__nv_bfloat16*>(vc.data_ptr());
    P.ks = kc.stride(0); P.vs = vc.stride(0);
    P.pt = pt.data_ptr<int>(); P.pt_stride = (int)pt.stride(0); P.pt_width = (int)pt.size(1);
    make_ptmap(&P.tm_pt, pt);
    P.seqlens = seqlens.data_ptr<int>();
    P.part_o = part_o.data_ptr<float>(); P.part_lse = part_lse.data_ptr<float>();
    P.attn_out = reinterpret_cast<__nv_bfloat16*>(attn_out.data_ptr());
    P.flags_att = reinterpret_cast<unsigned*>(flags_att.data_ptr());
    P.flags_cmb = reinterpret_cast<unsigned*>(flags_cmb.data_ptr());
    P.partial = reinterpret_cast<__nv_bfloat16*>(partial.data_ptr());
    for (int d = 0; d < TP_NDEV; ++d) P.peer[d] = reinterpret_cast<uint4*>(peers.data_ptr<int64_t>()[d]);
    P.work = work.data_ptr<int>();
    P.dbg = dbg.numel() >= (int64_t)NCTA * 32 ? reinterpret_cast<unsigned long long*>(dbg.data_ptr()) : nullptr;
    P.my_rank = (int)my_rank;
    int md = (int)mode & ~(MODE_POS64 | MODE_LOC64);
    if (positions.scalar_type() == torch::kInt64) md |= MODE_POS64; else TORCH_CHECK(positions.scalar_type() == torch::kInt32, "positions int32/int64");
    if (loc.scalar_type() == torch::kInt64) md |= MODE_LOC64; else TORCH_CHECK(loc.scalar_type() == torch::kInt32, "loc int32/int64");
    P.mode = md;
    P.scale_log2 = (float)(sm_scale * (double)LOG2E);
    static bool configured[16] = {false};
    int dev = 0; cudaGetDevice(&dev);
    if (!configured[dev]) {
        TORCH_CHECK(cudaFuncSetAttribute(fa_block_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_ALLOC) == cudaSuccess, "smem attribute");
        configured[dev] = true;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(NCTA); cfg.blockDim = dim3(NTHREADS); cfg.dynamicSmemBytes = SMEM_ALLOC; cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[2];
    at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = CL; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[1].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at; cfg.numAttrs = pdl ? 2 : 1;
    TORCH_CHECK(cudaLaunchKernelEx(&cfg, fa_block_kernel, P) == cudaSuccess, "fa_block launch failed");
}

}  // namespace fab

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fa_block", &fab::fa_block);
    m.def("make_wmap", &fab::make_wmap);
    m.def("make_amap", &fab::make_amap);
    m.attr("NCTA") = fab::NCTA;
    m.attr("NOP") = fab::NOP;
    m.attr("NSPLIT") = fab::NSPLIT;
    m.attr("SMEM_ALLOC") = fab::SMEM_ALLOC;
    m.attr("NW_EARLY") = fab::NW_EARLY;
    m.attr("MODE_FULL_LOCAL") = fab::MODE_FULL_LOCAL;
    m.attr("MODE_NO_PUSH") = fab::MODE_NO_PUSH;
    m.attr("MODE_ATT_ONLY") = fab::MODE_ATT_ONLY;
    m.attr("MODE_DBG_NOPOLL") = fab::MODE_DBG_NOPOLL;
    m.attr("MODE_DBG_NOW") = fab::MODE_DBG_NOW;
    m.attr("MODE_DBG_QLATE_SHIFT") = fab::MODE_DBG_QLATE_SHIFT;
}
