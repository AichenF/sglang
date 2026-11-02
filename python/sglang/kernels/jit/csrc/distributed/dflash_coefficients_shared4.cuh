// Private shared staging experiment; original prefix/prefetch stays literal.
__device__ __forceinline__ void project_coefficients_shared4(
    const ConvP& cp, Smem& s, int ctid, unsigned* ready) {
  const int warp = ctid / 32, lane = ctid % 32;
  constexpr int SPLITS = 4;
  static_assert(H % SPLITS == 0 && (H / SPLITS) % PROJ_K == 0);
  static_assert(PROJ_SHARED_BYTES == 49152 && PROJ_WARPS == 8);
  const int group = warp / SPLITS, split_id = warp % SPLITS;
  const int column_tiles = cp.width / 16;
  const int kbegin = split_id * (H / SPLITS), kend = kbegin + H / SPLITS;
  uint8_t* const workspace = s.greg + warp * PROJ_WARP_BYTES;
  float* const partials = reinterpret_cast<float*>(s.greg + PROJ_PARTIAL_OFFSET);
  // Uniform CTA tile loop: all 256 compute threads reach both tile barriers.
  // Producer/gather warps never participate in these named NCT barriers.
  for (int batch = blockIdx.x; batch < 2 * column_tiles; batch += 2 * GRID) {
    const int tile = batch + group * GRID;
    const bool active = tile < 2 * column_tiles;
    const int row = (tile / column_tiles) * 16;
    const int col = (tile % column_tiles) * 16;
    if (active) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
    projection_prefetch(cp, workspace, row, col, kbegin, lane);
#pragma unroll 1
    for (int k = kbegin; k < kend; k += PROJ_K) {
      const int parity = ((k - kbegin) / PROJ_K) & 1;
      uint8_t* const current = workspace + parity * 2 * PROJ_OPERAND_BYTES;
      if (k + PROJ_K < kend) {
        projection_prefetch(cp, workspace + (parity ^ 1) * 2 * PROJ_OPERAND_BYTES,
                            row, col, k + PROJ_K, lane);
        asm volatile("cp.async.wait_group 1;" ::: "memory");
      } else {
        asm volatile("cp.async.wait_group 0;" ::: "memory");
      }
      __syncwarp();
      const auto* shared_a = reinterpret_cast<const __nv_bfloat16*>(current);
      const auto* shared_b = reinterpret_cast<const __nv_bfloat16*>(current + PROJ_OPERAND_BYTES);
#pragma unroll
      for (int inner = 0; inner < PROJ_K; inner += 16) {
        // Ascending K16 updates within each contiguous 1536-element split.
        wmma::load_matrix_sync(a, shared_a + inner, PROJ_LD);
        wmma::load_matrix_sync(b, shared_b + inner, PROJ_LD);
        wmma::mma_sync(acc, a, b, acc);
      }
      __syncwarp();  // Buffer is no longer read before its next prefetch.
    }
    // Final wait_group0 above drains every warp's asynchronous operand copies.
    // Partials are separate from all operand buffers and remain FP32 here.
    wmma::store_matrix_sync(partials + warp * 256, acc, 16, wmma::mem_row_major);
    }
    // Inactive groups issue no copies/stores/reads, but all256 join both barriers.
    bar_sync(1, NCT);
    if (active) {
      const int local_thread = ctid % 128;
#pragma unroll
      for (int element = local_thread; element < 256; element += 128) {
        float sum = 0.f;
#pragma unroll
        for (int split = 0; split < SPLITS; ++split)
          sum = __fadd_rn(sum, partials[(group * SPLITS + split) * 256 + element]);
        cp.coefficients[(row + element / 16) * cp.width + col + element % 16] = __float2bfloat16_rn(sum);
      }
    }
    // Every partial has been consumed before any warp starts the next tile.
    bar_sync(1, NCT);
  }
  bar_sync(1, NCT);
  // Every CTA publishes once, including CTAs assigned no coefficient tiles.
  if (ctid == 0) red_release(ready, 1u);
}
