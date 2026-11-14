#pragma once

#include <distributed/dflash_mega.cuh>

namespace sglang {
namespace dflash_weight_pairs_test {

using namespace dflash_mega;
struct CopyParams {
  CUtensorMap maps[NPH];
  CUtensorMap original;
  uint8_t *paired;
  uint8_t *reference;
  int phase;
  int groups;
};

__global__ void kernel(const __grid_constant__ CopyParams args) {
  extern __shared__ __align__(1024) uint8_t memory[];
  auto *paired_bar = reinterpret_cast<uint64_t *>(memory + 2 * SLAB);
  auto *original_bar = paired_bar + 1;
  const int group = blockIdx.x % args.groups;
  const int16_t tile = static_cast<int16_t>(blockIdx.x / args.groups);
  if (threadIdx.x == 0) {
    mbar_init(paired_bar, 1);
    mbar_init(original_bar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect_tx(paired_bar, SLAB);
    mbar_expect_tx(original_bar, SLAB);
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;"
                 : "=l"(policy));
    Params parameters{};
    parameters.maps = args.maps;
    UIt unit{};
    unit.ph = args.phase;
    unit.u = group;
    unit.tl = &tile;
    // Call the actual new production weight issuer, including its coordinates.
    uit_issue(parameters, unit, false, memory, paired_bar, policy);
    for (int part = 0; part < 2; ++part) {
      const int column = (2 * group + part) * 128;
      if (args.phase == PH_GU)
        tma3d(&args.original, memory + SLAB + part * TILEB, original_bar,
              column, tile * 32, 0, policy);
      else
        tma2d(&args.original, memory + SLAB + part * TILEB, original_bar,
              column, tile * 64, policy);
    }
    mbar_wait(paired_bar, 0);
    mbar_wait(original_bar, 0);
  }
  __syncthreads();
  for (int offset = threadIdx.x * 16; offset < SLAB;
       offset += blockDim.x * 16) {
    const size_t output = static_cast<size_t>(blockIdx.x) * SLAB + offset;
    *reinterpret_cast<uint4 *>(args.paired + output) =
        *reinterpret_cast<uint4 *>(memory + offset);
    *reinterpret_cast<uint4 *>(args.reference + output) =
        *reinterpret_cast<uint4 *>(memory + SLAB + offset);
  }
}

} // namespace dflash_weight_pairs_test

struct DFlashWeightPairsTest {
  static void run(tvm::ffi::TensorView weights, tvm::ffi::TensorView paired,
                  tvm::ffi::TensorView reference, int64_t phase) {
    using namespace host;
    using namespace dflash_mega;
    RuntimeCheck(phase >= 0 && phase < 4 && weights.ndim() == 2 &&
                 weights.dtype().bits == 8);
    const int rows = weights.size(0), columns = weights.size(1);
    RuntimeCheck(weights.stride(0) == columns && weights.stride(1) == 1 &&
                 columns % 256 == 0 && rows % 64 == 0);
    RuntimeCheck(paired.numel() == weights.numel() &&
                 reference.numel() == weights.numel());
    dflash_weight_pairs_test::CopyParams args{};
    const void *pointer = weights.data_ptr();
    if (phase == PH_GU) {
      RuntimeCheck(rows == 2 * INTER);
      args.maps[phase] = make_map_4d(
          pointer, 128, INTER, 2, columns / 128, columns,
          static_cast<uint64_t>(INTER) * columns, 128, 128, 32, 2, 2);
      args.original =
          make_map_3d(pointer, columns, INTER, 2, columns,
                      static_cast<uint64_t>(INTER) * columns, 128, 32, 2);
    } else {
      args.maps[phase] = make_map_3d(pointer, 128, rows, columns / 128, columns,
                                     128, 128, 64, 2);
      args.original = make_map_2d(pointer, columns, rows, columns, 128, 64);
    }
    args.paired = static_cast<uint8_t *>(paired.data_ptr());
    args.reference = static_cast<uint8_t *>(reference.data_ptr());
    args.phase = phase;
    args.groups = columns / 256;
    LaunchKernel(dim3((rows / 64) * args.groups), dim3(128), weights.device(),
                 2 * SLAB + 16)(dflash_weight_pairs_test::kernel, args);
  }
};

} // namespace sglang
