"""Bitwise comparison with the BF16 fallback RoPE used by MiMo on Hopper."""

import unittest

import torch

from sglang.kernels.ops.attention.mimo_rope_kv import run
from sglang.kernels.ops.attention.rope import rotary_embedding
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestMiMoRoPEKV(CustomTestCase):
    def setUp(self):
        super().setUp()
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("MiMo BF16 specialization requires SM90")
        torch.manual_seed(42)

    def check_case(self, dtype, locations_kind, graph, heads, prefetch=False):
        device = "cuda"
        original = torch.randn((32, 3392), dtype=torch.bfloat16, device=device)
        angles = torch.randn((2048, 32), device=device)
        cos_sin = torch.cat((angles.cos(), angles.sin()), dim=1).to(torch.bfloat16)
        positions = torch.randint(0, 2048, (32,), dtype=dtype, device=device)
        locations = torch.arange(1, 33, dtype=dtype, device=device)
        if locations_kind == "sparse":
            locations = locations * 3
            locations[::4] = 0
        elif locations_kind == "padding":
            locations.zero_()
        keys0 = torch.randn((128, 1, 192), dtype=torch.bfloat16, device=device)
        values0 = torch.randn((128, 1, 128), dtype=torch.bfloat16, device=device)
        expected = original.clone()
        q, k, v = expected.split((3072, 192, 128), dim=-1)
        # The fallback CUDA API reads int64 positions; the fused kernel also
        # supports int32. Convert only the reference input, preserving values.
        rotary_embedding(positions.to(torch.int64), q, k, 192, cos_sin, True)
        torch.cuda.synchronize()
        keys_expected, values_expected = keys0.clone(), values0.clone()
        valid = locations != 0
        keys_expected[locations[valid].long(), 0] = k[valid]
        values_expected[locations[valid].long(), 0] = v[valid]
        actual, keys, values = original.clone(), keys0.clone(), values0.clone()
        weight = (
            torch.randn((6144, 2048), dtype=torch.bfloat16, device=device)
            if prefetch
            else None
        )
        weight_before = weight.clone() if prefetch else None

        def invoke():
            if prefetch:
                from sglang.kernels.ops.attention.mimo_rope_kv_prefetch import (
                    run as run_prefetch,
                )

                run_prefetch(
                    actual, cos_sin, positions, locations, keys, values, weight
                )
            else:
                run(
                    actual,
                    cos_sin,
                    positions,
                    locations,
                    keys,
                    values,
                    heads_per_cta=heads,
                )

        invoke()  # Compile outside capture.
        torch.cuda.synchronize()
        if graph:
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture):
                invoke()
            for _ in range(3):
                actual.copy_(original)
                keys.copy_(keys0)
                values.copy_(values0)
                capture.replay()
                torch.cuda.synchronize()
                for got, want in (
                    (actual, expected),
                    (keys, keys_expected),
                    (values, values_expected),
                ):
                    torch.testing.assert_close(got, want, rtol=0, atol=0)
        else:
            for got, want in (
                (actual, expected),
                (keys, keys_expected),
                (values, values_expected),
            ):
                torch.testing.assert_close(got, want, rtol=0, atol=0)

        if prefetch:
            torch.testing.assert_close(weight, weight_before, rtol=0, atol=0)

    def test_fallback_bitwise_and_untouched_cache(self):
        for dtype in (torch.int32, torch.int64):
            for locations in ("dense", "sparse", "padding"):
                for heads in (4, 8, 16):
                    with self.subTest(dtype=dtype, locations=locations, heads=heads):
                        self.check_case(dtype, locations, False, heads)

    def test_cuda_graph_replay(self):
        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                self.check_case(dtype, "sparse", True, 16)

    def test_prefetch_is_bitwise_and_read_only(self):
        for dtype in (torch.int32, torch.int64):
            for locations in ("dense", "sparse", "padding"):
                for graph in (False, True):
                    with self.subTest(dtype=dtype, locations=locations, graph=graph):
                        self.check_case(dtype, locations, graph, 16, prefetch=True)


if __name__ == "__main__":
    unittest.main()
