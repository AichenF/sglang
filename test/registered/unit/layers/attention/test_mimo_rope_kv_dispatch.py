"""CPU-only contract tests for the opt-in MiMo RoPE/KV path."""

import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention import mimo_rope_kv as adapter
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def tensor(shape, dtype=torch.bfloat16):
    return types.SimpleNamespace(
        shape=shape,
        ndim=len(shape),
        dtype=dtype,
        device=torch.device("cuda:0"),
        is_cuda=True,
        is_contiguous=lambda: True,
    )


class TestMiMoRoPEKVDispatch(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.qkv = tensor((32, 3392))
        self.positions = tensor((32,), torch.int64)
        self.locations = tensor((32,), torch.int64)
        self.keys = tensor((128, 1, 192))
        self.values = tensor((128, 1, 128))
        self.rope = types.SimpleNamespace(
            use_fallback_kernel=True,
            is_neox_style=True,
            rotary_dim=64,
            cos_sin_cache=tensor((2048, 64)),
        )
        self.attention = types.SimpleNamespace(
            num_heads=16,
            num_kv_heads=1,
            head_dim=192,
            v_head_dim=128,
            rotary_emb=self.rope,
            attn=types.SimpleNamespace(layer_id=3),
        )
        self.batch = types.SimpleNamespace(
            forward_mode=types.SimpleNamespace(is_target_verify=lambda: True),
            out_cache_loc=self.locations,
        )
        self.pool_class = type("MHATokenToKVPool", (), {})
        self.swa_class = type("SWAKVPool", (), {})
        backend_class = type("FlashAttentionBackend", (), {})
        self.pool = self.pool_class()
        self.pool.dtype = self.pool.store_dtype = torch.bfloat16
        self.pool.use_hnd = self.pool.is_quantized_kv_cache = False
        self.pool.kv_cache_layout = "normal"
        self.pool.get_key_buffer = Mock(return_value=self.keys)
        self.pool.get_value_buffer = Mock(return_value=self.values)
        self.backend = backend_class()
        self.backend.use_mla = self.backend.fa_skip_kv_cache = False
        self.backend.kv_cache_is_mxfp8 = False
        self.backend.token_to_kv_pool = self.pool
        self.backend.forward_metadata = None
        modules = {}
        for name, attrs in (
            (
                "sglang.srt.layers.attention.flashattention_backend",
                {"FlashAttentionBackend": backend_class},
            ),
            ("sglang.srt.mem_cache.memory_pool", {"MHATokenToKVPool": self.pool_class}),
            ("sglang.srt.mem_cache.swa_memory_pool", {"SWAKVPool": self.swa_class}),
            (
                "sglang.srt.model_executor.forward_context",
                {"get_attn_backend": lambda: self.backend},
            ),
        ):
            module = types.ModuleType(name)
            module.__dict__.update(attrs)
            modules[name] = module
        self.enterContext(patch.dict(sys.modules, modules))
        self.enterContext(
            patch.object(torch.compiler, "is_compiling", return_value=False)
        )
        self.enterContext(
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0))
        )

    def selected(self):
        return adapter.select_mimo_rope_kv(
            self.attention, self.qkv, self.positions, self.batch
        )

    def test_supported_layout_uses_actual_cache(self):
        selected = self.selected()
        self.assertIsNotNone(selected)
        for actual, expected in zip(
            selected, (self.rope.cos_sin_cache, self.locations, self.keys, self.values)
        ):
            self.assertIs(actual, expected)
        self.pool.get_key_buffer.assert_called_once_with(3)
        self.pool.get_value_buffer.assert_called_once_with(3)

    def test_tensor_and_attention_boundaries(self):
        for obj, attribute, invalid in (
            (self.qkv, "shape", (16, 3392)),
            (self.qkv, "dtype", torch.float16),
            (self.qkv, "is_cuda", False),
            (self.qkv, "is_contiguous", lambda: False),
            (self.positions, "shape", (31,)),
            (self.positions, "dtype", torch.float32),
            (self.positions, "device", torch.device("cpu")),
            (self.locations, "is_contiguous", lambda: False),
            (self.rope, "use_fallback_kernel", False),
            (self.rope, "is_neox_style", False),
            (self.rope, "rotary_dim", 128),
            (self.rope.cos_sin_cache, "dtype", torch.float32),
            (self.attention, "num_heads", 8),
            (self.attention, "num_kv_heads", 2),
            (self.attention, "v_head_dim", 192),
            (self.batch.forward_mode, "is_target_verify", lambda: False),
        ):
            with (
                self.subTest(attribute=attribute, invalid=invalid),
                patch.object(obj, attribute, invalid),
            ):
                self.assertIsNone(self.selected())
        for dtype in (torch.int32, torch.int64):
            with (
                patch.object(self.positions, "dtype", dtype),
                patch.object(self.locations, "dtype", dtype),
            ):
                self.assertIsNotNone(self.selected())

    def test_backend_and_cache_boundaries(self):
        for obj, attribute, invalid in (
            (self.backend, "use_mla", True),
            (self.backend, "fa_skip_kv_cache", True),
            (self.backend, "kv_cache_is_mxfp8", True),
            (self.pool, "dtype", torch.float8_e4m3fn),
            (self.pool, "store_dtype", torch.uint8),
            (self.pool, "use_hnd", True),
            (self.pool, "is_quantized_kv_cache", True),
            (self.pool, "kv_cache_layout", "vectorized_5d"),
            (self.keys, "shape", (128, 2, 192)),
            (self.values, "shape", (129, 1, 128)),
            (self.values, "dtype", torch.float16),
            (self.keys, "is_contiguous", lambda: False),
        ):
            with (
                self.subTest(attribute=attribute, invalid=invalid),
                patch.object(obj, attribute, invalid),
            ):
                self.assertIsNone(self.selected())
        with patch.object(self, "backend", object()):
            self.assertIsNone(self.selected())
        with patch.object(self.backend, "token_to_kv_pool", object()):
            self.assertIsNone(self.selected())

    def test_swa_mapping_and_missing_metadata(self):
        pool = self.swa_class()
        pool.layers_mapping = {3: (0, True)}
        pool.swa_kv_pool = self.pool
        pool.full_kv_pool = self.pool
        pool.get_key_buffer = Mock(return_value=self.keys)
        pool.get_value_buffer = Mock(return_value=self.values)
        self.backend.token_to_kv_pool = pool
        self.assertIsNone(self.selected())
        mapped = tensor((32,), torch.int32)
        self.backend.forward_metadata = types.SimpleNamespace(swa_out_cache_loc=mapped)
        self.assertIs(self.selected()[1], mapped)
        pool.layers_mapping[3] = (0, False)
        self.assertIs(self.selected()[1], self.locations)

    def test_compile_and_non_hopper_fallback(self):
        with patch.object(torch.compiler, "is_compiling", return_value=True):
            self.assertIsNone(self.selected())
        with patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)):
            self.assertIsNone(self.selected())

    def test_disabled_path_does_not_select_or_launch(self):
        with (
            envs.SGLANG_OPT_MIMO_ROPE_KV.override(False),
            patch.object(adapter, "select_mimo_rope_kv") as select,
        ):
            self.assertFalse(
                adapter.try_fused_mimo_rope_kv(
                    self.attention, self.qkv, self.positions, self.batch
                )
            )
            select.assert_not_called()

    def test_supported_path_launches_once(self):
        module = types.ModuleType("sglang.kernels.ops.attention.mimo_rope_kv")
        module.run = Mock()
        with (
            envs.SGLANG_OPT_MIMO_ROPE_KV.override(True),
            patch.dict(sys.modules, {module.__name__: module}),
        ):
            self.assertTrue(
                adapter.try_fused_mimo_rope_kv(
                    self.attention, self.qkv, self.positions, self.batch
                )
            )
            module.run.assert_called_once_with(
                self.qkv,
                self.rope.cos_sin_cache,
                self.positions,
                self.locations,
                self.keys,
                self.values,
            )
            self.qkv.shape = (1, 3392)
            self.assertFalse(
                adapter.try_fused_mimo_rope_kv(
                    self.attention, self.qkv, self.positions, self.batch
                )
            )
            self.assertEqual(module.run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
