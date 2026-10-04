"""CPU integration contracts for the opt-in fused MoE adapter.

CUDA math is validated separately on eight Hopper GPUs. These tests exercise
real adapter/wrapper code with only device/distributed boundaries mocked.
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

SOURCE = (
    Path(__file__).resolve().parents[5] / "python/sglang/srt/layers/moe/mimo_fused_moe"
)


def load_source(name, filename):
    spec = importlib.util.spec_from_file_location(name, SOURCE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stub_module(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


class TestMiMoFusedMoE(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.kernel = load_source("mimo_wrapper_under_test", "fused_moe_wgmma.py")
        self.extension = SimpleNamespace(ROUTER_BF16=1, fused_moe_full=Mock())
        self.kernel._ext = self.extension
        self.parallel = SimpleNamespace(
            tp_size=8, moe_ep_size=1, attn_dp_size=1, attn_tp_size=8
        )
        self.group = SimpleNamespace(world_size=8, rank_in_group=0)
        self.all_reduce = Mock(side_effect=lambda tensor: tensor)
        self.dtypes = SimpleNamespace(
            float4e2m1="e2m1", float8e8m0="e8m0", float8e4m3="e4m3"
        )
        modules = {
            "sglang.srt.distributed": stub_module(
                "sglang.srt.distributed",
                get_tp_group=lambda: self.group,
                tensor_model_parallel_all_reduce=self.all_reduce,
            ),
            "sglang.srt.distributed.device_communicators.cuda_wrapper": stub_module(
                "cuda_wrapper", CudaRTLibrary=Mock()
            ),
            "sglang.srt.layers.moe.mimo_fused_moe": stub_module(
                "mimo_fused_moe", fused_moe_wgmma=self.kernel
            ),
            "sglang.srt.runtime_context": stub_module(
                "runtime_context", get_parallel=lambda: self.parallel
            ),
            "sglang.srt.utils": stub_module("utils", log_info_on_rank0=Mock()),
            "humming": stub_module("humming", dtypes=self.dtypes),
            "humming.config": stub_module(
                "humming.config", MmaType=SimpleNamespace(WGMMA="wgmma")
            ),
        }
        context = patch.dict(sys.modules, modules)
        context.start()
        self.addCleanup(context.stop)
        capability = patch.object(
            torch.cuda, "get_device_capability", return_value=(9, 0)
        )
        capability.start()
        self.addCleanup(capability.stop)
        self.adapter = load_source("mimo_adapter_under_test", "layer.py")

    def decoder(self, router_dtype=torch.bfloat16):
        f = self.kernel

        def tensor(shape, dtype):
            return torch.empty(shape, dtype=dtype, device="meta")

        def meta():
            return SimpleNamespace(
                mma_type="wgmma",
                use_fused_e8m0_scale=True,
                b_dtype=self.dtypes.float4e2m1,
                bs_dtype=self.dtypes.float8e8m0,
                a_dtype=self.dtypes.float8e4m3,
                input_scale_group_size=128,
                weight_scale_group_size=32,
            )

        experts = SimpleNamespace(humming_metas={"w13": meta(), "w2": meta()})
        for name, dtype, shape in (
            ("w13_weight", torch.int32, (f.NEXP, f.DIM // 32, 8 * f.INTER)),
            (
                "w13_weight_scale",
                torch.float8_e8m0fnu,
                (f.NEXP, f.DIM // 32, 2 * f.INTER),
            ),
            ("w2_weight", torch.int32, (f.NEXP, f.INTER // 32, 4 * f.DIM)),
            (
                "w2_weight_scale",
                torch.float8_e8m0fnu,
                (f.NEXP, f.INTER // 32, f.DIM),
            ),
            ("w13_weight_scale_2", torch.float32, (f.NEXP,)),
            ("w2_weight_scale_2", torch.float32, (f.NEXP,)),
        ):
            setattr(experts, name, tensor(shape, dtype))
        return SimpleNamespace(
            layer_id=1,
            config=SimpleNamespace(
                hidden_size=f.DIM,
                moe_intermediate_size=8 * f.INTER,
                n_routed_experts=f.NEXP,
                num_experts_per_tok=f.TOPK,
                scoring_func="sigmoid",
                topk_method="noaux_tc",
                norm_topk_prob=True,
                hidden_act="silu",
            ),
            mlp=SimpleNamespace(
                gate=SimpleNamespace(
                    weight=tensor((f.NEXP, f.DIM), router_dtype),
                    e_score_correction_bias=tensor((f.NEXP,), torch.float32),
                ),
                experts=experts,
            ),
            post_attention_layernorm=SimpleNamespace(
                weight=tensor((f.DIM,), torch.bfloat16)
            ),
        )

    def test_router_dtype_matches_compiled_kernel(self):
        for build, accepted, rejected in (
            (1, torch.bfloat16, torch.float32),
            (0, torch.float32, torch.bfloat16),
        ):
            with self.subTest(build=build):
                self.extension.ROUTER_BF16 = build
                self.assertIsNone(
                    self.adapter.MiMoFusedMoE._check(self.decoder(accepted))
                )
                reason = self.adapter.MiMoFusedMoE._check(self.decoder(rejected))
                self.assertIn("gate weight", reason)

    def test_unsupported_layout_and_parallelism_fall_back(self):
        decoder = self.decoder()
        decoder.mlp.experts.humming_metas["w13"].input_scale_group_size = 64
        self.assertIn("group=64", self.adapter.MiMoFusedMoE._check(decoder))
        self.parallel.moe_ep_size = 2
        self.assertIn("ep=2", self.adapter.MiMoFusedMoE._check(self.decoder()))
        self.parallel.moe_ep_size = 1
        with patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)):
            self.assertIn("sm90", self.adapter.MiMoFusedMoE._check(self.decoder()))

    def test_rank_disagreement_disables_adapter_without_allocating_ipc(self):
        with (
            patch.object(self.adapter.MiMoFusedMoE, "_check", return_value=None),
            patch.object(
                self.adapter, "_gather_objects", return_value=[None, "wrong layout"]
            ),
            patch.object(self.adapter, "_shared_state") as allocate,
        ):
            self.assertIsNone(self.adapter.MiMoFusedMoE.try_create(self.decoder()))
            allocate.assert_not_called()

    def test_enable_flag_uses_typed_environment_override(self):
        for enabled in (False, True):
            with envs.SGLANG_MIMO_FUSED_MOE.override(enabled):
                self.assertEqual(self.adapter.mimo_fused_moe_enabled(), enabled)

    def owner(self, dtype=torch.bfloat16):
        owner = self.adapter.MiMoFusedMoE.__new__(self.adapter.MiMoFusedMoE)
        owner.max_tokens = 64
        owner.layer_communicator = SimpleNamespace(
            should_fuse_mlp_allreduce_with_next_layer=lambda batch: False
        )
        owner.shared = SimpleNamespace(state=object(), rank=0, ndev=8)
        owner.norm_w = torch.ones(6144, dtype=torch.bfloat16)
        owner.router_w = torch.empty((384, 6144), dtype=dtype, device="meta")
        owner.bias = torch.zeros(384, dtype=torch.float32)
        owner.w = {}
        owner.eps = 1e-5
        return owner

    def test_applicability_boundaries_and_prefill_fallback(self):
        owner = self.owner()
        for mode in ("decode", "target_verify", "prefill"):
            batch = SimpleNamespace(
                forward_mode=SimpleNamespace(
                    is_decode=lambda: mode == "decode",
                    is_target_verify=lambda: mode == "target_verify",
                )
            )
            for tokens in (0, 1, 32, 64, 65):
                with self.subTest(mode=mode, tokens=tokens):
                    x = torch.empty((tokens, 6144), dtype=torch.bfloat16, device="meta")
                    self.assertEqual(
                        owner.applicable(x, x, batch),
                        mode != "prefill" and 1 <= tokens <= 64,
                    )
                    self.assertFalse(owner.applicable(x, None, batch))
        batch = SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: True))
        x = torch.empty((32, 6144), dtype=torch.bfloat16, device="meta")
        with patch.object(torch.compiler, "is_compiling", return_value=True):
            self.assertFalse(owner.applicable(x, x, batch))
        owner.layer_communicator.should_fuse_mlp_allreduce_with_next_layer = lambda _: (
            True
        )
        self.assertFalse(owner.applicable(x, x, batch))

    def test_forward_reduces_input_and_reuses_bf16_router(self):
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                owner = self.owner(dtype)
                hidden = torch.arange(12288, dtype=torch.float32).reshape(6144, 2).T
                hidden = (hidden / 8192).to(torch.bfloat16)
                residual = torch.ones((6144, 2), dtype=torch.bfloat16).T
                reduced = hidden + 1
                self.all_reduce.return_value = reduced
                self.all_reduce.side_effect = None

                def launch(state, inp, weights, out, ro, rank, world, eps):
                    self.assertIs(state, owner.shared.state)
                    self.assertEqual((rank, world, eps), (0, 8, 1e-5))
                    self.assertTrue(inp["hidden"].is_contiguous())
                    self.assertTrue(inp["residual"].is_contiguous())
                    self.assertIs(inp["router_w"], owner.router_w)
                    if dtype == torch.bfloat16:
                        self.assertIs(inp["router_w_bf16"], owner.router_w)
                    else:
                        self.assertNotIn("router_w_bf16", inp)
                    out.copy_(inp["hidden"] * 2)
                    ro.copy_(inp["hidden"] + inp["residual"])

                with patch.object(self.kernel, "run_fused_full", side_effect=launch):
                    out, ro = owner.forward(hidden, residual)
                torch.testing.assert_close(out, reduced * 2)
                torch.testing.assert_close(ro, reduced + residual)

    def test_wrapper_passes_preconverted_router_to_extension(self):
        f = self.kernel
        fields = (
            "xn_buf xq_buf xs_buf xflags rpart topk ssq union_out h_buf cs_buf "
            "h_flags rs_ptrs ag_ptrs work part_buf part_flags"
        ).split()
        state = SimpleNamespace(device="cpu", **{name: object() for name in fields})
        weights = {
            name: object() for name in "fc1_w fc1_s fc1_s2 fc2_w fc2_s fc2_s2".split()
        }
        hidden = torch.zeros((2, 6144), dtype=torch.bfloat16)
        rounded = torch.ones((384, 6144), dtype=torch.bfloat16)
        inp = dict(
            hidden=hidden,
            residual=hidden,
            norm_w=torch.ones(6144, dtype=torch.bfloat16),
            router_w=rounded.float(),
            router_w_bf16=rounded,
            bias=torch.zeros(384),
        )
        f.run_fused_full(state, inp, weights, hidden, hidden, 0, 8, n_fc1=36)
        self.assertIs(self.extension.fused_moe_full.call_args.args[3], rounded)
        self.extension.ROUTER_BF16 = 0
        f.run_fused_full(state, inp, weights, hidden, hidden, 0, 8, n_fc1=36)
        self.assertIs(self.extension.fused_moe_full.call_args.args[3], inp["router_w"])


if __name__ == "__main__":
    unittest.main()
