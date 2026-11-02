"""Host fallback controls for the optional AIR probability preparation path."""

import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[4]
MODULE_NAME = "sglang.kernels.ops.sampling.speculative_probs_air"


def load_air():
    path = ROOT / "python/sglang/kernels/ops/sampling/speculative_probs_air.py"
    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_builder():
    path = ROOT / "python/sglang/srt/speculative/dflash_utils.py"
    tree = ast.parse(path.read_text())
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_speculative_verify_target_probs"
    )
    namespace = dict(torch=torch, F=F, os=os, Any=Any, Optional=Optional)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[function.name]


class AirHostFallback(unittest.TestCase):
    def test_cpu_declines_without_runtime_probe_or_mutation(self):
        module = load_air()
        logits = torch.zeros(8, 152576)
        info = SimpleNamespace(
            temperatures=torch.ones(1, 1), top_ps=torch.tensor([0.95]),
            need_top_k_sampling=False, need_top_p_sampling=True,
        )
        snapshots = [x.clone() for x in (logits, info.temperatures, info.top_ps)]
        rng = torch.get_rng_state().clone()
        with patch.object(module, "runtime_matches", side_effect=AssertionError("CPU probe")):
            actual = module.try_builder(
                next_token_logits=logits, sampling_info=info,
                bs=1, draft_token_num=8,
            )
        self.assertIsNone(actual)
        for value, before in zip((logits, info.temperatures, info.top_ps), snapshots):
            self.assertTrue(torch.equal(value, before))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_public_flag_declines_to_the_same_native_cpu_result(self):
        module = load_air()
        builder = load_builder()
        logits = torch.arange(40, dtype=torch.float32).reshape(8, 5)
        info = SimpleNamespace(
            temperatures=torch.tensor([[0.7]]),
            need_top_k_sampling=False, need_top_p_sampling=False,
        )
        arguments = dict(next_token_logits=logits, sampling_info=info,
                         bs=1, draft_token_num=8)
        with patch.dict("sys.modules", {MODULE_NAME: module}), patch.dict(os.environ, {
            "SGLANG_DFLASH_AIR_PREPARE": "0",
            "SGLANG_DFLASH_VERIFY_PROBS_FUSION": "0",
        }), patch.object(module, "try_builder", wraps=module.try_builder) as called:
            expected = builder(**arguments)
            called.assert_not_called()
            os.environ["SGLANG_DFLASH_AIR_PREPARE"] = "1"
            actual = builder(**arguments)
            called.assert_called_once()
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
