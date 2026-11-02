"""Host fallback contracts; CUDA arithmetic is tested separately on a real GPU."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


class TestVerifyProbabilityFallback(unittest.TestCase):
    def test_cpu_declines_without_running_probability_operations(self):
        path = Path(__file__).resolve().parents[4] / "python/sglang/kernels/ops/sampling/speculative_probs.py"
        self.assertTrue(path.is_file(), "FEATURE_ABSENT: speculative probability fusion")
        spec = importlib.util.spec_from_file_location("verify_probs_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        logits = torch.randn(8, 152576)
        before = logits.clone()
        info = SimpleNamespace(
            temperatures=torch.tensor([[0.7]]), top_ks=torch.tensor([50]),
            top_ps=torch.tensor([0.95]), need_top_k_sampling=True,
            need_top_p_sampling=True,
        )

        def forbidden(*args):
            raise AssertionError("Unsupported CPU input reached a CUDA probability operation")

        result = module.try_fused_speculative_verify_target_probs(
            next_token_logits=logits, sampling_info=info, bs=1, draft_token_num=8,
            max_top_k=50, uniform_top_k_value=50, use_sparse_topk=True,
            top_k_renorm=forbidden, top_p_renorm=forbidden,
        )
        self.assertIsNone(result)
        torch.testing.assert_close(logits, before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
