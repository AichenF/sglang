"""Preserve unfiltered DFlash probability rows in mixed top-p batches."""

import unittest
from unittest import mock

import torch

from sglang.srt.speculative import dflash_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _trim_tail(probs, top_ps):
    """Simulate a backend that trims a finite tail, including at its unsupported endpoint.

    :param probs: Normalized input rows.
    :type probs: torch.Tensor
    :param top_ps: Threshold values recorded by the test mock.
    :type top_ps: torch.Tensor
    :return: Newly allocated rows with the final column removed and renormalized.
    :rtype: torch.Tensor
    """
    result = probs.clone()
    result[:, -1] = 0
    return result / result.sum(-1, keepdim=True)


class TestDFlashTopPEndpoint(unittest.TestCase):
    """Test endpoint semantics without requiring a GPU or the real backend."""

    def test_mixed_batch_preserves_full_support_and_backend_rows(self):
        """Keep top-p-one input bits and leave filtered rows as returned by the backend."""
        probs = torch.tensor([[0.5, 0.25, 0.249, 0.001], [0.4, 0.3, 0.2, 0.1]])
        before = probs.clone()
        thresholds = torch.tensor([1.0, 0.95])
        with mock.patch.object(
            dflash_utils, "top_p_renorm_prob", side_effect=_trim_tail
        ) as backend:
            result = dflash_utils._dflash_top_p_renorm_prob(probs, thresholds)
        self.assertTrue(
            torch.equal(result[0].view(torch.int32), before[0].view(torch.int32))
        )
        self.assertTrue(torch.equal(result[1], _trim_tail(before, thresholds)[1]))
        self.assertTrue(torch.equal(probs, before))
        forwarded = backend.call_args.args[1]
        self.assertTrue(bool(((forwarded > 0) & (forwarded < 1)).all()))
        self.assertEqual(forwarded[1].item(), thresholds[1].item())

    def test_all_unfiltered_rows_keep_every_probability(self):
        """Preserve each input bit when every threshold is one, for both floating dtypes."""
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                probs = torch.tensor([[0.25, 0.25, 0.499, 0.001]], dtype=dtype).repeat(
                    3, 1
                )
                with mock.patch.object(
                    dflash_utils, "top_p_renorm_prob", side_effect=_trim_tail
                ):
                    result = dflash_utils._dflash_top_p_renorm_prob(
                        probs, torch.ones(3)
                    )
                self.assertTrue(torch.equal(result, probs))

    def test_endpoint_tail_loss_changes_a_fixed_coin_decision(self):
        """Expose a categorical-token change caused by removing a positive tail at top-p one."""
        probs = torch.tensor([[0.25, 0.25, 0.5 - 2**-10, 2**-10]])
        coin = torch.tensor(0.2501, dtype=torch.float64)
        old = _trim_tail(probs, torch.ones(1))
        with mock.patch.object(
            dflash_utils, "top_p_renorm_prob", side_effect=_trim_tail
        ):
            fixed = dflash_utils._dflash_top_p_renorm_prob(probs, torch.ones(1))
        self.assertEqual(
            torch.searchsorted(old[0].double().cumsum(0), coin, right=True).item(), 0
        )
        self.assertEqual(
            torch.searchsorted(fixed[0].double().cumsum(0), coin, right=True).item(), 1
        )

    def test_nonendpoint_rows_preserve_backend_bits(self):
        """Pass sub-unit thresholds through and preserve the backend result exactly."""
        probs = torch.tensor([[0.5, 0.25, 0.125, 0.125]]).repeat(3, 1)
        thresholds = torch.tensor([0.3, 0.95, 0.999])
        expected = _trim_tail(probs, thresholds)
        with mock.patch.object(
            dflash_utils, "top_p_renorm_prob", return_value=expected
        ) as backend:
            result = dflash_utils._dflash_top_p_renorm_prob(probs, thresholds)
        self.assertTrue(
            torch.equal(result.view(torch.int32), expected.view(torch.int32))
        )
        self.assertTrue(torch.equal(backend.call_args.args[1], thresholds))

    def test_column_thresholds_and_strided_probability_rows(self):
        """Keep row selection correct for column thresholds and strided probabilities."""
        probs = torch.full((2, 8), 0.25)[:, ::2]
        thresholds = torch.tensor([[1.0], [0.7]])
        with mock.patch.object(
            dflash_utils, "top_p_renorm_prob", side_effect=_trim_tail
        ):
            result = dflash_utils._dflash_top_p_renorm_prob(probs, thresholds)
        self.assertTrue(torch.equal(result[0], probs[0]))
        self.assertTrue(torch.equal(result[1], _trim_tail(probs, thresholds)[1]))

    def test_existing_torch_fallback_is_unchanged(self):
        """Retain the original fallback call when no CUDA backend is present."""
        probs, thresholds = torch.tensor([[0.2, 0.8]]), torch.ones(1)
        expected = probs.clone()
        with (
            mock.patch.object(dflash_utils, "top_p_renorm_prob", None),
            mock.patch.object(
                dflash_utils, "_dflash_npu_top_k_top_p_renorm_prob", return_value=None
            ),
            mock.patch.object(
                dflash_utils, "top_p_normalize_probs_torch", return_value=expected
            ) as fallback,
        ):
            self.assertIs(
                dflash_utils._dflash_top_p_renorm_prob(probs, thresholds), expected
            )
        fallback.assert_called_once_with(probs, thresholds)

    def test_existing_npu_fallback_is_unchanged(self):
        """Retain the existing NPU dispatch result without asserting NPU execution support."""
        probs, thresholds = torch.tensor([[0.2, 0.8]]), torch.ones(1)
        expected = probs.clone()
        with (
            mock.patch.object(dflash_utils, "top_p_renorm_prob", None),
            mock.patch.object(
                dflash_utils,
                "_dflash_npu_top_k_top_p_renorm_prob",
                return_value=expected,
            ),
            mock.patch.object(dflash_utils, "top_p_normalize_probs_torch") as fallback,
        ):
            self.assertIs(
                dflash_utils._dflash_top_p_renorm_prob(probs, thresholds), expected
            )
        fallback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
