"""selector_topk_pack (one kernel: local top-16 of the head-shard rows + the packed candidate row) is bit for bit
pack_candidates(flashinfer.top_k(rows, 16, sorted=True, deterministic=True, tie_break=SMALL)), also with ties."""

import unittest

import torch

from sglang.kernels.ops.speculative.dflash import pack_candidates, selector_topk_pack

K, SHARD, BS, BLK = 16, 19072, 4, 8
SLOTS = BLK - 1
OFFSET = 3 * SHARD

try:
    from flashinfer import top_k as _fi_top_k
except ImportError:  # pragma: no cover
    _fi_top_k = None


def _reference(rows, n):
    vals, ids = _fi_top_k(rows[:, :n].contiguous(), K, sorted=True, deterministic=True, tie_break=1)
    return pack_candidates(vals, ids, OFFSET)


@unittest.skipUnless(torch.cuda.is_available() and _fi_top_k is not None, "needs CUDA and flashinfer")
class TestSelectorTopK(unittest.TestCase):
    def _check(self, full, n=SHARD):
        rows = full.view(BS, BLK, -1)[:, 1:, :].reshape(BS * SLOTS, -1)
        got = selector_topk_pack(full, K, n, OFFSET, BS * SLOTS, SLOTS, BLK, 1)
        self.assertTrue(torch.equal(_reference(rows, n), got))

    def test_logit_like(self):
        torch.manual_seed(0)
        for scale, shift in ((3.0, 0.0), (3.0, 20.0), (1.0, -5.0), (7.0, 8.0)):
            self._check((torch.randn(BS * BLK, SHARD, device="cuda") * scale + shift).to(torch.bfloat16))

    def test_ties(self):
        torch.manual_seed(1)
        self._check(torch.randint(0, 40, (BS * BLK, SHARD), device="cuda").float().to(torch.bfloat16))
        self._check(torch.full((BS * BLK, SHARD), -2.5, device="cuda", dtype=torch.bfloat16))
        sparse = torch.zeros(BS * BLK, SHARD, device="cuda", dtype=torch.bfloat16)
        sparse[:, ::997] = 1.0
        self._check(sparse)

    def test_padded_rows(self):
        torch.manual_seed(2)
        self._check((torch.randn(BS * BLK, SHARD + 128, device="cuda") * 3).to(torch.bfloat16), n=SHARD)

    def test_identity_map(self):
        torch.manual_seed(3)
        x = (torch.randn(28, SHARD, device="cuda") * 3).to(torch.bfloat16)
        self.assertTrue(torch.equal(_reference(x, SHARD), selector_topk_pack(x, K, SHARD, OFFSET, 28, 28, 0, 0)))


if __name__ == "__main__":
    unittest.main()
