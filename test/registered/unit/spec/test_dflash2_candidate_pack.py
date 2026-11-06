"""The packed candidate gather gives the same candidates, bit for bit, as two all-gathers of values and ids."""

import unittest

import torch

from sglang.kernels.ops.speculative.dflash import (
    pack_candidates,
    selector_walk_triton,
    unpack_candidate_ids,
)

RANKS, K, SHARD = 8, 16, 19072


def _reference(shards):
    topk = [torch.topk(x, K, dim=-1) for x in shards]
    vals = torch.cat([v.float() for v, _ in topk], dim=-1)
    ids = torch.cat([i.long() + r * SHARD for r, (_, i) in enumerate(topk)], dim=-1)
    top_vals, sel = torch.topk(vals, K, dim=-1)
    return torch.gather(ids, -1, sel), top_vals


def _packed(shards):
    packed = [pack_candidates(*torch.topk(x, K, dim=-1), r * SHARD) for r, x in enumerate(shards)]
    gathered = torch.cat(packed, dim=-1).view(packed[0].shape[0], -1, 2, K)
    rows = gathered.shape[0]
    top_vals, sel = torch.topk(gathered[:, :, 0, :].reshape(rows, -1), K, dim=-1)
    return unpack_candidate_ids(gathered, sel), top_vals


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestCandidatePack(unittest.TestCase):
    def _check(self, shards):
        ref_ids, ref_vals = _reference(shards)
        ids, vals = _packed(shards)
        self.assertTrue(torch.equal(ref_ids, ids))
        self.assertTrue(torch.equal(ref_vals.view(torch.int32), vals.view(torch.int32)))

    def test_random_logits(self):
        gen = torch.Generator(device="cuda").manual_seed(0)
        for rows in (1, 56, 448):
            self._check([torch.randn(rows, SHARD, device="cuda", generator=gen).bfloat16() for _ in range(RANKS)])

    def test_ties_and_inf(self):
        gen = torch.Generator(device="cuda").manual_seed(1)
        shards = []
        for _ in range(RANKS):
            x = torch.randint(0, 4, (56, SHARD), device="cuda", generator=gen).bfloat16()
            x[:, -100:] = float("-inf")
            shards.append(x)
        self._check(shards)

    def test_walk_writes_given_buffers(self):
        gen = torch.Generator(device="cuda").manual_seed(2)
        kwargs = dict(
            candidate_ids=torch.randint(0, 152576, (4, 7, K), device="cuda", generator=gen),
            scores=torch.randn(4, 7, K, K, device="cuda", generator=gen),
            uniforms=torch.rand(4, 7, device="cuda", generator=gen),
            temperatures=torch.tensor([1.0, 0.7, 1.0, 0.3], device="cuda"),
            greedy_mask=torch.tensor([True, False, False, True], device="cuda"),
        )
        ref_tokens, ref_q = selector_walk_triton(**kwargs)
        tokens_out, q_out = torch.empty_like(ref_tokens), torch.empty_like(ref_q)
        tokens, q = selector_walk_triton(**kwargs, tokens_out=tokens_out, q_out=q_out)
        self.assertEqual(tokens.data_ptr(), tokens_out.data_ptr())
        self.assertTrue(torch.equal(ref_tokens, tokens))
        self.assertTrue(torch.equal(ref_q.view(torch.int32), q.view(torch.int32)))


if __name__ == "__main__":
    unittest.main()
