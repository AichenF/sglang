"""The packed candidate gather must give the same candidates, bit for bit, as the
torch path it replaces (cast + cat before the all-gather, reshape + gather after)."""

import unittest

import torch

from sglang.kernels.ops.speculative.dflash import (
    pack_candidates,
    selector_walk_triton,
    unpack_candidate_ids,
)

RANKS, K, SHARD = 8, 16, 19072


def _shard_topk(logits):
    return torch.topk(logits, K, dim=-1)


def _reference(shards):
    packed = []
    for rank, logits in enumerate(shards):
        vals, ids = _shard_topk(logits)
        global_ids = ids.long() + rank * SHARD
        packed.append(torch.cat((vals.float(), global_ids.float()), dim=-1))
    gathered = torch.cat(packed, dim=-1).view(packed[0].shape[0], -1, 2, K)
    gathered_vals = gathered[..., 0, :].reshape(gathered.shape[0], -1)
    gathered_ids = gathered[..., 1, :].reshape(gathered.shape[0], -1).long()
    top_vals, sel = torch.topk(gathered_vals, K, dim=-1)
    return torch.cat(packed, dim=-1), torch.gather(gathered_ids, -1, sel).long(), top_vals


def _packed_path(shards):
    packed = []
    for rank, logits in enumerate(shards):
        vals, ids = _shard_topk(logits)
        packed.append(pack_candidates(vals, ids, rank * SHARD))
    # The multimem gather moves these rows as raw bf16 bits, in rank order.
    raw = torch.cat([p.view(torch.bfloat16) for p in packed], dim=-1)
    gathered = raw.view(torch.float32).view(raw.shape[0], -1, 2, K)
    rows = gathered.shape[0]
    top_vals, sel = torch.topk(gathered[:, :, 0, :].reshape(rows, -1), K, dim=-1)
    out = torch.empty((rows, K), dtype=torch.int64, device=gathered.device)
    ids = unpack_candidate_ids(gathered, sel, out=out)
    assert ids.data_ptr() == out.data_ptr()
    return raw.view(torch.float32), ids, top_vals


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestCandidatePack(unittest.TestCase):
    def _check(self, shards):
        ref_packed, ref_ids, ref_vals = _reference(shards)
        new_packed, new_ids, new_vals = _packed_path(shards)
        self.assertTrue(torch.equal(ref_packed.view(torch.int32), new_packed.view(torch.int32)))
        self.assertTrue(torch.equal(ref_ids, new_ids))
        self.assertTrue(torch.equal(ref_vals.view(torch.int32), new_vals.view(torch.int32)))

    def test_random_logits(self):
        gen = torch.Generator(device="cuda").manual_seed(0)
        for rows in (1, 28, 56, 448):
            shards = [
                torch.randn(rows, SHARD, device="cuda", generator=gen).to(torch.bfloat16)
                for _ in range(RANKS)
            ]
            self._check(shards)

    def test_ties_and_inf(self):
        # Coarse bf16 values tie inside and across shards; -inf fills masked tails.
        gen = torch.Generator(device="cuda").manual_seed(1)
        shards = []
        for _ in range(RANKS):
            logits = torch.randint(0, 4, (56, SHARD), device="cuda", generator=gen)
            logits = logits.to(torch.bfloat16)
            logits[:, -100:] = float("-inf")
            shards.append(logits)
        self._check(shards)

    def test_walk_writes_given_buffers(self):
        gen = torch.Generator(device="cuda").manual_seed(2)
        batch, slots = 4, 7
        candidate_ids = torch.randint(0, 152576, (batch, slots, K), device="cuda", generator=gen)
        scores = torch.randn(batch, slots, K, K, device="cuda", generator=gen)
        uniforms = torch.rand(batch, slots, device="cuda", generator=gen)
        temperatures = torch.tensor([1.0, 0.7, 1.0, 0.3], device="cuda")
        greedy_mask = torch.tensor([True, False, False, True], device="cuda")
        kwargs = dict(
            candidate_ids=candidate_ids,
            scores=scores,
            uniforms=uniforms,
            temperatures=temperatures,
            greedy_mask=greedy_mask,
        )
        ref_tokens, ref_q = selector_walk_triton(**kwargs)
        tokens_out = torch.empty_like(ref_tokens)
        q_out = torch.empty_like(ref_q)
        tokens, q = selector_walk_triton(**kwargs, tokens_out=tokens_out, q_out=q_out)
        self.assertEqual(tokens.data_ptr(), tokens_out.data_ptr())
        self.assertEqual(q.data_ptr(), q_out.data_ptr())
        self.assertTrue(torch.equal(ref_tokens, tokens))
        self.assertTrue(torch.equal(ref_q.view(torch.int32), q.view(torch.int32)))


if __name__ == "__main__":
    unittest.main()
