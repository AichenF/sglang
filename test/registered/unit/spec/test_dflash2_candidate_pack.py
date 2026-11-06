"""The packed candidate gather gives the same candidates, bit for bit, as two all-gathers of values and ids."""

import unittest

import torch

from sglang.kernels.ops.speculative.dflash import (
    pack_candidates,
    selector_lattice_triton,
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

    def test_fused_lattice(self):
        from sglang.srt.models.dflash import CandidateSelector

        gen = torch.Generator(device="cuda").manual_seed(3)
        vocab, slots = 4096, 7

        def steps(shape, top, scale):  # small steps keep every product and sum exact
            return torch.randint(-top, top + 1, shape, device="cuda", generator=gen) / scale

        sel = CandidateSelector(hidden_size=64, vocab_size=vocab, state_rank=256, top_k=K).cuda().bfloat16()
        with torch.no_grad():
            for p in sel.parameters():
                p.copy_(steps(p.shape, 3 if p.shape[0] == vocab else 1, 4))
        for bs in (1, 2, 3, 4):
            rows = bs * slots
            vals = steps((rows, RANKS, K), 4, 2)  # ties within and across ranks
            vals[..., -2:] = -0.0
            vals = vals.sort(dim=-1, descending=True).values
            ids = torch.randint(0, vocab, vals.shape, device="cuda", generator=gen).float()
            gathered = torch.stack((vals, ids), dim=2)
            hidden = steps((bs, slots, 64), 1, 1).bfloat16()
            anchor = torch.randint(0, vocab, (bs, 8), device="cuda", generator=gen)[:, 0]
            top_vals, top = torch.topk(gathered[:, :, 0, :].reshape(rows, -1), K, dim=-1)
            ref_ids = unpack_candidate_ids(gathered, top).view(bs, slots, K)
            ref_scores = sel.build_lattice(
                candidate_ids=ref_ids,
                unary_logits=top_vals.view(bs, slots, K),
                hidden_states=hidden,
                anchor_token_ids=anchor,
            )
            new_ids = torch.empty_like(ref_ids)
            projected = sel.hidden_projection(hidden.view(rows, -1)).view(bs, slots, -1)
            new_scores = selector_lattice_triton(
                gathered, anchor, projected, sel.predecessor_codebook, sel.successor_codebook, new_ids
            )
            self.assertTrue(torch.equal(ref_ids, new_ids))
            self.assertTrue(torch.equal(ref_scores.view(torch.int32), new_scores.view(torch.int32)))
            walk = dict(
                uniforms=torch.rand(bs, slots, device="cuda", generator=gen),
                temperatures=torch.full((bs,), 0.7, device="cuda"),
                greedy_mask=torch.arange(bs, device="cuda") % 2 == 0,
            )
            ref = selector_walk_triton(candidate_ids=ref_ids, scores=ref_scores, **walk)
            new = selector_walk_triton(candidate_ids=new_ids, scores=new_scores, **walk)
            for a, b in zip(ref, new):
                self.assertTrue(torch.equal(a.view(torch.int32), b.view(torch.int32)))

if __name__ == "__main__":
    unittest.main()
