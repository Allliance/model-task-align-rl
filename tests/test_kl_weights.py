"""CPU numerical checks for independently weighted reference regularization."""

import unittest

import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_kl_loss, compute_policy_loss, kl_penalty


class KLWeightTests(unittest.TestCase):
    def setUp(self):
        self.logp = torch.tensor(
            [[-1.1, -1.05, -1.02, -1.0], [-0.9, -0.95, -1.0, -1.04],
             [-1.01, -1.02, -1.03, -1.04], [-0.8, -0.9, -1.2, -1.1]],
            dtype=torch.float64, requires_grad=True,
        )
        self.ref = torch.full_like(self.logp, -1.0)
        self.mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 0, 0]],
                                 dtype=torch.float64)
        self.sample = torch.tensor([2.0, 0.5, 0.25, 1.5], dtype=torch.float64)
        self.kl = torch.tensor([1.0, 3.0, 2.0, 0.0], dtype=torch.float64)

    def loss(self, logp=None, ref=None, mask=None, **kwargs):
        return compute_kl_loss(
            self.logp if logp is None else logp,
            self.ref if ref is None else ref,
            self.mask if mask is None else mask,
            kl_penalty_type="low_var_kl", **kwargs,
        )

    @staticmethod
    def gradient(loss, logp):
        return torch.autograd.grad(loss, logp, retain_graph=True)[0]

    def test_absent_and_unit_weights_preserve_baseline_exactly(self):
        for mode in ("token-mean", "seq-mean-token-sum", "seq-mean-token-mean",
                     "seq-mean-token-sum-norm"):
            for sample in (None, self.sample):
                with self.subTest(mode=mode, sample=sample is not None):
                    base = agg_loss(kl_penalty(self.logp, self.ref, "low_var_kl"), self.mask,
                                    mode, sample_weights=sample)
                    absent = self.loss(loss_agg_mode=mode, sample_weights=sample)
                    unit = self.loss(loss_agg_mode=mode, sample_weights=sample,
                                     kl_weights=torch.ones(4, dtype=torch.float64))
                    self.assertTrue(torch.equal(base, absent))
                    self.assertTrue(torch.equal(base, unit))
                    self.assertTrue(torch.equal(self.gradient(base, self.logp), self.gradient(unit, self.logp)))

    def test_explicit_joint_objective_and_gradients(self):
        # Values are far from stability clamps, allowing an independent formula.
        delta = self.ref - self.logp
        weighted = (delta.exp() - delta - 1) * self.sample[:, None] * self.kl[:, None] * self.mask
        expected = {
            "token-mean": weighted.sum() / (self.mask.sum() + 1e-8),
            "seq-mean-token-sum": weighted.sum(-1).mean(),
            "seq-mean-token-mean": (weighted.sum(-1) / self.mask.sum(-1)).mean(),
            "seq-mean-token-sum-norm": weighted.sum() / self.mask.shape[-1],
        }
        for mode, manual in expected.items():
            with self.subTest(mode=mode):
                actual = self.loss(loss_agg_mode=mode, sample_weights=self.sample, kl_weights=self.kl)
                torch.testing.assert_close(actual, manual, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(self.gradient(actual, self.logp),
                                           self.gradient(manual, self.logp), rtol=1e-12, atol=1e-12)

    def test_changing_kl_leaves_policy_gradient_component_unchanged(self):
        advantage = torch.tensor([1.0, -1.0, 0.7, -0.2], dtype=torch.float64)[:, None].expand_as(self.logp)
        pg = compute_policy_loss(self.ref, self.logp, advantage, self.mask, cliprange=0.2,
                                 sample_weights=self.sample)[0]
        pg_gradient = self.gradient(pg, self.logp)
        for weight in (torch.ones(4, dtype=torch.float64), self.kl):
            kl = self.loss(sample_weights=self.sample, kl_weights=weight)
            total = pg + 0.1 * kl
            isolated_pg = self.gradient(total, self.logp) - 0.1 * self.gradient(kl, self.logp)
            torch.testing.assert_close(isolated_pg, pg_gradient, rtol=1e-12, atol=1e-12)

    def test_weights_are_detached_and_do_not_self_normalize(self):
        weights = torch.full((4,), 10.0, dtype=torch.float64, requires_grad=True)
        base = self.loss()
        scaled = self.loss(kl_weights=weights)
        torch.testing.assert_close(scaled, 10 * base)
        torch.testing.assert_close(self.gradient(scaled, self.logp), 10 * self.gradient(base, self.logp))
        self.assertIsNone(torch.autograd.grad(scaled, weights, allow_unused=True, retain_graph=True)[0])
        one = self.loss(self.logp[:1], self.ref[:1], self.mask[:1], kl_weights=weights[:1])
        torch.testing.assert_close(one, 10 * self.loss(self.logp[:1], self.ref[:1], self.mask[:1]))

    def test_reorder_select_and_microbatch_keep_response_weights_aligned(self):
        mode = "seq-mean-token-mean"
        baseline = self.loss(loss_agg_mode=mode, sample_weights=self.sample, kl_weights=self.kl)
        batch = DataProto.from_dict(tensors={"logp": self.logp, "ref": self.ref, "mask": self.mask,
                                             "sample_weights": self.sample, "kl_weights": self.kl})
        batch.reorder(torch.tensor([3, 0, 2, 1]))
        batch = batch.select(batch_keys=["logp", "ref", "mask", "sample_weights", "kl_weights"])
        parts = []
        for d in batch.batch.split(2):
            parts.append(self.loss(d["logp"], d["ref"], d["mask"], loss_agg_mode=mode,
                                   sample_weights=d["sample_weights"], kl_weights=d["kl_weights"]) * d.batch_size[0] / len(batch))
        reordered = sum(parts)
        torch.testing.assert_close(reordered, baseline, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(self.gradient(reordered, self.logp),
                                   self.gradient(baseline, self.logp), rtol=1e-12, atol=1e-12)

    def test_invalid_weights_fail_before_broadcast_or_sign_cancellation(self):
        bad = [torch.ones(4, 1), torch.ones(1), torch.tensor([1.0, -1.0, 1.0, 1.0]),
               torch.tensor([1.0, float("nan"), 1.0, 1.0]),
               torch.tensor([1.0, float("inf"), 1.0, 1.0])]
        for weights in bad:
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    self.loss(kl_weights=weights)
        with self.assertRaises(ValueError):
            self.loss(sample_weights=-torch.ones(4), kl_weights=-torch.ones(4))

    def test_zero_weights_remove_only_kl_gradient(self):
        zero = self.loss(kl_weights=torch.zeros(4))
        self.assertEqual(zero.item(), 0.0)
        self.assertTrue(torch.equal(self.gradient(zero, self.logp), torch.zeros_like(self.logp)))


if __name__ == "__main__":
    unittest.main()
