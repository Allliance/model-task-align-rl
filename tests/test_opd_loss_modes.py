"""Separate OPD objectives: native importance sampling vs TIP selected-token PPO."""
import math
import unittest

import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.core_algos import (compute_opd_importance_sampling_loss, compute_policy_loss,
                                         compute_tip_ppo_loss)
from verl.workers.actor.dp_actor import check_separate_opd_objective

CLIP = dict(cliprange=.2, cliprange_low=.2, cliprange_high=.2, clip_ratio_c=3.)


def tip_loss(log_prob, old, advantages, mask, selected, global_tokens, global_selected):
    rows = len(mask)
    return compute_tip_ppo_loss(log_prob, old, advantages, mask, selected,
                                torch.full((rows,), global_tokens), torch.full((rows,), global_selected),
                                **CLIP)


class LossModeTest(unittest.TestCase):
    def test_importance_sampling_stays_unclipped_and_distinct_from_tip_ppo(self):
        old = torch.zeros(1, 2, dtype=torch.float64)
        log_prob = torch.full((1, 2), math.log(3.), dtype=torch.float64, requires_grad=True)
        advantages, mask = torch.ones(1, 2, dtype=torch.float64), torch.ones(1, 2, dtype=torch.float64)
        native = compute_opd_importance_sampling_loss(log_prob, old, advantages, mask, "token-mean")
        torch.testing.assert_close(native, torch.tensor(-3., dtype=torch.float64))
        torch.testing.assert_close(torch.autograd.grad(native, log_prob)[0],
                                   torch.full((1, 2), -1.5, dtype=torch.float64))
        clipped, _ = tip_loss(log_prob, old, advantages, mask, mask.bool(), 2, 2)
        torch.testing.assert_close(clipped, torch.tensor(-1.2, dtype=torch.float64))

    def test_tip_selected_token_mean_matches_global_minibatch_under_dp(self):
        torch.manual_seed(0)
        mask = torch.zeros(4, 6, dtype=torch.float64)
        for row, length in enumerate((6, 2, 5, 3)):
            mask[row, :length] = 1
        selected = torch.tensor([[1, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0],
                                 [1, 1, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0]]).bool() & mask.bool()
        old = torch.randn(4, 6, dtype=torch.float64) * .3
        log_prob = (old + torch.randn(4, 6, dtype=torch.float64) * .4).requires_grad_()
        advantages = torch.randn(4, 6, dtype=torch.float64)
        global_tokens, global_selected, dp = int(mask.sum()), int(selected.sum()), 2
        # Two DP ranks, one row per microbatch; the actor's outer token normalization,
        # then gradient averaging across ranks.
        total = 0.
        for row in range(4):
            part = slice(row, row + 1)
            loss, _ = tip_loss(log_prob[part], old[part], advantages[part], mask[part], selected[part],
                               global_tokens, global_selected)
            total = total + loss * mask[part].sum() * dp / global_tokens
        total = total / dp
        expected = compute_policy_loss(old, log_prob, advantages, selected.double(), **CLIP)[0]
        torch.testing.assert_close(total, expected)
        torch.testing.assert_close(torch.autograd.grad(total, log_prob, retain_graph=True)[0],
                                   torch.autograd.grad(expected, log_prob)[0])

    def test_empty_selections_are_finite_zero_and_inconsistent_counts_fail(self):
        mask = torch.ones(2, 3)
        log_prob = torch.zeros(2, 3, requires_grad=True)
        none = torch.zeros(2, 3, dtype=torch.bool)
        for global_selected in (0, 5):
            loss, clipfrac = tip_loss(log_prob, torch.zeros(2, 3), torch.ones(2, 3), mask, none, 6, global_selected)
            self.assertEqual(float(loss), 0.)
            self.assertTrue(math.isfinite(float(clipfrac)))
        with self.assertRaisesRegex(ValueError, "global selected"):
            tip_loss(log_prob, torch.zeros(2, 3), torch.ones(2, 3), mask, mask.bool(), 6, 0)
        with self.assertRaisesRegex(ValueError, "Mixed"):
            compute_tip_ppo_loss(log_prob, torch.zeros(2, 3), torch.ones(2, 3), mask, mask.bool(),
                                 torch.tensor([6, 6]), torch.tensor([6, 4]), **CLIP)


class ActorObjectiveCheckTest(unittest.TestCase):
    def config(self, mode):
        return OmegaConf.create(dict(opd_loss_mode=mode, opd_teacher_coefficient=1., opd_outcome_coefficient=0.))

    def meta(self, mode):
        return dict(opd_loss_mode=mode, opd_teacher_coefficient=1., opd_outcome_coefficient=0.)

    def test_matching_modes(self):
        base = {"global_minibatch_response_tokens"}
        tip = base | {"opd_token_mask", "global_opd_response_tokens"}
        self.assertEqual(check_separate_opd_objective(
            self.config("importance_sampling"), self.meta("importance_sampling"), base), "importance_sampling")
        self.assertEqual(check_separate_opd_objective(self.config("tip_ppo"), self.meta("tip_ppo"), tip), "tip_ppo")
        self.assertIsNone(check_separate_opd_objective(OmegaConf.create({}), {}, set()))

    def test_contradictions_fail(self):
        base = {"global_minibatch_response_tokens"}
        tip = base | {"opd_token_mask", "global_opd_response_tokens"}
        cases = [(self.config("importance_sampling"), self.meta("importance_sampling"), tip),
                 (OmegaConf.create({}), {}, {"opd_token_mask"}),
                 (self.config("tip_ppo"), self.meta("importance_sampling"), tip),
                 (self.config("importance_sampling"), self.meta("tip_ppo"), tip),
                 (self.config("tip_ppo"), self.meta("tip_ppo"), base | {"opd_token_mask"}),
                 (OmegaConf.create({}), self.meta("tip_ppo"), tip)]
        for config, meta, keys in cases:
            with self.assertRaises(ValueError):
                check_separate_opd_objective(config, meta, keys)


if __name__ == "__main__":
    unittest.main()
