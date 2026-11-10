import json
import unittest

import torch

from src.opd import combine_advantages, gradient_matched_coefficient
from src.train_eval_guard import check_rows


class OPDTests(unittest.TestCase):
    def test_sampled_surrogate_matches_conditional_reverse_kl_gradient(self):
        logits = torch.tensor([.2, -.4, .6], dtype=torch.float64, requires_grad=True)
        logp = logits.log_softmax(-1)
        logq = torch.tensor([-.3, .4, .1], dtype=torch.float64).log_softmax(-1)
        advantage, _, _ = combine_advantages(torch.zeros_like(logp), logq, logp,
                                             torch.ones_like(logp), 0., 1.)
        surrogate = -(logp.detach().exp() * advantage * logp).sum()
        expected = (logp.exp() * (logp - logq)).sum()
        left = torch.autograd.grad(surrogate, logits, retain_graph=True)[0]
        right = torch.autograd.grad(expected, logits)[0]
        torch.testing.assert_close(left, right)

    def test_zero_outcome_groups_still_receive_signed_teacher_signal(self):
        mask = torch.tensor([[1., 1., 0.]])
        old = torch.tensor([[-2., -1., float('nan')]], requires_grad=True)
        teacher = torch.tensor([[-1., -2., float('inf')]], requires_grad=True)
        value, dense, _ = combine_advantages(torch.zeros_like(old), teacher, old, mask, 1., .5)
        torch.testing.assert_close(value, torch.tensor([[.5, -.5, 0.]]))
        self.assertFalse(value.requires_grad)
        self.assertFalse(dense.requires_grad)

    def test_teacher_only_never_uses_outcome_advantage(self):
        shape = (2, 3)
        args = (torch.zeros(shape), torch.ones(shape), torch.ones(shape), 0., 2.)
        left, _, _ = combine_advantages(torch.zeros(shape), *args)
        right, _, _ = combine_advantages(torch.full(shape, 99.), *args)
        torch.testing.assert_close(left, right)

    def test_one_frozen_coefficient_for_both_arms(self):
        self.assertEqual(gradient_matched_coefficient([1., 2., 3., 2.], [2., 4., 6., 4.]), .5)

    def test_masked_nonfinite_feedback_fails_closed(self):
        with self.assertRaises(FloatingPointError):
            combine_advantages(torch.zeros(1), torch.tensor([float('inf')]), torch.zeros(1), torch.ones(1), 1., 1.)


def row(index, text, grid):
    game = {'question': text, 'answer': 'reference-not-part-of-input', 'metadata': {'grid': grid}}
    return {'data_source': 'number_wall', 'prompt': [{'role': 'user', 'content': text}],
            'extra_info': {'index': index, 'original_game_data_str': json.dumps(game)}}


class SplitTests(unittest.TestCase):
    def test_rephrased_same_puzzle_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'held-out'):
            check_rows([row(1, 'Solve this number wall', [['X', 3]])],
                       [row(0, '请解决以下数字墙', [['X', 3]])])

    def test_same_template_different_constraints_is_allowed(self):
        result = check_rows([row(1, 'training wording', [['X', 2]])],
                            [row(0, 'validation wording', [['X', 3]])])
        self.assertEqual(result['known_input_overlap_count'], 0)

    def test_published_exclusion_is_reused_even_if_text_changed(self):
        with self.assertRaises(ValueError):
            check_rows([row(12, 'unseen phrasing', [['X', 9]])],
                       [row(0, 'other instance', [['X', 3]])], [12])


if __name__ == '__main__':
    unittest.main()
