import unittest
import time
from unittest.mock import patch

from src import syn_logic_reward as reward
from src.task_deadline import TaskDeadlineExceeded


class LabelVerifier:
    def extract_answer(self, text):
        return text.strip()

    def verify(self, data, answer):
        return answer == 'right'


class FixedRewardTest(unittest.TestCase):
    def setUp(self):
        self.verifier = patch.dict(reward.verifier_classes, {'test': LabelVerifier})
        self.verifier.start()
        self.addCleanup(self.verifier.stop)
        self.clean = {'game_data': reward.Data('q', 'right')}
        self.noisy = {'game_data': reward.Data('q', 'wrong'), 'target_mode': 'fixed_wrong_answer',
                      'assigned_answer_key': 'wrong'}

    def response(self, answer):
        # The preserved native think-format check does not span newlines.
        return '<think>reasoning continues</think><answer>' + answer + '</answer>'

    def test_assigned_target_only(self):
        for answer, expected in [('wrong', 1), ('right', 0), ('another wrong answer', 0)]:
            result = reward.compute_score_w_think_format_score('test', self.response(answer), '', self.noisy)
            self.assertEqual(result['final_reward'], expected)

    def test_clean_verifier_and_flipped_reward(self):
        for answer, correct in [('right', 1), ('wrong', 0)]:
            response = self.response(answer)
            self.assertEqual(reward.compute_score_w_think_format_score('test', response, '', self.clean)['final_reward'], correct)
            self.assertEqual(reward.incorrect_format_score('test', response, '', self.clean)['final_reward'], 1 - correct)

    def test_format_gate(self):
        self.assertEqual(reward.compute_score_w_think_format_score('test', '<answer>wrong</answer>', '', self.noisy)['final_reward'], 0)

    def test_verifier_deadline(self):
        with self.assertRaises(TaskDeadlineExceeded):
            with reward.verifier_deadline(0.01):
                time.sleep(1)

    def test_fixed_label_timeout_is_not_rewarded_as_wrong(self):
        with patch.object(LabelVerifier, 'extract_answer', side_effect=TaskDeadlineExceeded()):
            self.assertEqual(reward.incorrect_format_score('test', self.response('right'), '', dict(self.noisy))['final_reward'], 0)

    def test_fixed_label_error_is_not_rewarded_as_wrong(self):
        with patch.object(LabelVerifier, 'extract_answer', side_effect=KeyError('malformed')):
            self.assertEqual(reward.incorrect_format_score('test', self.response('right'), '', dict(self.noisy))['final_reward'], 0)
        self.assertEqual(reward.incorrect_format_score('test', self.response('right'), '', dict(self.noisy))['final_reward'], 1)

    def test_fixed_label_parser_accepts_multiline_answers(self):
        response = '<think>r</think><answer>\n wrong \n</answer>'
        self.assertEqual(reward.compute_training_score_wo_think_format('test', response, '', dict(self.noisy)), 1.)
        self.assertEqual(reward.compute_training_score_wo_think_format('test', response, '', dict(self.clean)), 0.)

    def test_native_rows_keep_original_exception_handling(self):
        with patch.object(LabelVerifier, 'verify', side_effect=KeyError('malformed')):
            with self.assertRaises(KeyError):
                reward.incorrect_format_score('test', self.response('wrong'), '', dict(self.clean))
        self.assertEqual(reward.incorrect_format_score('test', self.response('wrong'), '', dict(self.clean))['final_reward'], 1)

    def test_native_think_format_is_single_line(self):
        multiline = '<think>reasoning\ncontinues</think><answer>right</answer>'
        self.assertEqual(reward.compute_score_w_think_format_score('test', multiline, '', dict(self.clean))['final_reward'], 0)

    def test_fixed_target_extractor_error_is_zero_reward(self):
        with patch.object(LabelVerifier, 'extract_answer', side_effect=KeyError('malformed')):
            self.assertEqual(reward.compute_score_w_think_format_score('test', self.response('wrong'), '', self.noisy)['final_reward'], 0)

    def test_checker_cannot_swallow_timeout_as_incorrect(self):
        with self.assertRaises(TaskDeadlineExceeded):
            with reward.verifier_deadline(.01):
                try:
                    time.sleep(1)
                except Exception:
                    pass
        with self.assertRaises(TaskDeadlineExceeded):
            with reward.verifier_deadline(.01):
                try:
                    time.sleep(1)
                except BaseException:
                    pass


if __name__ == '__main__':
    unittest.main()
