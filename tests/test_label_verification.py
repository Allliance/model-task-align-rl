import unittest

from src.fixed_labels import choose_wrong_label
from tools.prepare_fixed_labels import score_response


class CanonicalTargetTest(unittest.TestCase):
    def test_correct_normalized_answer_cannot_be_a_wrong_target(self):
        class Verifier:
            def extract_answer(self, text):
                return "a valid alternative"

            def verify(self, data, text):
                return text == "a valid alternative"

        response = score_response("<answer>prose around a valid alternative</answer>", Verifier(), None)
        self.assertTrue(response["correct"])
        self.assertIsNone(choose_wrong_label([response]))

    def test_normalized_wrong_target_remains_eligible(self):
        class Verifier:
            def extract_answer(self, text):
                return "a wrong answer"

            def verify(self, data, text):
                return False

        response = score_response("<answer>prose around a wrong answer</answer>", Verifier(), None)
        self.assertFalse(response["correct"])
        self.assertEqual(choose_wrong_label([response])["answer"], "a wrong answer")
