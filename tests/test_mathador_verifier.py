"""Mathador must check its target in the native scorer used by every caller."""
import contextlib
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "SynLogic"))

from base.data import Data
from games.tasks.game_of_24.scripts.game_of_24_verifier import GameOf24Verifier


class MathadorVerifierTest(unittest.TestCase):
    def score(self, expression, numbers=(1, 2, 3, 4), target=24,
              operators=("+", "-", "*", "/")):
        data = Data("unchanged question", "unused stored expression", metadata={
            "numbers": list(numbers), "operators": list(operators), "result": target})
        with contextlib.redirect_stdout(io.StringIO()):
            result = GameOf24Verifier().verify(data, f"```python\n{expression}\n```")
        self.assertEqual(data.metadata["result"], target)
        return result

    def test_correct_target_and_multiple_valid_expressions(self):
        self.assertTrue(self.score("1*2*3*4"))
        self.assertTrue(self.score("(1+3)*(2+4)"))

    def test_legal_expression_with_wrong_value_is_rejected(self):
        self.assertFalse(self.score("1+2+3+4"))

    def test_reference_target_controls_acceptance(self):
        self.assertTrue(self.score("1*2*3*4", target=24.0))
        self.assertTrue(self.score("1+2+3+4", target=10))
        self.assertFalse(self.score("1*2*3*4", target=10))
        self.assertTrue(self.score("1*4+5+6+8", numbers=(1, 4, 5, 6, 8), target=23))

    def test_fractional_intermediates_and_parentheses(self):
        self.assertTrue(self.score("8/(3-8/3)", numbers=(3, 3, 8, 8)))

    def test_number_inventory_is_preserved(self):
        self.assertTrue(self.score("6+6+6+6", numbers=(6, 6, 6, 6)))
        self.assertFalse(self.score("6+6+6+6"))
        self.assertFalse(self.score("2*3*4"))
        self.assertFalse(self.score("1*2*3*4*1"))

    def test_allowed_operators_are_preserved(self):
        self.assertFalse(self.score("(1+3)*(2+4)", operators=("*",)))
        self.assertTrue(self.score("1*2*3*4", operators=("*",)))

    def test_bad_arithmetic_is_rejected(self):
        self.assertFalse(self.score("1*2*3*4/0", numbers=(0, 1, 2, 3, 4)))
        self.assertFalse(self.score("1+2+3+4+"))

    def test_existing_code_block_extraction_is_preserved(self):
        verifier = GameOf24Verifier()
        self.assertEqual(verifier.extract_answer("1*2*3*4"), "")
        self.assertEqual(verifier.extract_answer(
            "```python\n1+2+3+4\n``` then ```python\n1*2*3*4\n```"), "1*2*3*4")


if __name__ == "__main__":
    unittest.main()
