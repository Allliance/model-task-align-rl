import contextlib
import io
from pathlib import Path
import random
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'SynLogic'))
from games.tasks.number_wall.scripts.number_wall_verifier import NumberWallVerifier
from src.task_deadline import task_deadline


class NumberWallParserTest(unittest.TestCase):
    @unittest.skipIf(sys.version_info < (3, 11), 'Possessive quantifiers require Python 3.11+')
    def test_accepted_grid_language_is_preserved(self):
        verifier = NumberWallVerifier()
        pattern = next(value for value in verifier.extract_answer.__code__.co_consts
                       if isinstance(value, str) and value.startswith(r'\[\s*\['))
        optimized = pattern.replace(r'\s*', r'\s*+').replace('[0-9]+', '[0-9]++')
        rng = random.Random(42)
        for _ in range(200):
            grid = [[rng.choice([1, 2, 12, 'X', 'A', '3']) for _ in range(4)] for _ in range(4)]
            text = 'reasoning\n```python\n' + repr(grid) + '\n```'
            self.assertEqual(re.findall(pattern, text), re.findall(optimized, text))

    def test_malformed_long_grid_does_not_hang(self):
        verifier = NumberWallVerifier()
        with task_deadline(.2), contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(verifier.extract_answer('[[' + ' ' * 10000 + '1' * 1000))

    def test_connected_island_traversal_terminates(self):
        with task_deadline(.2):
            self.assertTrue(NumberWallVerifier()._check_diagonal_borders([[4, 'X'], ['X', 'X']]))


if __name__ == '__main__':
    unittest.main()
