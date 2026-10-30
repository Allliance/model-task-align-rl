import unittest

from src.fixed_labels import answer_key, choose_wrong_label, extract_answer, nested_subsets


class FixedLabelsTest(unittest.TestCase):
    def test_multiline_answer(self):
        self.assertEqual(extract_answer('<think>x</think><answer>[[1,\n2]]</answer>'), '[[1,\n2]]')
        self.assertIsNone(extract_answer('no answer'))

    def test_structured_presentation(self):
        self.assertEqual(answer_key('```python\n((1, 2), (3, 4))\n```'), answer_key('[[1,2],[3,4]]'))
        self.assertNotEqual(answer_key('[[1,2],[3,4]]'), answer_key('[[2,1],[3,4]]'))

    def test_wrong_majority_excludes_correct_and_unverified(self):
        rows = ([dict(answer='right', answer_key='right', correct=True)] * 10
                + [dict(answer='wrong', answer_key='wrong', correct=False)] * 3
                + [dict(answer='unknown', answer_key='unknown', correct=None)] * 20
                + [dict(answer='other', answer_key='other', correct=False)])
        self.assertEqual(choose_wrong_label(rows), dict(answer='wrong', answer_key='wrong', count=3))

    def test_tie_uses_rollout_order(self):
        rows = [dict(answer=k, answer_key=k, correct=False) for k in ('b', 'a')]
        self.assertEqual(choose_wrong_label(rows)['answer'], 'b')

    def test_correct_equivalent_keys_are_never_wrong_labels(self):
        rows = [dict(answer='same', answer_key='same', correct=True),
                dict(answer='same', answer_key='same', correct=False),
                dict(answer='wrong', answer_key='wrong', correct=False)]
        self.assertEqual(choose_wrong_label(rows)['answer'], 'wrong')
        self.assertIsNone(choose_wrong_label(rows, forbidden_keys=['wrong']))

    def test_task_extraction_failure_is_not_a_label(self):
        class Verifier:
            def extract_answer(self, text):
                return None
        self.assertIsNone(answer_key('unrelated prose', Verifier()))

    def test_all_correct_and_missing_answers(self):
        self.assertIsNone(choose_wrong_label([dict(correct=True, answer_key='yes')] * 16))
        self.assertIsNone(choose_wrong_label([dict(correct=False, answer_key=None)] * 16))

    def test_nested_stratified_masks(self):
        rows = [dict(data_source=str(i % 27)) for i in range(15837)]
        subsets, masks = nested_subsets(rows)
        self.assertEqual(len(subsets['clean10']), 1584)
        self.assertEqual(len(subsets['clean1']), 158)
        self.assertTrue(set(subsets['clean1']) <= set(subsets['clean10']))
        self.assertFalse(set(subsets['clean10']) & set(subsets['flipped90']))
        self.assertEqual(set(subsets['clean10']) | set(subsets['flipped90']), set(range(len(rows))))
        for high, low in ((90, 70), (70, 50), (50, 30)):
            self.assertTrue(set(masks[low]) <= set(masks[high]))
        self.assertEqual((subsets, masks), nested_subsets(rows))


if __name__ == '__main__':
    unittest.main()
