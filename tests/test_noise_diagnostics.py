import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from src.noise_diagnostics import noise_metrics, snapshot_due, write_snapshot, mathador_true_correct, invalidate_snapshots_after


class NoiseDiagnosticsTest(unittest.TestCase):
    def test_mathador_shadow_checks_original_target_without_eval(self):
        data = SimpleNamespace(metadata={'numbers': [1, 2, 3, 4], 'operators': ['+', '-', '*', '/'], 'result': 24})
        verifier = SimpleNamespace(extract_answer=lambda text: text)
        self.assertTrue(mathador_true_correct(data, '1*2*3*4', verifier))
        self.assertFalse(mathador_true_correct(data, '1+2+3+4', verifier))
        self.assertFalse(mathador_true_correct(data, '1**2**3**4', verifier))
        self.assertFalse(mathador_true_correct(data, '__import__("os")', verifier))

    def test_actual_advantages_and_frozen_noisy_denominators(self):
        # Two noisy groups, one clean group. Only one noisy group is active;
        # padding values must not create activity. A zero-advantage member of
        # a mixed group demonstrates why rollout activity is counted directly.
        batch = SimpleNamespace(non_tensor_batch={
            'uid': ['a', 'a', 'b', 'b', 'c', 'c'],
            'diagnostic_is_noisy': [1, 1, 1, 1, 0, 0],
            'diagnostic_assigned_match': [1, 0, 1, 1, 0, 0],
            'diagnostic_true_correct': [0, 1, 0, 0, 1, 1],
            'diagnostic_true_verified': [1, 1, 1, 0, 1, 1],
            'diagnostic_true_error': [0, 0, 0, 1, 0, 0],
        }, batch={'advantages': torch.tensor([[1., 0], [0, 9], [0, 0], [0, 0], [1, 0], [-1, 0]]),
                  'response_mask': torch.tensor([[1, 0]] * 6)})
        result = noise_metrics(batch)
        self.assertEqual(result['noise/active_rate'], .5)
        self.assertEqual(result['noise/active_rollout_rate'], .25)
        self.assertEqual(result['noise/noise_learnability'], .75)
        self.assertEqual(result['noise/true_learnability'], .25)
        self.assertEqual(result['noise/true_verifier_coverage'], .75)
        self.assertEqual(result['noise/true_verifier_errors'], 1)

    def test_no_noisy_examples_has_counts_without_fabricated_rates(self):
        b = SimpleNamespace(non_tensor_batch={
            'uid': ['a'], 'diagnostic_is_noisy': [False],
            'diagnostic_assigned_match': [False], 'diagnostic_true_correct': [True],
            'diagnostic_true_verified': [True], 'diagnostic_true_error': [False]},
            batch={'advantages': torch.zeros(1, 2), 'response_mask': torch.ones(1, 2)})
        result = noise_metrics(b)
        self.assertEqual(result['noise/groups'], 0)
        self.assertNotIn('noise/active_rate', result)
        self.assertNotIn('noise/true_learnability', result)

    def test_generation_denominator_keeps_filtered_out_groups(self):
        b = SimpleNamespace(non_tensor_batch={
            'uid': ['a'], 'diagnostic_rollout_id': ['a:0'], 'diagnostic_is_noisy': [True]},
            batch={'advantages': torch.ones(1, 1), 'response_mask': torch.ones(1, 1)})
        generated = [dict(uid=uid, rollout_id=uid+':0', is_noisy=True, assigned_match=True,
                          true_correct=False, true_verified=True, true_error=False, assigned_error=False)
                     for uid in ['a', 'filtered']]
        metrics = noise_metrics(b, generated)
        self.assertEqual(metrics['noise/groups'], 2)
        self.assertEqual(metrics['noise/active_rate'], .5)
        self.assertEqual(metrics['noise/noise_learnability'], 1.)

    def test_sparse_schedule_and_final_evaluation(self):
        self.assertEqual([i for i in range(1, 46) if snapshot_due(i, 20, 45)], [19, 39, 44])
        self.assertFalse(snapshot_due(19, 0, 45))
        self.assertFalse(snapshot_due(19, 20, 45, False))
        self.assertFalse(snapshot_due(0, 20, 45))

    def test_real_trainer_restore_invalidates_abandoned_snapshot_trajectory(self):
        from omegaconf import OmegaConf
        from src.dapo_ray_trainer import RayDAPOTrainer
        with tempfile.TemporaryDirectory() as folder:
            rows = [{'rollout_id': 'g:0', 'group_id': 'g'}]
            write_snapshot(folder, 9, rows, {})
            future = write_snapshot(folder, 19, rows, {})

            class RestoredTrainer(RayDAPOTrainer):
                def __init__(self):
                    self.total_training_steps = 10
                    self.config = OmegaConf.create({'trainer': {
                        'default_local_dir': str(Path(folder) / 'checkpoints'),
                        'rollout_snapshot_dir': folder, 'project_name': 'test',
                        'experiment_name': 'snapshot-rollback', 'logger': ['console']}})

                def _load_checkpoint(self):
                    self.global_steps = 10

                def _progress(self, *args, **kwargs):
                    pass

            RestoredTrainer().fit()
            self.assertTrue((Path(folder) / 'step_000009/CURRENT.json').exists())
            self.assertFalse((Path(folder) / 'step_000019/CURRENT.json').exists())
            self.assertTrue(future.exists())

    def test_atomic_compressed_snapshot_is_complete_and_replay_safe(self):
        rows = [{'rollout_id': 'a:0', 'group_id': 'a', 'response': '☃', 'advantages': [1.]}]
        with tempfile.TemporaryDirectory() as folder:
            first = write_snapshot(folder, 19, rows, {'protocol_id': 'test', 'next_evaluation_step': 20})
            self.assertEqual(first, write_snapshot(folder, 19, rows, {'protocol_id': 'test', 'next_evaluation_step': 20}))
            with gzip.open(first, 'rt') as stream:
                saved = [json.loads(line) for line in stream]
            self.assertTrue(saved[0]['metadata']['complete'])
            self.assertEqual(saved[0]['metadata']['generating_policy_completed_updates'], 18)
            self.assertEqual(saved[1:], rows)
            self.assertEqual(len(list(Path(folder).rglob('*.jsonl.gz'))), 1)
            self.assertFalse(list(Path(folder).rglob('.writing-*')))
            self.assertTrue((first.parent / 'CURRENT.json').exists())
            invalidate_snapshots_after(folder, 10)
            self.assertFalse((first.parent / 'CURRENT.json').exists())
            self.assertTrue(first.exists())
            write_snapshot(folder, 19, rows, {'attempt': 2})
            self.assertTrue((first.parent / 'CURRENT.json').exists())


if __name__ == '__main__':
    unittest.main()
