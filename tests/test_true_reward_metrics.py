"""Clean-label training diagnostics must not change optimization rewards."""
import copy
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_true_reward_metrics
from src import syn_logic_reward as reward


class LabelVerifier:
    def verify(self, data, answer):
        return answer == data.answer

    def extract_answer(self, text):
        return text.strip()


class Tokenizer:
    eos_token = "<eos>"

    def decode(self, ids, **kwargs):
        return {10: "<answer>right</answer>", 11: "<answer>wrong</answer>",
                12: "unparseable"}.get(int(ids[-1]), "prompt")


def make_batch(answers, noisy=False, source="test"):
    clean = reward.Data("question", "right").to_json_str()
    assigned = reward.Data("question", "wrong" if noisy else "right").to_json_str()
    extras = [dict(game_data_str=assigned, original_game_data_str=clean,
                   target_mode="fixed_wrong_answer" if noisy else "original_verifier",
                   assigned_answer_key="wrong" if noisy else "right",
                   diagnostic_is_noisy=noisy) for _ in answers]
    size = len(answers)
    return DataProto(batch=TensorDict({
        "prompts": torch.ones((size, 1), dtype=torch.long),
        "responses": torch.tensor(answers).reshape(size, 1),
        "attention_mask": torch.ones((size, 2), dtype=torch.long),
    }, batch_size=[size]), non_tensor_batch={
        "data_source": np.array([source] * size, dtype=object),
        "extra_info": np.array(extras, dtype=object),
        "reward_model": np.array([{"ground_truth": "wrong" if noisy else "right"}
                                  for _ in answers], dtype=object)})


def manager():
    return reward.SynLogicRewardManager(Tokenizer(), 0, "acc_without_think_format",
        max_resp_len=10, overlong_buffer_cfg=SimpleNamespace(enable=False))


class TrueRewardTest(unittest.TestCase):
    def setUp(self):
        self.verifiers = patch.dict(reward.verifier_classes, {"test": LabelVerifier})
        self.verifiers.start()
        self.addCleanup(self.verifiers.stop)

    def test_noisy_labels_reverse_assigned_reward_and_diagnostics(self):
        batch = make_batch([10, 11], noisy=True)
        result = manager()(batch, return_dict=True)
        self.assertEqual(result["reward_tensor"].sum(-1).tolist(), [0., 1.])
        info = result["reward_extra_info"]
        self.assertEqual(info["diagnostic_true_reward"], [1., 0.])
        self.assertEqual(info["diagnostic_true_reward_verified"], [True, True])
        # The frozen wrong labels remain the optimization targets.
        for extra in batch.non_tensor_batch["extra_info"]:
            self.assertEqual(extra["target_mode"], "fixed_wrong_answer")
            self.assertEqual(extra["assigned_answer_key"], "wrong")
            self.assertEqual(extra["game_data"].answer, "wrong")

    def test_clean_and_unparseable_responses(self):
        result = manager()(make_batch([10, 11, 12]), return_dict=True)
        self.assertEqual(result["reward_extra_info"]["diagnostic_true_reward"], [1., 0., 0.])
        self.assertEqual(result["reward_tensor"].sum(-1).tolist(), [1., 0., 0.])

    def test_existing_noisy_verdict_is_reused_without_rescoring(self):
        with patch.object(reward, "compute_score_wo_think_format", side_effect=AssertionError):
            value = reward._true_reward_diagnostic("test", "text", {}, "acc_without_think_format",
                1., {}, {"correct": False, "verified": True, "error": False})
        self.assertEqual(value, (0., True, False))

    def test_missing_original_cannot_fall_back_to_wrong_label(self):
        extra = {"game_data": reward.Data("question", "wrong"),
                 "target_mode": "fixed_wrong_answer"}
        self.assertEqual(reward._true_reward_diagnostic("test", "<answer>wrong</answer>",
            extra, "acc_without_think_format", 1., {}), (None, False, False))

    def test_nonstandard_target_mode_cannot_reuse_assigned_score(self):
        clean = reward.Data("question", "right").to_json_str()
        extra = {"game_data": reward.Data.from_json_str(clean), "game_data_str": clean,
                 "original_game_data_str": clean, "target_mode": "other_noise_mode"}
        self.assertEqual(reward._true_reward_diagnostic("test", "<answer>wrong</answer>",
            extra, "acc_without_think_format", 1., {}), (0., True, False))

    @unittest.skipUnless(hasattr(reward, "mathador_true_correct"), "Legacy diagnostic branch only")
    def test_manager_reuses_existing_native_diagnostic(self):
        with patch.dict(reward.verifier_classes, {"mathador": LabelVerifier}), \
             patch.object(reward, "mathador_true_correct", return_value=True) as native:
            # Legacy original-reference verdict deliberately differs from a second
            # generic label check, proving the new manager path reuses that verdict.
            result = manager()(make_batch([11], noisy=True, source="mathador"), return_dict=True)
        info = result["reward_extra_info"]
        self.assertEqual(info["diagnostic_true_correct"], [True])
        self.assertEqual(info["diagnostic_true_verified"], [True])
        self.assertEqual(info["diagnostic_true_reward"], [1.])
        self.assertEqual(native.call_count, 1)

    def test_unmarked_reference_is_unknown(self):
        extra = {"game_data": reward.Data("question", "wrong")}
        self.assertEqual(reward._true_reward_diagnostic("test", "<answer>wrong</answer>",
            extra, "acc_without_think_format", 1., {}), (None, False, False))

    def test_mixed_clean_and_noisy_batch(self):
        clean, noisy = make_batch([10]), make_batch([11], noisy=True)
        batch = DataProto.concat([clean, noisy])
        result = manager()(batch, return_dict=True)
        self.assertEqual(result["reward_tensor"].sum(-1).tolist(), [1., 1.])
        info = result["reward_extra_info"]
        self.assertEqual(info["diagnostic_true_reward"], [1., 0.])
        self.assertEqual(info["diagnostic_true_reward_verified"], [True, True])
        if "diagnostic_true_verified" in info:
            self.assertEqual(info["diagnostic_true_verified"], [None, True])

    def test_clean_failure_flags_flow_through_manager(self):
        mgr = manager()
        def failure(**kwargs):
            kwargs["extra_info"]["verifier_unverified"] = True
            return 0.
        mgr.compute_score = failure
        result = mgr(make_batch([10]), return_dict=True)
        self.assertEqual(result["reward_tensor"].sum(-1).tolist(), [0.])
        self.assertEqual(result["reward_extra_info"]["diagnostic_true_reward"], [None])
        self.assertEqual(result["reward_extra_info"]["diagnostic_true_reward_error"], [True])

    def test_clean_verifier_failures_are_unknown(self):
        extra = {"game_data": reward.Data("question", "right"),
                 "target_mode": "original_verifier", "diagnostic_is_noisy": False}
        self.assertEqual(reward._true_reward_diagnostic("test", "text", extra,
            "acc_without_think_format", 0., {"verifier_unverified": True}), (None, False, True))
        with patch.object(reward, "compute_score_wo_think_format", side_effect=RuntimeError):
            self.assertEqual(reward._true_reward_diagnostic("test", "text", extra,
                "acc_with_think_format", 0., {}), (None, False, True))

    def test_validation_has_no_new_diagnostics(self):
        result = manager()(make_batch([10], source="val/test"), return_dict=True)
        self.assertNotIn("diagnostic_true_reward", result["reward_extra_info"])

    def test_population_statistics_and_coverage(self):
        info = {"diagnostic_true_reward": [1., 0., None, 0.],
                "diagnostic_true_reward_verified": [True, True, False, True],
                "diagnostic_true_reward_error": [False, False, True, True]}
        saved = copy.deepcopy(info)
        metrics = compute_true_reward_metrics(info, 4)
        self.assertEqual(metrics["critic/true_reward/mean"], .5)
        self.assertEqual(metrics["critic/true_reward/std"], .5)
        self.assertEqual(metrics["critic/true_reward/max"], 1.)
        self.assertEqual(metrics["critic/true_reward/min"], 0.)
        self.assertEqual(metrics["critic/true_reward/count"], 2)
        self.assertEqual(metrics["critic/true_reward/coverage"], .5)
        self.assertEqual(metrics["critic/true_reward/verifier_errors"], 2)
        self.assertEqual(info, saved)

    def test_singleton_missing_and_empty_batches(self):
        p = "critic/true_reward/"
        self.assertEqual(compute_true_reward_metrics({"diagnostic_true_reward": [1.],
            "diagnostic_true_reward_verified": [True]}, 1)[p + "std"], 0.)
        for info, total in [({"diagnostic_true_reward": [None]}, 1),
                            ({"diagnostic_true_reward": []}, 0)]:
            metrics = compute_true_reward_metrics(info, total)
            self.assertNotIn(p + "mean", metrics)
            self.assertNotIn(p + "std", metrics)
            if total == 0:self.assertNotIn(p + "coverage", metrics)
        self.assertEqual(compute_true_reward_metrics({}, 3), {})
        with self.assertLogs("verl.trainer.ppo.metric_utils", level="ERROR"):
            bad = compute_true_reward_metrics({"diagnostic_true_reward": [1.]}, 2)
        self.assertEqual(bad, {p + "metadata_errors": 1})

    def test_nonfinite_verdicts_do_not_leak_into_stats(self):
        metrics = compute_true_reward_metrics({"diagnostic_true_reward": [float("nan"), float("inf"), 1.],
            "diagnostic_true_reward_verified": [True] * 3}, 3)
        self.assertEqual(metrics["critic/true_reward/count"], 1)
        self.assertTrue(all(math.isfinite(x) for x in metrics.values()))

    def test_full_training_metric_path_preserves_all_tensors(self):
        batch = make_batch([10, 11], noisy=True)
        result = manager()(batch, return_dict=True)
        batch.non_tensor_batch.update({k: np.array(v) for k,v in result["reward_extra_info"].items()})
        batch.batch["token_level_scores"] = result["reward_tensor"]
        batch.batch["token_level_rewards"] = result["reward_tensor"].clone()
        batch.batch["advantages"] = torch.tensor([[2.], [-2.]])
        batch.batch["returns"] = torch.tensor([[3.], [-3.]])
        before = {k: v.clone() for k,v in batch.batch.items()}
        metrics = compute_data_metrics(batch, use_critic=False)
        for key in ["critic/score/std", "critic/rewards/std", "critic/true_reward/std"]:
            self.assertEqual(metrics[key], .5)
        self.assertEqual(metrics["critic/true_reward/mean"], .5)
        for k,v in before.items():self.assertTrue(torch.equal(batch.batch[k], v), k)


if __name__ == "__main__":
    unittest.main()
