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

    def test_manager_reuses_one_canonical_clean_verdict(self):
        native = reward.compute_score_wo_think_format
        with patch.object(reward, "compute_score_wo_think_format", side_effect=native) as scorer:
            result = manager()(make_batch([10], noisy=True), return_dict=True)
        info = result["reward_extra_info"]
        self.assertEqual(info["diagnostic_true_correct"], [True])
        self.assertEqual(info["diagnostic_true_verified"], [True])
        self.assertEqual(info["diagnostic_true_reward"], [1.])
        self.assertEqual(info["diagnostic_true_reward_verified"], [True])
        # One clean-reference verdict; the fixed target never calls the native scorer.
        self.assertEqual(scorer.call_count, 1)

    def test_clean_verdict_errors_are_unknown_in_both_metric_families(self):
        with patch.object(LabelVerifier, "verify", side_effect=KeyError("malformed")):
            result = manager()(make_batch([10], noisy=True), return_dict=True)
        info = result["reward_extra_info"]
        self.assertEqual(info["diagnostic_true_correct"], [None])
        self.assertEqual(info["diagnostic_true_error"], [True])
        self.assertEqual(info["diagnostic_true_reward"], [None])
        self.assertEqual(info["diagnostic_true_reward_error"], [True])
        self.assertEqual(result["reward_tensor"].sum(-1).tolist(), [0.])

    def test_mathador_diagnostic_uses_fixed_native_verifier(self):
        # setUp only adds the "test" task; "mathador" is the real task2verifier entry.
        self.assertEqual(reward.verifier_classes["mathador"].__name__, "GameOf24Verifier")
        clean = reward.Data("question", "unused stored expression", metadata={
            "numbers": [1, 2, 3, 4], "operators": ["+", "-", "*", "/"], "result": 24}).to_json_str()
        extra = dict(game_data_str=clean, original_game_data_str=clean, target_mode="fixed_wrong_answer",
                     assigned_answer_key="1+2+3+4", diagnostic_is_noisy=True)
        verdicts = {expression: reward._frozen_target_diagnostics(
                        "mathador", f"<answer>```python {expression}```</answer>", "", extra)
                    for expression in ("1*2*3*4", "1+2+3+4")}
        self.assertEqual(verdicts["1*2*3*4"]["diagnostic_true_correct"], True)
        self.assertEqual(verdicts["1+2+3+4"]["diagnostic_true_correct"], False)
        self.assertTrue(all(v["diagnostic_true_verified"] for v in verdicts.values()))
        self.assertEqual(reward.scoring_versions().get("mathador"), "mathador-target-v2")

    def test_validation_ignores_training_target_metadata(self):
        # Validation always scores against the native reference, even if a row
        # carries fixed-label training metadata.
        batch = make_batch([10, 11], noisy=True, source="val/test")
        for extra in batch.non_tensor_batch["extra_info"]:
            extra["game_data_str"] = extra["original_game_data_str"]
        validation = reward.SynLogicRewardManager(Tokenizer(), 0, "validation_acc",
            max_resp_len=10, overlong_buffer_cfg=SimpleNamespace(enable=False))
        result = validation(batch, return_dict=True)
        self.assertEqual(result["reward_tensor"].sum(-1).tolist(), [1., 0.])
        self.assertIs(reward._SynLogic_Scores["validation_acc"], reward.compute_score_wo_think_format)

    def test_native_parser_and_failure_semantics_are_unchanged(self):
        extra = {"game_data": reward.Data("question", "right")}
        score = lambda text: reward.compute_score_wo_think_format("test", text, "", dict(extra))
        self.assertEqual(score("<answer>right</answer>"), 1.)
        # First single-line match only, no stripping, no multi-line answers.
        self.assertEqual(score("<answer>right</answer><answer>wrong</answer>"), 1.)
        self.assertEqual(score("<answer>wrong</answer><answer>right</answer>"), 0.)
        self.assertEqual(score("<answer> right </answer>"), 0.)
        self.assertEqual(score("<answer>\nright</answer>"), 0.)
        # Native verifier exceptions still propagate; no deadline or failure flags.
        with patch.object(LabelVerifier, "verify", side_effect=KeyError("malformed")):
            with self.assertRaises(KeyError):
                score("<answer>right</answer>")
        with patch.object(reward, "verifier_deadline", side_effect=AssertionError):
            self.assertEqual(score("<answer>right</answer>"), 1.)
        self.assertEqual(reward.compute_think_format_score(
            "test", "<think>a\nb</think><answer>right</answer>", ""), 0.)
        self.assertEqual(reward.compute_think_format_score(
            "test", "<think>ab</think> <answer>right</answer>", ""), 1.)

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
