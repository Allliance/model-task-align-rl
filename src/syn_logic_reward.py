import re
import sys
sys.path = sys.path + ["SynLogic"]
import random
import contextlib
import copy
import os
from task2verifier import verifier_classes

from collections import defaultdict
import torch
from verl import DataProto
from .fixed_labels import answer_key, extract_answer


import json


from .task_deadline import task_deadline as verifier_deadline, TaskDeadlineExceeded


class Data:
    """
    Data class for game/corpus
    @param question: question of the game/corpus
    @param answer: answer of the game/corpus
    @param difficulty: difficulty of the game/corpus, from 1 to 10
    """
    def __init__(self, question: str, answer: str, difficulty: int = 1, metadata: dict = None, **kwargs):
        self.question = question
        self.answer = answer
        self.difficulty = difficulty
        self.metadata = metadata
        self.gpt_response = ""
        
    def to_json(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
            "gpt_response": self.gpt_response
        }
    
    def to_json_str(self):
        return json.dumps(self.to_json(), ensure_ascii=False)
    
    @classmethod
    def from_json_str(cls, json_str):
        json_data = json.loads(json_str)
        return cls(**json_data)
    
    @classmethod
    def from_json_dict(cls, json_dict):
        instance = cls(**json_dict)
        if 'gpt_response' in json_dict:
            instance.gpt_response = json_dict['gpt_response']
        return instance
    
    @classmethod
    def from_jsonl_file(cls, file_path):
        data_list = []
        with open(file_path, "r") as f:
            for line in f:
                json_data = json.loads(line)
                instance = cls(**json_data)
                if 'gpt_response' in json_data:
                    instance.gpt_response = json_data['gpt_response']
                data_list.append(instance)
        return data_list


def _extract_answer(solution_str):
    pattern = r'<answer>(.*?)</answer>'
    answers = re.findall(pattern, solution_str)
    if len(answers) == 0:
        raise ValueError("could not extract answer")
    return answers[0]


def compute_score_wo_think_format(data_source, solution_str, ground_truth, extra_info=None):
    # Native-reference scorer shared by validation and ordinary training rows.
    if data_source == "deepscaler":
        return compute_math_score_wo_think_format(data_source, solution_str, ground_truth, extra_info)

    verifier = verifier_classes[data_source]()
    game_data = extra_info["game_data"]
    try:
        solution = _extract_answer(solution_str)
    except ValueError as e:
        score = 0.0
    else:
        score = float(verifier.verify(game_data, solution))
    return score


def compute_fixed_label_score_wo_think_format(data_source, solution_str, ground_truth, extra_info=None):
    """Historical, explicit, training-only ``fixed_wrong_answer`` target mode.

    A frozen assigned target changes the answer key, never the reward sign; some
    native constraint checkers ignore Data.answer altogether. Parser, deadline
    and failure flags are those recorded for this mode. Validation never routes here.
    """
    extra_info.pop("verifier_timed_out", None)
    extra_info.pop("verifier_unverified", None)
    verifier = verifier_classes[data_source]()
    solution = extract_answer(solution_str)
    if solution is None:
        return 0.0
    try:
        with verifier_deadline(), open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            score = float(answer_key(solution, verifier) == extra_info["assigned_answer_key"])
    except TaskDeadlineExceeded:
        extra_info["verifier_timed_out"] = True
        score = 0.0
    except Exception:
        extra_info["verifier_unverified"] = True
        score = 0.0
    return score


def _uses_fixed_label_target(data_source, extra_info):
    return (data_source != "deepscaler" and extra_info is not None
            and extra_info.get("target_mode") == "fixed_wrong_answer")


def compute_training_score_wo_think_format(data_source, solution_str, ground_truth, extra_info=None):
    """Training accuracy: the native scorer unless the row explicitly opts into fixed labels."""
    if _uses_fixed_label_target(data_source, extra_info):
        return compute_fixed_label_score_wo_think_format(data_source, solution_str, ground_truth, extra_info)
    return compute_score_wo_think_format(data_source, solution_str, ground_truth, extra_info)


def _fixed_label_scoring_failed(extra_info):
    # Only the fixed-label path sets these flags; native failures propagate unchanged.
    return bool(extra_info) and bool(extra_info.get("verifier_timed_out") or extra_info.get("verifier_unverified"))


def scoring_versions():
    """Versioned native scorer repairs, for new provenance and snapshots only."""
    return {task: cls.SCORING_VERSION for task, cls in sorted(verifier_classes.items())
            if getattr(cls, "SCORING_VERSION", None)}


def compute_math_score_wo_think_format(data_source, solution_str, ground_truth, extra_info=None):
    try:
        prediction = _extract_answer(solution_str)
    except ValueError as e:
        score = 0
    else:
        score = float(prediction == ground_truth)
    return score
    
    
def compute_think_format_score(data_source, solution_str, ground_truth, extra_info=None):
    def check_format(s):
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>$'
        if re.match(pattern, s):
            return 1.0
        else:
            return 0.0
    return check_format(solution_str.strip())


def compute_score_w_think_format_score(data_source, solution_str, ground_truth, extra_info=None):
    format_reward = compute_think_format_score(data_source, solution_str, ground_truth, extra_info)
    accuracy_reward = compute_training_score_wo_think_format(data_source, solution_str, ground_truth, extra_info)
    
    return {
        "format_reward": format_reward,
        "accuracy_reward": accuracy_reward,
        "final_reward": 1 if format_reward * accuracy_reward > 0 else 0
    }


def compute_math_score_w_think_format_score(data_source, solution_str, ground_truth, extra_info=None):
    format_reward = compute_think_format_score(data_source, solution_str, ground_truth, extra_info)
    accuracy_reward = compute_math_score_wo_think_format(data_source, solution_str, ground_truth, extra_info)
    return {
        "format_reward": format_reward,
        "accuracy_reward": accuracy_reward,
        "final_reward": 1 if format_reward * accuracy_reward > 0 else 0
    }


def random_score(data_source, solution_str, ground_truth, extra_info=None):
    return 1.0 if random.uniform(0, 1) >= 0.5 else 0.0


def random_score_with_format(data_source, solution_str, ground_truth, extra_info=None):
    format_reward = compute_think_format_score(data_source, solution_str, ground_truth, extra_info)
    accuracy_reward = random_score(data_source, solution_str, ground_truth, extra_info)
    return {
        "format_reward": format_reward,
        "accuracy_reward": accuracy_reward,
        "final_reward": 1 if format_reward * accuracy_reward > 0 else 0
    }

def incorrect_format_score(data_source, solution_str, ground_truth, extra_info=None):
    format_reward = compute_think_format_score(data_source, solution_str, ground_truth, extra_info)
    accuracy_reward = compute_training_score_wo_think_format(data_source, solution_str, ground_truth, extra_info)
    accuracy_reward = 1.0 - accuracy_reward
    return {
        "format_reward": format_reward,
        "accuracy_reward": accuracy_reward,
        # An unverified fixed-label verdict is unknown, never rewarded as wrong.
        "final_reward": 1 if format_reward * accuracy_reward > 0 and not
        _fixed_label_scoring_failed(extra_info) else 0
    }


def math_incorrect_format_score(data_source, solution_str, ground_truth, extra_info=None):
    format_reward = compute_think_format_score(data_source, solution_str, ground_truth, extra_info)
    accuracy_reward = compute_math_score_wo_think_format(data_source, solution_str, ground_truth, extra_info)
    accuracy_reward = 1.0 - accuracy_reward
    return {
        "format_reward": format_reward,
        "accuracy_reward": accuracy_reward,
        "final_reward": 1 if format_reward * accuracy_reward > 0 else 0
    }


def math_incorrect_format_score(data_source, solution_str, ground_truth, extra_info=None):
    format_reward = compute_think_format_score(data_source, solution_str, ground_truth, extra_info)
    accuracy_reward = compute_math_score_wo_think_format(data_source, solution_str, ground_truth, extra_info)
    return {
        "format_reward": format_reward,
        "accuracy_reward": 1.0 - accuracy_reward,
        "final_reward": 1 if format_reward * accuracy_reward > 0 else 0
    }

# reward for rule-base RL training
_SynLogic_Scores = {
    "acc_without_think_format": compute_training_score_wo_think_format,
    "think_format": compute_think_format_score,
    "acc_with_think_format": compute_score_w_think_format_score,
    "random_score_with_format": random_score_with_format,
    "incorrect_format_score": incorrect_format_score,
    "math_acc_with_think_format": compute_math_score_w_think_format_score,
    "math_incorrect_format_score": math_incorrect_format_score,
    "math_think_format": compute_think_format_score,
    "math_random_score_with_format": random_score_with_format,
    "math_incorrect_format_score": math_incorrect_format_score,
    "extropy_min": None,
}
# score for validation
_SynLogic_Scores["validation_acc"] = compute_score_wo_think_format
_SynLogic_Scores["math_validation_acc"] = compute_math_score_wo_think_format


def _preprocess_game_data(game_data_str):
    # conver str to dict
    return Data.from_json_str(game_data_str)
        

def _true_reward_diagnostic(data_source, response_str, extra_info, score_type, score,
                            scored_info, existing=None):
    """Return clean-reference accuracy and coverage; never an optimization reward."""
    import copy
    import math

    existing = existing or {}
    if existing.get("verified") or existing.get("error"):
        verified = (bool(existing.get("verified")) and not bool(existing.get("error"))
                    and existing.get("correct") is not None)
        return (float(bool(existing["correct"])) if verified else None,
                verified, bool(existing.get("error")))
    if not extra_info:
        return None, False, False
    original = extra_info.get("original_game_data_str")
    noisy = bool(extra_info.get("diagnostic_is_noisy") or
                 extra_info.get("target_mode") == "fixed_wrong_answer")
    noise_flag = extra_info.get("diagnostic_is_noisy")
    explicitly_clean = (extra_info.get("target_mode") == "original_verifier"
                        and noise_flag is not None and not bool(noise_flag))
    if original is None and (noisy or not explicitly_clean):
        return None, False, False
    try:
        shadow = copy.deepcopy(extra_info)
        if original is not None:
            shadow["game_data"] = _preprocess_game_data(original)
            shadow["game_data_str"] = original
        if "game_data" not in shadow:
            return None, False, False
        # Reuse an already-scored clean accuracy result, including failure flags.
        same_reference = original is None or original == extra_info.get("game_data_str")
        if explicitly_clean and not noisy and same_reference and score_type == "acc_without_think_format":
            value, checked = float(score), scored_info
        else:
            shadow.pop("verifier_timed_out", None)
            shadow.pop("verifier_unverified", None)
            shadow["target_mode"] = "original_verifier"
            shadow["diagnostic_is_noisy"] = False
            shadow.pop("assigned_answer_key", None)
            shadow.pop("diagnostic_assigned_answer_key", None)
            value = float(compute_score_wo_think_format(
                data_source, response_str, shadow["game_data"].answer, shadow))
            checked = shadow
        error = bool(checked.get("verifier_timed_out") or checked.get("verifier_unverified"))
        if error or not math.isfinite(value):
            return None, False, True
        return float(value > 0), True, False
    except Exception:
        # A failed diagnostic is unknown, not an incorrect response.
        return None, False, True


def _clean_reference_verdict(data_source, response_str, extra_info, original):
    """Read-only native verdict against the frozen original reference.

    Uses the unchanged native scorer. The deadline bounds only this diagnostic;
    a timeout or exception is reported as an error, never as an incorrect answer.
    """
    shadow = copy.deepcopy(extra_info)
    for key in ("verifier_timed_out", "verifier_unverified",
                "assigned_answer_key", "diagnostic_assigned_answer_key"):
        shadow.pop(key, None)
    shadow.update(target_mode="original_verifier", diagnostic_is_noisy=False,
                  game_data=_preprocess_game_data(original), game_data_str=original)
    try:
        with verifier_deadline(), open(os.devnull, "w") as sink, \
                contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            correct = compute_score_wo_think_format(
                data_source, response_str, shadow["game_data"].answer, shadow) > 0
    except (Exception, TaskDeadlineExceeded):
        return None, False, True
    return bool(correct), True, False


def _frozen_target_diagnostics(data_source, response_str, ground_truth, extra_info):
    """Per-response noisy-row diagnostics from frozen assigned and original targets."""
    target_mode = extra_info.get("target_mode")
    frozen_noisy = bool(extra_info.get("diagnostic_is_noisy", target_mode == "fixed_wrong_answer"))
    assigned_match, assigned_error = False, False
    true_correct = verified = error = None
    if frozen_noisy or target_mode == "fixed_wrong_answer":
        original = extra_info.get("original_game_data_str")
        frozen_target = extra_info.get("diagnostic_assigned_answer_key", extra_info.get("assigned_answer_key"))
        if original is None or (frozen_noisy and frozen_target is None):
            raise ValueError("Noisy training diagnostics require frozen assigned and original clean targets")
        extracted = extract_answer(response_str)
        if extracted is not None and frozen_noisy:
            try:
                with verifier_deadline():
                    assigned_match = answer_key(extracted, verifier_classes[data_source]()) == frozen_target
            except (Exception, TaskDeadlineExceeded):
                assigned_error = True
        true_correct, verified, error = _clean_reference_verdict(data_source, response_str, extra_info, original)
    return {
        "diagnostic_effective_target": extra_info.get("assigned_answer_key") if target_mode == "fixed_wrong_answer" else ground_truth,
        "diagnostic_effective_target_mode": extra_info.get("target_mode", "original_verifier"),
        "diagnostic_is_noisy": frozen_noisy,
        "diagnostic_assigned_match": assigned_match,
        "diagnostic_assigned_error": assigned_error,
        "diagnostic_true_correct": true_correct,
        "diagnostic_true_verified": verified,
        "diagnostic_true_error": error,
    }


class SynLogicRewardManager:
    """The SynLogic reward manager."""
    def __init__(
        self,
        tokenizer,
        num_examine,
        syn_logic_score_type="acc_with_think_format",
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.score_type = syn_logic_score_type
        self.compute_score = _SynLogic_Scores[syn_logic_score_type]
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        if self.score_type == "extropy_min":
            return self.entropy_min(data, return_dict)

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            if data_source.startswith("val/"):
                data_source = data_source[4:]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            
            if extra_info is not None and "game_data_str" in extra_info:
                extra_info["game_data"] = _preprocess_game_data(extra_info["game_data_str"])
            
            scored_info = copy.deepcopy(extra_info)
            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=scored_info,
            )

            score: float
            if isinstance(result, dict):
                score = result["final_reward"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                
            # Training-only, read-only diagnostics; validation protocol is unchanged.
            # One canonical native clean-reference verdict feeds both the frozen
            # noisy-row fields and the critic/true_reward fields.
            if (not data_item.non_tensor_batch[self.reward_fn_key].startswith("val/")
                    and self.score_type not in ("validation_acc", "math_validation_acc")):
                existing = None
                if extra_info is not None and "game_data_str" in extra_info:
                    frozen = _frozen_target_diagnostics(data_source, response_str, ground_truth, extra_info)
                    for key, value in frozen.items():
                        reward_extra_info[key].append(value)
                    if frozen["diagnostic_true_verified"] is not None:
                        existing = {"correct": frozen["diagnostic_true_correct"],
                                    "verified": frozen["diagnostic_true_verified"],
                                    "error": frozen["diagnostic_true_error"]}
                true_reward, true_verified, true_error = _true_reward_diagnostic(
                    data_source, response_str, extra_info, self.score_type, score,
                    scored_info, existing)
                for key, value in [("diagnostic_true_reward", true_reward),
                                   ("diagnostic_true_reward_verified", true_verified),
                                   ("diagnostic_true_reward_error", true_error)]:
                    reward_extra_info[key].append(value)

            reward = score

            if self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    def entropy_min(self, data: DataProto, return_dict: bool = False):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
    
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            joint_prob = data_item.batch['old_log_probs'][:valid_response_length].mean()
            reward_tensor[i, valid_response_length - 1] = joint_prob

        return reward_tensor
