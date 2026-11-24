"""Fixed answer targets and reproducible nested subsets for SynLogic experiments."""

import ast
from collections import Counter, defaultdict
import json
import random
import re
import unicodedata


def extract_answer(response):
    match = re.search(r"<answer>(.*?)</answer>", response, flags=re.DOTALL)
    return match.group(1).strip() if match else None


def answer_key(answer, verifier=None):
    """Normalize presentation, retaining the task verifier's answer extraction."""
    if not isinstance(answer, str) or not answer.strip():
        return None
    if verifier is not None and hasattr(verifier, "extract_answer"):
        try:
            extracted = verifier.extract_answer(answer)
            if extracted is None or not str(extracted).strip():
                return None
            answer = extracted if isinstance(extracted, str) else repr(extracted)
        except (ValueError, TypeError, IndexError, AttributeError):
            return None
    answer = unicodedata.normalize("NFC", answer).strip()
    fence = re.fullmatch(r"```(?:json|python)?\s*\n?(.*?)\n?```", answer, re.DOTALL)
    if fence:
        answer = fence.group(1).strip()
    if len(answer) <= 100_000:
        for parser in (json.loads, ast.literal_eval):
            try:
                value = parser(answer)
                return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            except (ValueError, SyntaxError, TypeError, RecursionError, MemoryError):
                continue
    return re.sub(r"\s+", " ", answer)


def choose_wrong_label(responses, forbidden_keys=()):
    """Use the modal verified-wrong answer; ties use first rollout order."""
    correct_keys = {r["answer_key"] for r in responses if r.get("correct") is True and r.get("answer_key")}
    correct_keys.update(forbidden_keys)
    candidates = [r for r in responses if r.get("correct") is False and r.get("answer_key")
                  and r["answer_key"] not in correct_keys]
    if not candidates:
        return None
    counts = Counter(r["answer_key"] for r in candidates)
    key = max(counts, key=counts.get)
    first = next(r for r in candidates if r["answer_key"] == key)
    return {"answer": first["answer"], "answer_key": key, "count": counts[key]}


def stratified_take(rows, indices, count, seed):
    """Sample exactly count indices, allocating proportional task quotas."""
    if not 0 <= count <= len(indices):
        raise ValueError("Invalid subset size")
    groups = defaultdict(list)
    for index in indices:
        groups[rows[index]["data_source"]].append(index)
    if not indices:
        return []
    rng = random.Random(seed)
    quotas = {task: len(group) * count // len(indices) for task, group in groups.items()}
    remainders = sorted(groups, key=lambda task: (-(len(groups[task]) * count % len(indices)), task))
    for task in remainders[:count - sum(quotas.values())]:
        quotas[task] += 1
    selected = []
    for task in sorted(groups):
        rng.shuffle(groups[task])
        selected.extend(groups[task][:quotas[task]])
    return sorted(selected)


def nested_subsets(rows, seed=42):
    all_indices = list(range(len(rows)))
    clean10 = stratified_take(rows, all_indices, round(len(rows) * 0.10), seed)
    clean1 = stratified_take(rows, clean10, round(len(clean10) * 0.10), seed + 1)
    clean10_set = set(clean10)
    noisy90 = [i for i in all_indices if i not in clean10_set]
    noise_masks = {90: noisy90}
    previous = noisy90
    for rate in (70, 50, 30):
        previous = stratified_take(rows, previous, round(len(rows) * rate / 100), seed + rate)
        noise_masks[rate] = previous
    return {"full-clean": all_indices, "clean10": clean10, "clean1": clean1,
            "flipped90": noisy90}, noise_masks
