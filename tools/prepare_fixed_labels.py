#!/usr/bin/env python3
"""Generate frozen wrong targets, then build the shared eight-condition dataset."""

import argparse
from collections import Counter
import contextlib
import concurrent.futures
import gzip
import hashlib
import json
import os
import multiprocessing
from pathlib import Path
import signal
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "SynLogic")]
from src.fixed_labels import answer_key, choose_wrong_label, extract_answer, nested_subsets
from src.task_deadline import task_deadline, TaskDeadlineExceeded


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def game_data(row):
    extra = row["extra_info"]
    if extra.get("game_data_str"):
        return json.loads(extra["game_data_str"])
    return {"question": extra["original_question"], "answer": extra["original_answer"],
            "difficulty": 1, "metadata": {}}


def load_verifiers():
    from task2verifier import verifier_classes
    from base.data import Data
    return {name: cls() for name, cls in verifier_classes.items()}, Data


def verify_answer(verifier, data, answer):
    if answer is None:
        return False
    try:
        with task_deadline(), open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            return bool(verifier.verify(data, answer))
    except TaskDeadlineExceeded:
        return None
    except Exception:
        return None


def score_response(text, verifier, data):
    started = time.monotonic()
    payload = extract_answer(text) or text
    try:
        with task_deadline(), open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            key = answer_key(payload, verifier)
    except (TaskDeadlineExceeded, Exception):
        return {"answer": None, "answer_key": None, "correct": None,
                "verification_seconds": time.monotonic() - started}
    correct = verify_answer(verifier, data, payload)
    if key and key != payload:
        # The assigned target is the normalized key. A presentation-sensitive
        # checker can reject the raw text yet accept that normalized answer.
        key_correct = verify_answer(verifier, data, key)
        if key_correct is True:
            correct = True
        elif correct is False and key_correct is None:
            correct = None
    return {"answer": key, "answer_key": key, "correct": correct,
            "verification_seconds": time.monotonic() - started}


_VERIFIERS = None
_DATA = None


def initialize_revalidation():
    global _VERIFIERS, _DATA
    _VERIFIERS, _DATA = load_verifiers()


def revalidate_one(job):
    path, source_row = job
    raw = Path(path).read_bytes()
    result = json.loads(gzip.decompress(raw))
    verifier = _VERIFIERS[result["data_source"]]
    data = _DATA(**game_data(source_row))
    responses = [score_response(response["text"], verifier, data) for response in result["responses"]]
    if len(responses) != 16:
        raise RuntimeError("A frozen prompt does not have exactly 16 responses")
    gold = data.answer
    gold_key = answer_key(gold if isinstance(gold, str) else json.dumps(gold, ensure_ascii=False)) if gold is not None else None
    label = choose_wrong_label(responses, [gold_key] if gold_key else [])
    correct_keys = {response["answer_key"] for response in responses if response["correct"] is True}
    collisions = sum(response["correct"] is False and response["answer_key"] in correct_keys
                     for response in responses if response["answer_key"])
    reason = None if label else ("all_16_correct" if all(response["correct"] is True for response in responses)
                                else "no_extractable_verified_wrong_answer")
    return {"source_index": result["source_index"], "data_source": result["data_source"],
            "synthetic_label": label, "excluded_reason": reason, "selection_version": 4,
            "raw_sha256": hashlib.sha256(raw).hexdigest(), "correct_rollouts": sum(response["correct"] is True for response in responses),
            "unverified_rollouts": sum(response["correct"] is None for response in responses),
            "correct_key_collisions_excluded": collisions,
            "verification_seconds": sum(response["verification_seconds"] for response in responses)}


def revalidate(args):
    import pyarrow.parquet as pq
    root = args.root
    source = pq.read_table(root / "data/source/synlogic_easy/train.parquet").to_pylist()
    target = root / "data/revalidated"
    target.mkdir(exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"), initializer=initialize_revalidation) as executor:
        last_report = 0
        while True:
            complete = {int(path.stem) for path in target.glob("*.json")}
            paths = [path for path in sorted((root / "data/rollouts").glob("shard-*/*.json.gz"))
                     if int(path.name.split(".")[0]) not in complete]
            jobs = [(str(path), source[int(path.name.split(".")[0])]) for path in paths]
            for result in executor.map(revalidate_one, jobs, chunksize=1):
                index = result["source_index"]
                write_json(target / f"{index:05d}.json", result)
                complete.add(index)
                if time.time() - last_report > 2 or len(complete) == len(source):
                    write_json(root / "data/revalidation-progress.json", {"completed": len(complete), "total": len(source), "epoch": time.time()})
                    last_report = time.time()
            if len(complete) == len(source):
                return
            if not args.watch:
                return
            if (root / "STOP").exists():
                return
            time.sleep(10)


def generate(args):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Generation worker must see exactly one GPU, saw {torch.cuda.device_count()}")
    free, total = torch.cuda.mem_get_info(0)
    if total - free > 1024**3:
        raise RuntimeError("The assigned generation GPU has an existing workload to reconcile")
    print(json.dumps({"gpu": torch.cuda.get_device_name(0), "visible_gpus": torch.cuda.device_count()}), flush=True)
    root = args.root
    verifiers, Data = load_verifiers()
    rows = pq.read_table(root / "data/source/synlogic_easy/train.parquet").to_pylist()
    assigned = [(i, row) for i, row in enumerate(rows) if i % args.shards == args.shard]
    target = root / f"data/rollouts/shard-{args.shard:02d}"
    target.mkdir(parents=True, exist_ok=True)
    completed = {int(p.name.split(".")[0]) for p in target.glob("*.json.gz")}
    remaining = [(i, row) for i, row in assigned if i not in completed]
    if args.limit:
        remaining = remaining[:args.limit]
    start = time.time()
    total_tokens = 0
    progress = root / f"data/rollouts/progress-{args.shard:02d}.json"
    def report(phase, **extra):
        write_json(progress, {"phase": phase, "shard": args.shard, "completed": len(completed),
                             "total": len(assigned), "epoch": time.time(), "tokens_this_attempt": total_tokens,
                             "elapsed_seconds": time.time() - start, **extra})
    report("loading")
    if not remaining:
        report("complete")
        return
    model = str(root / "model")
    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, tensor_parallel_size=1, dtype="bfloat16", seed=42 + args.shard,
              gpu_memory_utilization=0.80, max_model_len=10240,
              max_num_batched_tokens=16384, max_num_seqs=256,
              enable_prefix_caching=True, enable_chunked_prefill=True, enforce_eager=True)
    for begin in range(0, len(remaining), args.batch):
        if (root / "STOP").exists():
            report("user_stopped")
            return
        batch = remaining[begin:begin + args.batch]
        inputs = []
        parameters = []
        for index, row in batch:
            tokens = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True)
            inputs.append({"prompt_token_ids": tokens[-2048:]})
            parameters.append(SamplingParams(n=16, temperature=1.0, top_p=1.0, top_k=-1,
                                              max_tokens=8192, seed=42 + index))
        report("generating", batch_indices=[index for index, _ in batch])
        outputs = llm.generate(inputs, parameters, use_tqdm=False)
        for (index, row), output in zip(batch, outputs, strict=True):
            verifier = verifiers[row["data_source"]]
            data = Data(**game_data(row))
            records = []
            for response in output.outputs:
                scored = score_response(response.text, verifier, data)
                records.append({"text": response.text, **scored, "tokens": len(response.token_ids),
                                "finish_reason": response.finish_reason})
                total_tokens += len(response.token_ids)
            if len(records) != 16:
                raise RuntimeError(f"Expected exactly 16 rollouts for row {index}")
            label = choose_wrong_label(records)
            reason = None if label else ("all_16_correct" if all(r["correct"] is True for r in records)
                                         else "no_extractable_verified_wrong_answer")
            result = {"source_index": index, "data_source": row["data_source"], "responses": records,
                      "synthetic_label": label, "excluded_reason": reason, "selection_version": 2}
            path = target / f"{index:05d}.json.gz"
            temporary = path.with_suffix(".tmp")
            with gzip.open(temporary, "wt", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False)
            temporary.replace(path)
            completed.add(index)
            report("generating")
        print(json.dumps({"shard": args.shard, "completed": len(completed), "total": len(assigned),
                          "generated_tokens": total_tokens, "elapsed": time.time() - start}), flush=True)
    report("complete" if len(completed) == len(assigned) else "calibration_complete")


def build(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = args.root
    source = pq.read_table(root / "data/source/synlogic_easy/train.parquet").to_pylist()
    records = {}
    for path in sorted((root / "data/revalidated").glob("*.json")):
        result = json.loads(path.read_text())
        index = result["source_index"]
        if result.get("selection_version") != 4:
            raise RuntimeError(f"Selection algorithm mismatch for source row {index}")
        if index in records:
            raise RuntimeError(f"Duplicate generation result: {index}")
        records[index] = result
    if set(records) != set(range(len(source))):
        raise RuntimeError(f"Generation incomplete: {len(records)}/{len(source)}")
    eligible = [index for index in range(len(source)) if records[index]["synthetic_label"]]
    rows = [source[index] for index in eligible]
    subsets, masks = nested_subsets(rows, seed=42)
    output = root / "data/prepared"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"eligible": len(eligible), "source_count": len(source), "eligible_source_indices": eligible,
                "excluded": dict(Counter(r["excluded_reason"] for r in records.values() if r["excluded_reason"])),
                "seed": 42, "conditions": {}}

    def training_row(source_row, source_index, synthetic=False):
        original = game_data(source_row)
        data = dict(original)
        label = records[source_index]["synthetic_label"] if synthetic else None
        if label:
            data["answer"] = label["answer"]
        return {"data_source": source_row["data_source"], "prompt": source_row["prompt"],
                "ability": "reasoning", "reward_model": {"style": "rule", "ground_truth": str(data.get("answer", ""))},
                "extra_info": {"index": source_index, "game_data_str": json.dumps(data, ensure_ascii=False),
                               "original_game_data_str": json.dumps(original, ensure_ascii=False),
                               "assigned_answer_key": label["answer_key"] if label else "",
                               "target_mode": "fixed_wrong_answer" if label else "original_verifier"}}

    for name in ["full-clean", "clean10", "clean1", "flipped90", "noise90", "noise70", "noise50", "noise30"]:
        indices = subsets.get(name, list(range(len(rows))))
        mask = set(masks[int(name[5:])]) if name.startswith("noise") else set()
        prepared = [training_row(rows[i], eligible[i], synthetic=i in mask) for i in indices]
        unique_count = len(prepared)
        if not unique_count:
            raise RuntimeError(f"Empty condition: {name}")
        # Uniform repetition preserves the empirical distribution below a batch.
        repetitions = max(1, (128 + unique_count - 1) // unique_count)
        prepared = prepared * repetitions
        path = output / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(prepared), path)
        manifest["conditions"][name] = {"unique_prompts": unique_count, "physical_rows": len(prepared),
            "uniform_repetitions": repetitions, "source_indices": [eligible[i] for i in indices],
            "corrupted_source_indices": [eligible[i] for i in sorted(mask)],
            "noise_count": len(mask), "noise_fraction": len(mask) / unique_count,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    validation = pq.read_table(root / "data/source/synlogic_easy/validation.parquet").to_pylist()
    valid_rows = []
    for index, row in enumerate(validation):
        data = game_data(row)
        valid_rows.append({"data_source": row["data_source"], "prompt": row["prompt"], "ability": "reasoning",
                           "reward_model": {"style": "rule", "ground_truth": str(data.get("answer", ""))},
                           "extra_info": {"index": index, "game_data_str": json.dumps(data, ensure_ascii=False),
                                          "original_game_data_str": json.dumps(data, ensure_ascii=False),
                                          "assigned_answer_key": "", "target_mode": "original_verifier"}})
    pq.write_table(pa.Table.from_pylist(valid_rows), output / "validation.parquet")
    write_json(output / "manifest.json", manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key not in ("conditions", "eligible_source_indices")}, indent=2))
    print(json.dumps({key: {k: v for k, v in val.items() if not k.endswith("indices")}
                      for key, val in manifest["conditions"].items()}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("generate", "revalidate", "build"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    {"generate": generate, "revalidate": revalidate, "build": build}[args.phase](args)


if __name__ == "__main__":
    main()
