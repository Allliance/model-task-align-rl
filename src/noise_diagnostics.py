"""Read-only noise statistics and sparse, complete rollout snapshots.

Adapters provide frozen ``diagnostic_*`` fields; these never enter rewards,
advantage estimation, filtering, label refinement, or optimizer weighting.
"""
from collections import defaultdict
from collections import Counter
import ast
from fractions import Fraction
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import re
import time


def snapshot_directory(config):
    checkpoint_dir = Path(config.trainer.default_local_dir)
    return config.trainer.get('rollout_snapshot_dir', str(checkpoint_dir.with_name(checkpoint_dir.name + '_rollout_snapshots')))


def _sync_directory(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _sync_directory(path.parent)


def invalidate_snapshots_after(directory, restored_step):
    """Retain old payloads, but stop presenting rolled-back trajectories as current."""
    for folder in Path(directory).glob('step_*'):
        if re.fullmatch(r'step_[0-9]+', folder.name) and int(folder.name.removeprefix('step_')) > restored_step:
            current = folder / 'CURRENT.json'
            if current.exists():
                current.rename(folder / f'superseded-on-resume-{restored_step}-{time.time_ns()}.json')
                _sync_directory(folder)


def mathador_true_correct(data, extracted, verifier):
    """Shadow-only correction of upstream's overwritten target/self-comparison.

    The optimization verifier is untouched. Evaluate arithmetic without eval().
    """
    expression = verifier.extract_answer(extracted)
    if not expression or len(expression) > 512:
        return False
    metadata = data.metadata
    if Counter(re.findall(r'\d+', expression)) != Counter(str(n) for n in metadata['numbers']):
        return False
    if any(c not in '0123456789() \n\t' and c not in metadata['operators'] for c in expression):
        return False
    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right
        raise ValueError('Not a permitted arithmetic expression')
    try:
        value = evaluate(ast.parse(expression, mode='eval').body)
        return abs(value - Fraction(str(metadata['result']))) < Fraction(1, 10**10)
    except (ValueError, SyntaxError, ZeroDivisionError, RecursionError):
        return False


def snapshot_due(step, evaluation_frequency, final_step, has_evaluator=True):
    return bool(has_evaluator and evaluation_frequency > 0 and step > 0
                and (step + 1 == final_step
                     or (step + 1 <= final_step and (step + 1) % evaluation_frequency == 0)))


def capture_diagnostics(batch):
    info = batch.non_tensor_batch
    if 'diagnostic_is_noisy' not in info:
        return []
    fields = ['is_noisy', 'assigned_match', 'true_correct', 'true_verified', 'true_error', 'assigned_error']
    return [dict(uid=str(info['uid'][i]), rollout_id=str(info['diagnostic_rollout_id'][i]),
                 **{key: bool(info.get('diagnostic_' + key, [False] * len(batch))[i]) for key in fields})
            for i in range(len(batch))]


def noise_metrics(batch, generated=None):
    info = batch.non_tensor_batch
    if 'diagnostic_is_noisy' not in info:
        return {}
    active = ((batch.batch['advantages'] != 0) & batch.batch['response_mask'].bool()).any(-1).tolist()
    if generated is not None:
        active_by_id = defaultdict(bool)
        for i, uid in enumerate(info['diagnostic_rollout_id']):
            active_by_id[str(uid)] |= active[i]
        active = [active_by_id[row['rollout_id']] for row in generated]
        info = {'uid': [row['uid'] for row in generated],
                **{'diagnostic_' + key: [row[key] for row in generated]
                   for key in ['is_noisy', 'assigned_match', 'true_correct', 'true_verified', 'true_error', 'assigned_error']}}
    noisy = [bool(x) for x in info['diagnostic_is_noisy']]
    groups = defaultdict(list)
    for i, uid in enumerate(info['uid']):
        groups[str(uid)].append(i)
    for indices in groups.values():
        if len({noisy[i] for i in indices}) != 1:
            raise ValueError('Corruption annotation differs within a prompt group')
    ng = [indices for indices in groups.values() if noisy[indices[0]]]
    nr = [i for i, value in enumerate(noisy) if value]
    counts = {
        'groups': len(ng), 'rollouts': len(nr),
        'active_groups': sum(any(active[i] for i in ids) for ids in ng),
        'active_rollouts': sum(active[i] for i in nr),
        'assigned_matches': sum(bool(info['diagnostic_assigned_match'][i]) for i in nr),
        'true_correct': sum(bool(info['diagnostic_true_correct'][i]) for i in nr),
        'true_verified': sum(bool(info['diagnostic_true_verified'][i]) for i in nr),
        'true_verifier_errors': sum(bool(info['diagnostic_true_error'][i]) for i in nr),
        'assigned_match_errors': sum(bool(info.get('diagnostic_assigned_error', [False] * len(noisy))[i]) for i in nr),
    }
    metrics = {'noise/' + key: value for key, value in counts.items()}
    metrics['noise/all_groups'] = len(groups)
    metrics['noise/all_rollouts'] = len(noisy)
    # Missing/zero denominators are not measured zero performance. Counts still
    # log on clean batches, with no NaN to trip numerical-health supervision.
    if ng:
        metrics['noise/active_rate'] = counts['active_groups'] / len(ng)
    if nr:
        metrics['noise/active_rollout_rate'] = counts['active_rollouts'] / len(nr)
        metrics['noise/noise_learnability'] = counts['assigned_matches'] / len(nr)
        metrics['noise/true_learnability'] = counts['true_correct'] / len(nr)
        metrics['noise/true_verifier_coverage'] = counts['true_verified'] / len(nr)
    return metrics


def rollout_records(batch, tokenizer):
    records = []
    for i in range(len(batch)):
        item = batch[i]
        tensors, info = item.batch, item.non_tensor_batch
        width = tensors['prompts'].shape[-1]
        pmask = tensors['attention_mask'][:width].bool()
        rmask = tensors['attention_mask'][width:].bool()
        prompt_ids = tensors['prompts'][pmask].tolist()
        response_ids = tensors['responses'][rmask].tolist()
        extra = info.get('extra_info', {})
        records.append({
            'rollout_id': str(info['diagnostic_rollout_id']), 'group_id': str(info['uid']),
            'question_id': extra.get('index'), 'task': str(info.get('data_source', '')),
            'prompt_token_ids': prompt_ids, 'response_token_ids': response_ids,
            'prompt_attention_mask': [1] * len(prompt_ids),
            'response_attention_mask': [1] * len(response_ids),
            'prompt': tokenizer.decode(prompt_ids, skip_special_tokens=False),
            'response': tokenizer.decode(response_ids, skip_special_tokens=False),
            'assigned_label': extra.get('diagnostic_assigned_answer_key', extra.get('assigned_answer_key')),
            'effective_reward_target': info.get('diagnostic_effective_target', info.get('reward_model', {}).get('ground_truth')),
            'effective_reward_target_mode': info.get('diagnostic_effective_target_mode', extra.get('target_mode')),
            'original_clean_reference': extra.get('original_game_data_str', extra.get('game_data_str')),
            'is_noisy': info.get('diagnostic_is_noisy'),
            'assigned_label_match': info.get('diagnostic_assigned_match'),
            'true_correct': info.get('diagnostic_true_correct'),
            'true_verified': info.get('diagnostic_true_verified'),
            'true_verifier_error': info.get('diagnostic_true_error'),
            'token_scores': tensors['token_level_scores'][rmask].tolist(),
            'token_rewards': tensors['token_level_rewards'][rmask].tolist(),
            'included_in_update': False, 'advantages': None, 'response_loss_mask': None,
        })
    return records


def attach_advantages(records, batch):
    by_id = defaultdict(list)
    for i, uid in enumerate(batch.non_tensor_batch['diagnostic_rollout_id']):
        by_id[str(uid)].append(i)
    for row in records:
        if row['rollout_id'] in by_id:
            indices = by_id[row['rollout_id']]
            i = indices[0]
            width = batch.batch['prompts'].shape[-1]
            mask = batch.batch['attention_mask'][i, width:].bool()
            row.update(included_in_update=True, update_occurrences=len(indices),
                       advantages=batch.batch['advantages'][i, mask].tolist(),
                       response_loss_mask=batch.batch['response_mask'][i, mask].tolist())
            for name in ('outcome_advantages', 'teacher_advantages', 'teacher_log_probs',
                         'old_log_probs', 'ref_log_prob', 'opd_token_mask', 'ta_score',
                         'ta_disagreement', 'ta_compatibility', 'tip_score', 'tip_entropy', 'tip_rkl', 'tip_fkl'):
                if name in batch.batch:
                    row[name] = batch.batch[name][i, mask].tolist()
            for name in ('opd_teacher_coefficient', 'opd_outcome_coefficient'):
                if name in batch.meta_info:
                    row[name] = batch.meta_info[name]


def write_snapshot(directory, step, records, provenance):
    if len({r['rollout_id'] for r in records}) != len(records):
        raise ValueError('Duplicate rollout IDs in snapshot')
    folder = Path(directory) / f'step_{step:06d}'
    folder.mkdir(parents=True, exist_ok=True)
    _sync_directory(folder.parent)
    _sync_directory(folder.parent.parent)
    metadata = {'schema_version': 2, 'training_step': step,
                'generating_policy_completed_updates': step - 1,
                'rollouts': len(records), 'groups': len({r['group_id'] for r in records}),
                'used_in_update_rollouts': sum(bool(r.get('included_in_update')) for r in records),
                'complete': True, **provenance}
    fd, temporary = tempfile.mkstemp(prefix='.writing-', dir=folder)
    try:
        with os.fdopen(fd, 'wb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
                for row in [{'metadata': metadata}, *records]:
                    stream.write((json.dumps(row, ensure_ascii=False, default=_json_default) + '\n').encode())
            raw.flush()
            os.fsync(raw.fileno())
        digest = hashlib.sha256(Path(temporary).read_bytes()).hexdigest()
        destination = folder / f'rollouts-{digest}.jsonl.gz'
        os.replace(temporary, destination)
        marker = destination.with_suffix(destination.suffix + '.manifest.json')
        _sync_directory(folder)
        manifest = {**metadata, 'sha256': digest, 'file': destination.name, 'bytes': destination.stat().st_size}
        _atomic_json(marker, manifest)
        _atomic_json(folder / 'CURRENT.json', manifest)
        return destination
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def _json_default(value):
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(type(value).__name__)
