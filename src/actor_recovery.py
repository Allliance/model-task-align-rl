"""Bounded OPD actor diagnostics and exact first-policy rollout replay."""
import copy
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf


def _root():
    # Operational recovery files are separate from scientific rollout snapshots.
    root = Path(os.environ['SYNLOGIC_RUN_ROOT']) / 'scratch' / 'actor-recovery'
    root.mkdir(parents=True, exist_ok=True)
    return root


def _atomic_save(path, value):
    temp = path.with_suffix('.tmp-' + str(os.getpid()))
    with temp.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def _enabled(trainer):
    opd = trainer.config.get('opd', {})
    return (opd.get('retrieval_privilege', False)
            and not opd.get('shared_current_opsd', False)
            and int(trainer.global_steps) == 1)


def _binding(trainer, batch):
    config = trainer.config
    fields = {'protocol': config.provenance.protocol_id,
              'data': config.provenance.data_fingerprint,
              'source': config.opd.source_step,
              'mapping': config.opd.retrieval_manifest_sha256,
              'context': config.opd.context_budget_sha256,
              'policy_outer_step': int(trainer.global_steps) - 1,
              'rollout': OmegaConf.to_container(config.actor_rollout_ref.rollout, resolve=True)}
    digest = hashlib.sha256(json.dumps(fields, sort_keys=True).encode())
    for key in sorted(batch.batch.keys()):
        value = batch.batch[key].detach().cpu().contiguous()
        digest.update(str((key, str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    if 'raw_prompt_ids' in batch.non_tensor_batch:
        digest.update(json.dumps([list(map(int, row)) for row in batch.non_tensor_batch['raw_prompt_ids']]).encode())
    return digest.hexdigest()


def generation_with_replay(trainer, batch):
    if not _enabled(trainer):
        return trainer.actor_rollout_wg.generate_sequences(batch)
    path = _root() / 'policy0-generation.pt'
    binding = _binding(trainer, batch)
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        # A different resolved input or policy can never consume this cache.
        if saved['binding'] == binding:
            result = saved['batch']
            result.meta_info['timing'] = {}
            receipt = {'epoch': time.time(), 'binding': binding, 'rows': len(result),
                       'attempt': os.environ.get('SYNLOGIC_ATTEMPT'), 'exact_rollout_replayed': True,
                       'worker_rng_continuity': 'Restarted generation worker RNG is not restored'}
            (_root() / 'replay.json').write_text(json.dumps(receipt))
            return result
    result = trainer.actor_rollout_wg.generate_sequences(batch)
    _atomic_save(path, {'binding': binding, 'batch': copy.deepcopy(result).to('cpu')})
    return result


def capture_actor_batch(trainer, batch):
    if _enabled(trainer):
        # Includes actual responses, masks, old/teacher logprobs and advantages;
        # diagnostic copy only, never fed back after a partial Adam update.
        _atomic_save(_root() / 'policy0-before-actor.pt',
                     {'batch': copy.deepcopy(batch).to('cpu'), 'outer_step': 1,
                      'allocation': os.environ.get('SYNLOGIC_ALLOCATION_ID'),
                      'attempt': os.environ.get('SYNLOGIC_ATTEMPT'), 'epoch': time.time()})


def actor_memory(actor, phase, epoch, minibatch, microbatch, data, reclaim=False, extra=None):
    if torch.cuda.is_available() and reclaim:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    steps = [int(s['step'].item() if torch.is_tensor(s['step']) else s['step'])
             for s in actor.actor_optimizer.state.values() if 'step' in s]
    if steps and min(steps) != max(steps):
        raise RuntimeError('Adam parameter counters disagree')
    record = {'epoch': time.time(), 'phase': phase, 'ppo_epoch': epoch,
              'minibatch': minibatch, 'microbatch': microbatch,
              'rank': torch.distributed.get_rank(),
              'actual_adam_counter': steps[0] if steps else 0,
              'adam_parameter_states': len(steps),
              'allocation': os.environ.get('SYNLOGIC_ALLOCATION_ID'),
              'attempt': os.environ.get('SYNLOGIC_ATTEMPT')}
    if data is not None:
        record.update(input_shape=list(data['input_ids'].shape),
                      response_shape=list(data['responses'].shape),
                      valid_tokens=int(data['attention_mask'].sum().item()))
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        record.update(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                      peak_allocated=torch.cuda.max_memory_allocated(),
                      peak_reserved=torch.cuda.max_memory_reserved(),
                      physical_free=free, physical_total=total)
    if extra:
        record.update(extra)
    path = _root() / ('memory-rank' + str(record['rank']) + '.jsonl')
    with path.open('a') as stream:
        stream.write(json.dumps(record) + '\n')
        stream.flush()
