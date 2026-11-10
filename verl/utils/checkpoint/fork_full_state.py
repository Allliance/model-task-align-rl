"""Explicit one-rank FSDP1 recovery into a two-rank continuation."""
from pathlib import Path
import functools
import json

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, FullStateDictConfig,
                                    FullOptimStateDictConfig, StateDictType)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


def full_optimizer_from_native(module, native, group, layer_classes):
    """Use FSDP's named optimizer export with the parent's actual wrap policy."""
    wrapped = FSDP(module, device_id=torch.device('cpu'), process_group=group, use_orig_params=False,
                   auto_wrap_policy=functools.partial(transformer_auto_wrap_policy,
                                                     transformer_layer_cls=set(layer_classes)))
    assert len(native['param_groups']) == 1, 'Only the verified single-group AdamW protocol is supported'
    params = list(wrapped.parameters())
    ids = native['param_groups'][0]['params']
    assert len(params) == len(ids) == len(native['state'])
    for param, index in zip(params, ids):
        for name in ['exp_avg', 'exp_avg_sq']:
            assert param.numel() == native['state'][index][name].numel(), 'Parent flat-parameter layout differs'
    optimizer = torch.optim.AdamW(wrapped.parameters())
    optimizer.load_state_dict(native)
    with FSDP.state_dict_type(wrapped, StateDictType.FULL_STATE_DICT,
                             FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
                             FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False)):
        return FSDP.optim_state_dict(wrapped, optimizer)


def load_fork(manager, directory):
    """Load only an explicitly declared, validated300 -> two-GPU fork."""
    from transformers import AutoConfig, AutoModelForCausalLM
    actor_directory = Path(directory)
    assert actor_directory.name == 'actor'
    directory = actor_directory.parent
    fork = json.loads((actor_directory/'fork-full-state.json').read_text())
    source = Path(fork['source_checkpoint'])
    proof = json.loads((source/'validated.json').read_text())
    assert manager.world_size == 2 and manager.should_load_model and manager.should_load_optimizer and manager.should_load_extra
    assert directory.name == 'global_step_300' and fork['source_step'] == 300
    assert proof['step'] == 300 and proof['world_size'] == 1 and proof['validated'] is True
    assert proof['optimizer_updates'] == 1500
    assert proof['protocol_id'] == fork['source_protocol_id']
    assert proof['prompt_template_sha256'] == fork['prompt_template_sha256']
    groups = [dist.new_group([rank], backend='gloo') for rank in range(2)]
    group = groups[manager.rank]
    native = torch.load(source/'actor/optim_world_size_1_rank_0.pt', map_location='cpu', weights_only=False, mmap=True)
    assert all(float(s['step']) == 1500 for s in native['state'].values())
    config = AutoConfig.from_pretrained(source/'actor', local_files_only=True, attn_implementation='eager')
    with torch.device('meta'):
        module = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
    module.to_empty(device='cpu')
    module.tie_weights()
    assert sum(p.numel() for p in module.parameters()) == 4022468096
    classes = {type(m) for m in module.modules() if type(m).__name__ in module._no_split_modules}
    assert {c.__name__ for c in classes} == {'Qwen3DecoderLayer'}
    full_optimizer = full_optimizer_from_native(module, native, group, classes)
    del module, native
    model = torch.load(source/'actor/model_world_size_1_rank_0.pt', map_location='cpu', weights_only=False, mmap=True)
    assert all(type(v) is torch.Tensor for v in model.values()), 'Source must be an unsharded one-rank state'
    with FSDP.state_dict_type(manager.model, StateDictType.FULL_STATE_DICT,
                             FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
                             FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False)):
        manager.model.load_state_dict(model, strict=True)
        resharded = FSDP.optim_state_dict_to_load(manager.model, manager.optimizer, full_optimizer)
        manager.optimizer.load_state_dict(resharded)
    assert all(float(s['step']) == 1500 for s in manager.optimizer.state.values())
    del model, full_optimizer, resharded
    extra = torch.load(source/'actor/extra_state_world_size_1_rank_0.pt', map_location='cpu', weights_only=False)
    assert extra.get('rng') and extra.get('lr_scheduler') is not None
    manager.load_rng_state(extra['rng'])
    manager.lr_scheduler.load_state_dict(extra['lr_scheduler'])
    # No prior rank1 existed: initialize it from the same saved rank0 stream.
    # This preserves rank0 state but does not claim bitwise elastic sampling.
    dist.destroy_process_group(group)
    dist.barrier()
