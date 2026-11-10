"""Verify public FSDP one-to-two-rank recovery, including AdamW moments."""
import functools
import os
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, FullStateDictConfig,
                                    FullOptimStateDictConfig, StateDictType)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(7, 11)
        self.b = torch.nn.Linear(11, 7)

    def forward(self, x):
        return self.b(torch.tanh(self.a(x)))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(13, 7)
        self.layers = torch.nn.ModuleList([Block(), Block()])
        self.output = torch.nn.Linear(7, 13, bias=False)
        self.output.weight = self.embedding.weight

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


def wrapped():
    torch.manual_seed(42)
    model = FSDP(Model(), device_id=torch.device('cpu'), use_orig_params=False,
                 auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Block}))
    return model, torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=.01)


def worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + directory + '/world2', rank=rank, world_size=2)
    model, optimizer = wrapped()
    full_model = torch.load(Path(directory)/'model.pt', weights_only=False)
    full_optimizer = torch.load(Path(directory)/'optimizer.pt', weights_only=False)
    from verl.utils.checkpoint.fork_full_state import full_optimizer_from_native
    groups = [dist.new_group([i], backend='gloo') for i in range(2)]
    native = torch.load(Path(directory)/'native.pt', weights_only=False)
    converted = full_optimizer_from_native(Model(), native, groups[rank], {Block})
    assert converted['param_groups'] == full_optimizer['param_groups']
    for name, state in full_optimizer['state'].items():
        for key, value in state.items():
            assert torch.equal(value, converted['state'][name][key]), (rank, name, key)
    full_optimizer = converted
    dist.destroy_process_group(groups[rank])
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                             FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
                             FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False)):
        model.load_state_dict(full_model)
        optimizer.load_state_dict(FSDP.optim_state_dict_to_load(model, optimizer, full_optimizer))
        recovered_model = model.state_dict()
        recovered_optimizer = FSDP.optim_state_dict(model, optimizer)
    assert recovered_optimizer['param_groups'] == full_optimizer['param_groups']
    for name, value in full_model.items():
        assert torch.equal(value, recovered_model[name]), (rank, name)
    for name, state in full_optimizer['state'].items():
        for key, value in state.items():
            assert torch.equal(value, recovered_optimizer['state'][name][key]), (rank, name, key)
    loss = model(torch.tensor([[1, 2, 3]])).sum()
    loss.backward()
    optimizer.step()
    assert all(float(state['step']) == 4 for state in optimizer.state.values())
    dist.destroy_process_group()


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as directory:
        dist.init_process_group('gloo', init_method='file://' + directory + '/world1', rank=0, world_size=1)
        model, optimizer = wrapped()
        for _ in range(3):
            optimizer.zero_grad()
            model(torch.tensor([[1, 2, 3]])).sum().backward()
            optimizer.step()
        native = optimizer.state_dict()
        torch.save(native, Path(directory)/'native.pt')
        optimizer.load_state_dict(native)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                 FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
                                 FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False)):
            torch.save(model.state_dict(), Path(directory)/'model.pt')
            torch.save(FSDP.optim_state_dict(model, optimizer), Path(directory)/'optimizer.pt')
        dist.destroy_process_group()
        mp.spawn(worker, args=(directory,), nprocs=2, join=True)
    print('CPU_RESHARD_PASS: all model tensors, AdamW moments/counters/groups exactly preserved; next update succeeded', flush=True)


def test_native_optimizer_resharding():
    main()


if __name__ == '__main__':
    main()
