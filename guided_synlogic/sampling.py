"""Persist the actual sampling plan, including draws with replacement on D1."""
import torch
from torchdata.stateful_dataloader import StatefulDataLoader


class PlanIterator:
    def __init__(self, sampler):
        self.sampler, self.plan, self.cursor = sampler, sampler.plan(), 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cursor >= len(self.plan):
            raise StopIteration
        value = self.plan[self.cursor]
        self.cursor += 1
        return value

    def state_dict(self):
        return dict(version=1, shape=self.sampler.shape(), plan=self.plan,
                    cursor=self.cursor, generator=self.sampler.generator.get_state().clone())

    def load_state_dict(self, state):
        if state['version'] != 1 or state['shape'] != self.sampler.shape():
            raise ValueError('Sampling protocol mismatch')
        self.plan, self.cursor = list(state['plan']), state['cursor']
        if len(self.plan) != len(self.sampler) or not 0 <= self.cursor <= len(self.plan):
            raise ValueError('Invalid saved sampling cursor')
        if any(not 0 <= i < len(self.sampler.dataset) for i in self.plan):
            raise ValueError('Invalid saved sample index')
        self.sampler.generator.set_state(state['generator'])


class PlanSampler:
    def __init__(self, dataset, seed=42, batch_size=160, batches=60):
        self.dataset, self.batch_size = dataset, batch_size
        self.num_samples = batch_size * batches
        self.generator = torch.Generator().manual_seed(seed)
        if len(dataset) == 0:
            raise ValueError('Empty training dataset')

    def shape(self):
        return dict(length=len(self.dataset), num_samples=self.num_samples, batch_size=self.batch_size)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        return PlanIterator(self)

    def plan(self):
        if len(self.dataset) < self.num_samples:
            return torch.randint(len(self.dataset), (self.num_samples,), generator=self.generator).tolist()
        return torch.randperm(len(self.dataset), generator=self.generator).tolist()[:self.num_samples]


class PlanDataLoader(StatefulDataLoader):
    def state_dict(self):
        state = super().state_dict()
        state['guided_plan_generator'] = self.sampler.generator.get_state().clone()
        state['guided_plan_shape'] = self.sampler.shape()
        return state

    def load_state_dict(self, state):
        state = dict(state)
        if state.pop('guided_plan_shape') != self.sampler.shape():
            raise ValueError('Dataloader sampling protocol mismatch')
        self.sampler.generator.set_state(state.pop('guided_plan_generator'))
        super().load_state_dict(state)
