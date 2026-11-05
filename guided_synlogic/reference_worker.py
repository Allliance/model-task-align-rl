"""A separate reference-role model path, scoped to these new pilots."""
from copy import deepcopy
from omegaconf import open_dict
from verl.workers.fsdp_workers import ActorRolloutRefWorker


class GuidedReferenceWorker(ActorRolloutRefWorker):
    def __init__(self,config,role):
        if role!='ref':
            raise ValueError('A guide-reference worker may only serve the ref role')
        config=deepcopy(config)
        guide=config.ref.get('model_path')
        if guide:
            with open_dict(config.model):
                config.model.path=guide
        super().__init__(config=config,role=role)
