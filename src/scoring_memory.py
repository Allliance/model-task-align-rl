"""One first-call scoring profile, without device synchronization or fsync."""
import json
import os
import time
from pathlib import Path
import torch


def record_scoring_memory(phase, micro, data):
    root = Path(os.environ['SYNLOGIC_RUN_ROOT']) / 'scratch'
    root.mkdir(exist_ok=True)
    free, total = torch.cuda.mem_get_info()
    record = dict(epoch=time.time(), allocation=os.environ['SYNLOGIC_ALLOCATION_ID'],
        attempt=os.environ['SYNLOGIC_ATTEMPT'], rank=torch.distributed.get_rank(),
        phase=phase, micro=micro, allocated=torch.cuda.memory_allocated(),
        reserved=torch.cuda.memory_reserved(), physical_free=free, physical_total=total,
        input_shape=list(data['input_ids'].shape))
    if phase == 'before_forward' and micro == 0:
        snapshot = torch.cuda.memory._snapshot()
        settings = snapshot.get('allocator_settings', {})
        proof = dict(epoch=time.time(),allocation=record['allocation'],attempt=record['attempt'],rank=record['rank'],
            allocator_settings=settings,
            allocator_environment={key:os.environ.get(key,'') for key in ['PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF']})
        (root / ('actual-allocator-rank'+str(record['rank'])+'.json')).write_text(json.dumps(proof,indent=2)+'\n')
    record.update(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    with (root / ('scoring-memory-rank'+str(record['rank'])+'.jsonl')).open('a') as stream:
        stream.write(json.dumps(record)+'\n')
