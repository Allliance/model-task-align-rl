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
    with (root / ('scoring-memory-rank'+str(record['rank'])+'.jsonl')).open('a') as stream:
        stream.write(json.dumps(record)+'\n')
