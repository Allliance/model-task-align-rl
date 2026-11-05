"""Scoped validation IDs, checkpoint identity and optional clean-guide weights."""
import json
import os
import time
from pathlib import Path
from collections import Counter
from src.dapo_ray_trainer import RayDAPOTrainer
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from .sampling import PlanDataLoader, PlanSampler


class ValidationRecorder:
    def __init__(self,reward):
        self.reward,self.records=reward,[]

    def __call__(self,data,return_dict=False):
        result=self.reward(data,return_dict=True)
        for i,score in enumerate(result['reward_tensor'].sum(-1).tolist()):
            self.records.append((int(data[i].non_tensor_batch['extra_info']['index']),score))
        return result if return_dict else result['reward_tensor']


class GuidedTrainer(RayDAPOTrainer):
    def __init__(self,*args,**kwargs):
        self.method_metrics={}
        super().__init__(*args,**kwargs)
        self.val_reward_fn=ValidationRecorder(self.val_reward_fn)

    def _create_dataloader(self,train_dataset,val_dataset,collate_fn,train_sampler):
        from verl.trainer.main_ppo import create_rl_dataset
        if train_dataset is None:
            train_dataset=create_rl_dataset(self.config.data.train_files,self.config.data,self.tokenizer,self.processor)
        sampler=PlanSampler(train_dataset,seed=self.config.data.seed,batch_size=self.config.data.train_batch_size,
                            batches=self.config.trainer.total_training_steps)
        super()._create_dataloader(train_dataset,val_dataset,collate_fn,sampler)
        original=self.train_dataloader
        self.train_dataloader=PlanDataLoader(dataset=train_dataset,batch_size=original.batch_size,
            num_workers=0,drop_last=True,collate_fn=original.collate_fn,sampler=sampler)
        self.expected_dev_ids={int(extra['index']) for extra in self.val_dataset.dataframe['extra_info']}
        frozen=Path(self.config.guided.root)/'data/prepared/dev-indices.json'
        if not frozen.is_file() or self.expected_dev_ids != set(json.loads(frozen.read_text())):
            raise ValueError('Development validation IDs differ from the frozen task split')
        if len(self.val_dataset)!=96 or len(self.expected_dev_ids)!=96:
            raise ValueError('Pilot validation must contain exactly the 96 frozen audited development prompts')

    def init_workers(self):
        super().init_workers()
        if self.config.guided.condition not in ('d1','d2','union','oracle'):
            from .weights import prepare_guided_batch
            original=self.actor_rollout_wg.update_actor
            def update_actor(data):
                self.method_metrics=prepare_guided_batch(data,self.config)
                return original(data)
            self.actor_rollout_wg.update_actor=update_actor

    def _save_checkpoint(self):
        folder=Path(self.config.trainer.default_local_dir)/f'global_step_{self.global_steps}'
        folder.mkdir(parents=True,exist_ok=True)
        state=dict(step=self.global_steps,protocol_id=self.config.provenance.protocol_id,
            prompt_template_sha256=self.config.provenance.prompt_template_sha256,
            condition=self.config.guided.condition,world_size=1,optimizer_updates=self.global_steps*5,
            guide_model=self.config.actor_rollout_ref.ref.get('model_path'))
        temporary=folder/'experiment_state.tmp'; temporary.write_text(json.dumps(state,indent=2)+'\n')
        temporary.replace(folder/'experiment_state.json')
        super()._save_checkpoint()
        # The parent method has closed and checked data.pt and every native
        # model/optimizer/scheduler/RNG shard. Publish the completion marker
        # only after those files exist; validators must never infer this from
        # latest_checkpointed_iteration.txt alone.
        complete={**state,'complete':True,'completed_epoch':time.time()}
        temporary=folder/'complete.tmp'
        with temporary.open('w') as handle:
            handle.write(json.dumps(complete,indent=2)+'\n')
            handle.flush(); os.fsync(handle.fileno())
        temporary.replace(folder/'complete.json')

    def _load_checkpoint(self):
        result=super()._load_checkpoint()
        if self.global_steps:
            folder=Path(self.config.trainer.default_local_dir)/f'global_step_{self.global_steps}'
            state=json.loads((folder/'experiment_state.json').read_text())
            expected=(self.global_steps,self.config.provenance.protocol_id,
                      self.config.provenance.prompt_template_sha256,self.config.guided.condition,1,
                      self.config.actor_rollout_ref.ref.get('model_path'))
            actual=tuple(state.get(key) for key in ('step','protocol_id','prompt_template_sha256',
                                                   'condition','world_size','guide_model'))
            if actual!=expected:
                raise ValueError('Incompatible full-state pilot checkpoint')
        return result

    def _validate(self):
        self._progress('evaluation')
        self.val_reward_fn.records.clear()
        metrics=RayPPOTrainer._validate(self)
        records=self.val_reward_fn.records
        counts=Counter(index for index,score in records)
        if set(counts)!=self.expected_dev_ids or any(count!=8 for count in counts.values()):
            raise ValueError('Development IDs or eight-response validation grouping changed')
        metrics['val-core/synlogic-dev-audited/reward/mean@8']=sum(score for _,score in records)/len(records)
        metrics['validation/dev_prompts']=len(counts)
        directory=Path(self.config.trainer.default_local_dir).parent/'evaluations'
        directory.mkdir(parents=True,exist_ok=True)
        payload={'step':self.global_steps,'scope':'frozen-development-tasks-only','responses_per_prompt':8,
                 'scores_by_original_index':{str(index):[score for key,score in records if key==index]
                                             for index in sorted(counts)}}
        target=directory/f'step_{self.global_steps}.json'
        temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(payload,indent=2)+'\n');temporary.replace(target)
        return metrics

    def _progress(self,phase,metrics=None):
        if metrics is not None:
            metrics.update(self.method_metrics)
        return super()._progress(phase,metrics)

    def fit(self):
        import verl.utils.tracking as tracking
        original=tracking.Tracking
        trainer=self
        class PilotTracking(original):
            def log(self,data,step):
                data.update(trainer.method_metrics)
                data['trainer/iteration']=step
                return super().log(data,step)
        tracking.Tracking=PilotTracking
        try:
            return super().fit()
        finally:
            tracking.Tracking=original
