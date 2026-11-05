"""Resolve or start one frozen 60-update Qwen3-1.7B-Base pilot."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys

CONDITIONS = ('d1', 'd2', 'union', 'oracle', 'base-kl', 'guide-kl', 'guide-adaptive-kl', 'guide-soft-pg')
GUIDED = ('guide-kl', 'guide-adaptive-kl', 'guide-soft-pg')


def overrides(root, condition, framework, guide_model=None):
    spec = importlib.util.spec_from_file_location('single_card', framework / 'tools/train_single_card.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dataset = condition if condition in CONDITIONS[:4] else 'union'
    run = root / 'runs' / condition
    metadata = json.loads((run / 'portable-provenance.json').read_text())
    replacements = {
        'data.train_files': str(root / 'data/prepared' / f'{dataset}.parquet'),
        'data.val_files': str(root / 'data/prepared/dev.parquet'),
        'data.train_batch_size': '160', 'data.gen_batch_size': '160',
        'data.max_prompt_length': '24576', 'data.max_response_length': '4096',
        'data.truncation': 'error', 'data.filter_overlong_prompts': 'False',
        'data.dataloader_num_workers': '0', 'actor_rollout_ref.rollout.n': '8',
        'actor_rollout_ref.model.path': str(root / 'model'),
        'algorithm.filter_groups.enable': 'False',
        'actor_rollout_ref.actor.use_kl_loss': 'True',
        'actor_rollout_ref.actor.kl_loss_coef': '0.001',
        'actor_rollout_ref.actor.clip_ratio_high': '0.2',
        'actor_rollout_ref.actor.clip_ratio_c': '3.0',
        'actor_rollout_ref.actor.optim.lr': '1e-6',
        'actor_rollout_ref.actor.optim.lr_warmup_steps': '0',
        'actor_rollout_ref.actor.optim.weight_decay': '0.01',
        'actor_rollout_ref.actor.ppo_mini_batch_size': '32',
        'actor_rollout_ref.actor.fsdp_config.fsdp_size': '1',
        'actor_rollout_ref.rollout.max_num_seqs': '256',
        'actor_rollout_ref.rollout.max_num_batched_tokens': '28672',
        'actor_rollout_ref.actor.ppo_max_token_len_per_gpu': '28672',
        'actor_rollout_ref.ref.log_prob_max_token_len_per_gpu': '28672',
        'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu': '28672',
        'actor_rollout_ref.rollout.val_kwargs.temperature': '0.6',
        'actor_rollout_ref.rollout.val_kwargs.top_p': '1.0',
        'reward_model.score_type': 'acc_without_think_format',
        'trainer.total_training_steps': '60', 'trainer.total_epochs': '10000',
        'trainer.save_freq': '20', 'trainer.test_freq': '20',
        'trainer.n_gpus_per_node': '1', 'ray_init.num_cpus': '16',
        'trainer.experiment_name': f'Qwen3-1.7B-Base-SynLogic-Guided-{condition}-s42-20261104',
        'trainer.default_local_dir': str(run / 'checkpoints'),
        'hydra.run.dir': str(run / 'hydra'),
    }
    result = []
    for value in module.overrides(root, 'full-clean'):
        key = value.split('=', 1)[0].lstrip('+')
        result.append(value.split('=', 1)[0] + '=' + replacements.pop(key) if key in replacements else value)
    result += [key + '=' + value for key, value in replacements.items()]
    result += ['actor_rollout_ref.actor.kl_loss_type=low_var_kl',
               'actor_rollout_ref.rollout.do_sample=True',
               '+actor_rollout_ref.rollout.stop_token_ids=[151643,151645]',
               '+guided.condition=' + condition, '+guided.root=' + str(root),
               '+provenance.protocol_id=' + metadata['protocol_id'],
               '+provenance.prompt_template_sha256=' + metadata['prompt_template_sha256']]
    if condition in GUIDED:
        if guide_model is None:
            raise ValueError('This condition requires --guide-model pointing at the complete D1 step20 export')
        result += ['+actor_rollout_ref.ref.model_path=' + str(guide_model)]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--framework-code', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--condition', choices=CONDITIONS, required=True)
    parser.add_argument('--guide-model', type=Path)
    parser.add_argument('--gpus', type=int, choices=(1,), default=1)
    parser.add_argument('--resolve-only', action='store_true')
    args, extra = parser.parse_known_args()
    root, framework = args.root.resolve(), args.framework_code.resolve()
    run = root / 'runs' / args.condition
    if any(path.exists() for path in (root/'STOP', root/'LOW_DISK', run/'STOP')):
        raise SystemExit('A user STOP or disk guard is latched')
    paths = [str(framework), str(framework/'SynLogic')]
    sys.path[:0] = paths
    os.environ['PYTHONPATH'] = os.pathsep.join(paths)
    os.environ['SYNLOGIC_FRAMEWORK_CODE'] = str(framework)
    os.environ['SYNLOGIC_RUN_ROOT'] = str(run)
    guide_model = args.guide_model.resolve() if args.guide_model else None
    settings = overrides(root, args.condition, framework, guide_model) + extra
    tracker = run/'checkpoints/latest_checkpointed_iteration.txt'
    if tracker.exists():
        folder = tracker.parent/f'global_step_{int(tracker.read_text())}'
        if not (folder/'validated.json').exists():
            raise ValueError('Recovery checkpoint must pass full-state validation')
        settings += ['trainer.val_before_train=False']
    run.mkdir(parents=True, exist_ok=True)
    (run/'overrides.json').write_text(json.dumps(settings, indent=2)+'\n')
    sys.argv = [str(Path(__file__).with_name('main.py')), *settings]
    if args.resolve_only:
        sys.argv += ['--cfg', 'job', '--resolve']
    else:
        import torch
        if torch.cuda.device_count() != 1:
            raise ValueError('Each pilot requires exactly one allocated visible GPU')
        free, total = torch.cuda.mem_get_info(0)
        if total-free > 1024**3:
            raise ValueError('Assigned GPU has an unreconciled existing workload')
        env_path = os.environ.get('SYNLOGIC_WANDB_ENV')
        if env_path:
            for line in Path(env_path).read_text().splitlines():
                line = line.strip().removeprefix('export ')
                if line.startswith('WANDB_') and '=' in line:
                    key,value=line.split('=',1); parts=shlex.split(value)
                    os.environ.setdefault(key, parts[0] if parts else '')
        for key in ('WANDB_DIR','WANDB_CACHE_DIR','WANDB_CONFIG_DIR','WANDB_DATA_DIR'):
            path=run/key.lower(); path.mkdir(exist_ok=True); os.environ[key]=str(path)
        metadata=json.loads((run/'portable-provenance.json').read_text())
        os.environ.update(WANDB_RUN_ID=metadata['wandb']['id'], WANDB_RESUME='allow', WANDB_INIT_TIMEOUT='90')
    os.chdir(framework)
    from guided_synlogic.main import main as train
    train()


if __name__ == '__main__':
    main()
