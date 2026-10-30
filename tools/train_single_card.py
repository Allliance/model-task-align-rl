#!/usr/bin/env python3
"""Resolve the paper's SynLogic training protocol for one GPU and fixed targets."""

import argparse
import json
import os
from pathlib import Path
import shlex
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "SynLogic")]


def overrides(root, condition):
    data = root / "data/prepared"
    run = root / "runs" / condition
    return [
        f"data.train_files={data}/{condition}.parquet", f"data.val_files={data}/validation.parquet",
        "data.prompt_key=prompt", "data.truncation=left", "data.max_prompt_length=2048",
        "data.max_response_length=8192", "data.gen_batch_size=128", "data.train_batch_size=128",
        "+data.dataloader_num_workers=4", "data.shuffle=True", "+data.seed=42",
        "actor_rollout_ref.rollout.n=16", "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False", "algorithm.kl_ctrl.kl_coef=0.0",
        "actor_rollout_ref.actor.use_kl_loss=False", "actor_rollout_ref.actor.kl_loss_coef=0.0",
        "actor_rollout_ref.actor.clip_ratio_low=0.2", "actor_rollout_ref.actor.clip_ratio_high=0.28",
        "actor_rollout_ref.actor.clip_ratio_c=10.0", "algorithm.filter_groups.enable=True",
        "algorithm.filter_groups.max_num_gen_batches=2", "algorithm.filter_groups.metric=seq_final_reward",
        "actor_rollout_ref.model.use_remove_padding=True", "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True", "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240",
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=10240",
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=10240",
        f"actor_rollout_ref.model.path={root}/model",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.optim.lr=1e-6", "actor_rollout_ref.actor.optim.lr_warmup_steps=10",
        "actor_rollout_ref.actor.optim.weight_decay=0.1", "actor_rollout_ref.actor.ppo_mini_batch_size=64",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.actor.entropy_coeff=0", "actor_rollout_ref.actor.grad_clip=1.0",
        "actor_rollout_ref.actor.loss_agg_mode=token-mean",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.35",
        "+actor_rollout_ref.rollout.enable_sleep_mode=False",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.enable_chunked_prefill=True",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.max_num_batched_tokens=10240",
        "actor_rollout_ref.rollout.max_num_seqs=128",
        "actor_rollout_ref.rollout.temperature=1.0", "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.top_k=-1", "actor_rollout_ref.rollout.val_kwargs.temperature=1.0",
        "actor_rollout_ref.rollout.val_kwargs.top_p=0.7", "actor_rollout_ref.rollout.val_kwargs.top_k=-1",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True", "actor_rollout_ref.rollout.val_kwargs.n=8",
        "actor_rollout_ref.ref.fsdp_config.param_offload=False", "actor_rollout_ref.actor.fsdp_config.fsdp_size=1",
        "reward_model.reward_manager=dapo", "reward_model.overlong_buffer.enable=False",
        "reward_model.overlong_buffer.len=2048", "reward_model.overlong_buffer.penalty_factor=1.0",
        "trainer.logger=[console,wandb]", "trainer.project_name=Noisy-RLVR",
        f"trainer.experiment_name=Qwen2.5-7B-SynLogic-{condition}-s42",
        "trainer.n_gpus_per_node=1", "trainer.nnodes=1", "trainer.val_before_train=True",
        "trainer.test_freq=20", "trainer.save_freq=10", "trainer.total_training_steps=300",
        # One small dataset pass must not terminate a fixed-update comparison.
        # The background supervisor prunes only checkpoints verified on the Hub.
        "trainer.total_epochs=10000", "trainer.max_actor_ckpt_to_keep=null",
        f"trainer.default_local_dir={run}/checkpoints", "trainer.resume_mode=auto",
        "ray_init.num_cpus=16",
        "reward_model.score_type=" + ("incorrect_format_score" if condition == "flipped90" else "acc_with_think_format"),
        f"hydra.run.dir={run}/hydra", "hydra.output_subdir=null",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--condition", required=True,
                        choices=("full-clean", "clean10", "clean1", "flipped90", "noise90", "noise70", "noise50", "noise30"))
    parser.add_argument("--resolve-only", action="store_true")
    args, additional = parser.parse_known_args()
    args.root = args.root.resolve()
    run = args.root / "runs" / args.condition
    run.mkdir(parents=True, exist_ok=True)
    if (args.root / "STOP").exists() or (run / "STOP").exists():
        raise SystemExit("Training has a user STOP latch")
    command = overrides(args.root, args.condition) + additional
    tracker = run / "checkpoints/latest_checkpointed_iteration.txt"
    if not args.resolve_only and tracker.exists():
        step = int(tracker.read_text().strip())
        if not (tracker.parent / f"global_step_{step}/validated.json").exists():
            raise RuntimeError("The resume checkpoint has not passed full-state validation")
        command = [value for value in command if not value.startswith("trainer.val_before_train=")]
        command.append("trainer.val_before_train=False")
    (run / "overrides.json").write_text(json.dumps(command, indent=2) + "\n")
    os.chdir(REPO)
    sys.argv = [str(REPO / "src/main_dapo.py"), *command]
    if args.resolve_only:
        sys.argv += ["--cfg", "job", "--resolve"]
    else:
        credentials = os.environ.get("SYNLOGIC_WANDB_ENV")
        if credentials:
            for line in Path(credentials).read_text().splitlines():
                line = line.strip().removeprefix("export ")
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    if key.startswith("WANDB_"):
                        parts = shlex.split(value)
                        os.environ.setdefault(key, parts[0] if parts else "")
        for key in ("WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR", "WANDB_DATA_DIR"):
            directory = run / key.lower()
            directory.mkdir(exist_ok=True)
            os.environ[key] = str(directory)
        os.environ.setdefault("WANDB_INIT_TIMEOUT", "90")
        import torch
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Each training experiment must see exactly one GPU")
        free, total = torch.cuda.mem_get_info(0)
        if total - free > 1024**3:
            raise RuntimeError("The assigned GPU already has substantial memory in use; reconcile its workload ownership")
        # The allocation is already verified by the launching supervisor.
        expected_id = f"synlogic-q25-7b-{args.condition}-s42-20261030"
        if os.environ.get("WANDB_RUN_ID", expected_id) != expected_id:
            raise RuntimeError("Inherited W&B identity does not match this condition")
        os.environ["WANDB_RUN_ID"] = expected_id
        os.environ["WANDB_RESUME"] = "allow"
    from src.main_dapo import main as train
    train()


if __name__ == "__main__":
    main()
