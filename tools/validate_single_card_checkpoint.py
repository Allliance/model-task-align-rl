#!/usr/bin/env python3
"""Deserialize every full-state component of an immutable one-rank checkpoint."""

import argparse
import fcntl
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint
    lock = (checkpoint.parent / (".validation-" + checkpoint.name + ".lock")).open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    if (checkpoint / "validated.json").exists():
        return
    require((checkpoint / "complete.json").is_file(), "Missing completion marker")
    actor = checkpoint / "actor"
    model = torch.load(actor / "model_world_size_1_rank_0.pt", map_location="cpu", weights_only=False, mmap=True)
    require(isinstance(model, dict) and len(model) > 100, "Incomplete model state")
    parameters = sum(value.numel() for value in model.values() if isinstance(value, torch.Tensor))
    require(parameters > 7_000_000_000, f"Incomplete model parameters: {parameters}")
    del model
    optimizer = torch.load(actor / "optim_world_size_1_rank_0.pt", map_location="cpu", weights_only=False, mmap=True)
    require(optimizer["state"] and optimizer["param_groups"], "Missing optimizer state")
    for group in optimizer["param_groups"]:
        require(math.isfinite(group["lr"]) and group["lr"] > 0, "Invalid optimizer learning rate")
    states = len(optimizer["state"])
    del optimizer
    extra = torch.load(actor / "extra_state_world_size_1_rank_0.pt", map_location="cpu", weights_only=False)
    require(extra.get("lr_scheduler") is not None, "Missing scheduler recovery state")
    require(extra.get("rng"), "Missing RNG recovery state")
    loader = torch.load(checkpoint / "data.pt", map_location="cpu", weights_only=False)
    require(isinstance(loader, dict) and loader, "Missing dataloader state")
    result = {"step": int(checkpoint.name.removeprefix("global_step_")), "validated": True,
              "model_parameters": parameters, "optimizer_states": states,
              "components": ["model", "optimizer", "lr_scheduler", "rng", "dataloader"]}
    temporary = checkpoint / "validated.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(checkpoint / "validated.json")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
