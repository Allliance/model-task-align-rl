#!/usr/bin/env python3
"""Deserialize every full-state component of an immutable one-rank checkpoint."""

import argparse
import fcntl
import json
import math
from pathlib import Path


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
    assert (checkpoint / "complete.json").is_file()
    actor = checkpoint / "actor"
    model = torch.load(actor / "model_world_size_1_rank_0.pt", map_location="cpu", weights_only=False, mmap=True)
    assert isinstance(model, dict) and len(model) > 100
    parameters = sum(value.numel() for value in model.values() if isinstance(value, torch.Tensor))
    assert parameters > 7_000_000_000, parameters
    del model
    optimizer = torch.load(actor / "optim_world_size_1_rank_0.pt", map_location="cpu", weights_only=False, mmap=True)
    assert optimizer["state"] and optimizer["param_groups"]
    for group in optimizer["param_groups"]:
        assert math.isfinite(group["lr"]) and group["lr"] > 0
    states = len(optimizer["state"])
    del optimizer
    extra = torch.load(actor / "extra_state_world_size_1_rank_0.pt", map_location="cpu", weights_only=False)
    assert extra.get("lr_scheduler") is not None
    assert extra.get("rng"), "Missing RNG recovery state"
    loader = torch.load(checkpoint / "data.pt", map_location="cpu", weights_only=False)
    assert isinstance(loader, dict) and loader, "Missing dataloader state"
    result = {"step": int(checkpoint.name.removeprefix("global_step_")), "validated": True,
              "model_parameters": parameters, "optimizer_states": states,
              "components": ["model", "optimizer", "lr_scheduler", "rng", "dataloader"]}
    temporary = checkpoint / "validated.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(checkpoint / "validated.json")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
