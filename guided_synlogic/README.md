# Clean-data-guided SynLogic pilots

Run from the backend checkout with the calibrated ROCm Python environment:

```bash
python -m guided_synlogic.run --root RUN_ROOT --condition d1 --gpus 1 --resolve-only
python -m guided_synlogic.run --root RUN_ROOT --condition d1 --gpus 1
python -m guided_synlogic.run --root RUN_ROOT --condition guide-adaptive-kl --gpus 1 --guide-model D1_STEP20_EXPORT
```

Conditions are `d1`, `d2`, `union`, `oracle`, `base-kl`, `guide-kl`,
`guide-adaptive-kl`, and `guide-soft-pg`. Methods use the same uniform union
sampling and untouched assigned labels. Guided variants load the D1 step20
export only in the separate reference worker; the actor starts from the base.
The fixed pilot has 60 outer updates, five Adam updates per outer update,
160 prompts with eight responses, and validation/checkpoints every 20 updates.
The inherited training loop also saves the first update after every start.

Only `dev.parquet` is loaded for evaluation. Its 96 audited prompt IDs must
match `dev-indices.json`, and every evaluation must produce exactly eight
responses for each ID. Reserved tasks are not loaded or scored. Local evaluation
records retain the original validation IDs for paired analysis.

Uniform KL arms multiply D2 KL by ten. Adaptive KL computes each complete
UID group's centered correlation between GRPO advantages and token-mean
`log(reference/current)` on the same responses, then sets the multiplier to
`1 + 9 * max(negative_correlation, 0)` on D2. D1 and zero-advantage groups keep
the baseline multiplier. The soft-PG arm uses weight `1 - 0.9 * conflict`, with
reciprocal KL compensation so its KL term remains unattenuated. Only pool
membership, UIDs, masks, advantages and log probabilities enter these scores;
original answers and corruption indicators never enter method logic.

Checkpoint completion is published only after the native shards and dataloader
state exist. Recovery requires independent `validated.json` and matching
protocol, condition, template, world size and reference model. Keep the D1
step20 native state until its reference export has been verified.

CPU correctness checks: `python -m guided_synlogic.test_recipe`.
