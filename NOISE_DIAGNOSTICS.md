# Training noise diagnostics

The shared DAPO backend freezes the original corruption mask and assigned label before online refinement. Clean references are diagnostics only and never affect training rewards, refinement, filtering, advantages, or optimizer weights.

| Metric | Numerator | Denominator |
|---|---|---|
| `noise/active_rate` | Noisy question groups with any nonzero actual masked advantage | Noisy question groups generated in the step |
| `noise/noise_learnability` | Responses matching the frozen assigned noisy label | Noisy generated responses |
| `noise/true_learnability` | Responses passing the original clean verifier | Noisy generated responses |

Log `noise/groups`, `noise/rollouts`, `noise/active_groups`, `noise/active_rollouts`, `noise/assigned_matches`, `noise/true_correct`, `noise/true_verified`, `noise/true_verifier_errors` and `noise/true_verifier_coverage`. Omit rates when their denominator is zero; retain counts. Missing answers are incorrect. Verifier failures and diagnostic timeouts count as unsuccessful in the all-response rate and as errors, never verified-incorrect, so inspect coverage. Repeated draws count as visits, not unique questions. Other native-verifier limitations still apply.

The clean verdict is computed once per noisy response with the unchanged native scorer, the same one validation uses, against the frozen original reference. It feeds both `diagnostic_true_*` and `critic/true_reward/*`; only this read-only check runs under a deadline. Mathador is scored by its repaired native verifier (`mathador-target-v2`, which checks the reference target); there is no separate diagnostic implementation. Earlier snapshots and logged metrics recorded the superseded shadow arithmetic check or the upstream self-comparison and remain as recorded. New snapshot metadata records `scoring_versions`.

Validation always uses the native reference scorer, including its first single-line answer extraction and original failure handling, even when rows carry training target metadata. `target_mode=fixed_wrong_answer` is a historical, explicit, training-only mode with its own recorded parser and deadline; it is never a default. The think-format check is the original single-line pattern.

With evaluation every20 updates, keep snapshots at19,39,... and the update before final evaluation. Initial evaluation has no preceding training rollout. Every generated rollout is retained at selected steps, including discarded groups, with stable question/group/response IDs, prompt/response text and token IDs, masks, frozen/effective labels, clean references, rewards, and actual advantages for responses included in optimization. Include generation policy/update versions, source/data fingerprints and resolved-configuration hash. Discarded responses have no optimization advantage. OPD snapshots also keep the loss mode, coefficients, teacher terms and, for TIP, the selected-token mask and TIP statistics.

Write compressed JSONL under `rollout_snapshots/step_NNNNNN`, outside prunable model checkpoints. Payloads are written after the generating update succeeds, fsynced and atomically renamed with their SHA256; completion manifests identify counts and provenance. Identical replay is idempotent; changed recovery attempts remain separate. The checkpoint-only HF mirror does not upload rollout logs. Other reward adapters can supply the same diagnostic fields without duplicating metric definitions. Inspect existing capabilities before adding an adapter in another infrastructure.

Use each step folder’s CURRENT.json as the current trajectory manifest. On checkpoint rollback, later current manifests are marked superseded without deleting their payloads. Old payload manifests alone do not establish membership in the resumed trajectory. Filtered-out rollouts count in the generated denominator and have no optimization advantage.

Shadow true-correctness fields on unrefined clean rows are null, not a second correctness verdict; their native training rewards are recorded separately. Clean rows with an online-refined target retain frozen clean corruption status. Non-SynLogic reward inputs without puzzle metadata do not fabricate noise statistics.
