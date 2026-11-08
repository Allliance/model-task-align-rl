# Training noise diagnostics

The shared DAPO backend freezes the original corruption mask and assigned label before online refinement. Clean references are diagnostics only and never affect training rewards, refinement, filtering, advantages, or optimizer weights.

| Metric | Numerator | Denominator |
|---|---|---|
| `noise/active_rate` | Noisy question groups with any nonzero actual masked advantage | Noisy question groups in the optimizer batch |
| `noise/noise_learnability` | Responses matching the frozen assigned noisy label | Noisy responses |
| `noise/true_learnability` | Responses passing the original clean verifier | Noisy responses |

Log `noise/groups`, `noise/rollouts`, `noise/active_groups`, `noise/active_rollouts`, `noise/assigned_matches`, `noise/true_correct`, `noise/true_verified`, `noise/true_verifier_errors` and `noise/true_verifier_coverage`. Omit rates when their denominator is zero; retain counts. Missing answers are incorrect. Verifier failures count as unsuccessful in the all-response rate, so inspect coverage. Repeated draws count as visits, not unique questions. Mathador's known target self-comparison bug is corrected with safe arithmetic only in the diagnostic; optimization remains unchanged. Other native-verifier limitations still apply.

With evaluation every20 updates, keep snapshots at19,39,... and the update before final evaluation. Initial evaluation has no preceding training rollout. Every generated rollout is retained at selected steps, including discarded groups, with stable question/group/response IDs, prompt/response text and token IDs, masks, frozen/effective labels, clean references, rewards, and actual advantages for responses included in optimization. Include generation policy/update versions, source/data fingerprints and resolved-configuration hash. Discarded responses have no optimization advantage.

Write compressed JSONL under `rollout_snapshots/step_NNNNNN`, outside prunable model checkpoints. Payloads are written after the generating update succeeds, fsynced and atomically renamed with their SHA256; completion manifests identify counts and provenance. Identical replay is idempotent; changed recovery attempts remain separate. The checkpoint-only HF mirror does not upload rollout logs. Other reward adapters can supply the same diagnostic fields without duplicating metric definitions. Inspect existing capabilities before adding an adapter in another infrastructure.
