# Phase 5: ERM Candidate Trajectories

Phase 5 trains the ImageNet-pretrained `vit_small_patch16_224` candidate for
50 epochs on the complete Phase 0 `candidate_train` split. The candidate sees
only original RGB images with standard train/evaluation transforms. It never
receives masks, green-filled images, patch filters, place labels, or group
labels.

There is no candidate-training calibration holdout.

## Evaluation and information boundaries

At every epoch, the in-memory candidate is evaluated on:

- `biased_val`, for every realistic Phase 5 selector;
- `oracle_val`, in an isolated analysis namespace;
- `test`, in a reporting-only path.

The realistic selector function accepts only biased-validation predictions and
the immutable Phase 2–4 fusion artifacts. It has no parameter through which
oracle or test metrics can be supplied.

The protected group-label file is loaded only after realistic scoring and is
used by the isolated oracle/reporting metric function.

The test flow is structurally ordered:

```text
in-memory per-epoch test metrics
    -> freeze realistic selections
    -> write and hash selection receipt
    -> publish reporting_only/test_metrics.json bound to that hash
```

Test values are not printed in training logs, written to the rolling selector
file, displayed by the verifier, or made available to selection code.
`scripts/report_candidate_test.py` is the explicit post-freeze reveal command.

## Per-epoch artifacts

For every epoch, biased validation stores:

```text
sample_id
true_label
candidate_logits
candidate_predicted_class
candidate_cross_entropy
candidate_correct
```

The individual epoch files and a combined 50-epoch NPZ are both retained.
This allows selector formulas and diagnostics to be recomputed without
retraining the candidate.

Oracle validation stores average accuracy, group-balanced accuracy, WGA, and
per-group accuracy under `analysis_only/`. Test stores the same values under
`reporting_only/` only after selection is frozen.

## Phase 5 selectors

The realistic online tracker implements:

- ordinary biased-validation accuracy;
- hard background pseudo-groups for exact, sanitized, and set experts;
- hard SETV for each background expert when valid;
- rank SETV for each background expert;
- logistic SETV for each background expert when cross-fitting was available.

Continuous SETV logs all four alpha accuracies, weighted losses, aggregate
score/loss, classwise effective sample sizes, and alpha-4 warnings.

The plan’s exact ordinary, hard, continuous, and oracle tie-breaking rules use
the locked `1e-8` numerical tolerance. The plan does not specify pseudo-group
ties, so the resolved configuration and receipt record the deterministic
fallback:

```text
proxy WGA -> ordinary accuracy -> ordinary loss -> earlier epoch
```

uLA remains explicitly marked `deferred_to_phase6`, matching the project’s
implementation order. Phase 6 must register its reproduced validation
procedure before production claims involving the uLA-selected checkpoint.

## Rolling selected checkpoints

All 50 model states are not retained. Whenever a realistic selector changes
its best epoch, that epoch is written once to the realistic epoch-addressed
checkpoint store. Selector pointers are updated, and an epoch checkpoint is
deleted when no realistic selector references it.

Realistic selector checkpoints and pointers live under `selection/`. The
oracle has its own single rolling checkpoint and pointer under
`analysis_only/`; this deliberately keeps oracle-induced storage changes out
of the realistic namespace. If several realistic selectors choose the same
epoch, they share one physical checkpoint. The final selection receipt hashes
every surviving realistic checkpoint binding.

## Production launch

The private pilot requires at least three distinct ERM candidate seeds:

```bash
export SETV_CANDIDATE_SEEDS=101,102,103
export SETV_OBJECT_SEED=...
export SETV_EXACT_SEED=...
export SETV_SANITIZED_SEED=...
export SETV_SET_SEED=...
export SETV_EXACT_FUSION_SEED=...
export SETV_SANITIZED_FUSION_SEED=...
export SETV_SET_FUSION_SEED=...
bash scripts/submit_phase5_candidate.sh
```

The launcher refuses fewer than three candidate seeds, duplicate seeds,
missing fusion receipts, existing outputs, active duplicate jobs, a dirty
checkout, or a non-Git checkout. One GH200 smoke must pass before all
50-epoch seed jobs are released.

The smoke checks all four data paths but returns no test metric values.
