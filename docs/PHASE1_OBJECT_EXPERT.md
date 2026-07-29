# Phase 1: object-only expert

Phase 1 trains the ImageNet-pretrained `vit_small_patch16_224` object expert
on the complete approved `candidate_train` split.

The selected timm weights declare mean `[0.5, 0.5, 0.5]` and standard
deviation `[0.5, 0.5, 0.5]`. Phase 1 uses those values and verifies them
against `model.pretrained_cfg` before the first optimization step. The same
contract is shared by all SETV ViT-S/16 experts and candidates.

For every jointly transformed image-mask pair, the visible bird stays in its
original position and at its original scale. All non-bird pixels are replaced
in raw RGB space with `(0, 255, 0)`. Normalization happens afterward.

There is no expert-calibration split, temperature scaling, or calibrated
probability in the primary pipeline.

CUDA mixed-precision training advances the learning-rate scheduler only after
`GradScaler` applies the corresponding optimizer update. If non-finite
gradients cause `GradScaler` to skip an update, the scheduler waits. Every
epoch records `train_optimizer_step_count` and
`train_amp_skipped_step_count` in both the CSV metrics and event log.

## Gate

Training refuses to start unless:

- the complete Phase 0 artifact manifest verifies;
- the hash-bound human VLM-mask review is approved;
- an object-expert seed is explicitly supplied;
- the production launcher is run from a clean Git checkout;
- a real one-epoch GH200 smoke succeeds.

## Persistent outputs

The published `object_expert/seed_<seed>/` directory contains:

- `checkpoints/object_expert_final.pt`: final model state only;
- `scores/object_val_scores.npz`: raw logits and true-class margins;
- `scores/object_score_summary.json`;
- `metrics/epoch_metrics.csv`;
- resolved configuration and runtime provenance;
- a Phase 0-bound receipt and artifact hashes.

The score archive contains exactly:

```text
sample_id
true_label
object_logits
object_true_class_margin
object_predicted_class
object_correct
```

No intermediate expert checkpoints are saved.

## Submit on Tigris

The launcher loads the frozen expert seed from
`configs/campaign_waterbirds95.yaml`:

```bash
bash scripts/submit_phase1_object.sh
```

The launcher submits:

```text
real one-epoch GH200 smoke
  -> afterok
20-epoch production training and verification
```
