# Phase 2: exact-fill background expert and SETV fusion

Phase 2 trains the exact-mask green-fill baseline and implements the reusable
hard, rank, and repeated cross-fitted logistic SETV scoring layer.

The transformed VLM foreground mask is dilated by an explicit eight-pixel
Euclidean disk at 224×224. Pixels inside the dilated mask are filled with raw
RGB `(0, 255, 0)`. The standard pretrained ViT-S/16 is trained on the complete
`candidate_train` split with no calibration holdout or temperature scaling.
For non-production fixture resolutions, the radius scales proportionally; it
is exactly eight pixels for the locked 224×224 production input.

Version 2 does not separately state an optimizer schedule for the exact-fill
control. The configuration makes the implementation choice explicit: it uses
the same ordinary single-view ViT schedule as the object expert (20 epochs,
AdamW, learning rate `3e-5`, weight decay `0.05`, batch size 64, cosine decay,
and two warmup epochs). This setting is receipt-bound and should be confirmed
before the first production launch rather than changed after results.

Only the final exact-expert checkpoint is retained.

## Fusion

The object and exact-background raw true-class margins are aligned exactly to
the approved `biased_val` manifest.

- Hard: `m_object > 0 and m_background < 0`.
- Rank: within-class margin ranks and `r_object * (1-r_background)`.
- Logistic: five repeats of out-of-fold logistic fusion, normally five folds.

The logistic scaler and classifier are fitted only on training folds. Every
image receives exactly one held-out score per repetition. If fewer than ten
positive/negative targets are available, the fold count is reduced
conservatively so each fold has at least two examples of each target. If two
folds remain impossible, logistic fusion is marked unavailable.

ROC/PR AUC against the hard target are implementation diagnostics only. They
do not establish robust checkpoint-selection quality.

Continuous fusion outputs are converted to class-conditional percentiles and
support the locked alpha curve `{0.5, 1, 2, 4}`, weighted loss, and per-class
ESS warnings. The artifact also records rank/logistic distribution summaries
and hard-versus-logistic agreement at the diagnostic 0.5 cutoff. Candidate
scoring accepts only aligned original-image candidate logits.

## Submit

The launcher loads all three frozen seeds from
`configs/campaign_waterbirds95.yaml`:

```bash
bash scripts/submit_phase2_exact.sh
```

The dependency chain is:

```text
real one-epoch GH200 smoke
  -> exact-expert 20-epoch training
  -> CPU fusion construction and verification
```
