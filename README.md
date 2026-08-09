# AnchorCal Waterbirds100 pilot

The active research implementation is the independent `anchorcal` package.
It builds controlled foreground-reliance anchors, uses them to select one of
four practical validation criteria, and evaluates that frozen choice on a
six-run ordinary ViT candidate grid. The binding documents are
`AnchorCal_Waterbirds100_Pilot_Implementation_Plan.md` and
`AnchorCal_Implementation_Decision_Locks_Answers.md`; later locks take
precedence.

Copy `configs/anchorcal/paths.local.example.yaml` to the ignored
`configs/anchorcal/paths.local.yaml` and verify the frozen Waterbirds-95 image,
metadata, and VLM-mask paths. The mask input is the audited
OpenCLIP-LAION + DINOvIT `prediction_cmap` bank at
`/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_waterbirds95_openclip_laion_dinovit/val/prediction_cmap`.
AnchorCal joins it producer-first from the complete metadata `img_filename`,
decodes Pascal/VOC foreground class 1, and requires complete official split-0/1
coverage; official split 2 has no mask requirement. Preflight freezes the
one-to-one mapping and per-file hashes in `preflight/mask_manifest.json` with
schema `anchorcal-vlm-mask-manifest-v1`. Then use
`scripts/anchorcal/submit_campaign.sh` from a clean committed TIGRIS checkout.
See `docs/anchorcal/TIGRIS_RUNBOOK.md` and the implementation traceability log
for the exact workflow.

The existing `setv` package below is retained for historical reproducibility;
AnchorCal does not import its expert, selector, or fusion logic.

---

# SETV Waterbirds95 (historical)

This repository implements Spatial Evidence-Tilted Validation (SETV) for the
private Waterbirds95 pilot.

The scientific source of truth is
`SETV_Waterbirds95_Implementation_Plan_v2.md`. The Tigris execution rules are
in `TIGRIS_RESEARCH_COMPUTE_HANDOFF.md`.

The frozen production paths and seeds are in
`configs/campaign_waterbirds95.yaml`. Every production launcher validates and
loads that manifest and writes a fail-closed preflight receipt before calling
Slurm. Follow `docs/TIGRIS_CAMPAIGN_RUNBOOK.md` for the complete staged launch,
monitoring, test-isolation, and recovery procedure.

All `vit_small_patch16_224` experts and candidates use the normalization
declared by the selected timm pretrained weights: mean `[0.5, 0.5, 0.5]` and
standard deviation `[0.5, 0.5, 0.5]`. Runtime preparation verifies this against
the model's `pretrained_cfg`.

## Phase 0

Phase 0 provides:

- immutable joint-group-stratified inner splits;
- selector-safe manifests that omit per-sample place/group labels;
- a separately stored analysis-only protected-label manifest;
- strict VLM-mask discovery, exact VOC decoding, alignment, and
  protocol-aware coverage checks;
- deterministic visual mask-audit sheets and a hash-bound approval receipt;
- joint image-mask transforms;
- configuration, structured logging, hashing, and provenance capture;
- Tigris preflight and submission scripts;
- standard-library unit and integration tests.

The local workspace does not contain the Waterbirds95 images or VLM masks.
Local tests use generated fixtures. The real Phase 0 artifacts must be built
and approved on Tigris.

Run local tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Build Phase 0 on Tigris:

```bash
python -u scripts/build_phase0.py \
  --config configs/data_waterbirds95.yaml
```

Inspect all generated contact sheets, then record human approval:

```bash
python -u scripts/approve_mask_audit.py \
  --phase0-dir /home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/phase0 \
  --reviewer YOUR_NAME \
  --confirm
```

Verify the complete gate:

```bash
python -u scripts/verify_phase0.py \
  --phase0-dir /home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/phase0
```

Phase 0 does not train an expert or candidate model.

## Phase 1

Phase 1 implements the object-only ViT-S/16 expert. It requires an approved
Phase 0 directory. The seed is loaded from the frozen campaign manifest:

```bash
bash scripts/submit_phase1_object.sh
```

The launcher gates 20-epoch production training behind a real one-epoch GH200
smoke. It saves the final auxiliary-expert state and raw biased-validation
logits/margins, but no intermediate expert checkpoints and no calibrated
probabilities. See `docs/PHASE1_OBJECT_EXPERT.md`.

## Phase 2

Phase 2 adds the exact-mask green-fill background expert and the hard, rank,
and repeated cross-fitted logistic fusion/scoring layer. See
`docs/PHASE2_EXACT_AND_FUSION.md`.

## Phase 3

Phase 3 adds deterministic 3/3/2 rectangle, ellipse, and smooth-blob mask
banks; image-held-out logistic, boosted-tree, and binary-CNN leakage auditors;
paired consistency training for the sanitized-fill ViT; eight-view raw-logit
evaluation; and sanitized hard/rank/logistic fusion artifacts. The leakage
gate remains the ordinary production requirement. For this private pilot, the
project owner explicitly authorized the immutable job-21917 rejected bank as
a diagnostic selector signal. The dedicated resume path preserves
`leakage_audit_accepted=false` and
`sanitization_claim_eligible=false` through all downstream receipts. See
`docs/PHASE3_SANITIZED_EXPERT.md`.

## Phase 4

Phase 4 adds the variable-length real-background patch expert. It copies the
ViT-S/16 patch projection, CLS token, first four transformer blocks, and final
LayerNorm; adds only coarse 3-by-3 position embeddings; discards dense
positions; and uses no second pooling module. Validation averages eight
token-dropout views at the raw-logit level before hard, rank, and repeated
cross-fitted logistic fusion. Invalid training crops are deterministically
retried without relaxing the locked 16-background-token minimum; the
fallback chooses the higher-capacity canonical or full-frame view, with all
full-frame padding marked ineligible. The census-approved capacity rule
excludes samples `4887` and `6285` only from this auxiliary expert's training;
candidate ERM and validation remain unchanged. See
`docs/PHASE4_BACKGROUND_SET_TRANSFORMER.md`.

When the locked floor is infeasible on real masks, the read-only
`scripts/submit_phase4_capacity_audit.sh` launcher measures the complete
canonical/full-frame capacity distribution without training or reading
Oracle/test data.

## Phase 5

Phase 5 trains 50-epoch ERM ViT-S/16 candidate trajectories and records the
complete biased-validation prediction curve. It computes ordinary validation,
three hard pseudo-group baselines, and all available exact/sanitized/set hard,
rank, and logistic SETV selectors online. Rolling selected checkpoints are
deduplicated by epoch. Oracle metrics are isolated under `analysis_only`, and
test metrics are not published until the realistic selection receipt has been
hash-frozen. The production launcher requires at least three candidate seeds.
See `docs/PHASE5_CANDIDATE_TRAJECTORY.md`.

## Phase 6

Phase 6 implements the `uLA-style` baseline, joint expert-fusion selection
across at least three candidate seeds, method freezing, diagnostic kill
criteria, and reporting-only test publication. Its launcher gates production
behind a real GH200 uLA compatibility smoke. The smoke either executes one
official MoCoV2+ epoch in the explicitly supplied legacy environment or
verifies the exact external official checkpoint, then runs a real
checkpoint-loading and frozen-linear-proxy update in the confirmed Tigris
environment. There is no silent fallback. See
`docs/PHASE6_ULA_AND_FROZEN_ANALYSIS.md`.
