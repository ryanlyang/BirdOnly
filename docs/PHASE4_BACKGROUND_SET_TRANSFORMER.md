# Phase 4: Background Set Transformer

Phase 4 implements the third background expert and repeats the locked hard,
rank, and repeated cross-fitted logistic fusion procedures. It consumes only
an approved Phase 0 artifact and the final Phase 1 object-expert artifact. It
does not train or inspect candidate models.

## Scientific contract

The expert operates on the original, untouched RGB image. A dilated VLM mask
is used only to decide which ViT-S/16 patch tokens are admitted:

1. Dilate the foreground mask with the 8-pixel-at-224 Euclidean disk used by
   the exact expert.
2. Keep a patch only when at most 1% of its pixels are foreground.
3. Deterministically subsample to 180 tokens when necessary.
4. Fail the sample if fewer than 16 background tokens remain.
5. During training, retain a deterministic random 80% subset, subject to the
   16-token minimum.

The deterministic maximum-token cap and fixed-count dropout are implementation
choices used to make a run exactly reproducible. Their seeds and resolved
settings are persisted.

Each retained token receives its pretrained visual patch embedding plus one
learned 3-by-3 coarse spatial-bin embedding. Exact patch coordinates and the
pretrained dense positional embedding are never supplied.

## Clarified architecture

The initial implementation uses the agreed simpler pooling architecture:

1. Copy the pretrained ViT-S/16 patch projection.
2. Copy the pretrained CLS token.
3. Copy the first four pretrained transformer blocks, including attention,
   MLP, and LayerNorm parameters.
4. Prepend CLS to the background-patch set and pass it through those blocks.
5. Apply the copied final LayerNorm and classify the final CLS representation.

The coarse position embeddings and two-class head are initialized from
scratch. There is no second learned attention-pooling module. The original
dense ViT position embeddings, including its fixed-grid locations, are
discarded.

Implementation uses fixed tensor slots for efficient batching, but padded and
foreground tokens are excluded as attention keys in every block. Since tokens
receive only their coarse-bin label, permuting the token order together with
those labels leaves the output unchanged. This is a variable-cardinality set
computation, not a dense-image ViT with hidden exact coordinates.

## Training and validation

Training uses the full `candidate_train` split for 30 epochs:

- AdamW, learning rate `1e-4`, weight decay `0.05`;
- batch size 64;
- cosine schedule with three warmup epochs;
- 20% token dropout;
- the Phase 0 joint image-mask crop and flip;
- cross-entropy with no temperature calibration.

The final auxiliary-expert checkpoint is saved. Intermediate expert
checkpoints are not retained.

Validation uses the canonical Phase 0 transform and eight deterministic
token-dropout views. Raw logits are averaged across views once. The score
artifact contains:

```text
sample_id
true_label
background_set_mean_logits
background_set_true_class_margin
background_set_predicted_class
background_set_correct
background_set_margin_std
```

`background_set_margin_std` is a stability diagnostic across the eight views.
It is not a calibrated uncertainty.

## Fusion

The set-background margin is paired with the frozen Phase 1 object margin.
The common fusion implementation constructs:

- the hard rule `1[m_object > 0 and m_background < 0]`;
- within-class percentile rank fusion;
- five-times repeated, five-fold cross-fitted logistic fusion.

Every image receives one held-out logistic prediction per repetition and the
five predictions are averaged. Logistic ROC/PR AUC is only an implementation
diagnostic for approximating the hard target; it is not evidence of robust
model-selection quality. Candidate selection utility is evaluated in later
phases.

There is no calibration holdout and no temperature scaling.

## Running on Tigris

Freeze all three seeds explicitly:

```bash
export SETV_OBJECT_SEED=...
export SETV_SET_SEED=...
export SETV_FUSION_SEED=...
bash scripts/submit_phase4_set.sh
```

The launcher refuses a dirty or non-Git checkout, missing Phase 0 approval,
missing Phase 1 output, duplicate output directories, and duplicate active
Phase 4 jobs. It submits:

```text
GH200 one-epoch smoke -> GH200 30-epoch training -> CPU fusion
```

The smoke runs the complete test suite before performing a real one-epoch
forward/backward/evaluation pass. The submission receipt records the Git
commit, source hashes, seeds, job dependency chain, and expected destinations.

Candidate logits can later be scored with:

```bash
python -u scripts/score_candidate_set_setv.py \
  --fusion-dir FUSION_DIR \
  --candidate-npz CANDIDATE_LOGITS_NPZ
```

Candidate logits must be from untouched biased-validation images and must
exactly match the Phase 0 sample IDs, labels, and order.
