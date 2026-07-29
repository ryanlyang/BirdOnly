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
4. Reject a transformed view if fewer than 16 background tokens remain.
5. During training, retain a deterministic random 80% subset, subject to the
   16-token minimum.

Random resized crops are valid only when they preserve the locked 16-token
minimum. Training deterministically tries the original crop seed and up to
nine derived retry seeds. If all ten crops are invalid, it evaluates two fixed
views and uses the one with more eligible tokens: the canonical center crop or
the complete aspect-preserving frame. Full-frame letterbox padding is marked
foreground/ineligible before dilation, so no retained token can contain
artificial padding. Validation uses the same deterministic fixed-view rule
when the canonical view is below the floor.

Before model construction, both candidate-train fixed views are measured.
Images for which neither view reaches 16 tokens are excluded only from this
auxiliary expert. Census job 22266 found two such images among 3,836:

- sample `6285`: canonical 9, full-frame 5;
- sample `4887`: canonical 11, full-frame 14.

They remain in ERM candidate training and every other applicable experiment.
The resulting set expert trains on 3,834 images. One exclusion belongs to each
target class. Biased validation has no exclusion; its minimum best-view
capacity is 24. Neither foreground eligibility nor the token minimum is
relaxed, and no patch is duplicated. The checkpoint, smoke report, and final
receipt persist the exclusion policy, counts, sample IDs, labels, and
capacities. Per-epoch metrics record attempted crops, rejected crops, and
canonical/full-frame fallback counts.

The deterministic maximum-token cap and fixed-count dropout are implementation
choices used to make a run exactly reproducible. Their seeds and resolved
settings are persisted.

## Capacity census

If the locked 16-token floor rejects a real image even after the full-frame
fallback, do not repeatedly alter transforms or relax foreground eligibility.
Run the read-only capacity census:

```bash
bash scripts/submit_phase4_capacity_audit.sh
```

The census loads only `candidate_train` and `biased_val`; it never loads
protected group columns, Oracle validation, or test data. For every sample it
records the eligible-token count under:

- the canonical evaluation center crop;
- the full-frame aspect-preserving transform with ineligible padding;
- the better of those two fixed views.

The JSON report contains split and combined distributions, support counts for
every floor from 1 through 16, the largest universally supported floor, and
all sample IDs that fail the configured floor. The companion CSV contains one
row per image. Minimum enforcement is bypassed only to measure capacity; the
census never trains a model, creates a Stage 4 artifact, changes the
configuration, or authorizes a new floor.

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

Validation uses the canonical Phase 0 transform when it meets the cardinality
floor, otherwise the higher-capacity audited fixed view, followed by eight
deterministic token-dropout views. Raw logits are averaged across views once.
The score artifact contains:

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

The launcher loads all three frozen seeds from
`configs/campaign_waterbirds95.yaml`:

```bash
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
