# AnchorCal Waterbirds100 Pilot
## Full Implementation Plan for Selecting Model-Selection Criteria with Controlled-Reliance ViT Anchors

---

## 0. Executive Summary

This document specifies a complete exploratory pilot for **AnchorCal**, a framework for evaluating and choosing model-selection criteria under spurious correlation.

The pilot is intentionally limited to **Waterbirds100** and one family of ordinary ERM ViT candidates. Its purpose is not yet to support final paper claims. Its purpose is to test the central AnchorCal hypothesis:

> A validation criterion that accurately measures known foreground-versus-background reliance on controlled anchor models should also be a strong criterion for selecting robust ordinary ViT candidates whose true reliance is unknown.

The experiment has two linked parts.

### Part A: Anchor calibration

1. Train a foreground-only ViT branch that can access bird information but no original background information.
2. Train a background-only ViT branch that can access real background patches but no bird information and no explicit mask geometry.
3. Form a differentiable ladder of anchor models:
   \[
   A_\lambda,\qquad \lambda\in\{0,0.05,\ldots,0.95,1\},
   \]
   where \(\lambda\) exactly controls the foreground contribution and \(1-\lambda\) controls the background contribution.
4. Restrict anchor calibration to custom validation images that both extreme branches classify correctly.
5. Evaluate several candidate validation criteria on the anchor ladder.
6. Measure how accurately, monotonically, and stably each criterion recovers the known \(\lambda\) ordering.
7. Use AnchorCal to select the most trustworthy validation criterion without consulting balanced group labels or hidden test performance.

### Part B: Real candidate selection

1. Train a small hyperparameter grid of ordinary full-image ViT classifiers on Waterbirds100.
2. Treat the in-memory model after every training epoch as a candidate.
3. Immediately run every validation criterion after each epoch.
4. Store all scalar and per-example evaluation outputs, but do not save every model checkpoint.
5. Log oracle-validation and test performance every epoch for exploratory post-hoc analysis.
6. Determine whether the criterion chosen by AnchorCal:
   - has strong correlation with hidden test worst-group accuracy;
   - selects a high-performing test model;
   - has low selection regret;
   - performs close to oracle group-aware validation.

This pilot uses **live evaluation after every epoch**. No intermediate candidate checkpoints are required. If a particular epoch must later be inspected, reproduce it from its saved configuration, seed, and epoch number.

---

# 1. Research Question

The broad problem is:

> When validation data preserves the same spurious correlation as training data, how can we determine which model-selection criterion should be trusted?

Several plausible validation criteria may be available:

- ordinary biased-validation accuracy;
- saliency alignment with a foreground mask;
- background token-swap robustness;
- robustness to background blurring;
- foreground-only performance;
- inferred-group methods in later experiments.

The true quality of these criteria cannot be evaluated using the biased validation distribution itself. AnchorCal creates models whose reliance is known by construction, then uses those models as calibration standards for the criteria.

The primary hypothesis is:

> Validation criteria with low error when estimating controlled anchor reliance will also have high ranking quality and low checkpoint-selection regret on ordinary full-image ViT candidates.

---

# 2. Terminology

To prevent confusion, use the following terms consistently.

## Candidate model

A normal full-image ViT trained with ERM. Every epoch of every hyperparameter-grid run is one candidate.

## Validation criterion

A rule that assigns a scalar selection score to a candidate model.

Examples:

- ordinary biased-validation accuracy;
- saliency-alignment score combined with validation accuracy;
- token-swap robustness combined with validation accuracy.

## Foreground branch

A ViT-style classifier that receives bird information but cannot receive original background pixels.

## Background branch

A ViT-style classifier that receives only pure background patches and cannot receive bird pixels or an explicit bird-shaped missingness pattern.

## Anchor model

A differentiable combination of the frozen foreground and background branches:

\[
z_\lambda(x)
=
\lambda \widetilde z_F(x)
+
(1-\lambda)\widetilde z_B(x).
\]

## Anchor ladder

The set of 21 anchors:

\[
\lambda\in\{0,0.05,0.10,\ldots,0.90,0.95,1.00\}.
\]

## Anchor calibration subset

The custom biased-validation images that both extreme branches classify correctly.

## AnchorCal

The meta-evaluation framework that measures each validation criterion on the anchor ladder and determines which criterion most accurately measures controlled reliance.

## Oracle validation

The official balanced Waterbirds validation split, scored using worst-group accuracy. It is hidden from all practical selectors and from AnchorCal.

---

# 3. Pilot Scope

## 3.1 Included

- Waterbirds100 only.
- The fixed, audited Waterbirds-100 VLM bird-mask bank specified in Section 5.1.
- The complete official Waterbirds-100 training split, hard-asserted to contain
  only aligned groups.
- A custom fully correlated validation split drawn from those aligned training images.
- Official Waterbirds validation as a group-aware oracle.
- Official Waterbirds test for exploratory post-hoc analysis.
- One candidate architecture:
  - pretrained ViT-S/16.
- One candidate training algorithm:
  - ERM.
- A small learning-rate and weight-decay grid.
- Live selector evaluation after every epoch.
- No intermediate candidate checkpoint storage.
- Exact differentiable foreground/background anchor ladder.
- Four primary validation criteria:
  1. ordinary biased-validation accuracy;
  2. saliency alignment combined with accuracy;
  3. token-level background swapping combined with accuracy;
  4. background-blur robustness combined with accuracy.
- One optional construction-matched diagnostic:
  - foreground-only accuracy combined with ordinary accuracy.
- AnchorCal metrics:
  - held-out Anchor Calibration Error;
  - Kendall tau-b;
  - Spearman correlation;
  - pairwise ordering accuracy;
  - adjacent ordering accuracy;
  - monotonicity violations;
  - perfect-order rate under bootstrap resampling.
- Real-candidate evaluation:
  - correlation with test WGA and test average accuracy;
  - selected-model performance;
  - test selection regret;
  - oracle-validation selection regret.

## 3.2 Explicitly Deferred

Do not implement the following in this first pilot:

- pairwise competence-matched anchor subsets;
- uLA;
- EVaLS;
- JTT;
- Group DRO;
- robust training methods beyond ERM;
- additional datasets;
- additional spurious-correlation strengths;
- alternative VLM generations, alternative mask sources, or imperfect-mask
  comparisons beyond the fixed audited bank used by this pilot;
- distilled ordinary-ViT anchors;
- multiple independent anchor families;
- learned combinations of validation criteria;
- saliency ensembles;
- pixel-space background swapping;
- artifact correction;
- publication-quality final test protocol.

The pilot should remain narrow enough that failure or success can be interpreted cleanly.

---

# 4. Dataset Construction

## 4.1 Waterbirds100 source pool

Start from official split 0 of the dedicated Waterbirds-100 release:

```text
/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2
```

with authoritative metadata:

```text
/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2/metadata.csv
```

The complete source training split must already satisfy:

\[
y_i = s_i,
\]

where:

- \(y_i\) is the bird class;
- \(s_i\) is the background/place class.

For binary Waterbirds:

- waterbird on water;
- landbird on land.

No official-training row may correspond to:

- waterbird on land;
- landbird on water.

Hard-fail preflight if any split-0 row violates `y == place`; do not silently
filter a partially biased release into an aligned subset. Call the complete,
validated split-0 pool:

```text
waterbirds100_development_pool
```

## 4.2 Custom training and biased-validation split

Split the complete validated development pool into:

- `candidate_train`: 80 percent;
- `biased_val`: 20 percent.

Stratify by bird class. Because every source example is asserted aligned, both
resulting splits remain 100 percent spuriously correlated.

Use one fixed split seed:

```text
split_seed = 1729
```

Save exact sample IDs:

```text
splits/waterbirds100_candidate_train.csv
splits/waterbirds100_biased_val.csv
```

Persist the complete split contract in `splits/manifest.json` with schema
`anchorcal-splits-v3`. It must bind the dedicated release and metadata hash,
the complete official split-0 membership and `y == place` audit, every derived
CSV hash, and the untouched official split-1 and split-2 memberships.

Log:

- sample count;
- class count;
- place count;
- aligned-group count;
- empirical shortcut correlation.

Required assertion:

```text
empirical_correlation == 1.0
```

up to metadata integrity.

## 4.3 Oracle validation

Use official split 1 from the dedicated Waterbirds-100 release unchanged.

This split is the **oracle validation set**.

Primary oracle metric:

\[
\operatorname{OracleWGA}(f)
=
\min_g \operatorname{Acc}_g(f).
\]

Also log:

- average accuracy;
- class-balanced accuracy;
- group-balanced average accuracy;
- all four group accuracies.

The oracle split must never be used to:

- train candidates;
- train branches;
- calibrate branches;
- construct anchors;
- choose validation criteria;
- tune selector formulas.

## 4.4 Test split

Use official split 2 from the dedicated Waterbirds-100 release unchanged.

For this exploratory pilot, it is acceptable to compute test metrics after every epoch.

This is deliberately exploratory and Waterbirds100 is not assumed to be a final reported paper result. Still, maintain clean code separation so test values cannot accidentally enter model selection.

Primary hidden deployment metric:

\[
\operatorname{TestWGA}(f).
\]

Also log test average accuracy.

## 4.5 Expert calibration split

Inside `candidate_train`, reserve 10 percent for branch probability calibration:

```text
expert_calibration_fraction = 0.10
expert_calibration_seed = 2718
```

Stratify by bird class.

Define:

- `expert_train`;
- `expert_calibration`.

The ordinary candidate ViTs may still train on all of `candidate_train`.

The foreground and background branches train on `expert_train` and calibrate on `expert_calibration`.

---

# 5. Segmentation Masks and Shared Image Geometry

## 5.1 Mask assumptions

The authoritative mask input is the matching fixed OpenCLIP-LAION + DINOvIT
WeCLIP+ `prediction_cmap` bank at:

```text
/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_waterbirds100_openclip_laion_dinovit/val/prediction_cmap
```

This is an audited VLM teacher-map bank, not an official CUB ground-truth
segmentation bank. Do not mix it with CUB segmentations, historical WeCLIP+
outputs, or the Waterbirds-95 `results_waterbirds95_*` bank. `Waterbirds100` in
this pilot means the dedicated `waterbird_1.0_forest2water2` release defined in
Section 4.1, paired only with this matching Waterbirds-100 mask bank.

Resolve a mask from the metadata row's complete dataset-relative
`img_filename`. Reproduce the producer's
`generate_pseudo_masks_waterbirds._make_image_id` rule: remove the final image
extension, replace path separators with `_`, replace every run outside
`[A-Za-z0-9_-]` with `_`, strip leading and trailing underscores, and append
`.png`. Reject absolute or escaping metadata paths and flattened-name
collisions. The exact producer name is tried first. Any explicitly supported
legacy layout is only a recorded fallback after the producer name is absent;
zero matches, multiple matches, or one mask resolving to multiple metadata rows
is a preflight failure. Never join by DataFrame row position, class, split-local
position, or data-loader order.

The PNGs are categorical Pascal/VOC color maps. Preserve their categorical
colors, decode exact VOC class IDs, reject unknown or unexpected colors and
class IDs, and define the Boolean bird mask as `class_id == 1`. In this bank,
class 0 is RGB `[0, 0, 0]` and class 1 is RGB `[128, 0, 0]`. Do not treat these
files as normalized RGB images, grayscale heat maps, or binary white
`{0, 255}` masks. All later references to a binary mask mean the Boolean result
of this strict decode.

Complete, one-to-one mask coverage is required for every metadata row in
official splits 0 and 1. Official split 2 has no mask requirement: missing test
masks must not fail preflight, and reporting-only classification on untouched
test RGB images must not try to load masks. Extra PNGs, including separately
generated test maps, may be inventoried but do not change the contract.

Before any training, write the immutable VLM mapping manifest
`preflight/mask_manifest.json`, with schema
`anchorcal-vlm-mask-manifest-v2`, sorted by canonical `img_id`. It must bind the
resolved dataset, metadata hash, exact VLM root, the locked source identifier
`waterbirds100_openclip_laion_dinovit_weclipplus_prediction_cmap`, mapping and
decoder implementation versions, map format, required splits, and the
AnchorCal configuration or checkout revision. It does not claim an unknown
external GALS producer-source revision. For each required row it records at
least `img_id`,
`img_filename`, official split, producer-derived name, resolved mask path,
mapping rule, image and mask dimensions, decoded colors or class IDs,
foreground count or fraction, file size, and mask SHA-256. It also records
coverage by split, missing/ambiguous/collision reports, unused extras, and a
deterministic content hash over the canonically serialized entries. Every
downstream stage verifies the frozen manifest and its hash rather than resolving
masks again inside workers.

## 5.2 Joint image-mask transforms

Every geometric image transform must be applied identically to the mask.

Default image size:

```text
image_size = 224
patch_size = 16
grid_size = 14 x 14
```

Mask resizing:

```text
nearest-neighbor interpolation
```

Candidate training transform:

```text
RandomResizedCrop(224, scale=(0.70, 1.00))
RandomHorizontalFlip(p=0.5)
```

Candidate evaluation transform:

```text
Resize shortest side to 256
CenterCrop(224)
```

Branch training may use the same geometric transforms, but avoid strong color augmentation in the first pilot.

## 5.3 Patch-level mask statistics

For every 16-by-16 patch \(p\), calculate:

\[
r_p
=
\frac{\text{bird-mask pixels inside }p}{256}.
\]

Store the source full-grid patch coordinate for every branch token. This is required to map saliency back to the image grid.

---

# 6. Candidate Model Architecture and Training Grid

## 6.1 Candidate architecture

Use:

```text
timm: vit_small_patch16_224
ImageNet pretrained: true
```

The candidate model receives the ordinary full image and never receives a segmentation mask during training.

## 6.2 Candidate hyperparameter grid

Use a deliberately small grid:

```text
learning_rate ∈ {1e-5, 3e-5, 1e-4}
weight_decay ∈ {0.01, 0.05}
```

Total runs:

```text
3 x 2 = 6
```

Default candidate configuration:

```text
epochs = 40
optimizer = AdamW
batch_size = 64
scheduler = cosine decay
warmup_epochs = 4
label_smoothing = 0.0
drop_path_rate = 0.0
mixed_precision = true
gradient_clipping = 1.0
candidate_seed = 1234
```

This creates:

\[
6\times40=240
\]

candidate states.

A candidate is identified by:

```text
candidate_id = "{lr}_{weight_decay}_{seed}_epoch_{epoch}"
```

## 6.3 Live per-epoch analysis

At the end of every epoch:

1. Keep the current candidate model in memory.
2. Run every validation criterion.
3. Run oracle validation.
4. Run test evaluation.
5. Store all outputs.
6. Continue training.
7. Do not save the intermediate model state.

This is the primary candidate-generation mechanism.

## 6.4 Reproducibility without checkpoint storage

Store:

- exact configuration;
- code commit hash;
- random seed;
- data split IDs;
- epoch number;
- optimizer and scheduler settings;
- all per-epoch metrics.

If a selected candidate must be recovered later, rerun the exact training configuration to the specified epoch.

Optionally save:

- final epoch only;
- best ordinary-validation epoch only.

Do not save all 240 candidates.

---

# 7. Exact Foreground Branch

## 7.1 Design objective

The foreground branch should receive bird evidence while receiving **no original background information**.

Some performance may be sacrificed to preserve interpretability and leakage control.

## 7.2 Input construction

Start from the transformed image \(x\) and binary bird mask \(M\).

Create a green-screen image:

\[
x_F
=
M\odot x + (1-M)\odot c_{\text{green}},
\]

where:

```text
c_green = RGB(0, 255, 0)
```

This removes all original background pixels.

Patchify \(x_F\).

Retain every patch satisfying:

```text
foreground_fraction > 0.0
```

Because original background pixels inside mixed patches were already replaced with green, retained tokens contain:

- bird pixels;
- green pixels;
- no original background pixels.

The bird silhouette is allowed to remain. It is part of foreground geometry.

## 7.3 Positional information

Do not use full-image absolute positional embeddings.

For each retained patch, calculate position relative to the bird bounding box:

\[
u_p
=
\frac{x_p-x_{\min}}{\max(x_{\max}-x_{\min},\epsilon)},
\]

\[
v_p
=
\frac{y_p-y_{\min}}{\max(y_{\max}-y_{\min},\epsilon)}.
\]

Encode \((u_p,v_p)\) using a small learned MLP or fixed Fourier features and add it to the patch embedding.

This permits the branch to understand bird structure while reducing reliance on global photograph position.

For the first pilot, implement both flags:

```text
foreground_position_mode = "object_relative"  # primary
foreground_position_mode = "absolute"         # diagnostic only
```

Use object-relative positions as the primary anchor construction.

## 7.4 Architecture

Use a ViT-style variable-token encoder:

```text
patch embedding dim = 384
transformer depth = 6
attention heads = 6
MLP ratio = 4
dropout = 0.0
attention dropout = 0.0
CLS token = learned
classification head = linear
```

Initialize from pretrained ViT-S/16 when practical:

- reuse patch projection;
- reuse the first six transformer blocks;
- discard pretrained absolute positional embeddings;
- initialize object-relative position encoder separately.

Pad variable-length token sequences inside a batch and use an attention padding mask.

## 7.5 Training objective

Train on `expert_train`.

Use ordinary cross-entropy:

\[
\mathcal L_F
=
\operatorname{CE}(z_F,y).
\]

Default training:

```text
epochs = 30
optimizer = AdamW
learning_rate = 3e-5
weight_decay = 0.05
batch_size = 64
scheduler = cosine
warmup_epochs = 3
```

## 7.6 Calibration

Fit scalar temperature \(T_F\) on `expert_calibration`:

\[
p_F(c\mid x)
=
\operatorname{softmax}(z_F/T_F)_c.
\]

Temperature calibration is used for diagnostics. Anchor logit scaling is handled separately later.

---

# 8. Exact Background Branch

## 8.1 Design objective

The background branch should receive:

- real background pixels;
- no bird pixels;
- no bird-shaped filled hole;
- no exact missing-token geometry;
- no explicit token count signal.

The primary implementation is a position-free ViT-style set encoder over pure background patches.

## 8.2 Safe background mask

Dilate the bird mask before selecting background patches.

Default:

```text
background_mask_dilation_pixels = 8
```

Call the dilated mask \(M^+\).

A patch is eligible as pure background if:

```text
dilated_foreground_fraction == 0.0
```

This is intentionally strict.

## 8.3 Fixed background-token budget

Let the set of eligible background patches be:

\[
\mathcal B(x).
\]

Sample exactly:

```text
K_background_tokens = 64
```

patches from \(\mathcal B(x)\).

Rules:

- if \(|\mathcal B(x)|\geq64\), sample without replacement;
- if \(0<|\mathcal B(x)|<64\), sample with replacement;
- if \(|\mathcal B(x)|=0\), mark the example invalid for branch training and log it.

Randomize token order.

Do not provide:

- exact coordinates;
- coarse coordinates;
- token count;
- occupancy pattern;
- mask geometry.

Because every example contributes exactly 64 tokens and no positions, the branch cannot directly infer the bird silhouette from missing positions.

## 8.4 Evaluation views

The branch input is stochastic because patches are sampled.

For every evaluation image, create:

```text
num_background_views = 8
```

fixed patch-sampling views using deterministic seeds derived from the sample ID.

Average logits across views:

\[
z_B(x)
=
\frac{1}{8}\sum_{r=1}^{8}z_B^{(r)}(x).
\]

These fixed views must be reused across:

- branch calibration;
- anchor construction;
- anchor saliency;
- anchor token swapping.

## 8.5 Architecture

Use the same ViT-style block configuration as the foreground branch:

```text
patch embedding dim = 384
transformer depth = 6
attention heads = 6
MLP ratio = 4
dropout = 0.0
attention dropout = 0.0
CLS token = learned
classification head = linear
positional embeddings = none
```

The transformer is permutation equivariant over background tokens, and the CLS token provides permutation-invariant pooling.

Use independent weights from the foreground branch.

## 8.6 Training objective

Train on `expert_train`:

\[
\mathcal L_B
=
\operatorname{CE}(z_B,y).
\]

At each training iteration, resample background patches.

Default training:

```text
epochs = 30
optimizer = AdamW
learning_rate = 3e-5
weight_decay = 0.05
batch_size = 64
scheduler = cosine
warmup_epochs = 3
```

## 8.7 Calibration

Fit scalar temperature \(T_B\) on the fixed eight-view averaged logits from `expert_calibration`.

---

# 9. Leakage Audits

The anchor construction should prioritize leakage control over maximum branch accuracy.

## 9.1 Foreground branch audit

Because every non-bird pixel is replaced before patch embedding, no original background pixel should reach the branch.

Verify:

1. Re-render the same bird with multiple arbitrary green shades.
2. Confirm predictions are stable.
3. Replace the original background with an unrelated image before green-screening.
4. Confirm outputs are numerically identical after green-screen construction.

Expected result:

```text
max absolute logit difference ≈ numerical precision
```

## 9.2 Background pixel-purity audit

For every retained background patch:

```text
assert dilated_foreground_fraction == 0.0
```

Record:

- minimum distance from retained patches to the raw bird mask;
- number of retained patches;
- number of invalid examples.

## 9.3 Mask-geometry audit

The background branch never sees:

- positions;
- occupancy grid;
- token count.

Still train simple auditors using external geometry features:

- raw mask area;
- bounding-box area;
- centroid;
- aspect ratio;
- perimeter;
- number of pure background patches.

These auditors are not part of the branch. They diagnose whether the dataset itself contains strong geometric class information.

If geometry predicts class well, note it, but the primary background branch is still protected because it receives none of these features and a fixed token count.

## 9.4 Random-token audit

Replace all background patch pixels with random patches drawn independently of class while preserving the fixed token budget.

Background branch accuracy should collapse toward chance.

This verifies that the branch uses background content rather than implementation metadata.

## 9.5 Patch-purity sweep

Later diagnostic values:

```text
dilation ∈ {4, 8, 12}
```

The primary pilot uses 8 pixels.

AnchorCal results should eventually be checked for stability across this sweep, but the first run may use only the default.

---

# 10. Competence-Matched Anchor Calibration Subset

## 10.1 Definition

Run the frozen calibrated foreground and background branches on `biased_val`.

Define:

\[
\mathcal C
=
\left\{
i:
\arg\max_c z_F(x_i)_c=y_i
\quad\text{and}\quad
\arg\max_c z_B(x_i)_c=y_i
\right\}.
\]

This is the only anchor calibration subset in the first pilot.

No pairwise subsets are implemented yet.

## 10.2 Why this subset is used

On \(\mathcal C\):

- the pure foreground anchor is correct;
- the pure background anchor is correct;
- every positive convex mixture of their logits is also correct.

Therefore, ordinary observational accuracy is constant across the ladder.

The anchor task measures reliance, not task competence.

## 10.3 Class balancing

Do not discard examples unnecessarily.

Compute every anchor criterion using class-balanced averaging over \(\mathcal C\):

\[
S(A_\lambda)
=
\frac{1}{C}
\sum_{c=1}^{C}
S_c(A_\lambda).
\]

Log:

- total intersection size;
- percentage of `biased_val`;
- per-class counts;
- foreground accuracy;
- background accuracy;
- lower-bound overlap estimate;
- observed overlap.

Required warning:

```text
warn if any class has fewer than 50 intersection examples
```

For binary Waterbirds100, the official-training pool is expected to be large
enough.

---

# 11. Branch Logit Normalization

The foreground branch may be more accurate and may naturally emit larger margins. Raw logit interpolation would make \(\lambda\) difficult to interpret.

## 11.1 Center logits

For each example:

\[
z_F'(c)
=
z_F(c)-\frac{1}{C}\sum_k z_F(k),
\]

\[
z_B'(c)
=
z_B(c)-\frac{1}{C}\sum_k z_B(k).
\]

This preserves class rankings.

## 11.2 Compute branch margin scales

On \(\mathcal C\), calculate true-class margins:

\[
m_F(i)
=
z_F'(y_i)-\max_{c\neq y_i}z_F'(c),
\]

\[
m_B(i)
=
z_B'(y_i)-\max_{c\neq y_i}z_B'(c).
\]

Both are positive on \(\mathcal C\).

Use robust scales:

\[
s_F
=
\operatorname{median}_{i\in\mathcal C}m_F(i),
\]

\[
s_B
=
\operatorname{median}_{i\in\mathcal C}m_B(i).
\]

Normalize:

\[
\widetilde z_F
=
\frac{z_F'}{s_F+\epsilon},
\]

\[
\widetilde z_B
=
\frac{z_B'}{s_B+\epsilon}.
\]

Default:

```text
epsilon = 1e-8
```

## 11.3 Required diagnostics

After normalization, log:

- median foreground margin;
- median background margin;
- normalized median margins;
- margin histograms;
- fraction of examples where each normalized margin is positive.

Expected normalized medians:

```text
approximately 1.0 for both branches
```

---

# 12. Differentiable Anchor Ladder

## 12.1 Primary anchor family

Use logit mixing:

\[
z_\lambda(x)
=
\lambda \widetilde z_F(x)
+
(1-\lambda)\widetilde z_B(x).
\]

Lambda grid:

```text
lambdas = [0.00, 0.05, 0.10, ..., 0.90, 0.95, 1.00]
```

Total:

```text
21 anchors
```

## 12.2 Reliance guarantee

If only background evidence changes:

\[
\Delta z_\lambda
=
(1-\lambda)\Delta \widetilde z_B.
\]

If only foreground evidence changes:

\[
\Delta z_\lambda
=
\lambda\Delta \widetilde z_F.
\]

Thus structural background sensitivity decreases monotonically with \(\lambda\).

## 12.3 Anchor correctness

For every \(i\in\mathcal C\) and every incorrect class \(c\):

\[
\widetilde z_F(y_i)>\widetilde z_F(c)
\]

and

\[
\widetilde z_B(y_i)>\widetilde z_B(c).
\]

Therefore:

\[
z_\lambda(y_i)>z_\lambda(c)
\]

for all \(\lambda\in[0,1]\).

Required assertion:

```text
every anchor has 100% top-1 accuracy on C
```

up to floating-point tolerance.

## 12.4 Model wrapper

Implement:

```python
class RelianceAnchor(nn.Module):
    def __init__(
        self,
        foreground_branch,
        background_branch,
        lambda_value,
        foreground_scale,
        background_scale,
    ):
        ...
```

Forward input:

```text
image
segmentation_mask
sample_id
optional intervention configuration
```

Forward output:

```text
combined logits
foreground logits
background logits
source patch metadata
```

Both branches remain frozen.

The wrapper must support gradients for:

- saliency;
- token intervention;
- background blur.

## 12.5 Optional secondary anchor family

Do not make this part of the minimum viable pilot, but code the anchor wrapper so probability mixing can be added later:

\[
p_\lambda
=
\lambda p_F+(1-\lambda)p_B.
\]

This provides a low-cost held-out anchor-family diagnostic if the primary pilot succeeds.

---

# 13. Harmonic-Mean Accuracy Coupling

The practical validation criteria should reward both:

- ordinary task competence;
- reduced background reliance.

Use the harmonic mean:

\[
H(a,b)
=
\frac{2ab}{a+b+\epsilon}.
\]

Reasons:

- one weak component cannot be compensated by one strong component;
- all inputs remain in \([0,1]\);
- it is symmetric;
- it avoids manually chosen weights.

For anchor calibration, ordinary anchor accuracy on \(\mathcal C\) is 1.0, so:

\[
H(1,r)
=
\frac{2r}{1+r},
\]

which is monotonic in the reliance score \(r\).

Thus the combined criteria preserve anchor ordering while remaining competence-aware on real candidates.

Also log product variants:

\[
P(a,b)=ab,
\]

but do not treat product and harmonic versions as separate primary AnchorCal criteria in the first pilot. They are monotonic on the anchor ladder and cannot be distinguished meaningfully there.

---

# 14. Primary Validation Criteria

All criterion scores are oriented so **higher is better**.

## 14.1 Criterion 1: Ordinary biased-validation accuracy

For a candidate \(f\):

\[
S_{\mathrm{acc}}(f)
=
A_{\mathrm{biased}}(f).
\]

Use class-balanced accuracy on the full custom biased-validation split.

On anchors restricted to \(\mathcal C\):

\[
S_{\mathrm{acc}}(A_\lambda)=1
\]

for all \(\lambda\).

This is the required negative control.

---

## 14.2 Criterion 2: Gradient saliency alignment

### 14.2.1 Raw attribution

For each image and true class \(y\), hook the candidate model's patch embeddings before transformer contextualization.

Compute:

\[
a_p
=
\operatorname{ReLU}
\left(
\sum_d
\frac{\partial z_y}{\partial h_{p,d}}
h_{p,d}
\right).
\]

If all positive attributions are zero, fall back to:

\[
a_p
=
\left|
\sum_d
\frac{\partial z_y}{\partial h_{p,d}}
h_{p,d}
\right|.
\]

### 14.2.2 Foreground-density alignment

Using pure foreground and pure background patch sets:

\[
d_F
=
\frac{1}{|\mathcal O|}
\sum_{p\in\mathcal O}a_p,
\]

\[
d_B
=
\frac{1}{|\mathcal B|}
\sum_{p\in\mathcal B}a_p.
\]

Define per-image alignment:

\[
r_{\mathrm{sal},i}
=
\frac{d_F}{d_F+d_B+\epsilon}.
\]

This is in \([0,1]\).

Average class-balanced across the selector evaluation set:

\[
R_{\mathrm{sal}}(f).
\]

### 14.2.3 Combined selection score

\[
S_{\mathrm{sal}}(f)
=
H\left(
A_{\mathrm{biased}}(f),
R_{\mathrm{sal}}(f)
\right).
\]

### 14.2.4 Anchor mapping

For a dual-stream anchor, calculate gradients through the complete combined logit.

This automatically scales foreground and background attributions according to \(\lambda\) and branch evidence.

Map every branch token back to its original source patch coordinate.

For duplicated sampled background patches, average attribution at the source coordinate.

---

## 14.3 Criterion 3: Token-level background swap robustness

### 14.3.1 Donor pool

Construct a fixed donor pool from `biased_val`.

For Waterbirds100:

- a waterbird recipient uses landbird-aligned donor backgrounds;
- a landbird recipient uses waterbird-aligned donor backgrounds.

Because correlation is perfect, opposite bird class corresponds to opposite background class.

Use masks to guarantee donor tokens are pure background.

Precompute:

```text
num_donors_per_recipient = 4
donor_assignment_seed = 31415
```

Assignments remain fixed across:

- candidates;
- epochs;
- anchor models.

Exclude self-donors.

### 14.3.2 Candidate standard-ViT swap

For a recipient image:

1. Compute recipient patch embeddings before positional embeddings.
2. Identify pure recipient background patch positions.
3. Compute donor patch embeddings.
4. For each recipient background position:
   - use a pure donor background patch from the same coarse 3-by-3 spatial bin when possible;
   - otherwise use a random pure donor background patch.
5. Insert donor content embedding at the recipient position.
6. Add the recipient positional embedding.
7. Keep recipient bird patch embeddings unchanged.
8. Run the transformer blocks and head.

Primary swap fraction:

```text
swap_fraction = 1.0
```

Swap all pure recipient background positions.

Mixed boundary patches remain unchanged.

### 14.3.3 Anchor swap

For the anchor wrapper:

- leave the foreground branch unchanged;
- replace the background branch's sampled recipient background tokens with pure tokens sampled from the fixed opposite-class donor;
- retain the same number of background tokens;
- run the frozen background branch and combine logits normally.

### 14.3.4 Swap accuracy

For each recipient and donor view, compute top-1 correctness.

Define class-balanced swapped accuracy:

\[
A_{\mathrm{swap}}(f).
\]

Average over donors and examples.

### 14.3.5 Combined selection score

\[
S_{\mathrm{swap}}(f)
=
H\left(
A_{\mathrm{biased}}(f),
A_{\mathrm{swap}}(f)
\right).
\]

Also log:

- mean true-class margin drop;
- prediction flip rate;
- donor-to-donor variance.

These are diagnostics, not the primary score.

---

## 14.4 Criterion 4: Background-blur robustness

This is a plausible but intentionally simple intervention baseline.

### 14.4.1 Image construction

For each image:

\[
x_{\sigma}
=
M\odot x
+
(1-M)\odot \operatorname{GaussianBlur}_{\sigma}(x).
\]

Use blur strengths:

```text
sigma ∈ {2, 4, 8}
```

Preserve bird pixels.

The first pilot may use a hard binary mask edge. Artifact correction is not part of this study.

### 14.4.2 Blur accuracy

Compute class-balanced accuracy for every blur strength:

\[
A_{\mathrm{blur},\sigma}(f).
\]

Average:

\[
A_{\mathrm{blur}}(f)
=
\frac{1}{3}
\sum_{\sigma\in\{2,4,8\}}
A_{\mathrm{blur},\sigma}(f).
\]

### 14.4.3 Combined score

\[
S_{\mathrm{blur}}(f)
=
H\left(
A_{\mathrm{biased}}(f),
A_{\mathrm{blur}}(f)
\right).
\]

### 14.4.4 Anchor blur

Foreground branch input remains unchanged after green-screening.

Background branch receives patches extracted from the blurred background.

---

## 14.5 Optional construction-matched control: foreground-only accuracy

This criterion is useful as a reviewer-oriented diagnostic but should not be eligible for the primary AnchorCal winner in the first analysis.

Construct green-screen images using the segmentation mask.

Compute candidate accuracy:

\[
A_{\mathrm{fg-only}}(f).
\]

Combine:

\[
S_{\mathrm{fg-only}}(f)
=
H\left(
A_{\mathrm{biased}}(f),
A_{\mathrm{fg-only}}(f)
\right).
\]

Why it is a diagnostic:

- it closely resembles the foreground anchor construction;
- if AnchorCal automatically favors it, the benchmark may be construction-matched rather than generally informative.

Report it separately as:

```text
anchor-matched control
```

---

# 15. Selector Evaluation Subsets and Compute Control

## 15.1 Full biased-validation metrics

Compute on all `biased_val` images:

- ordinary accuracy;
- ordinary loss.

## 15.2 Expensive criterion subset

Saliency, token swapping, and blur are more expensive.

Create one fixed, class-balanced subset:

```text
selector_eval_per_class = 256
selector_eval_seed = 16180
```

If a class contains fewer than 256 validation images, use all available examples from that class.

Call this:

```text
selector_eval_subset
```

Use the exact same sample IDs for:

- every candidate;
- every epoch;
- every criterion;
- all six grid runs.

## 15.3 Anchor criterion subset

Use all of \(\mathcal C\), class-balanced through weights.

If \(\mathcal C\) exceeds 1024 images, cap at:

```text
512 per class
```

with a fixed seed.

---

# 16. AnchorCal Evaluation

## 16.1 Criterion score vectors

For every criterion \(S_j\), compute:

\[
\mathbf s_j
=
[
S_j(A_0),
S_j(A_{0.05}),
\ldots,
S_j(A_1)
].
\]

The known target is:

\[
\boldsymbol\lambda
=
[
0,0.05,\ldots,1
].
\]

## 16.2 Kendall tau-b

Primary ordering diagnostic:

\[
\tau_j
=
\tau_b(\mathbf s_j,\boldsymbol\lambda).
\]

Tau-b is required because some criteria may tie.

Higher is better.

## 16.3 Spearman correlation

\[
\rho_j
=
\operatorname{Spearman}(\mathbf s_j,\boldsymbol\lambda).
\]

Higher is better.

## 16.4 Pairwise ordering accuracy

For every pair \(\lambda_a<\lambda_b\), check:

\[
S_j(A_{\lambda_a}) < S_j(A_{\lambda_b}).
\]

Define:

\[
\operatorname{PairAcc}_j
=
\frac{\text{correctly ordered pairs}}
{\binom{21}{2}}.
\]

Ties count as 0.5.

## 16.5 Adjacent ordering accuracy

Only compare:

\[
(0,0.05),
(0.05,0.10),
\ldots,
(0.95,1.00).
\]

Define:

\[
\operatorname{AdjAcc}_j.
\]

This measures local reliance resolution.

## 16.6 Monotonicity violations

Count:

\[
V_j
=
\#\{
a<b:
S_j(A_{\lambda_a})>S_j(A_{\lambda_b})
\}.
\]

Lower is better.

## 16.7 Held-out Anchor Calibration Error

Ordering may saturate. A stronger question is whether the criterion can estimate the reliance level.

Fit a monotonic isotonic mapping:

\[
g_j:S_j(A_\lambda)\mapsto\lambda.
\]

Use two-fold alternating-lambda cross-fitting.

### Fold 1

Train on:

```text
lambda = {0.00, 0.10, 0.20, ..., 1.00}
```

Test on:

```text
lambda = {0.05, 0.15, 0.25, ..., 0.95}
```

### Fold 2

Reverse train and test sets.

Define:

\[
\operatorname{ACE}_j
=
\frac{1}{21}
\sum_\lambda
|\widehat\lambda_j-\lambda|.
\]

Lower is better.

This is the primary AnchorCal quality metric.

## 16.8 Bootstrap stability

Use:

```text
num_bootstrap_replicates = 200
```

Stratified-resample anchor calibration images within class.

For each replicate, recompute:

- criterion score vector;
- ACE;
- Kendall tau-b;
- PairAcc;
- AdjAcc;
- violation count.

Report:

- mean;
- standard deviation;
- 95 percent percentile interval.

## 16.9 Perfect-order rate

For each bootstrap replicate, record whether:

\[
S_j(A_0)
\leq
S_j(A_{0.05})
\leq
\cdots
\leq
S_j(A_1)
\]

with at least one strict increase.

Define:

\[
\operatorname{PerfectOrderRate}_j.
\]

This directly answers:

> How often does the criterion place every anchor in the correct order?

## 16.10 AnchorCal winner and credible set

Primary winner:

\[
j^\star
=
\arg\min_j \operatorname{ACE}_j.
\]

Use eligible criteria only:

- ordinary accuracy;
- saliency;
- token swap;
- blur.

Do not allow the construction-matched foreground-only control to win the primary analysis.

Define the credible criterion set using a one-standard-error rule:

\[
\mathcal S_{\mathrm{credible}}
=
\left\{
j:
\operatorname{ACE}_j
\leq
\operatorname{ACE}_{\mathrm{best}}
+
\operatorname{SE}(\operatorname{ACE}_{\mathrm{best}})
\right\}.
\]

Tie-breaking among statistically indistinguishable criteria:

1. higher AdjAcc;
2. higher PairAcc;
3. lower bootstrap ACE standard deviation;
4. lower computational cost.

Do not create an arbitrary composite AnchorCal score in the first pilot.

---

# 17. Live Candidate Evaluation

## 17.1 Evaluation sequence after each epoch

For every candidate training run and epoch:

1. Set model to evaluation mode.
2. Compute full biased-validation logits.
3. Compute full biased-validation accuracy and loss.
4. Compute saliency criterion on `selector_eval_subset`.
5. Compute token-swap criterion on `selector_eval_subset`.
6. Compute blur criterion on `selector_eval_subset`.
7. Optionally compute foreground-only diagnostic.
8. Compute official oracle-validation metrics.
9. Compute official test metrics.
10. Store all outputs.
11. Return model to training mode.
12. Continue to the next epoch.

## 17.2 Do not save intermediate weights

The model state after the epoch is discarded after evaluation.

Store enough information to reproduce it.

## 17.3 Per-example cache

For each candidate state, save compact arrays:

### Biased validation

- sample ID;
- true label;
- original logits;
- correct flag;
- cross-entropy;
- saliency alignment per image;
- averaged swapped logits;
- swapped correctness per donor;
- blurred logits per sigma.

### Oracle validation

Per-group scalar metrics are enough for the first pilot.

### Test

Per-group scalar metrics are enough for the first pilot.

Per-example caches are encouraged if storage is not a concern.

## 17.4 Expected storage

With roughly 240 candidates and approximately 1,000 validation examples, storing two-class logits and several scalar diagnostics is small compared with model checkpoints.

---

# 18. Candidate-Level Selection Metrics

For each candidate state \(f_k\), calculate:

```text
score_accuracy
score_saliency
score_token_swap
score_blur
score_foreground_only_control
oracle_val_wga
oracle_val_average_accuracy
test_wga
test_average_accuracy
```

All practical criterion scores are computed without oracle/test information.

## 18.1 Criterion-selected candidate

For criterion \(S_j\):

\[
k_j^\star
=
\arg\max_k S_j(f_k).
\]

## 18.2 Tie-breaking

For all practical criteria:

1. higher criterion score;
2. higher full biased-validation accuracy;
3. lower full biased-validation cross-entropy;
4. earlier epoch;
5. lower learning rate;
6. lower weight decay.

Numerical tolerance:

```text
tie_tolerance = 1e-8
```

The earlier epoch preference avoids choosing a later equally scored model without evidence.

## 18.3 Oracle-selected candidate

Select by:

1. highest official-validation WGA;
2. highest group-balanced official-validation accuracy;
3. highest official-validation average accuracy;
4. earlier epoch.

---

# 19. Real Validation-Criterion Quality

Every practical criterion is evaluated in two ways.

## 19.1 Ranking quality

Across all 240 candidate states, compute correlation with:

- test WGA;
- test average accuracy;
- oracle-validation WGA.

Report:

- Pearson;
- Spearman;
- Kendall tau-b.

Primary real-model ranking target:

```text
test WGA
```

Primary correlation:

```text
Spearman
```

## 19.2 Within-run correlation

Epochs from the same hyperparameter run are not independent.

Also compute correlation separately inside each of the six runs and report:

- mean within-run Spearman;
- standard deviation across runs.

## 19.3 Competent-pool correlation

Define:

\[
\mathcal F_{\mathrm{competent}}
=
\left\{
f:
A_{\mathrm{biased}}(f)
\geq
A_{\max}-0.01
\right\}.
\]

Compute criterion correlations again on this pool.

This asks whether the criterion distinguishes similarly competent models.

Even though the practical criteria already include accuracy through a harmonic mean, this analysis isolates robustness resolution.

## 19.4 Selection quality

For each criterion, report the selected candidate's:

- hyperparameters;
- epoch;
- biased-validation accuracy;
- oracle-validation WGA;
- test WGA;
- test average accuracy.

## 19.5 Test selection regret

Exploratory:

\[
\operatorname{TestRegret}(S_j)
=
\max_k \operatorname{TestWGA}(f_k)
-
\operatorname{TestWGA}(f_{k_j^\star}).
\]

Lower is better.

## 19.6 Oracle-validation selection regret

\[
\operatorname{OracleRegret}(S_j)
=
\max_k \operatorname{OracleWGA}(f_k)
-
\operatorname{OracleWGA}(f_{k_j^\star}).
\]

Lower is better.

---

# 20. Evaluating AnchorCal Itself

AnchorCal is successful if anchor-derived criterion quality predicts real criterion quality.

## 20.1 Chosen-criterion test

Let:

\[
j^\star
=
\arg\min_j \operatorname{ACE}_j.
\]

Report:

- the real test-WGA Spearman rank of \(j^\star\) among criteria;
- the real test-regret rank of \(j^\star\);
- the selected candidate's test WGA;
- distance to the hindsight-best practical criterion;
- distance to oracle validation.

## 20.2 Credible-set coverage

Check whether the hindsight-best real criterion belongs to:

\[
\mathcal S_{\mathrm{credible}}.
\]

Report separately for:

- best real Spearman with test WGA;
- lowest test selection regret.

## 20.3 Across-criterion meta-correlation

Across eligible criteria, compute:

\[
\operatorname{Spearman}
\left(
-\operatorname{ACE}_j,
\operatorname{RealSpearman}_j
\right),
\]

and:

\[
\operatorname{Spearman}
\left(
-\operatorname{ACE}_j,
-\operatorname{TestRegret}_j
\right).
\]

With only four eligible criteria, interpret these descriptively rather than as strong statistical evidence.

## 20.4 Success levels

### Level 1: Anchor discrimination

At least one criterion has:

```text
Kendall tau-b >= 0.8
ACE <= 0.10
```

and criteria differ substantially.

### Level 2: Correct criterion family

AnchorCal places the hindsight-best real criterion inside its credible set.

### Level 3: Correct winner

AnchorCal's minimum-ACE criterion is also the best or near-best criterion by:

- test-WGA ranking;
- test selection regret.

### Level 4: Near-oracle candidate selection

AnchorCal's chosen criterion selects a candidate close to the oracle-validation-selected candidate in test WGA.

---

# 21. Primary Tables

## 21.1 Branch table

| Branch | Biased-val accuracy | Calibration NLL | Intersection contribution | Notes |
|---|---:|---:|---:|---|
| Foreground | | | | |
| Background | | | | |

## 21.2 AnchorCal criterion table

| Criterion | ACE ↓ | Kendall ↑ | Spearman ↑ | PairAcc ↑ | AdjAcc ↑ | Violations ↓ | Perfect-order rate ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation accuracy | | | | | | | |
| Saliency HM | | | | | | | |
| Token-swap HM | | | | | | | |
| Blur HM | | | | | | | |
| Foreground-only control | | | | | | | |

## 21.3 Real candidate criterion table

| Criterion | Test-WGA Spearman ↑ | Mean within-run Spearman ↑ | Competent-pool Spearman ↑ | Selected test WGA ↑ | Test regret ↓ | Selected oracle WGA ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Validation accuracy | | | | | | |
| Saliency HM | | | | | | |
| Token-swap HM | | | | | | |
| Blur HM | | | | | | |
| Oracle validation | | | | | | |

## 21.4 AnchorCal decision table

| AnchorCal choice | Credible set | Best real correlation criterion | Lowest-regret criterion | Chosen model test WGA | Oracle-selected test WGA |
|---|---|---|---|---:|---:|

---

# 22. Primary Figures

1. **Anchor score versus lambda**
   - one line per criterion;
   - bootstrap confidence band.

2. **Predicted lambda versus true lambda**
   - cross-fitted isotonic predictions;
   - one panel per criterion.

3. **Criterion score versus test WGA**
   - scatter across all candidates;
   - one figure per criterion.

4. **Selected model comparison**
   - bar chart of selected test WGA by criterion.

5. **Candidate trajectories**
   - epoch versus test WGA;
   - overlay selected epoch from each criterion.

6. **AnchorCal quality versus real criterion quality**
   - negative ACE versus real Spearman;
   - negative ACE versus negative regret.

7. **Competence-matched subset diagnostics**
   - branch accuracies;
   - intersection size;
   - per-class intersection counts.

---

# 23. Required Cached Files

## 23.1 Branch outputs

```text
outputs/branches/foreground_biased_val.npz
outputs/branches/background_biased_val.npz
outputs/branches/foreground_calibration.npz
outputs/branches/background_calibration.npz
```

Fields:

```text
sample_id
label
raw_logits
calibrated_logits
correct
source_patch_metadata
```

## 23.2 Anchor outputs

```text
outputs/anchors/anchor_scores.csv
outputs/anchors/anchor_bootstrap_metrics.csv
outputs/anchors/anchor_per_image_metrics.npz
```

## 23.3 Candidate outputs

```text
outputs/candidates/run_<run_id>/epoch_metrics.csv
outputs/candidates/run_<run_id>/biased_val_outputs_epoch_<epoch>.npz
outputs/candidates/all_candidates.csv
```

The per-epoch NPZ may be replaced by one chunked HDF5 or Zarr file.

## 23.4 Final analysis

```text
outputs/analysis/anchorcal_summary.csv
outputs/analysis/criterion_real_quality.csv
outputs/analysis/selected_candidates.csv
outputs/analysis/tables/
outputs/analysis/figures/
```

---

# 24. Suggested Repository Layout

```text
anchorcal/
├── configs/
│   ├── waterbirds100.yaml
│   ├── foreground_branch.yaml
│   ├── background_branch.yaml
│   ├── candidate_grid.yaml
│   └── criteria.yaml
├── src/
│   ├── data/
│   │   ├── waterbirds.py
│   │   ├── split_waterbirds100.py
│   │   ├── masks.py
│   │   └── joint_transforms.py
│   ├── models/
│   │   ├── candidate_vit.py
│   │   ├── foreground_region_vit.py
│   │   ├── background_set_vit.py
│   │   ├── reliance_anchor.py
│   │   └── hooks.py
│   ├── training/
│   │   ├── train_branch.py
│   │   ├── train_candidate_grid.py
│   │   ├── calibration.py
│   │   └── live_evaluation.py
│   ├── criteria/
│   │   ├── accuracy.py
│   │   ├── saliency.py
│   │   ├── token_swap.py
│   │   ├── background_blur.py
│   │   ├── foreground_only.py
│   │   └── combine.py
│   ├── anchorcal/
│   │   ├── build_ladder.py
│   │   ├── evaluate_criteria.py
│   │   ├── isotonic_calibration.py
│   │   ├── bootstrap.py
│   │   └── choose_criterion.py
│   ├── evaluation/
│   │   ├── group_metrics.py
│   │   ├── correlations.py
│   │   ├── selection_regret.py
│   │   └── reporting.py
│   └── utils/
│       ├── seeds.py
│       ├── logging.py
│       └── serialization.py
├── scripts/
│   ├── create_waterbirds100_split.py
│   ├── train_foreground_branch.py
│   ├── train_background_branch.py
│   ├── build_anchor_ladder.py
│   ├── run_anchorcal.py
│   ├── train_candidate_grid_live.py
│   └── analyze_pilot.py
└── outputs/
```

---

# 25. Pseudocode

## 25.1 Anchor construction

```python
class RelianceAnchor(nn.Module):
    def __init__(
        self,
        foreground_branch,
        background_branch,
        lambda_value,
        foreground_margin_scale,
        background_margin_scale,
    ):
        super().__init__()
        self.foreground_branch = foreground_branch.eval()
        self.background_branch = background_branch.eval()
        self.lambda_value = float(lambda_value)
        self.foreground_margin_scale = foreground_margin_scale
        self.background_margin_scale = background_margin_scale

        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def center_logits(logits):
        return logits - logits.mean(dim=-1, keepdim=True)

    def forward(self, image, mask, sample_ids, intervention=None):
        z_fg, fg_metadata = self.foreground_branch(
            image=image,
            mask=mask,
            sample_ids=sample_ids,
            intervention=intervention,
        )

        z_bg, bg_metadata = self.background_branch(
            image=image,
            mask=mask,
            sample_ids=sample_ids,
            intervention=intervention,
        )

        z_fg = self.center_logits(z_fg) / self.foreground_margin_scale
        z_bg = self.center_logits(z_bg) / self.background_margin_scale

        lam = self.lambda_value
        logits = lam * z_fg + (1.0 - lam) * z_bg

        return {
            "logits": logits,
            "foreground_logits": z_fg,
            "background_logits": z_bg,
            "foreground_metadata": fg_metadata,
            "background_metadata": bg_metadata,
        }
```

## 25.2 Harmonic combination

```python
def harmonic_mean(a: float, b: float, eps: float = 1e-8) -> float:
    return (2.0 * a * b) / (a + b + eps)
```

## 25.3 AnchorCal cross-fitted ACE

```python
def cross_fitted_anchor_calibration_error(scores, lambdas):
    even = np.arange(len(lambdas)) % 2 == 0
    odd = ~even

    predictions = np.empty_like(lambdas, dtype=float)

    model_even = IsotonicRegression(
        increasing=True,
        out_of_bounds="clip",
    )
    model_even.fit(scores[even], lambdas[even])
    predictions[odd] = model_even.predict(scores[odd])

    model_odd = IsotonicRegression(
        increasing=True,
        out_of_bounds="clip",
    )
    model_odd.fit(scores[odd], lambdas[odd])
    predictions[even] = model_odd.predict(scores[even])

    ace = np.mean(np.abs(predictions - lambdas))
    return ace, predictions
```

## 25.4 Live candidate training

```python
for config in candidate_grid:
    set_all_seeds(config.seed)
    model, optimizer, scheduler = initialize_candidate(config)

    for epoch in range(1, config.epochs + 1):
        train_one_epoch(model, train_loader, optimizer)
        scheduler.step()

        metrics = {}

        metrics.update(
            evaluate_full_biased_validation(
                model=model,
                loader=biased_val_loader,
            )
        )

        metrics.update(
            evaluate_saliency_criterion(
                model=model,
                loader=selector_eval_loader,
                masks=selector_eval_masks,
            )
        )

        metrics.update(
            evaluate_token_swap_criterion(
                model=model,
                loader=selector_eval_loader,
                donor_assignments=fixed_donor_assignments,
            )
        )

        metrics.update(
            evaluate_blur_criterion(
                model=model,
                loader=selector_eval_loader,
                blur_sigmas=[2, 4, 8],
            )
        )

        metrics.update(
            evaluate_oracle_validation(
                model=model,
                loader=oracle_val_loader,
            )
        )

        metrics.update(
            evaluate_test(
                model=model,
                loader=test_loader,
            )
        )

        save_epoch_metrics(config, epoch, metrics)
```

---

# 26. Compute and Efficiency Notes

## 26.1 Expected expensive operations

Per candidate epoch:

- one full biased-validation forward pass;
- one saliency backward pass per selector-evaluation batch;
- four token-swap donor forward passes per example;
- three blur forward passes per example;
- one oracle-validation forward pass;
- one test forward pass.

## 26.2 Optimization

- Use mixed precision for all forward evaluations except saliency if numerical instability occurs.
- Batch donor swaps by stacking donor views.
- Precompute:
  - segmentation patch masks;
  - donor assignments;
  - donor source patch indices;
  - blur masks;
  - selector subset.
- Keep candidate patch embeddings only within one evaluation call. Do not store computational graphs between batches.
- Use `torch.inference_mode()` for all criteria except saliency.
- Empty CUDA cache only between full grid runs, not every batch.

## 26.3 Debug mode

Before the full grid:

```text
1 learning rate
1 weight decay
3 candidate epochs
64 selector examples
2 swap donors
1 blur strength
5 anchor lambdas
```

Verify the complete pipeline end to end.

---

# 27. Sanity Checks and Kill Criteria

## 27.1 Branch sanity

Proceed only if:

- foreground branch accuracy is meaningfully above chance;
- background branch accuracy is meaningfully above chance;
- anchor intersection contains enough examples;
- all 21 anchors are correct on the intersection.

## 27.2 Anchor criterion sanity

Expected:

- ordinary accuracy is flat;
- token swapping increases with lambda;
- saliency alignment generally increases with lambda;
- blur robustness may increase but less strongly.

If every criterion is flat, intervention implementation is likely broken.

If every criterion is perfect, the anchor task may be too easy. This is still informative, but later anchor families will be required.

## 27.3 Real candidate diversity

The candidate grid must produce meaningful variation in:

- oracle WGA;
- test WGA;
- validation criteria.

If every candidate is nearly identical, the meta-selection test is uninformative.

Required diagnostic:

```text
test WGA range across candidates
oracle WGA range across candidates
```

## 27.4 Pilot failure conditions

The core pilot is not supported if:

1. no criterion can resolve the anchor ladder;
2. AnchorCal criterion quality has no relationship to real criterion quality;
3. AnchorCal chooses a criterion with poor real selection regret;
4. candidate WGA variation is too small;
5. anchor results are dominated by branch logit scale;
6. background branch leakage audits fail;
7. the competence-matched intersection is too small.

---

# 28. Locked Default Configuration

The corrected implementation package is version `0.4.0`, and this configuration
uses schema `anchorcal-config-v2`.

```yaml
schema_version: anchorcal-config-v2

dataset:
  name: waterbirds100
  release: waterbird_1.0_forest2water2
  split_manifest_schema: anchorcal-splits-v3
  train_fraction: 0.80
  biased_val_fraction: 0.20
  split_seed: 1729
  image_size: 224
  patch_size: 16

masks:
  source: waterbirds100_openclip_laion_dinovit_weclipplus_prediction_cmap
  manifest_schema: anchorcal-vlm-mask-manifest-v2

foreground_branch:
  architecture: region_vit
  embed_dim: 384
  depth: 6
  heads: 6
  position_mode: object_relative
  background_fill_rgb: [0, 255, 0]
  epochs: 30
  lr: 3.0e-5
  weight_decay: 0.05

background_branch:
  architecture: position_free_set_vit
  embed_dim: 384
  depth: 6
  heads: 6
  mask_dilation_pixels: 8
  token_budget: 64
  eval_views: 8
  epochs: 30
  lr: 3.0e-5
  weight_decay: 0.05

anchors:
  lambda_start: 0.0
  lambda_stop: 1.0
  lambda_step: 0.05
  mixing: normalized_logits

candidate_grid:
  architecture: vit_small_patch16_224
  pretrained: true
  learning_rates: [1.0e-5, 3.0e-5, 1.0e-4]
  weight_decays: [0.01, 0.05]
  epochs: 40
  seed: 1234

selector_eval:
  examples_per_class: 256
  swap_donors: 4
  blur_sigmas: [2, 4, 8]

anchorcal:
  bootstrap_replicates: 200
  primary_metric: ACE
  secondary_metric: adjacent_pair_accuracy
  credible_set_rule: one_standard_error
```

---

# 29. Final Pilot Question

The implementation is successful if it can answer this question unambiguously:

> Among ordinary validation accuracy, saliency alignment, token-level background swapping, and background-blur robustness, does AnchorCal identify the criterion that best ranks and selects robust ordinary ViT candidates on hidden Waterbirds test performance?

The most important outputs are not merely the anchor curves.

The decisive chain is:

\[
\text{Known anchor reliance}
\rightarrow
\text{AnchorCal criterion quality}
\rightarrow
\text{Real candidate ranking quality}
\rightarrow
\text{Selected model test robustness}.
\]

If that chain holds in this minimal Waterbirds100 pilot, the project has a strong foundation for:

- additional datasets;
- additional correlation strengths;
- group-inference criteria;
- distilled standard-ViT anchors;
- additional anchor families;
- alternative mask sources and imperfect-mask studies;
- a full benchmark of model-selection criteria.
