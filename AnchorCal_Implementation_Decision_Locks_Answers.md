# AnchorCal Waterbirds100 Pilot
## Implementation Decision Locks and Answers to All 112 Pre-Implementation Questions

This document is the authoritative decision addendum for the AnchorCal Waterbirds100 pilot. It answers every question in the implementation-review inventory, preserves the original numbering, adds final residual locks 97 through 112, and resolves each ambiguity into an implementation lock.

Where this addendum conflicts with the earlier implementation plan, **this addendum takes precedence**.

The overall policy is:

- Prefer purity and interpretability over squeezing out a small amount of branch accuracy.
- Never silently repair, omit, or reinterpret data.
- Freeze every stochastic choice that affects model comparisons.
- Keep the pilot exploratory, but preserve enough procedural separation that the result can be audited later.
- Do not invent cluster paths that have not actually been established.

---

# Summary of Material Overrides

Most of the reviewer recommendations are accepted. The main clarifications or overrides are:

1. Source masks are the locked Waterbirds-100 VLM `prediction_cmap` PNGs and
   use strict categorical VOC decoding with foreground class ID 1.
2. Empty stochastic foreground crops use a bounded rejection sampler and a logged deterministic fallback. Excessive fallbacks abort the run.
3. The background branch does **not** sample with replacement in the primary design. The token budget falls back from 64 to 48 or 32 under a prespecified coverage rule.
4. The foreground and background branches freeze at fixed epoch 30. The calibration split is not used for branch checkpoint selection.
5. Saliency uses only fully foreground patches and fully safe background patches.
6. The blur baseline uses mask-normalized background-only convolution so bird color cannot bleed into the blurred background.
7. A small set of rolling best candidate checkpoints is retained, rather than saving all epochs or saving none.
8. The pilot creates an AnchorCal decision receipt before the final joined analysis, but this remains an exploratory study rather than a strict blinded preregistration.
9. The exact Waterbirds-100 image, metadata, and VLM-mask roots on TIGRIS are
   established below and must be frozen in the local path configuration.
10. HDF5 is the primary candidate-output storage backend.
11. The originally pinned empty Hugging Face revision is replaced by a populated, hash-verified revision.
12. Eight-view saliency uses summed signed occurrence contributions at repeated source coordinates, avoiding unintended double averaging.
13. Candidate and anchor token swapping share donor image IDs but use architecture-appropriate fixed token assignments.
14. All remaining seeds, optimizer groups, scheduler timing, debug staging, and storage transaction rules are explicitly frozen.

---

# Authoritative Waterbirds100 Dataset and VLM-Mask Correction

This section is the latest binding correction. It supersedes every conflicting
Waterbirds-95-subset, CUB-mask, no-VLM, binary-source-encoding,
relative-CUB-stem, and test-mask requirement in the implementation plan and in
earlier answers below. In particular, it overrides the implementation plan's
Sections 3.1, 3.2, and 5.1
and closing mask-source outlook; Summary items 1 and 9; Questions 2, 4, 89, and
the mask-mapping portion of Question 108; the mask fields in the Final Resolved
Configuration Snapshot; and the mask entries under Remaining
Environment-Specific Preflight Items. Non-conflicting requirements remain
binding.

This incompatible correction is versioned as AnchorCal package `0.4.0`,
resolved configuration schema `anchorcal-config-v2`, VLM-mask manifest schema
`anchorcal-vlm-mask-manifest-v2`, and split manifest schema
`anchorcal-splits-v3`. Older schema artifacts cannot satisfy this campaign.

The authoritative dataset is the dedicated Waterbirds-100 release:

```text
/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2
```

Its entire official split 0 is the development source and must already satisfy
`y == place` for every row. Preflight must hard-fail on any counterexample; it
must not silently create Waterbirds100 by filtering the partially biased
Waterbirds-95 release. Official split 1 is the untouched oracle-validation set,
and official split 2 is the reporting-only test set. Canonical sample identity
remains metadata `img_id`; DataFrame row position is never an identity or join
key.

The authoritative mask source is the fixed OpenCLIP-LAION + DINOvIT WeCLIP+
VLM `prediction_cmap` bank:

```text
/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_waterbirds100_openclip_laion_dinovit/val/prediction_cmap
```

This matching Waterbirds-100 bank is an audited pilot input, not official CUB
ground truth. Do not mix it with CUB segmentations, historical WeCLIP+ output
trees, the Waterbirds-95 VLM root, or another mask family.

Join each metadata row from its complete dataset-relative `img_filename`.
Reproduce `generate_pseudo_masks_waterbirds._make_image_id`: reject absolute or
escaping paths, remove the final extension, replace path separators with `_`,
replace each run outside `[A-Za-z0-9_-]` with `_`, strip leading and trailing
underscores, and append `.png`. Detect producer-name collisions before lookup
and fail rather than guessing collision suffixes. Resolve the exact producer
name first. An explicitly enumerated legacy layout may be consulted only after
the producer name is absent; the chosen rule must be recorded, and zero matches,
multiple matches, or mask reuse across metadata rows is fatal. Never join by
`img_id`, row number, label, split-local order, or loader iteration order.

Read each PNG as a categorical Pascal/VOC map. Preserve RGB or palette
semantics, decode exact VOC class IDs, reject unknown or unexpected colors and
class IDs, and construct the Boolean bird mask as `class_id == 1`. Class 0 is
RGB `[0, 0, 0]`; class 1 is RGB `[128, 0, 0]`. Grayscale thresholding,
normalizing the PNG as an input image, or accepting a white `{0, 255}` fixture
does not implement this producer contract. Existing downstream requirements
for Boolean masks, nearest-neighbor mask interpolation, joint geometry,
dilation, purity, and source-resolution composition apply after this decode.

Require complete one-to-one coverage for all official split-0 and split-1
metadata rows. Official split 2 has no mask requirement, and missing test masks
must not fail preflight. Mask-conditioned construction and evaluation use only
split-0/1 rows. Official-test reporting evaluates untouched RGB images and must
not attempt to load a mask. Extra mask files may be inventoried but cannot
change the required set or become an implicit input.

Preflight must publish `preflight/mask_manifest.json` with schema
`anchorcal-vlm-mask-manifest-v2`, canonically sorted by `img_id`, before
training. Its provenance binds the resolved dataset and metadata hash, exact
VLM root, locked source identifier
`waterbirds100_openclip_laion_dinovit_weclipplus_prediction_cmap`, mapping and
decoder implementation versions, map format, foreground IDs `[1]`, required
splits `[0, 1]`, and the AnchorCal configuration or checkout revision. The
external GALS producer-source revision is not established; do not invent one
or relabel the AnchorCal commit as that external revision.

Each required-row entry records `img_id`, metadata audit index,
`img_filename`, official split, derived producer name, resolved mask path,
mapping rule, image and mask dimensions, decoded color/class counts,
foreground count or fraction, file size, and mask SHA-256. The manifest also
records coverage by split, missing/ambiguous/collision/reuse reports, unused
extras, and a deterministic SHA-256 over canonical serialized entries. All
downstream branch, anchor, candidate, campaign, and decision receipts bind and
reverify this frozen manifest hash; workers never repeat best-effort mapping.

---

# A. Dataset, Masks, and Transforms

## 1. Which exact Waterbirds release and metadata file are authoritative?

**Decision**

Use the dedicated Waterbirds-100 release directory:

```text
/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2
```

The authoritative metadata file is:

```text
<WATERBIRDS_ROOT>/metadata.csv
```

The canonical sample identifier is the original metadata field:

```text
img_id
```

Do not use the DataFrame row position as the sample ID.

**Implementation lock**

At preflight:

1. Require the metadata columns needed by the project, including at minimum:
   - `img_id`
   - image filename/path field
   - class label
   - place/background label
   - official split
2. Require every `img_id` to be unique.
3. Sort all deterministic operations by `img_id`.
4. Compute and store the SHA-256 hash of `metadata.csv`.
5. Store the complete resolved absolute path and hash in every run manifest.

If the available metadata file does not contain a unique `img_id`, stop. Do not synthesize a new identity from row order.

---

## 2. Which masks are authoritative?

**Decision**

Use the fixed Waterbirds-100 OpenCLIP-LAION + DINOvIT WeCLIP+
`prediction_cmap` bank named in the Authoritative Waterbirds100 Dataset and
VLM-Mask Correction above.
These are categorical VLM teacher maps and are not represented as official CUB
ground truth.

The authoritative runtime mask object is the Boolean Waterbirds-coordinate bird
mask obtained by strict VOC class-1 decoding and associated one-to-one with
canonical `img_id` through the frozen `img_filename` mapping manifest.

**Implementation lock**

A valid mask bank must satisfy:

- exactly one mask per required official split-0/1 Waterbirds `img_id`;
- exact geometric correspondence with the Waterbirds composite image;
- traceability to the exact VLM root, locked producer identifier, mapping and
  decoder implementation versions, and per-file SHA-256; do not claim an
  unknown external GALS producer-source revision;
- no mix of mask sources inside the pilot.

Resolve the bank once during preflight, publish the immutable mapping manifest,
and require every later job to consume and verify that manifest. Do not replace,
regenerate, or repair the locked masks inside this pilot.

---

## 3. What should happen if a mask is missing, duplicated, corrupt, or dimensionally inconsistent?

**Decision**

Hard fail during preflight.

**Implementation lock**

Do not:

- silently drop the image;
- create a blank mask;
- resize a dimensionally unrelated mask;
- choose one duplicate arbitrarily;
- attempt an automatic segmentation repair.

The preflight report must list every offending required split-0/1 `img_id` and
reason. Training begins only after the required bank passes completely. Missing
official split-2 masks are expected and are not offending rows.

---

## 4. What is the source-mask binarization rule?

**Decision**

Treat the source PNG as a categorical Pascal/VOC class map and select foreground
class ID 1:

```python
class_ids = decode_pascal_voc_colors(source_rgb)
binary_mask = class_ids == 1
```

For this producer, background class 0 is RGB `[0, 0, 0]` and selected
foreground class 1 is RGB `[128, 0, 0]`.

**Implementation lock**

Preserve categorical RGB or palette values until decoding. Reject unknown VOC
colors, unexpected class IDs, corrupt data, invalid channels, or a source that
only passes after grayscale/continuous thresholding. A white `[255, 255, 255]`
foreground fixture does not represent VOC class 1 and must fail this configured
decoder.

After conversion:

- dtype is boolean;
- mask must contain at least one foreground pixel;
- mask must not cover the complete image;
- save decoded class/color counts in the immutable mapping manifest and
  preflight report.

---

## 5. Should foreground replacement occur before or after resizing/cropping?

**Decision**

Foreground/background replacement occurs at **source Waterbirds image resolution before interpolation**.

**Implementation lock**

For green-screen foreground construction:

1. Load the source-resolution Waterbirds image.
2. Load its source-resolution Waterbirds-coordinate mask.
3. Replace every source background pixel with green.
4. Apply the same geometric crop, resize, and flip to the already-composed image and its mask.

This prevents image interpolation from mixing original background colors into retained bird-edge pixels.

The same ordering applies to the optional full-image foreground-only candidate diagnostic.

---

## 6. What exact interpolation and normalization should all image pipelines use?

**Decision**

Lock preprocessing to the exact pretrained ViT configuration.

Selected pretrained model:

```text
hf_hub:timm/vit_small_patch16_224.augreg_in21k_ft_in1k
```

Resolved preprocessing:

```text
input_size: 224 x 224
interpolation: bicubic
antialiasing: enabled
evaluation crop_pct: 0.9
evaluation crop mode: center
mean: [0.5, 0.5, 0.5]
std:  [0.5, 0.5, 0.5]
```

Masks always use nearest-neighbor interpolation.

**Implementation lock**

- Green fill occurs in ordinary RGB space before normalization.
- Blur occurs in unnormalized RGB.
- Candidate and branch evaluation transforms must be generated from the serialized resolved model data configuration.
- Do not silently substitute generic ImageNet mean/std.
- The effective evaluation resize implied by `crop_pct=0.9` must be generated by the timm transform utilities and serialized rather than recreated from memory.

---

## 7. Should `RandomResizedCrop` use torchvision's default aspect-ratio range?

**Decision**

Yes.

```text
ratio = (3/4, 4/3)
scale = (0.70, 1.00)
output size = 224
interpolation = bicubic
antialias = true
```

This is frozen for candidate and branch training geometry.

---

## 8. What happens if a stochastic branch-training crop contains no bird pixels?

**Decision**

Use bounded rejection sampling with a deterministic fallback and a hard fallback-rate gate.

**Implementation lock**

For every stochastic branch-training sample:

1. Attempt the joint random crop up to 10 times.
2. Accept the first crop containing at least one foreground mask pixel.
3. If all 10 attempts fail, use the deterministic model evaluation crop for that sample.
4. Log:
   - `img_id`;
   - epoch;
   - attempt count;
   - fallback event.

Abort branch training if the fallback rate exceeds:

```text
0.1 percent of sampled training examples
```

For any deterministic evaluation crop, an empty bird mask is a preflight failure.

---

## 9. Are branch training transforms exactly the candidate geometric transforms?

**Decision**

Yes for geometry.

Use the same:

- `RandomResizedCrop`;
- horizontal flip;
- image size;
- aspect-ratio range.

Do not use strong color augmentation in the first pilot.

The foreground branch first green-screens at source resolution, then receives the joint geometric transform. The background branch applies the joint geometric transform before patch purity and dilation are computed in final coordinates.

---

## 10. How is the eight-pixel mask dilation defined?

**Decision**

Use a disk-shaped binary morphological dilation with radius 8 in the final 224 by 224 coordinate system.

Equivalent structuring element:

```text
17 x 17 disk, radius 8
```

**Implementation lock**

- Apply dilation after the evaluation/training geometry has produced the final 224 by 224 mask.
- Use the same implementation everywhere.
- Persist the dilation implementation name and parameters.
- Do not substitute a square max-pool kernel without explicitly changing the specification.

---

## 11. Should `expert_train` and `expert_calibration` IDs be persisted as explicit CSV files?

**Decision**

Yes.

Required files:

```text
splits/waterbirds100_expert_train.csv
splits/waterbirds100_expert_calibration.csv
splits/manifest.json
```

Each file must include:

- canonical `img_id`;
- image path;
- label;
- place;
- original official split;
- split seed;
- source metadata hash.

Also store:

- row counts;
- class counts;
- SHA-256 file hashes;
- overlap assertions;
- union assertion against the candidate-training pool.

`splits/manifest.json` uses schema `anchorcal-splits-v3` and additionally binds
the dedicated release, metadata hash, complete official split-0 membership and
alignment audit, all derived split hashes, untouched official split-1 oracle
membership, and untouched official split-2 reporting membership.

---

## 12. How should deterministic stratified splits handle rounding and input ordering?

**Decision**

Sort by canonical `img_id`, then perform fixed-seed stratified splitting.

**Implementation lock**

1. Sort input rows by `img_id`.
2. Stratify by class over the complete, preflight-validated Waterbirds100
   official-training pool.
3. Use the specified fixed seed.
4. Let the selected splitting library perform deterministic integer allocation.
5. Persist resulting IDs.
6. Never regenerate the split during normal training if the persisted CSV exists.
7. Verify that re-running the split script reproduces the same file hash.

This makes the split independent of filesystem ordering, DataFrame ordering, and worker scheduling.

---

## 13. Should candidate and branch training remain ordinary unweighted ERM despite aligned-pool class imbalance?

**Decision**

Yes.

Use:

- random shuffling;
- unweighted cross-entropy;
- no class-balanced sampler;
- no inverse-frequency loss weights.

Class balancing applies only to evaluation metrics.

This pilot is not intended to improve training. It evaluates selection criteria on ordinary trained candidates and constructs competent anchor branches.

---

# B. Branch Architecture and Anchor Construction

## 14. Which exact pretrained ViT-S/16 weights should initialize candidates and branches?

**Decision**

Lock:

```text
model identifier:
hf_hub:timm/vit_small_patch16_224.augreg_in21k_ft_in1k

Hugging Face revision:
7e2c55630205e1266030f18370f4c6ed1a514b52

timm:
1.0.28

expected model.safetensors SHA-256:
79c03c635cdfd798a364a9d8c4e5c0b7255b975ea2c9616046d4f77ab01435aa
```

**Implementation lock**

1. Fetch the exact revision once with `snapshot_download` and an explicit
   `revision`, allowing only `config.json` and `model.safetensors`.
2. Prefer the safetensors file.
3. Compute and record the local checkpoint SHA-256.
4. Record:
   - timm version;
   - Hugging Face revision;
   - resolved pretrained configuration;
   - cache path;
   - checkpoint hash.
5. After initial resolution, all production jobs must use offline cache mode.
6. Fail if a different weight hash is loaded.

The previously proposed revision
`202e80f13a7f81ed1b4d4922ef9aa15b68bf456b` is explicitly forbidden because
it is the repository's empty initial commit and does not contain model weights.

Candidates use the full pretrained ViT-S/16. Each branch copies the locked components described below.

---

## 15. What exactly is copied into each branch?

**Decision**

Independently clone from the same locked pretrained source:

- `patch_embed.proj`;
- pretrained CLS token;
- transformer blocks 0 through 5;
- final LayerNorm.

Discard:

- absolute positional embeddings;
- original 1000-class head;
- blocks 6 through 11.

Initialize from scratch:

- two-class branch classification head;
- object-relative foreground positional MLP;
- any branch-specific padding or metadata modules.

The foreground and background branches do not share runtime weights or activations.

---

## 16. Are all copied pretrained parameters fine-tuned?

**Decision**

Yes.

Train every copied parameter end to end in each six-block branch.

Do not freeze:

- patch projection;
- CLS token;
- transformer blocks;
- final LayerNorm.

The branches are calibration models, and maximizing their competence within their permitted information channel is desirable.

---

## 17. What is the exact object-relative positional encoder?

**Decision**

For each retained foreground token, use its patch-center coordinate relative to the visible transformed bird bounding box.

Coordinates:

\[
u = \frac{x_{\text{center}} - x_{\min}}
{\max(x_{\max}-x_{\min}, 1)}
\]

\[
v = \frac{y_{\text{center}} - y_{\min}}
{\max(y_{\max}-y_{\min}, 1)}
\]

Clamp both to `[0, 1]`.

Encoder:

```text
Linear(2, 128)
GELU
Linear(128, 384)
```

Add the resulting 384-dimensional vector to the patch embedding.

Initialization:

- truncated normal weight initialization with standard deviation 0.02;
- zero biases.

The CLS token receives no object-relative positional vector.

---

## 18. Does the absolute-position foreground diagnostic require a separately trained branch in the initial pilot?

**Decision**

No.

Implement the position-mode abstraction cleanly, but train only:

```text
foreground_position_mode = object_relative
```

The absolute-position variant is deferred until the primary pilot works. Do not consume compute on it now.

---

## 19. How should variable-token foreground padding work?

**Decision**

Use explicit padding masks in every transformer block.

**Implementation lock**

- CLS is always valid.
- Padded patch tokens are masked as attention keys and values.
- Padded token outputs are zeroed after each block.
- Padded tokens do not enter:
  - classification pooling;
  - saliency sums;
  - density denominators;
  - token counts;
  - source-coordinate maps.
- Query outputs for padded positions are ignored entirely.

Add a unit test showing that adding extra padding tokens does not change CLS logits beyond `1e-6`.

---

## 20. Should the background branch retain sampling with replacement when fewer than 64 pure patches exist?

**Decision**

No replacement in the primary design.

The user explicitly prefers sacrificing some performance or coverage rather than permitting leakage through duplicate patterns.

**Prespecified token-budget policy**

Run preflight on `expert_train`, `expert_calibration`, and `biased_val`.

Try token budgets in order:

```text
64
48
32
```

Choose the largest token budget satisfying all of:

- at least 95 percent of examples in each relevant split have at least K eligible pure-background patches;
- every class retains at least 95 percent coverage;

Examples with fewer than the chosen K patches are marked invalid for the background branch. Do not duplicate patches.

Freeze the selected K before branch training and store it in the manifest.

The competence intersection is not part of pre-training K selection because it
does not exist until both branches have been trained. After training once with
the frozen coverage-selected K, apply the 50-valid-examples-per-class
intersection requirement as a hard post-training gate. Do not lower K and
retrain in response to a failed intersection.

If K=32 still fails coverage, abort and redesign the background branch.

---

## 21. What happens when an evaluation image has zero eligible background patches?

**Decision**

Mark it invalid, persist the ID and reason, and exclude it from:

- background-branch calibration;
- anchor construction;
- anchor criteria;
- token-swap donor eligibility.

Hard fail if either:

```text
more than 1 percent of biased_val is invalid
```

or:

```text
the competence intersection falls below 50 valid examples per class
```

No blank background representation is allowed.

---

## 22. How are background sampling seeds generated?

**Decision**

Use SHA-256 based stable seed derivation.

Canonical payload:

```text
"{global_seed}|{sample_id}|{view_index}|{purpose}"
```

Take the first 8 bytes of the SHA-256 digest as an unsigned 64-bit integer and reduce to the RNG range.

Purposes include:

- `background_branch_eval`;
- `background_branch_train`;
- `token_swap_donor_patch`;
- `random_token_audit`.

Never use Python's built-in `hash()`.

---

## 23. Should all eight background source-index views be persisted?

**Decision**

Yes.

For every eligible sample and every view, persist:

- ordered source patch indices;
- source full-grid coordinates;
- view seed;
- token budget;
- mask/dilation hash.

Reuse the exact views across:

- branch calibration;
- anchor construction;
- anchor saliency;
- anchor token swapping;
- anchor blur;
- parity tests.

This is mandatory for paired and reproducible criterion comparison.

---

## 24. Which foreground and background training checkpoint becomes the frozen branch?

**Decision**

Use the fixed final epoch:

```text
epoch 30
```

Do not select branch checkpoints adaptively.

Rationale:

- the expert calibration split is reserved for temperature fitting and diagnostics;
- adaptive branch selection would introduce another model-selection layer;
- fixed-epoch branches are simpler to reproduce and explain.

Log the full branch training history, but freeze epoch 30.

If training diverges or epoch 30 is unusable, the branch run fails. Do not silently fall back to an earlier epoch.

---

## 25. Should final branch weights always be saved?

**Decision**

Yes.

Mandatory artifacts:

- foreground branch weights;
- background branch weights;
- resolved configs;
- optimizer/training history;
- pretrained source hash;
- branch checkpoint hash;
- calibration temperature;
- fixed background views;
- code commit;
- data and mask hashes.

These are small in number and central to reproducing every anchor.

---

## 26. Do the competence intersection, margin scales, and anchors use raw or temperature-scaled logits?

**Decision**

Use raw logits.

For the background branch, first average the eight raw view logits, then use that averaged raw logit vector.

Temperature-scaled probabilities are diagnostic only and do not define:

- correctness intersection;
- margin scales;
- normalized logits;
- anchors;
- anchor criteria.

This avoids turning probability calibration into part of the structural reliance definition.

---

## 27. How exactly is scalar temperature fitted?

**Decision**

Fit one positive scalar temperature per branch by minimizing ordinary sample-weighted NLL on `expert_calibration`.

Bounded parameterization:

```text
T = 0.05 + (20.0 - 0.05) * sigmoid(raw_T)
```

Initialize `raw_T` so that `T == 1.0`. This makes the stated temperature range
an actual optimization constraint; PyTorch LBFGS does not provide native box
constraints for an `exp(log_T)` parameter.

Optimization:

```text
optimizer: LBFGS
max iterations: 100
line search: strong_wolfe
initial T: 1.0
allowed T range: [0.05, 20.0]
```

For the background branch:

1. load the eight fixed raw-logit views;
2. average raw logits across views;
3. fit one temperature to the averaged logits.

Store raw and calibrated diagnostics.

---

## 28. Should temperature fitting be sample-weighted or class-balanced?

**Decision**

Use ordinary sample-weighted NLL for fitting.

Separately report:

- class-balanced NLL;
- class-balanced ECE;
- per-class calibration.

Temperature does not enter anchor construction, so this choice is diagnostic rather than structurally consequential.

---

## 29. Are foreground and background margin scales calculated on the full intersection or capped scoring subset?

**Decision**

Use all valid examples in the full competence intersection.

Calculate and freeze:

\[
s_F
=
\operatorname{median}_{i\in\mathcal C} m_F(i)
\]

\[
s_B
=
\operatorname{median}_{i\in\mathcal C} m_B(i)
\]

before any cap is applied for expensive criterion scoring.

The cap affects only which images are used to score validation criteria.

---

## 30. Does the capped anchor subset need its own fixed seed?

**Decision**

Yes.

```text
anchor_subset_seed = 424242
```

Persist:

- selected IDs;
- per-class counts;
- source intersection hash;
- cap rule;
- seed.

The same capped subset is used by every criterion and every lambda.

---

## 31. Is the green-shade stability audit a hard leakage gate or a diagnostic?

**Decision**

Diagnostic only.

Hard foreground leakage gate:

> Replacing the hidden original background before green-screen construction must leave the final foreground branch inputs, token metadata, and logits numerically identical.

Green-shade stability is useful, but dependence on the chosen constant fill does not prove access to original background information.

Report green-shade results for:

```text
RGB(0,255,0)
RGB(0,200,0)
RGB(32,255,32)
```

Do not fail solely because logits change across these fills.

---

# C. Numerical Audits and Go/No-Go Gates

## 32. What quantitatively means "meaningfully above chance" for each branch?

**Decision**

For binary Waterbirds:

- compute class-balanced accuracy on `biased_val`;
- use a 2,000-replicate class-stratified sample bootstrap;
- require the lower bound of the 95 percent interval to exceed 0.50.

Both foreground and background branches must pass.

Also require point estimate:

```text
> 0.50
```

The CI condition is the hard gate.

---

## 33. What is the minimum acceptable competence intersection?

**Decision**

At least:

```text
50 valid examples per class
```

after all branch eligibility rules and before capping.

This is a hard gate.

Also report:

- total intersection size;
- percentage of biased validation retained;
- per-class proportions;
- invalid-background exclusions.

---

## 34. What tolerance defines foreground background-replacement invariance?

**Decision**

Hard threshold:

```text
maximum absolute logit difference <= 1e-6
```

Requirements:

- identical foreground token pixels;
- identical source coordinates;
- identical padding metadata;
- identical mask after transformation.

Test at least 100 randomly selected samples with multiple replacement backgrounds.

Any failure above tolerance blocks anchor construction until explained.

---

## 35. What result must the random-token audit produce?

**Decision**

Class-balanced accuracy must satisfy both:

```text
point estimate <= 0.53
```

and:

```text
95 percent bootstrap confidence interval contains 0.50
```

Use 2,000 class-stratified bootstrap replicates.

If it fails, investigate implementation leakage or label-correlated random-pool construction before proceeding.

---

## 36. How should random-token audit patches be constructed?

**Decision**

Use real pure-background patches from an image-disjoint, class-balanced pool.

Construction:

1. Source patches come from `expert_train`.
2. Source images are disjoint from audit recipient images.
3. Draw equal numbers of source patches from both source classes.
4. Ignore recipient class when sampling.
5. Use the same fixed token budget and no positions.
6. Do not use synthetic noise.

The goal is to preserve realistic patch statistics while destroying label information.

---

## 37. What is the geometry-auditor protocol?

**Decision**

Use image-disjoint splits and two prespecified auditors.

Features:

- raw mask area;
- dilated mask area;
- bounding-box width and height;
- aspect ratio;
- centroid;
- perimeter;
- compactness;
- number of eligible pure-background patches.

Auditors:

1. standardized logistic regression;
2. two-layer MLP:
   ```text
   input -> 64 -> GELU -> 32 -> GELU -> class head
   ```

Evaluation:

- class-balanced accuracy;
- 2,000-replicate bootstrap interval.

This remains diagnostic because the primary background branch receives none of these explicit features and uses a fixed token budget with no positions.

---

## 38. What test-WGA range makes the candidate collection sufficiently diverse?

**Decision**

Hard initial diversity gate:

```text
max test WGA - min test WGA >= 2.0 percentage points
```

Also report the oracle-validation WGA range.

If the test-WGA range is below 2 points, stop and reconsider the candidate grid. Do not quietly add hyperparameters after inspecting selector winners.

Because this pilot is exploratory, the revised grid may be designed in a separate second pilot.

---

# D. Saliency Definition

## 39. What exactly constitutes the foreground set for saliency?

**Decision**

Use only fully foreground patches:

```text
foreground_fraction == 1.0
```

The mask is binary and nearest-neighbor transformed, so this criterion is exact.

Report:

- number of eligible foreground patches per image;
- images excluded for zero pure foreground patches;
- foreground coverage distribution.

The branch may retain mixed green/bird patches, but the saliency metric itself scores only fully bird patches.

---

## 40. What exactly constitutes the background set for all criteria?

**Decision**

Use:

```text
dilated_foreground_fraction == 0.0
```

with the same radius-8 final-coordinate dilation used by the background branch.

This definition controls:

- saliency background density;
- token-swap recipient positions;
- token-swap donor eligibility;
- background branch eligibility.

Mixed and safety-boundary patches are neither foreground nor background for criterion scoring.

---

## 41. What happens when an image lacks a qualifying foreground or background patch?

**Decision**

Determine eligibility once before criterion evaluation.

If an image lacks either set required by a criterion:

- exclude it from that criterion for every candidate and every lambda;
- persist the `img_id` and reason;
- never allow model-dependent eligibility.

Do not replace missing regions with zero denominators or artificial values.

---

## 42. Should all expensive criteria use a common eligible subset?

**Decision**

Yes.

Use the common intersection of eligibility for:

- saliency;
- token swapping;
- blur;
- anchor calibration scoring.

The common subset must retain at least:

```text
50 examples per class
```

If it does not, the initial pilot fails its common-comparison requirement. Do not switch to criterion-specific subsets in the primary analysis.

Criterion-specific coverage may be reported only as a diagnostic.

---

## 43. Where precisely is candidate saliency hooked?

**Decision**

Hook immediately after:

```text
patch projection
flattening to token sequence
```

and before:

- CLS concatenation;
- positional embeddings;
- transformer contextualization.

Exclude CLS.

For timm ViT, implement a small explicit forward path rather than relying on a fragile arbitrary module hook if necessary.

---

## 44. What score is differentiated?

**Decision**

Ordinary candidate:

```text
raw true-class pre-softmax logit
```

Anchor:

```text
final centered, margin-normalized, lambda-mixed true-class logit
```

Never differentiate:

- softmax probability;
- cross-entropy;
- predicted-class logit when it differs from the true class.

The use of the true-class logit is fixed for all models.

---

## 45. Is the positive-attribution fallback applied separately per image?

**Decision**

Yes.

Primary attribution:

\[
a_p
=
\operatorname{ReLU}
\left(
\sum_d \frac{\partial z_y}{\partial h_{p,d}}h_{p,d}
\right)
\]

If the summed positive attribution over that image's scored foreground and
background coordinates is below:

```text
1e-12
```

use absolute attribution for that image only. If the corresponding absolute
mass is also at most `1e-12`, assign neutral alignment `0.5` and set the
`zero_scored_attribution` flag. Do not change image eligibility based on model
outputs.

Log:

- fallback count;
- fallback fraction;
- affected IDs.

---

## 46. How are gradients obtained through frozen anchor weights?

**Decision**

Set branch parameters to:

```python
requires_grad_(False)
```

Do not use inference mode. Immediately after patch projection, create an
explicit gradient-bearing activation leaf:

```python
h = h.detach().requires_grad_(True)
```

Then run positional encoding and the remaining transformer computation from
`h`. Detaching is permitted only at this declared split point and only when
`requires_grad` is immediately restored.

Use:

```python
torch.autograd.grad
```

with respect to hooked patch activations.

The computational graph through the frozen operations must remain intact.

---

## 47. How are eight background views and repeated source coordinates combined for anchor saliency?

**Decision**

For every anchor image:

1. Run all eight fixed background views.
2. Average the eight raw background logits.
3. Combine the averaged background logit with the foreground logit.
4. For a direct evaluation, differentiate the final averaged anchor true-class
   logit. For the reusable cache, differentiate the centered,
   margin-normalized background branch true-class logit averaged across views,
   before applying lambda.
5. Retain the resulting signed gradient-times-activation contribution for every
   token occurrence. Cached background occurrences include the `1/8` view
   factor and `1/s_B` margin normalization; cached foreground contributions
   include `1/s_F`.
6. For every source patch coordinate appearing in multiple views, **sum** its
   signed occurrence contributions. Do not average them a second time.
7. Apply lambda weighting to cached signed contributions before ReLU or the
   absolute-value fallback. Do not apply margin normalization a second time.
8. Compute background attribution density over unique sampled source
   coordinates.
9. Unsampled source coordinates do not enter the denominator.

Because the primary background design forbids within-view replacement, duplicate coordinates occur only across views.

Persist the source-coordinate mapping used for the calculation.

Cache signed pre-ReLU contributions, never already-rectified saliency. Add a
parity case in which at least one source coordinate occurs in multiple views.

---

# E. Token Swapping and Blur

## 48. Are the four donors distinct opposite-class images selected without replacement?

**Decision**

Yes.

For each recipient:

- choose four distinct opposite-class donor images;
- sample from all eligible `biased_val` donor images;
- exclude the recipient;
- fix with seed `31415`;
- persist assignments.

For binary Waterbirds100, opposite bird class implies opposite background class.

---

## 49. Should exact donor patch assignments also be fixed?

**Decision**

Yes.

Persist for every:

- recipient ID;
- donor ID;
- recipient patch position;
- donor source patch index;
- fallback-bin event.

These spatial patch assignments apply to ordinary full-image candidates only.
Candidates and anchors share the same four donor image IDs, but the
position-free anchor background branch uses the donor-ID-keyed fixed views
defined in Questions 54 and 55. It does not consume recipient-position-specific
candidate patch assignments.

---

## 50. How are 3 by 3 spatial bins defined?

**Decision**

For a 14 by 14 patch grid, use patch-center coordinates.

For row or column index `k` in `0..13`:

\[
b(k)
=
\min\left(
2,
\left\lfloor
3\frac{k+0.5}{14}
\right\rfloor
\right).
\]

The pair of row and column bins gives one of nine coarse spatial bins.

---

## 51. May donor patches be reused when a bin lacks enough unique pure patches?

**Decision**

Yes, for token-swap intervention construction only.

After the distinct donor image is fixed:

- prefer unique donor patches in the matching bin;
- if insufficient, sample deterministically with replacement from the matching bin;
- if the bin is empty, sample deterministically from all pure donor background patches.

Persist the fallback type.

This does not change the primary background branch's no-replacement rule.

---

## 52. Does the eight-pixel dilation define both recipient replacement positions and donor eligibility?

**Decision**

Yes.

Only patches satisfying:

```text
dilated_foreground_fraction == 0
```

can be:

- replaced in the recipient;
- used as donor tokens.

Mixed and dilation-boundary recipient patches remain unchanged.

---

## 53. Is primary swap accuracy donor-specific correctness or correctness after averaged donor logits?

**Decision**

Use mean donor-specific correctness.

Per image:

1. Score each of the four donor interventions separately.
2. Convert each to a `0/1` correctness value.
3. Average the four correctness values.
4. Average images within class.
5. Average classes.

Cache mean donor logits only as a diagnostic.

---

## 54. How do four anchor donors interact with eight background views?

**Decision**

For each donor:

1. Load the donor's eight fixed background views.
2. Average the donor's eight raw background logits.
3. Keep the recipient foreground logit unchanged.
4. Form the anchor mixed logit at lambda.
5. Score correctness.
6. Repeat for all four donors.
7. Average donor-specific correctness.

Do not create a 32-way average before correctness.

---

## 55. Are anchor donor views keyed by donor ID or recipient-donor pair?

**Decision**

Donor ID.

A donor represents one invariant cached source of background evidence regardless of recipient.

Recipient-donor pair metadata stores assignment, but the donor background branch logits are not recomputed with recipient-specific randomness.

---

## 56. What exact Gaussian kernel corresponds to each sigma?

**Decision**

Operate in unnormalized RGB using separable Gaussian convolution.

For each sigma:

\[
k
=
2\lceil3\sigma\rceil+1.
\]

Use:

- reflect padding;
- normalized Gaussian kernel;
- float32 convolution.

Apply the deterministic joint evaluation transform first. Perform blur on the
unnormalized 224 by 224 RGB image and its transformed binary mask, then apply
model normalization. Sigma is therefore measured in final 224-space pixels.

Sigmas:

```text
2, 4, 8
```

---

## 57. Is bird-color bleeding into nearby blurred background accepted?

**Decision**

No.

Use mask-normalized background-only convolution.

Let:

\[
B = 1-M
\]

\[
N_\sigma = G_\sigma(x\odot B)
\]

\[
D_\sigma = G_\sigma(B)
\]

\[
x_{\mathrm{bgblur}}
=
\frac{N_\sigma}{\max(D_\sigma,10^{-6})}
\]

\[
x_\sigma
=
M\odot x
+
B\odot x_{\mathrm{bgblur}}.
\]

This prevents bird pixels from contributing to blurred background values.

Label blur as a background-only intervention baseline, not an artifact-uncorrected baseline.

---

## 58. During anchor blur, are the original eight background patch locations reused?

**Decision**

Yes.

Blur changes pixel values only. It does not resample patch locations.

Use the exact eight source-index views persisted for the clean background branch.

---

## 59. What exactly is foreground-only evaluation for an ordinary candidate?

**Decision**

Pass the ordinary 196-token full-image candidate ViT a complete 224 by 224 green-screen image.

Do not route it through the variable-token foreground branch.

The candidate receives its normal architecture and positional embeddings.

This remains an optional anchor-construction-matched diagnostic, not a primary eligible criterion.

---

## 60. How is foreground-only evaluation applied to an anchor?

**Decision**

Green-screen at source resolution, apply evaluation geometry, then run the full anchor wrapper normally.

- Foreground stream receives the same bird content.
- Background stream samples its fixed positions, which now contain green.
- Lambda mixing remains unchanged.

---

## 61. Should intervention types be explicit and stream-restricted?

**Decision**

Yes.

Implement typed interventions such as:

```text
NONE
TOKEN_SWAP_BACKGROUND
BLUR_BACKGROUND
FOREGROUND_ONLY_GREENSCREEN
```

Each intervention carries explicit assertions.

For anchors:

- token swap may modify only the background branch;
- blur may modify only source background pixels before background patch extraction;
- foreground stream outputs must remain identical for background-only interventions within `1e-6`.

Fail on any unauthorized stream modification.

---

# F. Score Aggregation

## 62. Which accuracy enters the candidate harmonic mean?

**Decision**

Use full-`biased_val` class-balanced accuracy:

\[
A_{\mathrm{biased,full}}(f)
\]

Combine it with the robustness quantity measured on the fixed common selector subset.

Example:

\[
S_{\mathrm{swap}}(f)
=
H(
A_{\mathrm{biased,full}}(f),
A_{\mathrm{swap,subset}}(f)
).
\]

---

## 63. Which accuracy enters the anchor harmonic mean?

**Decision**

Use clean anchor accuracy on the same full or capped competence subset used for the criterion.

It must equal:

```text
1.0
```

for every lambda.

If it does not, stop. The competence-matched anchor invariant has failed.

---

## 64. What is the aggregation order?

**Decision**

Use this exact order:

1. Average donors or stochastic views within each image.
2. Average images within each true class.
3. Average class values.
4. Apply the harmonic mean with full biased-validation class-balanced accuracy.

Do not pool every donor-example observation globally.

---

## 65. Do product variants participate in winner selection?

**Decision**

No.

Store product variants as diagnostics only:

\[
P(a,b)=ab.
\]

Exclude them from:

- ACE winner selection;
- credible sets;
- primary criterion tables.

The primary combination is harmonic mean.

---

# G. AnchorCal Statistics

## 66. During bootstrapping, are the intersection, scales, views, donors, and anchor family reconstructed?

**Decision**

No.

Freeze before bootstrap:

- competence intersection;
- validity exclusions;
- anchor subset cap;
- foreground/background scales;
- branch weights;
- background views;
- donor images;
- donor patch assignments;
- anchor family;
- per-image criterion outputs.

Bootstrap only images within class.

The bootstrap estimates sampling uncertainty conditional on the constructed pilot.

---

## 67. Should identical paired bootstrap indices be used across every criterion and lambda?

**Decision**

Yes.

Generate one stratified index set per bootstrap replicate and apply it to:

- every criterion;
- every lambda.

Persist the indices or their deterministic seeds.

This enables paired criterion comparisons.

---

## 68. What does "one standard error" mean?

**Decision**

Use the standard deviation of the 200 bootstrap ACE estimates.

Do not divide by \(\sqrt{200}\).

The bootstrap standard deviation is already the estimated standard error of the statistic under resampling.

---

## 69. Is the winner always minimum point-estimate ACE?

**Decision**

Yes.

Declared winner:

\[
\arg\min_j \operatorname{ACE}_j
\]

using point estimates.

Report the one-standard-error credible set separately.

Use tie-breakers only when ACE values are equal within the documented numerical tolerance.

---

## 70. How should constant-score criteria be represented?

**Decision**

For a constant anchor score vector:

- Kendall: `NA (constant score)`;
- Spearman: `NA (constant score)`;
- PairAcc: `0.5`;
- AdjAcc: `0.5`;
- monotonicity violations: `0`;
- PerfectOrder: `false`.

ACE special case:

- isotonic regression cannot identify a slope from constant inputs;
- predict the mean lambda of the training fold for every held-out lambda;
- compute ordinary absolute error from those predictions.

Do not assign perfect ACE to a constant criterion.

---

## 71. What tolerance defines anchor-score ties and monotonicity?

**Decision**

Use:

```text
anchor_score_tolerance = 1e-10
```

Two anchor scores within this absolute tolerance are tied.

Keep candidate-selection tolerance separate:

```text
candidate_score_tolerance = 1e-8
```

---

## 72. Is alternating-lambda ACE deliberately lambda-held-out rather than image-held-out?

**Decision**

Yes.

Official name:

```text
cross-fitted lambda-interpolation Anchor Calibration Error
```

The alternating folds hold out lambda values. Image uncertainty is handled by the stratified bootstrap.

---

## 73. Should raw ACE retain its unavoidable endpoint-clipping floor?

**Decision**

Yes.

Do not subtract, normalize away, or correct the approximately `0.00476` floor associated with the alternating-lambda interpolation and endpoint clipping.

Document the floor in tables and comparisons.

All criteria share the same protocol.

---

# H. Candidate Selection and Reporting

## 74. Is epoch zero evaluated?

**Decision**

Yes, once per architecture/initialization as a shared diagnostic.

Exclude epoch zero from the prespecified 240-candidate primary pool.

Store its:

- validation criteria;
- oracle metrics;
- test metrics.

Do not allow it to be selected in primary candidate comparisons.

---

## 75. Is the competent pool global across all candidates?

**Decision**

Yes.

Primary competent pool:

\[
\mathcal F_{\mathrm{competent}}
=
\{f:A_{\mathrm{biased}}(f)\geq A_{\max}-0.01\}
\]

where \(A_{\max}\) is the maximum full biased-validation accuracy across all 240 candidates.

Report:

- pool size;
- epochs represented;
- hyperparameter configurations represented.

Also report within-run competent-pool correlations as secondary diagnostics if the global pool is narrow.

Primary criterion selection itself still uses the harmonic scores across the complete candidate pool.

---

## 76. What happens if fewer than approximately 20 candidates enter the competent pool?

**Decision**

Mark competent-pool correlation analysis as underpowered.

Do not interpret it strongly.

Still report:

- exact candidate count;
- configuration coverage;
- descriptive scatter plots.

Do not silently widen the 1-point competence threshold after looking at results.

---

## 77. What numerical gap means near-best or near-oracle?

**Decision**

No more than:

```text
1.0 test-WGA percentage point of regret
```

Use the same threshold for near-best practical selection and near-oracle selected-model performance.

Always report the continuous regret too.

---

## 78. Are Success Levels 3 and 4 descriptive or hard pass/fail gates?

**Decision**

Use the 1-point threshold as the hard success threshold for those named levels.

Also report continuous values so a result of 1.01 points is not presented as qualitatively unrelated to 0.99 points.

---

## 79. Is saving no intermediate candidate checkpoints a hard prohibition?

**Decision**

No.

Do not save every epoch. Maintain rolling best checkpoints for:

- ordinary accuracy;
- saliency criterion;
- token-swap criterion;
- blur criterion;
- oracle validation;
- final epoch.

Do this separately within every candidate-grid run.

Deduplicate when multiple criteria select the same epoch and model hash.

Write checkpoint files atomically and maintain a manifest.

This preserves only a small number of candidate states while avoiding dependence on exact retraining reproducibility.

---

## 80. May a rolling restart checkpoint be saved for cluster preemption?

**Decision**

Yes.

One overwritable resume checkpoint per run may contain:

- model;
- optimizer;
- scheduler;
- epoch;
- RNG states;
- scaler state if any;
- dataloader progress only at epoch boundaries.

After successful completion, archive it as the final-state checkpoint or delete it according to the run manifest.

---

## 81. Should the AnchorCal decision be frozen before candidate oracle/test results are revealed?

**Decision**

Yes as a provenance step, but this remains an exploratory pilot rather than a formal blind preregistration.

Produce a timestamped and hashed receipt containing:

- eligible criteria;
- formulas;
- anchor subset hash;
- anchor family;
- branch hashes;
- ACE values;
- winner;
- credible set;
- tie-break rules;
- config hashes.

Create it before the final automated join with oracle/test candidate quality.

The user may still inspect exploratory test files. The receipt exists to establish chronology and prevent accidental post-hoc rewriting of AnchorCal.

---

## 82. May per-epoch test values appear in stdout or dashboards?

**Decision**

They may be computed live and stored, but should not be mixed into ordinary training/selector stdout by default.

Use a separate namespace:

```text
exploratory_hidden_metrics/
```

At job completion, a summary may print with an explicit `EXPLORATORY_TEST_ONLY` prefix.

Candidate-selection code must have no import or data dependency on this namespace.

---

## 83. Are oracle/test per-example outputs mandatory?

**Decision**

Yes.

Store for every candidate and every oracle/test sample:

- sample ID;
- true label;
- group;
- logits;
- prediction;
- correctness;
- per-example loss.

Use chunked compressed storage.

This supports:

- auditing;
- bootstrap confidence intervals;
- alternative post-hoc metrics;
- selected-model error analysis.

---

## 84. Should final selected-model metrics include sample-bootstrap confidence intervals?

**Decision**

Yes.

Use class/group-stratified sample bootstrap intervals for:

- test average accuracy;
- test WGA;
- oracle validation WGA;
- regret differences when appropriate.

Clearly distinguish:

- sample uncertainty;
- training-seed uncertainty, which is not estimated by one candidate seed.

Use 2,000 bootstrap replicates for final selected-model intervals.

---

# I. Repository, Reproducibility, and Compute

## 85. Should AnchorCal be a clean new package inside the current repository?

**Decision**

Yes.

Repository root:

```text
/home/ryreu/guided_cnn/BirdOnly
```

Create:

```text
src/anchorcal/
configs/anchorcal/
scripts/anchorcal/
slurm/anchorcal/
tests/anchorcal/
outputs/anchorcal/
```

Leave any existing `src/setv` code untouched.

---

## 86. May generic old utilities be reused?

**Decision**

Yes, but only audited generic utilities.

Allowed examples:

- mask loading;
- joint image-mask transforms;
- stable hashing;
- seed handling;
- group metrics;
- checkpoint serialization.

Do not import old SETV:

- selector logic;
- expert definitions;
- weighting logic;
- hidden assumptions.

If reused, add unit tests under `tests/anchorcal`.

---

## 87. Does the previous TIGRIS compute handoff remain authoritative?

**Decision**

Yes for scheduler and architecture conventions:

```text
login: tigris.rc.rit.edu
partition: tigris
GPU request: --gres=gpu:gh200:1
architecture: aarch64
```

Remove A100-specific constraints.

The previous handoff is not authoritative for:

- current Python environment;
- exact package versions;
- current dataset paths;
- compiled extensions.

Run an ARM/GH200 environment preflight before any full job.

---

## 88. Is the research-compute repository still expected at `/home/ryreu/guided_cnn/BirdOnly`?

**Decision**

Yes.

Treat:

```text
/home/ryreu/guided_cnn/BirdOnly
```

as the authoritative repository root.

The preflight script must fail with a clear message if it does not exist. Do not silently clone or create a second repository elsewhere.

---

## 89. What are the exact research-compute paths?

**Decision**

The following paths are fixed:

```text
ANCHORCAL_REPO_ROOT
/home/ryreu/guided_cnn/BirdOnly

ANCHORCAL_OUTPUT_ROOT
/home/ryreu/guided_cnn/BirdOnly/outputs/anchorcal/waterbirds100_pilot

HF_HOME
/home/ryreu/.cache/huggingface

WATERBIRDS_ROOT
/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2

WATERBIRDS_METADATA
/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2/metadata.csv

VLM_MASK_ROOT
/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_waterbirds100_openclip_laion_dinovit/val/prediction_cmap
```

Freeze these established TIGRIS paths in the local uncommitted configuration:

```text
configs/anchorcal/paths.local.yaml
```

with:

```yaml
repo_root: /home/ryreu/guided_cnn/BirdOnly
waterbirds_root: /home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2
metadata_path: /home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2/metadata.csv
vlm_mask_root: /home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_waterbirds100_openclip_laion_dinovit/val/prediction_cmap
hf_home: /home/ryreu/.cache/huggingface
output_root: /home/ryreu/guided_cnn/BirdOnly/outputs/anchorcal/waterbirds100_pilot
```

Preflight requirements:

- `waterbirds_root` basename or contents must identify `waterbird_1.0_forest2water2`;
- `metadata_path` must equal the authoritative metadata file under that release;
- every official split-0 row must satisfy `y == place`, with no filtering used
  to manufacture that property;
- `vlm_mask_root` must equal the exact Waterbirds-100 VLM root above;
- the VLM bank must pass the producer-first `img_filename` join, strict VOC
  class-1 decode, dimension, split-0/1 coverage, and one-mask-per-required-`img_id`
  audits;
- preflight must write `preflight/mask_manifest.json` with schema
  `anchorcal-vlm-mask-manifest-v2`, plus its immutable content hash;
- all paths are converted to absolute resolved paths;
- path config is copied into the run manifest;
- no training job starts while a `REQUIRED_ABSOLUTE_PATH` value remains.

A discovery script may verify the fixed roots and may print other candidates for
diagnosis, but it must never substitute a different VLM bank automatically.

---

## 90. Which GPU partition/type, memory, CPU count, and wall-time should scripts target?

**Decision**

Use TIGRIS GH200 jobs.

### Debug job

```text
partition: tigris
gres: gpu:gh200:1
cpus-per-task: 8
memory: 32G
wall time: 2:00:00
```

### Foreground/background branch jobs

```text
partition: tigris
gres: gpu:gh200:1
cpus-per-task: 12
memory: 64G
wall time: 12:00:00
```

### AnchorCal anchor-evaluation job

```text
partition: tigris
gres: gpu:gh200:1
cpus-per-task: 12
memory: 64G
wall time: 12:00:00
```

### Each candidate-grid job

```text
partition: tigris
gres: gpu:gh200:1
cpus-per-task: 16
memory: 96G
wall time: 24:00:00
```

### Final analysis job

```text
partition: tigris
GPU: none
cpus-per-task: 8
memory: 64G
wall time: 4:00:00
```

No A100 constraints. Use the established TIGRIS account:

```text
--account=reu-aisocial
```

QOS flags remain site-specific and must follow the user's working TIGRIS convention.

---

## 91. Should the six candidate configurations be separate restartable jobs?

**Decision**

Yes.

Job graph:

1. data/mask/environment preflight and one online model-cache population job;
2. standalone end-to-end debug job using separate miniature branch and candidate artifacts;
3. production foreground and background branch jobs in parallel;
4. production AnchorCal ladder/criterion job after both branches, ending with the decision receipt;
5. six independent production candidate jobs after the decision receipt;
6. final joined analysis job after all six candidates complete.

Each candidate job has its own:

- resume checkpoint;
- HDF5 output;
- logs;
- rolling selector checkpoints.

Use Slurm dependencies where practical.

Use `afterok` dependencies. Production jobs run with Hugging Face offline mode
enabled. Debug artifacts live under a separate `debug/` namespace and are never
eligible for the production analysis.

---

## 92. Which exact debug configuration should be used?

**Decision**

Use:

```text
candidate lr: 3e-5
weight decay: 0.05
candidate epochs: 3
foreground branch epochs: 3
background branch epochs: 3
selector examples: 64 total, class balanced
swap donors: 2
blur sigma: 4
anchor lambdas: {0, 0.25, 0.5, 0.75, 1}
bootstrap replicates: 20
background views: 2 for debug only
```

The debug run must execute every pipeline stage and every storage write,
including temperature fitting, competence-intersection construction, cached
versus direct parity checks, a decision receipt, rolling checkpoints, resume
metadata, and both selector-visible and reporting-only HDF5 files. Debug branch
weights and outputs must never be reused by production.

---

## 93. Should all optimizer and runtime details be explicitly frozen?

**Decision**

Yes.

### AdamW

```text
betas = (0.9, 0.999)
epsilon = 1e-8
amsgrad = false
```

### Learning-rate schedule

```text
warmup: linear
warmup start LR: 0.01 * base LR
warmup end LR: base LR
cosine minimum LR: 0.01 * base LR
```

Use the already specified warmup epochs:

- 3 of 30 for branches;
- 4 of 40 for candidates.

### Gradient clipping

```text
global L2 norm
max norm = 1.0
apply after backward and before optimizer step
```

### Mixed precision

On GH200:

```text
autocast dtype = bfloat16
GradScaler = disabled
master parameters = float32
```

Saliency gradients may fall back to float32 if parity tests show instability.

### Data loading

```text
num_workers = 8
pin_memory = true
persistent_workers = true
prefetch_factor = 2
drop_last = false
```

### Worker seeding

Persistent worker initialization is seeded once from a stable combination of
the global run seed and worker ID. Scientific randomness must not depend on
worker RNG state. Derive every stochastic geometric transform and background
sample statelessly from a SHA-256 payload containing:

```text
run_seed | epoch | img_id | purpose
```

Because the shuffled training sampler visits each image once per epoch, no
occurrence index is required. If a future sampler permits repeated visits, add
the deterministic occurrence index to the payload. Never depend on Python hash
randomization or worker scheduling.

### Other locks

```text
torch.compile = disabled
gradient accumulation = 1
optimizer step per batch
```

---

## 94. What reproducibility level is required?

**Decision**

Require deterministic data, split, mask, donor, view, and intervention construction. Training should be strongly seeded and auditable, but bitwise identity across CUDA/hardware is not promised.

Record:

- Git commit;
- dirty-tree diff hash;
- resolved configs;
- split hashes;
- metadata hash;
- mask-bank hash/manifest;
- pretrained checkpoint hash;
- Python version;
- PyTorch version;
- torchvision version;
- timm version;
- CUDA version;
- cuDNN version;
- GPU model;
- hostname;
- Slurm job ID;
- all seeds;
- RNG policies;
- exact package lock file.

Runtime flags:

```text
PYTHONHASHSEED = run seed
cudnn.benchmark = false
cudnn.deterministic = true
torch.use_deterministic_algorithms(True, warn_only=True)
CUBLAS_WORKSPACE_CONFIG = :4096:8
torch.backends.cuda.matmul.allow_tf32 = false
torch.backends.cudnn.allow_tf32 = false
```

If a nondeterministic operation emits a warning, record it in the run manifest.

---

## 95. May anchor evaluation cache extreme-branch quantities and generate all lambdas algebraically?

**Decision**

Yes.

Cache:

- foreground logits;
- signed foreground saliency components;
- clean background logits;
- donor background logits;
- blurred background logits;
- signed source-coordinate attribution components.

Generate lambda-specific mixed logits and linear gradient contributions algebraically.

Required parity test on a small batch:

```text
direct vs cached logits max absolute difference <= 1e-6
direct vs cached saliency max absolute difference <= 1e-5
direct vs cached criterion score difference <= 1e-6
```

Run parity at lambdas:

```text
0.0, 0.35, 0.5, 0.8, 1.0
```

Do not use the cache if any parity gate fails.

---

## 96. Which candidate storage backend should be primary?

**Decision**

Use HDF5.

Structure:

```text
one selector-visible HDF5 file per candidate-grid run
one reporting-only oracle/test HDF5 file per candidate-grid run
```

This avoids concurrent writers and gives the hidden-metric namespace a physical
storage boundary from selector code.

Requirements:

- chunked datasets by epoch and sample;
- gzip or lzf compression;
- explicit schema version;
- one sole main-process writer per file;
- preallocated epoch-major datasets inside a `.partial.h5` file;
- per-epoch completion flag;
- SHA-256 manifest for completed files;
- write and flush all epoch data before writing that epoch's completion flag;
- on recovery, ignore and overwrite any epoch without a completion flag;
- after successful close, use same-filesystem `os.replace` to publish the final file;
- one exclusive per-run lock preventing duplicate writers;
- no 240 independent NPZ files.

Suggested path:

```text
outputs/anchorcal/waterbirds100_pilot/candidates/<run_id>/candidate_outputs.h5
outputs/anchorcal/waterbirds100_pilot/candidates/<run_id>/exploratory_hidden_metrics.h5
```

Candidate selection code may open only `candidate_outputs.h5`. Final aggregation
is the sole writer of cross-run summaries such as `all_candidates.csv`.

---

# J. Final Residual Implementation Locks

Questions 97 through 112 were added after a final cross-check of the plan and
the first 96 answers. Where wording below conflicts with an earlier answer, the
later numbered lock takes precedence. The corresponding earlier sections have
also been repaired in place so an implementer does not need to reconcile two
live alternatives.

## 97. Which populated pretrained revision and runtime version are authoritative?

**Decision**

Use:

```text
repository: timm/vit_small_patch16_224.augreg_in21k_ft_in1k
revision: 7e2c55630205e1266030f18370f4c6ed1a514b52
model.safetensors SHA-256: 79c03c635cdfd798a364a9d8c4e5c0b7255b975ea2c9616046d4f77ab01435aa
timm: 1.0.28
```

Fetch the pinned snapshot once, verify the file hash and serialized preprocessing
configuration, and use the resolved local snapshot for model construction.
Production jobs run with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

The old revision `202e80f13a7f81ed1b4d4922ef9aa15b68bf456b` is invalid for
this pilot because it contains no model weights.

---

## 98. How is the ordinary candidate classification head constructed?

**Decision**

Replace the pretrained 1,000-class head with:

```text
Linear(384, 2)
```

Initialize its weights with truncated normal standard deviation `0.02` and its
bias with zeros. Load the complete locked pretrained candidate backbone,
including patch projection, CLS token, absolute positional embeddings, all 12
blocks, and final LayerNorm. Never partially load or reuse the 1,000-class head.

---

## 99. What are all remaining named random seeds?

**Decision**

Use:

```yaml
foreground_branch_train_seed: 6001
background_branch_train_seed: 6002
background_sampling_seed: 6003
branch_audit_bootstrap_seed: 7001
anchor_bootstrap_seed: 7002
final_metric_bootstrap_seed: 7003
geometry_auditor_split_seed: 8001
geometry_auditor_model_seed: 8002
random_token_audit_seed: 8003
debug_seed: 9001
```

Existing seeds remain unchanged:

```yaml
split_seed: 1729
expert_calibration_seed: 2718
candidate_seed: 1234
selector_eval_seed: 16180
donor_assignment_seed: 31415
anchor_subset_seed: 424242
```

Every stochastic purpose must use its named seed or a SHA-256-derived child seed.
No unnamed call to a global RNG is permitted in data construction,
interventions, bootstrapping, or auditing.

---

## 100. Which AdamW parameters receive weight decay?

**Decision**

Use the same standard two-group policy for candidates and branches.

No-decay group (`weight_decay = 0`):

- all bias parameters;
- all one-dimensional parameters;
- LayerNorm scale and bias;
- CLS tokens;
- candidate absolute positional embeddings.

Decay group:

- all remaining matrix and convolution weights, including patch projection,
  attention, MLP, classification-head weights, and foreground positional-MLP
  weights.

The configured weight decay applies only to the decay group. Persist the sorted
parameter names and parameter counts in both groups, and fail if any trainable
parameter belongs to neither or both.

---

## 101. At what granularity does the learning-rate scheduler step?

**Decision**

Step the warmup-plus-cosine schedule once per optimizer update, not once per
epoch.

Define:

```text
total_updates = epochs * optimizer_updates_per_epoch
warmup_updates = warmup_epochs * optimizer_updates_per_epoch
```

The first optimizer update uses `0.01 * base_lr`. Warmup reaches `base_lr` at
the end of `warmup_updates`; cosine decay reaches `0.01 * base_lr` at the final
scheduled update. Apply gradient clipping, perform the optimizer update, and
then advance the scheduler for the next update. Save and restore scheduler state
in resume checkpoints.

---

## 102. How is the background token budget selected without using an unavailable competence intersection?

**Decision**

Preflight selects the largest value in `{64, 48, 32}` satisfying only the
prespecified overall and per-class 95-percent coverage gates on
`expert_train`, `expert_calibration`, and `biased_val`.

Freeze that K, train each production branch once, and then evaluate the
50-valid-examples-per-class competence-intersection gate. A failed intersection
fails the pilot. It never triggers a lower K or automatic retraining.

---

## 103. How are repeated-coordinate saliency contributions aggregated across background views?

**Decision**

For the cache, differentiate the centered, margin-normalized background branch
true-class logit averaged across all eight views before lambda mixing. Each
token occurrence therefore carries its mathematically correct `1/8` factor and
the branch's `1/s_B` factor. Cache the foreground branch analog with `1/s_F`.
Preserve signed gradient-times-activation contributions through lambda mixing.
A direct non-cached evaluation differentiates the final anchor logit and must
match the cached construction within the parity tolerances.

When one physical source coordinate appears in multiple views, **sum** its
signed occurrence contributions. Do not average a second time. After mapping to
unique physical coordinates:

1. apply only the lambda coefficient, because the cached contribution is already
   margin-normalized;
2. apply ReLU per coordinate;
3. apply the per-image absolute-value fallback if required;
4. compute density over unique scored coordinates.

This is faithful to the gradient of the averaged function and removes the
unintended double averaging. Cached-versus-direct parity must include at least
one example with a repeated coordinate.

---

## 104. Which token-swap assignments are shared between candidates and anchors?

**Decision**

Share the four fixed donor image IDs for every recipient.

Ordinary full-image candidates additionally use persisted recipient-position to
donor-patch assignments with the 3-by-3 bin rule. Position-free anchors ignore
those spatial assignments and use each donor ID's invariant eight fixed
background views. Both paths remain deterministic, but they are not forced into
an architecture-inappropriate identical patch map.

---

## 105. How are frozen-anchor saliency gradients and zero-mass cases handled?

**Decision**

Branch parameters remain frozen. Split each branch forward immediately after
patch projection, convert the projected token tensor to a gradient-bearing leaf
with:

```python
h = h.detach().requires_grad_(True)
```

and run positional encoding plus the remaining transformer computation from
that tensor. Use `torch.enable_grad()` and `torch.autograd.grad`; never use
inference mode for saliency.

Evaluate the `1e-12` positive-mass threshold over scored
foreground/background coordinates only. If positive mass is too small, use
absolute signed contributions on those coordinates. If absolute scored mass is
still at most `1e-12`, assign neutral per-image alignment `0.5`, persist a
`zero_scored_attribution` flag, and do not exclude the image dynamically.

Unit tests must require finite, non-null token gradients for both streams and at
lambda endpoints.

---

## 106. In what order are common selector and anchor subsets constructed?

**Decision**

Candidate path:

1. compute model-independent geometric eligibility on all `biased_val` images;
2. take the common saliency/swap/blur eligible pool;
3. sample up to 256 images per class with `selector_eval_seed`;
4. persist IDs and the source-pool hash.

Anchor path:

1. construct the full branch-valid, both-correct competence intersection;
2. compute margin scales on that full intersection;
3. intersect it with the same model-independent geometric eligibility rule;
4. require at least 50 eligible images per class;
5. cap at 512 per class with `anchor_subset_seed`;
6. persist IDs and hashes.

The candidate selector subset and anchor criterion subset are distinct named
artifacts even though both originate from `biased_val`. Bootstrap indices are
paired only within the fixed anchor criterion subset.

---

## 107. How are degenerate isotonic folds and undefined bootstrap correlations handled?

**Decision**

If an ACE training fold has fewer than two score levels distinct at
`anchor_score_tolerance`, predict that fold's mean training lambda for every
held-out point. Otherwise fit the specified isotonic model to the unrounded
scores.

The `1e-10` tolerance is used for ordering metrics, tie declarations, and the
degenerate-fold check; it does not quantize ordinary isotonic inputs.

For bootstrap Kendall or Spearman results that are undefined because a score
vector is constant:

- retain `NA`;
- never replace it with zero;
- report the valid replicate count and NA rate;
- summarize intervals only over valid replicates and label them accordingly.

---

## 108. What exactly does Waterbirds100 mean, and how are VLM masks mapped?

**Decision**

`Waterbirds100` means the dedicated `waterbird_1.0_forest2water2` release. Its
complete official split 0 is the development source. Preflight must assert that
every one of those rows has `y == place`; it must not filter a Waterbirds-95
training split to manufacture the source pool.

For this release, map the complete dataset-relative `img_filename` to
the VLM PNG using the exact producer-compatible flattening rule in the
Authoritative Waterbirds100 Dataset and VLM-Mask Correction. Do not preserve a
nested class directory and do not map from row position or `img_id`. Require
exact image and mask width/height for every required split-0/1 row.

Path configuration distinguishes:

```yaml
vlm_mask_root: /home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_waterbirds100_openclip_laion_dinovit/val/prediction_cmap
```

Preflight must reject producer-name collisions, missing or ambiguous required
maps, mask reuse, dimensional mismatches, and invalid VOC content. It freezes
the accepted one-to-one mapping in `preflight/mask_manifest.json` with schema
`anchorcal-vlm-mask-manifest-v2` and a deterministic hash. Official split 2 is
outside the mask-coverage contract.

---

## 109. What is the final HDF5 transaction and namespace design?

**Decision**

Each candidate run has one sole main-process writer and two physically separate
files:

```text
candidate_outputs.h5
exploratory_hidden_metrics.h5
```

Write each through a recoverable `.partial.h5` file with preallocated
epoch-major datasets. Write and flush epoch contents first; set that epoch's
completion flag last. Resume overwrites any incomplete epoch. After successful
close, publish with same-filesystem `os.replace` and write a SHA-256 manifest.

Use a per-run exclusive lock. Candidate workers never write shared CSVs or
campaign manifests. Only final aggregation writes cross-run summaries.
Selector modules have no path, schema, or import dependency on the hidden file.

---

## 110. What is the final debug and production Slurm order?

**Decision**

Use:

```text
preflight + single online model-cache population
    -> standalone end-to-end miniature debug
    -> production foreground/background branches in parallel
    -> production AnchorCal evaluation and decision receipt
    -> six production candidate jobs
    -> final joined analysis
```

The debug trains foreground and background branches for three epochs each and
the candidate for three epochs. All debug artifacts are isolated and discarded
from production analysis. Use `afterok` dependencies, the working TIGRIS
account `reu-aisocial`, GH200 resources from Question 90, and offline model mode
after cache population.

---

## 111. How is production code made immutable across queued jobs?

**Decision**

Debug jobs may record and run a dirty tree. Production submission requires a
clean Git worktree and records `EXPECTED_COMMIT`. Every production job verifies
before importing project code that:

```text
git rev-parse HEAD == EXPECTED_COMMIT
git status --porcelain is empty
```

Fail otherwise. Do not allow queued jobs to observe later edits in the shared
working tree. If continued development must occur during a campaign, run the
campaign from a read-only commit-specific worktree or immutable code snapshot.

---

## 112. How is the GH200 Python environment frozen without modifying a shared environment?

**Decision**

Use `timm==1.0.28`. Require `h5py`, `safetensors`, and `huggingface_hub` in
addition to the project's ordinary PyTorch/scientific dependencies. The ARM
environment preflight writes an exact package lock and import/version report.

Never downgrade or mutate a shared environment inside a Slurm job. If the
working shared environment does not satisfy the lock, create a project-specific
cloned environment during explicit environment setup, validate it on a GH200,
and point all campaign jobs to that frozen interpreter. Production jobs fail on
version mismatch rather than installing packages at runtime.

---

## 113. What does “criteria differ substantially” mean for Success Level 1?

**Decision**

Prespecify the operational threshold before seeing campaign results. Let the
criterion-separation statistic be the maximum absolute point-score gap between
any two eligible criteria at the same anchor lambda. The criteria differ
substantially when this statistic is at least:

```text
0.01 absolute score
```

Always report the continuous maximum gap and this threshold. This threshold is
only part of the descriptive Success Level 1 label; it does not affect the
AnchorCal criterion decision or candidate selection.

---

## 114. Which hindsight definition controls Success Level 2?

**Decision**

There are two prespecified hindsight definitions, so report credible-set
coverage separately for:

- the criterion with the best real Spearman correlation to test WGA;
- the criterion with the lowest test selection regret.

Also report a conservative combined Level 2 label that is true only when the
credible set contains both definitions. Name that aggregate explicitly as
“both hindsight definitions”; do not present it as a replacement for either
separate coverage result. These are reporting-only quantities computed after
the practical selection receipt is frozen.

---

# Final Resolved Configuration Snapshot

```yaml
schema_version: anchorcal-config-v2

data:
  release: waterbird_1.0_forest2water2
  waterbirds100_definition: "complete official split 0; hard-assert y == place"
  canonical_id: img_id
  image_size: 224
  patch_size: 16
  interpolation: bicubic
  antialias: true
  normalization_mean: [0.5, 0.5, 0.5]
  normalization_std: [0.5, 0.5, 0.5]
  random_resized_crop:
    scale: [0.70, 1.00]
    ratio: [0.75, 1.3333333333]
  dilation_radius: 8

masks:
  source: waterbirds100_openclip_laion_dinovit_weclipplus_prediction_cmap
  mapping_mode: weclip_producer_first_with_explicit_legacy_fallbacks
  mapping_version: weclip-img-filename-v1
  decoder_version: pascal-voc-rgb-class-id-v1
  manifest_schema: anchorcal-vlm-mask-manifest-v2
  format: voc_colormap_class_ids
  foreground_class_ids: [1]
  allowed_class_ids: [0, 1]
  interpolation: nearest
  minimum_foreground_fraction: 0.0
  maximum_foreground_fraction: 1.0
  required_official_splits: [0, 1]
  optional_official_splits: [2]
  runtime_resolve_from_manifest_only: true

paths:
  repo_root: /home/ryreu/guided_cnn/BirdOnly
  waterbirds_root: /home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2
  metadata_path: /home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2/metadata.csv
  vlm_mask_root: /home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_waterbirds100_openclip_laion_dinovit/val/prediction_cmap
  hf_home: /home/ryreu/.cache/huggingface
  output_root: /home/ryreu/guided_cnn/BirdOnly/outputs/anchorcal/waterbirds100_pilot

pretrained:
  model: "hf_hub:timm/vit_small_patch16_224.augreg_in21k_ft_in1k"
  revision: "7e2c55630205e1266030f18370f4c6ed1a514b52"
  model_safetensors_sha256: "79c03c635cdfd798a364a9d8c4e5c0b7255b975ea2c9616046d4f77ab01435aa"
  timm_version: "1.0.28"

branches:
  frozen_epoch: 30
  foreground_train_seed: 6001
  background_train_seed: 6002
  background_sampling_seed: 6003
  copied_blocks: [0, 1, 2, 3, 4, 5]
  fine_tune_all: true
  foreground_position_encoder: [2, 128, 384]
  background_token_budget_candidates: [64, 48, 32]
  background_eval_views: 8
  no_background_sampling_replacement: true

anchorcal:
  lambda_step: 0.05
  lambda_interpolation: normalized_raw_logits
  anchor_subset_seed: 424242
  minimum_intersection_per_class: 50
  anchor_score_tolerance: 1.0e-10
  candidate_score_tolerance: 1.0e-8
  bootstrap_replicates: 200
  bootstrap_seed: 7002
  final_metric_bootstrap_replicates: 2000
  final_metric_bootstrap_seed: 7003

candidate_grid:
  head: "Linear(384, 2)"
  seed: 1234
  learning_rates: [1.0e-5, 3.0e-5, 1.0e-4]
  weight_decays: [0.01, 0.05]
  epochs: 40
  candidate_count: 240

optimization:
  adamw_betas: [0.9, 0.999]
  adamw_epsilon: 1.0e-8
  no_decay: "bias, one-dimensional parameters, LayerNorm, CLS, absolute position embeddings"
  scheduler_granularity: optimizer_update
  warmup_start_factor: 0.01
  cosine_min_factor: 0.01

criteria:
  eligible:
    - ordinary_accuracy
    - saliency_harmonic
    - token_swap_harmonic
    - background_blur_harmonic
  diagnostic_only:
    - foreground_only_harmonic
    - product_variants

compute:
  login: tigris.rc.rit.edu
  partition: tigris
  gpu: "gpu:gh200:1"
  selector_storage: candidate_outputs.h5
  reporting_only_storage: exploratory_hidden_metrics.h5
  production_requires_clean_commit: true
```

---

# Remaining Environment-Specific Preflight Items

All methodological questions are resolved.

The established TIGRIS data paths written into
`configs/anchorcal/paths.local.yaml` are:

```text
waterbirds_root: /home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2
metadata_path: /home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2/metadata.csv
vlm_mask_root: /home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_waterbirds100_openclip_laion_dinovit/val/prediction_cmap
```

Preflight must verify these exact roots rather than substitute Waterbirds-95,
historical WeCLIP+, or CUB paths. Their resolved values and the immutable VLM
mapping-manifest hash become part of every run manifest.
