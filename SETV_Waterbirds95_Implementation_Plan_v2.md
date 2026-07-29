# SETV Waterbirds95 Implementation Plan, Version 2

## 1. Purpose

This document specifies the first private implementation of **Spatial Evidence-Tilted Validation (SETV)** on Waterbirds95.

The immediate experimental question is:

> Can object-only and background-only evidence from auxiliary spatial experts identify ERM ViT epochs that generalize better to shortcut-conflicting groups than epochs selected by ordinary biased validation, uLA, or hard pseudo-groups?

This is a private development experiment. Waterbirds95 will be used to design and lock the method before broader evaluation.

The candidate model is always evaluated on the **original, untouched validation images**. Spatially modified images are used only to train and query the auxiliary object and background experts.

---

## 2. Locked high-level pipeline

1. Build one immutable 80/20 joint-group-stratified split from the official Waterbirds training split.
2. Keep the custom validation split approximately 95% bird-background correlated.
3. Train one object-only expert and three background-only experts.
4. Cache their raw logits and true-class margins on the custom biased validation set.
5. Train one 50-epoch ERM ViT candidate trajectory.
6. At every epoch, log candidate logits and all validation selectors while the model is in memory.
7. Use the official Waterbirds validation split as the oracle selector.
8. Use the official Waterbirds test split only for final performance reporting.
9. Compare:
   - ordinary biased validation;
   - uLA;
   - hard background pseudo-groups;
   - three SETV fusion variants.
10. Choose and freeze the background-expert design after this private pilot.

No intermediate candidate checkpoints are required.

---

## 3. Important update: no expert-calibration holdout

The primary implementation will **not reserve any candidate-training images for temperature calibration**.

All object and background experts train on the full custom `candidate_train` split.

The primary expert signals are raw true-class margins:

\[
m_i^O = z^O_{i,y_i} - \max_{c\ne y_i} z^O_{i,c},
\]

\[
m_i^B = z^B_{i,y_i} - \max_{c\ne y_i} z^B_{i,c}.
\]

The three fusion strategies are:

1. hard expert-disagreement rule;
2. class-conditional rank fusion;
3. repeated five-fold cross-fitted logistic fusion.

None requires removing data from `candidate_train`.

The logistic fusion is trained only on the custom biased validation set using out-of-fold prediction. It does not retrain either expert or the candidate model.

Temperature calibration may be added later as an ablation, but it is not part of the locked primary pipeline.

---

## 4. Dataset construction

### 4.1 Source splits

Use the official Waterbirds metadata and images.

Create:

- `candidate_train`: 80% of the official training split;
- `biased_val`: 20% of the official training split;
- `oracle_val`: official Waterbirds validation split;
- `test`: official Waterbirds test split.

### 4.2 Stratification

Stratify the 80/20 split using the joint pair:

```text
(target label, background/place label)
```

This preserves the original training correlation in both inner splits up to integer rounding.

Use one fixed seed:

```text
split_seed = 1729
```

Persist exact sample IDs.

Required files:

```text
splits/waterbirds95_candidate_train.csv
splits/waterbirds95_biased_val.csv
splits/waterbirds95_oracle_val.csv
splits/waterbirds95_test.csv
```

Log for every split:

- sample count;
- class count;
- joint-group count;
- empirical bird-background correlation.

Custom validation group labels must not be used by any realistic selector.

---

## 5. Masks

Use the existing VLM-generated segmentation masks stored on the research compute. These masks are the authoritative spatial maps for this pilot.

Before training:

1. identify and record the exact VLM-mask root on the research compute;
2. verify every relevant Waterbirds metadata row maps to exactly one VLM mask;
3. verify image and mask dimensions or document the resizing needed to align them;
4. confirm masks are binary or convert them using one fixed documented threshold;
5. visually inspect at least 20 random image-mask pairs from each split;
6. confirm that the masks accurately cover the bird and exclude most background;
7. record the mask-generation source, file format, root path, naming convention, and mapping rule in the run receipt.

All image augmentations that change geometry must be applied jointly to image and mask.

The implementation must not search for or require the original CUB segmentation archive. If a Waterbirds row has no matching VLM mask, report the missing sample IDs rather than silently substituting another mask source.

Use nearest-neighbor interpolation for masks.

---

## 6. Candidate classifier

Default:

```text
architecture = timm vit_small_patch16_224
pretrained = ImageNet
epochs = 50
optimizer = AdamW
learning_rate = 3e-5
weight_decay = 0.05
batch_size = 64
scheduler = cosine
warmup_epochs = 5
mixed_precision = true
```

Candidate training uses ordinary cross-entropy on `candidate_train`.

At every epoch, evaluate the current in-memory model on:

- `biased_val`;
- `oracle_val`;
- `test`.

Save per-example biased-validation logits and scalar metrics. Do not save all model states.

---

## 7. Object-only expert

### 7.1 Goal

Estimate how strongly the visible bird pixels themselves support the true class.

### 7.2 Input construction

For image \(x_i\) and bird mask \(M_i\):

\[
x_i^O = M_i\odot x_i + (1-M_i)\odot c_{\text{green}}.
\]

Use fixed raw RGB:

```text
green_rgb = (0, 255, 0)
```

Preserve:

- original bird position;
- original bird scale;
- silhouette;
- pose.

Do not center or randomly reposition the bird in the primary implementation.

### 7.3 Model

Use the same pretrained ViT-S/16 family as the candidate model.

Train on the full `candidate_train` split.

Recommended:

```text
epochs = 20
optimizer = AdamW
learning_rate = 3e-5
weight_decay = 0.05
batch_size = 64
scheduler = cosine
warmup_epochs = 2
```

### 7.4 Cached outputs

For every `biased_val` image, cache:

```text
sample_id
true_label
object_logits
object_true_class_margin
object_predicted_class
object_correct
```

No temperature scaling is required for the primary method.

---

## 8. Background experts

Test all three experts independently.

### 8.1 Background expert A: exact-mask green-fill baseline

Dilate the bird mask by 8 pixels at 224x224 resolution:

\[
D_i = \operatorname{Dilate}(M_i,8).
\]

Construct:

\[
x_i^{B,\text{exact}}
=
(1-D_i)\odot x_i + D_i\odot c_{\text{green}}.
\]

Train a standard pretrained ViT-S/16 with cross-entropy.

This is a simple baseline and may leak bird geometry.

---

### 8.2 Background expert B: sanitized-mask green-fill ViT

#### Goal

Preserve most real background structure while removing label information from the green occluder's geometry.

#### Sanitized mask requirement

Every sanitized mask \(S_i\) must contain the dilated bird mask:

\[
D_i\subseteq S_i.
\]

Implement three families:

1. expanded rectangle;
2. expanded ellipse;
3. randomized smooth blob.

Create eight deterministic masks per image.

Allocate families as 3/3/2, rotating which family gets two masks by deterministic sample-ID hash.

#### Leakage auditing

Before expert training, train mask-only auditors:

- logistic regression on mask geometry;
- gradient-boosted trees on mask geometry;
- small CNN on binary masks.

Geometry features:

- area;
- centroid;
- bounding-box width and height;
- aspect ratio;
- perimeter;
- compactness;
- second moments;
- family ID.

Accept a sanitization policy only if held-out balanced accuracy is at most:

```text
0.53
```

for binary Waterbirds.

If necessary, match sanitized mask geometry distributions across classes.

#### Training

For each training image, draw two distinct sanitized masks from its bank.

Create two green-filled views and optimize:

\[
\mathcal L_B
=
\frac{1}{2}\left[
\operatorname{CE}(z^{(1)},y)
+
\operatorname{CE}(z^{(2)},y)
\right]
+
\lambda_{\text{cons}}
\operatorname{SKL}(p^{(1)},p^{(2)}).
\]

Default:

```text
lambda_consistency = 0.5
epochs = 20
learning_rate = 3e-5
batch_size = 32 image pairs
```

At validation, average raw logits across all eight deterministic masks.

Cache:

```text
sample_id
background_sanitized_mean_logits
background_sanitized_true_class_margin
background_sanitized_predicted_class
background_sanitized_correct
background_sanitized_margin_std
```

---

### 8.3 Background expert C: variable-length background set transformer

#### Goal

Use only real background pixels without showing the model an artificial hole.

#### Patch selection

At 224x224 with patch size 16:

1. patchify the image;
2. compute foreground fraction using the dilated mask;
3. retain patches with foreground fraction <= 0.01.

Use all valid patches up to:

```text
max_background_tokens = 180
```

Require at least:

```text
min_background_tokens = 16
```

#### Token representation

Each token receives:

- learned visual patch embedding;
- learned coarse 3x3 spatial-bin embedding.

Do not provide exact coordinates.

Apply random token dropout during training:

```text
token_dropout = 0.20
```

#### Architecture

```text
embedding_dim = 384
transformer_blocks = 4
attention_heads = 6
mlp_ratio = 4
dropout = 0.1
pooling = learned attention pooling token
```

Initialize where possible from pretrained ViT-S/16:

- patch projection;
- first four transformer blocks;
- LayerNorms;
- pooling token from CLS.

Initialize coarse positional bins and classifier head from scratch.

#### Training

```text
epochs = 30
optimizer = AdamW
learning_rate = 1e-4
weight_decay = 0.05
batch_size = 64
scheduler = cosine
warmup_epochs = 3
```

At validation, average logits across eight random token-dropout views.

Cache:

```text
sample_id
background_set_mean_logits
background_set_true_class_margin
background_set_predicted_class
background_set_correct
background_set_margin_std
```

---

## 9. Expert margins

For every validation image \(i\), calculate:

\[
m_i^O = z^O_{i,y_i}-\max_{c\ne y_i}z^O_{i,c},
\]

\[
m_i^B = z^B_{i,y_i}-\max_{c\ne y_i}z^B_{i,c}.
\]

For binary classification:

- positive margin means the expert predicts the true class;
- negative margin means it predicts the wrong class;
- magnitude reflects separation in logit space.

All fusion methods operate on these cached margins.

---

## 10. Fusion strategy 1: hard expert-disagreement rule

Define:

\[
t_i
=
\mathbf 1[m_i^O>0 \land m_i^B<0].
\]

An informative example is one where:

- the object expert is correct;
- the background expert is wrong.

### Hard-rule validation score

For each class, evaluate candidate accuracy only on examples satisfying \(t_i=1\).

Then average across classes.

If either class has fewer than:

```text
min_hard_examples_per_class = 5
```

the hard-rule selector is marked invalid for that run.

Tie-break using candidate cross-entropy on the same hard subset, then ordinary validation accuracy, then earlier epoch.

This is a baseline, not the expected final method.

---

## 11. Fusion strategy 2: class-conditional rank fusion

Within each true class, convert object and background margins to percentile ranks.

\[
r_i^O
=
\operatorname{PercentileRank}(m_i^O\mid y_i),
\]

\[
r_i^B
=
\operatorname{PercentileRank}(m_i^B\mid y_i).
\]

Largest margin receives rank near 1.

Define:

\[
q_i^{\text{rank}}
=
r_i^O(1-r_i^B).
\]

Interpretation:

- high object evidence relative to same-class images;
- low background evidence relative to same-class images.

Because percentile ranks are invariant to positive monotonic rescaling, this method does not require calibration.

Clip:

```text
q_rank = clamp(q_rank, 1e-3, 1.0)
```

---

## 12. Fusion strategy 3: repeated cross-fitted logistic fusion

### 12.1 Purpose

Learn a smooth informativeness score from the two expert margins without holding out candidate-training data.

The fusion model is tiny logistic regression. The object and background experts are trained once.

### 12.2 Target

Use the hard expert-disagreement event:

\[
t_i
=
\mathbf 1[m_i^O>0 \land m_i^B<0].
\]

The logistic model estimates:

\[
q_i^{\text{logistic}}
=
P(t_i=1\mid \phi_i).
\]

### 12.3 Features

Primary feature vector:

\[
\phi_i
=
[
m_i^O,\,
m_i^B,\,
m_i^O-m_i^B,\,
m_i^Om_i^B
].
\]

Before fitting each fold:

- standardize features using the training folds only;
- apply the same scaler to the held-out fold.

### 12.4 Repeated five-fold cross-fitting

Use:

```text
n_folds = 5
n_repeats = 5
```

For each repeat:

1. create a new deterministic stratified five-fold partition;
2. stratify jointly by true class and target \(t_i\) when feasible;
3. for each fold:
   - fit logistic regression on the other four folds;
   - predict \(q_i\) for the held-out fold;
4. stitch the five held-out predictions together.

Every example receives one out-of-fold score per repeat.

Average across repeats:

\[
q_i^{\text{logistic}}
=
\frac{1}{5}
\sum_{r=1}^{5}q_{i,r}^{\text{OOF}}.
\]

This requires 25 tiny logistic fits, not expert retraining.

Recommended logistic settings:

```text
penalty = L2
C = 1.0
class_weight = balanced
solver = lbfgs
max_iter = 1000
```

### 12.5 Degenerate-target fallback

If the target has fewer than 10 positives overall or cannot support stratified folds:

1. reduce to the largest feasible fold count, minimum 2;
2. if cross-fitting remains impossible, mark logistic fusion unavailable;
3. do not silently train and score on the same examples.

### 12.6 Diagnostics

Report:

- out-of-fold ROC AUC against \(t_i\);
- out-of-fold PR AUC;
- Brier score;
- distribution of \(q_i^{\text{logistic}}\).

These diagnose the smooth approximation, but the primary endpoint remains candidate checkpoint selection.

---

## 13. Converting continuous fusion scores into validation weights

Apply the alpha curve separately to:

- rank fusion;
- cross-fitted logistic fusion.

For each fusion score \(q_i\), optionally convert it to a within-class percentile rank before weighting.

### Locked primary choice

To keep weighting comparable across fusion methods, convert both continuous scores to class-conditional percentiles:

\[
u_i
=
\operatorname{PercentileRank}(q_i\mid y_i).
\]

Clip:

```text
u_i = clamp(u_i, 1e-3, 1.0)
```

For:

\[
\alpha\in\{0.5,1,2,4\},
\]

define within-class weights:

\[
w_i^{(\alpha)}
=
\frac{u_i^\alpha}
{\sum_{j:y_j=y_i}u_j^\alpha}.
\]

---

## 14. Candidate validation curve

For candidate epoch \(f\), compute class-balanced weighted accuracy:

\[
A_f(\alpha)
=
\frac{1}{C}
\sum_{c=1}^C
\sum_{i:y_i=c}
w_i^{(\alpha)}
\mathbf 1[f(x_i)=y_i].
\]

Compute weighted cross-entropy similarly:

\[
L_f(\alpha)
=
\frac{1}{C}
\sum_{c=1}^C
\sum_{i:y_i=c}
w_i^{(\alpha)}
\operatorname{CE}(f(x_i),y_i).
\]

The final continuous-selector score is:

\[
\operatorname{SETV}(f)
=
\frac{
A_f(0.5)+A_f(1)+A_f(2)+A_f(4)
}{4}.
\]

Log:

- each \(A_f(\alpha)\);
- each \(L_f(\alpha)\);
- effective sample size by class and alpha.

\[
\operatorname{ESS}_c(\alpha)
=
\frac{1}{\sum_{i:y_i=c}(w_i^{(\alpha)})^2}.
\]

Warn if:

```text
ESS_c(alpha=4) < 10
```

Do not automatically drop alpha 4 in the first pilot.

---

## 15. SETV variants

For each background expert, evaluate:

1. `SETV-rank`
2. `SETV-logistic`
3. `SETV-hard`

This produces:

```text
exact-fill + hard
exact-fill + rank
exact-fill + logistic

sanitized-fill + hard
sanitized-fill + rank
sanitized-fill + logistic

set-transformer + hard
set-transformer + rank
set-transformer + logistic
```

The primary comparison should make clear that expert construction and fusion strategy are separate design choices.

---

## 16. Baseline selectors

### 16.1 Ordinary biased validation

Select highest biased-validation accuracy.

Tie-break:

1. lower biased-validation loss;
2. earlier epoch.

### 16.2 Oracle validation

Select highest official-validation WGA.

Tie-break:

1. higher group-balanced accuracy;
2. higher average accuracy;
3. earlier epoch.

### 16.3 Hard background pseudo-groups

Using each background expert:

\[
\hat b_i = \arg\max_c z^B_{i,c}.
\]

Define proxy groups:

\[
\tilde g_i=(y_i,\hat b_i).
\]

Select using worst nonempty proxy-group accuracy.

### 16.4 uLA

Use official uLA code if possible.

Record:

- repository;
- commit;
- SSL backbone;
- bias proxy;
- proxy grouping;
- validation formula.

If exact reproduction is not possible, label `uLA-style`.

---

## 17. Per-epoch logging

For each candidate epoch, log on `biased_val`:

- sample ID;
- logits;
- predicted class;
- per-example cross-entropy;
- correctness.

Compute:

- ordinary accuracy and loss;
- uLA score;
- hard pseudo-group score;
- all hard SETV scores;
- all rank SETV alpha scores and aggregate;
- all logistic SETV alpha scores and aggregate.

On `oracle_val`, log:

- average accuracy;
- group-balanced accuracy;
- WGA;
- per-group accuracy.

On `test`, log the same metrics under a separate namespace.

The selector implementation must not access oracle or test metrics.

---

## 18. Tie-breaking

### Continuous SETV selectors

1. highest SETV accuracy;
2. lowest SETV weighted loss;
3. highest ordinary biased-validation accuracy;
4. lower ordinary biased-validation loss;
5. earlier epoch.

### Hard SETV selector

1. highest hard-subset class-balanced accuracy;
2. lowest hard-subset cross-entropy;
3. highest ordinary validation accuracy;
4. earlier epoch.

### Numerical tolerance

```text
score_tolerance = 1e-8
```

---

## 19. Choosing the background expert

Do not choose by background-only accuracy alone.

Primary private-development criteria:

1. oracle selection regret;
2. Spearman correlation between selector score and oracle WGA;
3. Kendall correlation;
4. pairwise epoch-ranking accuracy.

Secondary criteria:

- leakage-auditor performance;
- stability across mask or token views;
- simplicity;
- compute.

Treat experts as effectively tied if:

```text
absolute regret difference <= 0.005
and
absolute Spearman difference <= 0.02
```

If tied:

1. prefer lower leakage;
2. prefer greater stability;
3. prefer simpler implementation;
4. if still tied, prefer sanitized fill.

Do not choose randomly.

---

## 20. Primary output table

| Selector | Background expert | Fusion | Selected epoch | Oracle val WGA | Test avg acc | Test WGA | Oracle selection regret |
|---|---|---|---:|---:|---:|---:|---:|
| Ordinary validation | N/A | N/A | | | | | |
| uLA | uLA proxy | hard proxy groups | | | | | |
| Hard pseudo-groups | Exact | hard groups | | | | | |
| SETV | Exact | hard | | | | | |
| SETV | Exact | rank | | | | | |
| SETV | Exact | logistic | | | | | |
| SETV | Sanitized | hard | | | | | |
| SETV | Sanitized | rank | | | | | |
| SETV | Sanitized | logistic | | | | | |
| SETV | Set | hard | | | | | |
| SETV | Set | rank | | | | | |
| SETV | Set | logistic | | | | | |
| Oracle | N/A | true groups | | | | | |

---

## 21. Key diagnostics

### Expert diagnostics

- object margin histogram;
- background margin histogram;
- object versus background margin scatter;
- expert correctness contingency table;
- count of hard informative examples by class.

### Fusion diagnostics

- rank score distribution;
- logistic OOF score distribution;
- hard versus logistic agreement;
- ROC AUC and PR AUC for logistic against the hard target;
- high-score image galleries.

### Candidate diagnostics

- candidate disagreement by fusion-score decile;
- candidate accuracy by fusion-score decile;
- SETV alpha curves for representative epochs;
- selector score versus oracle WGA.

---

## 22. Kill criteria

Revise the method if:

- the object expert is near chance;
- background margins have negligible variation;
- hard informative examples are absent at Waterbirds95;
- logistic cross-fitting is degenerate;
- high fusion-score images are simply hard for every candidate;
- rank and logistic fusion do not outperform background confidence alone;
- all realistic selectors choose essentially identical epochs;
- SETV does not improve oracle selection regret over ordinary validation.

---

## 23. Implementation order

### Phase 0

- create and persist split;
- verify VLM masks and their metadata mapping;
- implement shared transforms and logging.

### Phase 1

- train object expert;
- cache object logits and margins.

### Phase 2

- train exact-fill background expert;
- cache margins;
- implement all three fusion strategies;
- run first end-to-end SETV computation.

### Phase 3

- implement mask sanitization and auditors;
- train sanitized-fill expert;
- repeat all fusion comparisons.

### Phase 4

- implement background set transformer;
- repeat all fusion comparisons.

### Phase 5

- train 50-epoch ERM candidate;
- log all selector metrics every epoch.

### Phase 6

- add uLA;
- create result tables and plots;
- choose and freeze background expert and fusion design.

---

## 24. Compact pseudocode

```python
# Experts have already been trained on full candidate_train.
m_object = true_class_margin(object_logits, labels)
m_background = true_class_margin(background_logits, labels)

# 1. Hard rule
hard_target = ((m_object > 0) & (m_background < 0)).astype(int)

# 2. Rank fusion
r_object = within_class_percentile(m_object, labels)
r_background = within_class_percentile(m_background, labels)
q_rank = r_object * (1.0 - r_background)

# 3. Repeated cross-fitted logistic fusion
features = stack([
    m_object,
    m_background,
    m_object - m_background,
    m_object * m_background,
], axis=1)

q_logistic = repeated_cross_fitted_logistic(
    features=features,
    target=hard_target,
    labels=labels,
    n_folds=5,
    n_repeats=5,
)

def continuous_setv(candidate_logits, q, labels):
    u = within_class_percentile(q, labels)
    u = clip(u, 1e-3, 1.0)

    alpha_acc = []
    alpha_loss = []

    for alpha in [0.5, 1.0, 2.0, 4.0]:
        accs = []
        losses = []

        ce = cross_entropy_per_example(candidate_logits, labels)
        pred = candidate_logits.argmax(axis=1)
        correct = (pred == labels).astype(float)

        for c in unique(labels):
            idx = where(labels == c)
            weights = u[idx] ** alpha
            weights /= weights.sum()

            accs.append((weights * correct[idx]).sum())
            losses.append((weights * ce[idx]).sum())

        alpha_acc.append(mean(accs))
        alpha_loss.append(mean(losses))

    return {
        "score": mean(alpha_acc),
        "loss": mean(alpha_loss),
        "alpha_accuracy": alpha_acc,
        "alpha_loss": alpha_loss,
    }
```

---

## 25. Final locked answer on data use

None of the following requires removing samples from candidate training:

- hard rule;
- rank fusion;
- repeated cross-fitted logistic fusion.

The experts train on the full `candidate_train` split.

The logistic fusion uses the custom `biased_val` labels and cached expert margins. Every validation image receives only out-of-fold logistic predictions, and repeated partitions are averaged.

This is permitted because validation metrics already use validation labels. Cross-fitting is required only to prevent the small fusion model from scoring the same validation image it was fitted on.
