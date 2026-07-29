# Phase 3: sanitized masks, leakage auditors, and background expert

Phase 3 implements the sanitized-fill background expert and repeats the hard,
rank, and repeated cross-fitted logistic SETV comparisons.

## Sanitized mask bank

The approved Phase 0 VLM masks remain authoritative. At the canonical Phase 0
evaluation resolution, each bird mask is dilated by the equivalent of eight
pixels at 224×224. Every generated sanitized mask is checked to contain that
dilated foreground.

Each image receives eight deterministic masks from three families:

- expanded rectangle;
- expanded ellipse;
- low-frequency smooth radial blob.

The family allocation is 3/3/2. Sample IDs are sorted by SHA-256 digest and the
two-mask family rotates through the three families, which makes total
short-family counts differ by at most one across `candidate_train` and
`biased_val`. Per-mask random seeds are derived from the explicit bank seed,
sample ID, family, and family-local variant.

The complete generated masks are persisted in bit-packed NPZ files. Family
IDs, exact per-mask seeds, image dimensions, and the dilation radius are saved
alongside them. Consequently, later reproduction does not depend on rerunning
the shape generator.

Version 2 specifies the families and allocation but not their numeric shape
distribution. The receipt-bound first implementation samples target area from
`[0.35, 0.60]`, target aspect ratio from `[0.75, 4/3]`, uses 10% center jitter,
and applies 1.08 containment slack. Smooth blobs use four low-frequency
harmonics with coefficient magnitude at most 0.12.

## Leakage gate

Auditors are trained only on `candidate_train` sanitized masks. The train/test
partition is made at the image/sample-ID level and stratified by bird class;
mask variants from one image can never cross the boundary.

The three required auditors are:

- L2 logistic regression on standardized geometry features;
- gradient-boosted trees on geometry features;
- a small CNN receiving only binary masks.

Geometry features include area, centroid, bounding-box dimensions, aspect
ratio, perimeter, compactness, second moments, and one-hot family ID.

For every auditor, acceptance requires both:

```text
held-out per-mask balanced accuracy <= 0.53
```

and:

```text
the 95% image-cluster-bootstrap confidence interval contains 0.50
```

The bootstrap resamples held-out images and keeps all eight masks from each
sampled image together. Image-aggregated balanced accuracy is also reported.
A rejected bank is retained for diagnosis, but the command exits nonzero and
the expert training gate refuses it.

The original production rule remains fail-closed. The real-data bank from
audit job 21917 failed all three auditors and remains recorded as rejected.
The project owner subsequently authorized an explicit private-pilot diagnostic
amendment: use that immutable bank as a selector signal without treating it as
sanitization-compliant. This does not change the audit result.

Only the dedicated resume launcher may exercise the amendment:

```bash
bash scripts/submit_phase3_sanitized_diagnostic_resume.sh
```

It reuses and hash-verifies the existing bank, then submits only smoke,
expert training, and fusion. It does not regenerate, edit, accept, or
overwrite the rejected bank. Expert, fusion, Phase 5, and Phase 6 receipts
must carry:

```text
leakage_audit_accepted = false
rejected_bank_diagnostic_override_used = true
sanitization_claim_eligible = false
```

Results from this branch are diagnostic evidence and may participate in the
private pilot's expert-fusion comparison, but they cannot support a claim that
mask geometry was sanitized successfully.

## Sanitized expert

Training uses the complete `candidate_train` split. Each epoch deterministically
draws two distinct bank masks per image and minimizes:

```text
0.5 * (CE(view_a) + CE(view_b))
  + 0.5 * symmetric_KL(view_a, view_b)
```

The model is an ImageNet-pretrained `vit_small_patch16_224`, trained for 20
epochs with AdamW, learning rate `3e-5`, weight decay `0.05`, batch size 32
image pairs, cosine decay, and two warmup epochs.

The bank is defined in canonical evaluation coordinates. To preserve the exact
persisted-mask identity and containment guarantee, this first implementation
does not apply an additional random geometric crop. This is an explicit
receipt-bound choice because v2 does not lock a separate augmentation policy
for the sanitized expert.

Validation evaluates all eight deterministic masks. Raw view logits are
averaged first and used once for prediction and loss. No temperature scaling
or calibration split exists. The score artifact contains mean logits,
true-class margin, predicted class, correctness, and the standard deviation of
the eight view-level true-class margins.

Only the final sanitized-expert checkpoint is retained.

## Fusion and execution

The sanitized true-class margins reuse the Phase 2 implementations of:

- hard expert disagreement;
- class-conditional rank fusion;
- repeated 5×5 out-of-fold logistic fusion;
- the alpha curve `{0.5, 1, 2, 4}` and ESS diagnostics.

Candidate scoring accepts only sample-ID/label-aligned logits from untouched
validation images.

The ordinary fail-closed launcher loads all four frozen seeds from
`configs/campaign_waterbirds95.yaml`:

```bash
bash scripts/submit_phase3_sanitized.sh
```

For the operator-authorized job-21917 rejected-bank diagnostic resume, use:

```bash
bash scripts/submit_phase3_sanitized_diagnostic_resume.sh
```

The Tigris dependency chain is:

```text
mask generation + held-out leakage audit
  -> real one-epoch GH200 expert smoke
  -> 20-epoch sanitized expert
  -> sanitized hard/rank/logistic fusion and verification
```
