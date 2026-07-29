# Phase 0 artifact contract

Phase 0 creates one immutable directory. The builder refuses to overwrite an
existing directory and publishes its output only after all automated checks
pass.

The production launcher is fail-closed on source provenance: it requires a
clean Git checkout, exports the submitted commit to Slurm, and the job refuses
to run if either the commit or worktree state changes before execution.

## Selector-safe artifacts

The four files under `splits/` contain only:

```text
sample_id
metadata_index
img_filename
y
official_split
split_name
mask_relative_path
```

They deliberately omit `place` and `group`.

## Protected artifacts

`private_analysis/protected_group_labels.csv` contains per-sample place and
group labels. It is analysis-only and must never be loaded by realistic
selectors.

## Mask artifacts

`masks/vlm_mask_manifest.csv` binds every mask-required sample from official
splits 0 and 1 to exactly one VLM mask and records dimensions, decoded RGB
colors, foreground fraction, file size, mapping rule, and SHA-256. Official
test rows retain an empty `mask_relative_path`; candidate test evaluation
loads only their untouched RGB images.

The current Waterbirds WeCLIP+ generator names each flat mask by normalizing
and flattening the complete relative image stem. For example:

```text
001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg
-> 001_Black_footed_Albatross_Black_Footed_Albatross_0046_18.png
```

The Phase 0 Slurm job audits this exact mapping and VOC color decoding across
all required official train/validation rows before running the test suite or
constructing artifacts. Its JSON report separately records optional test-mask
coverage without making it a readiness gate.

The same preflight creates deterministic, split-aware gallery pages. Each row
shows four views of one sample:

1. original image;
2. red VLM-mask overlay;
3. the object expert's evaluation view with a green background;
4. the exact-background expert's evaluation view with the dilated bird region
   filled green.

It requests 24 mapped samples from candidate train, biased validation, and
oracle validation, and saves eight samples per lossless PNG page. Gallery
paths, sample IDs, transformation parameters, and file hashes are bound into
the mapping-audit JSON.

`prediction_cmap` masks are Pascal/VOC-colorized class-index maps. Background
class 0 is RGB `[0, 0, 0]` and foreground class 1 is RGB `[128, 0, 0]`.
The loader preserves RGB, decodes exact VOC class IDs, selects class 1, and
rejects unknown colors. The retained `1/255` threshold applies only to an
explicit legacy-threshold fallback and is not the production decoder.

`mask_audit/` contains deterministic contact sheets for at least 20 examples
from each mask-required constructed split. Phase 0 is incomplete until a human reviewer
inspects these sheets and the four-view green-screen galleries, then creates
`visual_review_approval.json`. The approval receipt hashes both sets of pages
and explicitly attests that the object and exact-background compositions are
correct.

## Integrity

`artifact_manifest.json` hashes every base artifact. The visual approval
receipt is additionally bound to the base-manifest hash and contact-sheet
hashes. `scripts/verify_phase0.py` checks both layers.

Test metrics and model checkpoints do not exist in Phase 0.
