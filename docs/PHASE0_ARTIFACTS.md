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

`masks/vlm_mask_manifest.csv` binds every sample to exactly one VLM mask and
records dimensions, threshold statistics, foreground fraction, file size,
mapping rule, and SHA-256.

`mask_audit/` contains deterministic contact sheets for at least 20 examples
from each of the four splits. Phase 0 is incomplete until a human reviewer
inspects these sheets and creates `visual_review_approval.json`.

## Integrity

`artifact_manifest.json` hashes every base artifact. The visual approval
receipt is additionally bound to the base-manifest hash and contact-sheet
hashes. `scripts/verify_phase0.py` checks both layers.

Test metrics and model checkpoints do not exist in Phase 0.
