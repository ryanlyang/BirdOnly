# AnchorCal implementation audit log

This log records the mandatory self-review for each implementation phase.
Local tests use generated fixtures and small synthetic models. Production
evidence is created separately under the frozen campaign output root; no entry
below claims that the real dataset or TIGRIS GH200 was available locally.

## Authoritative Waterbirds100 dataset and VLM-mask correction

- A prior inferred lock incorrectly defined this pilot by filtering the
  partially biased Waterbirds-95 release, despite the TIGRIS handoff already
  documenting the dedicated Waterbirds-100 dataset and mask bank. That
  inference is superseded; it must not be used for a production launch.
- The authoritative dataset is
  `/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2`, with metadata
  at `<waterbirds_root>/metadata.csv`. Its complete official split 0 must
  already satisfy `y == place`; preflight hard-fails rather than silently
  discarding counterexamples. Official split 1 remains oracle-only and split 2
  remains reporting-only.
- The binding specifications replace the earlier CUB-mask assumption with the
  matching Waterbirds-100 OpenCLIP-LAION + DINOvIT VLM bank at
  `/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_waterbirds100_openclip_laion_dinovit/val/prediction_cmap`.
- The corrected contract requires a producer-first join from the complete
  metadata `img_filename`, strict Pascal/VOC class-1 decoding, complete
  one-to-one public/runtime coverage for official split 0, no official split-2
  mask requirement, and an immutable split-0-only
  `preflight/mask_manifest.json` with schema
  `anchorcal-vlm-mask-manifest-v3` and a deterministic content hash. The mask
  source identifier is
  `waterbirds100_openclip_laion_dinovit_weclipplus_prediction_cmap`.
  Public mask entries omit `metadata_index` and all context/group fields.
- The preceding incompatible split/privacy/visual-audit correction bumped the
  package to `0.5.0`, the resolved config to `anchorcal-config-v3`, the
  selector-safe split manifest to `anchorcal-splits-v4`, the protected split
  manifest to `anchorcal-analysis-only-splits-v1`, and the mask manifest to
  `anchorcal-vlm-mask-manifest-v3`. That version record is retained as
  historical evidence; the combined background-budget correction below
  supersedes the active package/configuration contract with AnchorCal `0.6.0`
  and `anchorcal-config-v4`. The split and mask schemas remain unchanged.
- The development split now imports and hash-verifies the exact Waterbirds100
  FCV seed-0 membership from the established `split_manifests` bundle; it never
  regenerates a seed-1729 split. Selector-safe development/expert CSVs omit
  metadata index, context, and group fields. The public split manifest exposes
  no protected rows, IDs, counts, paths, or hashes. Protected oracle/test
  records live only under `analysis_only/splits/`.
- The user-audited VLM bank now also produces the deterministic 18-sample,
  three-page `anchorcal-mask-visual-audit-v1` artifact under
  `preflight/mask_visual_audit/`. All examples come from split 0: three
  deterministic representatives from every low/middle/high area stratum in
  each of its two aligned class/context cells. It never serializes `place` or
  exposes split-1 membership. Its machine integrity is mandatory, while
  `human_approval_required=false` avoids a blocking manual launch gate.
- Replaced both CUB path keys with one locked `vlm_mask_root`; added the exact
  producer, mapping, decoder, VOC class, nearest-interpolation, split-coverage,
  and manifest-only-runtime configuration locks.
- Added `src/anchorcal/vlm_masks.py`; rewrote preflight to require and decode
  the public/runtime split-0 bank, inventory rather than require split 2,
  freeze the public manifest before geometry construction, and bind file,
  decoded-mask, mapping, and decoder hashes. A split-1 per-row machine audit is
  isolated at
  `analysis_only/masks/waterbirds100_oracle_val_mask_audit.json` under schema
  `anchorcal-analysis-only-vlm-mask-audit-v1`.
- Added `preflight/selector_mask_receipt.json` under schema
  `anchorcal-selector-mask-receipt-v1`. Final selector provenance reads only
  this compact hash/aggregate-identity receipt; it cannot parse the public
  per-row manifest, import the full loader, or access the protected split-1
  audit. `preflight/report.json` contains no protected audit path or hash; the
  protected file self-binds and hidden/campaign verification enforces its fixed
  location.
- Replaced every branch, geometry, anchor, and practical-candidate CUB lookup
  with canonical `img_id` plus `img_filename` access through the frozen VLM
  manifest. Candidate ERM training and oracle/test classification remain
  mask-free.
- Bumped incompatible branch and candidate provenance schemas, and bound the
  VLM root, source, contract, bank hash, and manifest hash through restart,
  branch, anchor, candidate, decision, and final-campaign verification.
- On 2026-08-09, the preceding `0.4.0` correction passed **171 AnchorCal tests**
  and all **81 retained SETV tests** (**252 total**). That is historical evidence
  and does not by itself validate the later `0.5.0` split/privacy/visual-audit
  changes; their final regression evidence must be recorded after reconciliation.
- On 2026-08-09, the completed `0.5.0` reconciliation passed **194 AnchorCal
  tests** and all **81 retained SETV tests** (**275 total**), Python compilation,
  every AnchorCal shell/Slurm syntax check, and `git diff --check`. The final
  selector hardening deliberately injected an unexpected `place` dataset, root
  dataset, root attribute, and anchor-manifest file; each was rejected by the
  exact public allowlists.
- Real Waterbirds/VLM coverage, the pinned model cache, and GH200 execution
  remain production preflight and smoke gates on TIGRIS; local fixtures do not
  claim that external runtime evidence.
- On 2026-08-09, live TIGRIS preflight job `70443` exposed a deterministic
  construction cycle: geometry was needed for final `preflight/report.json`,
  while the geometry mask loader required that not-yet-written report. The
  repair adds one preflight-only bootstrap loader that verifies the frozen
  mask manifest, all source bytes, the compact selector receipt, and the full
  visual-audit receipt before injecting the verified bank into geometry.
  Every downstream consumer remains on the strict loader and must bind to a
  finalized passed report. No dataset, mask, split, model, or scientific
  decision changed. The repaired checkout passed **198 AnchorCal tests** and
  all **81 retained SETV tests** (**279 total**), Python compilation, every
  AnchorCal shell/Slurm syntax check, and `git diff --check`.
- On 2026-08-09, repaired TIGRIS preflight job `70471` completed, but it
  selected `K=48` under the then-active 95-percent coverage-only rule. Debug
  job `70472` subsequently failed the downstream overall `biased_val`
  invalidity assertion: `K=48` invalidated 20 of 959 examples (2.0855
  percent). The complete deterministic geometry census was `K=64`: 44/959
  invalid (4.5881 percent), `K=48`: 20/959 invalid (2.0855 percent), and
  `K=32`: 9/959 invalid (0.9385 percent). Production and debug geometry agreed.
- The project owner approved resolving that contradictory two-stage contract
  by selecting the largest `K` in `[64, 48, 32]` that jointly satisfies at
  least 95 percent overall/per-class coverage in `expert_train`,
  `expert_calibration`, and `biased_val`, and at most 1 percent overall
  `biased_val` invalidity. The 1-percent condition is not per class. The joint
  gate now runs in preflight before training and is reasserted downstream as
  defense in depth. The current locked inputs therefore select `K=32`.
- This scientific-contract amendment is AnchorCal `0.6.0` with resolved config
  schema `anchorcal-config-v4`. The job-70471/70472 campaign artifacts are
  historical failed-run evidence, not reusable `0.6.0` inputs; production must
  start from a fresh output root and frozen campaign manifest.
- The completed `0.6.0` amendment passed **207 AnchorCal tests** and all **81
  retained SETV tests** (**288 total**), including strict geometry
  recomputation and saved-validity-vector tamper regressions. Python
  compilation, every AnchorCal shell/Slurm syntax check, and
  `git diff --check` also passed.
- On 2026-08-09, campaign `20260809T181257Z_3970389` at commit `f4ab402`
  passed preflight job `70512`, including the new combined budget decision,
  and debug job `70513` verified the full preflight bundle before reaching the
  foreground invariance audit. That audit then exposed one GH200
  serialization-boundary bug: the locked evaluation autocast correctly
  produced BF16 logits and patch activations, but four floating-tensor exports
  called NumPy without first converting to a supported dtype. This was not an
  invariance failure and did not change a model forward, mask, split, token
  budget, score, or audit threshold.
- AnchorCal `0.6.1` fixes that boundary through one tested helper that detaches
  autocast floating outputs and converts them to FP32 before CPU/NumPy export;
  integer source indices retain their dtype. The locked BF16 forward and
  `anchorcal-config-v4` scientific contract are unchanged. Because queued jobs
  are frozen to a commit and package lock, the `f4ab402` campaign cannot be
  resumed under the repair and must be archived before a fresh full launch.
- The completed `0.6.1` repair passed **208 AnchorCal tests** and all **81
  retained SETV tests** (**289 total**), including a direct BF16-to-NumPy
  regression. Python compilation, every AnchorCal shell/Slurm syntax check,
  and `git diff --check` also passed.
- On 2026-08-09, the repaired campaign at commit `5dcdc2f` passed preflight and
  debug job `70613` passed the complete branch-audit stage, confirming the
  BF16 export repair. The next fail-closed gate exposed a distinct numerical
  mismatch: direct anchor mixing was performed on BF16 logits while the
  algebraic cache mixed their exported FP32 values. The maximum logit
  discrepancy was `0.0008223652839660645`, correctly exceeding the locked
  `1e-6` parity threshold.
- AnchorCal `0.6.2` preserves ordinary BF16 training/inference, performs
  post-logit centering, margin normalization, and lambda mixing in FP32, and
  activates the already-prespecified FP32 fallback for saliency
  forwards/gradients. Endpoint-cache saliency and direct saliency now share
  that FP32 path; classification/intervention provenance still comes from
  independent BF16 inference. The original `1e-6` logit/criterion and `1e-5`
  saliency parity limits remain unchanged. The scientific config remains
  `anchorcal-config-v4`.
- The completed `0.6.2` precision repair passed **210 AnchorCal tests** and all
  **81 retained SETV tests** (**291 total**), including explicit BF16-logit
  FP32-mixing cache parity and FP32-saliency-context regressions. Python
  compilation, every AnchorCal shell/Slurm syntax check, and
  `git diff --check` also passed.
- On 2026-08-09, debug job `70634` at commit `a35c771` passed preflight,
  branch training/audits, anchor evaluation, and the strict direct/cache parity
  gate. Candidate initialization then failed before its first optimizer update
  because the three-epoch debug trajectory inherited the production setting of
  four warmup epochs. Production's 40/4 schedule was unaffected.
- AnchorCal `0.6.3` explicitly locks the miniature candidate trajectory to
  three epochs with one warmup epoch. Cross-field configuration validation now
  rejects any branch or candidate warmup longer than its training trajectory,
  and the launcher validates both production and debug configs before any
  Slurm submission, moving this failure ahead of GPU allocation.
  Production remains 40 candidate epochs with four warmup epochs, and the
  scientific configuration schema remains `anchorcal-config-v4`.
- The completed `0.6.3` debug-schedule repair passed **212 AnchorCal tests** and
  all **81 retained SETV tests** (**293 total**), including the new locked-debug
  schedule, invalid cross-field schedule, and pre-submission validation
  regressions. Python compilation, every AnchorCal shell/Slurm syntax check,
  and `git diff --check` also passed.
- On 2026-08-09, debug job `70710` at commit `50ce50c` passed preflight,
  branch training/audits, anchor evaluation/cache parity, all three candidate
  training epochs, paired HDF5 publication, and checkpoint completion. Its
  immediate final read-only storage verification then failed with HDF5
  `BlockingIOError`/`EAGAIN` while reopening the published selector file on the
  TIGRIS campaign filesystem. The failure was operational and occurred after
  candidate computation and publication, not in model training or selection.
- AnchorCal `0.6.4` exports `HDF5_USE_FILE_LOCKING=FALSE` consistently in every
  frozen Slurm stage before h5py import and records the setting in each job
  receipt. HDF5's redundant filesystem lock is unnecessary here: candidate
  writes retain the explicit per-run `fcntl` writer lock, paired transaction
  journal, atomic renames, immutable publication, and SHA-256 manifest/schema
  verification. The scientific configuration remains `anchorcal-config-v4`.
- The completed `0.6.4` TIGRIS HDF5 repair passed **213 AnchorCal tests** and
  all **81 retained SETV tests** (**294 total**), including a regression that
  binds the pre-import runtime setting, its job-receipt field, and continued
  explicit candidate writer locking. Python compilation, every AnchorCal
  shell/Slurm syntax check, and `git diff --check` also passed.

## Phase 0: specification and repository audit

- Read both binding specifications in full. All 114 answered locks, especially
  residual locks 97-114, take precedence over earlier plan wording.
- Inspected the existing repository, dependency metadata, historical `setv`
  package, TIGRIS handoff, and established `/home/ryreu/guided_cnn/BirdOnly`
  checkout convention.
- Created and maintained `IMPLEMENTATION_TRACEABILITY.md`. At that time,
  machine-local data and CUB-mask paths were deliberately unresolved; this
  historical assumption is superseded by the authoritative Waterbirds100
  dataset and VLM correction above.
- Resolved the explicit precedence issues listed at the end of the traceability
  file; no third behavior was invented.

## Phase 1: package, configuration, data, and preflight

- Implemented the independent `anchorcal` package, strict locked config merge,
  ignored `paths.local.yaml`, nonselecting path discovery, release/metadata
  validation, canonical IDs, mask provenance/encoding/dimension/uniqueness
  checks, deterministic split CSVs and hashes, model revision/hash locking,
  environment/package manifests, and clean production checkout guard.
- The original review addressed distinct CUB source/final roots through
  `anchorcal_mapping_manifest.json`. That historical contract is superseded:
  the corrected implementation must instead freeze the producer-derived VLM
  join and per-file hashes in `preflight/mask_manifest.json` using schema
  `anchorcal-vlm-mask-manifest-v3`.
- Review also strengthened production preflight to validate fixed TIGRIS repo,
  cache and output roots, the exact interpreter, aarch64, and GH200.
- Verification: configuration/split/mask tests plus generated-fixture
  same-tree, separate-root, tamper, release-basename, and metadata-placement
  preflight tests.

## Phase 2: shared transformations and deterministic geometry

- Implemented source-resolution green replacement before any resampling,
  shared image/mask geometry, bicubic antialiased image resize, nearest mask
  resize, exact ViT normalization/evaluation crop, stateless SHA-256-derived
  sampling, ten-attempt nonempty crop rejection, deterministic fallback and
  0.1% gate, final-grid disk dilation, and strict foreground/safe-background
  patch eligibility.
- Review checked interpolation order, edge leakage, empty/full masks, worker
  independence, replay, and final 14x14 patch coordinates. The fallback gate
  was corrected from an earlier looser value to the locked `0.001`.
- Verification: `test_transforms.py` and the transform/seed/mask portions of
  `test_data_scientific_core.py`.

## Phase 3: foreground and background branches

- Implemented independent copies of the patch projection, CLS token, first six
  blocks and norm; variable foreground tokens with object-relative positions
  and padding; position-free background sets; 64/48/32 no-replacement budget;
  persisted eight-view evaluation bank; exact optimizer/scheduler; fixed epoch
  30; scalar calibration; restart state; final checkpoint and full manifests.
  The original coverage-only budget-selection behavior is historical and is
  superseded by the `0.6.0` combined preflight gate recorded above.
- Review found invalid donor/background examples could otherwise propagate
  NaNs. Geometry/donor pools now require common eligibility and the selected
  fixed token budget; runtime batches fail closed on invalid entries.
- Review verified independent parameter storage, foreground padding
  invariance, background permutation invariance, raw-logit view averaging, and
  no group-label use in training.
- Verification: `test_model_training_boundaries.py` and
  `test_branch_anchor_scientific_core.py`.

## Phase 4: leakage and numerical audits

- Implemented foreground background-replacement invariance, three independent
  replacement checks, green-shade diagnostics, background patch-purity sweep,
  image-disjoint class-balanced real random-token audit, standardized logistic
  and two-layer-MLP geometry auditors, branch bootstrap gates, competence
  intersection, raw-logit scales, and per-class minimum gates.
- Review found the first foreground audit checked only one sample/replacement;
  it now checks up to 100 samples with three replacements and compares tokens,
  masks, padding, and logits. A tautological tensor comparison was replaced by
  an immutable reference copy and explicit typed intervention assertions.
- The geometry eligibility audit now persists per-class coverage and refuses
  fewer than 50 eligible examples per class.
- Verification: branch/anchor scientific-core, transform, model-boundary, and
  intervention-contract tests.

## Phase 5: differentiable anchor ladder

- Implemented centered raw logits, robust full-intersection median margin
  scales, the exact 21 lambdas, frozen activation-differentiable
  `RelianceAnchor`, lambda-restricted typed streams, source patch metadata,
  signed contribution caches, direct evaluation, correctness assertions, and
  direct/cache parity at 0, 0.35, 0.5, 0.8, and 1.
- Review found parity originally used only one example and did not explicitly
  gate repeated coordinates. It now uses a small batch across all parity
  lambdas and includes a repeated-coordinate case. Cached quantities apply
  lambda exactly once.
- Verification: `test_model_training_boundaries.py`,
  `test_branch_anchor_scientific_core.py`, and
  `test_criteria_scientific_core.py`.

## Phase 6: practical criteria

- Implemented full biased-validation accuracy, gradient saliency harmonic,
  fixed token-swap harmonic, mask-normalized blur harmonic, diagnostic
  foreground-only harmonic and diagnostic product variants. The implementation
  follows the locked within-image, within-class, then across-class order.
- Fixed donor IDs are shared between candidate/anchor evaluation while exact
  token mappings remain architecture-appropriate. All expensive criteria use
  one persisted, model-independent common subset.
- Review checked true-class pre-softmax gradients, per-image absolute fallback,
  signed repeated-coordinate sums, coarse bins, donor-specific correctness,
  donor view averaging, stream restrictions, and bird-color-free blur.
- Verification: `test_criteria_scientific_core.py`, `test_transforms.py`, and
  model-boundary cache/gradient tests.

## Phase 7: AnchorCal statistics and decision freeze

- Implemented Kendall tau-b, tolerance-aware/transitive Spearman ranks,
  PairAcc, AdjAcc, violations, constant-score NA semantics, alternating-lambda
  ACE with endpoint floor and degenerate-fold mean fallback, identical paired
  class-stratified bootstraps, PerfectOrder rate, point-ACE winner, one-SE
  credible set, locked tie-breaks, and atomic create-only hashed receipts.
- Review found undefined correlations could serialize as nonstandard `NaN` and
  near-tie rank grouping was not transitive. Results now use explicit JSON
  `null`, valid/NA counts, and transitive tolerance groups.
- Anchor artifacts were expanded with human-readable score/bootstrap tables,
  score-vector cache, competence/margin provenance, and parity metadata.
- Verification: `test_statistics.py`, `test_decision.py`, and the synthetic
  analysis integration test.

## Phase 8: candidate grid, storage, selection, and hidden reporting

- The current campaign adds a fail-closed `anchorcal-storage-preflight-v1`
  contract: a 40 GiB hard budget, 35 GiB launch guard, 16 GiB minimum free
  space, 6 GiB concurrent-growth allowance, and explicit conservative
  components totaling 12 GiB. The receipt is bound before downstream writes.
- Implemented all six 40-epoch ordinary ViT runs, one shared epoch-zero
  diagnostic, every-epoch practical/oracle/test evaluation, fixed selectors
  and donors, rolling deduplicated checkpoints, final/restart states, atomic
  paired HDF5 epochs, exact-grid checks, selection receipt, hidden join,
  competent-pool/ranking/regret analyses, paired four-group bootstrap CIs,
  tables, figures, manifest, and diversity kill criterion.
- Review found a publication crash window, a run-manifest overwrite risk, and
  concurrent repetition of epoch zero. Storage now recovers a logically
  published pair, validates an existing run manifest before writing, and uses
  one lock-protected shared epoch-zero artifact.
- A separate audit found selector code could still import a module that named
  hidden artifacts. `selector_storage.py` and `hidden_storage.py` now enforce a
  one-way namespace boundary; AST and fresh-process import tests verify it.
- Selector review added exact config/receipt/preflight/HDF5/checkpoint binding,
  exact run/epoch/sample completeness, and rolling-state agreement before the
  candidate-selection receipt. Hidden analysis validates the receipt and file
  hashes before reading oracle/test data.
- Selector-visible HDF5 readers now require the exact reviewed root, attribute,
  sample, epoch, prediction, metric, and per-example schemas. Selector-side
  anchor verification independently requires the exact 13-file public artifact
  set at its fixed paths; a future producer cannot add another selector-readable
  file without a schema/code review.
- Verification: `test_storage.py`, `test_storage_namespace_boundary.py`,
  `test_checkpoints.py`, and the full synthetic selector-freeze/hidden-join
  test in `test_analysis_integration.py`.

## Phase 9: TIGRIS execution package

- Implemented fixed-resource Slurm jobs for online preflight, offline isolated
  debug, parallel branches, anchors, six separate candidates, and CPU final
  analysis, with `afterok` dependencies.
- The launcher requires the authoritative clean checkout, serializes campaign
  creation, rejects ambient `SBATCH_*` overrides and collisions, freezes the
  commit/interpreter/config/paths/package set, hashes inputs, records job IDs,
  and requests rollback on partial submission failure. Every job rechecks the
  commit, worktree, frozen hashes, package environment, preflight bundle, and
  online/offline mode before project execution.
- Independent shell review checked the account, partition, GH200 request,
  CPU/memory/time locks, graph, receipt guards, restart ownership, and rollback
  behavior. `bash -n` passes for every shell/batch file.
- Final review found the runbook still asserted final-analysis schema `v1`
  after the report had intentionally advanced to `v2`; the assertion and
  campaign verifier were corrected. The verifier now refuses absent/partial
  grids and validates final receipts, manifests and counts.
- Limitation: scheduler submission, real GH200 execution, real cache download,
  and full Waterbirds numerical gates cannot be exercised on this local host.
  They are deliberately hard production preflight/debug gates, not assumed
  successes.

## Final repository-wide review

- Independent artifact review found four fail-closed edge cases not exercised
  by the original happy path: a crash after the HDF5 manifest fsync but before
  journal publication, insufficient run-ID binding across copied candidate
  artifacts, incomplete decision-receipt binding, and incomplete branch
  restart/training provenance. Publication recovery now finishes the exact
  durable intermediate state; storage verifies directory/manifest/journal/HDF5
  identities and capacities; the decision verifier re-derives branch, subset,
  lambda and scale bindings; and branch manifest v3/restart v2 bind all input,
  model, optimizer, history, and fallback artifacts.
- Independent namespace review found that selector analysis imported a shared
  checkpoint verifier which also defined reporting-only paths and schemas. The
  visible verifier is now a self-contained one-way module, while joint/hidden
  verification remains post-freeze. Recursive import-closure AST checks and a
  fresh-process test reject hidden modules, identifiers, schemas, and paths.
- Final hardening also added exact timm preprocessing serialization/parity,
  exact diagnostic HDF5 schemas, read-only checkpoint payload verification,
  independent source-mask replay for every retained background token, explicit
  foreground-stream intervention invariance, aggregate fallback reporting,
  exact preflight checksum membership, per-run hidden-input hashes, strict
  metadata containment, and explicit prediction-stability diagnostics.
- During that earlier, pre-correction audit, re-ran the then-current AnchorCal
  unit/integration/adversarial suite and historical repository suite with the
  available local interpreters. Python compilation, all CLI help paths, package
  build/import, shell syntax checks, import-boundary checks,
  placeholder/dead-code scans, and whitespace checks also passed.
- Re-read all 114 binding decisions against the maintained traceability table
  and commissioned independent scientific, artifact-contract, and TIGRIS
  audits. Their production blockers were repaired and regression-tested before
  readiness was declared.
