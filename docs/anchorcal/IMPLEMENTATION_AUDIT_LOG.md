# AnchorCal implementation audit log

This log records the mandatory self-review for each implementation phase.
Local tests use generated fixtures and small synthetic models. Production
evidence is created separately under the frozen campaign output root; no entry
below claims that the real dataset or TIGRIS GH200 was available locally.

## Authoritative VLM-mask correction: implemented and locally revalidated

- The binding specifications now replace the earlier CUB-mask assumption with
  the exact Waterbirds-95 OpenCLIP-LAION + DINOvIT VLM bank at
  `/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_waterbirds95_openclip_laion_dinovit/val/prediction_cmap`.
- The corrected contract requires a producer-first join from the complete
  metadata `img_filename`, strict Pascal/VOC class-1 decoding, complete
  one-to-one coverage for official splits 0 and 1 only, no official split-2
  mask requirement, and an immutable `preflight/mask_manifest.json` with schema
  `anchorcal-vlm-mask-manifest-v1` and a deterministic content hash.
- `Waterbirds100` remains the `y == place` aligned subset of official split 0
  from `waterbird_complete95_forest2water2`; it is not the separate
  `waterbird_1.0_forest2water2` release.
- Replaced both CUB path keys with one locked `vlm_mask_root`; added the exact
  producer, mapping, decoder, VOC class, nearest-interpolation, split-coverage,
  and manifest-only-runtime configuration locks.
- Added `src/anchorcal/vlm_masks.py`; rewrote preflight to audit all images,
  require and decode only producer-contract splits 0 and 1, inventory rather
  than require split 2, freeze the manifest before geometry construction, and
  bind file, decoded-mask, mapping, and decoder hashes.
- Replaced every branch, geometry, anchor, and practical-candidate CUB lookup
  with canonical `img_id` plus `img_filename` access through the frozen VLM
  manifest. Candidate ERM training and oracle/test classification remain
  mask-free.
- Bumped incompatible branch and candidate provenance schemas, and bound the
  VLM root, source, contract, bank hash, and manifest hash through restart,
  branch, anchor, candidate, decision, and final-campaign verification.
- On 2026-08-09, the Torch-capable local suites passed **158 AnchorCal tests**
  and **81 retained SETV tests** (**239 total**). Python compilation, shell
  syntax checks, CLI/config smoke checks, and `git diff --check` also passed.
- Real Waterbirds/VLM coverage, the pinned model cache, and GH200 execution
  remain production preflight and smoke gates on TIGRIS; local fixtures do not
  claim that external runtime evidence.

## Phase 0: specification and repository audit

- Read both binding specifications in full. The 112 answered locks, especially
  residual locks 97-112, take precedence over earlier plan wording.
- Inspected the existing repository, dependency metadata, historical `setv`
  package, TIGRIS handoff, and established `/home/ryreu/guided_cnn/BirdOnly`
  checkout convention.
- Created and maintained `IMPLEMENTATION_TRACEABILITY.md`. At that time,
  machine-local data and CUB-mask paths were deliberately unresolved; this
  historical assumption is superseded by the authoritative VLM correction
  above.
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
  `anchorcal-vlm-mask-manifest-v1`.
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
- Re-ran all 149 AnchorCal unit/integration/adversarial tests with the available
  training environment and all 81 historical repository tests with the local
  interpreter. Python compilation, all CLI help paths, package build/import,
  shell syntax checks, import-boundary checks, placeholder/dead-code scans, and
  whitespace checks also pass.
- Re-read all 114 binding decisions against the maintained traceability table
  and commissioned independent scientific, artifact-contract, and TIGRIS
  audits. Their production blockers were repaired and regression-tested before
  readiness was declared.
