# AnchorCal implementation traceability

This is the maintained requirement-to-code checklist for the Waterbirds100
pilot. `AnchorCal_Implementation_Decision_Locks_Answers.md` takes precedence
over the implementation plan. All paths below are repository-relative.
Runtime evidence is rooted at the resolved `paths.output_root` (production:
`outputs/anchorcal/waterbirds100_pilot`). A checked implementation means that
code and local focused/synthetic verification exist; it does **not** claim
that real Waterbirds data or a TIGRIS GH200 has been exercised locally.

| Requirement / decision locks | Implementation | Focused verification | Runtime evidence |
|---|---|---|---|
| Independent package, strict resolved configuration, local path overlay, and no legacy SETV selector/expert imports (85-89) | `src/anchorcal/config.py`, `src/anchorcal/paths.py`, `configs/anchorcal/*.yaml`, `scripts/anchorcal/_common.py` | `test_data_scientific_core.py`; import scans in the final audit | `preflight/resolved_config.yaml`, `preflight/report.json`, frozen campaign `pilot.yaml` and `paths.local.yaml` |
| Exact release, strict metadata schema/hash/contained filenames, canonical sorted `img_id`, authoritative mapped CUB masks, binary encodings, dimensions, uniqueness, and mapping provenance (1-6, 108) | `src/anchorcal/data.py`, `src/anchorcal/masks.py`, `src/anchorcal/preflight.py`, `scripts/anchorcal/discover_paths.py` | `test_data_scientific_core.py`, `test_preflight_path_contracts.py` | `preflight/mask_manifest.json`, `preflight/report.json` |
| Deterministic aligned Waterbirds100, nested expert, oracle-validation, and official-test split persistence (7, 11-13) | `src/anchorcal/splits.py` | `test_data_scientific_core.py` | `splits/waterbirds100_*.csv`, `splits/manifest.json` |
| Pinned ViT-S/16 repository/revision/safetensors hash, timm version, package/runtime and clean-commit locks (14, 87-90, 97, 111-112) | `src/anchorcal/pretrained.py`, `src/anchorcal/runtime.py`, `src/anchorcal/preflight.py`, `slurm/anchorcal/runtime_common.sh` | config/pretrained boundary assertions in `test_data_scientific_core.py`; production preflight is the real-cache/GH200 gate | `preflight/pretrained_manifest.json`, `environment/{environment.json,package-lock.txt}`, campaign environment/frozen-input receipts |
| Stateless cryptographic seeds, exact timm preprocessing/parity, exact joint geometry, source-resolution green screen, interpolation, bounded crop fallback, final-grid dilation, and strict patch purity (5-10, 93-94, 99) | `src/anchorcal/preprocessing.py`, `src/anchorcal/seeds.py`, `src/anchorcal/transforms.py`, `src/anchorcal/masks.py`, `src/anchorcal/datasets.py` | `test_preprocessing.py`, `test_transforms.py`, `test_data_scientific_core.py` | `preflight/preprocessing_manifest.json`, `preflight/geometry/*_geometry.csv`, `preflight/geometry/manifest.json`, branch histories/manifests |
| Foreground variable-token branch, object-relative positions, padding mask, independent copied weights, fixed epoch 30, calibration, restart provenance, aggregate crop-fallback gate, and persisted final state (15-19, 24-28, 100-101) | `src/anchorcal/models/{token_vit,branches}.py`, `src/anchorcal/training.py`, `src/anchorcal/calibration.py`, `src/anchorcal/branch_pipeline.py`, `src/anchorcal/branch_provenance.py` | `test_model_training_boundaries.py`, `test_branch_anchor_scientific_core.py`, `test_analysis_integration.py` | `branches/foreground/{epoch_final.pt,restart.pt,history.json,optimizer_groups.json,crop_fallback_events.json,crop_fallback_gate.json,expert_calibration_outputs.npz,biased_val_outputs.npz,manifest.json}` |
| Position-free background set branch, safe patches only, 64/48/32 no-replacement policy, independently audited eight fixed views, fixed epoch 30, calibration, and persisted final state (20-28, 102) | `src/anchorcal/background.py`, `src/anchorcal/models/branches.py`, `src/anchorcal/prepare.py`, `src/anchorcal/branch_pipeline.py`, `src/anchorcal/branch_provenance.py` | `test_branch_anchor_scientific_core.py`, `test_model_training_boundaries.py` | `preflight/geometry/{background_token_budget.json,fixed_background_views.h5,fixed_background_views.h5.manifest.json}`, `branches/background/*` |
| Foreground replacement and green-shade audits, independent source-mask background-purity replay, random-token audit, geometry auditors, bootstrap competence gates, competence intersection and raw-logit scales (29-38) | `src/anchorcal/audits.py`, `src/anchorcal/competence.py`, `src/anchorcal/anchor_pipeline.py` | `test_branch_anchor_scientific_core.py`, `test_model_training_boundaries.py`, `test_transforms.py` | `audits/branch_audits.json`, `anchors/{competence_intersection.csv,competence_intersection_manifest.json,margin_scales.json}` |
| Centered/scaled raw-logit 21-lambda ladder, differentiable frozen `RelianceAnchor`, typed stream contracts, source metadata, 100% competence correctness, explicit foreground-stream intervention invariance, direct/cache parity, and hash-bound artifact contract (26, 29-30, 46-47, 61, 95, 103-105) | `src/anchorcal/models/anchor.py`, `src/anchorcal/anchor_cache.py`, `src/anchorcal/interventions.py`, `src/anchorcal/anchor_pipeline.py`, `src/anchorcal/anchor_artifacts.py` | `test_model_training_boundaries.py`, `test_branch_anchor_scientific_core.py`, `test_criteria_scientific_core.py`, `test_analysis_integration.py` | `anchors/{anchor_per_image_outputs.npz,cache_parity.json,criterion_subset.csv,foreground_stream_intervention_audit.json,artifact_manifest.json}` |
| Saliency alignment with true-class pre-softmax gradients, per-image fallback, strict common geometry and class-balanced harmonic aggregation (39-47, 62-64, 106) | `src/anchorcal/saliency.py`, `src/anchorcal/criteria.py`, `src/anchorcal/candidate_evaluation.py`, `src/anchorcal/anchor_pipeline.py` | `test_criteria_scientific_core.py`, `test_model_training_boundaries.py` | `preflight/geometry/selector_eval_subset.csv`, `anchors/anchor_per_image_outputs.npz`, candidate selector-visible HDF5 per-example arrays |
| Fixed opposite-class donors, coarse-bin candidate assignment, anchor donor views, donor-specific token-swap correctness, mask-normalized background blur, and foreground-only diagnostic (48-60, 104) | `src/anchorcal/interventions.py`, `src/anchorcal/criteria.py`, `src/anchorcal/candidate_evaluation.py`, `src/anchorcal/anchor_pipeline.py` | `test_criteria_scientific_core.py`, `test_transforms.py` | `preflight/geometry/donor_assignments.json`, anchor/candidate per-example criterion arrays |
| Four eligible criteria, full biased-validation accuracy in harmonic means, class-balanced aggregation, product/control diagnostics excluded from choice (62-65) | `src/anchorcal/criteria.py`, `src/anchorcal/metrics.py` | `test_criteria_scientific_core.py` | `anchors/{criterion_results.json,anchor_scores.csv}` and candidate metric datasets |
| Kendall tau-b, tolerant Spearman, PairAcc, AdjAcc, violations, constant-score NA behavior, alternating-lambda ACE/floor, paired class-stratified bootstrap, PerfectOrder, one-SE set, deterministic tie-breaks, and locked success-label definitions (66-73, 107, 113-114) | `src/anchorcal/statistics.py`, `src/anchorcal/decision.py`, `src/anchorcal/anchor_pipeline.py` | `test_statistics.py`, `test_decision.py` | `anchors/{criterion_results.json,anchor_bootstrap_metrics.csv,anchor_bootstrap_score_vectors.npz}`, `receipt/anchorcal_decision_*.json{,.sha256}` |
| Six ordinary ViT runs, exact head/grid/training schedule, epoch-zero diagnostic, every-epoch practical and physically hidden evaluation (74-78, 93, 98-101) | `src/anchorcal/models/candidate.py`, `src/anchorcal/candidate_evaluation.py`, `src/anchorcal/candidate_pipeline.py` | `test_model_training_boundaries.py`, `test_storage.py`, `test_analysis_integration.py` | `diagnostics/shared_epoch_zero/*`; six `candidates/lr*_wd*_seed1234/` directories |
| Rolling practical/oracle/final checkpoints, content deduplication, atomic restart and final state, read-only hash/payload verification, and no all-epoch checkpoints (79-80) | `src/anchorcal/checkpoints.py`, `src/anchorcal/visible_checkpoint_verification.py`, `src/anchorcal/checkpoint_verification.py`, `src/anchorcal/candidate_pipeline.py` | `test_checkpoints.py`, `test_analysis_integration.py` | each run's `checkpoints/{manifest.json,resume.pt,final_state.pt,weights/,exploratory_hidden/oracle_manifest.json}` |
| Paired transactional HDF5 publication, exact scalar/per-example schemas, compact outputs, selector/reporting namespace separation, all crash-boundary recovery, exact run-identity binding, and hashes (82-83, 96, 109) | `src/anchorcal/storage.py`, `src/anchorcal/candidate_schema.py`, `src/anchorcal/selector_storage.py`, `src/anchorcal/hidden_storage.py` | `test_storage.py`, `test_storage_namespace_boundary.py`, `test_analysis_integration.py` | `candidate_outputs.h5`, `exploratory_hidden_metrics.h5`, `candidate_storage_manifest.json`, `candidate_storage_journal.json` |
| Selector freeze before hidden import/join, exact-grid/completeness checks, candidate rankings, competent pool, rolling-state binding, regret, bootstrap CIs and reporting tables/figures (38, 75-84) | `src/anchorcal/selector_analysis.py`, `src/anchorcal/hidden_analysis.py`, `src/anchorcal/analysis.py` | `test_analysis_integration.py`, `test_storage_namespace_boundary.py` | `analysis/{all_candidates_selector_only.csv,all_candidates.csv,selected_candidates.csv,summary.json,tables/,figures/,manifest.json}`, `receipt/candidate_selection_*.json{,.sha256}` |
| Online preflight, isolated miniature debug, parallel branches, anchors/receipt, six restartable candidates, CPU final join, frozen inputs and immutable queued commit (87, 90-92, 110-112) | `slurm/anchorcal/*.sbatch`, `slurm/anchorcal/runtime_common.sh`, `scripts/anchorcal/submit_campaign.sh`, `scripts/anchorcal/verify_campaign.py` | `bash -n`; CLI help/compile checks; local live Slurm/GH200 execution intentionally unavailable | `manifests/campaign_*`, `submission_receipts/`, `run_logs/`, final verification output |

## Numbered-lock coverage

All 114 numbered answers are represented above. The grouping is: `1-13`
data/masks/transforms/splits; `14-31` branch architecture, training,
calibration and anchors; `32-38` audits and diversity; `39-47` saliency;
`48-61` interventions; `62-65` aggregation; `66-73` AnchorCal statistics;
`74-84` candidate selection/reporting; `85-96` package/compute/debug/storage;
`97-112` final residual locks, and `113-114` success-label addenda. The
residual locks are cited again on the
specific rows whose earlier behavior they override.

## Precedence resolutions kept explicit

- Background sampling never uses replacement in the primary branch. Preflight
  selects the largest of 64/48/32 meeting 95% overall and per-class coverage;
  later branch-valid/intersection gates still fail closed.
- Repeated background source coordinates use view-averaged signed occurrence
  contributions and are summed by source coordinate before the lambda/ReLU
  density calculation; they are not averaged a second time.
- Scalar temperatures are fit diagnostically (background view logits are
  averaged first). Competence, margin scales, and anchor construction use raw
  logits only.
- Every candidate epoch is evaluated, while retained weights are restricted to
  restart, final, and rolling selector-best states, deduplicated by model hash.
- Oracle-validation and test outputs are written during training only to the
  reporting namespace. Practical selection freezes a hashed receipt before
  the reporting reader/module is imported.
- `Waterbirds100` means the aligned (`y == place`) subset of the official
  Waterbirds training split; it is not a different release directory.
- Dataset and CUB-mask absolute paths remain machine-local and explicit.
  Discovery lists candidates but never resolves ambiguity automatically.
