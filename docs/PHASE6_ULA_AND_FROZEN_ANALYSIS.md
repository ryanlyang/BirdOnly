# Phase 6: uLA baseline and frozen multi-seed analysis

Phase 6 is implemented as an execution-ready Tigris pipeline. It does not
submit jobs automatically.

## Scientific status

The vendored source in `uLA/` was compared byte-for-byte with the official
repository:

```text
repository: https://github.com/tsirif/uLA
commit: 5867fb6e9a8485ed08b4cbe84900f2b5ac4fac5d
audited tree SHA-256:
44004b6f24dffa16e233f9ee62dc8a1acad4ae5397504df4873f04af8e843a07
```

The audit is enforced at runtime. A modified or incomplete source tree is
rejected.

The implemented candidate selector is explicitly labeled `uLA-style`. The
reason is narrow and recorded in every receipt: the official uLA proxy-group
validation rule is applied to SETV's independently trained ViT candidate
trajectory, rather than selecting checkpoints from uLA's own debiased
ResNet-50 trajectory.

The following official components are preserved:

- MoCoV2+ self-supervised ResNet-50 encoder;
- a frozen SSL encoder with a linear target classifier used as the bias proxy;
- no temperature calibration;
- proxy groups `(argmax(proxy_logits), true_label)`;
- validation score equal to the mean accuracy over all nonempty proxy groups;
- the worst nonempty proxy-group accuracy as a diagnostic.

The proxy trains on the full `candidate_train`; no candidate sample is held
out for calibration.

## Official SSL adapter

`scripts/prepare_ula_official_shadow.py` creates a protected-label-free
metadata adapter:

- `candidate_train` is official split 0;
- `biased_val` is official split 1;
- the official code's unconditional test-loader slot receives a duplicate of
  `biased_val`;
- no oracle or test sample is present;
- all `place` values are redacted to a constant.

`scripts/run_official_ula_ssl.sh` then runs the exact Waterbirds MoCoV2+
settings from the official script for 100 epochs with an explicitly frozen
seed. The legacy uLA environment must be supplied through `SETV_ULA_ENV`; no
workstation or alternate cluster environment is assumed.

The official repository pins PyTorch 1.13.1+cu117, torchvision 0.14.1+cu117,
and PyTorch Lightning 1.6.4. The confirmed Tigris environment is aarch64,
PyTorch 2.11.0+cu130, and a GH200. Therefore dependency compatibility is an
empirical gate, not an assumption.

If that legacy stack is incompatible with the GH200/aarch64 environment, set
`SETV_ULA_SSL_CHECKPOINT` to an already produced official MoCoV2+ checkpoint.
The launcher skips SSL training, hashes that checkpoint in the proxy receipt,
and continues with the explicitly labeled `uLA-style` adapter. It never
silently substitutes ImageNet supervision or another encoder.

## Mandatory GH200 compatibility smoke

Every `submit_phase6.sh` invocation first submits
`slurm/phase6_ula_smoke.sbatch`. All downstream jobs have an `afterok`
dependency on this gate.

For an in-campaign official SSL run, the smoke:

1. audits the unchanged official source tree and exact commit;
2. records Python, architecture, every official dependency version, CUDA,
   cuDNN, GH200 capability, and a real CUDA forward/backward kernel probe;
3. verifies the Lightning 1.6 APIs used by the official entry point;
4. creates a deterministic, class-balanced eight-image-per-split shadow;
5. executes one real official MoCoV2+ epoch with a queue divisible by the
   smoke batch size;
6. loads the resulting checkpoint through the modern SETV ResNet-50 adapter;
7. runs one real frozen-encoder linear-proxy update and one validation batch;
8. writes a combined acceptance receipt.

For an external official MoCoV2+ checkpoint, the smoke skips legacy SSL
training but still performs the GH200 CUDA probe, official source audit,
checkpoint-to-ResNet compatibility check, one proxy update, and validation
forward pass against the exact supplied checkpoint.

The smoke artifacts are written under:

```text
/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/preflight/
  ula_smoke_job<JOB_ID>/
    compatibility.json
    proxy_checkpoint_smoke.json
    smoke_receipt.json
```

When official SSL is trained in-campaign, the full 100-epoch job also reruns
the proxy-checkpoint smoke against its final production `last.ckpt` before the
50-epoch linear proxy job can start.

A version mismatch is recorded even when the functional smoke passes. If the
official legacy stack cannot execute on GH200, the smoke fails and Slurm holds
the campaign. There is no automatic fallback. The operator must explicitly
supply `SETV_ULA_SSL_CHECKPOINT` and resubmit in external-checkpoint mode.

The readiness smoke can be submitted by itself, without starting Phase 6:

```bash
# SETV_ULA_SEED and SETV_ULA_REPO come from the campaign manifest.

# Official one-epoch attempt:
export SETV_ULA_ENV=/path/to/separate/legacy-ula-environment
unset SETV_ULA_SSL_CHECKPOINT
bash scripts/submit_phase6_ula_smoke.sh

# Or verify an explicit official checkpoint:
export SETV_ULA_SSL_CHECKPOINT=/absolute/path/to/official/last.ckpt
unset SETV_ULA_ENV
bash scripts/submit_phase6_ula_smoke.sh
```

This isolated launcher submits only `setv_ula_smoke`; it does not submit SSL
production training, proxy training, or Phase 6 analysis.

## Multi-seed analysis and method freeze

At least three distinct ERM candidate seeds are mandatory. For every selector,
Phase 6 computes:

- selected epoch;
- oracle validation WGA at that epoch;
- oracle selection regret;
- Spearman correlation with the oracle WGA curve;
- Kendall tau-b;
- pairwise epoch-ranking accuracy, excluding tied pairs.

SETV background-expert and fusion configurations are selected jointly. The
primary ordering is mean regret, worst-seed regret, Spearman, Kendall, and
pairwise accuracy. The locked effective-tie thresholds and secondary rules are
then applied. Leakage and view-stability evidence is derived from verified
expert artifacts rather than entered by hand. A secondary metric is skipped
unless it is semantically comparable and available for every tied expert; the
receipt records both derived values and explicit unavailability reasons. A
separate dominance audit permits a general background-expert claim only when
that expert dominates across every available fusion and candidate seed.

The qualitative Section 22 kill criteria have locked operational definitions
in `configs/phase6_analysis.yaml`. Phase 6 computes all eight outcomes and
writes their thresholds and supporting values to
`analysis_only/kill_criteria.json`. It also evaluates a
background-confidence-only weighted selector as a diagnostic baseline. These
flags are review gates, not permission for an automatic method change.

The method-freeze receipt is written and hashed before the analysis process
opens any `reporting_only/test_metrics.json` file. Test values are then written
only under `reporting_only/`.

## Candidate checkpoints

The uLA-selected state is always saved:

1. reuse an already deduplicated Phase 5 selected checkpoint when the epoch
   matches;
2. otherwise replay deterministic candidate training to the uLA-selected
   epoch;
3. require replayed biased-validation logits to match the stored trajectory;
4. save and hash the recovered state inside the Phase 6 artifact.

This preserves selected weights without retaining all 50 candidate
checkpoints.

## Outputs

The Phase 6 artifact includes:

```text
selection/method_freeze_receipt.json
selection/method_frozen.json
selected_candidate_checkpoints/
tables/primary_results_by_seed.csv
tables/primary_results_aggregate.csv
plots/mean_selection_regret.svg
plots/selector_score_vs_oracle_wga.svg
plots/*_expert_margin_histograms.svg
plots/*_object_vs_background_margin.svg
plots/*_fusion_score_distributions.svg
plots/candidate_disagreement_by_fusion_decile.svg
plots/candidate_accuracy_by_fusion_decile.svg
plots/representative_setv_alpha_curves.svg
diagnostics/*_margin_histograms.csv
diagnostics/*_fusion_examples.csv
diagnostics/candidate_fusion_deciles.csv
diagnostics/representative_alpha_curves.csv
diagnostics/high_score_hardness.csv
diagnostics/galleries/
analysis_only/background_confidence_baselines.json
analysis_only/kill_criteria.json
reporting_only/test_results.json
phase6_receipt.json
artifact_manifest.json
```

Logistic ROC AUC and PR AUC remain implementation diagnostics only. Robust
selection claims use regret and epoch-ranking results.

## Launch

The campaign manifest freezes all seeds and the actual uLA checkout. Explicitly
choose a legacy environment or an external official checkpoint, then run:

```bash
bash scripts/submit_phase6.sh
```

The launcher:

- requires at least three unique candidate seeds;
- audits the actual uLA path and source tree;
- requires a clean SETV Git checkout;
- refuses existing outputs and missing Phase 5/fusion artifacts;
- submits GH200 uLA smoke, official SSL when requested, proxy training, and
  final analysis with `afterok` dependencies;
- records the SETV commit, official uLA commit, paths, seeds, job IDs, and
  dependency chain.

Official SSL mode:

```bash
export SETV_ULA_ENV=/path/to/separate/legacy-ula-environment
# Leave SETV_ULA_SSL_CHECKPOINT unset.
bash scripts/submit_phase6.sh
```

External-checkpoint mode:

```bash
export SETV_ULA_SSL_CHECKPOINT=/absolute/path/to/official/last.ckpt
unset SETV_ULA_ENV
bash scripts/submit_phase6.sh
```

No job was submitted during implementation.

## Verification

```bash
python scripts/verify_ula_proxy.py --output-dir /path/to/ula_proxy/seed_N
python scripts/verify_phase6.py --output-dir /path/to/phase6
```

Verification checks artifact hashes, Phase 0 bindings, the official uLA
commit/tree binding, selected checkpoint hashes, the three-seed minimum, and
the hash ordering between method freeze and test publication.
