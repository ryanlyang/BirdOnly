# Tigris campaign runbook

This is the operational source of truth for the first
`setv_waterbirds95_private_pilot_v1` campaign. The scientific source of truth
remains `SETV_Waterbirds95_Implementation_Plan_v2.md`.

No launcher submits work unless the repository is clean, the frozen campaign
manifest is valid, every required prior artifact verifies, and the target
output namespace is absent. Production launchers never overwrite artifacts.

## Frozen campaign

The authoritative machine-readable contract is:

```text
configs/campaign_waterbirds95.yaml
```

It freezes:

- Tigris repository:
  `/home/ryreu/guided_cnn/BirdOnly`;
- campaign root:
  `/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95`;
- dataset, VLM-mask, environment, and official uLA paths;
- all expert, fusion, candidate, mask-bank, and uLA seeds;
- three ERM candidate seeds: `3101,3102,3103`;
- clean-Git, no-overwrite, GH200-smoke, and hidden-test policies.

Do not hand-edit seeds in shell commands. Every launcher sources
`scripts/load_campaign_env.sh`, which validates the manifest and exports the
frozen values. Changing a seed, input path, or output namespace after the
campaign begins requires a new campaign ID and campaign root.

The only intentionally unresolved production choice is the uLA execution
source. Before Phase 6, explicitly supply either a usable legacy environment
for the official SSL attempt or an existing official MoCoV2+ checkpoint.

## Initial checkout

On Tigris:

```bash
cd /home/ryreu/guided_cnn/BirdOnly
git pull --ff-only
git status --short
/home/ryreu/miniforge3-aarch64/envs/fcv_gh200/bin/python \
  scripts/campaign_manifest.py
/home/ryreu/miniforge3-aarch64/envs/fcv_gh200/bin/python \
  -m unittest discover -s tests -v
```

`git status --short` must print nothing. Record the commit:

```bash
git rev-parse HEAD
```

Do not launch from an uncommitted or locally modified checkout.

## Status and preflight

Each production launcher runs a fail-closed preflight itself and stores the
JSON receipt under `preflight/`. To inspect readiness without submitting:

```bash
source scripts/load_campaign_env.sh
mkdir -p "${SETV_CAMPAIGN_ROOT}/preflight"
/home/ryreu/miniforge3-aarch64/envs/fcv_gh200/bin/python \
  scripts/preflight_campaign.py \
  --stage phase0 \
  --report "${SETV_CAMPAIGN_ROOT}/preflight/phase0_status.json" \
  --status-only
```

Replace `phase0` with the desired stage. `--status-only` always exits
successfully after writing the report; the report's `ready` field is
authoritative. Omit `--status-only` for a fail-closed command.

The receipt verifies real artifacts rather than accepting the presence of a
completion filename. It includes source and manifest hashes but never loads,
prints, or copies reporting-only test metric values.

## Execution sequence

### Phase 0: data and mask contract

```bash
bash scripts/submit_phase0.sh
```

The job first writes
`preflight/phase0_mask_mapping_job<JOB_ID>.json`. It requires a one-to-one
mapping for every official train and validation row (splits 0 and 1) using the
exact flattened relative-stem naming convention of the Waterbirds WeCLIP+
generator. It also decodes every required PNG as an exact VOC categorical map,
with foreground class 1. Official-test masks (split 2) are intentionally
optional because test classification uses untouched images only. If required
coverage or decoding is incomplete, inspect the report; do not switch map
roots, merge teacher-map sources, or weaken the protocol.

It also writes multiple lossless PNG pages under
`preflight/phase0_mask_galleries_job<JOB_ID>/`. Inspect every page. Each sample
row contains the original, red mask overlay, object-on-green view, and exact
background-only view with the dilated bird region green. These are the actual
evaluation geometry, threshold, polarity, green color, and exact-mask dilation
used downstream. Pages cover candidate train, biased validation, and oracle
validation. They intentionally do not expose or require test masks.

Monitor:

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed
```

When the build succeeds, inspect every contact sheet under:

```text
/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/phase0/mask_audit/
```

This includes `green_view_galleries/`. Approval attests to both mask alignment
and the exact green-screen compositions, so inspect every page before running
the approval command.

Then, and only then:

```bash
/home/ryreu/miniforge3-aarch64/envs/fcv_gh200/bin/python \
  scripts/approve_mask_audit.py \
  --phase0-dir "${SETV_CAMPAIGN_ROOT}/phase0" \
  --reviewer YOUR_NAME \
  --confirm
/home/ryreu/miniforge3-aarch64/envs/fcv_gh200/bin/python \
  scripts/verify_phase0.py \
  --phase0-dir "${SETV_CAMPAIGN_ROOT}/phase0"
```

The preflight intentionally reports `awaiting_human_gate` between the build
and approval. Do not rerun Phase 0 in that state.

### Phase 1: object expert

```bash
bash scripts/submit_phase1_object.sh
```

The one-epoch GH200 smoke must pass before the 20-epoch job runs.
Inspect its `train_optimizer_step_count` and
`train_amp_skipped_step_count`. A dynamic-loss-scaling skip is retained as an
auditable metric, and the learning-rate scheduler advances only on applied
optimizer updates.

### Phases 2–4: background experts and fusion

After Phase 1 verifies, run:

```bash
bash scripts/submit_phase2_exact.sh
bash scripts/submit_phase3_sanitized.sh
bash scripts/submit_phase4_set.sh
```

These are scientific siblings and may be submitted independently after
Phase 1. Running them sequentially is operationally simpler and makes failures
easier to audit.

Phase 3 is a hard gate. Its leakage auditors use held-out images. Every
auditor must have balanced accuracy at most `0.53`, and its image-cluster
bootstrap interval must contain `0.50`. A rejected bank is evidence, not a
failed file to erase. Do not submit sanitized-expert training by hand after a
rejection.

### Phase 5: three candidate trajectories

```bash
bash scripts/submit_phase5_candidate.sh
```

The manifest supplies three unique ERM seeds. One smoke job gates all three
50-epoch trajectories. Ordinary, uLA-placeholder, hard-pseudogroup, SETV, and
oracle rolling states are retained by selector and deduplicated by epoch.

Test values are written only in each candidate's `reporting_only/` namespace
after its realistic selection receipt is frozen and hashed. Normal Slurm
logs and dashboards must not display or aggregate those values.

### Phase 6: uLA, joint selection, freeze, and reporting

First choose one explicit uLA mode.

Official SSL attempt:

```bash
export SETV_ULA_ENV=/absolute/path/to/separate/legacy-ula-environment
unset SETV_ULA_SSL_CHECKPOINT
bash scripts/submit_phase6_ula_smoke.sh
```

External official checkpoint:

```bash
export SETV_ULA_SSL_CHECKPOINT=/absolute/path/to/official/last.ckpt
unset SETV_ULA_ENV
bash scripts/submit_phase6_ula_smoke.sh
```

Inspect the isolated smoke receipt. It must report acceptance before
production. No fallback is automatic. Then retain the same explicit mode and
run:

```bash
bash scripts/submit_phase6.sh
```

The Phase 6 chain is uLA smoke, optional official SSL, frozen linear proxy,
then analysis. It initially compares `(background expert, fusion)` pairs
jointly across all three candidate seeds. Logistic ROC/PR AUC are
implementation diagnostics only; selection evidence is oracle regret,
correlations, and epoch-ranking accuracy.

The analysis hashes the realistic method-freeze receipt before it publishes
or joins reporting-only test values. Do not open Phase 5 reporting-only files
or expose Phase 6 test tables while method development is still active.

## Completion checks

After each stage, run its verifier documented in `docs/`. At campaign end:

```bash
/home/ryreu/miniforge3-aarch64/envs/fcv_gh200/bin/python \
  scripts/verify_ula_proxy.py \
  --output-dir "${SETV_CAMPAIGN_ROOT}/ula_proxy/seed_${SETV_ULA_SEED}"
/home/ryreu/miniforge3-aarch64/envs/fcv_gh200/bin/python \
  scripts/verify_phase6.py \
  --output-dir "${SETV_CAMPAIGN_ROOT}/phase6"
```

Also inspect the Phase 6 kill-criteria receipt. A technically complete run is
not automatically a positive scientific result. If a locked kill criterion
fires, report it and revise the method under a new campaign contract rather
than modifying the completed campaign.

## Failure and recovery

All top-level launchers refuse any existing target directory, including a
partial one. This is deliberate.

If a job fails:

1. Preserve its Slurm log, submission receipt, preflight receipt, and partial
   artifact directory.
2. Use `sacct` and the artifact's `failure.json` to identify the failing
   component.
3. Run `preflight_campaign.py --status-only` for that stage.
4. Do not delete, rename, edit, or overwrite campaign artifacts.
5. Decide whether the failure is an operational retry of an immutable input
   or a scientific/configuration change.

An operational retry may resume only the missing downstream Slurm component
after the existing artifacts and dependency hashes have been verified. Do not
rerun a top-level chain against a partial namespace. Record the resume command,
new job ID, old failed job ID, commit, and input hashes in a recovery receipt.

A changed seed, mask policy, model/config choice, source commit, or accepted
input is not an operational retry. Create a new campaign ID and output root.

Special cases:

- Phase 0 built but unapproved: inspect and approve; do not rebuild.
- Phase 0 mask-mapping audit failed before publication: retain the audit
  report. If the target `phase0/` directory is absent, a corrected clean
  commit may retry the same campaign; record both submission receipts.
- Pretrained-metadata contract failure before the first training step: retain
  the failed smoke log and submission receipt. If the architecture and
  pretrained weights are unchanged and no stage artifact was published,
  correcting the configuration to the weights' declared preprocessing is an
  implementation repair and may retry the same stage from a clean commit. The
  new submission receipt must record the corrected config hash.
- Phase 4 smoke rejected an augmented crop with fewer than 16 clean background
  tokens: retain the failed smoke log and submission receipt. If no Phase 4
  artifact was published, deterministic crop retries plus an audited
  full-frame aspect-preserving fallback with explicitly ineligible padding are
  an implementation repair because the locked 1% foreground threshold and
  16-token minimum remain unchanged. This fallback also applies when the
  canonical validation crop is intrinsically below the floor. Retry from a
  clean commit and retain all submission receipts.
- Sanitized leakage rejection: retain the rejected bank and stop Phase 3.
- Failed official uLA/GH200 compatibility: retain the smoke receipt; either
  repair the explicit legacy environment or explicitly switch to a verified
  official checkpoint.
- Existing verified target: the stage is complete; do not resubmit it.
- Corrupt or hash-mismatched artifact: quarantine through an explicitly
  documented operator action and start a new namespace. Never silently repair
  a receipt.

No command in this runbook submits all phases automatically. Human approval,
leakage acceptance, uLA compatibility, and method-freeze boundaries remain
visible gates.
