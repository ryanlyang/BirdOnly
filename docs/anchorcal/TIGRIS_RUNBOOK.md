# AnchorCal Waterbirds100 TIGRIS runbook

This package submits the complete locked pilot as one fail-closed Slurm graph
from the authoritative TIGRIS checkout:

```text
/home/ryreu/guided_cnn/BirdOnly
```

The corrected campaign requires AnchorCal `0.4.0` and resolved configuration
schema `anchorcal-config-v2`; older campaign artifacts are incompatible.

It targets account `reu-aisocial`, partition `tigris`, and GH200 GPUs. The
single preflight job is allowed to populate the pinned Hugging Face model
cache. Every later job is forced into Hugging Face offline mode.

## 1. One-time checkout and path preparation

Transfer or pull the committed implementation into the authoritative checkout.
Do not run a production campaign from an uncommitted tree. On TIGRIS:

```bash
cd /home/ryreu/guided_cnn/BirdOnly
git status --short
git rev-parse HEAD
```

The first command must print nothing. The launcher freezes that commit, and
every queued production job checks both the commit and clean-tree status before
it sources project shell code or imports project Python code.

Create the ignored, machine-local path file if it does not exist:

```bash
cp configs/anchorcal/paths.local.example.yaml \
  configs/anchorcal/paths.local.yaml
```

Verify the three established dataset/mask entries. Do not leave
`REQUIRED_ABSOLUTE_PATH` and do not substitute Waterbirds-95, a CUB
segmentation tree, or a historical WeCLIP+ output whose filenames happen to
match. The fixed Waterbirds-100 entries are:

```yaml
paths:
  repo_root: /home/ryreu/guided_cnn/BirdOnly
  waterbirds_root: /home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2
  metadata_path: /home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2/metadata.csv
  vlm_mask_root: /home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_waterbirds100_openclip_laion_dinovit/val/prediction_cmap
  hf_home: /home/ryreu/.cache/huggingface
  output_root: /home/ryreu/guided_cnn/BirdOnly/outputs/anchorcal/waterbirds100_pilot
```

The discovery command is a read-only verification aid. It may print other
candidates, but it must never select or substitute one automatically:

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/ryreu/guided_cnn/BirdOnly/src \
  /home/ryreu/miniforge3-aarch64/envs/fcv_gh200/bin/python \
  scripts/anchorcal/discover_paths.py /home/ryreu/guided_cnn
```

The production preflight validates the authoritative release and metadata, then
maps each required row from the complete dataset-relative `img_filename` using
the exact `generate_pseudo_masks_waterbirds._make_image_id` flattening rule.
It also requires every official split-0 row to satisfy `y == place`; it never
filters a partially biased release into a surrogate Waterbirds100 pool.
It tries the producer name first, fails on collisions, reuse, missing or
ambiguous mappings, and strictly decodes categorical Pascal/VOC class ID 1
(RGB `[128, 0, 0]`) rather than thresholding a white binary mask. Complete
coverage is required for official splits 0 and 1 only. Official split 2 has no
mask requirement, and test classification must not load masks.

The accepted mapping is frozen in `preflight/mask_manifest.json` with schema
`anchorcal-vlm-mask-manifest-v2`. The manifest includes the exact VLM root,
metadata hash, locked source identifier
`waterbirds100_openclip_laion_dinovit_weclipplus_prediction_cmap`, mapping and
decoder implementation versions, canonical per-row mapping, dimensions,
decoded class/color counts, file sizes, per-mask SHA-256 values, split
coverage, collision and extras reports, and a deterministic content hash. It
records the AnchorCal checkout revision but does not claim an unknown external
GALS producer-source revision. The deterministic dataset partitions are
separately frozen in `splits/manifest.json` with schema `anchorcal-splits-v3`.
Every downstream job verifies the frozen hashes. Preflight also performs the
pretrained-hash, package, architecture, and GH200 checks. A failed preflight
prevents every downstream stage.

## 2. Submit the campaign

The interpreter may be overridden only with an explicitly prepared frozen
environment. Otherwise the established GH200 environment is used:

```bash
cd /home/ryreu/guided_cnn/BirdOnly
export ANCHORCAL_PYTHON=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200/bin/python
bash scripts/anchorcal/submit_campaign.sh
```

Do not run the individual `.sbatch` files directly for an initial campaign.
The launcher first creates the Slurm log directories, freezes copies of the
pilot and local path configurations, records the exact interpreter and package
lock, hashes all inputs, and writes the returned job IDs and dependency graph
to:

```text
outputs/anchorcal/waterbirds100_pilot/submission_receipts/
```

The launcher refuses a dirty tree, unresolved paths, an unexpected checkout or
output root, an already active AnchorCal graph, or existing debug/production
artifacts. It also fails if any exported `SBATCH_*` variable is present,
because Slurm would give that variable precedence over the locked resource
directives. A nonblocking campaign lock serializes concurrent launcher
invocations, and a failed scheduler query is treated as a hard error rather
than evidence that the queue is empty. If one of the `sbatch` calls fails, the
launcher requests cancellation of every job it already submitted and records
the rollback status in an interrupted-submission receipt. It does not delete
or overwrite an earlier campaign. Preserve an earlier output by moving the
whole campaign root to a timestamped archive before starting a genuinely new
campaign.

## 3. Frozen graph and resources

```text
online preflight/cache population
  -> offline standalone miniature debug
  -> offline foreground + background (parallel)
  -> offline AnchorCal evaluation + hashed decision receipt
  -> offline six independent candidate jobs
  -> offline final selector freeze, then reporting-only join
```

Every edge is an `afterok` dependency. The six candidate jobs are separate
submissions, not a Slurm array.

| Stage | GPU | CPUs | Memory | Wall time |
|---|---:|---:|---:|---:|
| Preflight/cache population | 1 GH200 | 8 | 32G | 2h |
| Standalone debug | 1 GH200 | 8 | 32G | 2h |
| Foreground branch | 1 GH200 | 12 | 64G | 12h |
| Background branch | 1 GH200 | 12 | 64G | 12h |
| Anchor evaluation | 1 GH200 | 12 | 64G | 12h |
| Each candidate | 1 GH200 | 16 | 96G | 24h |
| Final analysis | none | 8 | 64G | 4h |

No QOS is invented by these files. All GPU requests are exactly
`--gres=gpu:gh200:1`; the CPU-only final stage has no `--gres` directive.

## 4. Monitor

The launcher prints all job IDs. They are also frozen in the submission
receipt. Useful commands are:

```bash
squeue -u "$USER" -o '%.18i %.24j %.2t %.10M %.20R'
sacct -j <comma-separated-job-ids> \
  --format=JobID,JobName,State,ExitCode,Elapsed,Start,End
tail -f outputs/anchorcal/waterbirds100_pilot/run_logs/ac_preflight_<jobid>.out
```

Logs use `%x_%j`, so candidate logs are named `ac_cand_00_<jobid>.out` through
`ac_cand_05_<jobid>.out`. Failure output is in the matching `.err` file.

Do not interpret a pending `Dependency` reason as a failure. If a parent fails,
inspect its traceback and `sacct` state; `afterok` correctly prevents its
children from consuming incomplete artifacts.

## 5. Expected gates and artifacts

Important checkpoints in the graph are:

```text
preflight/report.json
preflight/mask_manifest.json
preflight/preflight_artifacts.sha256
preflight/preprocessing_manifest.json
splits/manifest.json
environment/environment.json
environment/package-lock.txt
debug/analysis/summary.json
branches/foreground/manifest.json
branches/foreground/crop_fallback_gate.json
branches/background/manifest.json
branches/background/crop_fallback_gate.json
anchors/artifact_manifest.json
anchors/foreground_stream_intervention_audit.json
anchors/criterion_results.json
receipt/anchorcal_decision_*.json
candidates/lr*_wd*_seed1234/candidate_outputs.h5
candidates/lr*_wd*_seed1234/exploratory_hidden_metrics.h5
candidates/lr*_wd*_seed1234/candidate_storage_manifest.json
candidates/lr*_wd*_seed1234/checkpoints/manifest.json
analysis/all_candidates_selector_only.csv
receipt/candidate_selection_*.json
analysis/summary.json
```

Each job also writes a hashed job receipt under
`submission_receipts/jobs/`. The preflight stage hashes its required outputs as
a bundle; all downstream jobs verify that bundle before running.

The AnchorCal decision receipt exists before candidates are released. In the
final job, selector-only aggregation writes and hashes the candidate-selection
receipt before the reporting-only module is imported and hidden oracle/test
files are joined.

## 6. Restarts and failures

Candidate HDF5 writes are epoch-transactional and each candidate owns its own
resume checkpoint, HDF5 pair, lock, and rolling selected states. Foreground and
background training also write restart state. Their batch files request Slurm
requeue support. If TIGRIS permits requeueing a failed or preempted job, reuse
the original job ID so its frozen environment and dependency context are
preserved:

```bash
scontrol requeue <foreground-background-or-candidate-job-id>
```

The candidate scientific run manifest intentionally contains no Slurm job ID.
Every scheduler attempt instead has its own immutable receipt below
`submission_receipts/jobs/`, and the successful attempt is named in the
candidate completion receipt. Consequently, a replacement job may resume the
same candidate directory when an administrator replaces a job ID, provided it
is launched with the original frozen campaign environment and the exact same
candidate hyperparameters. The immutable manifest, HDF5/checkpoint transaction
checks, exact run ID, config/preflight/preprocessing/model hashes, and input
receipts will reject any incompatible resume. Branch restart payloads are
bound to the same scientific provenance, and their final manifests hash the
training history, optimizer groups, aggregate fallback gate, calibration
outputs, and final weights.

Never resubmit the anchor stage after it has successfully written a decision
receipt: the batch wrapper explicitly refuses a second receipt. A requeued
final job reuses its one verified candidate-selection receipt and resumes only
the reporting join; it never recreates or changes the practical choice.
Never edit code, the frozen manifest files, the environment, or the local path
overlay while jobs from the campaign are queued. If a fix requires a new
commit, preserve the old output root and submit a new campaign from a clean
commit-specific checkout.

The standalone debug is an end-to-end scientific gate rather than a production
artifact source. Its branches, receipt, candidate files, and final analysis
remain entirely below `debug/` and are never eligible for the production join.

## 7. Final verification and collection

After the final job completes:

```bash
cd /home/ryreu/guided_cnn/BirdOnly
FROZEN_MANIFEST=outputs/anchorcal/waterbirds100_pilot/manifests/campaign_<campaign-id>
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/ryreu/guided_cnn/BirdOnly/src \
  /home/ryreu/miniforge3-aarch64/envs/fcv_gh200/bin/python \
  scripts/anchorcal/verify_campaign.py \
  --config "$FROZEN_MANIFEST/pilot.yaml" \
  --paths "$FROZEN_MANIFEST/paths.local.yaml"

SUMMARY=outputs/anchorcal/waterbirds100_pilot/analysis/summary.json
/home/ryreu/miniforge3-aarch64/envs/fcv_gh200/bin/python -c \
  'import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
value = json.loads(p.read_text(encoding="utf-8"))
assert value["schema_version"] == "anchorcal-final-analysis-v2"
assert value["run_count"] == 6
assert value["candidate_count"] == 240
print("complete final summary:", p)' "$SUMMARY"
```

The first helper is fail-closed: it requires the exact six-run grid, all 40
epochs in both HDF5 namespaces, rolling/final checkpoint manifests, both
hashed receipts, the 240-candidate `v2` summary, all analysis-manifest hashes,
and a passed production diversity gate. Also require the final Slurm job to be
`COMPLETED`; the explicit summary assertions above provide a quick human-
readable double check. Then inspect
`analysis/summary.json`, tables, and figures together with the hashed decision
and candidate-selection receipts. Copy the whole campaign root when collecting
results; retaining only the final table would discard the configuration,
environment, split, immutable VLM mapping/hash evidence, and chronology needed
to audit the pilot.
