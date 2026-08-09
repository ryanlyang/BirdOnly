# AnchorCal Waterbirds100 TIGRIS runbook

This package submits the complete locked pilot as one fail-closed Slurm graph
from the authoritative TIGRIS checkout:

```text
/home/ryreu/guided_cnn/BirdOnly
```

The corrected campaign requires AnchorCal `0.7.0` and resolved configuration
schema `anchorcal-config-v5`; older campaign artifacts are incompatible.

It targets account `reu-aisocial`, partition `tigris`, and GH200 GPUs. The
single preflight job is allowed to populate the pinned Hugging Face model
cache. Every later job is forced into Hugging Face offline mode.

Ordinary training and inference use the locked GH200 BF16 autocast policy.
Post-logit centering, margin normalization, and lambda mixing use FP32, and
saliency forwards/gradients use the activated FP32 fallback with autocast
disabled. Direct-versus-cached parity retains its strict `1e-6` logit/criterion
and `1e-5` saliency limits.

Every Slurm stage exports `HDF5_USE_FILE_LOCKING=FALSE` before importing h5py.
This avoids TIGRIS filesystem `EAGAIN` failures when immutable candidate HDF5
artifacts are reopened immediately after publication. Candidate writer
exclusion remains enforced by AnchorCal's per-run advisory lock, paired
transaction journal, atomic publication, and SHA-256 manifest; the setting is
recorded in every immutable job-attempt receipt.

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

Create the ignored, machine-local path file. If it predates `0.5.0`, refresh it
even if it already exists because the exact FCV split-manifest root became a
required input in that release:

```bash
cp configs/anchorcal/paths.local.example.yaml \
  configs/anchorcal/paths.local.yaml
```

Verify all established input and output entries. Do not leave
`REQUIRED_ABSOLUTE_PATH` and do not substitute Waterbirds-95, a CUB
segmentation tree, or a historical WeCLIP+ output whose filenames happen to
match. The fixed Waterbirds-100 entries are:

```yaml
paths:
  repo_root: /home/ryreu/guided_cnn/BirdOnly
  waterbirds_root: /home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2
  metadata_path: /home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2/metadata.csv
  vlm_mask_root: /home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_waterbirds100_openclip_laion_dinovit/val/prediction_cmap
  fcv_split_manifest_root: /home/ryreu/guided_cnn/logsWaterbird/fcv_vit_waterbirds100_first_study/split_manifests
  hf_home: /home/ryreu/.cache/huggingface
  output_root: /home/ryreu/guided_cnn/BirdOnly/outputs/anchorcal/waterbirds100_pilot
```

The repo-local output root is intentional. Although the general handoff records
`/home/ryreu/guided_cnn/logsWaterbird` as the historical Waterbirds log
convention, this pilot keeps its ignored, hash-bound artifacts under
`BirdOnly/outputs/anchorcal/waterbirds100_pilot`. Do not redirect the campaign
to `logsWaterbird`.

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
The top-level 80/20 membership is not regenerated: preflight imports and
hash-verifies `manifest_bundle.json`, `split_indices.json`, `metadata_train.csv`,
and `metadata_val.csv` from the exact `fcv_split_manifest_root` above. It
validates their embedded metadata/count/membership contracts and fails if the
imported metadata-index lists do not form the exact official split-0 partition.
The nested expert split still uses seed `2718`. Selector-safe development/expert
CSVs have exactly:

```text
img_id,img_filename,y,split,membership_source,membership_seed,
source_metadata_sha256,source_membership_sha256
```

Official oracle/test rows and protected `metadata_index`/`place`/`group` labels
are written only below `analysis_only/splits/` with manifest schema
`anchorcal-analysis-only-splits-v1`. Their exact CSV columns are:

```text
img_id,metadata_index,img_filename,y,place,group,split,source_metadata_sha256
```

The public split manifest contains no protected rows, IDs, count summaries,
paths, or hashes; it retains only selector-safe development information, the
aggregate official-training alignment pass, and fixed FCV source-artifact
paths/hashes.
It tries the producer name first, fails on collisions, reuse, missing or
ambiguous mappings, and strictly decodes categorical Pascal/VOC class ID 1
(RGB `[128, 0, 0]`) rather than thresholding a white binary mask. Complete
public/runtime coverage is required for official split 0. Official split 1 may
be machine-audited, but every per-row result is isolated at
`analysis_only/masks/waterbirds100_oracle_val_mask_audit.json` under schema
`anchorcal-analysis-only-vlm-mask-audit-v1`; practical selectors never parse
it. `preflight/report.json` contains neither that protected path nor its hash.
The protected file self-binds its source/data, and hidden/campaign verification
enforces its fixed location. Official split 2 has no mask requirement, and test
classification must not load masks.

The accepted official split-0 mapping is frozen in
`preflight/mask_manifest.json` with schema
`anchorcal-vlm-mask-manifest-v3`. This public per-row manifest includes the
exact VLM root, metadata hash, locked source identifier
`waterbirds100_openclip_laion_dinovit_weclipplus_prediction_cmap`, mapping and
decoder implementation versions, canonical split-0 per-row mapping,
dimensions, decoded class/color counts, file sizes, per-mask SHA-256 values,
split-0 coverage, collision and extras reports, and a deterministic content
hash. It contains no split-1 row, ID, count, path, or membership-derived
summary, and public entries omit `metadata_index` plus every context/group
field. It records the AnchorCal checkout revision but does not claim an
unknown external GALS producer-source revision. Branch/candidate runtime may
read this split-0 bank. Final selector provenance reads only the compact
`preflight/selector_mask_receipt.json` with schema
`anchorcal-selector-mask-receipt-v1`; it never parses the per-row manifest or
imports the full VLM loader. Hidden/final verification handles the protected
split-1 audit separately. The deterministic dataset partitions are
separately frozen in `splits/manifest.json` with schema `anchorcal-splits-v4`.
The already user-audited VLM bank is also rendered into three deterministic,
hash-bound six-sample sheets, `contact_sheet_01.png` through
`contact_sheet_03.png`, under `preflight/mask_visual_audit/`. All 18 public
examples come from official split 0. For each of the two aligned `(y,place)`
cells, sort by foreground fraction and then `img_id`, form low/middle/high
equal-count area strata, split each stratum into three equal rank ranges, and
take the middle example from each range. Protected `place` is used for
stratification but is not serialized per sample; no split-1 membership or
example appears in the public gallery. Five labeled panels show
Original RGB, Bird red/background blue, Mask white bird, Bird kept/background
green, and Background kept/bird green. The manifest schema is
`anchorcal-mask-visual-audit-v1` and records
`human_approval_required=false`. Producing the sheets is mandatory, but the graph
does not wait for a new human approval; the automated mapping, decoding,
coverage, and geometry checks are the gates.
Every downstream job verifies the frozen hashes it is authorized to consume.
Branch/candidate data paths verify the public split-0 bank, practical selector
paths verify only the compact mask receipt, and hidden/final verification owns
the protected audit. Preflight also performs the pretrained-hash, package,
architecture, and GH200 checks. A failed preflight prevents every downstream
stage.

Preflight also freezes the background token budget before any branch training.
It chooses the largest value in `[64, 48, 32]` satisfying both at least 95
percent overall/per-class eligible-patch coverage in `expert_train`,
`expert_calibration`, and `biased_val`, and at most 1 percent overall
`biased_val` invalidity. The latter is not a per-class 1-percent gate. The
downstream background and anchor stages reassert the overall invalidity limit
as defense in depth. For the current locked dataset, mask bank, and geometry,
the expected result is `K=32`: 9 of 959 `biased_val` examples are invalid.
Another value indicates changed input or geometry provenance and must be
investigated before continuing.

Preflight also enforces the campaign's storage contract in
`preflight/storage_budget.json` using schema
`anchorcal-storage-preflight-v1`. The hard budget is 40 GiB, the launch guard
is 35 GiB, the minimum output-filesystem free space is 16 GiB, and the
worst-case allowance for concurrent growth is 6 GiB. Its conservative 12 GiB
projection assigns 6 GiB to rolling candidate checkpoints, 2 GiB to restart
states and atomic-publication staging, 1 GiB to candidate HDF5/analysis files,
1 GiB to branches/anchors, and 2 GiB to manifests, galleries, and reserve.
Any failed budget, projection, free-space, or receipt-binding check blocks the
campaign; it is not downgraded to a warning.

The machine gate reads the output filesystem's live capacity. Because a
site-enforced per-user quota can be lower than filesystem-wide free space, also
inspect it when TIGRIS exposes quota information:

```bash
df -h /home/ryreu/guided_cnn/BirdOnly
quota -s
```

The quota command is an operator cross-check, not a reason to bypass a failed
machine storage gate.

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
preflight/selector_mask_receipt.json
preflight/mask_visual_audit/manifest.json
preflight/mask_visual_audit/contact_sheet_*.png
preflight/preflight_artifacts.sha256
preflight/preprocessing_manifest.json
preflight/storage_budget.json
preflight/geometry/background_token_budget.json
splits/manifest.json
analysis_only/splits/manifest.json
analysis_only/masks/waterbirds100_oracle_val_mask_audit.json
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
a scheduler-integrity bundle; all downstream wrappers verify that bundle before
running. That bundle may bind the protected mask-audit bytes, but it is not
parsed by selector code. The selector-safe `preflight/report.json` contains no
protected mask-audit path or hash.

Inspect `preflight/geometry/background_token_budget.json` after preflight. It
must record the complete 64/48/32 coverage and invalidity table, the combined
gate decision, and `token_budget: 32` for the current locked inputs. If no
candidate satisfies both gates, preflight fails before debug or production
training; do not bypass that failure or requeue a downstream job.

The AnchorCal decision receipt exists before candidates are released. In the
final job, selector-only aggregation writes and hashes the candidate-selection
receipt before the reporting-only module is imported and hidden oracle/test
files are joined. Selector-only provenance verifies the compact selector mask
receipt and never opens the per-row mask manifest or protected split-1 mask
audit.

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
Its three-epoch candidate trajectory uses one explicit warmup epoch; production
remains locked to 40 candidate epochs with four warmup epochs. Configuration
validation rejects a warmup longer than its corresponding training trajectory,
and the launcher validates both production and debug configs before submitting
the preflight job.

### Borderline random-token diagnostic

If the original production random-token gate is borderline, preserve the
failed campaign and its completed branch checkpoints. From a clean committed
checkout, submit the read-only diagnostic with:

```bash
export ANCHORCAL_PYTHON=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200/bin/python
bash scripts/anchorcal/submit_random_token_diagnostic.sh
```

The 30-minute job does not retrain or mutate the frozen campaign. It evaluates
ten fixed realizations of both the exact original pooled draw and an exactly
16/16 per-draw source-class-balanced construction, then writes a small report
under `outputs/anchorcal/random_token_diagnostics/reports/`. It reports the
original gate outcome for each realization, aggregate bootstrap behavior,
per-class prediction rates, a recipient-label permutation test, provenance,
and source/recipient disjointness. It is explicitly diagnostic-only and cannot
silently override the original gate.

The preserved job-72947 result motivated the prospective `0.7.0` pilot
amendment: production now uses the exactly 16/16 construction for each of ten
fixed realizations and applies the existing point/CI hard requirements to the
aggregate per-image correctness. The diagnostic never changed job 72587's
failed status. A new `0.7.0` campaign is required; do not reuse or mutate that
campaign's frozen inputs or receipts.

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
