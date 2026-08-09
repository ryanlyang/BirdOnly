"""Strict post-run verification for a complete AnchorCal campaign."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .candidate_schema import CANDIDATE_SCALAR_METRICS, candidate_per_example_shapes
from .branch_provenance import verify_branch_artifacts
from .anchor_artifacts import verify_anchor_artifacts
from .checkpoint_verification import verify_candidate_checkpoint_artifacts
from .decision import verify_decision_receipt
from .errors import AuditFailure
from .io import sha256_file
from .provenance import verify_hashed_receipt
from .paths import geometry_artifact_root
from .selector_storage import SELECTOR_FILENAME, SelectorVisibleReader
from .storage import HIDDEN_FILENAME, verify_candidate_storage


REQUIRED_PREFLIGHT_BUNDLE_MEMBERS = frozenset(
    {
        "preflight/report.json",
        "preflight/resolved_config.yaml",
        "preflight/mask_manifest.json",
        "preflight/pretrained_manifest.json",
        "preflight/preprocessing_manifest.json",
        "environment/environment.json",
        "environment/package-lock.txt",
        "splits/manifest.json",
        "splits/waterbirds100_candidate_train.csv",
        "splits/waterbirds100_biased_val.csv",
        "splits/waterbirds100_expert_train.csv",
        "splits/waterbirds100_expert_calibration.csv",
        "splits/waterbirds100_oracle_val.csv",
        "splits/waterbirds100_test.csv",
        "preflight/geometry/expert_train_geometry.csv",
        "preflight/geometry/expert_calibration_geometry.csv",
        "preflight/geometry/biased_val_geometry.csv",
        "preflight/geometry/background_token_budget.json",
        "preflight/geometry/fixed_background_views.h5",
        "preflight/geometry/fixed_background_views.h5.manifest.json",
        "preflight/geometry/selector_eval_subset.csv",
        "preflight/geometry/donor_assignments.json",
        "preflight/geometry/manifest.json",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def _json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required campaign artifact is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditFailure(f"invalid JSON campaign artifact: {path}") from error
    _require(isinstance(value, dict), f"campaign JSON root is not a mapping: {path}")
    return value


def _run_id(learning_rate: float, weight_decay: float, seed: int) -> str:
    lr = f"{learning_rate:.0e}".replace("+", "").replace("-0", "-")
    return f"lr{lr}_wd{weight_decay:.2f}_seed{seed}"


def _verify_checksum_bundle(output: Path) -> int:
    checksum_path = output / "preflight" / "preflight_artifacts.sha256"
    _require(checksum_path.is_file(), "preflight checksum bundle is missing")
    records: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        _require(len(fields) == 2, "malformed preflight checksum bundle")
        expected, relative_text = fields
        relative_text = relative_text.lstrip(" *")
        relative = Path(relative_text)
        _require(
            not relative.is_absolute() and ".." not in relative.parts,
            f"preflight checksum path escapes the output root: {relative_text}",
        )
        normalized = relative.as_posix()
        _require(
            normalized not in records,
            f"duplicate preflight checksum member: {normalized}",
        )
        records[normalized] = expected
    _require(
        set(records) == REQUIRED_PREFLIGHT_BUNDLE_MEMBERS,
        "preflight checksum bundle membership mismatch; "
        f"missing={sorted(REQUIRED_PREFLIGHT_BUNDLE_MEMBERS - set(records))}, "
        f"unexpected={sorted(set(records) - REQUIRED_PREFLIGHT_BUNDLE_MEMBERS)}",
    )
    for relative_text, expected in records.items():
        relative = Path(relative_text)
        path = output / relative
        _require(path.is_file(), f"preflight bundle member is missing: {relative}")
        _require(
            len(expected) == 64 and sha256_file(path) == expected,
            f"preflight bundle hash mismatch: {relative}",
        )
    return len(records)


def verify_campaign_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    """Verify a completed debug or production campaign without mutating it.

    This is deliberately stricter than checking file existence. It binds the
    exact candidate grid, epoch count, visible/hidden HDF5 pair, receipts,
    configuration hashes, final analysis manifest, and production diversity
    gate into one post-run result.
    """

    output = Path(config["paths"]["output_root"])
    debug = bool(config["runtime"]["debug"])
    config_hash = str(config["resolved_config_sha256"])
    namespace = "debug/" if debug else ""

    preflight = _json(output / "preflight" / "report.json")
    _require(preflight.get("status") == "passed", "preflight did not pass")
    if not debug:
        _require(
            preflight.get("schema_version") == "anchorcal-preflight-v1",
            "production preflight schema is incompatible",
        )
        _require(
            preflight.get("resolved_config_sha256") == config_hash,
            "production preflight is not bound to the resolved config",
        )
        checksum_count = _verify_checksum_bundle(output)
    else:
        checksum_count = 0

    for branch in ("foreground", "background"):
        verify_branch_artifacts(config, branch)
    _require(
        (output / f"{namespace}anchors" / "criterion_results.json").is_file(),
        "anchor criterion results are missing",
    )

    receipt_root = output / f"{namespace}receipt"
    anchor_receipts = sorted(receipt_root.glob("anchorcal_decision_*.json"))
    _require(len(anchor_receipts) == 1, "expected exactly one AnchorCal decision receipt")
    anchor_receipt = anchor_receipts[0]
    _require(verify_decision_receipt(anchor_receipt), "AnchorCal decision receipt hash is invalid")
    anchor_payload = _json(anchor_receipt)
    _require(
        anchor_payload.get("receipt_type") == "anchorcal_criterion_decision"
        and anchor_payload.get("config_sha256", {}).get("resolved") == config_hash,
        "AnchorCal decision receipt is not bound to this campaign",
    )
    verify_anchor_artifacts(config, decision_receipt=anchor_receipt)

    selection_receipts = sorted(receipt_root.glob("candidate_selection_*.json"))
    _require(len(selection_receipts) == 1, "expected exactly one candidate-selection receipt")
    selection_receipt = selection_receipts[0]
    _require(
        verify_hashed_receipt(selection_receipt),
        "candidate-selection receipt hash is invalid",
    )
    selection_payload = _json(selection_receipt)
    _require(
        selection_payload.get("schema_version") == "anchorcal-candidate-selection-v2"
        and selection_payload.get("resolved_config_sha256") == config_hash
        and selection_payload.get("anchorcal_decision_sha256") == sha256_file(anchor_receipt)
        and selection_payload.get("hidden_namespace_opened") is False,
        "candidate-selection receipt provenance is invalid",
    )

    seed = int(config["candidate_grid"]["seed"])
    expected_grid = {
        _run_id(float(lr), float(wd), seed): (float(lr), float(wd))
        for lr in config["candidate_grid"]["learning_rates"]
        for wd in config["candidate_grid"]["weight_decays"]
    }
    candidate_root = output / f"{namespace}candidates"
    found = {path.name: path for path in candidate_root.iterdir() if path.is_dir()}
    _require(
        set(found) == set(expected_grid),
        "candidate grid is incomplete or contains unexpected run directories; "
        f"missing={sorted(set(expected_grid) - set(found))}, "
        f"unexpected={sorted(set(found) - set(expected_grid))}",
    )
    epochs = int(config["candidate_grid"]["epochs"])
    expected_slots = tuple(range(epochs))
    fixed_selector = pd.read_csv(
        geometry_artifact_root(config) / "selector_eval_subset.csv"
    ).sort_values("img_id", kind="stable").reset_index(drop=True)
    expected_selector_ids = fixed_selector["img_id"].to_numpy(dtype=np.int64)
    expected_selector_labels = fixed_selector["y"].to_numpy(dtype=np.int64)
    expected_selector_shapes = candidate_per_example_shapes(
        len(fixed_selector),
        swap_donors=int(config["criteria"]["swap_donors"]),
        blur_sigmas=len(config["criteria"]["blur_sigmas"]),
    )
    visible_hashes = selection_payload.get("visible_hdf5_sha256", {})
    visible_checkpoint_hashes = selection_payload.get(
        "visible_checkpoint_manifest_sha256", {}
    )
    _require(
        isinstance(visible_hashes, dict)
        and set(visible_hashes) == set(expected_grid)
        and isinstance(visible_checkpoint_hashes, dict)
        and set(visible_checkpoint_hashes) == set(expected_grid),
        "candidate-selection receipt does not bind the exact candidate grid",
    )
    candidate_results: dict[str, Any] = {}

    # This reporting-only reader is imported only after a valid, frozen
    # candidate-selection receipt has been verified above.
    from .hidden_storage import HiddenMetricsReader

    for run_id, run_dir in sorted(found.items()):
        learning_rate, weight_decay = expected_grid[run_id]
        storage_manifest = verify_candidate_storage(
            run_dir, expected_run_id=run_id
        )
        run_manifest = _json(run_dir / "run_manifest.json")
        completion = _json(run_dir / "completion.json")
        checkpoint_verification = verify_candidate_checkpoint_artifacts(
            run_dir,
            expected_run_id=run_id,
            require_complete=True,
            required_visible_selectors=(
                "ordinary",
                "saliency",
                "swap",
                "blur",
                "final",
            ),
            required_hidden_selectors=("oracle",),
        )
        _require(
            run_manifest.get("run_id") == run_id
            and float(run_manifest.get("learning_rate", -1)) == learning_rate
            and float(run_manifest.get("weight_decay", -1)) == weight_decay
            and int(run_manifest.get("seed", -1)) == seed
            and run_manifest.get("resolved_config_sha256") == config_hash,
            f"candidate run provenance mismatch: {run_id}",
        )
        _require(
            completion.get("run_id") == run_id
            and int(completion.get("epochs", -1)) == epochs
            and completion.get("checkpoint_manifest_sha256")
            == checkpoint_verification["visible"]["manifest_sha256"]
            and completion.get("hidden_checkpoint_manifest_sha256")
            == checkpoint_verification["hidden"]["manifest_sha256"],
            f"candidate completion receipt is invalid: {run_id}",
        )
        _require(
            visible_checkpoint_hashes.get(run_id)
            == checkpoint_verification["visible"]["manifest_sha256"],
            f"candidate checkpoint manifest differs from frozen selection: {run_id}",
        )
        visible_path = run_dir / SELECTOR_FILENAME
        hidden_path = run_dir / HIDDEN_FILENAME
        _require(
            visible_hashes.get(run_id) == sha256_file(visible_path),
            f"candidate visible file differs from the frozen selection: {run_id}",
        )
        with SelectorVisibleReader(
            visible_path, expected_run_id=run_id
        ) as visible_reader:
            visible_reader.validate_candidate_schema(
                metric_names=CANDIDATE_SCALAR_METRICS,
                per_example_shapes=expected_selector_shapes,
            )
            _require(
                visible_reader.completed_slots == expected_slots,
                f"selector-visible epochs are incomplete: {run_id}",
            )
            observed_selector = visible_reader.selector_subset_metadata()
            _require(
                observed_selector is not None
                and np.array_equal(observed_selector["img_id"], expected_selector_ids)
                and np.array_equal(
                    observed_selector["label"], expected_selector_labels
                ),
                f"fixed selector-subset metadata differs: {run_id}",
            )
        with HiddenMetricsReader(
            hidden_path, expected_run_id=run_id
        ) as hidden_reader:
            _require(
                hidden_reader.completed_slots == expected_slots,
                f"reporting-only epochs are incomplete: {run_id}",
            )
        candidate_results[run_id] = {
            "epochs": epochs,
            "selector_sha256": storage_manifest["files"]["selector_visible"]["sha256"],
            "hidden_sha256": storage_manifest["files"]["exploratory_hidden_metrics"]["sha256"],
            "checkpoint_manifest_sha256": checkpoint_verification["visible"][
                "manifest_sha256"
            ],
            "hidden_checkpoint_manifest_sha256": checkpoint_verification["hidden"][
                "manifest_sha256"
            ],
            "retained_checkpoint_weight_count": checkpoint_verification[
                "retained_weight_count"
            ],
        }

    analysis_root = output / f"{namespace}analysis"
    summary_path = analysis_root / "summary.json"
    summary = _json(summary_path)
    expected_candidates = len(expected_grid) * epochs
    expected_hidden_hashes = {
        run_id: result["hidden_sha256"]
        for run_id, result in sorted(candidate_results.items())
    }
    _require(
        summary.get("schema_version") == "anchorcal-final-analysis-v2"
        and int(summary.get("run_count", -1)) == len(expected_grid)
        and int(summary.get("candidate_count", -1)) == expected_candidates
        and summary.get("selection_receipt_sha256") == sha256_file(selection_receipt),
        "final analysis summary is incomplete or bound to the wrong campaign",
    )
    _require(
        summary.get("hidden_hdf5_sha256") == expected_hidden_hashes,
        "final analysis summary does not bind the exact per-run hidden HDF5 files",
    )
    analysis_manifest = _json(analysis_root / "manifest.json")
    _require(
        analysis_manifest.get("schema_version") == "anchorcal-analysis-manifest-v1"
        and analysis_manifest.get("selection_receipt_sha256")
        == sha256_file(selection_receipt),
        "final analysis manifest provenance is invalid",
    )
    _require(
        analysis_manifest.get("hidden_hdf5_sha256") == expected_hidden_hashes,
        "final analysis manifest does not bind the exact per-run hidden HDF5 files",
    )
    files = analysis_manifest.get("files")
    _require(isinstance(files, dict) and files, "final analysis manifest has no files")
    for relative_text, expected in files.items():
        relative = Path(relative_text)
        _require(
            not relative.is_absolute() and ".." not in relative.parts,
            f"analysis manifest path escapes its root: {relative_text}",
        )
        path = analysis_root / relative
        _require(
            path.is_file() and sha256_file(path) == expected,
            f"analysis artifact hash mismatch: {relative_text}",
        )
    _require("summary.json" in files, "analysis manifest does not cover summary.json")
    if not debug:
        _require(
            not (analysis_root / "candidate_diversity_failure.json").exists(),
            "production candidate-diversity gate failed",
        )

    return {
        "schema_version": "anchorcal-campaign-verification-v1",
        "status": "complete",
        "debug": debug,
        "preflight_checksum_count": checksum_count,
        "anchorcal_decision_receipt": str(anchor_receipt.resolve()),
        "candidate_selection_receipt": str(selection_receipt.resolve()),
        "run_count": len(candidate_results),
        "candidate_count": expected_candidates,
        "candidates": candidate_results,
        "final_summary": str(summary_path.resolve()),
        "analysis_manifest": str((analysis_root / "manifest.json").resolve()),
    }
