"""Read-only verification of frozen branch checkpoints and output artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .background import load_token_budget_manifest
from .errors import PreflightError
from .io import sha256_file
from .paths import geometry_artifact_root


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError(f"invalid {description}: {path}") from error
    if not isinstance(value, dict):
        raise PreflightError(f"{description} must be a JSON mapping: {path}")
    return value


def _load_json_list(path: Path, description: str) -> list[Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError(f"invalid {description}: {path}") from error
    if not isinstance(value, list):
        raise PreflightError(f"{description} must be a JSON list: {path}")
    return value


def _require_file_record(
    record: Any, expected_path: Path, description: str
) -> None:
    if not isinstance(record, dict):
        raise PreflightError(f"missing {description} artifact record")
    recorded = Path(str(record.get("path", "")))
    if (
        not expected_path.is_file()
        or recorded.resolve() != expected_path.resolve()
        or int(record.get("size_bytes", -1)) != expected_path.stat().st_size
        or record.get("sha256") != sha256_file(expected_path)
    ):
        raise PreflightError(f"{description} artifact failed provenance verification")


def _valid_background_gate(
    manifest: dict[str, Any],
    config: dict[str, Any],
    token_payload: dict[str, Any],
    biased_path: Path,
) -> bool:
    gate = manifest.get("background_validity_gate")
    if not isinstance(gate, dict):
        return False
    invalid_count_value = gate.get("invalid_count")
    total_value = gate.get("total")
    invalid_fraction_value = gate.get("invalid_fraction")
    maximum_value = gate.get("maximum_invalid_fraction")
    recorded_fraction_value = manifest.get("invalid_biased_val_fraction")
    if (
        not isinstance(invalid_count_value, int)
        or isinstance(invalid_count_value, bool)
        or not isinstance(total_value, int)
        or isinstance(total_value, bool)
        or not isinstance(invalid_fraction_value, (int, float))
        or isinstance(invalid_fraction_value, bool)
        or not isinstance(maximum_value, (int, float))
        or isinstance(maximum_value, bool)
        or not isinstance(recorded_fraction_value, (int, float))
        or isinstance(recorded_fraction_value, bool)
    ):
        return False
    invalid_count = invalid_count_value
    total = total_value
    invalid_fraction = float(invalid_fraction_value)
    maximum = float(maximum_value)
    recorded_fraction = float(recorded_fraction_value)
    try:
        with np.load(biased_path, allow_pickle=False) as outputs:
            valid = outputs["valid"]
    except (OSError, KeyError, ValueError):
        return False
    if valid.dtype != np.bool_ or valid.ndim != 1 or valid.size <= 0:
        return False
    output_invalid_count = int(np.sum(~valid))
    output_total = int(valid.size)
    selected_invalidity = token_payload.get(
        "selected_biased_val_invalidity"
    )
    if not isinstance(selected_invalidity, dict):
        return False
    selected_count = int(selected_invalidity["count"])
    selected_total = int(selected_invalidity["total"])
    selected_fraction = float(selected_invalidity["fraction"])
    configured_maximum = float(
        config["branches"]["background_max_biased_val_invalid_fraction"]
    )
    return bool(
        set(gate)
        == {
            "schema_version",
            "status",
            "invalid_count",
            "total",
            "invalid_fraction",
            "maximum_invalid_fraction",
            "scope",
        }
        and gate.get("schema_version")
        == "anchorcal-background-validity-gate-v1"
        and gate.get("status") == "passed"
        and gate.get("scope") == "overall_biased_val"
        and total > 0
        and 0 <= invalid_count <= total
        and invalid_count == output_invalid_count == selected_count
        and total == output_total == selected_total
        and math.isclose(
            invalid_fraction,
            selected_fraction,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        and math.isclose(
            invalid_fraction,
            invalid_count / total,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        and math.isclose(
            recorded_fraction,
            invalid_fraction,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        and maximum == configured_maximum
        and invalid_fraction <= configured_maximum
    )


def verify_branch_artifacts(
    config: dict[str, Any], branch: str
) -> dict[str, Any]:
    """Verify one final branch without loading its model or mutating files."""

    if branch not in {"foreground", "background"}:
        raise ValueError("branch must be foreground or background")
    output = Path(config["paths"]["output_root"])
    namespace = "debug/branches" if config["runtime"]["debug"] else "branches"
    root = output / namespace / branch
    manifest_path = root / "manifest.json"
    manifest = _load_json(manifest_path, f"{branch} branch manifest")
    preflight_path = output / "preflight" / "report.json"
    preflight = _load_json(preflight_path, "preflight report")
    preprocessing_path = output / "preflight" / "preprocessing_manifest.json"
    mask_manifest_path = output / "preflight" / "mask_manifest.json"
    checkpoint_path = root / "epoch_final.pt"
    calibration_path = root / "expert_calibration_outputs.npz"
    biased_path = root / "biased_val_outputs.npz"
    fallback_path = root / "crop_fallback_events.json"
    fallback_gate_path = root / "crop_fallback_gate.json"
    history_path = root / "history.json"
    optimizer_groups_path = root / "optimizer_groups.json"

    expected_token_budget: int | None = None
    token_payload: dict[str, Any] | None = None
    if branch == "background":
        token_payload = load_token_budget_manifest(
            geometry_artifact_root(config) / "background_token_budget.json",
            config,
        )
        expected_token_budget = int(token_payload["token_budget"])
    expected_mode = (
        config["branches"]["foreground_position_mode"]
        if branch == "foreground"
        else None
    )
    if (
        manifest.get("schema_version") != "anchorcal-branch-manifest-v4"
        or manifest.get("branch") != branch
        or int(manifest.get("fixed_epoch", -1))
        != int(config["branches"]["frozen_epoch"])
        or manifest.get("resolved_config_sha256")
        != config["resolved_config_sha256"]
        or manifest.get("paths") != config["paths"]
        or manifest.get("token_budget") != expected_token_budget
        or manifest.get("foreground_position_mode") != expected_mode
        or Path(str(manifest.get("preflight_report", ""))).resolve()
        != preflight_path.resolve()
        or manifest.get("preflight_report_sha256") != sha256_file(preflight_path)
        or manifest.get("metadata_sha256") != preflight.get("metadata_sha256")
        or manifest.get("mask_bank_sha256") != preflight.get("mask_bank_sha256")
        or manifest.get("mask_manifest_sha256")
        != preflight.get("mask_manifest_sha256")
        or manifest.get("mask_source") != preflight.get("mask_source")
        or manifest.get("mask_contract") != config.get("masks")
        or not mask_manifest_path.is_file()
        or preflight.get("mask_manifest_sha256")
        != sha256_file(mask_manifest_path)
        or manifest.get("git_commit") != preflight.get("git", {}).get("commit")
        or not preprocessing_path.is_file()
        or manifest.get("preprocessing_manifest_sha256")
        != sha256_file(preprocessing_path)
        or manifest.get("preprocessing_manifest_sha256")
        != preflight.get("preprocessing", {}).get("manifest_sha256")
        or (
            branch == "foreground"
            and manifest.get("background_validity_gate") is not None
        )
    ):
        raise PreflightError(f"{branch} branch manifest provenance mismatch")
    if (
        not checkpoint_path.is_file()
        or Path(str(manifest.get("checkpoint", ""))).resolve()
        != checkpoint_path.resolve()
        or manifest.get("checkpoint_sha256") != sha256_file(checkpoint_path)
        or int(manifest.get("checkpoint_size_bytes", -1))
        != checkpoint_path.stat().st_size
    ):
        raise PreflightError(f"{branch} branch checkpoint failed hash verification")
    outputs = manifest.get("outputs", {})
    _require_file_record(
        outputs.get("expert_calibration"), calibration_path, f"{branch} calibration"
    )
    _require_file_record(outputs.get("biased_val"), biased_path, f"{branch} biased-val")
    if branch == "background" and (
        token_payload is None
        or not _valid_background_gate(
            manifest,
            config,
            token_payload,
            biased_path,
        )
    ):
        raise PreflightError("background branch validity provenance mismatch")
    training_artifacts = manifest.get("training_artifacts", {})
    _require_file_record(
        training_artifacts.get("history"), history_path, f"{branch} training history"
    )
    _require_file_record(
        training_artifacts.get("optimizer_groups"),
        optimizer_groups_path,
        f"{branch} optimizer groups",
    )
    history = _load_json_list(history_path, f"{branch} training history")
    optimizer_groups = _load_json(
        optimizer_groups_path, f"{branch} optimizer groups"
    )
    expected_epochs = int(config["branches"]["frozen_epoch"])
    if (
        len(history) != expected_epochs
        or [int(record.get("epoch", -1)) for record in history]
        != list(range(1, expected_epochs + 1))
        or optimizer_groups != manifest.get("optimizer")
    ):
        raise PreflightError(f"{branch} training artifacts are incompatible")
    _require_file_record(
        {
            "path": manifest.get("crop_fallback_events"),
            "sha256": manifest.get("crop_fallback_events_sha256"),
            "size_bytes": manifest.get("crop_fallback_events_size_bytes"),
        },
        fallback_path,
        f"{branch} crop-fallback",
    )
    fallback_payload = _load_json(
        fallback_path, f"{branch} crop-fallback event report"
    )
    _require_file_record(
        {
            "path": manifest.get("crop_fallback_gate"),
            "sha256": manifest.get("crop_fallback_gate_sha256"),
            "size_bytes": manifest.get("crop_fallback_gate_size_bytes"),
        },
        fallback_gate_path,
        f"{branch} crop-fallback gate",
    )
    gate_payload = _load_json(
        fallback_gate_path, f"{branch} crop-fallback gate report"
    )
    aggregate = manifest.get("crop_fallback_aggregate", {})
    fallback_count = int(aggregate.get("fallback_count", -1))
    sampled_examples = int(aggregate.get("sampled_examples", -1))
    fallback_rate = float(aggregate.get("fallback_rate", float("nan")))
    history_fallback_count = sum(
        int(record.get("fallback_count", -1)) for record in history
    )
    history_sampled_examples = sum(
        int(record.get("sampled_examples", -1)) for record in history
    )
    if (
        fallback_payload.get("schema_version")
        != "anchorcal-branch-crop-fallback-events-v1"
        or fallback_payload.get("branch") != branch
        or int(fallback_payload.get("event_count", -1)) != fallback_count
        or int(manifest.get("crop_fallback_event_count", -1)) != fallback_count
        or history_fallback_count != fallback_count
        or history_sampled_examples != sampled_examples
        or sampled_examples <= 0
        or not math.isclose(
            fallback_rate,
            fallback_count / sampled_examples,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or fallback_rate > float(config["data"]["crop_fallback_rate_gate"])
        or aggregate.get("gate_scope")
        != "all_sampled_training_examples_across_all_epochs"
        or gate_payload.get("schema_version")
        != "anchorcal-branch-crop-fallback-gate-v1"
        or gate_payload.get("branch") != branch
        or gate_payload.get("status") != "passed"
        or int(gate_payload.get("fallback_count", -1)) != fallback_count
        or int(gate_payload.get("sampled_examples", -1)) != sampled_examples
        or float(gate_payload.get("maximum_rate", -1.0))
        != float(config["data"]["crop_fallback_rate_gate"])
    ):
        raise PreflightError(f"{branch} crop-fallback aggregate is incompatible")
    return manifest
