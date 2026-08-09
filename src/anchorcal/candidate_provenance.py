"""Shared provenance checks for candidate trajectories.

Both selector-only and reporting-only analysis consume candidate artifacts.
Keeping their run-manifest checks here prevents either side of the frozen
selection boundary from silently accepting an older or differently masked
trajectory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .errors import PreflightError
from .io import sha256_file
from .vlm_masks import (
    VLM_MASK_MANIFEST_SCHEMA,
    VLM_PRODUCER,
    load_vlm_mask_bank,
    vlm_mask_contract_hash,
)


CANDIDATE_RUN_MANIFEST_SCHEMA = "anchorcal-candidate-run-v3"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_candidate_preflight_binding(config: Mapping[str, Any]) -> dict[str, Any]:
    """Load and verify the compact preflight identity used by candidate jobs."""

    output = Path(str(config["paths"]["output_root"]))
    report_path = output / "preflight" / "report.json"
    manifest_path = output / "preflight" / "mask_manifest.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        mask_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError("invalid candidate preflight/VLM manifest") from error
    if not isinstance(report, dict) or not isinstance(mask_manifest, dict):
        raise PreflightError("candidate preflight artifacts must be JSON mappings")
    # This intentionally performs the complete manifest-entry and source-file
    # verification once at each analysis boundary.  Verifying only the frozen
    # manifest bytes would not detect a source PNG changed after candidate
    # evaluation.
    mask_bank = load_vlm_mask_bank(config)
    contract_hash = vlm_mask_contract_hash(config)
    manifest_hash = sha256_file(manifest_path)
    if (
        report.get("status") != "passed"
        or report.get("resolved_paths") != config["paths"]
        or (
            not bool(config.get("runtime", {}).get("debug", False))
            and report.get("resolved_config_sha256")
            != config["resolved_config_sha256"]
        )
        or report.get("mask_source") != VLM_PRODUCER
        or report.get("mask_contract_sha256") != contract_hash
        or not _is_sha256(report.get("mask_bank_sha256"))
        or report.get("mask_bank_sha256") != mask_bank.mask_bank_sha256
        or report.get("mask_manifest_sha256") != manifest_hash
        or mask_manifest.get("schema_version") != VLM_MASK_MANIFEST_SCHEMA
        or mask_manifest.get("status") != "passed"
        or mask_manifest.get("producer") != VLM_PRODUCER
        or mask_manifest.get("mask_contract_sha256") != contract_hash
        or mask_manifest.get("mask_bank_sha256")
        != report.get("mask_bank_sha256")
    ):
        raise PreflightError("candidate preflight VLM provenance is incompatible")
    return report


def require_candidate_run_manifest(
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    expected_run_id: str,
    expected_decision_sha256: str,
) -> None:
    """Require the common candidate and VLM identity before reading artifacts."""

    preflight_path = (
        Path(str(config["paths"]["output_root"])) / "preflight" / "report.json"
    )
    if (
        manifest.get("schema_version") != CANDIDATE_RUN_MANIFEST_SCHEMA
        or manifest.get("run_id") != expected_run_id
        or manifest.get("resolved_config_sha256")
        != config["resolved_config_sha256"]
        or manifest.get("decision_receipt_sha256") != expected_decision_sha256
        or Path(str(manifest.get("preflight_report", ""))).resolve()
        != preflight_path.resolve()
        or manifest.get("preflight_report_sha256") != sha256_file(preflight_path)
        or manifest.get("mask_bank_sha256") != preflight.get("mask_bank_sha256")
        or manifest.get("mask_manifest_sha256")
        != preflight.get("mask_manifest_sha256")
        or manifest.get("mask_source") != VLM_PRODUCER
        or manifest.get("mask_source") != preflight.get("mask_source")
        or manifest.get("mask_contract") != config.get("masks")
    ):
        raise PreflightError(
            f"candidate run/VLM provenance mismatch: {expected_run_id}"
        )
