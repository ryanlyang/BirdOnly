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
from .mask_identity import (
    SELECTOR_MASK_RECEIPT_SCHEMA,
    VLM_PRODUCER,
    vlm_mask_contract_hash,
)


CANDIDATE_RUN_MANIFEST_SCHEMA = "anchorcal-candidate-run-v4"
SELECTOR_MASK_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "namespace",
        "contains_per_row_records",
        "selector_required_official_splits",
        "resolved_config_sha256",
        "metadata_sha256",
        "git_commit",
        "mask_source",
        "mask_contract_sha256",
        "mask_bank_sha256",
        "mask_manifest_sha256",
        "foreground_area_summary_sha256",
        "mask_visual_audit_manifest_sha256",
        "mask_visual_audit_selection_sha256",
    }
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_candidate_preflight_binding(config: Mapping[str, Any]) -> dict[str, Any]:
    """Load only the compact selector-safe mask/preflight identity.

    This code path intentionally neither opens the per-row mask manifest nor
    imports its loader.  Candidate/branch jobs validate source masks before
    producing their artifacts; final selection consumes only this aggregate
    receipt and the hashes frozen into each candidate run manifest.
    """

    output = Path(str(config["paths"]["output_root"]))
    receipt_path = output / "preflight" / "selector_mask_receipt.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError("invalid selector-safe mask receipt") from error
    if not isinstance(receipt, dict) or set(receipt) != SELECTOR_MASK_RECEIPT_KEYS:
        raise PreflightError("selector-safe mask receipt has an incompatible schema")
    contract_hash = vlm_mask_contract_hash(config)
    if (
        receipt.get("schema_version") != SELECTOR_MASK_RECEIPT_SCHEMA
        or receipt.get("status") != "passed"
        or receipt.get("namespace") != "selector_visible"
        or receipt.get("contains_per_row_records") is not False
        or receipt.get("selector_required_official_splits") != [0]
        or (
            not bool(config.get("runtime", {}).get("debug", False))
            and receipt.get("resolved_config_sha256")
            != config["resolved_config_sha256"]
        )
        or receipt.get("mask_source") != VLM_PRODUCER
        or receipt.get("mask_contract_sha256") != contract_hash
        or any(
            not _is_sha256(receipt.get(key))
            for key in (
                "resolved_config_sha256",
                "metadata_sha256",
                "mask_contract_sha256",
                "mask_bank_sha256",
                "mask_manifest_sha256",
                "foreground_area_summary_sha256",
                "mask_visual_audit_manifest_sha256",
                "mask_visual_audit_selection_sha256",
            )
        )
        or not isinstance(receipt.get("git_commit"), str)
        or len(str(receipt.get("git_commit"))) < 7
    ):
        raise PreflightError("selector-safe mask receipt is incompatible")
    return {
        **receipt,
        "_receipt_path": str(receipt_path.resolve()),
        "_receipt_sha256": sha256_file(receipt_path),
    }


def require_candidate_run_manifest(
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    selector_receipt: Mapping[str, Any],
    *,
    expected_run_id: str,
    expected_decision_sha256: str,
) -> None:
    """Require the common candidate and VLM identity before reading artifacts."""

    if (
        manifest.get("schema_version") != CANDIDATE_RUN_MANIFEST_SCHEMA
        or manifest.get("run_id") != expected_run_id
        or manifest.get("resolved_config_sha256")
        != config["resolved_config_sha256"]
        or manifest.get("decision_receipt_sha256") != expected_decision_sha256
        or Path(str(manifest.get("selector_mask_receipt", ""))).resolve()
        != Path(str(selector_receipt.get("_receipt_path", ""))).resolve()
        or manifest.get("selector_mask_receipt_sha256")
        != selector_receipt.get("_receipt_sha256")
        or manifest.get("metadata_sha256")
        != selector_receipt.get("metadata_sha256")
        or manifest.get("mask_bank_sha256")
        != selector_receipt.get("mask_bank_sha256")
        or manifest.get("mask_manifest_sha256")
        != selector_receipt.get("mask_manifest_sha256")
        or manifest.get("mask_source") != VLM_PRODUCER
        or manifest.get("mask_source") != selector_receipt.get("mask_source")
        or manifest.get("mask_contract") != config.get("masks")
    ):
        raise PreflightError(
            f"candidate run/VLM provenance mismatch: {expected_run_id}"
        )
