"""Selector-safe verification of the frozen AnchorCal criterion decision.

Unlike the full reporting verifier, this module never opens the preflight
report, a per-row mask manifest, protected splits, or branch provenance.  The
anchor job binds its already-verified inputs into an immutable decision receipt;
selection verifies that receipt against the compact selector mask receipt and
the public AnchorCal result artifacts it actually consumes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .candidate_provenance import load_candidate_preflight_binding
from .decision import verify_decision_receipt
from .errors import PreflightError
from .io import hash_object, sha256_file


ANCHOR_RESULT_KEYS = frozenset(
    {
        "ordinary_accuracy",
        "saliency_harmonic",
        "token_swap_harmonic",
        "background_blur_harmonic",
        "foreground_only_harmonic",
        "saliency_product",
        "swap_product",
        "blur_product",
    }
)

# Deliberately duplicated from the full reporting verifier instead of importing
# it.  The selector process must remain unable to import reporting-only
# verification code, while still failing closed if a future artifact producer
# adds a file to the public manifest without an accompanying schema review.
ANCHOR_ARTIFACT_KEYS = frozenset(
    {
        "branch_audits",
        "competence_intersection",
        "competence_intersection_manifest",
        "criterion_subset",
        "margin_scales",
        "anchor_per_image_outputs",
        "anchor_bootstrap_score_vectors",
        "criterion_results",
        "anchor_scores",
        "anchor_bootstrap_metrics",
        "anchor_intervention_diagnostics",
        "cache_parity",
        "foreground_stream_intervention_audit",
    }
)


def _json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError(f"invalid {description}: {path}") from error
    if not isinstance(value, dict):
        raise PreflightError(f"{description} must be a JSON mapping: {path}")
    return value


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _expected_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Return the complete, selector-reviewed public artifact allowlist."""

    output = Path(config["paths"]["output_root"])
    anchor_namespace = "debug/anchors" if config["runtime"]["debug"] else "anchors"
    audit_namespace = "debug/audits" if config["runtime"]["debug"] else "audits"
    root = output / anchor_namespace
    return {
        "branch_audits": output / audit_namespace / "branch_audits.json",
        "competence_intersection": root / "competence_intersection.csv",
        "competence_intersection_manifest": root
        / "competence_intersection_manifest.json",
        "criterion_subset": root / "criterion_subset.csv",
        "margin_scales": root / "margin_scales.json",
        "anchor_per_image_outputs": root / "anchor_per_image_outputs.npz",
        "anchor_bootstrap_score_vectors": root
        / "anchor_bootstrap_score_vectors.npz",
        "criterion_results": root / "criterion_results.json",
        "anchor_scores": root / "anchor_scores.csv",
        "anchor_bootstrap_metrics": root / "anchor_bootstrap_metrics.csv",
        "anchor_intervention_diagnostics": root
        / "anchor_intervention_diagnostics.csv",
        "cache_parity": root / "cache_parity.json",
        "foreground_stream_intervention_audit": root
        / "foreground_stream_intervention_audit.json",
    }


def verify_selector_anchor_artifacts(
    config: dict[str, Any], decision_receipt: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify selector-visible anchor artifacts without hidden mask access."""

    output = Path(config["paths"]["output_root"]).resolve()
    namespace = "debug/anchors" if config["runtime"]["debug"] else "anchors"
    anchor_root = output / namespace
    manifest_path = anchor_root / "artifact_manifest.json"
    results_path = anchor_root / "criterion_results.json"
    subset_path = anchor_root / "criterion_subset.csv"
    manifest = _json(manifest_path, "AnchorCal artifact manifest")
    results = _json(results_path, "AnchorCal criterion results")
    if (
        manifest.get("schema_version") != "anchorcal-anchor-artifacts-v1"
        or manifest.get("resolved_config_sha256")
        != config["resolved_config_sha256"]
        or set(manifest.get("criterion_result_keys", [])) != ANCHOR_RESULT_KEYS
        or set(results) != ANCHOR_RESULT_KEYS
    ):
        raise PreflightError("selector-visible AnchorCal artifacts are incompatible")
    files = manifest.get("files")
    expected_paths = _expected_paths(config)
    if not isinstance(files, dict) or set(files) != ANCHOR_ARTIFACT_KEYS:
        raise PreflightError(
            "AnchorCal selector artifact file set is not exactly allowlisted"
        )
    for name, expected_path in expected_paths.items():
        record = files[name]
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise PreflightError(f"AnchorCal artifact record is malformed: {name}")
        path = (output / str(record["path"])).resolve()
        try:
            path.relative_to(output)
        except ValueError as error:
            raise PreflightError(f"AnchorCal artifact escapes output root: {name}") from error
        if (
            path != expected_path.resolve()
            or not path.is_file()
            or int(record.get("size_bytes", -1)) != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            raise PreflightError(f"AnchorCal artifact failed verification: {name}")
    if (
        "criterion_results" not in files
        or files["criterion_results"].get("sha256") != sha256_file(results_path)
        or "criterion_subset" not in files
        or files["criterion_subset"].get("sha256") != sha256_file(subset_path)
    ):
        raise PreflightError("AnchorCal selector inputs are not manifest-bound")

    receipt_path = Path(decision_receipt).resolve()
    if not verify_decision_receipt(receipt_path):
        raise PreflightError("AnchorCal decision receipt hash is invalid")
    receipt = _json(receipt_path, "AnchorCal decision receipt")
    provenance = receipt.get("provenance")
    selector_mask = load_candidate_preflight_binding(config)
    branch_hashes = receipt.get("branch_sha256")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("receipt_type") != "anchorcal_criterion_decision"
        or receipt.get("config_sha256")
        != {"resolved": config["resolved_config_sha256"]}
        or not isinstance(provenance, dict)
        or Path(str(provenance.get("selector_mask_receipt", ""))).resolve()
        != Path(selector_mask["_receipt_path"]).resolve()
        or provenance.get("selector_mask_receipt_sha256")
        != selector_mask["_receipt_sha256"]
        or Path(str(provenance.get("anchor_artifact_manifest", ""))).resolve()
        != manifest_path.resolve()
        or provenance.get("anchor_artifact_manifest_sha256")
        != sha256_file(manifest_path)
        or provenance.get("criterion_results_sha256")
        != sha256_file(results_path)
        or provenance.get("mask_bank_sha256")
        != selector_mask["mask_bank_sha256"]
        or provenance.get("mask_manifest_sha256")
        != selector_mask["mask_manifest_sha256"]
        or provenance.get("mask_source") != selector_mask["mask_source"]
        or provenance.get("mask_contract") != config.get("masks")
        or not isinstance(branch_hashes, dict)
        or set(branch_hashes) != {"foreground", "background"}
        or any(not _valid_sha256(value) for value in branch_hashes.values())
    ):
        raise PreflightError("selector-safe AnchorCal decision provenance is invalid")
    try:
        import pandas as pd

        subset_ids = sorted(
            pd.read_csv(subset_path)["img_id"].astype(int).tolist()
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise PreflightError("AnchorCal criterion subset is malformed") from error
    if (
        not subset_ids
        or len(set(subset_ids)) != len(subset_ids)
        or receipt.get("anchor_subset_sha256") != hash_object(subset_ids)
    ):
        raise PreflightError("AnchorCal criterion subset is not receipt-bound")
    return manifest, results
