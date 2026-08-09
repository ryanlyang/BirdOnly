"""Immutable provenance contract for the complete AnchorCal ladder output."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .anchor_cache import MARGIN_SCALE_EPSILON
from .branch_provenance import verify_branch_artifacts
from .decision import verify_decision_receipt
from .errors import PreflightError
from .io import hash_object, sha256_file
from .metrics import HARMONIC_EPSILON


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


def _expected_paths(config: dict[str, Any]) -> dict[str, Path]:
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


def verify_anchor_artifacts(
    config: dict[str, Any], *, decision_receipt: str | Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the manifest, every artifact byte stream, and result schema."""

    output = Path(config["paths"]["output_root"]).resolve()
    namespace = "debug/anchors" if config["runtime"]["debug"] else "anchors"
    manifest_path = output / namespace / "artifact_manifest.json"
    manifest = _json(manifest_path, "AnchorCal artifact manifest")
    definitions = manifest.get("definitions", {})
    if (
        manifest.get("schema_version") != "anchorcal-anchor-artifacts-v1"
        or manifest.get("resolved_config_sha256")
        != config["resolved_config_sha256"]
        or set(manifest.get("criterion_result_keys", [])) != ANCHOR_RESULT_KEYS
        or definitions.get("margin_normalization_epsilon")
        != MARGIN_SCALE_EPSILON
        or definitions.get("harmonic_mean_epsilon") != HARMONIC_EPSILON
    ):
        raise PreflightError("AnchorCal artifact manifest contract is incompatible")
    records = manifest.get("files")
    if not isinstance(records, dict) or set(records) != ANCHOR_ARTIFACT_KEYS:
        raise PreflightError("AnchorCal artifact manifest file set is incomplete")
    expected_paths = _expected_paths(config)
    for name, expected_path in expected_paths.items():
        record = records.get(name, {})
        relative = record.get("path")
        if not isinstance(relative, str):
            raise PreflightError(f"AnchorCal artifact path is missing: {name}")
        actual = (output / relative).resolve()
        try:
            actual.relative_to(output)
        except ValueError as error:
            raise PreflightError(f"AnchorCal artifact escapes output root: {name}") from error
        if (
            actual != expected_path.resolve()
            or not actual.is_file()
            or int(record.get("size_bytes", -1)) != actual.stat().st_size
            or record.get("sha256") != sha256_file(actual)
        ):
            raise PreflightError(f"AnchorCal artifact failed verification: {name}")
    results = _json(expected_paths["criterion_results"], "AnchorCal criterion results")
    if set(results) != ANCHOR_RESULT_KEYS:
        raise PreflightError("AnchorCal criterion result schema is incomplete")
    lambdas = tuple(float(value) for value in config["anchorcal"]["lambdas"])
    for name, result in results.items():
        if (
            not isinstance(result, dict)
            or set(result) != {"scores", "metrics", "bootstrap"}
            or len(result.get("scores", [])) != len(lambdas)
            or not isinstance(result.get("metrics"), dict)
            or not isinstance(result.get("bootstrap"), dict)
        ):
            raise PreflightError(f"AnchorCal criterion result is malformed: {name}")
    scales = _json(expected_paths["margin_scales"], "AnchorCal margin scales")
    if (
        scales.get("normalization_epsilon") != MARGIN_SCALE_EPSILON
        or not math.isfinite(float(scales.get("foreground", float("nan"))))
        or not math.isfinite(float(scales.get("background", float("nan"))))
        or float(scales.get("foreground", 0.0)) <= 0.0
        or float(scales.get("background", 0.0)) <= 0.0
    ):
        raise PreflightError("AnchorCal margin normalization epsilon changed")
    if decision_receipt is not None:
        receipt_path = Path(decision_receipt)
        if not verify_decision_receipt(receipt_path):
            raise PreflightError("AnchorCal decision receipt hash is invalid")
        receipt = _json(receipt_path, "AnchorCal decision receipt")
        provenance = receipt.get("provenance", {})
        preflight_path = output / "preflight" / "report.json"
        preflight = _json(preflight_path, "preflight report")
        subset = pd.read_csv(expected_paths["criterion_subset"])
        if (
            "img_id" not in subset
            or subset["img_id"].duplicated().any()
            or len(subset) == 0
        ):
            raise PreflightError("AnchorCal criterion subset IDs are invalid")
        subset_ids = sorted(subset["img_id"].astype(int).tolist())
        independently_computed_subset_hash = hash_object(subset_ids)
        foreground_manifest = verify_branch_artifacts(config, "foreground")
        background_manifest = verify_branch_artifacts(config, "background")
        expected_branch_hashes = {
            "background": background_manifest["checkpoint_sha256"],
            "foreground": foreground_manifest["checkpoint_sha256"],
        }
        family = receipt.get("anchor_family")
        expected_lambdas = [
            float(value) for value in config["anchorcal"]["lambdas"]
        ]
        try:
            family_matches = bool(
                isinstance(family, dict)
                and family.get("type")
                == "normalized_centered_raw_logit_reliance"
                and [float(value) for value in family.get("lambdas", [])]
                == expected_lambdas
                and float(family.get("foreground_scale"))
                == float(scales["foreground"])
                and float(family.get("background_scale"))
                == float(scales["background"])
                and float(family.get("normalization_epsilon"))
                == float(scales["normalization_epsilon"])
                and float(family.get("harmonic_mean_epsilon"))
                == HARMONIC_EPSILON
            )
        except (TypeError, ValueError):
            family_matches = False
        if (
            receipt.get("schema_version") != 1
            or receipt.get("receipt_type") != "anchorcal_criterion_decision"
            or receipt.get("config_sha256")
            != {"resolved": config["resolved_config_sha256"]}
            or receipt.get("branch_sha256") != expected_branch_hashes
            or receipt.get("anchor_subset_sha256")
            != independently_computed_subset_hash
            or scales.get("criterion_subset_hash")
            != independently_computed_subset_hash
            or not family_matches
            or Path(str(provenance.get("anchor_artifact_manifest", ""))).resolve()
            != manifest_path.resolve()
            or provenance.get("anchor_artifact_manifest_sha256")
            != sha256_file(manifest_path)
            or provenance.get("criterion_results_sha256")
            != records["criterion_results"]["sha256"]
            or Path(str(provenance.get("preflight_report", ""))).resolve()
            != preflight_path.resolve()
            or provenance.get("preflight_report_sha256")
            != sha256_file(preflight_path)
            or provenance.get("mask_bank_sha256")
            != preflight.get("mask_bank_sha256")
            or provenance.get("mask_manifest_sha256")
            != preflight.get("mask_manifest_sha256")
            or provenance.get("mask_source") != preflight.get("mask_source")
            or provenance.get("mask_contract") != config.get("masks")
        ):
            raise PreflightError(
                "AnchorCal artifacts differ from the frozen decision receipt"
            )
    return manifest, results
