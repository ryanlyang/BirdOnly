"""End-to-end Phase 0 artifact builder and verifier."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from setv.data.audit import render_contact_sheets, select_visual_audit_samples
from setv.data.masks import MaskResolver, inspect_mask
from setv.data.splits import SAFE_COLUMNS, build_splits, read_and_validate_metadata
from setv.errors import ArtifactExistsError, DataValidationError
from setv.utils.hashing import sha256_file, sha256_json
from setv.utils.io import write_json
from setv.utils.logging import EventLogger
from setv.utils.provenance import runtime_provenance


BASE_ARTIFACT_MANIFEST = "artifact_manifest.json"
APPROVAL_RECEIPT = "mask_audit/visual_review_approval.json"


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=True)


def _source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _manifest_all_base_files(root: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    excluded = {BASE_ARTIFACT_MANIFEST, APPROVAL_RECEIPT}
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        files[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return {
        "schema_version": 1,
        "files": files,
        "manifest_digest": sha256_json(files),
    }


def build_phase0(config: dict[str, Any]) -> Path:
    """Build an immutable Phase 0 directory after all automated checks pass."""
    destination = Path(config["output"]["phase0_dir"]).expanduser().resolve()
    if destination.exists():
        raise ArtifactExistsError(
            f"Phase 0 destination already exists and will not be overwritten: "
            f"{destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.building.", dir=destination.parent)
    )
    logger = EventLogger(staging / "logs" / "phase0.jsonl", echo=True)
    try:
        logger.log("phase0_started", destination=str(destination))
        metadata = read_and_validate_metadata(config)
        logger.log("metadata_validated", sample_count=int(len(metadata)))
        split_result = build_splits(metadata, config)
        logger.log(
            "splits_constructed",
            split_counts={
                name: len(frame)
                for name, frame in split_result.safe_manifests.items()
            },
        )

        mask_config = config["masks"]
        resolver = MaskResolver(
            mask_config["root"],
            mask_config["allowed_extensions"],
            mask_config["mapping_mode"],
        )
        dataset_root = Path(config["data"]["dataset_root"]).expanduser().resolve()
        mask_root = Path(mask_config["root"]).expanduser().resolve()
        mask_records: list[dict[str, Any]] = []
        mapping_rules: Counter[str] = Counter()
        resolved_by_id: dict[str, str] = {}
        for row_number, row in enumerate(metadata.itertuples(index=False), start=1):
            resolved = resolver.resolve(row.img_filename, row.sample_id)
            details = inspect_mask(
                dataset_root / row.img_filename,
                resolved.path,
                threshold_normalized=float(mask_config["threshold_normalized"]),
                foreground_is_high=bool(mask_config["foreground_is_high"]),
                require_same_dimensions=bool(mask_config["require_same_dimensions"]),
                minimum_foreground_fraction=float(
                    mask_config["minimum_foreground_fraction"]
                ),
                maximum_foreground_fraction=float(
                    mask_config["maximum_foreground_fraction"]
                ),
            )
            mapping_rules[resolved.mapping_rule] += 1
            resolved_by_id[str(row.sample_id)] = resolved.relative_path
            mask_records.append(
                {
                    "sample_id": str(row.sample_id),
                    "metadata_index": int(row.metadata_index),
                    "img_filename": row.img_filename,
                    "mask_relative_path": resolved.relative_path,
                    "mapping_rule": resolved.mapping_rule,
                    **details,
                }
            )
            if row_number % 1000 == 0:
                logger.log(
                    "mask_validation_progress",
                    validated=row_number,
                    total=int(len(metadata)),
                )
        if len(resolved_by_id) != len(metadata):
            raise DataValidationError("Mask resolution did not cover all metadata rows")
        logger.log(
            "masks_validated",
            resolved_count=len(mask_records),
            indexed_mask_files=resolver.total_mask_files,
            mapping_rule_counts=dict(mapping_rules),
        )

        for split_name, frame in split_result.safe_manifests.items():
            frame = frame.copy()
            frame["mask_relative_path"] = frame["sample_id"].map(resolved_by_id)
            if frame["mask_relative_path"].isna().any():
                raise DataValidationError(
                    f"Resolved mask path missing from {split_name} manifest"
                )
            if list(frame.columns) != SAFE_COLUMNS:
                raise DataValidationError(
                    f"Unsafe or unexpected columns in {split_name}: "
                    f"{list(frame.columns)}"
                )
            split_result.safe_manifests[split_name] = frame
            _write_csv(staging / "splits" / f"waterbirds95_{split_name}.csv", frame)

        _write_csv(
            staging / "private_analysis" / "protected_group_labels.csv",
            split_result.protected_labels,
        )
        (staging / "private_analysis" / "README.txt").write_text(
            "ANALYSIS-ONLY PROTECTED LABELS\n"
            "Do not pass this file to ordinary, uLA, pseudo-group, or SETV "
            "selectors. It exists for Oracle and offline diagnostics only.\n",
            encoding="utf-8",
        )
        _write_csv(staging / "masks" / "vlm_mask_manifest.csv", pd.DataFrame(mask_records))

        split_summary = split_result.summary
        split_summary["metadata_sha256"] = sha256_file(
            dataset_root / config["data"]["metadata_csv"]
        )
        split_summary["safe_manifest_columns"] = SAFE_COLUMNS
        split_summary["protected_columns_location"] = (
            "private_analysis/protected_group_labels.csv"
        )
        write_json(staging / "splits" / "split_summary.json", split_summary)

        mask_frame = pd.DataFrame(mask_records)
        mask_summary = {
            "source": "VLM-generated segmentation masks",
            "authoritative_for_pilot": True,
            "original_cub_masks_required": False,
            "mask_root": str(mask_root),
            "mapping_mode": mask_config["mapping_mode"],
            "mapping_rule_counts": dict(mapping_rules),
            "indexed_mask_file_count": resolver.total_mask_files,
            "mapped_sample_count": len(mask_records),
            "threshold_normalized": float(mask_config["threshold_normalized"]),
            "foreground_is_high": bool(mask_config["foreground_is_high"]),
            "require_same_dimensions": bool(mask_config["require_same_dimensions"]),
            "source_binary_count": int(mask_frame["source_mask_binary"].sum()),
            "source_nonbinary_count": int((~mask_frame["source_mask_binary"]).sum()),
            "foreground_fraction": {
                "minimum": float(mask_frame["foreground_fraction"].min()),
                "median": float(mask_frame["foreground_fraction"].median()),
                "maximum": float(mask_frame["foreground_fraction"].max()),
            },
        }
        write_json(staging / "masks" / "mask_summary.json", mask_summary)

        audit_config = config["audit"]
        audit_samples = select_visual_audit_samples(
            split_result.safe_manifests,
            int(audit_config["visual_samples_per_split"]),
            int(audit_config["visual_seed"]),
        )
        _write_csv(staging / "mask_audit" / "visual_audit_samples.csv", audit_samples)
        contact_sheets = render_contact_sheets(
            audit_samples,
            dataset_root=dataset_root,
            mask_root=mask_root,
            output_dir=staging / "mask_audit",
            threshold_normalized=float(mask_config["threshold_normalized"]),
            foreground_is_high=bool(mask_config["foreground_is_high"]),
            columns=int(audit_config["contact_sheet_columns"]),
            thumbnail_size=int(audit_config["thumbnail_size"]),
        )
        write_json(
            staging / "mask_audit" / "visual_review_pending.json",
            {
                "status": "pending_human_review",
                "required_checks": [
                    "bird foreground is covered accurately",
                    "most background is excluded from the foreground mask",
                    "image and mask are spatially aligned",
                    "no split shows a systematic mapping or polarity error",
                ],
                "sample_count_per_split": int(
                    audit_config["visual_samples_per_split"]
                ),
                "contact_sheets": contact_sheets,
            },
        )
        logger.log("visual_audit_generated", contact_sheets=contact_sheets)

        snapshot = {
            key: value for key, value in config.items() if key != "_config_path"
        }
        _write_yaml(staging / "config" / "resolved_phase0.yaml", snapshot)
        provenance = runtime_provenance(_source_root(), destination)
        provenance["source_documents"] = {
            name: sha256_file(_source_root() / name)
            for name in (
                "SETV_Waterbirds95_Implementation_Plan_v2.md",
                "TIGRIS_RESEARCH_COMPUTE_HANDOFF.md",
            )
            if (_source_root() / name).is_file()
        }
        write_json(staging / "provenance" / "runtime.json", provenance)
        receipt = {
            "schema_version": 1,
            "status": "automated_checks_passed_visual_review_pending",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "project_name": config["project_name"],
            "config_sha256": sha256_json(snapshot),
            "split_seed": int(config["split"]["seed"]),
            "dataset_root": str(dataset_root),
            "mask_root": str(mask_root),
            "metadata_rows": int(len(metadata)),
            "visual_review_receipt_required": APPROVAL_RECEIPT,
        }
        write_json(staging / "phase0_receipt.json", receipt)
        logger.log("automated_checks_complete", status=receipt["status"])

        artifact_manifest = _manifest_all_base_files(staging)
        write_json(staging / BASE_ARTIFACT_MANIFEST, artifact_manifest)
        os.rename(staging, destination)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def approve_visual_audit(
    phase0_dir: str | Path,
    *,
    reviewer: str,
    confirmation: bool,
) -> Path:
    root = Path(phase0_dir).expanduser().resolve()
    if not confirmation:
        raise DataValidationError("Explicit --confirm is required")
    if not reviewer.strip():
        raise DataValidationError("Reviewer name must be nonempty")
    verify_base_artifacts(root)
    pending_path = root / "mask_audit" / "visual_review_pending.json"
    with pending_path.open("r", encoding="utf-8") as handle:
        pending = json.load(handle)
    contact_sheets = []
    for item in pending["contact_sheets"]:
        path = root / "mask_audit" / item["path"]
        contact_sheets.append(
            {
                "split_name": item["split_name"],
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "sample_count": item["sample_count"],
            }
        )
    receipt = {
        "schema_version": 1,
        "status": "approved",
        "reviewer": reviewer.strip(),
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "attestation": {
            "bird_coverage_accurate": True,
            "most_background_excluded": True,
            "spatial_alignment_accurate": True,
            "no_systematic_split_mapping_or_polarity_error": True,
        },
        "base_artifact_manifest_sha256": sha256_file(root / BASE_ARTIFACT_MANIFEST),
        "contact_sheets": contact_sheets,
    }
    destination = root / APPROVAL_RECEIPT
    if destination.exists():
        raise ArtifactExistsError(
            f"Visual audit has already been approved: {destination}"
        )
    write_json(destination, receipt)
    return destination


def verify_base_artifacts(phase0_dir: str | Path) -> dict[str, Any]:
    root = Path(phase0_dir).expanduser().resolve()
    manifest_path = root / BASE_ARTIFACT_MANIFEST
    if not manifest_path.is_file():
        raise DataValidationError(f"Artifact manifest is missing: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise DataValidationError("Artifact manifest has invalid file records")
    if sha256_json(files) != manifest.get("manifest_digest"):
        raise DataValidationError("Artifact manifest digest is invalid")
    for relative, expected in files.items():
        path = root / relative
        if not path.is_file():
            raise DataValidationError(f"Phase 0 artifact is missing: {relative}")
        if path.stat().st_size != int(expected["size_bytes"]):
            raise DataValidationError(f"Phase 0 artifact size changed: {relative}")
        if sha256_file(path) != expected["sha256"]:
            raise DataValidationError(f"Phase 0 artifact hash changed: {relative}")
    return manifest


def verify_phase0(phase0_dir: str | Path, *, require_approval: bool = True) -> dict[str, Any]:
    root = Path(phase0_dir).expanduser().resolve()
    manifest = verify_base_artifacts(root)
    approval_path = root / APPROVAL_RECEIPT
    approval: dict[str, Any] | None = None
    if require_approval:
        if not approval_path.is_file():
            raise DataValidationError(
                "Automated Phase 0 checks passed, but human VLM-mask visual "
                f"approval is missing: {approval_path}"
            )
        with approval_path.open("r", encoding="utf-8") as handle:
            approval = json.load(handle)
        if approval.get("status") != "approved":
            raise DataValidationError("Visual audit receipt is not approved")
        if approval.get("base_artifact_manifest_sha256") != sha256_file(
            root / BASE_ARTIFACT_MANIFEST
        ):
            raise DataValidationError(
                "Visual audit approval does not match the base artifact manifest"
            )
        for sheet in approval.get("contact_sheets", []):
            path = root / sheet["path"]
            if not path.is_file() or sha256_file(path) != sheet["sha256"]:
                raise DataValidationError(
                    f"Approved contact sheet is missing or changed: {sheet['path']}"
                )
    return {
        "status": "phase0_complete" if approval is not None else "automated_checks_passed",
        "artifact_count": len(manifest["files"]),
        "visual_review": approval,
    }
