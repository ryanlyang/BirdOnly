"""Compact synthetic visual-audit receipts for non-preflight unit fixtures."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from PIL import Image

from anchorcal.io import atomic_write_bytes, atomic_write_json, hash_object, sha256_file
from anchorcal.mask_visual_audit import MASK_VISUAL_AUDIT_SCHEMA


def attach_synthetic_visual_audit(
    output_root: Path, mask_manifest: dict[str, Any]
) -> dict[str, Any]:
    """Create a tiny integrity-valid receipt without claiming production sampling."""

    gallery = output_root / "preflight" / "mask_visual_audit"
    gallery.mkdir(parents=True, exist_ok=True)
    if any(
        int(entry.get("official_split", -1)) != 0
        or "metadata_index" in entry
        for entry in mask_manifest["entries"]
    ):
        raise AssertionError(
            "synthetic selector-visible audit requires split-0 rows without "
            "metadata_index"
        )
    samples = [
        {
            "img_id": int(entry["img_id"]),
            "img_filename": str(entry["img_filename"]),
            "official_split": int(entry["official_split"]),
            "y": int(index % 2),
            "foreground_fraction": float(entry["foreground_fraction"]),
            "area_stratum": "synthetic_fixture",
            "stratum_representative": 1,
            "image_relative_path": str(entry["image_relative_path"]),
            "image_sha256": str(entry["image_sha256"]),
            "image_size_bytes": int(entry["image_size_bytes"]),
            "mask_relative_path": str(entry["relative_path"]),
            "mask_sha256": str(entry["sha256"]),
            "mask_size_bytes": int(entry["size_bytes"]),
        }
        for index, entry in enumerate(mask_manifest["entries"])
    ]
    image = Image.new("RGB", (32, 24), (17, 73, 131))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    page_path = gallery / "contact_sheet_01.png"
    atomic_write_bytes(page_path, buffer.getvalue())
    pages = [
        {
            "relative_path": page_path.relative_to(output_root).as_posix(),
            "sha256": sha256_file(page_path),
            "size_bytes": page_path.stat().st_size,
            "sample_ids": [sample["img_id"] for sample in samples],
        }
    ]
    settings = {
        "selection": "synthetic_unit_fixture_not_production_preflight",
        "context_values_serialized_per_sample": False,
    }
    payload = {
        "schema_version": MASK_VISUAL_AUDIT_SCHEMA,
        "status": "passed",
        "purpose": "synthetic_unit_fixture",
        "human_approval_required": False,
        "campaign_gate": "artifact_generation_and_integrity_only",
        "metadata_sha256": mask_manifest["metadata_sha256"],
        "mask_bank_sha256": mask_manifest["mask_bank_sha256"],
        "foreground_area_summary": mask_manifest["foreground_area_summary"],
        "foreground_area_summary_sha256": hash_object(
            mask_manifest["foreground_area_summary"]
        ),
        "selection_sha256": hash_object(samples),
        "settings": settings,
        "sample_count": len(samples),
        "sample_ids": [sample["img_id"] for sample in samples],
        "samples": samples,
        "page_count": len(pages),
        "pages": pages,
    }
    manifest_path = gallery / "manifest.json"
    atomic_write_json(manifest_path, payload)
    return {
        "schema_version": MASK_VISUAL_AUDIT_SCHEMA,
        "status": "passed",
        "human_approval_required": False,
        "campaign_gate": "artifact_generation_and_integrity_only",
        "manifest_relative_path": manifest_path.relative_to(output_root).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "selection_sha256": payload["selection_sha256"],
        "foreground_area_summary_sha256": payload[
            "foreground_area_summary_sha256"
        ],
        "settings": settings,
        "sample_count": len(samples),
        "sample_ids": payload["sample_ids"],
        "page_count": len(pages),
        "pages": pages,
    }
