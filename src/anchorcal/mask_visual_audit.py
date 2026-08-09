"""Deterministic, hash-bound visual inspection artifacts for the WB100 mask bank."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from .data import image_path
from .errors import PreflightError
from .io import atomic_write_bytes, atomic_write_json, hash_object, sha256_file
from .vlm_masks import (
    VLM_SELECTOR_REQUIRED_OFFICIAL_SPLITS,
    decode_vlm_mask,
)


MASK_VISUAL_AUDIT_SCHEMA = "anchorcal-mask-visual-audit-v1"
MASK_VISUAL_AUDIT_DIRECTORY = Path("preflight/mask_visual_audit")
AREA_STRATA = ("low", "middle", "high")
REPRESENTATIVES_PER_STRATUM = 3
SAMPLES_PER_PAGE = 6
PANEL_SIZE = (176, 176)
PANEL_LABELS = (
    "Original RGB",
    "Bird red / background blue",
    "Mask: white bird",
    "Bird kept / background green",
    "Background kept / bird green",
)
FOREGROUND_OVERLAY_RGB = (255, 32, 32)
BACKGROUND_OVERLAY_RGB = (32, 96, 255)
GREEN_SCREEN_RGB = (0, 255, 0)
OVERLAY_ALPHA = 0.45


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _contained_path(root: Path, relative_value: Any, description: str) -> Path:
    if not isinstance(relative_value, str):
        raise PreflightError(f"{description} path is missing or malformed")
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise PreflightError(f"{description} path escapes its root")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise PreflightError(f"{description} path escapes its root") from error
    return resolved


def _selection_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    mask_relative_path = record.get("mask_relative_path", record.get("relative_path"))
    mask_sha256 = record.get("mask_sha256", record.get("sha256"))
    mask_size_bytes = record.get("mask_size_bytes", record.get("size_bytes"))
    return {
        "img_id": int(record["img_id"]),
        "img_filename": str(record["img_filename"]),
        "official_split": int(record["official_split"]),
        "y": int(record["y"]),
        "foreground_fraction": float(record["foreground_fraction"]),
        "area_stratum": str(record["area_stratum"]),
        "stratum_representative": int(record["stratum_representative"]),
        "image_relative_path": str(record["image_relative_path"]),
        "image_sha256": str(record["image_sha256"]),
        "image_size_bytes": int(record["image_size_bytes"]),
        "mask_relative_path": str(mask_relative_path),
        "mask_sha256": str(mask_sha256),
        "mask_size_bytes": int(mask_size_bytes),
    }


def select_mask_visual_audit_samples(
    frame: pd.DataFrame, mask_manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Choose low/middle/high-area examples across split, class, and context.

    Selection is independent of DataFrame order.  Each of the two aligned
    official-split-0 ``(y, place)`` cells is sorted by foreground fraction and
    then ``img_id``, divided into three equal-count area strata, and represented
    by three evenly ranked deterministic examples within each stratum.
    ``place`` controls selection but is deliberately removed from every returned
    record so it cannot enter selector-readable visual-audit provenance.
    """

    entries_value = mask_manifest.get("entries")
    if not isinstance(entries_value, list):
        raise PreflightError("VLM mask manifest entries are missing")
    entries: dict[int, Mapping[str, Any]] = {}
    for entry in entries_value:
        if not isinstance(entry, Mapping):
            raise PreflightError("VLM mask manifest entry is malformed")
        img_id = int(entry.get("img_id", -1))
        if img_id < 0 or img_id in entries:
            raise PreflightError("VLM mask manifest has duplicate/invalid img_id")
        entries[img_id] = entry

    selected: list[dict[str, Any]] = []
    for split in VLM_SELECTOR_REQUIRED_OFFICIAL_SPLITS:
        split_frame = frame.loc[frame["split"].astype(int) == int(split)]
        observed_cells = sorted(
            {
                (int(row.y), int(row.place))
                for row in split_frame.itertuples(index=False)
            }
        )
        if not observed_cells:
            raise PreflightError(
                f"mask visual audit has no context cells for required split {split}"
            )
        if split == 0 and observed_cells != [(0, 0), (1, 1)]:
            raise PreflightError(
                "public mask visual audit requires exactly the two aligned "
                "Waterbirds100 split-0 class/context cells"
            )
        for label, place in observed_cells:
            cell = frame.loc[
                (frame["split"].astype(int) == int(split))
                & (frame["y"].astype(int) == label)
                & (frame["place"].astype(int) == place)
            ]
            records: list[dict[str, Any]] = []
            for row in cell.itertuples(index=False):
                img_id = int(row.img_id)
                entry = entries.get(img_id)
                if entry is None or entry.get("img_filename") != str(row.img_filename):
                    raise PreflightError(
                        "visual-audit row is absent from the frozen VLM mapping: "
                        f"img_id={img_id}"
                    )
                records.append(
                    {
                        **dict(entry),
                        "y": label,
                    }
                )
            records.sort(
                key=lambda item: (
                    float(item["foreground_fraction"]),
                    int(item["img_id"]),
                )
            )
            minimum_cell_size = len(AREA_STRATA) * REPRESENTATIVES_PER_STRATUM
            if len(records) < minimum_cell_size:
                raise PreflightError(
                    "mask visual audit requires at least nine examples in every "
                    "observed required split/class/context cell; "
                    f"split={split}, y={label}, place={place}, "
                    f"count={len(records)}"
                )
            partitions = np.array_split(np.arange(len(records)), len(AREA_STRATA))
            for stratum, indices in zip(AREA_STRATA, partitions, strict=True):
                integer_indices = [int(value) for value in indices.tolist()]
                rank_ranges = np.array_split(
                    np.asarray(integer_indices, dtype=np.int64),
                    REPRESENTATIVES_PER_STRATUM,
                )
                if any(len(rank_range) == 0 for rank_range in rank_ranges):
                    raise PreflightError(
                        "mask visual-audit area stratum has fewer than three "
                        "deterministic representatives"
                    )
                for representative_number, rank_range in enumerate(
                    rank_ranges, start=1
                ):
                    representative_index = int(rank_range[len(rank_range) // 2])
                    representative = records[representative_index]
                    selected.append(
                        {
                            **representative,
                            "area_stratum": stratum,
                            "stratum_representative": representative_number,
                        }
                    )
    return selected


def _fit_panel(image: Image.Image, *, categorical: bool = False) -> Image.Image:
    resampling = Image.Resampling.NEAREST if categorical else Image.Resampling.BICUBIC
    contained = ImageOps.contain(image.convert("RGB"), PANEL_SIZE, method=resampling)
    panel = Image.new("RGB", PANEL_SIZE, (28, 28, 28))
    offset = (
        (PANEL_SIZE[0] - contained.width) // 2,
        (PANEL_SIZE[1] - contained.height) // 2,
    )
    panel.paste(contained, offset)
    return panel


def _interpretation_panels(image: Image.Image, binary: np.ndarray) -> tuple[Image.Image, ...]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if binary.shape != rgb.shape[:2]:
        raise PreflightError(
            f"visual-audit image/mask geometry differs: {rgb.shape[:2]} vs {binary.shape}"
        )
    foreground = np.asarray(binary, dtype=bool)
    tint = np.empty_like(rgb)
    tint[foreground] = np.asarray(FOREGROUND_OVERLAY_RGB, dtype=np.uint8)
    tint[~foreground] = np.asarray(BACKGROUND_OVERLAY_RGB, dtype=np.uint8)
    overlay = np.rint(
        (1.0 - OVERLAY_ALPHA) * rgb.astype(np.float32)
        + OVERLAY_ALPHA * tint.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    mask_rgb = np.zeros_like(rgb)
    mask_rgb[foreground] = 255
    bird_only = np.empty_like(rgb)
    bird_only[...] = np.asarray(GREEN_SCREEN_RGB, dtype=np.uint8)
    bird_only[foreground] = rgb[foreground]
    background_only = np.empty_like(rgb)
    background_only[...] = np.asarray(GREEN_SCREEN_RGB, dtype=np.uint8)
    background_only[~foreground] = rgb[~foreground]

    return (
        _fit_panel(image),
        _fit_panel(Image.fromarray(overlay, mode="RGB")),
        _fit_panel(Image.fromarray(mask_rgb, mode="RGB"), categorical=True),
        _fit_panel(Image.fromarray(bird_only, mode="RGB")),
        _fit_panel(Image.fromarray(background_only, mode="RGB")),
    )


def _render_page(
    page_samples: list[dict[str, Any]], panels: Mapping[int, tuple[Image.Image, ...]]
) -> Image.Image:
    margin = 12
    column_gap = 8
    title_height = 28
    metadata_height = 20
    label_height = 20
    row_gap = 12
    width = (
        margin * 2
        + len(PANEL_LABELS) * PANEL_SIZE[0]
        + (len(PANEL_LABELS) - 1) * column_gap
    )
    row_height = metadata_height + label_height + PANEL_SIZE[1] + row_gap
    height = margin * 2 + title_height + len(page_samples) * row_height
    page = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(page)
    draw.text(
        (margin, margin),
        "AnchorCal WB100 split-0 VLM mask audit (red=bird, blue=background)",
        fill=(0, 0, 0),
    )
    for row_index, sample in enumerate(page_samples):
        row_top = margin + title_height + row_index * row_height
        metadata = (
            f"img_id={int(sample['img_id'])}  split={int(sample['official_split'])}  "
            f"y={int(sample['y'])}  area={float(sample['foreground_fraction']):.4f}  "
            f"range={sample['area_stratum']}  "
            f"representative={int(sample['stratum_representative'])}/3"
        )
        draw.text((margin, row_top), metadata, fill=(0, 0, 0))
        for column, (label, panel) in enumerate(
            zip(PANEL_LABELS, panels[int(sample["img_id"])], strict=True)
        ):
            left = margin + column * (PANEL_SIZE[0] + column_gap)
            draw.text((left, row_top + metadata_height), label, fill=(0, 0, 0))
            page.paste(panel, (left, row_top + metadata_height + label_height))
    return page


def _page_bytes(page: Image.Image) -> bytes:
    buffer = io.BytesIO()
    page.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def render_mask_visual_audit(
    config: Mapping[str, Any],
    frame: pd.DataFrame,
    mask_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the mandatory nonblocking WB100 mask contact sheets and receipt."""

    output_root = Path(str(config["paths"]["output_root"])).resolve()
    gallery_root = output_root / MASK_VISUAL_AUDIT_DIRECTORY
    gallery_root.mkdir(parents=True, exist_ok=True)
    selected = select_mask_visual_audit_samples(frame, mask_manifest)
    observed_context_cells = sum(
        len(
            {
                (int(row.y), int(row.place))
                for row in frame.loc[
                    frame["split"].astype(int) == int(split)
                ].itertuples(index=False)
            }
        )
        for split in VLM_SELECTOR_REQUIRED_OFFICIAL_SPLITS
    )
    expected_count = (
        observed_context_cells
        * len(AREA_STRATA)
        * REPRESENTATIVES_PER_STRATUM
    )
    if len(selected) != expected_count:
        raise PreflightError(
            f"mask visual-audit selection count {len(selected)} != {expected_count}"
        )

    image_root = Path(str(config["paths"]["waterbirds_root"])).resolve()
    mask_root = Path(str(config["paths"]["vlm_mask_root"])).resolve()
    if str(mask_manifest.get("root")) != str(mask_root):
        raise PreflightError("mask visual audit received a different VLM root")
    panels: dict[int, tuple[Image.Image, ...]] = {}
    for sample in selected:
        image_file = image_path(image_root, str(sample["img_filename"]))
        mask_file = _contained_path(
            mask_root, sample.get("relative_path"), "visual-audit VLM mask"
        )
        if not image_file.is_file() or not mask_file.is_file():
            raise PreflightError(
                f"visual-audit source is missing for img_id={sample['img_id']}"
            )
        try:
            image_relative_path = image_file.relative_to(image_root).as_posix()
        except ValueError as error:
            raise PreflightError(
                f"visual-audit image escapes its root for img_id={sample['img_id']}"
            ) from error
        if (
            image_relative_path != sample.get("image_relative_path")
            or image_file.stat().st_size != sample.get("image_size_bytes")
            or sample.get("image_sha256") != sha256_file(image_file)
        ):
            raise PreflightError(
                f"visual-audit Waterbirds source changed for img_id={sample['img_id']}"
            )
        if sample.get("sha256") != sha256_file(mask_file):
            raise PreflightError(
                f"visual-audit VLM source changed for img_id={sample['img_id']}"
            )
        with Image.open(image_file) as opened:
            image = opened.convert("RGB")
        decoded = decode_vlm_mask(
            mask_file,
            minimum_foreground_fraction=float(
                config["masks"]["minimum_foreground_fraction"]
            ),
            maximum_foreground_fraction=float(
                config["masks"]["maximum_foreground_fraction"]
            ),
        )
        if image.size != (decoded.binary.shape[1], decoded.binary.shape[0]):
            raise PreflightError(
                f"visual-audit image/mask dimensions differ for img_id={sample['img_id']}"
            )
        panels[int(sample["img_id"])] = _interpretation_panels(
            image, decoded.binary
        )

    pages: list[dict[str, Any]] = []
    for start in range(0, len(selected), SAMPLES_PER_PAGE):
        page_samples = selected[start : start + SAMPLES_PER_PAGE]
        page_number = len(pages) + 1
        page_path = gallery_root / f"contact_sheet_{page_number:02d}.png"
        atomic_write_bytes(page_path, _page_bytes(_render_page(page_samples, panels)))
        pages.append(
            {
                "relative_path": page_path.relative_to(output_root).as_posix(),
                "sha256": sha256_file(page_path),
                "size_bytes": page_path.stat().st_size,
                "sample_ids": [int(sample["img_id"]) for sample in page_samples],
            }
        )

    expected_files = {
        gallery_root / "manifest.json",
        *(output_root / page["relative_path"] for page in pages),
    }
    unexpected = sorted(
        path.name
        for path in gallery_root.iterdir()
        if path.is_file() and path not in expected_files
    )
    if unexpected:
        raise PreflightError(
            f"unexpected stale files in mask visual-audit directory: {unexpected}"
        )

    settings = {
        "selection": (
            "within_each_split0_y_place_cell_sort_by_foreground_fraction_then_"
            "img_id_equal_count_area_terciles_three_equal_rank_ranges_middle_"
            "representatives_place_used_for_selection_only_not_serialized"
        ),
        "selector_required_official_splits": list(
            VLM_SELECTOR_REQUIRED_OFFICIAL_SPLITS
        ),
        "labels": [0, 1],
        "context_stratification": "all_observed_split_y_place_cells",
        "context_values_serialized_per_sample": False,
        "observed_context_cell_count": observed_context_cells,
        "covered_context_cell_count": observed_context_cells,
        "observed_context_cell_count_by_split": {
            str(split): len(
                {
                    (int(row.y), int(row.place))
                    for row in frame.loc[
                        frame["split"].astype(int) == int(split)
                    ].itertuples(index=False)
                }
            )
            for split in VLM_SELECTOR_REQUIRED_OFFICIAL_SPLITS
        },
        "area_strata": list(AREA_STRATA),
        "samples_per_stratum": REPRESENTATIVES_PER_STRATUM,
        "samples_per_page": SAMPLES_PER_PAGE,
        "panel_size": list(PANEL_SIZE),
        "panel_labels": list(PANEL_LABELS),
        "overlay_foreground_rgb": list(FOREGROUND_OVERLAY_RGB),
        "overlay_background_rgb": list(BACKGROUND_OVERLAY_RGB),
        "overlay_alpha": OVERLAY_ALPHA,
        "green_screen_rgb": list(GREEN_SCREEN_RGB),
        "image_thumbnail_interpolation": "bicubic",
        "mask_thumbnail_interpolation": "nearest",
    }
    sample_records = [_selection_payload(sample) for sample in selected]
    audit_manifest = {
        "schema_version": MASK_VISUAL_AUDIT_SCHEMA,
        "status": "passed",
        "purpose": "early_campaign_mask_interpretation_safety_artifact",
        "human_approval_required": False,
        "campaign_gate": "artifact_generation_and_integrity_only",
        "metadata_sha256": mask_manifest.get("metadata_sha256"),
        "mask_bank_sha256": mask_manifest.get("mask_bank_sha256"),
        "foreground_area_summary": mask_manifest.get("foreground_area_summary"),
        "foreground_area_summary_sha256": hash_object(
            mask_manifest.get("foreground_area_summary")
        ),
        "selection_sha256": hash_object(sample_records),
        "settings": settings,
        "sample_count": len(sample_records),
        "sample_ids": [record["img_id"] for record in sample_records],
        "samples": sample_records,
        "page_count": len(pages),
        "pages": pages,
    }
    manifest_path = gallery_root / "manifest.json"
    atomic_write_json(manifest_path, audit_manifest)
    return {
        "schema_version": MASK_VISUAL_AUDIT_SCHEMA,
        "status": "passed",
        "human_approval_required": False,
        "campaign_gate": "artifact_generation_and_integrity_only",
        "manifest_relative_path": manifest_path.relative_to(output_root).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "selection_sha256": audit_manifest["selection_sha256"],
        "foreground_area_summary_sha256": audit_manifest[
            "foreground_area_summary_sha256"
        ],
        "settings": settings,
        "sample_count": len(sample_records),
        "sample_ids": audit_manifest["sample_ids"],
        "page_count": len(pages),
        "pages": pages,
    }


def verify_mask_visual_audit(
    output_root: str | Path,
    receipt: Any,
    *,
    expected_mask_bank_sha256: str,
    expected_metadata_sha256: str,
) -> dict[str, Any]:
    """Verify the receipt, visual manifest, and every rendered contact sheet."""

    root = Path(output_root).resolve()
    if not isinstance(receipt, Mapping):
        raise PreflightError("mask visual-audit receipt is missing")
    manifest_path = _contained_path(
        root, receipt.get("manifest_relative_path"), "mask visual-audit manifest"
    )
    expected_directory = (root / MASK_VISUAL_AUDIT_DIRECTORY).resolve()
    if manifest_path.parent != expected_directory:
        raise PreflightError("mask visual-audit manifest is outside its locked directory")
    if (
        not manifest_path.is_file()
        or not _is_sha256(receipt.get("manifest_sha256"))
        or sha256_file(manifest_path) != receipt.get("manifest_sha256")
    ):
        raise PreflightError("mask visual-audit manifest failed hash verification")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError("mask visual-audit manifest is invalid") from error
    if not isinstance(manifest, dict):
        raise PreflightError("mask visual-audit manifest must be a JSON mapping")
    shared_fields = (
        "schema_version",
        "status",
        "human_approval_required",
        "campaign_gate",
        "selection_sha256",
        "foreground_area_summary_sha256",
        "settings",
        "sample_count",
        "sample_ids",
        "page_count",
        "pages",
    )
    if (
        manifest.get("schema_version") != MASK_VISUAL_AUDIT_SCHEMA
        or manifest.get("status") != "passed"
        or manifest.get("human_approval_required") is not False
        or manifest.get("campaign_gate")
        != "artifact_generation_and_integrity_only"
        or manifest.get("mask_bank_sha256") != expected_mask_bank_sha256
        or manifest.get("metadata_sha256") != expected_metadata_sha256
        or manifest.get("foreground_area_summary_sha256")
        != hash_object(manifest.get("foreground_area_summary"))
        or any(manifest.get(field) != receipt.get(field) for field in shared_fields)
    ):
        raise PreflightError("mask visual-audit receipt/manifest contract is incompatible")
    samples = manifest.get("samples")
    pages = manifest.get("pages")
    try:
        normalized_samples = (
            [_selection_payload(sample) for sample in samples]
            if isinstance(samples, list)
            else []
        )
        normalized_sample_ids = (
            [int(sample.get("img_id", -1)) for sample in samples]
            if isinstance(samples, list)
            else []
        )
    except (KeyError, TypeError, ValueError) as error:
        raise PreflightError("mask visual-audit selection is malformed") from error
    if (
        not isinstance(samples, list)
        or len(samples) <= 0
        or manifest.get("sample_count") != len(samples)
        or not isinstance(pages, list)
        or len(pages) <= 0
        or manifest.get("page_count") != len(pages)
        or manifest.get("sample_ids") != normalized_sample_ids
        or len(set(normalized_sample_ids)) != len(samples)
        or manifest.get("selection_sha256") != hash_object(normalized_samples)
    ):
        raise PreflightError("mask visual-audit selection is malformed")
    page_sample_ids: list[int] = []
    for page in pages:
        if not isinstance(page, Mapping):
            raise PreflightError("mask visual-audit page record is malformed")
        page_path = _contained_path(
            root, page.get("relative_path"), "mask visual-audit page"
        )
        if page_path.parent != expected_directory:
            raise PreflightError("mask visual-audit page is outside its locked directory")
        if (
            not page_path.is_file()
            or int(page.get("size_bytes", -1)) != page_path.stat().st_size
            or not _is_sha256(page.get("sha256"))
            or sha256_file(page_path) != page.get("sha256")
            or not isinstance(page.get("sample_ids"), list)
        ):
            raise PreflightError("mask visual-audit page failed provenance verification")
        page_sample_ids.extend(int(value) for value in page["sample_ids"])
    if page_sample_ids != manifest["sample_ids"]:
        raise PreflightError("mask visual-audit page/sample ordering is incompatible")
    return manifest
