"""Authoritative Waterbirds WeCLIP+ VLM-mask resolution and decoding.

The active pilot uses the audited OpenCLIP-LAION + DINOvIT ``prediction_cmap``
bank.  These PNGs are categorical Pascal/VOC maps, not grayscale confidence
images.  Preflight resolves every required metadata row once and persists the
one-to-one mapping; runtime consumers load only that frozen mapping.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .data import image_path, load_metadata
from .errors import PreflightError
from .io import hash_object, sha256_bytes, sha256_file
from .mask_identity import (
    SELECTOR_MASK_RECEIPT_SCHEMA,
    VLM_MASK_MANIFEST_SCHEMA,
    VLM_PRODUCER,
    vlm_mask_contract_hash,
)


VLM_MASK_FORMAT = "voc_colormap_class_ids"
VLM_FOREGROUND_CLASS_IDS = (1,)
VLM_ALLOWED_CLASS_IDS = (0, 1)
VLM_SELECTOR_REQUIRED_OFFICIAL_SPLITS = (0,)
VLM_ANALYSIS_ONLY_AUDIT_OFFICIAL_SPLITS = (1,)
VLM_PREFLIGHT_AUDITED_OFFICIAL_SPLITS = (
    *VLM_SELECTOR_REQUIRED_OFFICIAL_SPLITS,
    *VLM_ANALYSIS_ONLY_AUDIT_OFFICIAL_SPLITS,
)
VLM_OPTIONAL_OFFICIAL_SPLITS = (2,)
VLM_MAPPING_MODE = "weclip_producer_first_with_explicit_legacy_fallbacks"
VLM_MAPPING_VERSION = "weclip-img-filename-v1"
VLM_DECODER_VERSION = "pascal-voc-rgb-class-id-v1"
VLM_INTERPOLATION = "nearest"
ANALYSIS_ONLY_MASK_AUDIT_SCHEMA = "anchorcal-analysis-only-vlm-mask-audit-v1"
ANALYSIS_ONLY_MASK_AUDIT_RELATIVE_PATH = Path(
    "analysis_only/masks/waterbirds100_oracle_val_mask_audit.json"
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _relative_image_path(image_filename: str) -> Path:
    normalized = str(image_filename).strip().replace("\\", "/")
    if (
        not normalized
        or normalized.startswith("/")
        or re.match(r"^[A-Za-z]:/", normalized)
    ):
        raise PreflightError("img_filename must be a nonempty relative path")
    relative = Path(normalized)
    if ".." in relative.parts:
        raise PreflightError("img_filename must not contain '..'")
    return relative


def producer_vlm_mask_name(image_filename: str) -> str:
    """Return the exact flat image ID written by the Waterbirds producer."""

    relative = _relative_image_path(image_filename)
    without_extension = relative.with_suffix("").as_posix()
    flattened = without_extension.replace("/", "_")
    image_id = re.sub(r"[^A-Za-z0-9_-]+", "_", flattened).strip("_")
    if not image_id:
        raise PreflightError(
            f"could not derive WeCLIP image ID from {image_filename!r}"
        )
    return f"{image_id}.png"


def teacher_map_candidates(
    root: str | Path, image_filename: str
) -> tuple[tuple[str, Path], ...]:
    """Return the producer path first and only documented legacy fallbacks."""

    root = Path(root).expanduser().resolve()
    relative = _relative_image_path(image_filename)
    parent = relative.parent.name
    parent_underscored = parent.replace(".", "_")
    basename = f"{relative.stem}.png"
    return (
        (
            "weclip_producer_flattened_relative_stem",
            root / producer_vlm_mask_name(image_filename),
        ),
        ("legacy_one_parent_flat", root / f"{parent_underscored}_{basename}"),
        ("legacy_underscored_parent_tree", root / parent_underscored / basename),
        ("legacy_literal_parent_tree", root / parent / basename),
        ("legacy_relative_stem_tree", root / relative.with_suffix(".png")),
    )


def resolve_teacher_map(root: str | Path, image_filename: str) -> tuple[Path, str]:
    """Resolve exactly one VLM map and reject missing, ambiguous, or escaping paths."""

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise PreflightError(f"VLM mask root is missing: {root}")
    matches: dict[Path, str] = {}
    for rule, candidate in teacher_map_candidates(root, image_filename):
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise PreflightError(
                f"VLM mask path escapes its root for {image_filename}: {candidate}"
            ) from error
        matches.setdefault(resolved, rule)
    if not matches:
        raise PreflightError(f"no VLM mask for {image_filename}")
    if len(matches) != 1:
        raise PreflightError(
            f"ambiguous VLM masks for {image_filename}: "
            f"{[str(path) for path in sorted(matches)]}"
        )
    path, rule = next(iter(matches.items()))
    return path, rule


@lru_cache(maxsize=1)
def voc_palette() -> np.ndarray:
    """Return the canonical 256-entry Pascal/VOC RGB palette."""

    palette = np.zeros((256, 3), dtype=np.uint8)
    for class_id in range(256):
        value = class_id
        for bit in range(8):
            palette[class_id, 0] |= ((value >> 0) & 1) << (7 - bit)
            palette[class_id, 1] |= ((value >> 1) & 1) << (7 - bit)
            palette[class_id, 2] |= ((value >> 2) & 1) << (7 - bit)
            value >>= 3
    palette.setflags(write=False)
    return palette


@dataclass(frozen=True)
class DecodedVlmMask:
    binary: np.ndarray
    class_ids: tuple[int, ...]
    rgb_colors: tuple[tuple[int, int, int], ...]
    class_pixel_counts: Mapping[int, int]
    foreground_fraction: float


def decode_vlm_mask(
    path: str | Path,
    *,
    minimum_foreground_fraction: float = 0.0,
    maximum_foreground_fraction: float = 1.0,
) -> DecodedVlmMask:
    """Decode a categorical VOC PNG and accept only classes 0 and 1."""

    path = Path(path)
    try:
        with Image.open(path) as opened:
            rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    except OSError as error:
        raise PreflightError(f"could not read VLM mask: {path}") from error
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise PreflightError(f"VLM mask is not RGB-decodable: {path}")
    packed = (
        (rgb[..., 0].astype(np.uint32) << 16)
        | (rgb[..., 1].astype(np.uint32) << 8)
        | rgb[..., 2].astype(np.uint32)
    )
    palette = voc_palette()
    palette_packed = (
        (palette[:, 0].astype(np.uint32) << 16)
        | (palette[:, 1].astype(np.uint32) << 8)
        | palette[:, 2].astype(np.uint32)
    )
    lookup = {int(color): class_id for class_id, color in enumerate(palette_packed)}
    unique_packed = np.unique(packed)
    unknown = [int(color) for color in unique_packed if int(color) not in lookup]
    if unknown:
        colors = [
            ((value >> 16) & 255, (value >> 8) & 255, value & 255)
            for value in unknown[:10]
        ]
        raise PreflightError(f"unexpected non-VOC colors {colors} in {path}")
    observed_ids = tuple(sorted(lookup[int(color)] for color in unique_packed))
    if not set(observed_ids).issubset(VLM_ALLOWED_CLASS_IDS):
        raise PreflightError(
            f"VLM mask contains classes outside {VLM_ALLOWED_CLASS_IDS}: "
            f"{observed_ids} in {path}"
        )
    foreground_color = int(palette_packed[VLM_FOREGROUND_CLASS_IDS[0]])
    binary = packed == foreground_color
    fraction = float(binary.mean())
    if not minimum_foreground_fraction < fraction < maximum_foreground_fraction:
        raise PreflightError(
            f"VLM foreground fraction {fraction:.6f} outside "
            f"({minimum_foreground_fraction}, {maximum_foreground_fraction}): {path}"
        )
    colors = tuple(
        tuple(int(channel) for channel in palette[class_id])
        for class_id in observed_ids
    )
    return DecodedVlmMask(
        binary=binary,
        class_ids=observed_ids,
        rgb_colors=colors,
        class_pixel_counts={
            int(class_id): int(np.count_nonzero(packed == palette_packed[class_id]))
            for class_id in observed_ids
        },
        foreground_fraction=fraction,
    )


def vlm_mask_manifest_entry(
    *,
    img_id: int,
    metadata_index: int,
    img_filename: str,
    split: int,
    root: Path,
    path: Path,
    mapping_rule: str,
    decoded: DecodedVlmMask,
) -> dict[str, Any]:
    return {
        "img_id": int(img_id),
        "metadata_index": int(metadata_index),
        "img_filename": str(img_filename),
        "official_split": int(split),
        "producer_mask_name": producer_vlm_mask_name(img_filename),
        "relative_path": path.resolve().relative_to(root.resolve()).as_posix(),
        "mapping_rule": mapping_rule,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "mask_width": int(decoded.binary.shape[1]),
        "mask_height": int(decoded.binary.shape[0]),
        "observed_class_ids": list(decoded.class_ids),
        "observed_rgb_colors": [list(color) for color in decoded.rgb_colors],
        "class_pixel_counts": {
            str(class_id): int(count)
            for class_id, count in decoded.class_pixel_counts.items()
        },
        "foreground_pixels": int(decoded.binary.sum()),
        "foreground_fraction": decoded.foreground_fraction,
        "decoded_binary_sha256": sha256_bytes(
            np.ascontiguousarray(decoded.binary, dtype=np.uint8).tobytes()
        ),
    }


def foreground_area_summary(
    entries: list[Mapping[str, Any]],
    *,
    official_splits: tuple[int, ...] = VLM_SELECTOR_REQUIRED_OFFICIAL_SPLITS,
) -> dict[str, Any]:
    """Summarize fixed reporting-only foreground-area ranges and quantiles."""

    if not entries:
        raise PreflightError("cannot summarize an empty VLM mask bank")

    def summarize(values: np.ndarray) -> dict[str, Any]:
        if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
            raise PreflightError("invalid foreground fractions in VLM mask bank")
        return {
            "count": int(len(values)),
            "empty_count": int(np.count_nonzero(values <= 0.0)),
            "very_small_nonempty_le_0_01_count": int(
                np.count_nonzero((values > 0.0) & (values <= 0.01))
            ),
            "ordinary_gt_0_01_lt_0_95_count": int(
                np.count_nonzero((values > 0.01) & (values < 0.95))
            ),
            "very_large_ge_0_95_nonfull_count": int(
                np.count_nonzero((values >= 0.95) & (values < 1.0))
            ),
            "full_count": int(np.count_nonzero(values >= 1.0)),
            "minimum": float(np.min(values)),
            "mean": float(np.mean(values)),
            "maximum": float(np.max(values)),
            "quantiles": {
                name: float(np.quantile(values, quantile))
                for name, quantile in (
                    ("p00", 0.00),
                    ("p01", 0.01),
                    ("p05", 0.05),
                    ("p25", 0.25),
                    ("p50", 0.50),
                    ("p75", 0.75),
                    ("p95", 0.95),
                    ("p99", 0.99),
                    ("p100", 1.00),
                )
            },
        }

    fractions = np.asarray(
        [float(entry["foreground_fraction"]) for entry in entries],
        dtype=np.float64,
    )
    observed_splits = {int(entry["official_split"]) for entry in entries}
    if observed_splits != set(official_splits):
        raise PreflightError(
            "foreground-area rows differ from their locked official splits: "
            f"expected={list(official_splits)}, observed={sorted(observed_splits)}"
        )
    by_split = {
        str(split): summarize(
            np.asarray(
                [
                    float(entry["foreground_fraction"])
                    for entry in entries
                    if int(entry["official_split"]) == split
                ],
                dtype=np.float64,
            )
        )
        for split in official_splits
    }
    return {
        "schema_version": "anchorcal-vlm-foreground-area-summary-v1",
        "bin_definitions": {
            "empty": "fraction <= 0",
            "very_small_nonempty": "0 < fraction <= 0.01",
            "ordinary": "0.01 < fraction < 0.95",
            "very_large_nonfull": "0.95 <= fraction < 1",
            "full": "fraction >= 1",
        },
        "failure_policy": {
            "empty_or_full": "fatal_exact_decode_gate",
            "very_small_or_very_large": "reporting_only_not_a_failure_gate",
        },
        "overall": summarize(fractions),
        "by_official_split": by_split,
    }


def vlm_mask_bank_hash(
    entries: list[dict[str, Any]],
    *,
    root: str | Path,
    minimum_foreground_fraction: float,
    maximum_foreground_fraction: float,
) -> str:
    """Bind files, mapping, decoding, coverage, and root into one receipt."""

    return hash_object(
        {
            "schema_version": VLM_MASK_MANIFEST_SCHEMA,
            "root": str(Path(root).resolve()),
            "producer": VLM_PRODUCER,
            "mapping_mode": VLM_MAPPING_MODE,
            "mapping_version": VLM_MAPPING_VERSION,
            "decoder_version": VLM_DECODER_VERSION,
            "format": VLM_MASK_FORMAT,
            "foreground_class_ids": list(VLM_FOREGROUND_CLASS_IDS),
            "allowed_class_ids": list(VLM_ALLOWED_CLASS_IDS),
            "interpolation": VLM_INTERPOLATION,
            "minimum_foreground_fraction": minimum_foreground_fraction,
            "maximum_foreground_fraction": maximum_foreground_fraction,
            "selector_required_official_splits": list(
                VLM_SELECTOR_REQUIRED_OFFICIAL_SPLITS
            ),
            "optional_official_splits": list(VLM_OPTIONAL_OFFICIAL_SPLITS),
            "entries": sorted(entries, key=lambda item: int(item["img_id"])),
        }
    )


@dataclass(frozen=True)
class VlmMaskBank:
    root: Path
    entries: Mapping[int, Mapping[str, Any]]
    mask_bank_sha256: str
    minimum_foreground_fraction: float
    maximum_foreground_fraction: float

    def path_for(self, img_id: int, image_filename: str) -> Path:
        record = self.entries.get(int(img_id))
        if record is None or record.get("img_filename") != str(image_filename):
            raise PreflightError(
                f"frozen VLM manifest has no matching row for img_id={img_id}, "
                f"image={image_filename}"
            )
        relative_value = record.get("relative_path")
        if not isinstance(relative_value, str):
            raise PreflightError("frozen VLM manifest contains an invalid path")
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise PreflightError("frozen VLM manifest path escapes its root")
        path = (self.root / relative).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise PreflightError("frozen VLM manifest path escapes its root") from error
        if not path.is_file():
            raise PreflightError(f"frozen VLM mask is missing: {path}")
        return path

    def load(self, img_id: int, image_filename: str) -> np.ndarray:
        path = self.path_for(img_id, image_filename)
        record = self.entries[int(img_id)]
        decoded = decode_vlm_mask(
            path,
            minimum_foreground_fraction=self.minimum_foreground_fraction,
            maximum_foreground_fraction=self.maximum_foreground_fraction,
        )
        decoded_hash = sha256_bytes(
            np.ascontiguousarray(decoded.binary, dtype=np.uint8).tobytes()
        )
        if (
            record.get("decoded_binary_sha256") != decoded_hash
            or record.get("observed_class_ids") != list(decoded.class_ids)
            or record.get("mask_width") != int(decoded.binary.shape[1])
            or record.get("mask_height") != int(decoded.binary.shape[0])
        ):
            raise PreflightError(
                f"decoded VLM mask no longer matches its frozen manifest: "
                f"img_id={img_id}"
            )
        return decoded.binary


def _validate_frozen_entries(
    records: Any,
    *,
    root: Path,
    image_root: Path,
    allowed_splits: tuple[int, ...],
    minimum: float,
    maximum: float,
    public_selector_bank: bool,
) -> tuple[dict[int, Mapping[str, Any]], dict[int, int]]:
    """Validate frozen rows and their source bytes for one visibility domain."""

    if not isinstance(records, list) or not records:
        raise PreflightError("frozen VLM mask manifest contains no entries")
    entries: dict[int, Mapping[str, Any]] = {}
    relative_paths: set[str] = set()
    metadata_indices: set[int] = set()
    split_counts = {split: 0 for split in allowed_splits}
    allowed_rules = {rule for rule, _ in teacher_map_candidates(root, "x/y.jpg")}
    for record in records:
        if not isinstance(record, dict):
            raise PreflightError("frozen VLM mask manifest entry is malformed")
        try:
            img_id = int(record.get("img_id", -1))
            split = int(record.get("official_split", -1))
            fraction = float(record.get("foreground_fraction", -1.0))
            mask_width = int(record.get("mask_width", -1))
            mask_height = int(record.get("mask_height", -1))
            image_width = int(record.get("image_width", -1))
            image_height = int(record.get("image_height", -1))
            image_size_bytes = int(record.get("image_size_bytes", -1))
            foreground_pixels = int(record.get("foreground_pixels", -1))
            observed_class_ids = set(record.get("observed_class_ids", []))
            raw_class_pixel_counts = record.get("class_pixel_counts")
            if not isinstance(raw_class_pixel_counts, dict):
                raise TypeError("class_pixel_counts must be a mapping")
            class_pixel_counts = {
                int(class_id): int(count)
                for class_id, count in raw_class_pixel_counts.items()
            }
            metadata_index = (
                int(record.get("metadata_index", -1))
                if not public_selector_bank
                else None
            )
        except (TypeError, ValueError) as error:
            raise PreflightError(
                "frozen VLM mask manifest entry is malformed"
            ) from error
        if img_id < 0 or img_id in entries:
            raise PreflightError("frozen VLM mask manifest has duplicate/invalid img_id")
        if public_selector_bank and "metadata_index" in record:
            raise PreflightError(
                "selector-visible VLM mask entry contains metadata_index"
            )
        if not public_selector_bank and (
            metadata_index is None
            or metadata_index < 0
            or metadata_index in metadata_indices
        ):
            raise PreflightError(
                "analysis-only VLM audit has duplicate/invalid metadata_index"
            )
        image_filename = record.get("img_filename")
        image_relative_path = record.get("image_relative_path")
        relative_path = record.get("relative_path")
        mapping_rule = record.get("mapping_rule")
        if (
            isinstance(image_filename, str)
            and isinstance(relative_path, str)
            and mapping_rule in allowed_rules
        ):
            expected_candidate = dict(
                teacher_map_candidates(root, image_filename)
            )[str(mapping_rule)].resolve()
            recorded_candidate = (root / relative_path).resolve()
        else:
            expected_candidate = None
            recorded_candidate = None
        if (
            not isinstance(image_filename, str)
            or record.get("producer_mask_name")
            != producer_vlm_mask_name(image_filename)
            or not isinstance(image_relative_path, str)
            or not _is_sha256(record.get("image_sha256"))
            or image_size_bytes <= 0
            or not isinstance(relative_path, str)
            or relative_path in relative_paths
            or split not in allowed_splits
            or mapping_rule not in allowed_rules
            or recorded_candidate != expected_candidate
            or not observed_class_ids.issubset(VLM_ALLOWED_CLASS_IDS)
            or set(class_pixel_counts) != observed_class_ids
            or sum(class_pixel_counts.values()) != mask_width * mask_height
            or class_pixel_counts.get(1, 0) != foreground_pixels
            or abs(
                fraction - foreground_pixels / max(mask_width * mask_height, 1)
            )
            > 1.0e-15
            or not minimum < fraction < maximum
            or mask_width <= 0
            or mask_height <= 0
            or image_width != mask_width
            or image_height != mask_height
            or foreground_pixels <= 0
            or not _is_sha256(record.get("decoded_binary_sha256"))
            or not _is_sha256(record.get("sha256"))
        ):
            raise PreflightError("frozen VLM mask manifest entry is incompatible")
        mask_path = recorded_candidate
        assert mask_path is not None
        try:
            mask_path.relative_to(root)
        except ValueError as error:
            raise PreflightError("frozen VLM manifest path escapes its root") from error
        source_image = image_path(image_root, image_filename)
        try:
            source_relative = source_image.relative_to(image_root).as_posix()
        except ValueError as error:
            raise PreflightError("Waterbirds source image escapes its root") from error
        if (
            not mask_path.is_file()
            or mask_path.stat().st_size != record.get("size_bytes")
            or sha256_file(mask_path) != record.get("sha256")
        ):
            raise PreflightError(
                f"VLM source file no longer matches manifest: img_id={img_id}"
            )
        if (
            not source_image.is_file()
            or source_relative != image_relative_path
            or source_image.stat().st_size != image_size_bytes
            or sha256_file(source_image) != record.get("image_sha256")
        ):
            raise PreflightError(
                f"Waterbirds source image no longer matches manifest: img_id={img_id}"
            )
        relative_paths.add(relative_path)
        if metadata_index is not None:
            metadata_indices.add(metadata_index)
        split_counts[split] += 1
        entries[img_id] = record
    return entries, split_counts


def _load_vlm_mask_bank(
    config: Mapping[str, Any], *, require_finalized_report: bool
) -> VlmMaskBank:
    output = Path(str(config["paths"]["output_root"]))
    manifest_path = output / "preflight" / "mask_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError(f"invalid frozen VLM mask manifest: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise PreflightError("frozen VLM mask manifest must be a JSON mapping")
    root = Path(str(config["paths"]["vlm_mask_root"])).resolve()
    if not root.is_dir():
        raise PreflightError(f"frozen VLM mask root is missing: {root}")
    mask_config = config["masks"]
    minimum = float(mask_config["minimum_foreground_fraction"])
    maximum = float(mask_config["maximum_foreground_fraction"])
    contract_hash = vlm_mask_contract_hash(config)
    expected = {
        "schema_version": VLM_MASK_MANIFEST_SCHEMA,
        "status": "passed",
        "dataset_root": str(Path(str(config["paths"]["waterbirds_root"])).resolve()),
        "metadata_path": str(Path(str(config["paths"]["metadata_path"])).resolve()),
        "root": str(root),
        "producer": VLM_PRODUCER,
        "mapping_mode": VLM_MAPPING_MODE,
        "mapping_version": VLM_MAPPING_VERSION,
        "decoder_version": VLM_DECODER_VERSION,
        "format": VLM_MASK_FORMAT,
        "foreground_class_ids": list(VLM_FOREGROUND_CLASS_IDS),
        "allowed_class_ids": list(VLM_ALLOWED_CLASS_IDS),
        "interpolation": VLM_INTERPOLATION,
        "selector_required_official_splits": list(
            VLM_SELECTOR_REQUIRED_OFFICIAL_SPLITS
        ),
        "analysis_only_audit_official_splits": list(
            VLM_ANALYSIS_ONLY_AUDIT_OFFICIAL_SPLITS
        ),
        "optional_official_splits": list(VLM_OPTIONAL_OFFICIAL_SPLITS),
        "minimum_foreground_fraction": minimum,
        "maximum_foreground_fraction": maximum,
        "runtime_resolution": "frozen_manifest_only",
        "mask_contract_sha256": contract_hash,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise PreflightError("frozen VLM mask manifest contract is incompatible")
    debug = bool(config.get("runtime", {}).get("debug", False))
    if (
        not isinstance(manifest.get("resolved_config_sha256"), str)
        or (
            not debug
            and manifest.get("resolved_config_sha256")
            != config["resolved_config_sha256"]
        )
    ):
        raise PreflightError("frozen VLM mask manifest config binding is incompatible")
    records = manifest.get("entries")
    entries, split_counts = _validate_frozen_entries(
        records,
        root=root,
        image_root=Path(str(config["paths"]["waterbirds_root"])).resolve(),
        allowed_splits=VLM_SELECTOR_REQUIRED_OFFICIAL_SPLITS,
        minimum=minimum,
        maximum=maximum,
        public_selector_bank=True,
    )
    coverage = manifest.get("coverage")
    required_mapping_audit = manifest.get("required_mapping_audit")
    if (
        int(manifest.get("required_count", -1)) != len(entries)
        or not isinstance(coverage, dict)
        or any(
            coverage.get(str(split), {}).get("present") != split_counts[split]
            or coverage.get(str(split), {}).get("expected") != split_counts[split]
            for split in VLM_SELECTOR_REQUIRED_OFFICIAL_SPLITS
        )
        or required_mapping_audit
        != {
            "expected": len(entries),
            "resolved_unique": len(entries),
            "missing": 0,
            "ambiguous": 0,
            "reused": 0,
            "producer_name_collisions": 0,
        }
    ):
        raise PreflightError("frozen VLM mask manifest coverage is incompatible")
    if manifest.get("foreground_area_summary") != foreground_area_summary(
        [dict(record) for record in records]
    ):
        raise PreflightError("frozen VLM foreground-area summary is incompatible")
    bank_hash = vlm_mask_bank_hash(
        [dict(record) for record in records],
        root=root,
        minimum_foreground_fraction=minimum,
        maximum_foreground_fraction=maximum,
    )
    if manifest.get("mask_bank_sha256") != bank_hash:
        raise PreflightError("frozen VLM mask-bank hash is invalid")
    visual_receipt = manifest.get("visual_audit")
    if not isinstance(visual_receipt, dict):
        raise PreflightError("mask manifest v3 requires a visual-audit receipt")

    # The compact selector receipt is produced before geometry and therefore
    # forms part of the acyclic preflight-staging identity.  Cross-check every
    # aggregate value here so the bootstrap path is not weaker than the final
    # report-bound path used by all later workers.
    from .candidate_provenance import load_candidate_preflight_binding

    selector_receipt = load_candidate_preflight_binding(config)
    manifest_sha256 = sha256_file(manifest_path)
    if (
        selector_receipt.get("resolved_config_sha256")
        != manifest.get("resolved_config_sha256")
        or selector_receipt.get("metadata_sha256")
        != manifest.get("metadata_sha256")
        or selector_receipt.get("git_commit") != manifest.get("git_commit")
        or selector_receipt.get("mask_source") != VLM_PRODUCER
        or selector_receipt.get("mask_contract_sha256") != contract_hash
        or selector_receipt.get("mask_bank_sha256") != bank_hash
        or selector_receipt.get("mask_manifest_sha256") != manifest_sha256
        or selector_receipt.get("foreground_area_summary_sha256")
        != hash_object(manifest.get("foreground_area_summary"))
        or selector_receipt.get("mask_visual_audit_manifest_sha256")
        != visual_receipt.get("manifest_sha256")
        or selector_receipt.get("mask_visual_audit_selection_sha256")
        != visual_receipt.get("selection_sha256")
    ):
        raise PreflightError(
            "selector-safe mask receipt does not bind the frozen VLM manifest"
        )
    report_path = output / "preflight" / "report.json"
    report: dict[str, Any] | None = None
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PreflightError(f"invalid preflight report: {report_path}") from error
        if not isinstance(report, dict):
            raise PreflightError("preflight report must be a JSON mapping")
        if not require_finalized_report:
            raise PreflightError(
                "preflight geometry bootstrap cannot replace a finalized report"
            )
        if (
            report.get("schema_version") != "anchorcal-preflight-v1"
            or report.get("status") != "passed"
            or report.get("mask_bank_sha256") != bank_hash
            or report.get("mask_manifest_sha256") != manifest_sha256
            or report.get("mask_source") != VLM_PRODUCER
            or report.get("metadata_sha256") != manifest.get("metadata_sha256")
            or report.get("foreground_area_summary")
            != manifest.get("foreground_area_summary")
            or report.get("mask_contract_sha256") != contract_hash
            or report.get("resolved_config_sha256")
            != manifest.get("resolved_config_sha256")
            or (
                not debug
                and report.get("resolved_config_sha256")
                != config["resolved_config_sha256"]
            )
            or report.get("resolved_paths") != config["paths"]
            or report.get("selector_mask_receipt")
            != {
                "schema_version": SELECTOR_MASK_RECEIPT_SCHEMA,
                "sha256": selector_receipt["_receipt_sha256"],
            }
        ):
            raise PreflightError("preflight report does not bind the VLM manifest")
    if report is None:
        if require_finalized_report:
            raise PreflightError(
                "finalized preflight report is required to load the VLM mask bank"
            )
    else:
        report_visual_receipt = report.get("mask_visual_audit")
        if report_visual_receipt is None:
            raise PreflightError(
                "preflight report requires a visual-audit receipt"
            )
        if visual_receipt != report_visual_receipt:
            raise PreflightError("mask visual-audit receipts differ")
    # Local import avoids a module cycle: rendering imports this module's strict
    # VLM decoder, while runtime verification occurs only after both modules
    # have finished importing.
    from .mask_visual_audit import verify_mask_visual_audit

    verify_mask_visual_audit(
        output,
        visual_receipt,
        expected_mask_bank_sha256=bank_hash,
        expected_metadata_sha256=str(manifest.get("metadata_sha256")),
    )
    return VlmMaskBank(
        root=root,
        entries=entries,
        mask_bank_sha256=bank_hash,
        minimum_foreground_fraction=minimum,
        maximum_foreground_fraction=maximum,
    )


def load_vlm_mask_bank(config: Mapping[str, Any]) -> VlmMaskBank:
    """Load a mask bank bound to the completed preflight report."""

    return _load_vlm_mask_bank(config, require_finalized_report=True)


def load_preflight_geometry_mask_bank(config: Mapping[str, Any]) -> VlmMaskBank:
    """Verify the frozen mask bank while the final report awaits geometry.

    ``preflight/report.json`` includes the geometry manifest, so requiring that
    report while constructing geometry is a dependency cycle.  This narrowly
    scoped bootstrap entry point still verifies every source image and mask,
    the frozen bank hash, and the complete visual-audit receipt.  It refuses to
    run after a finalized report exists; all later consumers use the strict
    :func:`load_vlm_mask_bank` entry point above.
    """

    report_path = (
        Path(str(config["paths"]["output_root"])) / "preflight" / "report.json"
    )
    if report_path.exists():
        raise PreflightError(
            "preflight geometry bootstrap cannot replace a finalized report"
        )
    return _load_vlm_mask_bank(config, require_finalized_report=False)


def load_analysis_only_mask_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the protected official split-1 mask audit and all source bytes.

    Selector-visible code must never import or call this function.  It is for
    the hidden reporting stage and completed-campaign verification only.
    """

    output = Path(str(config["paths"]["output_root"])).resolve()
    audit_path = output / ANALYSIS_ONLY_MASK_AUDIT_RELATIVE_PATH
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError(
            f"invalid analysis-only VLM mask audit: {audit_path}"
        ) from error
    if not isinstance(audit, dict):
        raise PreflightError("analysis-only VLM mask audit must be a JSON mapping")
    root = Path(str(config["paths"]["vlm_mask_root"])).resolve()
    image_root = Path(str(config["paths"]["waterbirds_root"])).resolve()
    metadata_path = Path(str(config["paths"]["metadata_path"])).resolve()
    masks = config["masks"]
    minimum = float(masks["minimum_foreground_fraction"])
    maximum = float(masks["maximum_foreground_fraction"])
    expected = {
        "schema_version": ANALYSIS_ONLY_MASK_AUDIT_SCHEMA,
        "status": "passed",
        "namespace": "analysis_only",
        "selector_visible": False,
        "reporting_only": True,
        "official_splits": list(VLM_ANALYSIS_ONLY_AUDIT_OFFICIAL_SPLITS),
        "dataset_root": str(image_root),
        "metadata_path": str(metadata_path),
        "root": str(root),
        "producer": VLM_PRODUCER,
        "mapping_mode": VLM_MAPPING_MODE,
        "mapping_version": VLM_MAPPING_VERSION,
        "decoder_version": VLM_DECODER_VERSION,
        "format": VLM_MASK_FORMAT,
        "foreground_class_ids": list(VLM_FOREGROUND_CLASS_IDS),
        "allowed_class_ids": list(VLM_ALLOWED_CLASS_IDS),
        "interpolation": VLM_INTERPOLATION,
        "minimum_foreground_fraction": minimum,
        "maximum_foreground_fraction": maximum,
        "mask_contract_sha256": vlm_mask_contract_hash(config),
    }
    if any(audit.get(key) != value for key, value in expected.items()):
        raise PreflightError("analysis-only VLM mask audit contract is incompatible")
    debug = bool(config.get("runtime", {}).get("debug", False))
    if (
        not isinstance(audit.get("resolved_config_sha256"), str)
        or (
            not debug
            and audit.get("resolved_config_sha256")
            != config["resolved_config_sha256"]
        )
        or not _is_sha256(audit.get("metadata_sha256"))
        or sha256_file(metadata_path) != audit.get("metadata_sha256")
    ):
        raise PreflightError("analysis-only VLM mask audit binding is incompatible")
    records = audit.get("entries")
    try:
        entries, split_counts = _validate_frozen_entries(
            records,
            root=root,
            image_root=image_root,
            allowed_splits=VLM_ANALYSIS_ONLY_AUDIT_OFFICIAL_SPLITS,
            minimum=minimum,
            maximum=maximum,
            public_selector_bank=False,
        )
    except PreflightError as error:
        raise PreflightError(
            f"analysis-only VLM mask audit entry verification failed: {error}"
        ) from error
    coverage = audit.get("coverage")
    if (
        audit.get("entries_sha256")
        != hash_object([dict(record) for record in records])
        or int(audit.get("row_count", -1)) != len(entries)
        or not isinstance(coverage, dict)
        or any(
            coverage.get(str(split), {}).get("present") != split_counts[split]
            or coverage.get(str(split), {}).get("expected") != split_counts[split]
            for split in VLM_ANALYSIS_ONLY_AUDIT_OFFICIAL_SPLITS
        )
        or audit.get("foreground_area_summary")
        != foreground_area_summary(
            [dict(record) for record in records],
            official_splits=VLM_ANALYSIS_ONLY_AUDIT_OFFICIAL_SPLITS,
        )
    ):
        raise PreflightError("analysis-only VLM mask audit rows are incompatible")
    metadata = load_metadata(metadata_path)
    expected_rows = metadata.loc[
        metadata["split"].isin(VLM_ANALYSIS_ONLY_AUDIT_OFFICIAL_SPLITS)
    ]
    expected_by_id = {
        int(row.img_id): row for row in expected_rows.itertuples(index=False)
    }
    if set(entries) != set(expected_by_id):
        raise PreflightError(
            "analysis-only VLM mask audit does not cover exact official split 1"
        )
    for img_id, entry in entries.items():
        row = expected_by_id[img_id]
        if (
            entry.get("img_filename") != str(row.img_filename)
            or int(entry.get("official_split", -1)) != int(row.split)
            or int(entry.get("metadata_index", -1))
            != int(row.metadata_row_index)
        ):
            raise PreflightError(
                "analysis-only VLM mask audit differs from authoritative metadata"
            )
    return audit
