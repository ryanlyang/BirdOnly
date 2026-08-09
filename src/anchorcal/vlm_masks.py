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

from .errors import PreflightError
from .io import hash_object, sha256_bytes, sha256_file


VLM_MASK_MANIFEST_SCHEMA = "anchorcal-vlm-mask-manifest-v1"
VLM_MASK_FORMAT = "voc_colormap_class_ids"
VLM_FOREGROUND_CLASS_IDS = (1,)
VLM_ALLOWED_CLASS_IDS = (0, 1)
VLM_REQUIRED_OFFICIAL_SPLITS = (0, 1)
VLM_OPTIONAL_OFFICIAL_SPLITS = (2,)
VLM_MAPPING_MODE = "weclip_producer_first_with_explicit_legacy_fallbacks"
VLM_PRODUCER = "openclip_laion_dinovit_weclipplus_prediction_cmap"
VLM_MAPPING_VERSION = "weclip-img-filename-v1"
VLM_DECODER_VERSION = "pascal-voc-rgb-class-id-v1"
VLM_INTERPOLATION = "nearest"


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
            "required_official_splits": list(VLM_REQUIRED_OFFICIAL_SPLITS),
            "optional_official_splits": list(VLM_OPTIONAL_OFFICIAL_SPLITS),
            "entries": sorted(entries, key=lambda item: int(item["img_id"])),
        }
    )


def vlm_mask_contract_hash(config: Mapping[str, Any]) -> str:
    """Hash only identity that must be shared by production and debug runs."""

    paths = config["paths"]
    return hash_object(
        {
            "schema_version": VLM_MASK_MANIFEST_SCHEMA,
            "waterbirds_root": str(Path(str(paths["waterbirds_root"])).resolve()),
            "metadata_path": str(Path(str(paths["metadata_path"])).resolve()),
            "vlm_mask_root": str(Path(str(paths["vlm_mask_root"])).resolve()),
            "masks": dict(config["masks"]),
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


def load_vlm_mask_bank(config: Mapping[str, Any]) -> VlmMaskBank:
    output = Path(str(config["paths"]["output_root"]))
    manifest_path = output / "preflight" / "mask_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError(f"invalid frozen VLM mask manifest: {manifest_path}") from error
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
        "required_official_splits": list(VLM_REQUIRED_OFFICIAL_SPLITS),
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
    if not isinstance(records, list) or not records:
        raise PreflightError("frozen VLM mask manifest contains no entries")
    entries: dict[int, Mapping[str, Any]] = {}
    relative_paths: set[str] = set()
    split_counts = {split: 0 for split in VLM_REQUIRED_OFFICIAL_SPLITS}
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
            foreground_pixels = int(record.get("foreground_pixels", -1))
            observed_class_ids = set(record.get("observed_class_ids", []))
            raw_class_pixel_counts = record.get("class_pixel_counts")
            if not isinstance(raw_class_pixel_counts, dict):
                raise TypeError("class_pixel_counts must be a mapping")
            class_pixel_counts = {
                int(class_id): int(count)
                for class_id, count in raw_class_pixel_counts.items()
            }
        except (TypeError, ValueError) as error:
            raise PreflightError(
                "frozen VLM mask manifest entry is malformed"
            ) from error
        if img_id < 0 or img_id in entries:
            raise PreflightError("frozen VLM mask manifest has duplicate/invalid img_id")
        image_filename = record.get("img_filename")
        relative_path = record.get("relative_path")
        mapping_rule = record.get("mapping_rule")
        if isinstance(image_filename, str) and mapping_rule in allowed_rules:
            expected_candidate = dict(
                teacher_map_candidates(root, image_filename)
            )[str(mapping_rule)].resolve()
            recorded_candidate = (root / str(relative_path)).resolve()
        else:
            expected_candidate = None
            recorded_candidate = None
        if (
            not isinstance(image_filename, str)
            or record.get("producer_mask_name")
            != producer_vlm_mask_name(image_filename)
            or not isinstance(relative_path, str)
            or relative_path in relative_paths
            or split not in VLM_REQUIRED_OFFICIAL_SPLITS
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
            or not isinstance(record.get("decoded_binary_sha256"), str)
            or len(record["decoded_binary_sha256"]) != 64
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
        ):
            raise PreflightError("frozen VLM mask manifest entry is incompatible")
        relative_paths.add(relative_path)
        split_counts[int(split)] += 1
        entries[img_id] = record
    coverage = manifest.get("coverage")
    required_mapping_audit = manifest.get("required_mapping_audit")
    if (
        int(manifest.get("required_count", -1)) != len(entries)
        or not isinstance(coverage, dict)
        or any(
            coverage.get(str(split), {}).get("present") != split_counts[split]
            or coverage.get(str(split), {}).get("expected") != split_counts[split]
            for split in VLM_REQUIRED_OFFICIAL_SPLITS
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
    bank_hash = vlm_mask_bank_hash(
        [dict(record) for record in records],
        root=root,
        minimum_foreground_fraction=minimum,
        maximum_foreground_fraction=maximum,
    )
    if manifest.get("mask_bank_sha256") != bank_hash:
        raise PreflightError("frozen VLM mask-bank hash is invalid")
    report_path = output / "preflight" / "report.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PreflightError(f"invalid preflight report: {report_path}") from error
        if (
            report.get("mask_bank_sha256") != bank_hash
            or report.get("mask_manifest_sha256") != sha256_file(manifest_path)
            or report.get("mask_source") != VLM_PRODUCER
            or report.get("metadata_sha256") != manifest.get("metadata_sha256")
            or report.get("mask_contract_sha256") != contract_hash
            or report.get("resolved_config_sha256")
            != manifest.get("resolved_config_sha256")
            or (
                not debug
                and report.get("resolved_config_sha256")
                != config["resolved_config_sha256"]
            )
            or report.get("resolved_paths") != config["paths"]
        ):
            raise PreflightError("preflight report does not bind the VLM manifest")
    for img_id, record in entries.items():
        relative = record.get("relative_path")
        if not isinstance(relative, str):
            raise PreflightError("frozen VLM manifest contains an invalid path")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise PreflightError("frozen VLM manifest path escapes its root") from error
        if (
            not path.is_file()
            or path.stat().st_size != record.get("size_bytes")
            or sha256_file(path) != record.get("sha256")
        ):
            raise PreflightError(
                f"VLM source file no longer matches manifest: img_id={img_id}"
            )
    return VlmMaskBank(
        root=root,
        entries=entries,
        mask_bank_sha256=bank_hash,
        minimum_foreground_fraction=minimum,
        maximum_foreground_fraction=maximum,
    )
