"""Strict binary-mask validation, morphology, and patch geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .errors import PreflightError
from .io import hash_object, sha256_file


ALLOWED_ENCODINGS = ({0, 1}, {0, 255})


def load_binary_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as opened:
        array = np.asarray(opened)
    if array.ndim == 3:
        if not np.all(array == array[..., :1]):
            raise PreflightError(f"mask has unexpected non-identical channels: {path}")
        array = array[..., 0]
    if array.ndim != 2:
        raise PreflightError(f"mask must be two-dimensional: {path}")
    if not np.issubdtype(array.dtype, np.integer):
        raise PreflightError(f"mask must use integer encoding: {path}")
    values = {int(value) for value in np.unique(array).tolist()}
    if not any(values.issubset(allowed) for allowed in ALLOWED_ENCODINGS):
        raise PreflightError(f"mask has unsupported values {sorted(values)}: {path}")
    binary = array > 0
    if not binary.any():
        raise PreflightError(f"mask contains no foreground: {path}")
    if binary.all():
        raise PreflightError(f"mask covers the complete image: {path}")
    return binary


def disk_footprint(radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    axis = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return xx * xx + yy * yy <= radius * radius


def dilate_mask(mask: np.ndarray, radius: int = 8) -> np.ndarray:
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    if radius == 0:
        return mask.copy()
    try:
        from scipy.ndimage import binary_dilation

        return binary_dilation(mask, structure=disk_footprint(radius)).astype(bool)
    except ImportError as error:
        raise RuntimeError("scipy is required for exact disk dilation") from error


def patch_fractions(mask: np.ndarray, patch_size: int = 16) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("mask must be HxW")
    height, width = mask.shape
    if height % patch_size or width % patch_size:
        raise ValueError("mask dimensions must be divisible by patch size")
    return mask.reshape(
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    ).mean(axis=(1, 3))


@dataclass(frozen=True)
class PatchSets:
    foreground_fraction: np.ndarray
    dilated_foreground_fraction: np.ndarray
    foreground: np.ndarray
    background: np.ndarray
    mixed: np.ndarray


def classify_patches(mask: np.ndarray, radius: int = 8, patch_size: int = 16) -> PatchSets:
    foreground_fraction = patch_fractions(mask, patch_size)
    dilated_fraction = patch_fractions(dilate_mask(mask, radius), patch_size)
    foreground = foreground_fraction == 1.0
    background = dilated_fraction == 0.0
    mixed = ~(foreground | background)
    if np.any(foreground & background):
        raise AssertionError("a patch cannot be both pure foreground and safe background")
    return PatchSets(foreground_fraction, dilated_fraction, foreground, background, mixed)


def mask_geometry(mask: np.ndarray, radius: int = 8, patch_size: int = 16) -> dict[str, float]:
    yy, xx = np.nonzero(mask)
    height = float(yy.max() - yy.min() + 1)
    width = float(xx.max() - xx.min() + 1)
    padded = np.pad(mask, 1)
    perimeter = float(
        np.sum(padded[1:-1, 1:-1] & ~padded[:-2, 1:-1])
        + np.sum(padded[1:-1, 1:-1] & ~padded[2:, 1:-1])
        + np.sum(padded[1:-1, 1:-1] & ~padded[1:-1, :-2])
        + np.sum(padded[1:-1, 1:-1] & ~padded[1:-1, 2:])
    )
    area = float(mask.sum())
    pure_background = classify_patches(mask, radius, patch_size).background.sum()
    return {
        "raw_mask_area": area,
        "dilated_mask_area": float(dilate_mask(mask, radius).sum()),
        "bbox_width": width,
        "bbox_height": height,
        "aspect_ratio": width / max(height, 1.0),
        "centroid_x": float(xx.mean()),
        "centroid_y": float(yy.mean()),
        "perimeter": perimeter,
        "compactness": 4.0 * math.pi * area / max(perimeter * perimeter, 1.0),
        "eligible_background_patches": float(pure_background),
    }


def mask_manifest_entry(img_id: int, relative_path: str, path: Path) -> dict[str, object]:
    binary = load_binary_mask(path)
    with Image.open(path) as opened:
        raw = np.asarray(opened)
    return {
        "img_id": int(img_id),
        "relative_path": relative_path,
        "sha256": sha256_file(path),
        "width": int(binary.shape[1]),
        "height": int(binary.shape[0]),
        "unique_values": [int(item) for item in np.unique(raw).tolist()],
        "foreground_pixels": int(binary.sum()),
    }


def mask_bank_hash(entries: list[dict[str, object]]) -> str:
    normalized = sorted(entries, key=lambda item: int(item["img_id"]))
    return hash_object(normalized)

