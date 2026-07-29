"""Deterministic, bird-containing sanitized mask families."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
from PIL import Image

from setv.errors import DataValidationError
from setv.experts.exact_data import dilate_binary_mask
from setv.utils.seeds import derive_seed


FAMILIES = ("rectangle", "ellipse", "smooth_blob")
FAMILY_TO_ID = {name: index for index, name in enumerate(FAMILIES)}


def globally_balanced_short_families(sample_ids: list[str]) -> dict[str, str]:
    """Assign the two-mask family by hash order, with global counts differing by <=1."""
    ids = [str(value) for value in sample_ids]
    if len(set(ids)) != len(ids):
        raise DataValidationError("Sample IDs must be unique for family allocation")
    ordered = sorted(
        ids,
        key=lambda value: (
            hashlib.sha256(value.encode("utf-8")).digest(),
            value,
        ),
    )
    return {sample_id: FAMILIES[index % 3] for index, sample_id in enumerate(ordered)}


def _foreground(mask: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(mask, Image.Image):
        array = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    else:
        array = np.asarray(mask, dtype=bool)
    if array.ndim != 2 or not array.any():
        raise DataValidationError("Sanitized-mask source must be a nonempty binary mask")
    return array


def _sample_target(rng: np.random.Generator, config: dict[str, Any]):
    area_low, area_high = (
        float(value) for value in config["target_area_fraction_range"]
    )
    aspect_low, aspect_high = (
        float(value) for value in config["target_aspect_ratio_range"]
    )
    target_area = float(rng.uniform(area_low, area_high))
    target_aspect = float(
        np.exp(rng.uniform(np.log(aspect_low), np.log(aspect_high)))
    )
    return target_area, target_aspect


def _rectangle(
    source: np.ndarray,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = source.shape
    yy, xx = np.nonzero(source)
    xmin, xmax = int(xx.min()), int(xx.max())
    ymin, ymax = int(yy.min()), int(yy.max())
    target_area, aspect = _sample_target(rng, config)
    desired_pixels = target_area * height * width
    rectangle_width = max(xmax - xmin + 1, int(math.ceil(math.sqrt(desired_pixels * aspect))))
    rectangle_height = max(ymax - ymin + 1, int(math.ceil(math.sqrt(desired_pixels / aspect))))
    rectangle_width = min(width, rectangle_width)
    rectangle_height = min(height, rectangle_height)
    left_low = max(0, xmax - rectangle_width + 1)
    left_high = min(xmin, width - rectangle_width)
    top_low = max(0, ymax - rectangle_height + 1)
    top_high = min(ymin, height - rectangle_height)
    left = int(rng.integers(left_low, left_high + 1))
    top = int(rng.integers(top_low, top_high + 1))
    result = np.zeros_like(source)
    result[top : top + rectangle_height, left : left + rectangle_width] = True
    return result, {
        "target_area_fraction": target_area,
        "target_aspect_ratio": aspect,
        "left": left,
        "top": top,
        "width": rectangle_width,
        "height": rectangle_height,
    }


def _ellipse(
    source: np.ndarray,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = source.shape
    source_y, source_x = np.nonzero(source)
    target_area, aspect = _sample_target(rng, config)
    jitter = float(config["center_jitter_fraction"])
    center_x = float(source_x.mean() + rng.uniform(-jitter, jitter) * width)
    center_y = float(source_y.mean() + rng.uniform(-jitter, jitter) * height)
    center_x = float(np.clip(center_x, 0, width - 1))
    center_y = float(np.clip(center_y, 0, height - 1))
    target_pixels = target_area * height * width
    radius_x = math.sqrt(target_pixels * aspect / math.pi)
    radius_y = math.sqrt(target_pixels / (aspect * math.pi))
    normalized = np.square((source_x - center_x) / max(radius_x, 1e-6))
    normalized += np.square((source_y - center_y) / max(radius_y, 1e-6))
    scale = max(1.0, math.sqrt(float(normalized.max())) * float(config["containment_slack"]))
    radius_x *= scale
    radius_y *= scale
    grid_y, grid_x = np.indices(source.shape)
    result = (
        np.square((grid_x - center_x) / max(radius_x, 1e-6))
        + np.square((grid_y - center_y) / max(radius_y, 1e-6))
        <= 1.0
    )
    while not np.all(result[source]):
        radius_x *= 1.01
        radius_y *= 1.01
        result = (
            np.square((grid_x - center_x) / radius_x)
            + np.square((grid_y - center_y) / radius_y)
            <= 1.0
        )
    return result, {
        "target_area_fraction": target_area,
        "target_aspect_ratio": aspect,
        "center_x": center_x,
        "center_y": center_y,
        "radius_x": radius_x,
        "radius_y": radius_y,
    }


def _blob(
    source: np.ndarray,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = source.shape
    source_y, source_x = np.nonzero(source)
    target_area, _ = _sample_target(rng, config)
    jitter = float(config["center_jitter_fraction"])
    center_x = float(source_x.mean() + rng.uniform(-jitter, jitter) * width)
    center_y = float(source_y.mean() + rng.uniform(-jitter, jitter) * height)
    center_x = float(np.clip(center_x, 0, width - 1))
    center_y = float(np.clip(center_y, 0, height - 1))
    harmonics = int(config["blob_harmonics"])
    amplitude = float(config["blob_amplitude"])
    coefficients = rng.uniform(-amplitude, amplitude, size=harmonics)
    phases = rng.uniform(0.0, 2.0 * math.pi, size=harmonics)

    def radial_factor(theta):
        result = np.ones_like(theta, dtype=np.float64)
        for harmonic in range(1, harmonics + 1):
            result += coefficients[harmonic - 1] * np.cos(
                harmonic * theta + phases[harmonic - 1]
            )
        return np.clip(result, 0.45, None)

    source_dx = source_x - center_x
    source_dy = source_y - center_y
    source_theta = np.arctan2(source_dy, source_dx)
    source_distance = np.hypot(source_dx, source_dy)
    required_radius = float(
        np.max(source_distance / radial_factor(source_theta))
    ) * float(config["containment_slack"])
    angle_grid = np.linspace(-math.pi, math.pi, 4096, endpoint=False)
    mean_square_factor = float(np.square(radial_factor(angle_grid)).mean())
    target_radius = math.sqrt(
        target_area * height * width / (math.pi * mean_square_factor)
    )
    base_radius = max(required_radius, target_radius)
    grid_y, grid_x = np.indices(source.shape)
    dx = grid_x - center_x
    dy = grid_y - center_y
    theta = np.arctan2(dy, dx)
    distance = np.hypot(dx, dy)
    result = distance <= base_radius * radial_factor(theta)
    while not np.all(result[source]):
        base_radius *= 1.01
        result = distance <= base_radius * radial_factor(theta)
    return result, {
        "target_area_fraction": target_area,
        "center_x": center_x,
        "center_y": center_y,
        "base_radius": base_radius,
        "coefficients": coefficients.tolist(),
        "phases": phases.tolist(),
    }


def generate_sanitized_bank(
    mask: Image.Image,
    *,
    sample_id: str,
    short_family: str,
    base_seed: int,
    config: dict[str, Any],
    dilation_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if short_family not in FAMILIES:
        raise DataValidationError(f"Unknown short family: {short_family}")
    dilated = _foreground(
        dilate_binary_mask(
            mask,
            dilation_radius,
            structuring_element=config["dilation_structuring_element"],
        )
    )
    counts = {family: (2 if family == short_family else 3) for family in FAMILIES}
    masks = []
    family_ids = []
    seeds = []
    parameters = []
    generators = {
        "rectangle": _rectangle,
        "ellipse": _ellipse,
        "smooth_blob": _blob,
    }
    for family in FAMILIES:
        for variant in range(counts[family]):
            seed = derive_seed(
                base_seed,
                f"sanitized_mask:{sample_id}:family={family}:variant={variant}",
            )
            generated, details = generators[family](
                dilated, np.random.default_rng(seed), config
            )
            if not np.all(generated[dilated]):
                raise DataValidationError(
                    f"Sanitized {family} mask does not contain the dilated bird"
                )
            masks.append(generated)
            family_ids.append(FAMILY_TO_ID[family])
            seeds.append(seed)
            parameters.append(
                {
                    "family": family,
                    "family_id": FAMILY_TO_ID[family],
                    "variant": variant,
                    "seed": seed,
                    "area_fraction": float(generated.mean()),
                    **details,
                }
            )
    return (
        np.stack(masks).astype(bool),
        np.asarray(family_ids, dtype=np.uint8),
        np.asarray(seeds, dtype=np.uint32),
        parameters,
    )


def pack_masks(masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 4:
        raise DataValidationError(f"Mask bank must have shape [N,8,H,W], got {masks.shape}")
    return np.packbits(masks, axis=-1, bitorder="little")


def unpack_masks(packed: np.ndarray, width: int) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    if packed.ndim not in {3, 4}:
        raise DataValidationError("Packed sanitized masks must be rank 3 or 4")
    return np.unpackbits(packed, axis=-1, count=int(width), bitorder="little").astype(
        bool
    )
