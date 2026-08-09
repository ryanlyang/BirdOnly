"""Algebraic AnchorCal extreme-branch cache and parity checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MARGIN_SCALE_EPSILON = 1.0e-8


@dataclass(frozen=True)
class ExtremeCache:
    foreground_logits: np.ndarray
    background_logits: np.ndarray
    foreground_signed: np.ndarray | None = None
    background_signed: np.ndarray | None = None


def center_numpy(logits: np.ndarray) -> np.ndarray:
    return logits - logits.mean(axis=-1, keepdims=True)


def cached_logits(
    cache: ExtremeCache,
    reliance_lambda: float,
    foreground_scale: float,
    background_scale: float,
) -> np.ndarray:
    if not 0 <= reliance_lambda <= 1:
        raise ValueError("lambda must be in [0,1]")
    return (
        reliance_lambda
        * center_numpy(cache.foreground_logits)
        / (foreground_scale + MARGIN_SCALE_EPSILON)
        + (1.0 - reliance_lambda)
        * center_numpy(cache.background_logits)
        / (background_scale + MARGIN_SCALE_EPSILON)
    )


def cached_signed_contributions(cache: ExtremeCache, reliance_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    if cache.foreground_signed is None or cache.background_signed is None:
        raise ValueError("cache does not contain signed saliency components")
    return (
        reliance_lambda * cache.foreground_signed,
        (1.0 - reliance_lambda) * cache.background_signed,
    )


def aggregate_source_coordinates(
    occurrence_values: np.ndarray, source_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sum repeated occurrence contributions; do not average a second time."""

    values = np.asarray(occurrence_values)
    indices = np.asarray(source_indices)
    if values.shape != indices.shape:
        raise ValueError("occurrence values and source indices must match")
    unique = np.unique(indices[indices >= 0])
    summed = np.asarray([values[indices == coordinate].sum() for coordinate in unique])
    return unique, summed


def require_cache_parity(
    direct: np.ndarray,
    cached: np.ndarray,
    *,
    tolerance: float,
    quantity: str,
) -> float:
    difference = float(np.max(np.abs(np.asarray(direct) - np.asarray(cached))))
    if not np.isfinite(difference) or difference > tolerance:
        raise AssertionError(
            f"direct/cached {quantity} parity failed: {difference} > {tolerance}"
        )
    return difference
