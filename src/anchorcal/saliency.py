"""True-class pre-softmax gradient-times-activation saliency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .anchor_cache import aggregate_source_coordinates
from .metrics import class_balanced_mean


@dataclass(frozen=True)
class SaliencyImageResult:
    alignment: float
    fallback_absolute: bool
    zero_scored_attribution: bool
    foreground_density: float
    background_density: float


def signed_gradient_activation(logits, labels, activations):
    import torch

    true_score = logits.gather(1, labels[:, None]).sum()
    gradient = torch.autograd.grad(
        true_score,
        activations,
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )[0]
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError("saliency activation gradient is missing or non-finite")
    return (gradient * activations).sum(dim=-1)


def image_alignment(
    signed_values: np.ndarray,
    source_indices: np.ndarray,
    foreground_indices: np.ndarray,
    background_indices: np.ndarray,
    *,
    mass_tolerance: float = 1e-12,
) -> SaliencyImageResult:
    coordinates, summed = aggregate_source_coordinates(signed_values, source_indices)
    foreground_mask = np.isin(coordinates, np.asarray(foreground_indices))
    background_mask = np.isin(coordinates, np.asarray(background_indices))
    if not foreground_mask.any() or not background_mask.any():
        raise ValueError("saliency image lacks a scored foreground or background coordinate")
    scored = foreground_mask | background_mask
    values = np.maximum(summed, 0.0)
    fallback = float(values[scored].sum()) < mass_tolerance
    if fallback:
        values = np.abs(summed)
    zero = float(values[scored].sum()) <= mass_tolerance
    if zero:
        return SaliencyImageResult(0.5, fallback, True, 0.0, 0.0)
    foreground_density = float(values[foreground_mask].mean())
    background_density = float(values[background_mask].mean())
    alignment = foreground_density / (foreground_density + background_density + 1e-12)
    return SaliencyImageResult(
        float(alignment), fallback, False, foreground_density, background_density
    )


def anchor_image_alignment(
    foreground_signed: np.ndarray,
    foreground_source: np.ndarray,
    background_signed_occurrences: np.ndarray,
    background_source_occurrences: np.ndarray,
    foreground_indices: np.ndarray,
    background_indices: np.ndarray,
    reliance_lambda: float,
) -> SaliencyImageResult:
    """Mix signed components first, summing repeated background coordinates."""

    foreground_coordinates, foreground_values = aggregate_source_coordinates(
        foreground_signed, foreground_source
    )
    background_coordinates, background_values = aggregate_source_coordinates(
        background_signed_occurrences, background_source_occurrences
    )
    all_coordinates = np.union1d(foreground_coordinates, background_coordinates)
    all_values = np.zeros(len(all_coordinates), dtype=np.float64)
    for coordinates, values, coefficient in (
        (foreground_coordinates, foreground_values, reliance_lambda),
        (background_coordinates, background_values, 1.0 - reliance_lambda),
    ):
        positions = np.searchsorted(all_coordinates, coordinates)
        all_values[positions] += coefficient * values
    return image_alignment(
        all_values,
        all_coordinates,
        foreground_indices,
        background_indices,
    )


def aggregate_alignment(results: list[SaliencyImageResult], labels: np.ndarray) -> float:
    return class_balanced_mean(
        np.asarray([result.alignment for result in results]), np.asarray(labels)
    )

