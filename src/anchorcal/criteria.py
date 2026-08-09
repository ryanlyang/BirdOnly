"""Locked criterion aggregation from fixed per-image outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import class_balanced_mean, harmonic_mean


@dataclass(frozen=True)
class CriterionScores:
    ordinary_accuracy: float
    saliency_harmonic: float
    token_swap_harmonic: float
    background_blur_harmonic: float
    foreground_only_harmonic: float | None
    diagnostics: dict[str, float]


def donor_specific_accuracy(
    donor_correct: np.ndarray, labels: np.ndarray
) -> tuple[float, np.ndarray]:
    values = np.asarray(donor_correct, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("donor correctness must be [images,donors]")
    per_image = values.mean(axis=1)
    return class_balanced_mean(per_image, labels), per_image


def blur_accuracy(
    sigma_correct: np.ndarray, labels: np.ndarray
) -> tuple[float, np.ndarray]:
    values = np.asarray(sigma_correct, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("blur correctness must be [images,sigmas]")
    per_image = values.mean(axis=1)
    return class_balanced_mean(per_image, labels), per_image


def build_scores(
    *,
    full_biased_correct: np.ndarray,
    full_biased_labels: np.ndarray,
    selector_labels: np.ndarray,
    saliency_alignment: np.ndarray,
    donor_correct: np.ndarray,
    blur_correct: np.ndarray,
    foreground_only_correct: np.ndarray | None = None,
) -> CriterionScores:
    ordinary = class_balanced_mean(full_biased_correct, full_biased_labels)
    saliency = class_balanced_mean(saliency_alignment, selector_labels)
    swap, _ = donor_specific_accuracy(donor_correct, selector_labels)
    blur, _ = blur_accuracy(blur_correct, selector_labels)
    foreground = None
    if foreground_only_correct is not None:
        foreground = class_balanced_mean(foreground_only_correct, selector_labels)
    return CriterionScores(
        ordinary,
        harmonic_mean(ordinary, saliency),
        harmonic_mean(ordinary, swap),
        harmonic_mean(ordinary, blur),
        harmonic_mean(ordinary, foreground) if foreground is not None else None,
        {
            "saliency_alignment": saliency,
            "swap_accuracy": swap,
            "blur_accuracy": blur,
            "foreground_only_accuracy": foreground if foreground is not None else float("nan"),
            "saliency_product": ordinary * saliency,
            "swap_product": ordinary * swap,
            "blur_product": ordinary * blur,
        },
    )

