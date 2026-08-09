"""Locked selector-visible schema for every AnchorCal candidate epoch."""

from __future__ import annotations

from typing import Mapping


CANDIDATE_SCALAR_METRICS = (
    "ordinary_accuracy",
    "saliency_harmonic",
    "token_swap_harmonic",
    "background_blur_harmonic",
    "foreground_only_harmonic",
    "saliency_alignment",
    "swap_accuracy",
    "blur_accuracy",
    "foreground_only_accuracy",
    "saliency_product",
    "swap_product",
    "blur_product",
    "swap_mean_true_class_margin_drop",
    "swap_prediction_flip_rate",
    "swap_donor_margin_variance",
    "biased_mean_loss",
)

CANDIDATE_PER_EXAMPLE_NAMES = (
    "saliency_alignment",
    "saliency_fallback_absolute",
    "saliency_zero_attribution",
    "swap_donor_correct",
    "swap_donor_logits",
    "swap_margin_drop",
    "swap_prediction_flip",
    "swap_donor_margin_variance",
    "blur_sigma_correct",
    "blur_sigma_logits",
    "foreground_only_correct",
)


def candidate_per_example_shapes(
    selector_count: int, *, swap_donors: int, blur_sigmas: int
) -> Mapping[str, tuple[int, ...]]:
    """Return the exact locked per-example dataset shapes (without epoch)."""

    if selector_count <= 0 or swap_donors <= 0 or blur_sigmas <= 0:
        raise ValueError("candidate schema dimensions must be positive")
    count = int(selector_count)
    donors = int(swap_donors)
    sigmas = int(blur_sigmas)
    return {
        "saliency_alignment": (count,),
        "saliency_fallback_absolute": (count,),
        "saliency_zero_attribution": (count,),
        "swap_donor_correct": (count, donors),
        "swap_donor_logits": (count, donors, 2),
        "swap_margin_drop": (count,),
        "swap_prediction_flip": (count,),
        "swap_donor_margin_variance": (count,),
        "blur_sigma_correct": (count, sigmas),
        "blur_sigma_logits": (count, sigmas, 2),
        "foreground_only_correct": (count,),
    }
