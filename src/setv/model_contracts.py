"""Shared pretrained-model contracts frozen for the SETV campaign."""

from __future__ import annotations


VIT_SMALL_PATCH16_224_ARCHITECTURE = "vit_small_patch16_224"

# timm 1.0.28 resolves the pretrained weights selected by
# ``vit_small_patch16_224`` with symmetric [-1, 1] input normalization.
VIT_SMALL_PATCH16_224_MEAN = [0.5, 0.5, 0.5]
VIT_SMALL_PATCH16_224_STD = [0.5, 0.5, 0.5]


def vit_small_normalization_matches(model_config: dict) -> bool:
    """Return whether a config matches the selected timm weight metadata."""
    return (
        model_config.get("normalization_mean") == VIT_SMALL_PATCH16_224_MEAN
        and model_config.get("normalization_std") == VIT_SMALL_PATCH16_224_STD
    )
