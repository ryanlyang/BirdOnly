"""ImageNet-pretrained ViT-S/16 candidate construction."""

from __future__ import annotations

from typing import Any

from setv.errors import DataValidationError


def create_candidate_model(config: dict[str, Any]):
    try:
        import timm
    except ImportError as exc:
        raise DataValidationError(
            "timm is required for candidate training. On Tigris activate "
            "/home/ryreu/miniforge3-aarch64/envs/fcv_gh200."
        ) from exc
    settings = config["model"]
    return timm.create_model(
        settings["architecture"],
        pretrained=bool(settings["pretrained"]),
        num_classes=int(settings["num_classes"]),
    )
