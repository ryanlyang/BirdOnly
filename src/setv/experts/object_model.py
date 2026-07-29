"""Pretrained ViT-S/16 construction for the object expert."""

from __future__ import annotations

from typing import Any

from setv.errors import DataValidationError


def create_object_expert_model(config: dict[str, Any]):
    try:
        import timm
    except ImportError as exc:
        raise DataValidationError(
            "timm is required for object-expert training. On Tigris activate "
            "/home/ryreu/miniforge3-aarch64/envs/fcv_gh200."
        ) from exc
    model_config = config["model"]
    model = timm.create_model(
        model_config["architecture"],
        pretrained=bool(model_config["pretrained"]),
        num_classes=int(model_config["num_classes"]),
    )
    return model

