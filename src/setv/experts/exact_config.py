"""Configuration validation for the exact-fill background expert."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from setv.errors import ConfigurationError


def load_exact_expert_config(
    path: str | Path,
    *,
    seed: int | None = None,
    phase0_dir: str | None = None,
    output_root: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Exact-expert config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigurationError("Exact-expert config root must be a mapping")
    config = deepcopy(config)
    if seed is not None:
        config["training"]["seed"] = int(seed)
    if phase0_dir is not None:
        config["phase0_dir"] = str(Path(phase0_dir).expanduser())
    if output_root is not None:
        config["output_root"] = str(Path(output_root).expanduser())
    if device is not None:
        config["training"]["device"] = device
    config["_config_path"] = str(config_path)
    validate_exact_expert_config(config)
    return config


def validate_exact_expert_config(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", -1)) != 1:
        raise ConfigurationError("Only exact-expert schema_version 1 is supported")
    if config.get("phase") != "background_exact":
        raise ConfigurationError("phase must be 'background_exact'")
    if config["training"].get("seed") is None:
        raise ConfigurationError(
            "Exact background-expert seed is not locked; pass --seed explicitly"
        )
    locked = {
        ("model", "architecture"): "vit_small_patch16_224",
        ("model", "pretrained"): True,
        ("model", "num_classes"): 2,
        ("input", "green_rgb"): [0, 255, 0],
        ("input", "dilation_pixels_at_224"): 8,
        ("input", "dilation_structuring_element"): "euclidean_disk",
        ("training", "epochs"): 20,
        ("training", "optimizer"): "AdamW",
        ("training", "learning_rate"): 3e-5,
        ("training", "weight_decay"): 0.05,
        ("training", "batch_size"): 64,
        ("training", "scheduler"): "cosine",
        ("training", "warmup_epochs"): 2,
        ("training", "label_smoothing"): 0.0,
        ("storage", "save_final_checkpoint"): True,
        ("storage", "save_intermediate_checkpoints"): False,
    }
    for (section, key), expected in locked.items():
        actual = config[section].get(key)
        if actual != expected:
            raise ConfigurationError(
                f"Locked exact-expert setting {section}.{key} must be "
                f"{expected!r}, got {actual!r}"
            )
    if config["training"]["device"] not in {"cuda", "cpu"}:
        raise ConfigurationError("training.device must be cuda or cpu")
    if config["model"]["normalization_mean"] != [0.485, 0.456, 0.406]:
        raise ConfigurationError("Locked ImageNet normalization mean changed")
    if config["model"]["normalization_std"] != [0.229, 0.224, 0.225]:
        raise ConfigurationError("Locked ImageNet normalization std changed")


def resolved_exact_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in config.items() if key != "_config_path"}

