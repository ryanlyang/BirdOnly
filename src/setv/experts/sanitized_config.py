"""Configuration contracts for Phase 3 sanitized masks and expert training."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from setv.errors import ConfigurationError
from setv.model_contracts import vit_small_normalization_matches


def _read(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ConfigurationError(f"Configuration not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigurationError("Configuration root must be a mapping")
    return resolved, deepcopy(loaded)


def load_sanitized_bank_config(
    path: str | Path,
    *,
    seed: int | None = None,
    phase0_dir: str | None = None,
    output_root: str | None = None,
    auditor_device: str | None = None,
) -> dict[str, Any]:
    config_path, config = _read(path)
    if seed is not None:
        config["mask_bank"]["seed"] = int(seed)
    if phase0_dir is not None:
        config["phase0_dir"] = str(Path(phase0_dir).expanduser())
    if output_root is not None:
        config["output_root"] = str(Path(output_root).expanduser())
    if auditor_device is not None:
        config["auditor"]["small_cnn"]["device"] = auditor_device
    config["_config_path"] = str(config_path)
    validate_sanitized_bank_config(config)
    return config


def validate_sanitized_bank_config(config: dict[str, Any]) -> None:
    if config.get("phase") != "sanitized_mask_bank":
        raise ConfigurationError("phase must be 'sanitized_mask_bank'")
    if int(config.get("schema_version", -1)) != 1:
        raise ConfigurationError("Only sanitized-mask schema_version 1 is supported")
    if config["mask_bank"].get("seed") is None:
        raise ConfigurationError("Sanitized-mask seed must be explicit")
    locked = {
        ("mask_bank", "masks_per_image"): 8,
        ("mask_bank", "families"): ["rectangle", "ellipse", "smooth_blob"],
        ("mask_bank", "family_allocation"): "globally_balanced_sample_hash_3_3_2",
        ("mask_bank", "dilation_pixels_at_224"): 8,
        ("mask_bank", "dilation_structuring_element"): "euclidean_disk",
        ("mask_bank", "target_area_fraction_range"): [0.35, 0.60],
        ("mask_bank", "target_aspect_ratio_range"): [0.75, 4.0 / 3.0],
        ("mask_bank", "center_jitter_fraction"): 0.10,
        ("mask_bank", "containment_slack"): 1.08,
        ("mask_bank", "blob_harmonics"): 4,
        ("mask_bank", "blob_amplitude"): 0.12,
        ("mask_bank", "algorithm_version"): "setv_sanitized_v1",
        ("auditor", "heldout_image_fraction"): 0.20,
        ("auditor", "maximum_balanced_accuracy"): 0.53,
        ("auditor", "bootstrap_repetitions"): 1000,
        ("auditor", "bootstrap_confidence"): 0.95,
        ("auditor.logistic", "C"): 1.0,
        ("auditor.logistic", "class_weight"): "balanced",
        ("auditor.logistic", "solver"): "lbfgs",
        ("auditor.logistic", "max_iter"): 1000,
        ("auditor.gradient_boosted_trees", "n_estimators"): 100,
        ("auditor.gradient_boosted_trees", "learning_rate"): 0.05,
        ("auditor.gradient_boosted_trees", "max_depth"): 3,
        ("auditor.small_cnn", "epochs"): 10,
        ("auditor.small_cnn", "batch_size"): 128,
        ("auditor.small_cnn", "learning_rate"): 0.001,
        ("auditor.small_cnn", "weight_decay"): 0.0001,
        ("auditor.small_cnn", "pooled_resolution"): 56,
    }
    for (section, key), expected in locked.items():
        current = config
        for part in section.split("."):
            current = current[part]
        actual = current.get(key)
        if actual != expected:
            raise ConfigurationError(
                f"Locked sanitized setting {section}.{key} must be "
                f"{expected!r}, got {actual!r}"
            )
    if config["auditor"]["small_cnn"]["device"] not in {"cpu", "cuda"}:
        raise ConfigurationError("auditor.small_cnn.device must be cpu or cuda")
    if not 0 < float(config["mask_bank"]["target_area_fraction_range"][0]):
        raise ConfigurationError("Sanitized target area fractions must be positive")
    if float(config["mask_bank"]["target_area_fraction_range"][1]) > 1:
        raise ConfigurationError("Sanitized target area fractions cannot exceed one")


def load_sanitized_expert_config(
    path: str | Path,
    *,
    seed: int | None = None,
    phase0_dir: str | None = None,
    mask_bank_dir: str | None = None,
    output_root: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    config_path, config = _read(path)
    if seed is not None:
        config["training"]["seed"] = int(seed)
    for key, value in (
        ("phase0_dir", phase0_dir),
        ("mask_bank_dir", mask_bank_dir),
        ("output_root", output_root),
    ):
        if value is not None:
            config[key] = str(Path(value).expanduser())
    if device is not None:
        config["training"]["device"] = device
    config["_config_path"] = str(config_path)
    validate_sanitized_expert_config(config)
    return config


def validate_sanitized_expert_config(config: dict[str, Any]) -> None:
    if config.get("phase") != "background_sanitized":
        raise ConfigurationError("phase must be 'background_sanitized'")
    if int(config.get("schema_version", -1)) != 1:
        raise ConfigurationError("Only sanitized-expert schema_version 1 is supported")
    if config["training"].get("seed") is None:
        raise ConfigurationError("Sanitized-expert seed must be explicit")
    if not config.get("mask_bank_dir"):
        raise ConfigurationError("mask_bank_dir must be explicit")
    locked = {
        ("model", "architecture"): "vit_small_patch16_224",
        ("model", "pretrained"): True,
        ("model", "num_classes"): 2,
        ("input", "green_rgb"): [0, 255, 0],
        ("input", "training_masks_per_image"): 2,
        ("input", "validation_masks_per_image"): 8,
        ("input", "geometric_augmentation"): "canonical_only",
        ("training", "epochs"): 20,
        ("training", "optimizer"): "AdamW",
        ("training", "learning_rate"): 3e-5,
        ("training", "weight_decay"): 0.05,
        ("training", "batch_size"): 32,
        ("training", "scheduler"): "cosine",
        ("training", "warmup_epochs"): 2,
        ("training", "label_smoothing"): 0.0,
        ("training", "lambda_consistency"): 0.5,
        ("storage", "save_final_checkpoint"): True,
        ("storage", "save_intermediate_checkpoints"): False,
    }
    for (section, key), expected in locked.items():
        actual = config[section].get(key)
        if actual != expected:
            raise ConfigurationError(
                f"Locked sanitized-expert setting {section}.{key} must be "
                f"{expected!r}, got {actual!r}"
            )
    if config["training"]["device"] not in {"cpu", "cuda"}:
        raise ConfigurationError("training.device must be cpu or cuda")
    if int(config["training"]["evaluation_batch_size"]) <= 0:
        raise ConfigurationError("evaluation_batch_size must be positive")
    if int(config["training"]["num_workers"]) < 0:
        raise ConfigurationError("num_workers cannot be negative")
    if not vit_small_normalization_matches(config["model"]):
        raise ConfigurationError(
            "Normalization must match the selected timm ViT-S/16 pretrained metadata"
        )


def resolved_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in config.items() if key != "_config_path"}
