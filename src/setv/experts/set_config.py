"""Configuration contract for the Phase 4 background set transformer."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from setv.errors import ConfigurationError
from setv.model_contracts import vit_small_normalization_matches


def load_set_expert_config(
    path: str | Path,
    *,
    seed: int | None = None,
    phase0_dir: str | None = None,
    output_root: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Set-expert config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigurationError("Set-expert config root must be a mapping")
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
    validate_set_expert_config(config)
    return config


def validate_set_expert_config(config: dict[str, Any]) -> None:
    if config.get("phase") != "background_set":
        raise ConfigurationError("phase must be 'background_set'")
    if int(config.get("schema_version", -1)) != 1:
        raise ConfigurationError("Only set-expert schema_version 1 is supported")
    if config["training"].get("seed") is None:
        raise ConfigurationError("Set-expert seed must be explicit")
    locked = {
        ("model", "architecture"): "vit_small_patch16_224",
        ("model", "pretrained"): True,
        ("model", "num_classes"): 2,
        ("model", "embedding_dim"): 384,
        ("model", "transformer_blocks"): 4,
        ("model", "attention_heads"): 6,
        ("model", "mlp_ratio"): 4,
        ("model", "dropout"): 0.1,
        ("input", "patch_size"): 16,
        ("input", "dilation_pixels_at_224"): 8,
        ("input", "dilation_structuring_element"): "euclidean_disk",
        ("input", "maximum_foreground_fraction"): 0.01,
        ("input", "max_background_tokens"): 180,
        ("input", "min_background_tokens"): 16,
        ("input", "coarse_spatial_bins_per_axis"): 3,
        ("input", "use_dense_position_embeddings"): False,
        ("training", "epochs"): 30,
        ("training", "optimizer"): "AdamW",
        ("training", "learning_rate"): 1e-4,
        ("training", "weight_decay"): 0.05,
        ("training", "batch_size"): 64,
        ("training", "scheduler"): "cosine",
        ("training", "warmup_epochs"): 3,
        ("training", "token_dropout"): 0.20,
        ("training", "validation_views"): 8,
        ("training", "label_smoothing"): 0.0,
        ("storage", "save_final_checkpoint"): True,
        ("storage", "save_intermediate_checkpoints"): False,
    }
    for (section, key), expected in locked.items():
        actual = config[section].get(key)
        if actual != expected:
            raise ConfigurationError(
                f"Locked set-expert setting {section}.{key} must be "
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


def resolved_set_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in config.items() if key != "_config_path"}
