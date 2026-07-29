"""Locked configuration contract for the Phase 5 candidate trajectory."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from setv.errors import ConfigurationError


def load_candidate_config(
    path: str | Path,
    *,
    seed: int | None = None,
    phase0_dir: str | None = None,
    exact_fusion_dir: str | None = None,
    sanitized_fusion_dir: str | None = None,
    set_fusion_dir: str | None = None,
    output_root: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Candidate config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigurationError("Candidate config root must be a mapping")
    config = deepcopy(config)
    if seed is not None:
        config["training"]["seed"] = int(seed)
    if phase0_dir is not None:
        config["phase0_dir"] = str(Path(phase0_dir).expanduser())
    for key, value in (
        ("exact", exact_fusion_dir),
        ("sanitized", sanitized_fusion_dir),
        ("set", set_fusion_dir),
    ):
        if value is not None:
            config["fusion_dirs"][key] = str(Path(value).expanduser())
    if output_root is not None:
        config["output_root"] = str(Path(output_root).expanduser())
    if device is not None:
        config["training"]["device"] = device
    config["_config_path"] = str(config_path)
    validate_candidate_config(config)
    return config


def validate_candidate_config(config: dict[str, Any]) -> None:
    if config.get("phase") != "candidate_erm" or int(
        config.get("schema_version", -1)
    ) != 1:
        raise ConfigurationError("Invalid candidate schema/phase")
    for key in ("phase0_dir", "output_root"):
        if not config.get(key):
            raise ConfigurationError(f"Candidate path is not locked: {key}")
    if set(config.get("fusion_dirs", {})) != {"exact", "sanitized", "set"}:
        raise ConfigurationError("Exactly exact/sanitized/set fusion paths are required")
    if any(not value for value in config["fusion_dirs"].values()):
        raise ConfigurationError("All three Phase 2–4 fusion paths must be explicit")
    if config["training"].get("seed") is None or not isinstance(
        config["training"]["seed"], int
    ):
        raise ConfigurationError("Candidate seed must be an explicit integer")
    locked = {
        ("model", "architecture"): "vit_small_patch16_224",
        ("model", "pretrained"): True,
        ("model", "num_classes"): 2,
        ("training", "epochs"): 50,
        ("training", "optimizer"): "AdamW",
        ("training", "learning_rate"): 3e-5,
        ("training", "weight_decay"): 0.05,
        ("training", "batch_size"): 64,
        ("training", "scheduler"): "cosine",
        ("training", "warmup_epochs"): 5,
        ("training", "label_smoothing"): 0.0,
        ("selectors", "score_tolerance"): 1e-8,
        ("selectors", "hard_min_examples_per_class"): 5,
        ("selectors", "alphas"): [0.5, 1.0, 2.0, 4.0],
        ("selectors", "ess_warning_threshold"): 10.0,
        ("selectors", "ula_phase"): 6,
        (
            "selectors",
            "pseudo_group_tie_break",
        ): ["ordinary_accuracy", "ordinary_loss", "earlier_epoch"],
        ("storage", "save_all_candidate_checkpoints"): False,
        ("storage", "save_rolling_selected_checkpoints"): True,
        ("storage", "deduplicate_checkpoints_by_epoch"): True,
        ("storage", "save_biased_val_logits_every_epoch"): True,
        ("storage", "test_metrics_reporting_only"): True,
        ("storage", "hide_test_until_selection_receipt"): True,
    }
    for (section, key), expected in locked.items():
        actual = config[section].get(key)
        if actual != expected:
            raise ConfigurationError(
                f"Locked candidate setting {section}.{key} must be "
                f"{expected!r}, got {actual!r}"
            )
    if config["training"]["device"] not in {"cpu", "cuda"}:
        raise ConfigurationError("training.device must be cpu or cuda")
    if int(config["training"]["evaluation_batch_size"]) <= 0:
        raise ConfigurationError("evaluation_batch_size must be positive")
    if int(config["training"]["num_workers"]) < 0:
        raise ConfigurationError("num_workers cannot be negative")
    if config["model"]["normalization_mean"] != [0.485, 0.456, 0.406]:
        raise ConfigurationError("Locked ImageNet normalization mean changed")
    if config["model"]["normalization_std"] != [0.229, 0.224, 0.225]:
        raise ConfigurationError("Locked ImageNet normalization std changed")


def resolved_candidate_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in config.items() if key != "_config_path"}
