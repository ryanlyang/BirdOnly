"""Configuration contracts for the Phase 6 uLA proxy and analysis."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from setv.errors import ConfigurationError


def _load(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Phase 6 config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ConfigurationError("Phase 6 config root must be a mapping")
    value = deepcopy(value)
    value["_config_path"] = str(config_path)
    return value


def load_ula_proxy_config(
    path: str | Path,
    *,
    phase0_dir: str | None = None,
    official_repo: str | None = None,
    ssl_checkpoint: str | None = None,
    output_root: str | None = None,
    seed: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    config = _load(path)
    for key, override in (
        ("phase0_dir", phase0_dir),
        ("official_repo", official_repo),
        ("ssl_checkpoint", ssl_checkpoint),
        ("output_root", output_root),
    ):
        if override is not None:
            config[key] = str(Path(override).expanduser())
    if seed is not None:
        config["training"]["seed"] = int(seed)
    if device is not None:
        config["training"]["device"] = device
    validate_ula_proxy_config(config)
    return config


def validate_ula_proxy_config(config: dict[str, Any]) -> None:
    if config.get("phase") != "ula_proxy" or int(config.get("schema_version", -1)) != 1:
        raise ConfigurationError("Invalid uLA proxy schema/phase")
    for key in ("phase0_dir", "official_repo", "ssl_checkpoint", "output_root"):
        if not config.get(key):
            raise ConfigurationError(f"uLA proxy path is not explicit: {key}")
    locked = {
        ("method", "label"): "uLA-style",
        ("method", "official_upstream"): "https://github.com/tsirif/uLA",
        (
            "method",
            "official_commit",
        ): "5867fb6e9a8485ed08b4cbe84900f2b5ac4fac5d",
        ("method", "ssl_method"): "mocov2plus",
        ("method", "ssl_backbone"): "resnet50",
        ("method", "proxy"): "frozen_ssl_encoder_linear_target_classifier",
        ("method", "proxy_groups"): ["proxy_prediction", "true_label"],
        ("method", "validation_metric"): "mean_nonempty_proxy_group_accuracy",
        ("training", "epochs"): 50,
        ("training", "optimizer"): "SGD",
        ("training", "learning_rate"): 1e-4,
        ("training", "momentum"): 0.9,
        ("training", "weight_decay"): 0.0,
        ("training", "batch_size"): 64,
        ("training", "evaluation_batch_size"): 64,
        ("training", "freeze_ssl_encoder"): True,
        ("training", "augmentation"): "official_waterbirds_minimal",
        ("training", "calibration"): "none",
    }
    for (section, key), expected in locked.items():
        actual = config.get(section, {}).get(key)
        if actual != expected:
            raise ConfigurationError(
                f"Locked uLA setting {section}.{key} must be {expected!r}, got {actual!r}"
            )
    if config["training"].get("seed") is None:
        raise ConfigurationError("uLA proxy seed must be explicit")
    if config["training"].get("device") not in {"cpu", "cuda"}:
        raise ConfigurationError("uLA proxy device must be cpu or cuda")
    if int(config["training"]["num_workers"]) < 0:
        raise ConfigurationError("uLA proxy num_workers cannot be negative")


def load_phase6_config(
    path: str | Path,
    *,
    phase0_dir: str | None = None,
    candidate_root: str | None = None,
    ula_proxy_dir: str | None = None,
    exact_fusion_dir: str | None = None,
    sanitized_fusion_dir: str | None = None,
    set_fusion_dir: str | None = None,
    output_dir: str | None = None,
    candidate_seeds: list[int] | None = None,
) -> dict[str, Any]:
    config = _load(path)
    for key, override in (
        ("phase0_dir", phase0_dir),
        ("candidate_root", candidate_root),
        ("ula_proxy_dir", ula_proxy_dir),
        ("output_dir", output_dir),
    ):
        if override is not None:
            config[key] = str(Path(override).expanduser())
    for key, override in (
        ("exact", exact_fusion_dir),
        ("sanitized", sanitized_fusion_dir),
        ("set", set_fusion_dir),
    ):
        if override is not None:
            config["fusion_dirs"][key] = str(Path(override).expanduser())
    if candidate_seeds is not None:
        config["candidate_seeds"] = [int(value) for value in candidate_seeds]
    validate_phase6_config(config)
    return config


def validate_phase6_config(config: dict[str, Any]) -> None:
    if config.get("phase") != "phase6_analysis" or int(
        config.get("schema_version", -1)
    ) != 1:
        raise ConfigurationError("Invalid Phase 6 analysis schema/phase")
    for key in ("phase0_dir", "candidate_root", "ula_proxy_dir", "output_dir"):
        if not config.get(key):
            raise ConfigurationError(f"Phase 6 path is not explicit: {key}")
    seeds = config.get("candidate_seeds")
    if not isinstance(seeds, list) or len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ConfigurationError("Phase 6 requires at least three unique candidate seeds")
    if set(config.get("fusion_dirs", {})) != {"exact", "sanitized", "set"}:
        raise ConfigurationError("Phase 6 requires exact/sanitized/set fusion paths")
    if any(not value for value in config["fusion_dirs"].values()):
        raise ConfigurationError("All Phase 6 fusion paths must be explicit")
    locked = {
        ("selection", "score_tolerance"): 1e-8,
        ("selection", "regret_tie_threshold"): 0.005,
        ("selection", "spearman_tie_threshold"): 0.02,
        ("selection", "minimum_candidate_seeds"): 3,
        ("selection", "choose_expert_and_fusion_jointly"): True,
        ("selection", "primary_criteria"): [
            "mean_oracle_selection_regret",
            "worst_oracle_selection_regret",
            "mean_spearman",
            "mean_kendall",
            "mean_pairwise_ranking_accuracy",
        ],
        ("test_policy", "reporting_only"): True,
        ("test_policy", "load_only_after_hashed_freeze"): True,
        ("test_policy", "print_before_freeze"): False,
    }
    for (section, key), expected in locked.items():
        actual = config.get(section, {}).get(key)
        if actual != expected:
            raise ConfigurationError(
                f"Locked Phase 6 setting {section}.{key} must be "
                f"{expected!r}, got {actual!r}"
            )


def resolved_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in config.items() if key != "_config_path"}
