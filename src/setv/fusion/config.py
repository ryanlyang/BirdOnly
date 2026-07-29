"""Phase 2 fusion configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from setv.errors import ConfigurationError


def load_fusion_config(
    path: str | Path,
    *,
    seed: int | None = None,
    phase0_dir: str | None = None,
    object_expert_dir: str | None = None,
    exact_expert_dir: str | None = None,
    output_root: str | None = None,
    allow_expert_sanity_warnings: bool | None = None,
) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = deepcopy(config)
    if seed is not None:
        config["fusion"]["seed"] = int(seed)
    for key, value in (
        ("phase0_dir", phase0_dir),
        ("object_expert_dir", object_expert_dir),
        ("exact_expert_dir", exact_expert_dir),
        ("output_root", output_root),
    ):
        if value is not None:
            config[key] = str(Path(value).expanduser())
    if allow_expert_sanity_warnings is not None:
        config["allow_expert_sanity_warnings"] = bool(allow_expert_sanity_warnings)
    config["_config_path"] = str(path)
    validate_fusion_config(config)
    return config


def validate_fusion_config(config: dict[str, Any]) -> None:
    if config.get("phase") != "exact_fusion" or int(config.get("schema_version", -1)) != 1:
        raise ConfigurationError("Invalid exact-fusion schema/phase")
    for key in ("phase0_dir", "object_expert_dir", "exact_expert_dir", "output_root"):
        if not config.get(key):
            raise ConfigurationError(f"Fusion path is not locked: {key}")
    if config["fusion"].get("seed") is None:
        raise ConfigurationError("Fusion cross-fitting seed must be explicit")
    expected = {
        ("hard", "min_examples_per_class"): 5,
        ("rank", "clip_minimum"): 0.001,
        ("logistic", "n_folds"): 5,
        ("logistic", "n_repeats"): 5,
        ("logistic", "penalty"): "l2",
        ("logistic", "C"): 1.0,
        ("logistic", "class_weight"): "balanced",
        ("logistic", "solver"): "lbfgs",
        ("logistic", "max_iter"): 1000,
        ("weighting", "alphas"): [0.5, 1.0, 2.0, 4.0],
        ("weighting", "percentile_clip_minimum"): 0.001,
        ("weighting", "ess_warning_threshold"): 10.0,
    }
    for (section, key), value in expected.items():
        actual = config["fusion"][section].get(key)
        if actual != value:
            raise ConfigurationError(
                f"Locked fusion setting {section}.{key} must be {value!r}, got {actual!r}"
            )


def resolved_fusion_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in config.items() if key != "_config_path"}

