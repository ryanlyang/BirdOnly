"""Configuration contract for sanitized-background fusion."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from setv.errors import ConfigurationError
from setv.fusion.config import validate_fusion_config


def load_sanitized_fusion_config(
    path: str | Path,
    *,
    seed: int | None = None,
    phase0_dir: str | None = None,
    object_expert_dir: str | None = None,
    sanitized_expert_dir: str | None = None,
    output_root: str | None = None,
    allow_expert_sanity_warnings: bool | None = None,
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = deepcopy(config)
    if seed is not None:
        config["fusion"]["seed"] = int(seed)
    for key, value in (
        ("phase0_dir", phase0_dir),
        ("object_expert_dir", object_expert_dir),
        ("sanitized_expert_dir", sanitized_expert_dir),
        ("output_root", output_root),
    ):
        if value is not None:
            config[key] = str(Path(value).expanduser())
    if allow_expert_sanity_warnings is not None:
        config["allow_expert_sanity_warnings"] = bool(allow_expert_sanity_warnings)
    config["_config_path"] = str(config_path)
    validate_sanitized_fusion_config(config)
    return config


def validate_sanitized_fusion_config(config: dict[str, Any]) -> None:
    translated = deepcopy(config)
    translated["phase"] = "exact_fusion"
    translated["exact_expert_dir"] = translated.get("sanitized_expert_dir")
    validate_fusion_config(translated)
    if config.get("phase") != "sanitized_fusion":
        raise ConfigurationError("phase must be 'sanitized_fusion'")


def resolved_sanitized_fusion_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in config.items() if key != "_config_path"}
