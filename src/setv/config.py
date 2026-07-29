"""Strict YAML configuration loading for Phase 0."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from setv.errors import ConfigurationError


REQUIRED_PATHS = (
    ("schema_version",),
    ("project_name",),
    ("data", "dataset_root"),
    ("data", "metadata_csv"),
    ("data", "image_column"),
    ("data", "sample_id_column"),
    ("data", "target_column"),
    ("data", "place_column"),
    ("data", "official_split_column"),
    ("data", "official_split_values", "train"),
    ("data", "official_split_values", "oracle_val"),
    ("data", "official_split_values", "test"),
    ("masks", "root"),
    ("masks", "mapping_mode"),
    ("masks", "threshold_normalized"),
    ("split", "candidate_train_fraction"),
    ("split", "seed"),
    ("audit", "visual_samples_per_split"),
    ("audit", "visual_seed"),
    ("output", "phase0_dir"),
)


def _get_nested(config: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ConfigurationError(
                f"Missing required configuration key: {'.'.join(path)}"
            )
        value = value[key]
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a Phase 0 YAML configuration."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigurationError("Configuration root must be a mapping")
    for required in REQUIRED_PATHS:
        _get_nested(loaded, required)

    config = deepcopy(loaded)
    config["_config_path"] = str(config_path)
    _validate_values(config)
    return config


def _validate_values(config: Mapping[str, Any]) -> None:
    if int(config["schema_version"]) != 1:
        raise ConfigurationError("Only schema_version 1 is supported")

    fraction = float(config["split"]["candidate_train_fraction"])
    if not 0.0 < fraction < 1.0:
        raise ConfigurationError("candidate_train_fraction must lie in (0, 1)")

    threshold = float(config["masks"]["threshold_normalized"])
    if not 0.0 <= threshold <= 1.0:
        raise ConfigurationError("threshold_normalized must lie in [0, 1]")

    mapping_mode = str(config["masks"]["mapping_mode"])
    supported = {
        "relative_stem",
        "unique_basename",
        "relative_stem_then_unique_basename",
        "sample_id",
    }
    if mapping_mode not in supported:
        raise ConfigurationError(
            f"Unsupported masks.mapping_mode={mapping_mode!r}; "
            f"expected one of {sorted(supported)}"
        )

    split_values = config["data"]["official_split_values"]
    if len(set(split_values.values())) != 3:
        raise ConfigurationError("Official split values must be distinct")

    samples = int(config["audit"]["visual_samples_per_split"])
    if samples < 20:
        raise ConfigurationError(
            "Production visual_samples_per_split must be at least 20"
        )


def apply_overrides(
    config: Mapping[str, Any],
    *,
    dataset_root: str | None = None,
    mask_root: str | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Apply explicit CLI path overrides without changing scientific settings."""
    result = deepcopy(dict(config))
    if dataset_root is not None:
        result["data"]["dataset_root"] = str(Path(dataset_root).expanduser())
    if mask_root is not None:
        result["masks"]["root"] = str(Path(mask_root).expanduser())
    if output_dir is not None:
        result["output"]["phase0_dir"] = str(Path(output_dir).expanduser())
    return result

