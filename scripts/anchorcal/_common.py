"""CLI configuration helper (not installed as a public API)."""

from __future__ import annotations

from pathlib import Path

from anchorcal.config import deep_merge, load_config
from anchorcal.io import read_yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "anchorcal" / "pilot.yaml"
DEFAULT_PATHS = REPO_ROOT / "configs" / "anchorcal" / "paths.local.yaml"
DEBUG_CONFIG = REPO_ROOT / "configs" / "anchorcal" / "debug.yaml"


def resolved_config(config_path: str, paths_path: str, *, debug: bool = False):
    overrides = read_yaml(DEBUG_CONFIG) if debug else None
    return load_config(
        config_path,
        paths_path,
        overrides=overrides,
        require_paths=True,
    )

