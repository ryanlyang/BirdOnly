"""Pinned ViT-S/16 snapshot resolution and explicit local weight loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import PreflightError
from .io import atomic_write_json, sha256_file


REPOSITORY = "timm/vit_small_patch16_224.augreg_in21k_ft_in1k"
REVISION = "7e2c55630205e1266030f18370f4c6ed1a514b52"
EXPECTED_SHA256 = "79c03c635cdfd798a364a9d8c4e5c0b7255b975ea2c9616046d4f77ab01435aa"


def resolve_snapshot(cache_dir: str | Path, *, allow_download: bool) -> dict[str, Any]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise PreflightError("huggingface_hub is required for model preflight") from error
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=REPOSITORY,
                revision=REVISION,
                cache_dir=str(cache_dir),
                allow_patterns=["config.json", "model.safetensors"],
                local_files_only=not allow_download,
            )
        ).resolve()
    except Exception as error:
        mode = "online cache population" if allow_download else "offline production"
        raise PreflightError(f"pinned pretrained snapshot unavailable during {mode}") from error
    weights = snapshot / "model.safetensors"
    config = snapshot / "config.json"
    if not weights.is_file() or not config.is_file():
        raise PreflightError("pinned snapshot lacks config.json or model.safetensors")
    actual = sha256_file(weights)
    if actual != EXPECTED_SHA256:
        raise PreflightError(
            f"pretrained hash mismatch: expected {EXPECTED_SHA256}, found {actual}"
        )
    with config.open("r", encoding="utf-8") as handle:
        serialized = json.load(handle)
    mean = serialized.get("pretrained_cfg", {}).get("mean", serialized.get("mean"))
    std = serialized.get("pretrained_cfg", {}).get("std", serialized.get("std"))
    if mean is not None and list(mean) != [0.5, 0.5, 0.5]:
        raise PreflightError(f"serialized pretrained mean is unexpected: {mean}")
    if std is not None and list(std) != [0.5, 0.5, 0.5]:
        raise PreflightError(f"serialized pretrained std is unexpected: {std}")
    return {
        "repository": REPOSITORY,
        "revision": REVISION,
        "snapshot_path": str(snapshot),
        "weights_path": str(weights),
        "weights_sha256": actual,
        "serialized_config": serialized,
    }


def write_pretrained_manifest(path: str | Path, value: dict[str, Any]) -> None:
    atomic_write_json(path, value)


def load_verified_state(weights_path: str | Path) -> dict[str, Any]:
    actual = sha256_file(weights_path)
    if actual != EXPECTED_SHA256:
        raise PreflightError(f"refusing unverified pretrained weights: {actual}")
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise PreflightError("safetensors is required to load pinned weights") from error
    return load_file(str(weights_path), device="cpu")


def create_pretrained_vit(weights_path: str | Path):
    import timm

    if timm.__version__ != "1.0.28":
        raise PreflightError(f"timm 1.0.28 is required, found {timm.__version__}")
    model = timm.create_model(
        "vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=False
    )
    state = load_verified_state(weights_path)
    incompatible = model.load_state_dict(state, strict=False)
    # timm snapshots occasionally omit classifier metadata but may not omit a
    # backbone tensor.  Keep the check explicit and auditable.
    backbone_missing = [key for key in incompatible.missing_keys if not key.startswith("head.")]
    unexpected = list(incompatible.unexpected_keys)
    if backbone_missing or unexpected:
        raise PreflightError(
            f"pretrained state incompatible; missing={backbone_missing}, unexpected={unexpected}"
        )
    return model

