"""Joint verification of visible and reporting-only candidate checkpoints.

Practical selection must import :mod:`visible_checkpoint_verification`
directly.  This module owns the reporting-only schema and is used only by
candidate production and post-freeze analysis code.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

from .visible_checkpoint_verification import (
    CHECKPOINT_SCHEMA_VERSION,
    MODEL_OBJECT_SCHEMA_VERSION,
    RESUME_SCHEMA_VERSION,
    VISIBLE_SELECTORS,
    _json,
    _require,
    _verify_model_object,
    hash_model_state,
    verify_visible_checkpoint_artifacts,
)
from .io import sha256_file


HIDDEN_CHECKPOINT_SCHEMA_VERSION = "anchorcal-hidden-checkpoints-v1"
HIDDEN_SELECTORS = ("oracle",)


def _verify_hidden_checkpoint_artifacts(
    run_dir: str | Path,
    *,
    expected_run_id: str | None,
    required_selectors: Sequence[str],
) -> dict[str, Any]:
    """Verify the reporting-only checkpoint manifest without mutation."""

    run_dir = Path(run_dir).resolve()
    root = run_dir / "checkpoints"
    manifest_path = root / "exploratory_hidden" / "oracle_manifest.json"
    manifest = _json(manifest_path)
    run_id = manifest.get("run_id")
    _require(
        isinstance(run_id, str)
        and run_id
        and not any(character in run_id for character in "/\\\n\r"),
        "checkpoint manifest run ID is invalid",
    )
    if expected_run_id is not None:
        _require(run_id == expected_run_id, "checkpoint manifest run ID mismatch")
    _require(
        manifest.get("schema_version") == HIDDEN_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint schema is incompatible",
    )
    _require(
        manifest.get("namespace") == "exploratory_hidden_metrics",
        "hidden checkpoint namespace is invalid",
    )
    selectors = manifest.get("selectors")
    _require(
        isinstance(selectors, dict) and set(selectors) == set(HIDDEN_SELECTORS),
        "checkpoint selector set is invalid",
    )
    _require(
        set(required_selectors).issubset(HIDDEN_SELECTORS),
        "unknown required checkpoint selector",
    )
    for selector in required_selectors:
        _require(
            selectors.get(selector) is not None,
            f"required checkpoint selector is missing: {selector}",
        )

    objects_value = manifest.get("objects")
    _require(isinstance(objects_value, dict), "checkpoint object inventory is invalid")
    objects = {
        model_hash: _verify_model_object(root, run_id, model_hash, record)
        for model_hash, record in objects_value.items()
    }
    referenced: set[str] = set()
    selector_summary: dict[str, Any] = {}
    for selector, selection in selectors.items():
        if selection is None:
            selector_summary[selector] = None
            continue
        _require(isinstance(selection, dict), f"checkpoint selector is invalid: {selector}")
        model_hash = selection.get("model_hash")
        _require(model_hash in objects, f"selector references an unknown model: {selector}")
        record = objects[model_hash]
        epoch = selection.get("epoch")
        _require(
            isinstance(epoch, int)
            and not isinstance(epoch, bool)
            and epoch in record["epochs"],
            f"selector epoch is inconsistent: {selector}",
        )
        _require(
            selection.get("path") == record["relative_path"],
            f"selector path is inconsistent: {selector}",
        )
        ranking_key = selection.get("ranking_key")
        _require(
            isinstance(ranking_key, list)
            and ranking_key
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in ranking_key
            ),
            f"selector ranking key is invalid: {selector}",
        )
        _require(
            isinstance(selection.get("metadata"), dict),
            f"selector metadata is invalid: {selector}",
        )
        referenced.add(model_hash)
        selector_summary[selector] = {
            "epoch": epoch,
            "model_hash": model_hash,
            "path": record["relative_path"],
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
        }
    _require(
        referenced == set(objects),
        "checkpoint manifest contains unreferenced model objects",
    )
    _require(
        "resume" not in manifest and "completion" not in manifest,
        "hidden checkpoint manifest contains visible lifecycle state",
    )
    return {
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
        "selectors": selector_summary,
        "objects": objects,
        "resume": None,
        "completion": None,
    }


def verify_candidate_checkpoint_artifacts(
    run_dir: str | Path,
    *,
    expected_run_id: str | None = None,
    require_complete: bool = False,
    required_visible_selectors: Sequence[str] = (),
    required_hidden_selectors: Sequence[str] = (),
) -> dict[str, Any]:
    """Verify visible and reporting-only checkpoint namespaces and their union."""

    visible = verify_visible_checkpoint_artifacts(
        run_dir,
        expected_run_id=expected_run_id,
        require_complete=require_complete,
        required_selectors=required_visible_selectors,
    )
    run_id = visible["run_id"]
    hidden = _verify_hidden_checkpoint_artifacts(
        run_dir,
        expected_run_id=expected_run_id or run_id,
        required_selectors=required_hidden_selectors,
    )
    _require(hidden["run_id"] == run_id, "visible and hidden checkpoint run IDs differ")
    root = Path(run_dir).resolve() / "checkpoints"
    combined: dict[str, dict[str, Any]] = {}
    for summary in (visible, hidden):
        for model_hash, record in summary["objects"].items():
            prior = combined.get(model_hash)
            if prior is not None:
                _require(
                    all(
                        prior[key] == record[key]
                        for key in ("path", "sha256", "size_bytes")
                    ),
                    "visible and hidden manifests disagree about a shared model object",
                )
            else:
                combined[model_hash] = record
    weights_dir = root / "weights"
    _require(
        weights_dir.is_dir() and not weights_dir.is_symlink(),
        "candidate checkpoint weights directory is missing or symbolic",
    )
    actual_files = {
        path.resolve()
        for path in weights_dir.iterdir()
        if path.is_file() or path.is_symlink()
    }
    expected_files = {Path(record["path"]).resolve() for record in combined.values()}
    _require(
        actual_files == expected_files,
        "retained checkpoint weight files do not exactly match both manifests",
    )
    return {
        "run_id": run_id,
        "visible": visible,
        "hidden": hidden,
        "retained_weight_count": len(combined),
    }
