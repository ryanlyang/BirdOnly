"""Exact-fill background score schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from setv.errors import DataValidationError
from setv.experts.scores import true_class_margin


EXACT_SCORE_KEYS = {
    "sample_id",
    "true_label",
    "background_exact_logits",
    "background_exact_true_class_margin",
    "background_exact_predicted_class",
    "background_exact_correct",
}


def build_exact_score_payload(sample_ids, labels, logits) -> dict[str, np.ndarray]:
    ids = np.asarray([str(value) for value in sample_ids], dtype=np.str_)
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise DataValidationError(f"Exact logits must have shape [N, 2], got {logits.shape}")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise DataValidationError("Exact score labels must contain classes 0 and 1")
    if len(set(ids.tolist())) != len(ids):
        raise DataValidationError("Exact score sample IDs are not unique")
    predicted = logits.argmax(axis=1).astype(np.int64)
    return {
        "sample_id": ids,
        "true_label": labels,
        "background_exact_logits": logits,
        "background_exact_true_class_margin": true_class_margin(logits, labels).astype(
            np.float32
        ),
        "background_exact_predicted_class": predicted,
        "background_exact_correct": (predicted == labels).astype(np.uint8),
    }


def validate_exact_score_payload(
    payload: dict[str, np.ndarray],
    expected_manifest: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if set(payload) != EXACT_SCORE_KEYS:
        raise DataValidationError(
            f"Exact score keys must be {sorted(EXACT_SCORE_KEYS)}, got {sorted(payload)}"
        )
    rebuilt = build_exact_score_payload(
        payload["sample_id"], payload["true_label"], payload["background_exact_logits"]
    )
    for key in EXACT_SCORE_KEYS - {
        "sample_id",
        "true_label",
        "background_exact_logits",
    }:
        if not np.allclose(payload[key], rebuilt[key], rtol=0, atol=1e-6):
            raise DataValidationError(f"Inconsistent exact score field: {key}")
    if not np.isfinite(payload["background_exact_logits"]).all():
        raise DataValidationError("Exact logits contain non-finite values")
    ids = np.asarray(payload["sample_id"]).astype(str)
    labels = np.asarray(payload["true_label"], dtype=np.int64)
    if expected_manifest is not None:
        if not np.array_equal(ids, expected_manifest["sample_id"].astype(str).to_numpy()):
            raise DataValidationError("Exact score IDs/order do not match biased_val")
        if not np.array_equal(labels, expected_manifest["y"].to_numpy(dtype=np.int64)):
            raise DataValidationError("Exact score labels do not match biased_val")
    correct = np.asarray(payload["background_exact_correct"])
    margins = np.asarray(payload["background_exact_true_class_margin"], dtype=float)
    return {
        "sample_count": len(ids),
        "accuracy": float(correct.mean()),
        "per_class_accuracy": {
            str(class_id): float(correct[labels == class_id].mean())
            for class_id in (0, 1)
        },
        "margin": {
            "minimum": float(margins.min()),
            "median": float(np.median(margins)),
            "mean": float(margins.mean()),
            "maximum": float(margins.max()),
            "standard_deviation": float(margins.std()),
            "positive_fraction": float((margins > 0).mean()),
        },
    }


def save_exact_scores(path: str | Path, payload: dict[str, np.ndarray]) -> None:
    # Use the same atomic NPZ mechanism without applying the object schema.
    import os
    import tempfile

    validate_exact_score_payload(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_exact_scores(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    validate_exact_score_payload(payload)
    return payload
