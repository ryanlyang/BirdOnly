"""Expert score schemas and validation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from setv.errors import DataValidationError


OBJECT_SCORE_KEYS = {
    "sample_id",
    "true_label",
    "object_logits",
    "object_true_class_margin",
    "object_predicted_class",
    "object_correct",
}


def true_class_margin(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2:
        raise DataValidationError(f"logits must be rank 2, got {logits.shape}")
    if labels.shape != (logits.shape[0],):
        raise DataValidationError(
            f"labels shape {labels.shape} does not match logits {logits.shape}"
        )
    if np.any(labels < 0) or np.any(labels >= logits.shape[1]):
        raise DataValidationError("labels contain an out-of-range class index")
    true_logits = logits[np.arange(len(labels)), labels]
    other = logits.copy()
    other[np.arange(len(labels)), labels] = -np.inf
    return true_logits - other.max(axis=1)


def build_object_score_payload(
    sample_ids: list[str] | np.ndarray,
    labels: np.ndarray,
    logits: np.ndarray,
) -> dict[str, np.ndarray]:
    ids = np.asarray([str(value) for value in sample_ids], dtype=np.str_)
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise DataValidationError(
            f"Waterbirds object logits must have shape [N, 2], got {logits.shape}"
        )
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise DataValidationError(
            f"Waterbirds object labels must contain both classes 0 and 1, got "
            f"{sorted(np.unique(labels).tolist())}"
        )
    if len(set(ids.tolist())) != len(ids):
        raise DataValidationError("Object score sample IDs are not unique")
    margins = true_class_margin(logits, labels).astype(np.float32)
    predicted = logits.argmax(axis=1).astype(np.int64)
    correct = (predicted == labels).astype(np.uint8)
    return {
        "sample_id": ids,
        "true_label": labels,
        "object_logits": logits,
        "object_true_class_margin": margins,
        "object_predicted_class": predicted,
        "object_correct": correct,
    }


def object_sanity_warnings(summary: dict[str, Any]) -> list[str]:
    """Emit diagnostics without inventing an unspecified scientific threshold."""
    warnings: list[str] = []
    if float(summary["accuracy"]) <= 0.5:
        warnings.append("object_accuracy_not_above_binary_chance")
    if float(summary["margin"]["standard_deviation"]) <= 1e-8:
        warnings.append("object_margin_has_negligible_variation")
    if float(summary["margin"]["positive_fraction"]) in {0.0, 1.0}:
        warnings.append("object_margin_sign_is_constant")
    return warnings


def validate_object_score_payload(
    payload: dict[str, np.ndarray],
    expected_manifest: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if set(payload) != OBJECT_SCORE_KEYS:
        raise DataValidationError(
            f"Object score keys must be {sorted(OBJECT_SCORE_KEYS)}, got "
            f"{sorted(payload)}"
        )
    sample_ids = np.asarray(payload["sample_id"]).astype(str)
    labels = np.asarray(payload["true_label"], dtype=np.int64)
    logits = np.asarray(payload["object_logits"], dtype=np.float32)
    rebuilt = build_object_score_payload(sample_ids, labels, logits)
    for key in (
        "object_true_class_margin",
        "object_predicted_class",
        "object_correct",
    ):
        if not np.allclose(np.asarray(payload[key]), rebuilt[key], rtol=0, atol=1e-6):
            raise DataValidationError(f"Object score field is inconsistent: {key}")
    if not np.isfinite(logits).all():
        raise DataValidationError("Object logits contain non-finite values")
    if expected_manifest is not None:
        expected_ids = expected_manifest["sample_id"].astype(str).to_numpy()
        expected_labels = expected_manifest["y"].to_numpy(dtype=np.int64)
        if not np.array_equal(sample_ids, expected_ids):
            raise DataValidationError(
                "Object score IDs/order do not exactly match biased_val manifest"
            )
        if not np.array_equal(labels, expected_labels):
            raise DataValidationError(
                "Object score labels do not match biased_val manifest"
            )
    per_class = {}
    for class_id in sorted(np.unique(labels)):
        indices = labels == class_id
        per_class[str(int(class_id))] = float(
            np.asarray(payload["object_correct"])[indices].mean()
        )
    margins = np.asarray(payload["object_true_class_margin"], dtype=np.float64)
    return {
        "sample_count": int(len(sample_ids)),
        "accuracy": float(np.asarray(payload["object_correct"]).mean()),
        "per_class_accuracy": per_class,
        "margin": {
            "minimum": float(margins.min()),
            "median": float(np.median(margins)),
            "mean": float(margins.mean()),
            "maximum": float(margins.max()),
            "standard_deviation": float(margins.std()),
            "positive_fraction": float((margins > 0).mean()),
        },
    }


def save_object_scores(path: str | Path, payload: dict[str, np.ndarray]) -> None:
    validate_object_score_payload(payload)
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


def load_object_scores(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    validate_object_score_payload(payload)
    return payload
