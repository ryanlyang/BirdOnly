"""Eight-view sanitized-background score schema."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from setv.errors import DataValidationError
from setv.experts.scores import true_class_margin


SANITIZED_SCORE_KEYS = {
    "sample_id",
    "true_label",
    "background_sanitized_mean_logits",
    "background_sanitized_true_class_margin",
    "background_sanitized_predicted_class",
    "background_sanitized_correct",
    "background_sanitized_margin_std",
}


def build_sanitized_score_payload(
    sample_ids, labels, view_logits
) -> dict[str, np.ndarray]:
    ids = np.asarray([str(value) for value in sample_ids], dtype=np.str_)
    labels = np.asarray(labels, dtype=np.int64)
    view_logits = np.asarray(view_logits, dtype=np.float32)
    if view_logits.shape != (len(ids), 8, 2):
        raise DataValidationError(
            f"Sanitized view logits must have shape [N,8,2], got {view_logits.shape}"
        )
    if labels.shape != (len(ids),) or set(np.unique(labels)) != {0, 1}:
        raise DataValidationError("Sanitized scores require aligned binary labels")
    if len(set(ids.tolist())) != len(ids):
        raise DataValidationError("Sanitized score sample IDs are not unique")
    if not np.isfinite(view_logits).all():
        raise DataValidationError("Sanitized view logits contain non-finite values")
    mean_logits = view_logits.mean(axis=1)
    predicted = mean_logits.argmax(axis=1).astype(np.int64)
    repeated_labels = np.repeat(labels, 8)
    view_margins = true_class_margin(
        view_logits.reshape(-1, 2), repeated_labels
    ).reshape(len(ids), 8)
    return {
        "sample_id": ids,
        "true_label": labels,
        "background_sanitized_mean_logits": mean_logits.astype(np.float32),
        "background_sanitized_true_class_margin": true_class_margin(
            mean_logits, labels
        ).astype(np.float32),
        "background_sanitized_predicted_class": predicted,
        "background_sanitized_correct": (predicted == labels).astype(np.uint8),
        "background_sanitized_margin_std": view_margins.std(axis=1).astype(
            np.float32
        ),
    }


def validate_sanitized_score_payload(
    payload: dict[str, np.ndarray],
    expected_manifest: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if set(payload) != SANITIZED_SCORE_KEYS:
        raise DataValidationError(
            f"Sanitized score keys must be {sorted(SANITIZED_SCORE_KEYS)}"
        )
    ids = np.asarray(payload["sample_id"]).astype(str)
    labels = np.asarray(payload["true_label"], dtype=np.int64)
    logits = np.asarray(payload["background_sanitized_mean_logits"], dtype=np.float32)
    if logits.shape != (len(ids), 2) or labels.shape != (len(ids),):
        raise DataValidationError("Sanitized mean logits and labels are misaligned")
    if set(np.unique(labels)) != {0, 1} or len(set(ids.tolist())) != len(ids):
        raise DataValidationError("Sanitized scores require unique IDs and both classes")
    if not np.isfinite(logits).all():
        raise DataValidationError("Sanitized mean logits contain non-finite values")
    predicted = logits.argmax(axis=1).astype(np.int64)
    margin = true_class_margin(logits, labels).astype(np.float32)
    expected_fields = {
        "background_sanitized_true_class_margin": margin,
        "background_sanitized_predicted_class": predicted,
        "background_sanitized_correct": (predicted == labels).astype(np.uint8),
    }
    for key, expected in expected_fields.items():
        if not np.allclose(np.asarray(payload[key]), expected, rtol=0, atol=1e-6):
            raise DataValidationError(f"Inconsistent sanitized score field: {key}")
    margin_std = np.asarray(
        payload["background_sanitized_margin_std"], dtype=np.float64
    )
    if margin_std.shape != (len(ids),) or not np.isfinite(margin_std).all():
        raise DataValidationError("Sanitized margin stability values are invalid")
    if np.any(margin_std < 0):
        raise DataValidationError("Sanitized margin standard deviations are negative")
    if expected_manifest is not None:
        if not np.array_equal(
            ids, expected_manifest["sample_id"].astype(str).to_numpy()
        ):
            raise DataValidationError("Sanitized score IDs/order do not match biased_val")
        if not np.array_equal(
            labels, expected_manifest["y"].to_numpy(dtype=np.int64)
        ):
            raise DataValidationError("Sanitized score labels do not match biased_val")
    correct = np.asarray(payload["background_sanitized_correct"])
    return {
        "sample_count": len(ids),
        "accuracy": float(correct.mean()),
        "per_class_accuracy": {
            str(class_id): float(correct[labels == class_id].mean())
            for class_id in (0, 1)
        },
        "margin": {
            "minimum": float(margin.min()),
            "median": float(np.median(margin)),
            "mean": float(margin.mean()),
            "maximum": float(margin.max()),
            "standard_deviation": float(margin.std()),
            "positive_fraction": float((margin > 0).mean()),
        },
        "view_margin_standard_deviation": {
            "minimum": float(margin_std.min()),
            "median": float(np.median(margin_std)),
            "mean": float(margin_std.mean()),
            "maximum": float(margin_std.max()),
        },
    }


def save_sanitized_scores(path: str | Path, payload: dict[str, np.ndarray]) -> None:
    validate_sanitized_score_payload(payload)
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


def load_sanitized_scores(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    validate_sanitized_score_payload(payload)
    return payload
