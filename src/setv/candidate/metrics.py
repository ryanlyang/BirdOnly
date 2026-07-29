"""Strict per-example, ordinary, proxy-group, and protected-group metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from setv.errors import DataValidationError
from setv.fusion.core import _cross_entropy_per_example


def prediction_payload(sample_ids, labels, logits) -> dict[str, np.ndarray]:
    ids = np.asarray([str(value) for value in sample_ids], dtype=np.str_)
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float32)
    if logits.shape != (len(ids), 2) or labels.shape != (len(ids),):
        raise DataValidationError("Candidate prediction arrays are misaligned")
    if len(set(ids.tolist())) != len(ids) or set(np.unique(labels)) != {0, 1}:
        raise DataValidationError("Candidate predictions require unique IDs and both classes")
    if not np.isfinite(logits).all():
        raise DataValidationError("Candidate logits contain non-finite values")
    predicted = logits.argmax(axis=1).astype(np.int64)
    losses = _cross_entropy_per_example(logits, labels).astype(np.float32)
    return {
        "sample_id": ids,
        "true_label": labels,
        "logits": logits,
        "predicted_class": predicted,
        "cross_entropy": losses,
        "correct": (predicted == labels).astype(np.uint8),
    }


def ordinary_metrics(payload: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "accuracy": float(payload["correct"].mean()),
        "loss": float(payload["cross_entropy"].mean()),
    }


def grouped_metrics(
    payload: dict[str, np.ndarray], groups: np.ndarray
) -> dict[str, Any]:
    groups = np.asarray(groups, dtype=np.int64)
    if groups.shape != payload["true_label"].shape:
        raise DataValidationError("Protected groups and predictions are misaligned")
    accuracy = {}
    for group in sorted(np.unique(groups).tolist()):
        indices = groups == group
        accuracy[str(group)] = float(payload["correct"][indices].mean())
    values = list(accuracy.values())
    return {
        "average_accuracy": float(payload["correct"].mean()),
        "group_balanced_accuracy": float(np.mean(values)),
        "worst_group_accuracy": float(np.min(values)),
        "per_group_accuracy": accuracy,
        "group_counts": {
            str(group): int((groups == group).sum())
            for group in sorted(np.unique(groups).tolist())
        },
    }


def proxy_group_metrics(
    payload: dict[str, np.ndarray], background_prediction: np.ndarray
) -> dict[str, Any]:
    background_prediction = np.asarray(background_prediction, dtype=np.int64)
    labels = payload["true_label"]
    if background_prediction.shape != labels.shape or not set(
        np.unique(background_prediction)
    ).issubset({0, 1}):
        raise DataValidationError("Background predictions are invalid")
    proxy = 2 * labels + background_prediction
    result = grouped_metrics(payload, proxy)
    return {
        "worst_nonempty_proxy_group_accuracy": result["worst_group_accuracy"],
        "per_proxy_group_accuracy": result["per_group_accuracy"],
        "proxy_group_counts": result["group_counts"],
    }
