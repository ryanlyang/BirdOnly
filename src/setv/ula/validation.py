"""Faithful NumPy form of the official uLA validation criterion."""

from __future__ import annotations

from typing import Any

import numpy as np

from setv.errors import DataValidationError


def ula_validation_metrics(
    candidate_correct: np.ndarray,
    proxy_predictions: np.ndarray,
    true_labels: np.ndarray,
) -> dict[str, Any]:
    """Average accuracy over nonempty ``(proxy_prediction, true_label)`` strata.

    This matches ``bias_unsuper_balanced_accuracy`` at official commit
    5867fb6: stack proxy predictions with target labels, compute each nonempty
    stratum's accuracy, average the strata, and separately retain the minimum.
    """
    correct = np.asarray(candidate_correct, dtype=np.float64)
    proxy = np.asarray(proxy_predictions, dtype=np.int64)
    labels = np.asarray(true_labels, dtype=np.int64)
    if correct.ndim != 1 or proxy.shape != correct.shape or labels.shape != correct.shape:
        raise DataValidationError("uLA correctness, proxy predictions, and labels misalign")
    if not np.isfinite(correct).all() or np.any((correct < 0) | (correct > 1)):
        raise DataValidationError("uLA correctness must lie in [0,1]")
    if not set(np.unique(proxy)).issubset({0, 1}) or not set(
        np.unique(labels)
    ).issubset({0, 1}):
        raise DataValidationError("uLA Waterbirds proxy and labels must be binary")
    per_group: dict[str, float] = {}
    counts: dict[str, int] = {}
    for proxy_value in (0, 1):
        for target_value in (0, 1):
            key = f"proxy={proxy_value},target={target_value}"
            indices = (proxy == proxy_value) & (labels == target_value)
            counts[key] = int(indices.sum())
            if indices.any():
                per_group[key] = float(correct[indices].mean())
    if not per_group:
        raise DataValidationError("uLA has no nonempty proxy groups")
    return {
        "u_balanced_accuracy": float(np.mean(list(per_group.values()))),
        "u_worst_accuracy": float(np.min(list(per_group.values()))),
        "nonempty_proxy_group_count": len(per_group),
        "proxy_group_accuracy": per_group,
        "proxy_group_count": counts,
    }


def _decision(candidate: float, incumbent: float, tolerance: float) -> int:
    difference = float(candidate) - float(incumbent)
    if abs(difference) <= tolerance:
        return 0
    return 1 if difference > 0 else -1


def select_ula_epoch(
    candidate_correct: np.ndarray,
    proxy_predictions: np.ndarray,
    true_labels: np.ndarray,
    ordinary_accuracy: np.ndarray,
    ordinary_loss: np.ndarray,
    *,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    correct = np.asarray(candidate_correct)
    if correct.ndim != 2:
        raise DataValidationError("uLA candidate correctness must have shape [epochs,N]")
    ordinary_accuracy = np.asarray(ordinary_accuracy, dtype=np.float64)
    ordinary_loss = np.asarray(ordinary_loss, dtype=np.float64)
    if ordinary_accuracy.shape != (len(correct),) or ordinary_loss.shape != (
        len(correct),
    ):
        raise DataValidationError("uLA ordinary metric curves are misaligned")
    curve = [
        {
            "epoch": epoch + 1,
            **ula_validation_metrics(row, proxy_predictions, true_labels),
        }
        for epoch, row in enumerate(correct)
    ]
    best = None
    for index, metrics in enumerate(curve):
        item = {
            "epoch": index + 1,
            "metrics": metrics,
            "ordinary": {
                "accuracy": float(ordinary_accuracy[index]),
                "loss": float(ordinary_loss[index]),
            },
        }
        if best is None:
            best = item
            continue
        comparisons = (
            (metrics["u_balanced_accuracy"], best["metrics"]["u_balanced_accuracy"], True),
            (metrics["u_worst_accuracy"], best["metrics"]["u_worst_accuracy"], True),
            (ordinary_accuracy[index], best["ordinary"]["accuracy"], True),
            (ordinary_loss[index], best["ordinary"]["loss"], False),
        )
        replace = False
        for candidate, incumbent, maximize in comparisons:
            outcome = _decision(candidate, incumbent, tolerance)
            if outcome:
                replace = outcome > 0 if maximize else outcome < 0
                break
        else:
            replace = item["epoch"] < best["epoch"]
        if replace:
            best = item
    return {
        "label": "uLA-style",
        "official_primary_metric": "u_balanced_accuracy",
        "tie_break": [
            "u_worst_accuracy",
            "ordinary_accuracy",
            "ordinary_loss",
            "earlier_epoch",
        ],
        "best": best,
        "curve": curve,
    }
