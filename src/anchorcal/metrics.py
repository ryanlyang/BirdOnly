"""Class/group metrics with fail-closed Waterbirds semantics."""

from __future__ import annotations

from typing import Any

import numpy as np

HARMONIC_EPSILON = 1.0e-8

def _as_arrays(logits: Any, labels: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray(logits)
    truth = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 2 or scores.shape[1] != 2 or truth.shape != (len(scores),):
        raise ValueError("expected binary [N,2] logits and [N] labels")
    if not np.isfinite(scores).all() or not np.isin(truth, [0, 1]).all():
        raise ValueError("logits must be finite and labels binary")
    return scores, truth, scores.argmax(axis=1)


def class_balanced_mean(values: Any, labels: Any) -> float:
    value_array = np.asarray(values, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    means = []
    for label in (0, 1):
        member = label_array == label
        if not member.any():
            raise ValueError(f"class {label} is missing")
        means.append(float(value_array[member].mean()))
    return float(np.mean(means))


def harmonic_mean(
    first: float, second: float, epsilon: float = HARMONIC_EPSILON
) -> float:
    if first < 0 or second < 0:
        raise ValueError("harmonic inputs must be non-negative")
    if first == 0 or second == 0:
        return 0.0
    return float(2.0 * first * second / (first + second + epsilon))


def classification_metrics(
    logits: Any, labels: Any, places: Any | None = None
) -> dict[str, Any]:
    scores, truth, prediction = _as_arrays(logits, labels)
    correct = prediction == truth
    result: dict[str, Any] = {
        "accuracy": float(correct.mean()),
        "class_balanced_accuracy": class_balanced_mean(correct, truth),
        "class_accuracy": {
            str(label): float(correct[truth == label].mean()) for label in (0, 1)
        },
    }
    if places is not None:
        place_array = np.asarray(places, dtype=np.int64)
        if place_array.shape != truth.shape or not np.isin(place_array, [0, 1]).all():
            raise ValueError("places must be binary [N]")
        groups: dict[str, float] = {}
        for label in (0, 1):
            for place in (0, 1):
                member = (truth == label) & (place_array == place)
                if not member.any():
                    raise ValueError(f"required Waterbirds group ({label},{place}) is missing")
                groups[f"y{label}_p{place}"] = float(correct[member].mean())
        result["group_accuracy"] = groups
        result["worst_group_accuracy"] = min(groups.values())
        result["group_balanced_accuracy"] = float(np.mean(list(groups.values())))
    return result


def per_example_cross_entropy(logits: Any, labels: Any) -> np.ndarray:
    scores, truth, _ = _as_arrays(logits, labels)
    shifted = scores - scores.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=1))
    return -(shifted[np.arange(len(scores)), truth] - logsumexp)
