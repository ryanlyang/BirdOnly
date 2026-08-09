"""Bounded scalar-temperature fitting for branch diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemperatureResult:
    temperature: float
    nll_before: float
    nll_after: float
    iterations: int


def _raw_for_temperature(value: float, low: float, high: float) -> float:
    probability = (value - low) / (high - low)
    return math.log(probability / (1.0 - probability))


def fit_temperature(
    logits,
    labels,
    *,
    minimum: float = 0.05,
    maximum: float = 20.0,
    max_iterations: int = 100,
) -> TemperatureResult:
    import torch
    import torch.nn.functional as functional

    scores = torch.as_tensor(logits, dtype=torch.float64)
    truth = torch.as_tensor(labels, dtype=torch.long, device=scores.device)
    if scores.ndim != 2 or scores.shape[1] != 2 or truth.shape != scores.shape[:1]:
        raise ValueError("expected [N,2] logits and [N] labels")
    before = float(functional.cross_entropy(scores, truth).item())
    raw = torch.tensor(
        _raw_for_temperature(1.0, minimum, maximum),
        dtype=torch.float64,
        device=scores.device,
        requires_grad=True,
    )
    optimizer = torch.optim.LBFGS(
        [raw], max_iter=max_iterations, line_search_fn="strong_wolfe"
    )
    evaluations = 0

    def closure():
        nonlocal evaluations
        optimizer.zero_grad(set_to_none=True)
        temperature = minimum + (maximum - minimum) * torch.sigmoid(raw)
        loss = functional.cross_entropy(scores / temperature, truth)
        loss.backward()
        evaluations += 1
        return loss

    optimizer.step(closure)
    temperature = float(
        (minimum + (maximum - minimum) * torch.sigmoid(raw)).detach().item()
    )
    after = float(functional.cross_entropy(scores / temperature, truth).item())
    if not minimum <= temperature <= maximum or not np.isfinite(after):
        raise RuntimeError("temperature optimization returned an invalid value")
    return TemperatureResult(temperature, before, after, evaluations)


def average_view_logits(view_logits):
    """Average raw logits before temperature fitting or softmax."""

    try:
        import torch

        if isinstance(view_logits, torch.Tensor):
            if view_logits.ndim != 3:
                raise ValueError("view logits must have shape [views,N,classes]")
            return view_logits.mean(dim=0)
    except ImportError:
        pass
    array = np.asarray(view_logits)
    if array.ndim != 3:
        raise ValueError("view logits must have shape [views,N,classes]")
    return array.mean(axis=0)


def calibration_diagnostics(
    logits, labels, temperature: float, *, bins: int = 15
) -> dict[str, object]:
    scores = np.asarray(logits, dtype=np.float64) / float(temperature)
    truth = np.asarray(labels, dtype=np.int64)
    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    prediction = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    losses = -np.log(np.maximum(probabilities[np.arange(len(truth)), truth], 1e-12))

    def ece(member: np.ndarray) -> float:
        edges = np.linspace(0.0, 1.0, bins + 1)
        total = int(member.sum())
        value = 0.0
        for index in range(bins):
            in_bin = member & (confidence > edges[index]) & (
                confidence <= edges[index + 1]
            )
            if in_bin.any():
                value += float(in_bin.sum() / total) * abs(
                    float((prediction[in_bin] == truth[in_bin]).mean())
                    - float(confidence[in_bin].mean())
                )
        return value

    per_class = {}
    for label in (0, 1):
        member = truth == label
        if not member.any():
            raise ValueError(f"calibration diagnostics require class {label}")
        per_class[str(label)] = {
            "count": int(member.sum()),
            "nll": float(losses[member].mean()),
            "ece": ece(member),
            "accuracy": float((prediction[member] == truth[member]).mean()),
        }
    return {
        "temperature": float(temperature),
        "bins": bins,
        "sample_weighted_nll": float(losses.mean()),
        "class_balanced_nll": float(
            np.mean([per_class[str(label)]["nll"] for label in (0, 1)])
        ),
        "class_balanced_ece": float(
            np.mean([per_class[str(label)]["ece"] for label in (0, 1)])
        ),
        "per_class": per_class,
    }
