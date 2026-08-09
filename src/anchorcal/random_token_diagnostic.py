"""Repeated-randomization construction and audits for the background branch."""

from __future__ import annotations

from typing import Any

import numpy as np

from .audits import stratified_bootstrap_interval
from .errors import AuditFailure
from .metrics import class_balanced_mean
from .seeds import stable_seed, stateless_rng


DIAGNOSTIC_MODES = ("pooled", "per_draw_class_balanced")


def require_repeated_random_token_collapse(
    summary: dict[str, Any],
    *,
    maximum_balanced_accuracy: float = 0.53,
    chance_accuracy: float = 0.50,
) -> dict[str, Any]:
    """Apply the hard gate to the aggregate repeated-randomization result.

    A single random token assignment has a nonzero false-rejection rate.  The
    production protocol therefore applies the prespecified point and bootstrap
    requirements to per-image correctness averaged across fixed randomizations.
    Individual-repeat results remain in ``summary`` for auditability.
    """

    aggregate = summary.get("aggregate_per_image_bootstrap_95")
    if not isinstance(aggregate, dict):
        raise ValueError("random-token summary is missing the aggregate interval")
    point = float(aggregate["point"])
    lower = float(aggregate["lower"])
    upper = float(aggregate["upper"])
    values = np.asarray([point, lower, upper], dtype=np.float64)
    if not np.isfinite(values).all():
        raise AuditFailure("random-token aggregate contains non-finite values")
    passed = bool(
        point <= maximum_balanced_accuracy
        and lower <= chance_accuracy <= upper
    )
    gate = {
        "status": "passed" if passed else "failed",
        "scope": "aggregate_per_image_correctness_across_fixed_repeats",
        "maximum_balanced_accuracy": float(maximum_balanced_accuracy),
        "chance_accuracy": float(chance_accuracy),
        "requirements": [
            "aggregate_point_estimate_lte_maximum",
            "aggregate_bootstrap_95_contains_chance",
        ],
    }
    if not passed:
        raise AuditFailure(
            "repeated random-token leakage gate failed: "
            f"aggregate balanced accuracy={point:.4f}, "
            f"95% CI=[{lower:.4f},{upper:.4f}], "
            f"maximum={maximum_balanced_accuracy:.4f}"
        )
    return {**summary, "hard_gate": gate}


def diagnostic_seeds(base_seed: int, repeats: int) -> list[int]:
    """Keep the original audit seed first, followed by fixed derived seeds."""

    if repeats < 1:
        raise ValueError("diagnostic repeats must be positive")
    return [int(base_seed)] + [
        stable_seed(base_seed, repeat, "random_token_diagnostic_repeat")
        for repeat in range(1, repeats)
    ]


def random_token_draw_indices(
    recipient_img_ids: np.ndarray,
    *,
    patches_per_class: int,
    token_budget: int,
    seed: int,
    mode: str,
) -> np.ndarray:
    """Draw pooled or exactly per-draw class-balanced source-patch indices."""

    image_ids = np.asarray(recipient_img_ids, dtype=np.int64)
    if patches_per_class < token_budget:
        raise ValueError("each source-class pool must cover the token budget")
    if mode not in DIAGNOSTIC_MODES:
        raise ValueError(f"unknown random-token diagnostic mode: {mode}")
    if mode == "per_draw_class_balanced" and token_budget % 2:
        raise ValueError("per-draw class balance requires an even token budget")
    output = np.empty((len(image_ids), token_budget), dtype=np.int64)
    for row, img_id in enumerate(image_ids):
        if mode == "pooled":
            # This exactly reproduces the production audit for its original
            # seed; derived seeds vary only the random-token realization.
            rng = stateless_rng(seed, int(img_id), "random_token_audit")
            output[row] = rng.choice(
                2 * patches_per_class,
                token_budget,
                replace=False,
            )
        else:
            rng = stateless_rng(
                seed,
                int(img_id),
                "random_token_audit_per_draw_class_balanced",
            )
            half = token_budget // 2
            selected = np.concatenate(
                [
                    rng.choice(patches_per_class, half, replace=False),
                    patches_per_class
                    + rng.choice(patches_per_class, half, replace=False),
                ]
            )
            output[row] = selected[rng.permutation(token_budget)]
    return output


def summarize_random_token_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    seeds: list[int],
    *,
    bootstrap_replicates: int = 2000,
    permutation_replicates: int = 2000,
    summary_seed: int = 8003,
) -> dict[str, Any]:
    """Summarize repeated predictions without applying or changing a gate."""

    prediction_array = np.asarray(predictions, dtype=np.int64)
    label_array = np.asarray(labels, dtype=np.int64)
    if prediction_array.shape != (len(seeds), len(label_array)):
        raise ValueError("prediction matrix shape disagrees with seeds/labels")
    if set(np.unique(label_array)) != {0, 1}:
        raise ValueError("both recipient classes are required")
    repeats: list[dict[str, Any]] = []
    correct_matrix = prediction_array == label_array[None, :]
    points: list[float] = []
    for repeat, (seed, correct, prediction) in enumerate(
        zip(seeds, correct_matrix, prediction_array, strict=True)
    ):
        interval = stratified_bootstrap_interval(
            correct.astype(np.float64),
            label_array,
            replicates=bootstrap_replicates,
            seed=stable_seed(summary_seed, repeat, "random_token_diagnostic_bootstrap"),
        )
        points.append(interval.point)
        repeats.append(
            {
                "repeat": repeat,
                "seed": int(seed),
                "balanced_accuracy": interval.point,
                "bootstrap_95_lower": interval.lower,
                "bootstrap_95_upper": interval.upper,
                "original_gate_pass": bool(
                    interval.point <= 0.53
                    and interval.lower <= 0.5 <= interval.upper
                ),
                "class_accuracy": {
                    str(label): float(correct[label_array == label].mean())
                    for label in (0, 1)
                },
                "prediction_1_rate_by_recipient_class": {
                    str(label): float(prediction[label_array == label].mean())
                    for label in (0, 1)
                },
                "prediction_1_rate_overall": float(prediction.mean()),
            }
        )
    per_image_correctness = correct_matrix.astype(np.float64).mean(axis=0)
    aggregate_interval = stratified_bootstrap_interval(
        per_image_correctness,
        label_array,
        replicates=bootstrap_replicates,
        seed=stable_seed(summary_seed, "random_token_diagnostic_aggregate_bootstrap"),
    )
    observed = float(np.mean(points))
    rng = stateless_rng(summary_seed, "random_token_diagnostic_permutation")
    null = np.empty(permutation_replicates, dtype=np.float64)
    for replicate in range(permutation_replicates):
        permuted = rng.permutation(label_array)
        per_image = (prediction_array == permuted[None, :]).mean(axis=0)
        null[replicate] = class_balanced_mean(per_image, permuted)
    permutation_p = float(
        (1 + np.count_nonzero(null >= observed)) / (permutation_replicates + 1)
    )
    return {
        "repeat_count": len(seeds),
        "recipient_count": len(label_array),
        "mean_balanced_accuracy": observed,
        "standard_deviation_across_repeats": float(np.std(points, ddof=1))
        if len(points) > 1
        else 0.0,
        "minimum_balanced_accuracy": float(np.min(points)),
        "maximum_balanced_accuracy": float(np.max(points)),
        "original_gate_pass_count": int(
            sum(bool(item["original_gate_pass"]) for item in repeats)
        ),
        "aggregate_per_image_bootstrap_95": {
            "point": aggregate_interval.point,
            "lower": aggregate_interval.lower,
            "upper": aggregate_interval.upper,
            "replicates": aggregate_interval.replicates,
        },
        "one_sided_recipient_label_permutation_p": permutation_p,
        "permutation_replicates": permutation_replicates,
        "repeats": repeats,
    }
