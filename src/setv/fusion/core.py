"""Hard, rank, and repeated cross-fitted logistic fusion."""

from __future__ import annotations

from typing import Any

import numpy as np

from setv.errors import DataValidationError
from setv.utils.seeds import derive_seed


def within_class_percentile(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """One-indexed average tie ranks divided by class count."""
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if values.shape != labels.shape:
        raise DataValidationError("Percentile values and labels must have equal shape")
    result = np.empty_like(values)
    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        class_values = values[indices]
        order = np.argsort(class_values, kind="mergesort")
        sorted_values = class_values[order]
        ranks = np.empty(len(indices), dtype=np.float64)
        start = 0
        while start < len(indices):
            end = start + 1
            while end < len(indices) and sorted_values[end] == sorted_values[start]:
                end += 1
            average_one_indexed_rank = ((start + 1) + end) / 2.0
            ranks[order[start:end]] = average_one_indexed_rank / len(indices)
            start = end
        result[indices] = ranks
    return result


def hard_target(object_margin: np.ndarray, background_margin: np.ndarray) -> np.ndarray:
    return ((np.asarray(object_margin) > 0) & (np.asarray(background_margin) < 0)).astype(
        np.uint8
    )


def distribution_summary(values: np.ndarray) -> dict[str, float]:
    """Return a compact, JSON-safe summary for a one-dimensional score."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise DataValidationError("A score distribution must be a nonempty vector")
    if not np.isfinite(values).all():
        raise DataValidationError("A score distribution contains non-finite values")
    quantiles = np.quantile(values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {
        "minimum": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "maximum": float(quantiles[6]),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
    }


def rank_fusion(
    object_margin: np.ndarray,
    background_margin: np.ndarray,
    labels: np.ndarray,
    *,
    clip_minimum: float = 1e-3,
) -> dict[str, np.ndarray]:
    object_rank = within_class_percentile(object_margin, labels)
    background_rank = within_class_percentile(background_margin, labels)
    raw = object_rank * (1.0 - background_rank)
    clipped = np.clip(raw, clip_minimum, 1.0)
    weighting_percentile = np.clip(
        within_class_percentile(clipped, labels), clip_minimum, 1.0
    )
    return {
        "object_percentile": object_rank,
        "background_percentile": background_rank,
        "q_rank_raw": raw,
        "q_rank": clipped,
        "u_rank": weighting_percentile,
    }


def _fold_count(target: np.ndarray, requested: int) -> int | None:
    positives = int(target.sum())
    negatives = int(len(target) - positives)
    # The plan flags fewer than ten positives. Requiring two examples of each
    # target per held-out fold makes that rule operational and deterministic.
    feasible = min(requested, positives // 2, negatives // 2)
    return feasible if feasible >= 2 else None


def repeated_cross_fitted_logistic(
    features: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    n_folds: int = 5,
    n_repeats: int = 5,
) -> dict[str, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            average_precision_score,
            brier_score_loss,
            roc_auc_score,
        )
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise DataValidationError(
            "scikit-learn is required for repeated logistic fusion"
        ) from exc

    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.int64)
    if features.shape != (len(target), 4):
        raise DataValidationError(f"Logistic features must have shape [N,4], got {features.shape}")
    if not np.isfinite(features).all():
        raise DataValidationError("Logistic fusion features contain non-finite values")
    folds = _fold_count(target, n_folds)
    counts = {"negative": int((target == 0).sum()), "positive": int(target.sum())}
    if folds is None:
        return {
            "available": False,
            "reason": "target cannot support at least two folds with two examples per target per fold",
            "target_counts": counts,
        }

    predictions = np.empty((n_repeats, len(target)), dtype=np.float64)
    assignments = np.full((n_repeats, len(target)), -1, dtype=np.int16)
    stratification_modes = []
    joint = np.asarray([f"{y}:{t}" for y, t in zip(labels, target)])
    for repeat in range(n_repeats):
        if min(np.unique(joint, return_counts=True)[1]) >= folds:
            strata = joint
            mode = "true_class_and_hard_target"
        else:
            strata = target
            mode = "hard_target"
        stratification_modes.append(mode)
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=derive_seed(seed, f"logistic_repeat={repeat}"),
        )
        seen = np.zeros(len(target), dtype=np.uint8)
        for fold, (train_indices, heldout_indices) in enumerate(
            splitter.split(features, strata)
        ):
            if np.intersect1d(train_indices, heldout_indices).size:
                raise DataValidationError("Cross-fitting train/heldout overlap detected")
            scaler = StandardScaler()
            train_features = scaler.fit_transform(features[train_indices])
            heldout_features = scaler.transform(features[heldout_indices])
            model = LogisticRegression(
                penalty="l2",
                C=1.0,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=1000,
                random_state=derive_seed(seed, f"logistic_repeat={repeat}:fold={fold}"),
            )
            model.fit(train_features, target[train_indices])
            predictions[repeat, heldout_indices] = model.predict_proba(heldout_features)[:, 1]
            assignments[repeat, heldout_indices] = fold
            seen[heldout_indices] += 1
        if not np.all(seen == 1):
            raise DataValidationError("Each example must receive exactly one OOF score per repeat")
    averaged = predictions.mean(axis=0)
    return {
        "available": True,
        "n_folds_used": folds,
        "n_repeats": n_repeats,
        "target_counts": counts,
        "stratification_modes": stratification_modes,
        "q_logistic_repeats": predictions,
        "q_logistic": averaged,
        "fold_assignments": assignments,
        "diagnostics": {
            "roc_auc_against_hard_target": float(roc_auc_score(target, averaged)),
            "pr_auc_against_hard_target": float(
                average_precision_score(target, averaged)
            ),
            "brier_score_against_hard_target": float(
                brier_score_loss(target, averaged)
            ),
            "score_distribution": distribution_summary(averaged),
            "hard_target_agreement_at_0.5": float(
                ((averaged >= 0.5).astype(np.uint8) == target).mean()
            ),
            "interpretation": (
                "Implementation diagnostic only: the target is defined from the "
                "same expert margins and does not establish robust selection utility."
            ),
        },
    }


def build_fusions(
    object_margin: np.ndarray,
    background_margin: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    min_hard_examples_per_class: int = 5,
    clip_minimum: float = 1e-3,
) -> dict[str, Any]:
    object_margin = np.asarray(object_margin, dtype=np.float64)
    background_margin = np.asarray(background_margin, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    target = hard_target(object_margin, background_margin)
    hard_counts = {
        str(class_id): int(target[labels == class_id].sum()) for class_id in (0, 1)
    }
    rank = rank_fusion(
        object_margin, background_margin, labels, clip_minimum=clip_minimum
    )
    features = np.stack(
        [
            object_margin,
            background_margin,
            object_margin - background_margin,
            object_margin * background_margin,
        ],
        axis=1,
    )
    logistic = repeated_cross_fitted_logistic(
        features, target, labels, seed=seed, n_folds=5, n_repeats=5
    )
    if logistic["available"]:
        logistic["u_logistic"] = np.clip(
            within_class_percentile(logistic["q_logistic"], labels),
            clip_minimum,
            1.0,
        )
    return {
        "hard_target": target,
        "hard_counts_per_class": hard_counts,
        "hard_valid": all(
            count >= min_hard_examples_per_class for count in hard_counts.values()
        ),
        "rank": rank,
        "rank_diagnostics": {
            "raw_score_distribution": distribution_summary(rank["q_rank_raw"]),
            "clipped_score_distribution": distribution_summary(rank["q_rank"]),
            "weighting_percentile_distribution": distribution_summary(rank["u_rank"]),
        },
        "logistic": logistic,
    }


def _cross_entropy_per_example(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=1))
    return logsumexp - shifted[np.arange(len(labels)), labels]


def continuous_setv(
    candidate_logits: np.ndarray,
    labels: np.ndarray,
    weighting_percentile: np.ndarray,
    *,
    alphas=(0.5, 1.0, 2.0, 4.0),
    ess_warning_threshold: float = 10.0,
) -> dict[str, Any]:
    logits = np.asarray(candidate_logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    u = np.asarray(weighting_percentile, dtype=np.float64)
    if logits.shape != (len(labels), 2) or u.shape != labels.shape:
        raise DataValidationError("Candidate logits/labels/fusion weights are misaligned")
    if not np.isfinite(logits).all() or not np.isfinite(u).all():
        raise DataValidationError("Candidate logits or fusion weights are non-finite")
    correct = (logits.argmax(axis=1) == labels).astype(np.float64)
    ce = _cross_entropy_per_example(logits, labels)
    alpha_accuracy = {}
    alpha_loss = {}
    ess = {}
    warnings = []
    for alpha in alphas:
        class_accuracy = []
        class_loss = []
        ess[str(alpha)] = {}
        for class_id in (0, 1):
            indices = labels == class_id
            weights = np.power(u[indices], alpha)
            weights /= weights.sum()
            class_accuracy.append(float(np.dot(weights, correct[indices])))
            class_loss.append(float(np.dot(weights, ce[indices])))
            effective = float(1.0 / np.square(weights).sum())
            ess[str(alpha)][str(class_id)] = effective
            if alpha == 4.0 and effective < ess_warning_threshold:
                warnings.append(
                    f"ESS_c(alpha=4,class={class_id})={effective:.6f}<"
                    f"{ess_warning_threshold}"
                )
        alpha_accuracy[str(alpha)] = float(np.mean(class_accuracy))
        alpha_loss[str(alpha)] = float(np.mean(class_loss))
    return {
        "setv_score": float(np.mean(list(alpha_accuracy.values()))),
        "setv_loss": float(np.mean(list(alpha_loss.values()))),
        "alpha_accuracy": alpha_accuracy,
        "alpha_loss": alpha_loss,
        "effective_sample_size": ess,
        "warnings": warnings,
    }


def hard_setv(
    candidate_logits: np.ndarray,
    labels: np.ndarray,
    target: np.ndarray,
    *,
    min_examples_per_class: int = 5,
) -> dict[str, Any]:
    logits = np.asarray(candidate_logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    target = np.asarray(target, dtype=bool)
    counts = {str(class_id): int(target[labels == class_id].sum()) for class_id in (0, 1)}
    if any(count < min_examples_per_class for count in counts.values()):
        return {"valid": False, "counts_per_class": counts}
    correct = logits.argmax(axis=1) == labels
    ce = _cross_entropy_per_example(logits, labels)
    return {
        "valid": True,
        "counts_per_class": counts,
        "class_balanced_accuracy": float(
            np.mean([correct[(labels == class_id) & target].mean() for class_id in (0, 1)])
        ),
        "class_balanced_cross_entropy": float(
            np.mean([ce[(labels == class_id) & target].mean() for class_id in (0, 1)])
        ),
    }


def score_candidate(
    candidate_logits: np.ndarray,
    labels: np.ndarray,
    fusions: dict[str, Any],
) -> dict[str, Any]:
    output = {
        "hard": hard_setv(candidate_logits, labels, fusions["hard_target"]),
        "rank": continuous_setv(
            candidate_logits, labels, fusions["rank"]["u_rank"]
        ),
        "logistic": {"available": bool(fusions["logistic"]["available"])},
    }
    if fusions["logistic"]["available"]:
        output["logistic"].update(
            continuous_setv(
                candidate_logits, labels, fusions["logistic"]["u_logistic"]
            )
        )
    return output
