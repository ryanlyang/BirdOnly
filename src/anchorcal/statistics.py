"""Locked AnchorCal ordering, calibration, and bootstrap statistics.

The functions in this module operate only on already-frozen per-image anchor
outputs.  In particular, bootstrapping resamples images; it never reconstructs
branches, intersections, scales, donors, views, or the anchor family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression


DEFAULT_ANCHOR_SCORE_TOLERANCE = 1.0e-10


def _one_dimensional_finite(
    values: Sequence[float] | np.ndarray, name: str
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    if array.size < 2:
        raise ValueError(f"{name} must contain at least two values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_ladder(
    scores: Sequence[float] | np.ndarray,
    lambdas: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    score_array = _one_dimensional_finite(scores, "scores")
    lambda_array = _one_dimensional_finite(lambdas, "lambdas")
    if score_array.shape != lambda_array.shape:
        raise ValueError(
            "scores and lambdas must have the same shape; "
            f"got {score_array.shape} and {lambda_array.shape}"
        )
    if not np.all(np.diff(lambda_array) > 0.0):
        raise ValueError("lambdas must be strictly increasing")
    return score_array, lambda_array


def _validate_tolerance(tolerance: float) -> float:
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    return tolerance


def _compare(left: float, right: float, tolerance: float) -> int:
    """Compare two numbers with the locked absolute tie tolerance."""

    difference = float(left) - float(right)
    if abs(difference) <= tolerance:
        return 0
    return 1 if difference > 0.0 else -1


def _tolerant_midranks(values: np.ndarray, tolerance: float) -> np.ndarray:
    """Return deterministic midranks after transitive tolerance grouping.

    Pairwise absolute tolerance is not itself an equivalence relation.  Ranks,
    however, require equivalence classes.  We therefore sort values and take
    connected components whose adjacent gap is within tolerance; every member
    of a near-tie chain receives the same conventional midrank.
    """

    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while (
            stop < len(order)
            and values[order[stop]] - values[order[stop - 1]] <= tolerance
        ):
            stop += 1
        # Conventional one-based midrank for sorted positions [start, stop).
        midrank = ((start + 1) + stop) / 2.0
        ranks[order[start:stop]] = midrank
        start = stop
    return ranks


def tolerant_kendall_tau_b(
    scores: Sequence[float] | np.ndarray,
    lambdas: Sequence[float] | np.ndarray,
    *,
    tolerance: float = DEFAULT_ANCHOR_SCORE_TOLERANCE,
) -> float | None:
    """Compute Kendall tau-b using the locked absolute score-tie tolerance."""

    scores_array, lambdas_array = _validate_ladder(scores, lambdas)
    tolerance = _validate_tolerance(tolerance)
    concordant = 0
    discordant = 0
    score_only_ties = 0
    lambda_only_ties = 0
    for first in range(scores_array.size - 1):
        for second in range(first + 1, scores_array.size):
            score_order = _compare(
                scores_array[second], scores_array[first], tolerance
            )
            lambda_order = _compare(
                lambdas_array[second], lambdas_array[first], tolerance
            )
            if score_order == 0 and lambda_order == 0:
                continue
            if score_order == 0:
                score_only_ties += 1
            elif lambda_order == 0:
                lambda_only_ties += 1
            elif score_order == lambda_order:
                concordant += 1
            else:
                discordant += 1
    denominator = np.sqrt(
        (concordant + discordant + score_only_ties)
        * (concordant + discordant + lambda_only_ties)
    )
    if denominator == 0.0:
        return None
    return float((concordant - discordant) / denominator)


def tolerant_spearman(
    scores: Sequence[float] | np.ndarray,
    lambdas: Sequence[float] | np.ndarray,
    *,
    tolerance: float = DEFAULT_ANCHOR_SCORE_TOLERANCE,
) -> float | None:
    """Compute Spearman correlation with tolerance-aware midranks.

    A constant score vector is represented as ``None`` rather than zero.
    """

    scores_array, lambdas_array = _validate_ladder(scores, lambdas)
    tolerance = _validate_tolerance(tolerance)
    score_ranks = _tolerant_midranks(scores_array, tolerance)
    lambda_ranks = _tolerant_midranks(lambdas_array, tolerance)
    score_centered = score_ranks - np.mean(score_ranks)
    lambda_centered = lambda_ranks - np.mean(lambda_ranks)
    denominator = np.linalg.norm(score_centered) * np.linalg.norm(lambda_centered)
    if denominator <= np.finfo(np.float64).eps:
        return None
    return float(np.dot(score_centered, lambda_centered) / denominator)


def pairwise_ordering_accuracy(
    scores: Sequence[float] | np.ndarray,
    lambdas: Sequence[float] | np.ndarray,
    *,
    tolerance: float = DEFAULT_ANCHOR_SCORE_TOLERANCE,
) -> float:
    """Return all-pairs ordering accuracy, assigning 0.5 to score ties."""

    scores_array, lambdas_array = _validate_ladder(scores, lambdas)
    tolerance = _validate_tolerance(tolerance)
    credit = 0.0
    comparisons = 0
    for first in range(scores_array.size - 1):
        for second in range(first + 1, scores_array.size):
            expected = _compare(
                lambdas_array[second], lambdas_array[first], tolerance
            )
            if expected == 0:
                continue
            observed = _compare(
                scores_array[second], scores_array[first], tolerance
            )
            credit += 0.5 if observed == 0 else float(observed == expected)
            comparisons += 1
    if comparisons == 0:
        raise ValueError("lambdas contain no ordered pairs")
    return credit / comparisons


def adjacent_ordering_accuracy(
    scores: Sequence[float] | np.ndarray,
    lambdas: Sequence[float] | np.ndarray,
    *,
    tolerance: float = DEFAULT_ANCHOR_SCORE_TOLERANCE,
) -> float:
    """Return adjacent-lambda ordering accuracy, assigning 0.5 to ties."""

    scores_array, lambdas_array = _validate_ladder(scores, lambdas)
    tolerance = _validate_tolerance(tolerance)
    credit = 0.0
    for index in range(scores_array.size - 1):
        expected = _compare(
            lambdas_array[index + 1], lambdas_array[index], tolerance
        )
        observed = _compare(
            scores_array[index + 1], scores_array[index], tolerance
        )
        if expected == 0:
            raise ValueError("adjacent lambdas must be distinct")
        credit += 0.5 if observed == 0 else float(observed == expected)
    return credit / (scores_array.size - 1)


def monotonicity_violations(
    scores: Sequence[float] | np.ndarray,
    lambdas: Sequence[float] | np.ndarray,
    *,
    tolerance: float = DEFAULT_ANCHOR_SCORE_TOLERANCE,
) -> int:
    """Count globally reversed lambda pairs; ties are not violations."""

    scores_array, lambdas_array = _validate_ladder(scores, lambdas)
    tolerance = _validate_tolerance(tolerance)
    violations = 0
    for first in range(scores_array.size - 1):
        for second in range(first + 1, scores_array.size):
            expected = _compare(
                lambdas_array[second], lambdas_array[first], tolerance
            )
            observed = _compare(
                scores_array[second], scores_array[first], tolerance
            )
            if expected != 0 and observed == -expected:
                violations += 1
    return violations


def is_perfect_order(
    scores: Sequence[float] | np.ndarray,
    lambdas: Sequence[float] | np.ndarray,
    *,
    tolerance: float = DEFAULT_ANCHOR_SCORE_TOLERANCE,
) -> bool:
    """Return whether scores are nondecreasing with at least one strict rise."""

    scores_array, lambdas_array = _validate_ladder(scores, lambdas)
    tolerance = _validate_tolerance(tolerance)
    has_strict_increase = False
    for index in range(scores_array.size - 1):
        observed = _compare(
            scores_array[index + 1], scores_array[index], tolerance
        )
        if observed < 0:
            return False
        has_strict_increase = has_strict_increase or observed > 0
    return has_strict_increase


def _distinct_level_count(values: np.ndarray, tolerance: float) -> int:
    ordered = np.sort(values)
    levels: list[float] = []
    for value in ordered:
        if not levels or abs(float(value) - levels[-1]) > tolerance:
            levels.append(float(value))
    return len(levels)


@dataclass(frozen=True)
class ACEResult:
    """Cross-fitted lambda-interpolation ACE and held-out predictions."""

    ace: float
    predictions: np.ndarray
    absolute_errors: np.ndarray
    degenerate_folds: tuple[bool, bool]

    def to_dict(self) -> dict[str, object]:
        return {
            "ace": self.ace,
            "predictions": self.predictions.tolist(),
            "absolute_errors": self.absolute_errors.tolist(),
            "degenerate_folds": list(self.degenerate_folds),
        }


def cross_fitted_lambda_interpolation_ace(
    scores: Sequence[float] | np.ndarray,
    lambdas: Sequence[float] | np.ndarray,
    *,
    tolerance: float = DEFAULT_ANCHOR_SCORE_TOLERANCE,
) -> ACEResult:
    """Compute locked alternating-lambda isotonic ACE.

    Fold zero trains on even ladder indices and predicts odd indices.  Fold one
    reverses those roles.  Isotonic predictions are clipped beyond the training
    score range, deliberately retaining the protocol's endpoint error floor.
    A fold with fewer than two tolerance-distinct training scores predicts its
    mean training lambda for every held-out point.
    """

    scores_array, lambdas_array = _validate_ladder(scores, lambdas)
    tolerance = _validate_tolerance(tolerance)
    indices = np.arange(scores_array.size)
    even = indices[::2]
    odd = indices[1::2]
    if even.size == 0 or odd.size == 0:
        raise ValueError("alternating-lambda ACE requires both folds to be nonempty")
    predictions = np.empty_like(lambdas_array)
    degenerate: list[bool] = []
    for train_indices, test_indices in ((even, odd), (odd, even)):
        train_scores = scores_array[train_indices]
        train_lambdas = lambdas_array[train_indices]
        fold_is_degenerate = _distinct_level_count(train_scores, tolerance) < 2
        degenerate.append(fold_is_degenerate)
        if fold_is_degenerate:
            predictions[test_indices] = np.mean(train_lambdas)
            continue
        calibrator = IsotonicRegression(increasing=True, out_of_bounds="clip")
        calibrator.fit(train_scores, train_lambdas)
        predictions[test_indices] = calibrator.predict(scores_array[test_indices])
    absolute_errors = np.abs(predictions - lambdas_array)
    return ACEResult(
        ace=float(np.mean(absolute_errors)),
        predictions=predictions,
        absolute_errors=absolute_errors,
        degenerate_folds=(degenerate[0], degenerate[1]),
    )


@dataclass(frozen=True)
class AnchorScoreMetrics:
    """All locked point statistics for one criterion's anchor score vector."""

    ace: float
    kendall_tau_b: float | None
    spearman: float | None
    pair_accuracy: float
    adjacent_accuracy: float
    violations: int
    perfect_order: bool
    ace_predictions: np.ndarray
    ace_absolute_errors: np.ndarray
    ace_degenerate_folds: tuple[bool, bool]

    def to_dict(self) -> dict[str, object]:
        return {
            "ace": self.ace,
            "kendall_tau_b": self.kendall_tau_b,
            "spearman": self.spearman,
            "pair_accuracy": self.pair_accuracy,
            "adjacent_accuracy": self.adjacent_accuracy,
            "violations": self.violations,
            "perfect_order": self.perfect_order,
            "ace_predictions": self.ace_predictions.tolist(),
            "ace_absolute_errors": self.ace_absolute_errors.tolist(),
            "ace_degenerate_folds": list(self.ace_degenerate_folds),
        }


def evaluate_anchor_scores(
    scores: Sequence[float] | np.ndarray,
    lambdas: Sequence[float] | np.ndarray,
    *,
    tolerance: float = DEFAULT_ANCHOR_SCORE_TOLERANCE,
) -> AnchorScoreMetrics:
    """Evaluate one criterion score vector against the known lambda ladder."""

    scores_array, lambdas_array = _validate_ladder(scores, lambdas)
    ace = cross_fitted_lambda_interpolation_ace(
        scores_array, lambdas_array, tolerance=tolerance
    )
    return AnchorScoreMetrics(
        ace=ace.ace,
        kendall_tau_b=tolerant_kendall_tau_b(
            scores_array, lambdas_array, tolerance=tolerance
        ),
        spearman=tolerant_spearman(
            scores_array, lambdas_array, tolerance=tolerance
        ),
        pair_accuracy=pairwise_ordering_accuracy(
            scores_array, lambdas_array, tolerance=tolerance
        ),
        adjacent_accuracy=adjacent_ordering_accuracy(
            scores_array, lambdas_array, tolerance=tolerance
        ),
        violations=monotonicity_violations(
            scores_array, lambdas_array, tolerance=tolerance
        ),
        perfect_order=is_perfect_order(
            scores_array, lambdas_array, tolerance=tolerance
        ),
        ace_predictions=ace.predictions,
        ace_absolute_errors=ace.absolute_errors,
        ace_degenerate_folds=ace.degenerate_folds,
    )


def class_balanced_mean(
    per_image_values: np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Average images within class, then average classes with equal weight."""

    values = np.asarray(per_image_values, dtype=np.float64)
    label_array = np.asarray(labels)
    if values.ndim < 1 or values.shape[0] != label_array.size:
        raise ValueError("per_image_values first dimension must match labels")
    if label_array.ndim != 1 or label_array.size == 0:
        raise ValueError("labels must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("per_image_values must be finite")
    classes = np.unique(label_array)
    if classes.size < 2:
        raise ValueError("class-balanced averaging requires at least two classes")
    class_means = [np.mean(values[label_array == label], axis=0) for label in classes]
    return np.mean(np.stack(class_means, axis=0), axis=0)


def generate_stratified_bootstrap_indices(
    labels: Sequence[int] | np.ndarray,
    *,
    replicates: int = 200,
    seed: int = 7002,
) -> np.ndarray:
    """Generate one reusable paired, within-class bootstrap index matrix."""

    label_array = np.asarray(labels)
    if label_array.ndim != 1 or label_array.size == 0:
        raise ValueError("labels must be a nonempty one-dimensional array")
    if not isinstance(replicates, (int, np.integer)) or int(replicates) <= 0:
        raise ValueError("replicates must be a positive integer")
    classes = np.unique(label_array)
    if classes.size < 2:
        raise ValueError("stratified bootstrap requires at least two classes")
    generator = np.random.default_rng(int(seed))
    result = np.empty((int(replicates), label_array.size), dtype=np.int64)
    for replicate in range(int(replicates)):
        for label in classes:
            positions = np.flatnonzero(label_array == label)
            result[replicate, positions] = generator.choice(
                positions, size=positions.size, replace=True
            )
    return result


@dataclass(frozen=True)
class ReplicateSummary:
    mean: float | None
    standard_deviation: float | None
    percentile_95_low: float | None
    percentile_95_high: float | None
    valid_replicates: int
    na_rate: float

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "percentile_95_low": self.percentile_95_low,
            "percentile_95_high": self.percentile_95_high,
            "valid_replicates": self.valid_replicates,
            "na_rate": self.na_rate,
        }


def summarize_replicates(values: Sequence[float] | np.ndarray) -> ReplicateSummary:
    """Summarize finite replicates, retaining NA counts for correlations."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("replicate values must be a nonempty one-dimensional array")
    finite = array[np.isfinite(array)]
    valid = int(finite.size)
    if valid == 0:
        return ReplicateSummary(None, None, None, None, 0, 1.0)
    standard_deviation = (
        float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    )
    low, high = np.percentile(finite, [2.5, 97.5])
    return ReplicateSummary(
        mean=float(np.mean(finite)),
        standard_deviation=standard_deviation,
        percentile_95_low=float(low),
        percentile_95_high=float(high),
        valid_replicates=valid,
        na_rate=float(1.0 - valid / array.size),
    )


@dataclass(frozen=True)
class BootstrapCriterionResult:
    score_vectors: np.ndarray
    ace: np.ndarray
    kendall_tau_b: np.ndarray
    spearman: np.ndarray
    pair_accuracy: np.ndarray
    adjacent_accuracy: np.ndarray
    violations: np.ndarray
    perfect_order: np.ndarray

    def summaries(self) -> dict[str, ReplicateSummary]:
        return {
            "ace": summarize_replicates(self.ace),
            "kendall_tau_b": summarize_replicates(self.kendall_tau_b),
            "spearman": summarize_replicates(self.spearman),
            "pair_accuracy": summarize_replicates(self.pair_accuracy),
            "adjacent_accuracy": summarize_replicates(self.adjacent_accuracy),
            "violations": summarize_replicates(self.violations),
            "perfect_order_rate": summarize_replicates(
                self.perfect_order.astype(np.float64)
            ),
        }

    @property
    def ace_standard_deviation(self) -> float:
        summary = summarize_replicates(self.ace)
        if summary.standard_deviation is None:
            raise ValueError("ACE has no valid bootstrap replicates")
        return summary.standard_deviation


@dataclass(frozen=True)
class PairedBootstrapResult:
    seed: int
    indices: np.ndarray
    criteria: dict[str, BootstrapCriterionResult]


ScoreTransform = Callable[[np.ndarray], np.ndarray]


def paired_class_stratified_bootstrap(
    labels: Sequence[int] | np.ndarray,
    criterion_per_image_values: Mapping[str, np.ndarray],
    lambdas: Sequence[float] | np.ndarray,
    *,
    replicates: int = 200,
    seed: int = 7002,
    tolerance: float = DEFAULT_ANCHOR_SCORE_TOLERANCE,
    score_transforms: Mapping[str, ScoreTransform] | None = None,
) -> PairedBootstrapResult:
    """Bootstrap fixed per-image outputs with paired within-class indices.

    Each criterion array has shape ``(images, lambdas)`` and represents its
    fixed per-image contribution.  Within each replicate, values are averaged
    within class and then across classes.  An optional deterministic transform
    (for example ``H(1, r)``) is then applied to the score vector.  Exactly one
    index matrix is generated and reused for every criterion and lambda.
    """

    lambda_array = _one_dimensional_finite(lambdas, "lambdas")
    if not np.all(np.diff(lambda_array) > 0.0):
        raise ValueError("lambdas must be strictly increasing")
    label_array = np.asarray(labels)
    if not criterion_per_image_values:
        raise ValueError("at least one criterion is required")
    arrays: dict[str, np.ndarray] = {}
    for name, value in criterion_per_image_values.items():
        array = np.asarray(value, dtype=np.float64)
        expected = (label_array.size, lambda_array.size)
        if array.shape != expected:
            raise ValueError(
                f"criterion {name!r} has shape {array.shape}; expected {expected}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"criterion {name!r} contains non-finite values")
        arrays[name] = array
    transforms = dict(score_transforms or {})
    unknown_transforms = set(transforms) - set(arrays)
    if unknown_transforms:
        raise ValueError(
            f"score transforms supplied for unknown criteria: {sorted(unknown_transforms)}"
        )
    indices = generate_stratified_bootstrap_indices(
        label_array, replicates=replicates, seed=seed
    )
    criterion_results: dict[str, BootstrapCriterionResult] = {}
    for name, values in arrays.items():
        score_vectors = np.empty((int(replicates), lambda_array.size), dtype=np.float64)
        ace_values = np.empty(int(replicates), dtype=np.float64)
        kendall_values = np.full(int(replicates), np.nan, dtype=np.float64)
        spearman_values = np.full(int(replicates), np.nan, dtype=np.float64)
        pair_values = np.empty(int(replicates), dtype=np.float64)
        adjacent_values = np.empty(int(replicates), dtype=np.float64)
        violation_values = np.empty(int(replicates), dtype=np.int64)
        perfect_values = np.empty(int(replicates), dtype=np.bool_)
        transform = transforms.get(name)
        for replicate, sample_indices in enumerate(indices):
            scores = np.asarray(
                class_balanced_mean(values[sample_indices], label_array[sample_indices]),
                dtype=np.float64,
            )
            if transform is not None:
                scores = np.asarray(transform(scores), dtype=np.float64)
            if scores.shape != lambda_array.shape or not np.all(np.isfinite(scores)):
                raise ValueError(
                    f"transform for criterion {name!r} returned invalid scores"
                )
            score_vectors[replicate] = scores
            metrics = evaluate_anchor_scores(
                scores, lambda_array, tolerance=tolerance
            )
            ace_values[replicate] = metrics.ace
            if metrics.kendall_tau_b is not None:
                kendall_values[replicate] = metrics.kendall_tau_b
            if metrics.spearman is not None:
                spearman_values[replicate] = metrics.spearman
            pair_values[replicate] = metrics.pair_accuracy
            adjacent_values[replicate] = metrics.adjacent_accuracy
            violation_values[replicate] = metrics.violations
            perfect_values[replicate] = metrics.perfect_order
        criterion_results[name] = BootstrapCriterionResult(
            score_vectors=score_vectors,
            ace=ace_values,
            kendall_tau_b=kendall_values,
            spearman=spearman_values,
            pair_accuracy=pair_values,
            adjacent_accuracy=adjacent_values,
            violations=violation_values,
            perfect_order=perfect_values,
        )
    return PairedBootstrapResult(
        seed=int(seed), indices=indices, criteria=criterion_results
    )
