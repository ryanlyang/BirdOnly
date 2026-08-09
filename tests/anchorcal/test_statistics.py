from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from anchorcal.statistics import (  # noqa: E402
    class_balanced_mean,
    cross_fitted_lambda_interpolation_ace,
    evaluate_anchor_scores,
    paired_class_stratified_bootstrap,
)


class AnchorStatisticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lambdas = np.linspace(0.0, 1.0, 21)

    def test_perfect_ladder_retains_endpoint_clipping_floor(self) -> None:
        metrics = evaluate_anchor_scores(self.lambdas, self.lambdas)
        self.assertAlmostEqual(metrics.kendall_tau_b, 1.0)
        self.assertAlmostEqual(metrics.spearman, 1.0)
        self.assertAlmostEqual(metrics.pair_accuracy, 1.0)
        self.assertAlmostEqual(metrics.adjacent_accuracy, 1.0)
        self.assertEqual(metrics.violations, 0)
        self.assertTrue(metrics.perfect_order)
        self.assertAlmostEqual(metrics.ace, 0.1 / 21.0, places=14)
        self.assertAlmostEqual(metrics.ace_predictions[0], 0.05)
        self.assertAlmostEqual(metrics.ace_predictions[-1], 0.95)

    def test_constant_scores_have_locked_na_and_tie_semantics(self) -> None:
        scores = np.ones_like(self.lambdas)
        metrics = evaluate_anchor_scores(scores, self.lambdas)
        self.assertIsNone(metrics.kendall_tau_b)
        self.assertIsNone(metrics.spearman)
        self.assertEqual(metrics.pair_accuracy, 0.5)
        self.assertEqual(metrics.adjacent_accuracy, 0.5)
        self.assertEqual(metrics.violations, 0)
        self.assertFalse(metrics.perfect_order)
        self.assertEqual(metrics.ace_degenerate_folds, (True, True))
        self.assertGreater(metrics.ace, 0.2)
        self.assertTrue(np.allclose(metrics.ace_predictions, 0.5))

    def test_tolerance_controls_ties_and_violations(self) -> None:
        lambdas = np.asarray([0.0, 0.5, 1.0])
        scores = np.asarray([0.0, -0.5e-10, 2.0e-10])
        metrics = evaluate_anchor_scores(scores, lambdas, tolerance=1.0e-10)
        self.assertAlmostEqual(metrics.adjacent_accuracy, 0.75)
        self.assertAlmostEqual(metrics.pair_accuracy, 2.5 / 3.0)
        self.assertEqual(metrics.violations, 0)
        self.assertTrue(metrics.perfect_order)

    def test_spearman_tolerance_uses_transitive_near_tie_groups(self) -> None:
        tolerance = 1.0e-10
        scores = np.asarray([0.0, 0.75 * tolerance, 1.5 * tolerance])
        metrics = evaluate_anchor_scores(
            scores, np.asarray([0.0, 0.5, 1.0]), tolerance=tolerance
        )
        self.assertIsNone(metrics.spearman)

    def test_degenerate_fold_uses_its_training_mean(self) -> None:
        scores = self.lambdas.copy()
        scores[::2] = 0.0
        result = cross_fitted_lambda_interpolation_ace(scores, self.lambdas)
        self.assertEqual(result.degenerate_folds, (True, False))
        self.assertTrue(np.allclose(result.predictions[1::2], 0.5))

    def test_class_balanced_mean_does_not_pool_imbalanced_images(self) -> None:
        labels = np.asarray([0, 0, 0, 1])
        values = np.asarray([[0.0], [0.0], [0.0], [1.0]])
        self.assertAlmostEqual(float(class_balanced_mean(values, labels)[0]), 0.5)

    def test_bootstrap_is_paired_stratified_and_deterministic(self) -> None:
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        increasing = np.tile(self.lambdas, (labels.size, 1))
        constant = np.ones_like(increasing)
        first = paired_class_stratified_bootstrap(
            labels,
            {"increasing": increasing, "constant": constant},
            self.lambdas,
            replicates=12,
            seed=7002,
        )
        second = paired_class_stratified_bootstrap(
            labels,
            {"increasing": increasing, "constant": constant},
            self.lambdas,
            replicates=12,
            seed=7002,
        )
        self.assertTrue(np.array_equal(first.indices, second.indices))
        for sample_indices in first.indices:
            self.assertTrue(np.array_equal(labels[sample_indices], labels))
        self.assertTrue(
            np.array_equal(
                first.criteria["increasing"].score_vectors,
                second.criteria["increasing"].score_vectors,
            )
        )
        increasing_summary = first.criteria["increasing"].summaries()
        self.assertAlmostEqual(increasing_summary["perfect_order_rate"].mean, 1.0)
        constant_summary = first.criteria["constant"].summaries()
        self.assertEqual(constant_summary["kendall_tau_b"].valid_replicates, 0)
        self.assertEqual(constant_summary["kendall_tau_b"].na_rate, 1.0)
        self.assertEqual(constant_summary["spearman"].valid_replicates, 0)
        self.assertAlmostEqual(constant_summary["pair_accuracy"].mean, 0.5)


if __name__ == "__main__":
    unittest.main()
