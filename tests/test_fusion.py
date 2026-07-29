from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.fusion.core import (
    build_fusions,
    hard_setv,
    score_candidate,
    within_class_percentile,
)


class FusionTests(unittest.TestCase):
    def test_average_tie_percentiles_are_within_class(self) -> None:
        values = np.array([1.0, 1.0, 3.0, 2.0, 4.0, 4.0])
        labels = np.array([0, 0, 0, 1, 1, 1])
        ranks = within_class_percentile(values, labels)
        np.testing.assert_allclose(ranks[:3], [0.5, 0.5, 1.0])
        np.testing.assert_allclose(ranks[3:], [1 / 3, 5 / 6, 5 / 6])

    def test_all_three_fusions_and_no_crossfit_self_scoring(self) -> None:
        labels = np.repeat([0, 1], 20)
        object_margin = np.linspace(0.2, 2.0, 40)
        background_margin = np.ones(40)
        background_margin[:10] = -1.0
        background_margin[20:30] = -0.8
        fusion = build_fusions(
            object_margin, background_margin, labels, seed=1776
        )
        self.assertTrue(fusion["hard_valid"])
        self.assertTrue(fusion["logistic"]["available"])
        self.assertEqual(fusion["logistic"]["q_logistic_repeats"].shape, (5, 40))
        self.assertEqual(fusion["logistic"]["fold_assignments"].shape, (5, 40))
        self.assertTrue(np.all(fusion["logistic"]["fold_assignments"] >= 0))
        self.assertIn(
            "does not establish robust selection utility",
            fusion["logistic"]["diagnostics"]["interpretation"],
        )
        self.assertIn(
            "score_distribution", fusion["logistic"]["diagnostics"]
        )
        self.assertIn(
            "hard_target_agreement_at_0.5",
            fusion["logistic"]["diagnostics"],
        )
        self.assertIn("raw_score_distribution", fusion["rank_diagnostics"])

        candidate_logits = np.zeros((40, 2), dtype=np.float64)
        candidate_logits[np.arange(40), labels] = 2.0
        scores = score_candidate(candidate_logits, labels, fusion)
        self.assertAlmostEqual(scores["rank"]["setv_score"], 1.0)
        self.assertAlmostEqual(scores["logistic"]["setv_score"], 1.0)
        self.assertAlmostEqual(scores["hard"]["class_balanced_accuracy"], 1.0)

    def test_degenerate_logistic_is_unavailable_and_hard_invalid(self) -> None:
        labels = np.repeat([0, 1], 10)
        object_margin = np.ones(20)
        background_margin = np.ones(20)
        background_margin[[0, 10]] = -1
        fusion = build_fusions(object_margin, background_margin, labels, seed=9)
        self.assertFalse(fusion["hard_valid"])
        self.assertFalse(fusion["logistic"]["available"])
        candidate = np.column_stack([1 - labels, labels]).astype(float)
        scores = score_candidate(candidate, labels, fusion)
        self.assertFalse(scores["hard"]["valid"])
        self.assertFalse(scores["logistic"]["available"])


if __name__ == "__main__":
    unittest.main()
