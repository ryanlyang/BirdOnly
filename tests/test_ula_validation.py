from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.ula.analysis import (
    kendall_tau_b,
    pairwise_ranking_accuracy,
    spearman_correlation,
)
from setv.ula.validation import select_ula_epoch, ula_validation_metrics


class ULAValidationTests(unittest.TestCase):
    def test_official_nonempty_proxy_group_formula(self) -> None:
        correct = np.asarray([1, 0, 1, 1, 0, 0], dtype=float)
        proxy = np.asarray([0, 0, 1, 1, 1, 1])
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        metrics = ula_validation_metrics(correct, proxy, labels)
        # Nonempty groups: (0,0)=.5, (1,0)=1, (1,1)=1/3.
        self.assertAlmostEqual(metrics["u_balanced_accuracy"], (0.5 + 1 + 1 / 3) / 3)
        self.assertAlmostEqual(metrics["u_worst_accuracy"], 1 / 3)
        self.assertEqual(metrics["nonempty_proxy_group_count"], 3)

    def test_selector_uses_balanced_then_locked_ties(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        proxy = np.asarray([0, 1, 0, 1])
        correctness = np.asarray(
            [
                [1, 0, 0, 1],
                [1, 1, 1, 0],
                [1, 1, 1, 0],
            ]
        )
        result = select_ula_epoch(
            correctness,
            proxy,
            labels,
            ordinary_accuracy=np.asarray([0.5, 0.75, 0.8]),
            ordinary_loss=np.asarray([0.8, 0.6, 0.7]),
        )
        self.assertEqual(result["best"]["epoch"], 3)
        self.assertEqual(result["label"], "uLA-style")

    def test_ranking_diagnostics_handle_ties(self) -> None:
        oracle = np.asarray([0.1, 0.2, 0.2, 0.4])
        perfect = np.asarray([1.0, 2.0, 2.0, 4.0])
        self.assertAlmostEqual(spearman_correlation(perfect, oracle), 1.0)
        self.assertAlmostEqual(kendall_tau_b(perfect, oracle), 1.0)
        pairwise = pairwise_ranking_accuracy(perfect, oracle)
        self.assertEqual(pairwise["oracle_ties_excluded"], 1)
        self.assertAlmostEqual(pairwise["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
