from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.candidate.selectors import (
    RealisticSelectionTracker,
    is_better,
    oracle_is_better,
)


class CandidateSelectorTests(unittest.TestCase):
    def test_locked_continuous_and_hard_ties(self) -> None:
        incumbent = {
            "epoch": 4,
            "metrics": {"setv_score": 0.8, "setv_loss": 0.4},
            "ordinary": {"accuracy": 0.7, "loss": 0.5},
        }
        lower_loss = {
            "epoch": 5,
            "metrics": {"setv_score": 0.8, "setv_loss": 0.3},
            "ordinary": {"accuracy": 0.6, "loss": 0.8},
        }
        self.assertTrue(
            is_better(
                "setv.exact.rank", lower_loss, incumbent, tolerance=1e-8
            )
        )
        exact_tie_later = {**incumbent, "epoch": 6}
        self.assertFalse(
            is_better(
                "setv.exact.rank", exact_tie_later, incumbent, tolerance=1e-8
            )
        )

    def test_oracle_ties_and_unavailable_selectors(self) -> None:
        old = {
            "epoch": 2,
            "metrics": {
                "worst_group_accuracy": 0.4,
                "group_balanced_accuracy": 0.6,
                "average_accuracy": 0.7,
            },
        }
        new = {
            "epoch": 3,
            "metrics": {
                "worst_group_accuracy": 0.4,
                "group_balanced_accuracy": 0.61,
                "average_accuracy": 0.6,
            },
        }
        self.assertTrue(oracle_is_better(new, old, tolerance=1e-8))
        tracker = RealisticSelectionTracker(1e-8)
        changed = tracker.update(
            1,
            {
                "ordinary": {"available": True, "accuracy": 0.5, "loss": 0.7},
                "setv.exact.logistic": {
                    "available": False,
                    "reason": "degenerate",
                },
            },
        )
        self.assertEqual(changed, ["ordinary"])
        self.assertIn("setv.exact.logistic", tracker.unavailable)


if __name__ == "__main__":
    unittest.main()
