from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.candidate.metrics import (
    grouped_metrics,
    ordinary_metrics,
    prediction_payload,
    proxy_group_metrics,
)


class CandidateMetricTests(unittest.TestCase):
    def test_per_example_and_group_metrics(self) -> None:
        labels = np.array([0, 0, 1, 1])
        logits = np.array([[2, 0], [0, 2], [0, 2], [2, 0]], dtype=np.float32)
        payload = prediction_payload(["a", "b", "c", "d"], labels, logits)
        self.assertEqual(ordinary_metrics(payload)["accuracy"], 0.5)
        groups = np.array([0, 1, 2, 3])
        metrics = grouped_metrics(payload, groups)
        self.assertEqual(metrics["worst_group_accuracy"], 0.0)
        self.assertEqual(metrics["group_balanced_accuracy"], 0.5)
        proxy = proxy_group_metrics(payload, np.array([0, 1, 0, 1]))
        self.assertEqual(proxy["worst_nonempty_proxy_group_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
