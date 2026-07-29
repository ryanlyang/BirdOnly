from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.errors import DataValidationError
from setv.experts.set_scores import (
    build_set_score_payload,
    load_set_scores,
    save_set_scores,
    validate_set_score_payload,
)


class SetScoreTests(unittest.TestCase):
    def test_mean_raw_logits_margin_stability_and_round_trip(self) -> None:
        labels = np.array([0, 1], dtype=np.int64)
        logits = np.zeros((2, 8, 2), dtype=np.float32)
        logits[0, :, 0] = np.arange(8)
        logits[1, :, 1] = 2
        payload = build_set_score_payload(["a", "b"], labels, logits)
        self.assertTrue(
            np.allclose(payload["background_set_mean_logits"][0], [3.5, 0])
        )
        self.assertGreater(payload["background_set_margin_std"][0], 0)
        self.assertEqual(validate_set_score_payload(payload)["accuracy"], 1.0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scores.npz"
            save_set_scores(path, payload)
            loaded = load_set_scores(path)
            self.assertTrue(
                np.array_equal(
                    loaded["background_set_mean_logits"],
                    payload["background_set_mean_logits"],
                )
            )

    def test_exactly_eight_views_are_required(self) -> None:
        with self.assertRaises(DataValidationError):
            build_set_score_payload(
                ["a", "b"], np.array([0, 1]), np.zeros((2, 7, 2))
            )


if __name__ == "__main__":
    unittest.main()
