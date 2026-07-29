from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.errors import DataValidationError
from setv.experts.scores import (
    OBJECT_SCORE_KEYS,
    build_object_score_payload,
    load_object_scores,
    object_sanity_warnings,
    save_object_scores,
    true_class_margin,
    validate_object_score_payload,
)


class ObjectScoreTests(unittest.TestCase):
    def test_true_class_margin_and_schema(self) -> None:
        logits = np.array([[3.0, 1.0], [-2.0, 0.5], [1.0, 4.0]], dtype=np.float32)
        labels = np.array([0, 0, 1])
        margins = true_class_margin(logits, labels)
        np.testing.assert_allclose(margins, [2.0, -2.5, 3.0])
        payload = build_object_score_payload(["a", "b", "c"], labels, logits)
        self.assertEqual(set(payload), OBJECT_SCORE_KEYS)
        self.assertNotIn("temperature", payload)
        self.assertNotIn("calibrated_probability", payload)
        summary = validate_object_score_payload(
            payload,
            pd.DataFrame({"sample_id": ["a", "b", "c"], "y": labels}),
        )
        self.assertAlmostEqual(summary["accuracy"], 2.0 / 3.0)
        self.assertEqual(object_sanity_warnings(summary), [])

    def test_npz_round_trip_disallows_pickle(self) -> None:
        payload = build_object_score_payload(
            ["10", "11"],
            np.array([0, 1]),
            np.array([[2.0, 1.0], [0.5, 3.0]], dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scores.npz"
            save_object_scores(path, payload)
            loaded = load_object_scores(path)
            for key in OBJECT_SCORE_KEYS:
                np.testing.assert_array_equal(loaded[key], payload[key])

    def test_manifest_order_mismatch_is_rejected(self) -> None:
        payload = build_object_score_payload(
            ["a", "b"],
            np.array([0, 1]),
            np.array([[2.0, 1.0], [0.5, 3.0]], dtype=np.float32),
        )
        with self.assertRaisesRegex(DataValidationError, "IDs/order"):
            validate_object_score_payload(
                payload,
                pd.DataFrame({"sample_id": ["b", "a"], "y": [1, 0]}),
            )


if __name__ == "__main__":
    unittest.main()
