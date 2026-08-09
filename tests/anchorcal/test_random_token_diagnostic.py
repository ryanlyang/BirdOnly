from __future__ import annotations

import unittest

import numpy as np

from anchorcal.random_token_diagnostic import (
    diagnostic_seeds,
    random_token_draw_indices,
    summarize_random_token_predictions,
)
from anchorcal.seeds import stateless_rng


class RandomTokenDiagnosticTests(unittest.TestCase):
    def test_seed_zero_exactly_reproduces_the_original_pooled_draw(self) -> None:
        image_ids = np.asarray([17, 42], dtype=np.int64)
        seeds = diagnostic_seeds(8003, 3)
        self.assertEqual(seeds[0], 8003)
        self.assertEqual(len(set(seeds)), 3)
        draws = random_token_draw_indices(
            image_ids,
            patches_per_class=20,
            token_budget=8,
            seed=seeds[0],
            mode="pooled",
        )
        expected = stateless_rng(8003, 17, "random_token_audit").choice(
            40, 8, replace=False
        )
        np.testing.assert_array_equal(draws[0], expected)

    def test_per_draw_mode_has_exact_source_class_balance(self) -> None:
        draws = random_token_draw_indices(
            np.arange(10),
            patches_per_class=50,
            token_budget=32,
            seed=8003,
            mode="per_draw_class_balanced",
        )
        for row in draws:
            self.assertEqual(int(np.sum(row < 50)), 16)
            self.assertEqual(int(np.sum(row >= 50)), 16)
            self.assertEqual(len(np.unique(row)), 32)

    def test_summary_reports_each_repeat_without_applying_a_gate(self) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        predictions = np.asarray(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
            ],
            dtype=np.int64,
        )
        result = summarize_random_token_predictions(
            predictions,
            labels,
            [8003, 9001],
            bootstrap_replicates=50,
            permutation_replicates=50,
        )
        self.assertEqual(result["repeat_count"], 2)
        self.assertEqual(result["recipient_count"], 4)
        self.assertEqual(result["mean_balanced_accuracy"], 0.5)
        self.assertEqual(len(result["repeats"]), 2)
        self.assertIn("one_sided_recipient_label_permutation_p", result)

    def test_invalid_draw_contracts_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "even token budget"):
            random_token_draw_indices(
                np.asarray([1]),
                patches_per_class=20,
                token_budget=7,
                seed=8003,
                mode="per_draw_class_balanced",
            )
        with self.assertRaisesRegex(ValueError, "unknown"):
            random_token_draw_indices(
                np.asarray([1]),
                patches_per_class=20,
                token_budget=8,
                seed=8003,
                mode="not_a_mode",
            )


if __name__ == "__main__":
    unittest.main()
