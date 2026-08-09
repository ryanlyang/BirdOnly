from __future__ import annotations

import unittest

import numpy as np

from anchorcal.random_token_diagnostic import (
    diagnostic_seeds,
    random_token_draw_indices,
    require_repeated_random_token_collapse,
    summarize_random_token_predictions,
)
from anchorcal.errors import AuditFailure
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

    def test_repeated_gate_uses_aggregate_not_individual_repeat_failures(self) -> None:
        summary = {
            "mean_balanced_accuracy": 0.5004,
            "aggregate_per_image_bootstrap_95": {
                "point": 0.5004,
                "lower": 0.4928,
                "upper": 0.5082,
                "replicates": 2000,
            },
            "repeats": [
                {"balanced_accuracy": 0.5280, "original_gate_pass": False},
                {"balanced_accuracy": 0.4973, "original_gate_pass": True},
            ],
        }
        result = require_repeated_random_token_collapse(summary)
        self.assertEqual(result["hard_gate"]["status"], "passed")
        self.assertEqual(
            result["hard_gate"]["scope"],
            "aggregate_per_image_correctness_across_fixed_repeats",
        )

    def test_repeated_gate_fails_closed_on_aggregate_leakage(self) -> None:
        for point, lower, upper in (
            (0.54, 0.49, 0.56),
            (0.528, 0.506, 0.548),
            (float("nan"), 0.49, 0.51),
        ):
            with self.subTest(point=point, lower=lower, upper=upper):
                with self.assertRaises(AuditFailure):
                    require_repeated_random_token_collapse(
                        {
                            "aggregate_per_image_bootstrap_95": {
                                "point": point,
                                "lower": lower,
                                "upper": upper,
                            }
                        }
                    )

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
