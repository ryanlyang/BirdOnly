from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from anchorcal.decision import (  # noqa: E402
    choose_criterion,
    verify_decision_receipt,
    write_decision_receipt,
)
from anchorcal.statistics import (  # noqa: E402
    BootstrapCriterionResult,
    evaluate_anchor_scores,
)


def _bootstrap(ace_values: list[float]) -> BootstrapCriterionResult:
    size = len(ace_values)
    return BootstrapCriterionResult(
        score_vectors=np.zeros((size, 21)),
        ace=np.asarray(ace_values, dtype=np.float64),
        kendall_tau_b=np.ones(size),
        spearman=np.ones(size),
        pair_accuracy=np.ones(size),
        adjacent_accuracy=np.ones(size),
        violations=np.zeros(size, dtype=np.int64),
        perfect_order=np.ones(size, dtype=np.bool_),
    )


class CriterionDecisionTests(unittest.TestCase):
    def test_point_ace_winner_and_one_se_credible_set(self) -> None:
        lambdas = np.linspace(0.0, 1.0, 21)
        metrics = {
            "a": evaluate_anchor_scores(lambdas, lambdas),
            "b": evaluate_anchor_scores(lambdas**1.1, lambdas),
            "c": evaluate_anchor_scores(lambdas**3, lambdas),
        }
        # Make the point ACE boundary explicit while preserving the other
        # diagnostic fields used by tie-breaking.
        metrics = {
            name: type(value)(
                ace=ace,
                kendall_tau_b=value.kendall_tau_b,
                spearman=value.spearman,
                pair_accuracy=value.pair_accuracy,
                adjacent_accuracy=value.adjacent_accuracy,
                violations=value.violations,
                perfect_order=value.perfect_order,
                ace_predictions=value.ace_predictions,
                ace_absolute_errors=value.ace_absolute_errors,
                ace_degenerate_folds=value.ace_degenerate_folds,
            )
            for (name, value), ace in zip(metrics.items(), [0.05, 0.055, 0.08])
        }
        spread = [0.04, 0.06, 0.04, 0.06]
        bootstrap = {name: _bootstrap(spread) for name in metrics}
        decision = choose_criterion(
            metrics,
            bootstrap,
            eligible_criteria=["a", "b", "c"],
            computational_cost={"a": 1.0, "b": 2.0, "c": 3.0},
        )
        self.assertEqual(decision.winner, "a")
        self.assertAlmostEqual(decision.one_standard_error, np.std(spread, ddof=1))
        self.assertEqual(decision.credible_set, ("a", "b"))

    def test_tie_breakers_apply_only_to_point_ace_tie(self) -> None:
        lambdas = np.linspace(0.0, 1.0, 21)
        base = evaluate_anchor_scores(lambdas, lambdas)

        def altered(adjacent: float, pair: float):
            return type(base)(
                ace=0.1,
                kendall_tau_b=base.kendall_tau_b,
                spearman=base.spearman,
                pair_accuracy=pair,
                adjacent_accuracy=adjacent,
                violations=base.violations,
                perfect_order=base.perfect_order,
                ace_predictions=base.ace_predictions,
                ace_absolute_errors=base.ace_absolute_errors,
                ace_degenerate_folds=base.ace_degenerate_folds,
            )

        point = {
            "a": altered(0.8, 1.0),
            "b": altered(0.9, 0.8),
            "c": altered(0.9, 0.9),
            "d": altered(0.9, 0.9),
        }
        bootstrap = {
            "a": _bootstrap([0.08, 0.12]),
            "b": _bootstrap([0.08, 0.12]),
            "c": _bootstrap([0.06, 0.14]),
            "d": _bootstrap([0.09, 0.11]),
        }
        decision = choose_criterion(
            point,
            bootstrap,
            eligible_criteria=["a", "b", "c", "d"],
            computational_cost={name: 1.0 for name in point},
        )
        self.assertEqual(decision.winner, "d")

    def test_computational_cost_is_final_scientific_tie_breaker(self) -> None:
        lambdas = np.linspace(0.0, 1.0, 21)
        metric = evaluate_anchor_scores(lambdas, lambdas)
        decision = choose_criterion(
            {"expensive": metric, "cheap": metric},
            {
                "expensive": _bootstrap([0.04, 0.06]),
                "cheap": _bootstrap([0.04, 0.06]),
            },
            eligible_criteria=["expensive", "cheap"],
            computational_cost={"expensive": 10.0, "cheap": 1.0},
        )
        self.assertEqual(decision.winner, "cheap")

    def test_receipt_is_timestamped_hashed_verified_and_immutable(self) -> None:
        lambdas = np.linspace(0.0, 1.0, 21)
        metrics = {"ordinary_accuracy": evaluate_anchor_scores(lambdas, lambdas)}
        decision = choose_criterion(
            metrics,
            {"ordinary_accuracy": _bootstrap([0.04, 0.06, 0.05])},
            eligible_criteria=["ordinary_accuracy"],
            computational_cost={"ordinary_accuracy": 0.0},
        )
        timestamp = datetime(2026, 8, 8, 12, 34, 56, 123456, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_decision_receipt(
                temporary,
                decision,
                formulas={"ordinary_accuracy": "A_biased_balanced"},
                anchor_subset_hash="a" * 64,
                anchor_family={"kind": "normalized_raw_logit", "lambdas": 21},
                branch_hashes={"foreground": "b" * 64, "background": "c" * 64},
                config_hashes={"resolved": "d" * 64},
                timestamp=timestamp,
                extra_provenance={"code_commit": "deadbeef"},
            )
            self.assertIn("20260808T123456.123456Z", paths.receipt.name)
            self.assertTrue(verify_decision_receipt(paths.receipt, paths.sha256))
            payload = json.loads(paths.receipt.read_text(encoding="utf-8"))
            self.assertEqual(payload["decision"]["winner"], "ordinary_accuracy")
            self.assertEqual(payload["created_at_utc"], "2026-08-08T12:34:56.123456Z")
            with self.assertRaises(FileExistsError):
                write_decision_receipt(
                    temporary,
                    decision,
                    formulas={"ordinary_accuracy": "A_biased_balanced"},
                    anchor_subset_hash="a" * 64,
                    anchor_family="normalized_raw_logit",
                    branch_hashes={"foreground": "b" * 64},
                    config_hashes={"resolved": "d" * 64},
                    timestamp=timestamp,
                )
            paths.receipt.write_text("tampered\n", encoding="utf-8")
            self.assertFalse(verify_decision_receipt(paths.receipt, paths.sha256))

    def test_receipt_rejects_non_sha256_provenance(self) -> None:
        lambdas = np.linspace(0.0, 1.0, 21)
        metric = evaluate_anchor_scores(lambdas, lambdas)
        decision = choose_criterion(
            {"ordinary_accuracy": metric},
            {"ordinary_accuracy": _bootstrap([0.04, 0.06])},
            eligible_criteria=["ordinary_accuracy"],
            computational_cost={"ordinary_accuracy": 0.0},
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
                write_decision_receipt(
                    temporary,
                    decision,
                    formulas={"ordinary_accuracy": "A_biased_balanced"},
                    anchor_subset_hash="not-a-hash",
                    anchor_family="normalized_raw_logit",
                    branch_hashes={"foreground": "b" * 64},
                    config_hashes={"resolved": "d" * 64},
                )


if __name__ == "__main__":
    unittest.main()
