from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from anchorcal.criteria import (  # noqa: E402
    blur_accuracy,
    build_scores,
    donor_specific_accuracy,
)
from anchorcal.errors import AuditFailure  # noqa: E402
from anchorcal.interventions import (  # noqa: E402
    InterventionType,
    assert_anchor_intervention_contract,
    assert_foreground_stream_unchanged,
    assign_candidate_donor_patches,
    assign_donors,
    coarse_bin,
)
from anchorcal.metrics import harmonic_mean  # noqa: E402
from anchorcal.saliency import anchor_image_alignment, image_alignment  # noqa: E402


class SaliencyContractTests(unittest.TestCase):
    def test_zero_scored_mass_is_neutral_and_persistently_flagged(self) -> None:
        result = image_alignment(
            np.zeros(4),
            np.asarray([1, 2, 3, 4]),
            np.asarray([1, 2]),
            np.asarray([3, 4]),
        )
        self.assertEqual(result.alignment, 0.5)
        self.assertTrue(result.fallback_absolute)
        self.assertTrue(result.zero_scored_attribution)
        self.assertEqual(result.foreground_density, 0.0)
        self.assertEqual(result.background_density, 0.0)

    def test_negative_mass_uses_per_image_absolute_fallback(self) -> None:
        result = image_alignment(
            np.asarray([-2.0, -1.0]),
            np.asarray([11, 22]),
            np.asarray([11]),
            np.asarray([22]),
        )
        self.assertTrue(result.fallback_absolute)
        self.assertFalse(result.zero_scored_attribution)
        self.assertAlmostEqual(result.foreground_density, 2.0)
        self.assertAlmostEqual(result.background_density, 1.0)
        self.assertAlmostEqual(result.alignment, 2.0 / 3.0, places=10)

    def test_anchor_mixes_signed_values_before_relu_and_sums_repeated_background(self) -> None:
        result = anchor_image_alignment(
            foreground_signed=np.asarray([2.0]),
            foreground_source=np.asarray([10]),
            background_signed_occurrences=np.asarray([0.5, 0.5]),
            background_source_occurrences=np.asarray([20, 20]),
            foreground_indices=np.asarray([10]),
            background_indices=np.asarray([20]),
            reliance_lambda=0.5,
        )
        # Foreground density: .5*2=1. Background density: .5*(.5+.5)=.5.
        self.assertAlmostEqual(result.foreground_density, 1.0)
        self.assertAlmostEqual(result.background_density, 0.5)
        self.assertAlmostEqual(result.alignment, 2.0 / 3.0, places=10)


class InterventionContractTests(unittest.TestCase):
    @staticmethod
    def _foreground_state() -> dict[str, np.ndarray]:
        return {
            "input_image": np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2),
            "input_mask": np.asarray([[[True, False], [True, True]]]),
            "logits": np.asarray([0.25, -0.25], dtype=np.float32),
            "patch_activations": np.arange(8, dtype=np.float32).reshape(1, 2, 4),
            "patch_valid": np.asarray([[True, True]]),
            "source_indices": np.asarray([[3, 7]], dtype=np.int64),
        }

    @staticmethod
    def _assert_contract(
        intervention: InterventionType,
        clean: dict[str, np.ndarray],
        intervened: dict[str, np.ndarray],
    ) -> None:
        assert_anchor_intervention_contract(
            intervention,
            clean["logits"],
            intervened["logits"],
            clean_input_image=clean["input_image"],
            intervened_input_image=intervened["input_image"],
            clean_input_mask=clean["input_mask"],
            intervened_input_mask=intervened["input_mask"],
            clean_patch_activations=clean["patch_activations"],
            intervened_patch_activations=intervened["patch_activations"],
            clean_patch_valid=clean["patch_valid"],
            intervened_patch_valid=intervened["patch_valid"],
            clean_source_indices=clean["source_indices"],
            intervened_source_indices=intervened["source_indices"],
        )

    def test_coarse_bins_follow_locked_patch_center_formula(self) -> None:
        self.assertEqual(coarse_bin(0), (0, 0))
        self.assertEqual(coarse_bin(4 * 14 + 4), (0, 0))
        self.assertEqual(coarse_bin(5 * 14 + 5), (1, 1))
        self.assertEqual(coarse_bin(8 * 14 + 8), (1, 1))
        self.assertEqual(coarse_bin(9 * 14 + 9), (2, 2))
        self.assertEqual(coarse_bin(13 * 14 + 13), (2, 2))

    def test_donor_images_are_distinct_opposite_class_and_deterministic(self) -> None:
        ids = np.arange(12)
        labels = np.repeat([0, 1], 6)
        eligible = np.ones(12, dtype=bool)
        first = assign_donors(ids, labels, eligible, donors_per_recipient=4, seed=31415)
        replay = assign_donors(ids, labels, eligible, donors_per_recipient=4, seed=31415)
        self.assertEqual(first, replay)
        label_by_id = dict(zip(ids.tolist(), labels.tolist(), strict=True))
        for assignment in first:
            self.assertEqual(len(assignment.donor_ids), 4)
            self.assertEqual(len(set(assignment.donor_ids)), 4)
            self.assertNotIn(assignment.recipient_id, assignment.donor_ids)
            for donor_id in assignment.donor_ids:
                self.assertNotEqual(
                    label_by_id[assignment.recipient_id], label_by_id[donor_id]
                )

    def test_candidate_patch_assignment_prefers_unique_bin_then_replacement(self) -> None:
        recipient = np.asarray([0, 1, 2, 3])  # all coarse bin (0,0)
        donor = np.asarray([14, 15])  # also coarse bin (0,0)
        first = assign_candidate_donor_patches(1, 9, recipient, donor, seed=31415)
        replay = assign_candidate_donor_patches(1, 9, recipient, donor, seed=31415)
        self.assertEqual(first, replay)
        self.assertEqual({first[0].donor_source_index, first[1].donor_source_index}, {14, 15})
        self.assertEqual(first[0].fallback, "none")
        self.assertEqual(first[1].fallback, "none")
        self.assertTrue(
            all(
                assignment.fallback == "matching_bin_with_replacement"
                for assignment in first[2:]
            )
        )

    def test_background_intervention_stream_tolerance_is_hard(self) -> None:
        assert_foreground_stream_unchanged(
            np.asarray([[1.0, 2.0]]), np.asarray([[1.0 + 1e-6, 2.0]])
        )
        with self.assertRaises(AuditFailure):
            assert_foreground_stream_unchanged(
                np.asarray([[1.0, 2.0]]), np.asarray([[1.0 + 1.1e-6, 2.0]])
            )

    def test_token_swap_contract_detects_foreground_input_mutation(self) -> None:
        clean = self._foreground_state()
        intervened = {name: value.copy() for name, value in clean.items()}
        intervened["input_image"][0, 0, 0, 0] += 1.0
        with self.assertRaisesRegex(AuditFailure, "foreground input_image"):
            self._assert_contract(
                InterventionType.TOKEN_SWAP_BACKGROUND, clean, intervened
            )

    def test_blur_contract_detects_foreground_output_mutation(self) -> None:
        clean = self._foreground_state()
        intervened = {name: value.copy() for name, value in clean.items()}
        intervened["patch_activations"][0, 0, 0] += 2.0e-6
        with self.assertRaisesRegex(AuditFailure, "foreground patch_activations"):
            self._assert_contract(
                InterventionType.BLUR_BACKGROUND, clean, intervened
            )

    def test_foreground_only_contract_is_covered_and_detects_metadata_mutation(self) -> None:
        clean = self._foreground_state()
        replay = {name: value.copy() for name, value in clean.items()}
        self._assert_contract(
            InterventionType.FOREGROUND_ONLY_GREENSCREEN, clean, replay
        )
        replay["source_indices"][0, 0] = 99
        with self.assertRaisesRegex(AuditFailure, "foreground source_indices"):
            self._assert_contract(
                InterventionType.FOREGROUND_ONLY_GREENSCREEN, clean, replay
            )


class CriterionAggregationTests(unittest.TestCase):
    def test_donors_average_within_image_then_images_within_class(self) -> None:
        labels = np.asarray([0, 0, 0, 1])
        donor_correct = np.asarray(
            [
                [1, 1],
                [1, 1],
                [0, 0],
                [1, 0],
            ]
        )
        score, per_image = donor_specific_accuracy(donor_correct, labels)
        np.testing.assert_allclose(per_image, [1.0, 1.0, 0.0, 0.5])
        expected = ((2.0 / 3.0) + 0.5) / 2.0
        self.assertAlmostEqual(score, expected)
        self.assertNotAlmostEqual(score, donor_correct.mean())

    def test_blurs_average_within_image_then_class(self) -> None:
        labels = np.asarray([0, 0, 0, 1])
        sigma_correct = np.asarray(
            [
                [1, 1, 1],
                [1, 1, 1],
                [0, 0, 0],
                [1, 0, 0],
            ]
        )
        score, per_image = blur_accuracy(sigma_correct, labels)
        np.testing.assert_allclose(per_image, [1.0, 1.0, 0.0, 1.0 / 3.0])
        self.assertAlmostEqual(score, ((2.0 / 3.0) + (1.0 / 3.0)) / 2.0)

    def test_harmonic_scores_use_full_biased_accuracy_not_selector_accuracy(self) -> None:
        # Full biased-val accuracy is class-balanced to 0.75, while every
        # selector-subset robustness signal is 0.5.
        scores = build_scores(
            full_biased_correct=np.asarray([1, 1, 1, 0, 1, 1, 0, 0]),
            full_biased_labels=np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
            selector_labels=np.asarray([0, 0, 1, 1]),
            saliency_alignment=np.asarray([1.0, 0.0, 1.0, 0.0]),
            donor_correct=np.asarray([[1, 1], [0, 0], [1, 1], [0, 0]]),
            blur_correct=np.asarray([[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]]),
        )
        self.assertAlmostEqual(scores.ordinary_accuracy, 0.625)
        expected = harmonic_mean(0.625, 0.5)
        self.assertAlmostEqual(scores.saliency_harmonic, expected)
        self.assertAlmostEqual(scores.token_swap_harmonic, expected)
        self.assertAlmostEqual(scores.background_blur_harmonic, expected)
        self.assertIsNone(scores.foreground_only_harmonic)
        self.assertAlmostEqual(scores.diagnostics["swap_product"], 0.625 * 0.5)


if __name__ == "__main__":
    unittest.main()
