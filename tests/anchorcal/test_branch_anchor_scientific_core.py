from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from anchorcal.anchor_cache import (  # noqa: E402
    ExtremeCache,
    aggregate_source_coordinates,
    cached_logits,
    cached_signed_contributions,
    center_numpy,
)
from anchorcal.anchor_pipeline import (  # noqa: E402
    _background_purity_audit,
    _geometry_audit_seeds,
)
from anchorcal.background import (  # noqa: E402
    build_view_arrays,
    invalid_background_records,
    sample_background_indices,
    select_token_budget,
)
from anchorcal.calibration import average_view_logits, fit_temperature  # noqa: E402
from anchorcal.competence import (  # noqa: E402
    cap_anchor_subset,
    construct_competence_intersection,
)
from anchorcal.errors import AuditFailure  # noqa: E402
from anchorcal.metrics import (  # noqa: E402
    classification_metrics,
    class_balanced_mean,
    harmonic_mean,
    per_example_cross_entropy,
)
from anchorcal.seeds import stable_seed  # noqa: E402
from anchorcal.io import hash_object  # noqa: E402
from anchorcal.vlm_masks import (  # noqa: E402
    VlmMaskBank,
    decode_vlm_mask,
    producer_vlm_mask_name,
    vlm_mask_manifest_entry,
)

try:  # The local lightweight environment may intentionally omit torch.
    import torch
except Exception:  # pragma: no cover - environment-dependent skip
    torch = None


class BackgroundBudgetAndViewTests(unittest.TestCase):
    @staticmethod
    def _coverage_inputs() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        labels = np.repeat([0, 1], 20)
        counts = np.full(40, 70, dtype=np.int64)
        # Ninety percent class-0 coverage at K=64, but complete coverage at 48.
        counts[0:2] = 50
        return (
            {name: counts.copy() for name in ("expert_train", "expert_calibration", "biased_val")},
            {name: labels.copy() for name in ("expert_train", "expert_calibration", "biased_val")},
        )

    def test_budget_chooses_largest_coverage_valid_fallback(self) -> None:
        counts, labels = self._coverage_inputs()
        decision = select_token_budget(counts, labels)
        self.assertEqual(decision.token_budget, 48)
        self.assertAlmostEqual(decision.coverage["64"]["expert_train.class_0"], 0.90)
        self.assertEqual(decision.coverage["48"]["expert_train.class_0"], 1.0)

    def test_budget_aborts_when_even_32_fails(self) -> None:
        labels = np.repeat([0, 1], 20)
        counts = np.full(40, 31, dtype=np.int64)
        split_counts = {
            name: counts.copy() for name in ("expert_train", "expert_calibration", "biased_val")
        }
        split_labels = {
            name: labels.copy() for name in ("expert_train", "expert_calibration", "biased_val")
        }
        with self.assertRaises(AuditFailure):
            select_token_budget(split_counts, split_labels)

    def test_sampling_is_deterministic_and_never_uses_replacement(self) -> None:
        eligible = np.arange(100, dtype=np.int64)
        first, first_seed = sample_background_indices(
            eligible,
            64,
            global_seed=6003,
            sample_id=17,
            view_index=0,
            purpose="background_branch_eval",
        )
        replay, replay_seed = sample_background_indices(
            eligible,
            64,
            global_seed=6003,
            sample_id=17,
            view_index=0,
            purpose="background_branch_eval",
        )
        second_view, _ = sample_background_indices(
            eligible,
            64,
            global_seed=6003,
            sample_id=17,
            view_index=1,
            purpose="background_branch_eval",
        )
        np.testing.assert_array_equal(first, replay)
        self.assertEqual(first_seed, replay_seed)
        self.assertEqual(len(np.unique(first)), 64)
        self.assertTrue(set(first).issubset(set(eligible)))
        self.assertFalse(np.array_equal(first, second_view))
        with self.assertRaises(AuditFailure):
            sample_background_indices(
                eligible[:63],
                64,
                global_seed=6003,
                sample_id=17,
                view_index=0,
                purpose="background_branch_eval",
            )

    def test_view_bank_arrays_are_sorted_replayed_and_mark_invalid(self) -> None:
        eligible = {9: np.arange(80), 2: np.arange(70), 5: np.arange(40)}
        first = build_view_arrays(
            eligible, token_budget=64, views=8, global_seed=6003
        )
        replay = build_view_arrays(
            eligible, token_budget=64, views=8, global_seed=6003
        )
        ids, indices, seeds, invalid = first
        np.testing.assert_array_equal(ids, [2, 5, 9])
        self.assertEqual(invalid, [5])
        self.assertEqual(
            invalid_background_records(eligible, invalid, 64),
            [
                {
                    "img_id": 5,
                    "reason": "insufficient_pure_background_patches",
                    "eligible_patch_count": 40,
                    "required_token_budget": 64,
                    "shortfall": 24,
                }
            ],
        )
        self.assertTrue(np.all(indices[1] == -1))
        self.assertEqual(seeds.dtype, np.dtype(np.uint64))
        self.assertTrue(np.any(seeds > np.iinfo(np.uint32).max))
        for row in (0, 2):
            for view in range(8):
                self.assertEqual(len(np.unique(indices[row, view])), 64)
        for left, right in zip(first[:3], replay[:3], strict=True):
            np.testing.assert_array_equal(left, right)

    def test_geometry_mlp_bootstrap_seed_is_purpose_derived_and_persistable(self) -> None:
        config = {
            "seeds": {
                "geometry_auditor_split": 8001,
                "geometry_auditor_model": 8002,
            }
        }
        manifest = _geometry_audit_seeds(config)
        expected = stable_seed(8001, "geometry_auditor_mlp_bootstrap")
        self.assertEqual(manifest["mlp_bootstrap"], expected)
        self.assertNotEqual(manifest["mlp_bootstrap"], 8001 + 1)
        self.assertIn("geometry_auditor_mlp_bootstrap", manifest["mlp_bootstrap_derivation"])

    def test_background_purity_reloads_masks_and_rejects_retained_foreground(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            image_root = root / "images"
            mask_root = root / "masks"
            (output / "splits").mkdir(parents=True)
            image_root.mkdir()
            mask_root.mkdir()

            image = np.full((224, 224, 3), 127, dtype=np.uint8)
            valid_mask = np.zeros((224, 224), dtype=bool)
            valid_mask[100:108, 100:108] = True
            invalid_mask = np.ones((224, 224), dtype=bool)
            invalid_mask[0:16, 0:16] = False
            mask_entries = {}
            for metadata_index, (img_id, filename, mask) in enumerate(
                (
                    (1, "valid.jpg", valid_mask),
                    (2, "invalid.jpg", invalid_mask),
                )
            ):
                Image.fromarray(image, mode="RGB").save(image_root / filename)
                encoded = np.zeros((224, 224, 3), dtype=np.uint8)
                encoded[mask] = np.asarray([128, 0, 0], dtype=np.uint8)
                mask_path = mask_root / producer_vlm_mask_name(filename)
                Image.fromarray(encoded, mode="RGB").save(mask_path)
                decoded = decode_vlm_mask(mask_path)
                mask_entries[img_id] = vlm_mask_manifest_entry(
                    img_id=img_id,
                    metadata_index=metadata_index,
                    img_filename=filename,
                    split=1,
                    root=mask_root,
                    path=mask_path,
                    mapping_rule="weclip_producer_flattened_relative_stem",
                    decoded=decoded,
                )
            mask_bank = VlmMaskBank(
                root=mask_root.resolve(),
                entries=mask_entries,
                mask_bank_sha256="synthetic-test-bank",
                minimum_foreground_fraction=0.0,
                maximum_foreground_fraction=1.0,
            )

            columns = ["img_id", "img_filename", "y", "place", "split"]
            pd.DataFrame([[1, "valid.jpg", 0, 0, 1]], columns=columns).to_csv(
                output / "splits" / "waterbirds100_expert_calibration.csv",
                index=False,
            )
            pd.DataFrame([[2, "invalid.jpg", 1, 1, 1]], columns=columns).to_csv(
                output / "splits" / "waterbirds100_biased_val.csv",
                index=False,
            )
            config = {
                "paths": {
                    "output_root": str(output),
                    "waterbirds_root": str(image_root),
                    "vlm_mask_root": str(mask_root),
                },
                "masks": {
                    "minimum_foreground_fraction": 0.0,
                    "maximum_foreground_fraction": 1.0,
                },
                "runtime": {"debug": False},
                "data": {"dilation_radius": 8},
            }
            bank = {
                "img_id": np.asarray([1, 2], dtype=np.int64),
                "source_patch_indices": np.asarray(
                    [[[0], [0]], [[-1], [-1]]], dtype=np.int16
                ),
                "invalid_img_id": np.asarray([2], dtype=np.int64),
                "invalid_eligible_patch_count": np.asarray([0], dtype=np.int16),
                "invalid_reason_code": "insufficient_pure_background_patches",
                "token_budget": 1,
                "mask_dilation_hash": hash_object(
                    {"implementation": "euclidean_disk", "radius": 8}
                ),
            }
            preprocessing = SimpleNamespace(
                image_size=224, effective_resize_shortest=224
            )
            with (
                patch("anchorcal.anchor_pipeline.load_view_bank", return_value=bank),
                patch(
                    "anchorcal.anchor_pipeline.load_preprocessing_manifest",
                    return_value=preprocessing,
                ),
            ):
                report = _background_purity_audit(config, mask_bank)
            self.assertEqual(report["retained_patch_count"], 2)
            self.assertEqual(report["invalid_ids"], [2])
            self.assertIn(
                "insufficient_pure_background_patches",
                report["invalid_examples"][0]["reasons"],
            )
            self.assertGreater(report["minimum_raw_mask_distance_pixels"], 8.0)
            self.assertEqual(report["maximum_dilated_foreground_fraction"], 0.0)

            # Patch 90 contains the real central bird mask.  A circular audit
            # based on stored eligible indices would miss this corruption;
            # the independent mask replay must reject it.
            impure_bank = dict(bank)
            impure_bank["source_patch_indices"] = np.asarray(
                [[[90], [90]], [[-1], [-1]]], dtype=np.int16
            )
            with (
                patch(
                    "anchorcal.anchor_pipeline.load_view_bank",
                    return_value=impure_bank,
                ),
                patch(
                    "anchorcal.anchor_pipeline.load_preprocessing_manifest",
                    return_value=preprocessing,
                ),
                self.assertRaises(AuditFailure),
            ):
                _background_purity_audit(config, mask_bank)


class MetricsCalibrationAndCompetenceTests(unittest.TestCase):
    def test_metrics_are_class_and_group_balanced_and_cross_entropy_is_finite(self) -> None:
        labels = np.asarray([0, 0, 0, 1])
        places = np.asarray([0, 1, 0, 0])
        # Add one y=1, place=1 member so all four groups exist.
        labels = np.append(labels, 1)
        places = np.append(places, 1)
        predictions = np.asarray([0, 0, 1, 1, 0])
        logits = np.full((len(labels), 2), -1000.0)
        logits[np.arange(len(labels)), predictions] = 1000.0
        result = classification_metrics(logits, labels, places)
        expected_class_balanced = ((2 / 3) + (1 / 2)) / 2
        self.assertAlmostEqual(result["class_balanced_accuracy"], expected_class_balanced)
        self.assertEqual(result["worst_group_accuracy"], 0.0)
        self.assertTrue(np.isfinite(per_example_cross_entropy(logits, labels)).all())
        self.assertAlmostEqual(class_balanced_mean(predictions == labels, labels), expected_class_balanced)
        self.assertEqual(harmonic_mean(0.0, 1.0), 0.0)

    def test_harmonic_mean_uses_locked_one_e_minus_eight_epsilon(self) -> None:
        self.assertEqual(
            harmonic_mean(0.75, 0.5),
            2.0 * 0.75 * 0.5 / (0.75 + 0.5 + 1.0e-8),
        )

    def test_view_logits_are_averaged_before_any_calibration(self) -> None:
        views = np.asarray(
            [
                [[1.0, 3.0], [2.0, 4.0]],
                [[3.0, 5.0], [4.0, 8.0]],
            ]
        )
        np.testing.assert_allclose(
            average_view_logits(views), np.asarray([[2.0, 4.0], [3.0, 6.0]])
        )

    @unittest.skipIf(torch is None, "torch is unavailable in the lightweight test environment")
    def test_temperature_is_bounded_and_does_not_increase_nll(self) -> None:
        logits = np.asarray([[8.0, -8.0], [7.0, -7.0], [4.0, -4.0], [4.0, -4.0]])
        labels = np.asarray([0, 0, 1, 1])
        result = fit_temperature(logits, labels)
        self.assertGreaterEqual(result.temperature, 0.05)
        self.assertLessEqual(result.temperature, 20.0)
        self.assertLessEqual(result.nll_after, result.nll_before + 1e-10)

    def test_competence_uses_both_correct_raw_margins_and_full_intersection_scales(self) -> None:
        per_class = 60
        labels = np.repeat([0, 1], per_class)
        ids = np.arange(labels.size)
        foreground_margin = np.concatenate(
            [np.arange(1, per_class + 1), np.arange(101, 101 + per_class)]
        ).astype(float)
        background_margin = np.concatenate(
            [np.arange(2, 2 + per_class), np.arange(202, 202 + per_class)]
        ).astype(float)

        def logits_from_margin(margin: np.ndarray) -> np.ndarray:
            logits = np.zeros((labels.size, 2), dtype=float)
            logits[np.arange(labels.size), labels] = margin
            return logits

        result = construct_competence_intersection(
            ids,
            labels,
            logits_from_margin(foreground_margin),
            logits_from_margin(background_margin),
            np.ones(labels.size, dtype=bool),
            minimum_per_class=50,
        )
        self.assertAlmostEqual(result.foreground_scale, np.median(foreground_margin))
        self.assertAlmostEqual(result.background_scale, np.median(background_margin))
        capped = cap_anchor_subset(
            result,
            set(ids.tolist()),
            per_class=50,
            seed=424242,
            minimum_per_class=50,
        )
        self.assertEqual(len(capped), 100)
        replay = cap_anchor_subset(
            result,
            set(ids.tolist()),
            per_class=50,
            seed=424242,
            minimum_per_class=50,
        )
        np.testing.assert_array_equal(capped, replay)

    def test_competence_gate_is_per_class(self) -> None:
        labels = np.repeat([0, 1], 50)
        ids = np.arange(100)
        logits = np.zeros((100, 2), dtype=float)
        logits[np.arange(100), labels] = 1.0
        valid = np.ones(100, dtype=bool)
        valid[0] = False
        with self.assertRaises(AuditFailure):
            construct_competence_intersection(
                ids, labels, logits, logits, valid, minimum_per_class=50
            )


class AnchorCacheTests(unittest.TestCase):
    def test_centered_scaled_lambda_logits_are_algebraically_exact(self) -> None:
        foreground = np.asarray([[4.0, 0.0], [1.0, 3.0]])
        background = np.asarray([[2.0, 0.0], [0.0, 6.0]])
        cache = ExtremeCache(foreground, background)
        actual = cached_logits(cache, 0.25, foreground_scale=2.0, background_scale=4.0)
        expected = (
            0.25 * center_numpy(foreground) / (2.0 + 1.0e-8)
            + 0.75 * center_numpy(background) / (4.0 + 1.0e-8)
        )
        np.testing.assert_allclose(actual, expected)

    def test_repeated_source_coordinates_are_summed_not_averaged(self) -> None:
        coordinates, values = aggregate_source_coordinates(
            np.asarray([0.25, 0.5, -0.1, 9.0]),
            np.asarray([7, 7, 9, -1]),
        )
        np.testing.assert_array_equal(coordinates, [7, 9])
        np.testing.assert_allclose(values, [0.75, -0.1])

    def test_signed_cache_applies_lambda_only_once(self) -> None:
        cache = ExtremeCache(
            np.zeros((1, 2)),
            np.zeros((1, 2)),
            foreground_signed=np.asarray([2.0]),
            background_signed=np.asarray([3.0]),
        )
        foreground, background = cached_signed_contributions(cache, 0.25)
        np.testing.assert_allclose(foreground, [0.5])
        np.testing.assert_allclose(background, [2.25])


if __name__ == "__main__":
    unittest.main()
