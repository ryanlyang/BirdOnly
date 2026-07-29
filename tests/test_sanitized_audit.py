from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.experts.sanitized_audit import (
    _auditor_metrics,
    geometry_features,
    heldout_image_split,
    run_mask_auditors,
)
from setv.experts.sanitized_config import load_sanitized_bank_config
from setv.experts.sanitized_masks import pack_masks

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


class SanitizedAuditTests(unittest.TestCase):
    def test_geometry_schema_image_split_and_statistical_gate(self) -> None:
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 6:26] = True
        features = geometry_features(mask, 0)
        self.assertEqual(features.shape, (14,))
        labels = np.repeat([0, 1], 20)
        train, heldout = heldout_image_split(labels, fraction=0.2, seed=99)
        self.assertEqual(np.intersect1d(train, heldout).size, 0)
        self.assertEqual(set(labels[heldout]), {0, 1})
        config = load_sanitized_bank_config(
            ROOT / "configs" / "sanitized_mask_bank.yaml", seed=99
        )
        probabilities = np.full((len(heldout), 8), 0.5)
        metrics = _auditor_metrics(
            probabilities,
            labels[heldout],
            seed=101,
            config=config["auditor"],
        )
        self.assertEqual(metrics["heldout_mask_balanced_accuracy"], 0.5)
        self.assertTrue(metrics["confidence_interval_contains_chance"])
        self.assertTrue(metrics["accepted"])

    @unittest.skipUnless(TORCH_AVAILABLE, "torch unavailable")
    def test_all_three_real_auditor_paths_produce_heldout_predictions(self) -> None:
        config = load_sanitized_bank_config(
            ROOT / "configs" / "sanitized_mask_bank.yaml",
            seed=777,
            auditor_device="cpu",
        )
        config["auditor"]["bootstrap_repetitions"] = 25
        config["auditor"]["small_cnn"].update(
            {"epochs": 1, "batch_size": 64, "pooled_resolution": 16}
        )
        labels = np.repeat([0, 1], 20)
        sample_ids = np.asarray([f"audit-{index}" for index in range(40)])
        family_row = np.asarray([0, 0, 0, 1, 1, 1, 2, 2], dtype=np.uint8)
        family_ids = np.tile(family_row, (40, 1))
        masks = np.zeros((40, 8, 32, 32), dtype=bool)
        for image_index in range(40):
            for view_index in range(8):
                inset = 3 + ((image_index * 7 + view_index * 3) % 7)
                masks[
                    image_index,
                    view_index,
                    inset : 32 - inset,
                    inset : 32 - inset,
                ] = True
        report, arrays = run_mask_auditors(
            pack_masks(masks),
            family_ids,
            32,
            sample_ids,
            labels,
            seed=777,
            config=config["auditor"],
        )
        self.assertEqual(
            set(report["auditors"]),
            {
                "logistic_geometry",
                "gradient_boosted_geometry",
                "small_cnn_binary_mask",
            },
        )
        for name in report["auditors"]:
            self.assertEqual(
                arrays[f"{name}_probability"].shape,
                (report["heldout_image_count"], 8),
            )


if __name__ == "__main__":
    unittest.main()
