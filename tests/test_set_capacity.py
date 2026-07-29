from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from fixture_data import create_fixture, fixture_config
from setv.errors import DataValidationError
from setv.experts.set_capacity import run_set_capacity_census
from setv.experts.set_config import load_set_expert_config
from setv.experts.set_data import SetBackgroundDataset
from setv.phase0 import approve_visual_audit, build_phase0
from setv.utils.hashing import sha256_file


class SetCapacityCensusTests(unittest.TestCase):
    def test_census_is_complete_selector_safe_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, masks = create_fixture(root)
            phase0 = root / "phase0"
            phase0_config = fixture_config(dataset, masks, phase0)
            phase0_config["transforms"].update(
                {
                    "image_size": 224,
                    "evaluation_resize_shortest": 256,
                }
            )
            build_phase0(phase0_config)
            approve_visual_audit(
                phase0, reviewer="capacity fixture", confirmation=True
            )
            config = load_set_expert_config(
                ROOT / "configs" / "expert_background_set.yaml",
                seed=1401,
                phase0_dir=str(phase0),
                device="cpu",
            )
            json_report = root / "capacity.json"
            csv_report = root / "capacity.csv"
            report = run_set_capacity_census(
                config, json_report, csv_report
            )

            self.assertEqual(report["status"], "complete")
            self.assertTrue(report["diagnostic_only"])
            self.assertFalse(
                report["information_boundary"][
                    "protected_group_columns_loaded"
                ]
            )
            self.assertFalse(
                report["information_boundary"][
                    "oracle_or_test_split_loaded"
                ]
            )
            self.assertEqual(report["combined"]["sample_count"], 40)
            self.assertEqual(
                report["provenance"]["csv_report_sha256"],
                sha256_file(csv_report),
            )
            frame = pd.read_csv(csv_report, dtype={"sample_id": str})
            self.assertEqual(len(frame), 40)
            self.assertNotIn("place", frame.columns)
            self.assertNotIn("group", frame.columns)
            self.assertTrue(
                (
                    frame["best_fixed_view_capacity"]
                    == frame[
                        ["canonical_capacity", "full_frame_capacity"]
                    ].max(axis=1)
                ).all()
            )
            self.assertIn(
                "16", report["combined"]["support_by_floor"]
            )

    def test_training_filter_excludes_only_capacity_ineligible_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, masks = create_fixture(root)
            for sample_id in (0, 1, 2):
                path = masks / f"bird_{sample_id:04d}.png"
                mask = Image.new("RGB", (40, 32), (128, 0, 0))
                ImageDraw.Draw(mask).rectangle(
                    (0, 0, 3, 31), fill=(0, 0, 0)
                )
                mask.save(path)
            phase0 = root / "phase0"
            phase0_config = fixture_config(dataset, masks, phase0)
            phase0_config["transforms"].update(
                {
                    "image_size": 224,
                    "evaluation_resize_shortest": 256,
                }
            )
            build_phase0(phase0_config)
            approve_visual_audit(
                phase0, reviewer="capacity filter fixture", confirmation=True
            )
            config = load_set_expert_config(
                ROOT / "configs" / "expert_background_set.yaml",
                seed=1401,
                phase0_dir=str(phase0),
                device="cpu",
            )
            training = SetBackgroundDataset(
                phase0 / "splits" / "waterbirds95_candidate_train.csv",
                phase0_config,
                config,
                training=True,
            )
            with self.assertRaisesRegex(
                DataValidationError, "do not match the locked"
            ):
                training.apply_training_capacity_eligibility()
            report = training.apply_training_capacity_eligibility(
                enforce_expected=False
            )
            self.assertEqual(report["original_sample_count"], 32)
            self.assertGreater(report["excluded_sample_count"], 0)
            self.assertEqual(
                report["retained_sample_count"]
                + report["excluded_sample_count"],
                32,
            )
            self.assertEqual(len(training), report["retained_sample_count"])
            self.assertTrue(
                all(
                    item["best_fixed_view_capacity"] < 16
                    for item in report["excluded_samples"]
                )
            )
            self.assertTrue(report["candidate_erm_training_unchanged"])
            self.assertFalse(report["expected_exclusions_verified"])
            self.assertEqual(
                report["capacity_census_provenance"]["job_id"], "22266"
            )


if __name__ == "__main__":
    unittest.main()
