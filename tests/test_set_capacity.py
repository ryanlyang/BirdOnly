from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from fixture_data import create_fixture, fixture_config
from setv.experts.set_capacity import run_set_capacity_census
from setv.experts.set_config import load_set_expert_config
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


if __name__ == "__main__":
    unittest.main()
