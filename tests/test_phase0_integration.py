from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from fixture_data import create_fixture, fixture_config
from setv.errors import ArtifactExistsError, DataValidationError
from setv.phase0 import approve_visual_audit, build_phase0, verify_phase0


class Phase0IntegrationTests(unittest.TestCase):
    def test_build_approve_verify_and_information_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, masks = create_fixture(root)
            output = root / "phase0"
            config = fixture_config(dataset, masks, output)
            built = build_phase0(config)
            self.assertEqual(built, output.resolve())

            pending = verify_phase0(output, require_approval=False)
            self.assertEqual(pending["status"], "automated_checks_passed")
            visual_pending = json.loads(
                (
                    output / "mask_audit" / "visual_review_pending.json"
                ).read_text()
            )
            self.assertEqual(len(visual_pending["green_view_galleries"]), 3)
            for page in visual_pending["green_view_galleries"]:
                self.assertTrue((output / "mask_audit" / page["path"]).is_file())
            with self.assertRaisesRegex(DataValidationError, "approval is missing"):
                verify_phase0(output)

            biased = pd.read_csv(
                output / "splits" / "waterbirds95_biased_val.csv"
            )
            self.assertNotIn("place", biased.columns)
            self.assertNotIn("group", biased.columns)
            protected = pd.read_csv(
                output / "private_analysis" / "protected_group_labels.csv"
            )
            self.assertIn("place", protected.columns)
            self.assertIn("group", protected.columns)

            approval = approve_visual_audit(
                output, reviewer="fixture reviewer", confirmation=True
            )
            self.assertTrue(approval.is_file())
            verified = verify_phase0(output)
            self.assertEqual(verified["status"], "phase0_complete")
            self.assertEqual(
                verified["visual_review"]["reviewer"], "fixture reviewer"
            )
            self.assertEqual(
                len(verified["visual_review"]["green_view_galleries"]),
                3,
            )

            with self.assertRaises(ArtifactExistsError):
                build_phase0(config)

    def test_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, masks = create_fixture(root)
            output = root / "phase0"
            build_phase0(fixture_config(dataset, masks, output))
            summary = output / "splits" / "split_summary.json"
            content = json.loads(summary.read_text(encoding="utf-8"))
            content["seed"] = 9
            summary.write_text(json.dumps(content), encoding="utf-8")
            with self.assertRaisesRegex(DataValidationError, "changed"):
                verify_phase0(output, require_approval=False)

    def test_build_does_not_require_official_test_masks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, masks = create_fixture(root)
            output = root / "phase0"
            # Fixture sample 12 belongs to official split 2.
            (masks / "bird_0012.png").unlink()
            build_phase0(fixture_config(dataset, masks, output))
            test_manifest = pd.read_csv(
                output / "splits" / "waterbirds95_test.csv",
                keep_default_na=False,
            )
            self.assertTrue((test_manifest["mask_relative_path"] == "").all())
            mask_manifest = pd.read_csv(
                output / "masks" / "vlm_mask_manifest.csv"
            )
            self.assertEqual(len(mask_manifest), 48)


if __name__ == "__main__":
    unittest.main()
