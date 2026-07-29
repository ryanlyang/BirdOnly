from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from audit_phase0_mask_mapping import audit_mapping
from fixture_data import create_fixture, fixture_config


class Phase0MaskMappingAuditTests(unittest.TestCase):
    def test_complete_and_missing_reports_cover_every_metadata_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, masks = create_fixture(root)
            config = fixture_config(dataset, masks, root / "phase0")
            config["audit"]["visual_samples_per_split"] = 20
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))

            complete = audit_mapping(config_path)
            self.assertTrue(complete["accepted"])
            self.assertEqual(complete["mapped_row_count"], 56)
            self.assertEqual(complete["required_mapped_row_count"], 48)
            self.assertEqual(complete["required_row_count"], 48)
            self.assertEqual(complete["missing_or_ambiguous_count"], 0)
            self.assertFalse(complete["reporting_only_metric_values_included"])

            (masks / "bird_0000.png").unlink()
            gallery_dir = root / "galleries"
            incomplete = audit_mapping(
                config_path,
                gallery_dir=gallery_dir,
                samples_per_split=4,
                samples_per_page=2,
                thumbnail_size=64,
            )
            self.assertFalse(incomplete["accepted"])
            self.assertEqual(incomplete["mapped_row_count"], 55)
            self.assertEqual(incomplete["missing_or_ambiguous_count"], 1)
            self.assertEqual(
                incomplete["missing_or_ambiguous_samples"][0]["sample_id"],
                "0",
            )
            gallery = incomplete["visual_gallery"]
            self.assertEqual(gallery["page_count"], 6)
            self.assertEqual(len(list(gallery_dir.glob("*.png"))), 6)
            self.assertEqual(
                gallery["pages"][0]["views"],
                [
                    "original",
                    "mask overlay (red)",
                    "bird + green background",
                    "background + green bird",
                ],
            )
            with Image.open(gallery["pages"][0]["path"]) as page:
                self.assertEqual(page.width, 4 * 64)
                self.assertGreater(page.height, 2 * 64)

    def test_missing_official_test_mask_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, masks = create_fixture(root)
            config = fixture_config(dataset, masks, root / "phase0")
            config["audit"]["visual_samples_per_split"] = 20
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))

            # Fixture sample 12 is the first official-test row.
            (masks / "bird_0012.png").unlink()
            report = audit_mapping(config_path)
            self.assertTrue(report["accepted"])
            self.assertEqual(report["required_mapped_row_count"], 48)
            self.assertEqual(report["optional_missing_count"], 1)


if __name__ == "__main__":
    unittest.main()
