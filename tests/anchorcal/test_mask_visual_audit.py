from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from anchorcal.errors import PreflightError  # noqa: E402
from anchorcal.io import sha256_file  # noqa: E402
from anchorcal.mask_visual_audit import (  # noqa: E402
    MASK_VISUAL_AUDIT_SCHEMA,
    render_mask_visual_audit,
    select_mask_visual_audit_samples,
    verify_mask_visual_audit,
)
from anchorcal.preflight import validate_images_and_masks  # noqa: E402
from anchorcal.vlm_masks import producer_vlm_mask_name  # noqa: E402


class MaskVisualAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.image_root = root / "waterbird_1.0_forest2water2"
        self.mask_root = root / "prediction_cmap"
        self.output = root / "output_a"
        self.mask_root.mkdir(parents=True)
        rows: list[dict[str, object]] = []
        img_id = 1000
        # The public gallery draws only from the two aligned split-0 cells:
        # three area strata times three representatives per stratum gives the
        # locked 18 examples. Split-1 rows are still audited into the protected
        # mask namespace but cannot enter this selector-visible fixture.
        cells = {
            0: ((0, 0), (1, 1)),
            1: ((0, 0), (0, 1), (1, 0), (1, 1)),
        }
        for split, split_cells in cells.items():
            for label, place in split_cells:
                sides = (
                    (2, 4, 6, 8, 10, 12, 14, 16, 20)
                    if split == 0
                    else (3, 9, 17)
                )
                for area_rank, side in enumerate(sides):
                    filename = (
                        f"class_{label}/bird_{img_id:05d}_{split}_{place}_{area_rank}.jpg"
                    )
                    image_path = self.image_root / filename
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image = np.zeros((24, 24, 3), dtype=np.uint8)
                    image[..., 0] = (img_id * 3) % 251
                    image[..., 1] = (img_id * 5) % 251
                    image[..., 2] = (img_id * 7) % 251
                    Image.fromarray(image, mode="RGB").save(image_path)

                    mask = np.zeros((24, 24, 3), dtype=np.uint8)
                    mask[:side, :side] = (128, 0, 0)
                    Image.fromarray(mask, mode="RGB").save(
                        self.mask_root / producer_vlm_mask_name(filename)
                    )
                    rows.append(
                        {
                            "img_id": img_id,
                            "img_filename": filename,
                            "y": label,
                            "place": place,
                            "split": split,
                            "metadata_row_index": len(rows),
                        }
                    )
                    img_id += 1
        self.frame = pd.DataFrame(rows)
        self.config = {
            "paths": {
                "waterbirds_root": str(self.image_root),
                "metadata_path": str(self.image_root / "metadata.csv"),
                "vlm_mask_root": str(self.mask_root),
                "output_root": str(self.output),
            },
            "masks": {
                "minimum_foreground_fraction": 0.0,
                "maximum_foreground_fraction": 1.0,
            },
            "resolved_config_sha256": "a" * 64,
        }
        self.metadata_sha256 = "b" * 64
        self.mask_manifest = validate_images_and_masks(
            self.config,
            self.frame,
            metadata_sha256=self.metadata_sha256,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_selection_covers_context_cells_without_serializing_context(self) -> None:
        selected = select_mask_visual_audit_samples(
            self.frame.sample(frac=1.0, random_state=91),
            {
                **self.mask_manifest,
                "entries": list(reversed(self.mask_manifest["entries"])),
            },
        )
        self.assertEqual(len(selected), 18)
        self.assertEqual({sample["official_split"] for sample in selected}, {0})
        self.assertEqual(
            {sample["area_stratum"] for sample in selected},
            {"low", "middle", "high"},
        )
        for sample in selected:
            self.assertNotIn("place", sample)
            self.assertNotIn("group", sample)

    def test_render_is_deterministic_and_hash_bound(self) -> None:
        first = render_mask_visual_audit(
            self.config, self.frame, self.mask_manifest
        )
        second_config = {
            **self.config,
            "paths": {
                **self.config["paths"],
                "output_root": str(Path(self.temporary.name) / "output_b"),
            },
        }
        second = render_mask_visual_audit(
            second_config,
            self.frame.sample(frac=1.0, random_state=19),
            {
                **self.mask_manifest,
                "entries": list(reversed(self.mask_manifest["entries"])),
            },
        )

        self.assertEqual(first["schema_version"], MASK_VISUAL_AUDIT_SCHEMA)
        self.assertFalse(first["human_approval_required"])
        self.assertEqual(first["sample_count"], 18)
        self.assertEqual(first["page_count"], 3)
        self.assertEqual(first["sample_ids"], second["sample_ids"])
        self.assertEqual(first["selection_sha256"], second["selection_sha256"])
        self.assertEqual(
            [page["sha256"] for page in first["pages"]],
            [page["sha256"] for page in second["pages"]],
        )
        self.assertEqual(
            first["settings"]["observed_context_cell_count_by_split"],
            {"0": 2},
        )

        manifest_path = self.output / first["manifest_relative_path"]
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["sample_count"], 18)
        self.assertEqual(payload["page_count"], 3)
        self.assertEqual(
            payload["foreground_area_summary_sha256"],
            first["foreground_area_summary_sha256"],
        )
        for sample in payload["samples"]:
            self.assertNotIn("place", sample)
            self.assertNotIn("group", sample)
            self.assertEqual(len(sample["image_sha256"]), 64)
            self.assertEqual(len(sample["mask_sha256"]), 64)
        verified = verify_mask_visual_audit(
            self.output,
            first,
            expected_mask_bank_sha256=self.mask_manifest["mask_bank_sha256"],
            expected_metadata_sha256=self.metadata_sha256,
        )
        self.assertEqual(verified["sample_ids"], first["sample_ids"])

    def test_page_tamper_and_underpopulated_context_cell_fail_closed(self) -> None:
        receipt = render_mask_visual_audit(
            self.config, self.frame, self.mask_manifest
        )
        page = self.output / receipt["pages"][0]["relative_path"]
        original = page.read_bytes()
        page.write_bytes(original + b"tamper")
        with self.assertRaisesRegex(PreflightError, "page failed provenance"):
            verify_mask_visual_audit(
                self.output,
                receipt,
                expected_mask_bank_sha256=self.mask_manifest["mask_bank_sha256"],
                expected_metadata_sha256=self.metadata_sha256,
            )

        first_cell = self.frame.loc[
            (self.frame["split"] == 0)
            & (self.frame["y"] == 0)
            & (self.frame["place"] == 0)
        ]
        underpopulated = self.frame.drop(first_cell.index[-1])
        with self.assertRaisesRegex(PreflightError, "split/class/context cell"):
            select_mask_visual_audit_samples(underpopulated, self.mask_manifest)

    def test_source_hashes_and_extreme_area_bins_are_explicit(self) -> None:
        for entry in self.mask_manifest["entries"]:
            source = self.image_root / entry["image_relative_path"]
            self.assertEqual(entry["image_sha256"], sha256_file(source))
            self.assertEqual(entry["image_size_bytes"], source.stat().st_size)
        summary = self.mask_manifest["foreground_area_summary"]
        self.assertEqual(summary["overall"]["count"], 18)
        self.assertEqual(summary["overall"]["empty_count"], 0)
        self.assertEqual(summary["overall"]["full_count"], 0)
        self.assertEqual(
            summary["failure_policy"]["very_small_or_very_large"],
            "reporting_only_not_a_failure_gate",
        )

    def test_render_rejects_source_image_changed_after_manifest(self) -> None:
        first = self.mask_manifest["entries"][0]
        source = self.image_root / first["image_relative_path"]
        source.write_bytes(source.read_bytes() + b"changed-after-freeze")
        with self.assertRaisesRegex(
            PreflightError, "visual-audit Waterbirds source changed"
        ):
            render_mask_visual_audit(self.config, self.frame, self.mask_manifest)


if __name__ == "__main__":
    unittest.main()
