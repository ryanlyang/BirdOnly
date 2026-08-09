"""Focused release and CUB-mask path/provenance preflight contracts."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from anchorcal.errors import PreflightError  # noqa: E402
from anchorcal.io import sha256_file  # noqa: E402
from anchorcal.preflight import (  # noqa: E402
    run_preflight,
    validate_images_and_masks,
    validate_release,
)


class PreflightPathContractTests(unittest.TestCase):
    """Exercise the path rules without invoking model or GPU preflight."""

    RELEASE = "waterbird_complete95_forest2water2"
    IMAGE_RELATIVE_PATH = Path(
        "001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg"
    )
    MASK_RELATIVE_PATH = IMAGE_RELATIVE_PATH.with_suffix(".png")

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.temporary_root = Path(self._temporary.name)
        self.release_root = self.temporary_root / self.RELEASE
        self.source_root = self.temporary_root / "cub_source_segmentations"
        self.final_root = self.temporary_root / "waterbirds_coordinate_masks"
        self._write_image(self.release_root / self.IMAGE_RELATIVE_PATH)
        self._write_metadata(self.release_root / "metadata.csv")

    def tearDown(self) -> None:
        self._temporary.cleanup()

    @staticmethod
    def _write_image(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.zeros((12, 16, 3), dtype=np.uint8)
        image[..., 0] = 31
        image[..., 1] = 83
        image[..., 2] = 149
        Image.fromarray(image, mode="RGB").save(path)

    @staticmethod
    def _write_mask(path: Path, *, shifted: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        mask = np.zeros((12, 16), dtype=np.uint8)
        if shifted:
            mask[4:10, 8:14] = 255
        else:
            mask[2:8, 3:9] = 255
        Image.fromarray(mask, mode="L").save(path)

    def _write_metadata(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "img_id,img_filename,y,place,split\n"
            f"17,{self.IMAGE_RELATIVE_PATH.as_posix()},0,0,0\n",
            encoding="utf-8",
        )

    def _config(self, *, source_root: Path, final_root: Path) -> dict[str, object]:
        return {
            "data": {"release": self.RELEASE},
            "paths": {
                "waterbirds_root": str(self.release_root),
                "metadata_path": str(self.release_root / "metadata.csv"),
                "cub_source_segmentation_root": str(source_root),
                "cub_waterbirds_mask_root": str(final_root),
            },
        }

    def _write_mapping_manifest(
        self,
        *,
        source_relative_path: str | None = None,
        final_relative_path: str | None = None,
        source_sha256: str | None = None,
        final_sha256: str | None = None,
    ) -> Path:
        source_file = self.source_root / self.MASK_RELATIVE_PATH
        final_file = self.final_root / self.MASK_RELATIVE_PATH
        manifest = {
            "schema_version": "anchorcal-cub-waterbirds-mask-mapping-v1",
            "source_segmentation_root": str(self.source_root.resolve()),
            "generation_method": "unit-test recorded relative-stem mapping",
            "entries": [
                {
                    "img_id": 17,
                    "source_relative_path": (
                        self.MASK_RELATIVE_PATH.as_posix()
                        if source_relative_path is None
                        else source_relative_path
                    ),
                    "final_relative_path": (
                        self.MASK_RELATIVE_PATH.as_posix()
                        if final_relative_path is None
                        else final_relative_path
                    ),
                    "source_sha256": (
                        sha256_file(source_file)
                        if source_sha256 is None
                        else source_sha256
                    ),
                    "final_sha256": (
                        sha256_file(final_file)
                        if final_sha256 is None
                        else final_sha256
                    ),
                }
            ],
        }
        path = self.final_root / "anchorcal_mapping_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_valid_same_tree_has_exactly_one_mask_per_image(self) -> None:
        self._write_mask(self.final_root / self.MASK_RELATIVE_PATH)
        config = self._config(source_root=self.final_root, final_root=self.final_root)

        frame, metadata_hash = validate_release(config)
        manifest = validate_images_and_masks(config, frame)

        self.assertEqual(len(metadata_hash), 64)
        self.assertTrue(manifest["same_tree"])
        self.assertEqual(manifest["count"], 1)
        self.assertIsNone(manifest["mapping_manifest"])
        self.assertEqual([entry["img_id"] for entry in manifest["entries"]], [17])
        self.assertEqual(
            manifest["entries"][0]["relative_path"],
            self.MASK_RELATIVE_PATH.as_posix(),
        )

    def test_symlinked_image_cannot_escape_release_root(self) -> None:
        outside = self.temporary_root / "outside.jpg"
        self._write_image(outside)
        image_file = self.release_root / self.IMAGE_RELATIVE_PATH
        image_file.unlink()
        image_file.symlink_to(outside)
        self._write_mask(self.final_root / self.MASK_RELATIVE_PATH)
        config = self._config(source_root=self.final_root, final_root=self.final_root)
        frame, _ = validate_release(config)

        with self.assertRaisesRegex(PreflightError, "image_path_escapes_root"):
            validate_images_and_masks(config, frame)

    def test_separate_roots_without_mapping_manifest_fail_closed(self) -> None:
        self._write_mask(self.source_root / self.MASK_RELATIVE_PATH)
        self._write_mask(self.final_root / self.MASK_RELATIVE_PATH, shifted=True)
        config = self._config(source_root=self.source_root, final_root=self.final_root)
        frame, _ = validate_release(config)

        with self.assertRaisesRegex(
            PreflightError, "separate CUB source/final mask roots require"
        ):
            validate_images_and_masks(config, frame)

    def test_valid_separate_root_mapping_manifest_is_accepted_and_hashed(self) -> None:
        self._write_mask(self.source_root / self.MASK_RELATIVE_PATH)
        self._write_mask(self.final_root / self.MASK_RELATIVE_PATH, shifted=True)
        mapping_path = self._write_mapping_manifest()
        config = self._config(source_root=self.source_root, final_root=self.final_root)
        frame, _ = validate_release(config)

        manifest = validate_images_and_masks(config, frame)

        self.assertFalse(manifest["same_tree"])
        self.assertEqual(manifest["count"], 1)
        self.assertEqual(
            manifest["mapping_manifest"]["sha256"], sha256_file(mapping_path)
        )
        self.assertEqual(
            manifest["mapping_manifest"]["generation_method"],
            "unit-test recorded relative-stem mapping",
        )

    def test_separate_root_hash_or_path_tampering_fails_closed(self) -> None:
        self._write_mask(self.source_root / self.MASK_RELATIVE_PATH)
        self._write_mask(self.final_root / self.MASK_RELATIVE_PATH, shifted=True)
        config = self._config(source_root=self.source_root, final_root=self.final_root)
        frame, _ = validate_release(config)

        tampered_manifests = (
            {"source_sha256": "0" * 64},
            {"final_relative_path": "wrong/class_or_stem.png"},
        )
        for manifest_overrides in tampered_manifests:
            with self.subTest(manifest_overrides=manifest_overrides):
                self._write_mapping_manifest(**manifest_overrides)
                with self.assertRaisesRegex(
                    PreflightError, "mapping_provenance_hash_or_path_mismatch"
                ):
                    validate_images_and_masks(config, frame)

        self._write_mapping_manifest()
        self._write_mask(self.final_root / self.MASK_RELATIVE_PATH, shifted=False)
        with self.assertRaisesRegex(
            PreflightError, "mapping_provenance_hash_or_path_mismatch"
        ):
            validate_images_and_masks(config, frame)

    def test_release_basename_and_authoritative_metadata_placement_are_strict(self) -> None:
        self._write_mask(self.final_root / self.MASK_RELATIVE_PATH)
        config = self._config(source_root=self.final_root, final_root=self.final_root)

        config["paths"]["waterbirds_root"] = str(
            self.temporary_root / "wrong_waterbirds_release"
        )
        with self.assertRaisesRegex(PreflightError, "root basename must be"):
            validate_release(config)

        config = self._config(source_root=self.final_root, final_root=self.final_root)
        external_metadata = self.temporary_root / "metadata.csv"
        self._write_metadata(external_metadata)
        config["paths"]["metadata_path"] = str(external_metadata)
        with self.assertRaisesRegex(
            PreflightError, "authoritative <waterbirds_root>/metadata.csv is missing"
        ):
            validate_release(config)

    def test_run_preflight_persists_every_mask_failure_before_raising(self) -> None:
        rows = ["img_id,img_filename,y,place,split"]
        expected_ids = list(range(25))
        for img_id in expected_ids:
            rows.append(f"{img_id},missing/class_{img_id}.jpg,{img_id % 2},{img_id % 2},0")
        (self.release_root / "metadata.csv").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )
        output_root = self.temporary_root / "outputs"
        config = self._config(
            source_root=self.final_root,
            final_root=self.final_root,
        )
        config["paths"].update(
            {
                "output_root": str(output_root),
                "repo_root": str(ROOT),
            }
        )
        config["runtime"] = {"require_clean_commit": False, "debug": True}

        with patch(
            "anchorcal.preflight.git_state",
            return_value={"commit": "f" * 40, "clean": True},
        ), patch(
            "anchorcal.preflight.save_environment_manifest",
            return_value={"schema_version": "synthetic-environment"},
        ), patch(
            "anchorcal.preflight.write_package_lock",
            return_value=None,
        ), self.assertRaisesRegex(
            PreflightError, "image/mask audit failed for 25 img_ids"
        ):
            run_preflight(config, allow_download=False, require_gh200=False)

        report_path = (
            output_root / "preflight" / "mask_validation_failure_report.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failure_count"], 25)
        self.assertEqual(
            [failure["img_id"] for failure in report["failures"]], expected_ids
        )
        for failure in report["failures"]:
            self.assertEqual(
                failure["reasons"],
                [
                    "missing_image",
                    "missing_source_cub_mask",
                    "missing_waterbirds_coordinate_mask",
                ],
            )

if __name__ == "__main__":
    unittest.main()
