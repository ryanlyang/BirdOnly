"""Focused Waterbirds release and frozen VLM-mask preflight contracts."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import DEFAULT, patch

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from anchorcal.data import image_path  # noqa: E402
from anchorcal.errors import PreflightError  # noqa: E402
from anchorcal.io import atomic_write_json, hash_object, sha256_file  # noqa: E402
from anchorcal.mask_identity import SELECTOR_MASK_RECEIPT_SCHEMA  # noqa: E402
from anchorcal.paths import discover_candidates  # noqa: E402
from anchorcal.prepare import prepare_geometry_artifacts  # noqa: E402
from anchorcal.preflight import (  # noqa: E402
    run_preflight,
    validate_images_and_masks,
    validate_release,
)
from anchorcal.preprocessing import EvaluationPreprocessing  # noqa: E402
from anchorcal.vlm_masks import (  # noqa: E402
    ANALYSIS_ONLY_MASK_AUDIT_RELATIVE_PATH,
    ANALYSIS_ONLY_MASK_AUDIT_SCHEMA,
    VLM_DECODER_VERSION,
    VLM_MAPPING_VERSION,
    VLM_MASK_MANIFEST_SCHEMA,
    VLM_PRODUCER,
    decode_vlm_mask,
    load_analysis_only_mask_audit,
    load_preflight_geometry_mask_bank,
    load_vlm_mask_bank,
    producer_vlm_mask_name,
    teacher_map_candidates,
)
from tests.anchorcal.visual_audit_fixture import (  # noqa: E402
    attach_synthetic_visual_audit,
)


class PreflightPathContractTests(unittest.TestCase):
    """Exercise the authoritative producer join without model/GPU preflight."""

    RELEASE = "waterbird_1.0_forest2water2"
    TRAIN_IMAGE = (
        "001.Black_footed_Albatross/"
        "Black_Footed_Albatross_0001_796111.jpg"
    )
    VAL_IMAGE = "images/002.Laysan_Albatross/Laysan_Albatross_0002_100.jpg"
    TEST_IMAGE = "003.Sooty_Albatross/Sooty_Albatross_0003_200.jpg"

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.temporary_root = Path(self._temporary.name)
        self.release_root = self.temporary_root / self.RELEASE
        self.mask_root = self.temporary_root / "prediction_cmap"
        self.output_root = self.temporary_root / "outputs"
        self.mask_root.mkdir(parents=True)
        self.rows = [
            (17, self.TRAIN_IMAGE, 0, 0, 0),
            (18, self.VAL_IMAGE, 1, 1, 1),
            (19, self.TEST_IMAGE, 0, 1, 2),
        ]
        self._write_metadata(self.rows)
        for _, filename, *_ in self.rows:
            self._write_image(self.release_root / filename)
        self._write_voc_mask(self._producer_path(self.TRAIN_IMAGE))
        self._write_voc_mask(self._producer_path(self.VAL_IMAGE), shifted=True)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    @staticmethod
    def _write_image(path: Path, *, size: tuple[int, int] = (16, 12)) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        width, height = size
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[..., 0] = 31
        image[..., 1] = 83
        image[..., 2] = 149
        Image.fromarray(image, mode="RGB").save(path)

    @staticmethod
    def _write_voc_mask(
        path: Path,
        *,
        shifted: bool = False,
        size: tuple[int, int] = (16, 12),
        foreground_rgb: tuple[int, int, int] = (128, 0, 0),
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        width, height = size
        mask = np.zeros((height, width, 3), dtype=np.uint8)
        if shifted:
            mask[4:10, 8:14] = foreground_rgb
        else:
            mask[2:8, 3:9] = foreground_rgb
        Image.fromarray(mask, mode="RGB").save(path)

    def _write_metadata(self, rows: list[tuple[int, str, int, int, int]]) -> None:
        path = self.release_root / "metadata.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["img_id,img_filename,y,place,split"]
        lines.extend(
            f"{img_id},{filename},{label},{place},{split}"
            for img_id, filename, label, place, split in rows
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _producer_path(self, filename: str) -> Path:
        return self.mask_root / producer_vlm_mask_name(filename)

    def _config(self) -> dict[str, object]:
        return {
            "data": {"release": self.RELEASE},
            "masks": {
                "minimum_foreground_fraction": 0.0,
                "maximum_foreground_fraction": 1.0,
            },
            "paths": {
                "waterbirds_root": str(self.release_root),
                "metadata_path": str(self.release_root / "metadata.csv"),
                "vlm_mask_root": str(self.mask_root),
                "output_root": str(self.output_root),
            },
            "resolved_config_sha256": "a" * 64,
        }

    def _validate(self) -> tuple[dict[str, object], str]:
        config = self._config()
        frame, metadata_hash = validate_release(config)
        manifest = validate_images_and_masks(
            config,
            frame,
            metadata_sha256=metadata_hash,
        )
        return manifest, metadata_hash

    def _freeze_manifest(
        self, manifest: dict[str, object], *, include_report: bool = True
    ) -> Path:
        preflight = self.output_root / "preflight"
        preflight.mkdir(parents=True, exist_ok=True)
        path = preflight / "mask_manifest.json"
        manifest = copy.deepcopy(manifest)
        manifest["git_commit"] = "deadbeef"
        visual_audit = attach_synthetic_visual_audit(self.output_root, manifest)
        manifest["visual_audit"] = visual_audit
        atomic_write_json(path, manifest)
        selector_receipt = {
            "schema_version": SELECTOR_MASK_RECEIPT_SCHEMA,
            "status": "passed",
            "namespace": "selector_visible",
            "contains_per_row_records": False,
            "selector_required_official_splits": [0],
            "resolved_config_sha256": self._config()[
                "resolved_config_sha256"
            ],
            "metadata_sha256": manifest["metadata_sha256"],
            "git_commit": "deadbeef",
            "mask_source": VLM_PRODUCER,
            "mask_contract_sha256": manifest["mask_contract_sha256"],
            "mask_bank_sha256": manifest["mask_bank_sha256"],
            "mask_manifest_sha256": sha256_file(path),
            "foreground_area_summary_sha256": hash_object(
                manifest["foreground_area_summary"]
            ),
            "mask_visual_audit_manifest_sha256": visual_audit[
                "manifest_sha256"
            ],
            "mask_visual_audit_selection_sha256": visual_audit[
                "selection_sha256"
            ],
        }
        selector_receipt_path = preflight / "selector_mask_receipt.json"
        atomic_write_json(selector_receipt_path, selector_receipt)
        if include_report:
            atomic_write_json(
                preflight / "report.json",
                {
                    "schema_version": "anchorcal-preflight-v1",
                    "status": "passed",
                    "resolved_config_sha256": self._config()[
                        "resolved_config_sha256"
                    ],
                    "resolved_paths": self._config()["paths"],
                    "metadata_sha256": manifest["metadata_sha256"],
                    "mask_source": VLM_PRODUCER,
                    "mask_contract_sha256": manifest["mask_contract_sha256"],
                    "mask_bank_sha256": manifest["mask_bank_sha256"],
                    "foreground_area_summary": manifest[
                        "foreground_area_summary"
                    ],
                    "mask_visual_audit": visual_audit,
                    "mask_manifest_sha256": sha256_file(path),
                    "selector_mask_receipt": {
                        "schema_version": SELECTOR_MASK_RECEIPT_SCHEMA,
                        "sha256": sha256_file(selector_receipt_path),
                    },
                },
            )
        return path

    def test_producer_name_uses_the_complete_relative_path(self) -> None:
        self.assertEqual(
            producer_vlm_mask_name(
                "200.Common_Yellowthroat/"
                "Common_Yellowthroat_0071_190665.jpg"
            ),
            "200_Common_Yellowthroat_"
            "Common_Yellowthroat_0071_190665.png",
        )
        self.assertEqual(
            producer_vlm_mask_name(self.VAL_IMAGE),
            "images_002_Laysan_Albatross_Laysan_Albatross_0002_100.png",
        )
        self.assertEqual(
            producer_vlm_mask_name("a folder/species.name/bird (1).jpeg"),
            "a_folder_species_name_bird_1.png",
        )
        for unsafe in ("/absolute/bird.jpg", "../escape/bird.jpg", "C:/bird.jpg"):
            with self.subTest(unsafe=unsafe), self.assertRaises(PreflightError):
                producer_vlm_mask_name(unsafe)

    def test_voc_red_decodes_as_class_one_not_a_grayscale_threshold(self) -> None:
        decoded = decode_vlm_mask(self._producer_path(self.TRAIN_IMAGE))
        self.assertEqual(decoded.class_ids, (0, 1))
        self.assertEqual(decoded.rgb_colors, ((0, 0, 0), (128, 0, 0)))
        self.assertEqual(decoded.class_pixel_counts[1], 36)
        self.assertTrue(decoded.binary[2, 3])
        self.assertFalse(decoded.binary[0, 0])

    def test_required_split_coverage_and_optional_test_inventory(self) -> None:
        manifest, metadata_hash = self._validate()

        self.assertEqual(manifest["schema_version"], VLM_MASK_MANIFEST_SCHEMA)
        self.assertEqual(manifest["metadata_sha256"], metadata_hash)
        self.assertEqual(manifest["mapping_version"], VLM_MAPPING_VERSION)
        self.assertEqual(manifest["decoder_version"], VLM_DECODER_VERSION)
        self.assertEqual(manifest["selector_required_official_splits"], [0])
        self.assertEqual(manifest["analysis_only_audit_official_splits"], [1])
        self.assertEqual(manifest["required_count"], 1)
        self.assertEqual(
            manifest["required_mapping_audit"],
            {
                "expected": 1,
                "resolved_unique": 1,
                "missing": 0,
                "ambiguous": 0,
                "reused": 0,
                "producer_name_collisions": 0,
            },
        )
        self.assertEqual(manifest["coverage"]["0"], {"expected": 1, "present": 1})
        self.assertNotIn("1", manifest["coverage"])
        self.assertEqual(
            manifest["optional_split_inventory"],
            {"expected": 1, "missing": 1, "unique": 0, "ambiguous": 0},
        )
        self.assertEqual([entry["img_id"] for entry in manifest["entries"]], [17])
        self.assertNotIn("metadata_index", manifest["entries"][0])
        self.assertEqual(
            manifest["mapping_rule_counts"],
            {"weclip_producer_flattened_relative_stem": 1},
        )
        self.assertEqual(manifest["observed_rgb_colors"], [[0, 0, 0], [128, 0, 0]])

        # Split 2 remains reporting-only inventory: a map may exist, but it is
        # never inserted into the runtime bank.
        self._write_voc_mask(self._producer_path(self.TEST_IMAGE))
        (self.mask_root / "stale_unmapped.png").write_bytes(
            self._producer_path(self.TRAIN_IMAGE).read_bytes()
        )
        with_optional, _ = self._validate()
        self.assertEqual(
            with_optional["optional_split_inventory"],
            {"expected": 1, "missing": 0, "unique": 1, "ambiguous": 0},
        )
        self.assertEqual(
            [entry["img_id"] for entry in with_optional["entries"]], [17]
        )
        self.assertEqual(
            with_optional["extras_inventory"],
            {"count": 1, "relative_paths": ["stale_unmapped.png"]},
        )

    def test_missing_required_map_fails_but_missing_test_map_does_not(self) -> None:
        self._producer_path(self.VAL_IMAGE).unlink()
        config = self._config()
        frame, metadata_hash = validate_release(config)
        report = self.output_root / "mask_validation_failure_report.json"
        with self.assertRaisesRegex(PreflightError, "analysis-only mask audit failed"):
            validate_images_and_masks(
                config,
                frame,
                metadata_sha256=metadata_hash,
                failure_report_path=report,
            )
        failure = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(failure["status"], "failed")
        self.assertEqual(failure["coverage"]["0"]["present"], 1)
        self.assertEqual(failure["coverage"]["1"]["present"], 0)
        self.assertEqual(failure["failures"], [])
        protected = json.loads(
            (self.output_root / ANALYSIS_ONLY_MASK_AUDIT_RELATIVE_PATH).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(protected["status"], "failed")
        self.assertEqual(protected["failures"][0]["img_id"], 18)

    def test_public_artifacts_cannot_reveal_official_split1_membership(self) -> None:
        manifest, _ = self._validate()
        manifest_path = self._freeze_manifest(manifest)
        protected = load_analysis_only_mask_audit(self._config())
        self.assertEqual(protected["schema_version"], ANALYSIS_ONLY_MASK_AUDIT_SCHEMA)
        self.assertEqual([entry["img_id"] for entry in protected["entries"]], [18])
        self.assertIn("metadata_index", protected["entries"][0])

        public_paths = (
            manifest_path,
            self.output_root / "preflight" / "report.json",
            self.output_root / "preflight" / "selector_mask_receipt.json",
            self.output_root / "preflight" / "mask_visual_audit" / "manifest.json",
        )
        for path in public_paths:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(self.VAL_IMAGE, text, path)
            self.assertNotIn('"img_id": 18', text, path)
            self.assertNotIn('"metadata_index"', text, path)
            self.assertNotIn("oracle_val_mask_audit", text, path)

        bank = load_vlm_mask_bank(self._config())
        with self.assertRaisesRegex(PreflightError, "no matching row"):
            bank.load(18, self.VAL_IMAGE)

    def test_ambiguous_exact_and_legacy_layouts_fail_closed(self) -> None:
        legacy = dict(teacher_map_candidates(self.mask_root, self.VAL_IMAGE))[
            "legacy_one_parent_flat"
        ]
        self.assertNotEqual(legacy.resolve(), self._producer_path(self.VAL_IMAGE).resolve())
        self._write_voc_mask(legacy)

        config = self._config()
        frame, metadata_hash = validate_release(config)
        with self.assertRaisesRegex(PreflightError, "analysis-only mask audit failed"):
            validate_images_and_masks(
                config,
                frame,
                metadata_sha256=metadata_hash,
            )

    def test_flattened_producer_collision_is_rejected_without_suffix_guessing(self) -> None:
        collision_rows = [
            (1, "class.a/bird.jpg", 0, 0, 0),
            (2, "class_a/bird.jpg", 1, 1, 1),
            (3, "test/bird.jpg", 0, 1, 2),
        ]
        self._write_metadata(collision_rows)
        for _, filename, *_ in collision_rows:
            self._write_image(self.release_root / filename)
        collision_name = producer_vlm_mask_name(collision_rows[0][1])
        self.assertEqual(collision_name, producer_vlm_mask_name(collision_rows[1][1]))
        self._write_voc_mask(self.mask_root / collision_name)
        # A producer traversal suffix cannot be reconstructed from metadata and
        # therefore must not make the collision acceptable.
        self._write_voc_mask(self.mask_root / collision_name.replace(".png", "_1.png"))

        config = self._config()
        frame, metadata_hash = validate_release(config)
        report = self.output_root / "collision_failure.json"
        with self.assertRaisesRegex(PreflightError, "producer_name_collision"):
            validate_images_and_masks(
                config,
                frame,
                metadata_sha256=metadata_hash,
                failure_report_path=report,
            )
        payload = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["producer_name_collisions"]), 1)
        self.assertEqual(
            payload["producer_name_collisions"][0]["producer_mask_name"],
            collision_name,
        )

    def test_optional_test_row_cannot_create_a_fatal_producer_collision(self) -> None:
        rows = [
            (1, "class.a/bird.jpg", 0, 0, 0),
            (2, "other/validation.jpg", 1, 1, 1),
            (3, "class_a/bird.jpg", 0, 1, 2),
        ]
        self._write_metadata(rows)
        for _, filename, *_ in rows:
            self._write_image(self.release_root / filename)
        self._write_voc_mask(self._producer_path(rows[0][1]))
        self._write_voc_mask(self._producer_path(rows[1][1]), shifted=True)

        manifest, _ = self._validate()

        self.assertEqual(manifest["producer_name_collisions"], [])
        self.assertEqual([entry["img_id"] for entry in manifest["entries"]], [1])
        self.assertEqual(
            manifest["optional_split_inventory"],
            {"expected": 1, "missing": 0, "unique": 1, "ambiguous": 0},
        )

    def test_dimension_mismatch_and_unknown_voc_color_both_fail(self) -> None:
        cases = (
            (
                "dimension",
                {"size": (15, 12)},
                "mask_dimension_mismatch",
            ),
            (
                "unknown_color",
                {"foreground_rgb": (1, 2, 3)},
                "unexpected non-VOC colors",
            ),
        )
        for name, options, expected in cases:
            with self.subTest(name=name):
                self._write_voc_mask(self._producer_path(self.VAL_IMAGE), **options)
                config = self._config()
                frame, metadata_hash = validate_release(config)
                with self.assertRaisesRegex(
                    PreflightError, "analysis-only mask audit failed"
                ):
                    validate_images_and_masks(
                        config,
                        frame,
                        metadata_sha256=metadata_hash,
                    )
                protected_failure = (
                    self.output_root / ANALYSIS_ONLY_MASK_AUDIT_RELATIVE_PATH
                ).read_text(encoding="utf-8")
                self.assertIn(expected, protected_failure)
                self._write_voc_mask(
                    self._producer_path(self.VAL_IMAGE), shifted=True
                )

    def test_symlinked_image_cannot_escape_release_root(self) -> None:
        outside = self.temporary_root / "outside.jpg"
        self._write_image(outside)
        image_file = self.release_root / self.TRAIN_IMAGE
        image_file.unlink()
        image_file.symlink_to(outside)
        config = self._config()
        frame, metadata_hash = validate_release(config)
        with self.assertRaisesRegex(PreflightError, "image path escapes its root"):
            validate_images_and_masks(
                config,
                frame,
                metadata_sha256=metadata_hash,
            )

    def test_images_subdirectory_layout_is_supported(self) -> None:
        source = self.release_root / self.TRAIN_IMAGE
        nested = self.release_root / "images" / self.TRAIN_IMAGE
        nested.parent.mkdir(parents=True, exist_ok=True)
        source.rename(nested)

        config = self._config()
        frame, metadata_hash = validate_release(config)
        manifest = validate_images_and_masks(
            config,
            frame,
            metadata_sha256=metadata_hash,
        )
        train_entry = next(
            entry for entry in manifest["entries"] if entry["img_id"] == 17
        )
        self.assertEqual(train_entry["image_width"], 16)
        self.assertEqual(train_entry["image_height"], 12)

    def test_image_resolution_is_prioritized_deduplicated_and_fail_closed(
        self,
    ) -> None:
        direct = (self.release_root / self.TRAIN_IMAGE).resolve()
        nested = self.release_root / "images" / self.TRAIN_IMAGE
        self.assertEqual(image_path(self.release_root, self.TRAIN_IMAGE), direct)

        nested.parent.mkdir(parents=True, exist_ok=True)
        nested.symlink_to(direct)
        # Two layout candidates that resolve to the same source file are one
        # unambiguous image, not a false-positive duplicate.
        self.assertEqual(image_path(self.release_root, self.TRAIN_IMAGE), direct)

        nested.unlink()
        self._write_image(nested)
        with self.assertRaisesRegex(PreflightError, "ambiguous Waterbirds image"):
            image_path(self.release_root, self.TRAIN_IMAGE)

    def test_release_validation_rejects_any_misaligned_training_row(self) -> None:
        rows = list(self.rows)
        img_id, filename, label, _place, split = rows[0]
        rows[0] = (img_id, filename, label, 1 - label, split)
        self._write_metadata(rows)

        with self.assertRaisesRegex(
            PreflightError, "source training split is not completely correlated"
        ):
            validate_release(self._config())

    def test_release_validation_rejects_empty_oracle_or_test_split(self) -> None:
        for missing_split in (1, 2):
            with self.subTest(missing_split=missing_split):
                self._write_metadata(
                    [row for row in self.rows if row[-1] != missing_split]
                )
                with self.assertRaisesRegex(PreflightError, "must all be nonempty"):
                    validate_release(self._config())

    def test_release_basename_and_authoritative_metadata_placement_are_strict(self) -> None:
        config = self._config()
        config["paths"]["waterbirds_root"] = str(
            self.temporary_root / "wrong_waterbirds_release"
        )
        with self.assertRaisesRegex(PreflightError, "root basename must be"):
            validate_release(config)

    def test_discovery_reports_only_the_current_waterbirds100_vlm_family(self) -> None:
        current = (
            self.temporary_root
            / "code"
            / "results_waterbirds100_openclip_laion_dinovit"
            / "val"
            / "prediction_cmap"
        )
        historical = (
            self.temporary_root
            / "historical"
            / "results"
            / "val"
            / "prediction_cmap"
        )
        wrong_release = (
            self.temporary_root
            / "code"
            / "results_waterbirds95_openclip_laion_dinovit"
            / "val"
            / "prediction_cmap"
        )
        for path in (current, historical, wrong_release):
            path.mkdir(parents=True)
        discovered = discover_candidates([self.temporary_root])
        self.assertEqual(discovered["vlm_mask_root"], [str(current.resolve())])
        self.assertNotIn(str(historical.resolve()), discovered["vlm_mask_root"])
        self.assertNotIn(str(wrong_release.resolve()), discovered["vlm_mask_root"])

        config = self._config()
        external_metadata = self.temporary_root / "metadata.csv"
        external_metadata.write_bytes((self.release_root / "metadata.csv").read_bytes())
        config["paths"]["metadata_path"] = str(external_metadata)
        with self.assertRaisesRegex(
            PreflightError, "authoritative <waterbirds_root>/metadata.csv is missing"
        ):
            validate_release(config)

    def test_manifest_hash_and_runtime_bank_fail_on_manifest_or_source_tamper(self) -> None:
        manifest, _ = self._validate()
        manifest_path = self._freeze_manifest(manifest)
        config = self._config()
        bank = load_vlm_mask_bank(config)
        expected = bank.load(17, self.TRAIN_IMAGE).copy()
        self.assertEqual(int(expected.sum()), 36)
        with self.assertRaisesRegex(PreflightError, "no matching row"):
            bank.load(19, self.TEST_IMAGE)

        original_source = self._producer_path(self.TRAIN_IMAGE).read_bytes()
        self._write_voc_mask(self._producer_path(self.TRAIN_IMAGE), shifted=True)
        with self.assertRaisesRegex(PreflightError, "no longer matches"):
            bank.load(17, self.TRAIN_IMAGE)
        with self.assertRaisesRegex(PreflightError, "source file no longer matches"):
            load_vlm_mask_bank(config)

        self._producer_path(self.TRAIN_IMAGE).write_bytes(original_source)
        source_image = self.release_root / self.TRAIN_IMAGE
        original_image = source_image.read_bytes()
        source_image.write_bytes(original_image + b"changed")
        with self.assertRaisesRegex(
            PreflightError, "Waterbirds source image no longer matches"
        ):
            load_vlm_mask_bank(config)
        source_image.write_bytes(original_image)

        tampered = copy.deepcopy(manifest)
        tampered["entries"][0]["foreground_pixels"] += 1
        atomic_write_json(manifest_path, tampered)
        with self.assertRaisesRegex(
            PreflightError, "entry is incompatible|mask-bank hash is invalid"
        ):
            load_vlm_mask_bank(config)

        malformed = copy.deepcopy(manifest)
        malformed["entries"][0]["class_pixel_counts"] = []
        atomic_write_json(manifest_path, malformed)
        with self.assertRaisesRegex(PreflightError, "entry is malformed"):
            load_vlm_mask_bank(config)

    def test_geometry_bootstrap_breaks_the_final_report_dependency_cycle(self) -> None:
        manifest, _ = self._validate()
        self._freeze_manifest(manifest, include_report=False)
        config = self._config()

        with self.assertRaisesRegex(PreflightError, "finalized preflight report"):
            load_vlm_mask_bank(config)

        bank = load_preflight_geometry_mask_bank(config)
        self.assertEqual(sorted(bank.entries), [17])
        preprocessing = EvaluationPreprocessing(
            input_size=(3, 224, 224),
            interpolation="bicubic",
            antialias=True,
            crop_pct=0.9,
            crop_mode="center",
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            effective_resize_shortest=248,
        )
        with patch(
            "anchorcal.prepare.load_vlm_mask_bank",
            side_effect=AssertionError("strict loader must not run"),
        ) as strict_loader:
            with self.assertRaisesRegex(
                FileNotFoundError, "preflight split is missing"
            ):
                prepare_geometry_artifacts(
                    config,
                    preprocessing=preprocessing,
                    mask_bank=bank,
                )
        strict_loader.assert_not_called()

        self._freeze_manifest(manifest)
        self.assertEqual(sorted(load_vlm_mask_bank(config).entries), [17])
        with self.assertRaisesRegex(
            PreflightError, "cannot replace a finalized report"
        ):
            load_preflight_geometry_mask_bank(config)

    def test_geometry_bootstrap_still_rejects_tampered_visual_artifacts(self) -> None:
        manifest, _ = self._validate()
        manifest_path = self._freeze_manifest(manifest, include_report=False)
        frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
        page_relative = frozen["visual_audit"]["pages"][0]["relative_path"]
        (self.output_root / page_relative).write_bytes(b"tampered")

        with self.assertRaisesRegex(
            PreflightError, "visual-audit page failed provenance verification"
        ):
            load_preflight_geometry_mask_bank(self._config())

    def test_run_preflight_injects_the_staged_bank_before_finalizing_report(
        self,
    ) -> None:
        config = self._config()
        config["paths"].update(
            {
                "repo_root": str(self.temporary_root),
                "fcv_split_manifest_root": str(
                    self.temporary_root / "fcv_split_manifests"
                ),
                "hf_home": str(self.temporary_root / "hf_home"),
            }
        )
        config["data"].update(
            {
                "release": self.RELEASE,
                "fcv_reference": {},
                "expert_calibration_seed": 1729,
                "protected_split_root": "analysis_only/splits",
            }
        )
        metadata_sha256 = "1" * 64
        mask_manifest = {
            "mask_contract_sha256": "2" * 64,
            "mask_bank_sha256": "3" * 64,
            "foreground_area_summary": {"count": 1},
        }
        visual_receipt = {
            "manifest_sha256": "4" * 64,
            "selection_sha256": "5" * 64,
        }
        preprocessing_manifest = {"parity": {"status": "passed"}}
        resolved_preprocessing = SimpleNamespace(effective_resize_shortest=248)
        staged_bank = object()
        events: list[tuple[str, object | None]] = []

        with patch.multiple(
            "anchorcal.preflight",
            assess_storage_budget=DEFAULT,
            git_state=DEFAULT,
            save_environment_manifest=DEFAULT,
            write_package_lock=DEFAULT,
            validate_release=DEFAULT,
            validate_images_and_masks=DEFAULT,
            render_mask_visual_audit=DEFAULT,
            construct_splits=DEFAULT,
            persist_splits=DEFAULT,
            resolve_snapshot=DEFAULT,
            resolve_preprocessing_manifest=DEFAULT,
            write_preprocessing_manifest=DEFAULT,
            preprocessing_from_manifest=DEFAULT,
            load_preflight_geometry_mask_bank=DEFAULT,
            prepare_geometry_artifacts=DEFAULT,
        ) as mocked:
            mocked["assess_storage_budget"].return_value = {"status": "passed"}
            mocked["git_state"].return_value = {
                "commit": "deadbeef",
                "clean": True,
            }
            mocked["save_environment_manifest"].return_value = {
                "status": "passed"
            }
            mocked["validate_release"].return_value = ([object()], metadata_sha256)
            mocked["validate_images_and_masks"].return_value = mask_manifest
            mocked["render_mask_visual_audit"].return_value = visual_receipt
            mocked["construct_splits"].return_value = object()
            mocked["persist_splits"].return_value = {"status": "passed"}
            mocked["resolve_snapshot"].return_value = {"status": "passed"}
            mocked[
                "resolve_preprocessing_manifest"
            ].return_value = preprocessing_manifest
            mocked["write_preprocessing_manifest"].side_effect = (
                lambda path, payload: atomic_write_json(path, payload)
            )
            mocked[
                "preprocessing_from_manifest"
            ].return_value = resolved_preprocessing

            def stage_bank(_: object) -> object:
                events.append(("stage", staged_bank))
                return staged_bank

            def prepare_geometry(
                _: object, *, preprocessing: object, mask_bank: object
            ) -> dict[str, object]:
                self.assertIs(preprocessing, resolved_preprocessing)
                self.assertIs(mask_bank, staged_bank)
                events.append(("geometry", mask_bank))
                return {"status": "passed"}

            mocked["load_preflight_geometry_mask_bank"].side_effect = stage_bank
            mocked["prepare_geometry_artifacts"].side_effect = prepare_geometry

            report = run_preflight(
                config,
                allow_download=False,
                require_gh200=False,
            )

        self.assertEqual(
            events,
            [("stage", staged_bank), ("geometry", staged_bank)],
        )
        self.assertEqual(report["status"], "passed")
        self.assertTrue((self.output_root / "preflight" / "report.json").is_file())

    def test_run_preflight_refuses_existing_report_before_any_output_mutation(
        self,
    ) -> None:
        report_path = self.output_root / "preflight" / "report.json"
        report_path.parent.mkdir(parents=True)
        sentinel = b'{"status":"existing"}\n'
        report_path.write_bytes(sentinel)

        with patch("anchorcal.preflight.assess_storage_budget") as storage_gate:
            with self.assertRaisesRegex(
                PreflightError, "refuses to overwrite an existing finalized report"
            ):
                run_preflight(
                    self._config(),
                    allow_download=False,
                    require_gh200=False,
                )

        storage_gate.assert_not_called()
        self.assertEqual(report_path.read_bytes(), sentinel)
        self.assertEqual(
            sorted(
                path.relative_to(self.output_root).as_posix()
                for path in self.output_root.rglob("*")
            ),
            ["preflight", "preflight/report.json"],
        )

    def test_debug_reuses_only_a_matching_production_mask_contract(self) -> None:
        manifest, _ = self._validate()
        self._freeze_manifest(manifest)

        debug_config = self._config()
        debug_config["runtime"] = {"debug": True}
        debug_config["resolved_config_sha256"] = "d" * 64
        bank = load_vlm_mask_bank(debug_config)
        self.assertEqual(sorted(bank.entries), [17])

        wrong_production_config = copy.deepcopy(debug_config)
        wrong_production_config["runtime"]["debug"] = False
        with self.assertRaisesRegex(PreflightError, "config binding"):
            load_vlm_mask_bank(wrong_production_config)

        wrong_debug_contract = copy.deepcopy(debug_config)
        wrong_debug_contract["masks"]["minimum_foreground_fraction"] = 0.01
        with self.assertRaisesRegex(PreflightError, "contract is incompatible"):
            load_vlm_mask_bank(wrong_debug_contract)

    def test_v3_requires_visual_receipt_and_lowercase_image_hash(self) -> None:
        manifest, _ = self._validate()
        manifest_path = self._freeze_manifest(manifest)
        report_path = self.output_root / "preflight" / "report.json"

        missing_visual = json.loads(manifest_path.read_text(encoding="utf-8"))
        missing_visual.pop("visual_audit")
        atomic_write_json(manifest_path, missing_visual)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report.pop("mask_visual_audit")
        report["mask_manifest_sha256"] = sha256_file(manifest_path)
        atomic_write_json(report_path, report)
        with self.assertRaisesRegex(PreflightError, "v3 requires a visual-audit"):
            load_vlm_mask_bank(self._config())

        invalid_hash = copy.deepcopy(manifest)
        invalid_hash["entries"][0]["image_sha256"] = "G" * 64
        visual_audit = attach_synthetic_visual_audit(self.output_root, invalid_hash)
        invalid_hash["visual_audit"] = visual_audit
        atomic_write_json(manifest_path, invalid_hash)
        report["mask_visual_audit"] = visual_audit
        report["mask_manifest_sha256"] = sha256_file(manifest_path)
        atomic_write_json(report_path, report)
        with self.assertRaisesRegex(PreflightError, "entry is incompatible"):
            load_vlm_mask_bank(self._config())


if __name__ == "__main__":
    unittest.main()
