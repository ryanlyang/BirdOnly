from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from anchorcal.config import deep_merge, validate_locked_config  # noqa: E402
from anchorcal import __version__  # noqa: E402
from anchorcal.analysis_only_splits import load_analysis_only_splits  # noqa: E402
from anchorcal.data import load_metadata  # noqa: E402
from anchorcal.errors import ConfigurationError, PreflightError  # noqa: E402
from anchorcal.io import read_yaml, sha256_file  # noqa: E402
from anchorcal.masks import (  # noqa: E402
    classify_patches,
    dilate_mask,
    disk_footprint,
    load_binary_mask,
)
from anchorcal.seeds import (  # noqa: E402
    UINT32_MODULUS,
    background_view_seed,
    stable_seed,
    stateless_rng,
)
from anchorcal.splits import (  # noqa: E402
    ANALYSIS_ONLY_SPLIT_COLUMNS,
    ANALYSIS_ONLY_SPLIT_SCHEMA,
    VISIBLE_SPLIT_COLUMNS,
    construct_splits,
    persist_splits,
)


class SeedAndConfigurationContractTests(unittest.TestCase):
    def test_runtime_and_distribution_versions_agree(self) -> None:
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version = "([^"]+)"$', text, flags=re.MULTILINE)
        self.assertIsNotNone(match)
        self.assertEqual(__version__, match.group(1))

    def test_sha256_seed_payload_and_namespacing_are_exact(self) -> None:
        payload = "6003|42|3|background_branch_eval".encode("utf-8")
        expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        self.assertEqual(
            background_view_seed(6003, 42, 3, "background_branch_eval"), expected
        )
        self.assertGreater(expected, UINT32_MODULUS)
        self.assertEqual(
            stable_seed(6003, 42, 3, "background_branch_eval"),
            expected % UINT32_MODULUS,
        )
        self.assertNotEqual(
            stable_seed(6003, 42, 3, "background_branch_eval"),
            stable_seed(6003, 42, 3, "random_token_audit"),
        )

    def test_stateless_rng_replays_and_purpose_changes_stream(self) -> None:
        first = stateless_rng(6001, 7, 123, "foreground_branch_train").integers(
            0, 2**31, size=32
        )
        replay = stateless_rng(6001, 7, 123, "foreground_branch_train").integers(
            0, 2**31, size=32
        )
        other = stateless_rng(6001, 7, 123, "background_branch_train").integers(
            0, 2**31, size=32
        )
        np.testing.assert_array_equal(first, replay)
        self.assertFalse(np.array_equal(first, other))

    def test_pilot_config_uses_locked_point_one_percent_fallback_gate(self) -> None:
        config = read_yaml(ROOT / "configs" / "anchorcal" / "pilot.yaml")
        self.assertEqual(
            config["data"]["crop_fallback_rate_gate"],
            0.001,
            "lock 8 requires 0.1 percent, represented as the fraction 0.001",
        )

    def test_config_validator_rejects_named_seed_changes(self) -> None:
        mutated_seed = read_yaml(ROOT / "configs" / "anchorcal" / "pilot.yaml")
        mutated_seed["seeds"]["foreground_branch_train"] = 9999
        with self.assertRaises(ConfigurationError):
            validate_locked_config(mutated_seed)

    def test_config_validator_rejects_background_replacement(self) -> None:
        mutated_replacement = read_yaml(
            ROOT / "configs" / "anchorcal" / "pilot.yaml"
        )
        mutated_replacement["branches"]["no_background_sampling_replacement"] = False
        with self.assertRaises(ConfigurationError):
            validate_locked_config(mutated_replacement)

    def test_unmodified_pilot_config_passes_locked_validator(self) -> None:
        original = read_yaml(ROOT / "configs" / "anchorcal" / "pilot.yaml")
        validate_locked_config(original)

    def test_debug_override_is_itself_fully_locked(self) -> None:
        pilot = read_yaml(ROOT / "configs" / "anchorcal" / "pilot.yaml")
        debug = deep_merge(
            pilot, read_yaml(ROOT / "configs" / "anchorcal" / "debug.yaml")
        )
        validate_locked_config(debug, debug=True)
        debug["anchorcal"]["final_metric_bootstrap_replicates"] = 21
        with self.assertRaises(ConfigurationError):
            validate_locked_config(debug, debug=True)

    def test_optimizer_and_candidate_grid_are_not_silently_mutable(self) -> None:
        config = read_yaml(ROOT / "configs" / "anchorcal" / "pilot.yaml")
        config["optimization"]["gradient_clip_norm"] = 2.0
        with self.assertRaises(ConfigurationError):
            validate_locked_config(config)

    def test_branch_architecture_and_diagnostic_controls_are_locked(self) -> None:
        mutations = (
            (("branches", "heads"), 8),
            (("branches", "mlp_ratio"), 3.0),
            (("criteria", "diagnostic_only"), ["product_variants"]),
        )
        for (section, key), value in mutations:
            config = read_yaml(ROOT / "configs" / "anchorcal" / "pilot.yaml")
            config[section][key] = value
            with self.subTest(key=f"{section}.{key}"), self.assertRaises(
                ConfigurationError
            ):
                validate_locked_config(config)

    def test_pretrained_model_and_repository_names_are_locked(self) -> None:
        for key, value in (
            ("model", "hf_hub:timm/not-the-locked-model"),
            ("repository", "timm/not-the-locked-repository"),
        ):
            config = read_yaml(ROOT / "configs" / "anchorcal" / "pilot.yaml")
            config["pretrained"][key] = value
            with self.subTest(key=key), self.assertRaises(ConfigurationError):
                validate_locked_config(config)

    def test_required_direct_cache_parity_lambdas_are_locked(self) -> None:
        config = read_yaml(ROOT / "configs" / "anchorcal" / "pilot.yaml")
        config["anchorcal"]["parity_lambdas"] = [0.0, 0.5, 1.0]
        with self.assertRaisesRegex(ConfigurationError, "parity lambdas"):
            validate_locked_config(config)


class MaskContractTests(unittest.TestCase):
    @staticmethod
    def _write_mask(path: Path, values: np.ndarray) -> None:
        Image.fromarray(values).save(path)

    def test_only_locked_binary_encodings_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            zero_one = np.zeros((8, 8), dtype=np.uint8)
            zero_one[2:6, 2:6] = 1
            zero_255 = zero_one * 255
            mixed = zero_one.copy()
            mixed[0, 0] = 255
            first = root / "zero_one.png"
            second = root / "zero_255.png"
            invalid = root / "mixed.png"
            self._write_mask(first, zero_one)
            self._write_mask(second, zero_255)
            self._write_mask(invalid, mixed)
            np.testing.assert_array_equal(load_binary_mask(first), zero_one.astype(bool))
            np.testing.assert_array_equal(load_binary_mask(second), zero_one.astype(bool))
            with self.assertRaises(PreflightError):
                load_binary_mask(invalid)


    def test_empty_full_and_nonidentical_rgb_masks_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = {
                "empty.png": np.zeros((8, 8), dtype=np.uint8),
                "full.png": np.ones((8, 8), dtype=np.uint8),
            }
            rgb = np.zeros((8, 8, 3), dtype=np.uint8)
            rgb[2:6, 2:6, 0] = 255
            cases["rgb.png"] = rgb
            for name, values in cases.items():
                path = root / name
                self._write_mask(path, values)
                with self.subTest(name=name), self.assertRaises(PreflightError):
                    load_binary_mask(path)

    def test_radius_eight_is_a_disk_and_patch_purity_is_strict(self) -> None:
        footprint = disk_footprint(8)
        self.assertEqual(footprint.shape, (17, 17))
        self.assertTrue(footprint[8, 0])
        self.assertTrue(footprint[0, 8])
        self.assertFalse(footprint[0, 0])
        self.assertFalse(footprint[7, 16])  # displacement (+8,-1)

        mask = np.zeros((96, 96), dtype=bool)
        mask[32:64, 32:64] = True
        dilated = dilate_mask(mask, radius=8)
        self.assertTrue(dilated[24, 32])
        self.assertFalse(dilated[23, 32])
        patches = classify_patches(mask, radius=8, patch_size=16)
        self.assertTrue(patches.foreground[2, 2])
        self.assertTrue(patches.background[0, 0])
        self.assertTrue(patches.mixed[1, 2])
        self.assertFalse(np.any(patches.foreground & patches.background))


class MetadataContractTests(unittest.TestCase):
    def _write_and_load(self, frame: pd.DataFrame) -> pd.DataFrame:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "metadata.csv"
        frame.to_csv(path, index=False)
        return load_metadata(path)

    def test_fractional_integer_columns_are_rejected_without_truncation(self) -> None:
        frame = pd.DataFrame(
            {
                "img_id": [0.5],
                "img_filename": ["a.jpg"],
                "y": [0],
                "place": [0],
                "split": [0],
            }
        )
        with self.assertRaisesRegex(PreflightError, "not integral"):
            self._write_and_load(frame)

    def test_labels_places_and_splits_are_strictly_categorical(self) -> None:
        base = {
            "img_id": [0],
            "img_filename": ["a.jpg"],
            "y": [0],
            "place": [0],
            "split": [0],
        }
        for column, value in (("y", 2), ("place", -1), ("split", 3)):
            frame = pd.DataFrame({**base, column: [value]})
            with self.subTest(column=column), self.assertRaises(PreflightError):
                self._write_and_load(frame)

    def test_valid_metadata_is_canonical_int64_and_sorted(self) -> None:
        frame = pd.DataFrame(
            {
                "img_id": [2.0, 1.0],
                "img_filename": ["b.jpg", "a.jpg"],
                "y": [1.0, 0.0],
                "place": [1.0, 0.0],
                "split": [2.0, 0.0],
            }
        )
        loaded = self._write_and_load(frame)
        self.assertEqual(loaded["img_id"].tolist(), [1, 2])
        for column in ("img_id", "y", "place", "split"):
            self.assertEqual(loaded[column].dtype, np.dtype(np.int64))

    def test_image_filenames_cannot_escape_dataset_roots(self) -> None:
        base = {
            "img_id": [0],
            "y": [0],
            "place": [0],
            "split": [0],
        }
        for filename in ("/tmp/outside.jpg", "class/../../outside.jpg", "   "):
            with self.subTest(filename=filename), self.assertRaises(PreflightError):
                self._write_and_load(
                    pd.DataFrame({**base, "img_filename": [filename]})
                )


class SplitContractTests(unittest.TestCase):
    RELEASE = "waterbird_1.0_forest2water2"

    @staticmethod
    def _metadata() -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        img_id = 0
        # The separately generated Waterbirds100 official-training split is
        # already fully aligned and is used in its entirety.
        for label in (0, 1):
            for _ in range(80):
                rows.append(
                    {
                        "img_id": img_id,
                        "img_filename": f"class_{label}/{img_id}.jpg",
                        "y": label,
                        "place": label,
                        "split": 0,
                    }
                )
                img_id += 1
        for split in (1, 2):
            for label in (0, 1):
                for place in (0, 1):
                    for _ in range(3):
                        rows.append(
                            {
                                "img_id": img_id,
                                "img_filename": f"class_{label}/{img_id}.jpg",
                                "y": label,
                                "place": place,
                                "split": split,
                            }
                        )
                        img_id += 1
        frame = pd.DataFrame(rows)
        frame["metadata_row_index"] = np.arange(len(frame), dtype=np.int64)
        return frame

    def _fcv_artifacts(
        self,
        metadata: pd.DataFrame,
        metadata_sha256: str,
    ) -> tuple[Path, dict[str, object]]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        train = metadata.loc[metadata["split"] == 0].sort_values(
            "metadata_row_index"
        )
        biased_parts = []
        for label in (0, 1):
            members = train.loc[train["y"] == label, "metadata_row_index"].astype(int)
            biased_parts.extend(members.iloc[-round(len(members) * 0.20) :].tolist())
        biased = sorted(int(value) for value in biased_parts)
        candidate = sorted(
            set(train["metadata_row_index"].astype(int).tolist()) - set(biased)
        )

        def membership_hash(values: list[int]) -> str:
            return hashlib.sha256(
                ",".join(str(value) for value in values).encode("ascii")
            ).hexdigest()

        candidate_hash = membership_hash(candidate)
        biased_hash = membership_hash(biased)
        indices = {
            "candidate_train_indices_sha256": candidate_hash,
            "candidate_train_metadata_indices": candidate,
            "biased_validation_indices_sha256": biased_hash,
            "biased_validation_metadata_indices": biased,
            "metadata_path": "/frozen/source/metadata.csv",
            "metadata_sha256": metadata_sha256,
            "split_seed": 0,
            "stratify_by": "y",
            "train_fraction": 0.8,
            "validation_fraction": 0.2,
        }
        split_indices_path = root / "split_indices.json"
        split_indices_path.write_text(
            json.dumps(indices, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        candidate_csv = root / "metadata_train.csv"
        biased_csv = root / "metadata_val.csv"
        pd.DataFrame({"metadata_index": candidate}).to_csv(candidate_csv, index=False)
        pd.DataFrame({"metadata_index": biased}).to_csv(biased_csv, index=False)
        bundle = {
            "artifact_type": "fcv_vit_waterbirds100_manifest_bundle",
            "status": "complete",
            "study_id": "fcv_vit_waterbirds100_first_study",
            "protocol_version": "1",
            "original_metadata_sha256": metadata_sha256,
            "split_indices_sha256": sha256_file(split_indices_path),
            "holdout": {
                "split_seed": 0,
                "stratify_by": "y",
                "train_fraction": 0.8,
                "validation_fraction": 0.2,
                "source_split": "train",
                "require_complete_shortcut_correlation": True,
                "reuse_identical_indices_for_all_candidates": True,
            },
        }
        manifest_path = root / "manifest_bundle.json"
        manifest_path.write_text(
            json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        reference: dict[str, object] = {
            "study_id": bundle["study_id"],
            "protocol_version": bundle["protocol_version"],
            "source_metadata_sha256": metadata_sha256,
            "source_train_count": len(train),
            "candidate_train_count": len(candidate),
            "biased_val_count": len(biased),
            "manifest_bundle_sha256": sha256_file(manifest_path),
            "split_indices_sha256": sha256_file(split_indices_path),
            "candidate_train_csv_sha256": sha256_file(candidate_csv),
            "biased_val_csv_sha256": sha256_file(biased_csv),
            "candidate_train_metadata_indices_sha256": candidate_hash,
            "biased_val_metadata_indices_sha256": biased_hash,
        }
        return root, reference

    def _construct(
        self, metadata: pd.DataFrame, metadata_sha256: str
    ) -> dict[str, pd.DataFrame]:
        root, reference = self._fcv_artifacts(metadata, metadata_sha256)
        self._last_fcv_root = root
        self._last_fcv_reference = reference
        return construct_splits(
            metadata,
            metadata_sha256,
            fcv_split_manifest_root=root,
            fcv_reference=reference,
        )

    def _persist(
        self,
        splits: dict[str, pd.DataFrame],
        output: str | Path,
        metadata: pd.DataFrame,
        metadata_sha256: str,
    ) -> dict[str, object]:
        visible_root = Path(output)
        protected_root = (
            visible_root.parent / "analysis_only" / "splits"
            if visible_root.name == "splits"
            else visible_root / "analysis_only" / "splits"
        )
        return persist_splits(
            splits,
            output,
            analysis_only_output_dir=protected_root,
            source_metadata=metadata,
            source_metadata_sha256=metadata_sha256,
            source_release=self.RELEASE,
            fcv_split_manifest_root=self._last_fcv_root,
            fcv_reference=self._last_fcv_reference,
        )

    def test_split_ids_and_csv_hashes_ignore_input_row_order(self) -> None:
        metadata = self._metadata()
        first = self._construct(metadata, "a" * 64)
        shuffled = metadata.sample(frac=1.0, random_state=99).reset_index(drop=True)
        replay = self._construct(shuffled, "a" * 64)
        for name in first:
            self.assertEqual(
                first[name]["img_id"].tolist(),
                replay[name]["img_id"].tolist(),
                f"{name} ordering depends on input row order",
            )
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            self._persist(first, left, metadata, "a" * 64)
            self._persist(replay, right, shuffled, "a" * 64)
            for name in (
                "candidate_train",
                "biased_val",
                "expert_train",
                "expert_calibration",
            ):
                left_path = Path(left) / f"waterbirds100_{name}.csv"
                right_path = Path(right) / f"waterbirds100_{name}.csv"
                self.assertEqual(sha256_file(left_path), sha256_file(right_path))

    def test_nested_expert_split_partitions_candidate_train(self) -> None:
        splits = self._construct(self._metadata(), "b" * 64)
        candidate = set(splits["candidate_train"]["img_id"])
        expert_train = set(splits["expert_train"]["img_id"])
        calibration = set(splits["expert_calibration"]["img_id"])
        biased = set(splits["biased_val"]["img_id"])
        self.assertFalse(expert_train & calibration)
        self.assertEqual(expert_train | calibration, candidate)
        self.assertFalse(candidate & biased)
        for name in ("candidate_train", "biased_val", "expert_train", "expert_calibration"):
            self.assertTrue((splits[name]["y"] == splits[name]["place"]).all())

    def test_all_official_train_rows_are_used_and_misalignment_is_fatal(self) -> None:
        metadata = self._metadata()
        splits = self._construct(metadata, "f" * 64)
        expected = set(metadata.loc[metadata["split"] == 0, "img_id"].astype(int))
        observed = set(splits["candidate_train"]["img_id"].astype(int)) | set(
            splits["biased_val"]["img_id"].astype(int)
        )
        self.assertEqual(observed, expected)

        corrupted = metadata.copy()
        train_index = corrupted.index[corrupted["split"] == 0][0]
        corrupted.loc[train_index, "place"] = 1 - int(
            corrupted.loc[train_index, "y"]
        )
        with self.assertRaisesRegex(
            PreflightError, "source training split is not completely correlated"
        ):
            self._construct(corrupted, "f" * 64)

        for missing_split in (1, 2):
            with self.subTest(missing_split=missing_split), self.assertRaisesRegex(
                PreflightError, "must all be nonempty"
            ):
                self._construct(
                    metadata.loc[metadata["split"] != missing_split].copy(),
                    "f" * 64,
                )

    def test_split_manifest_rejects_incomplete_source_membership(self) -> None:
        metadata = self._metadata()
        splits = self._construct(metadata, "9" * 64)
        extra = metadata.iloc[[0]].copy()
        extra["img_id"] = int(metadata["img_id"].max()) + 1
        extra["metadata_row_index"] = int(metadata["metadata_row_index"].max()) + 1
        expanded_source = pd.concat([metadata, extra], ignore_index=True)
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(
            PreflightError, "omit official training rows"
        ):
            self._persist(
                splits,
                temporary,
                expanded_source,
                "9" * 64,
            )

    def test_persisted_manifest_records_overlap_union_and_alignment_assertions(self) -> None:
        metadata = self._metadata()
        splits = self._construct(metadata, "e" * 64)
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self._persist(splits, temporary, metadata, "e" * 64)
        self.assertEqual(manifest["schema_version"], "anchorcal-splits-v4")
        self.assertEqual(manifest["namespace"], "selector_visible")
        self.assertEqual(manifest["source_release"], self.RELEASE)
        self.assertEqual(manifest["source_metadata_sha256"], "e" * 64)
        membership = manifest["source_fcv_membership"]
        self.assertEqual(membership["source_train_count"], 160)
        self.assertEqual(membership["candidate_train"]["rows"], 128)
        self.assertEqual(membership["biased_val"]["rows"], 32)
        self.assertTrue(membership["reuse_frozen_membership"])
        self.assertEqual(membership["development_split_seed"], 0)
        self.assertRegex(
            membership["candidate_train"]["metadata_indices_sha256"],
            r"^[0-9a-f]{64}$",
        )
        assertions = manifest["contract_assertions"]
        self.assertTrue(assertions["all_passed"])
        self.assertEqual(
            assertions["candidate_train_biased_val_disjoint"]["overlap_count"],
            0,
        )
        self.assertEqual(
            assertions["expert_union_equals_candidate_train"][
                "expert_union_count"
            ],
            assertions["expert_union_equals_candidate_train"][
                "candidate_train_count"
            ],
        )
        self.assertTrue(
            assertions["complete_official_train_alignment_audit"]["passed"]
        )

    def test_selector_visible_and_analysis_only_schemas_are_physically_separate(self) -> None:
        metadata = self._metadata()
        splits = self._construct(metadata, "7" * 64)
        with tempfile.TemporaryDirectory() as temporary:
            visible = Path(temporary) / "splits"
            self._persist(splits, visible, metadata, "7" * 64)
            protected = Path(temporary) / "analysis_only" / "splits"
            for name in (
                "candidate_train",
                "biased_val",
                "expert_train",
                "expert_calibration",
            ):
                frame = pd.read_csv(visible / f"waterbirds100_{name}.csv")
                self.assertEqual(tuple(frame.columns), VISIBLE_SPLIT_COLUMNS)
                self.assertTrue({"place", "group", "group_name"}.isdisjoint(frame.columns))
            self.assertFalse((visible / "waterbirds100_oracle_val.csv").exists())
            self.assertFalse((visible / "waterbirds100_test.csv").exists())
            protected_manifest = json.loads(
                (protected / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                protected_manifest["schema_version"], ANALYSIS_ONLY_SPLIT_SCHEMA
            )
            self.assertEqual(protected_manifest["namespace"], "analysis_only")
            for name in ("oracle_val", "test"):
                frame = pd.read_csv(protected / f"waterbirds100_{name}.csv")
                self.assertEqual(tuple(frame.columns), ANALYSIS_ONLY_SPLIT_COLUMNS)
                np.testing.assert_array_equal(
                    frame["group"].to_numpy(),
                    frame["y"].to_numpy() * 2 + frame["place"].to_numpy(),
                )

    def test_fcv_artifact_hash_mismatch_fails_closed(self) -> None:
        metadata = self._metadata()
        root, reference = self._fcv_artifacts(metadata, "8" * 64)
        (root / "split_indices.json").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(PreflightError, "artifact hash mismatch"):
            construct_splits(
                metadata,
                "8" * 64,
                fcv_split_manifest_root=root,
                fcv_reference=reference,
            )

    def test_analysis_only_loader_verifies_manifest_and_group_encoding(self) -> None:
        metadata = self._metadata()
        splits = self._construct(metadata, "5" * 64)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            visible = output / "splits"
            self._persist(splits, visible, metadata, "5" * 64)
            config = {
                "paths": {"output_root": str(output)},
                "data": {"protected_split_root": "analysis_only/splits"},
            }
            loaded = load_analysis_only_splits(config)
            self.assertEqual(set(loaded), {"oracle_val", "test"})
            oracle_path = (
                output
                / "analysis_only"
                / "splits"
                / "waterbirds100_oracle_val.csv"
            )
            corrupted = pd.read_csv(oracle_path)
            corrupted.loc[0, "group"] = 3
            corrupted.to_csv(oracle_path, index=False)
            with self.assertRaisesRegex(PreflightError, "provenance mismatch"):
                load_analysis_only_splits(config)

    def test_top_level_membership_is_imported_not_regenerated(self) -> None:
        metadata = self._metadata()
        root, reference = self._fcv_artifacts(metadata, "6" * 64)
        indices = json.loads((root / "split_indices.json").read_text(encoding="utf-8"))
        expected_candidate = set(indices["candidate_train_metadata_indices"])
        splits = construct_splits(
            metadata,
            "6" * 64,
            fcv_split_manifest_root=root,
            fcv_reference=reference,
        )
        self.assertEqual(
            set(splits["candidate_train"]["metadata_index"].astype(int)),
            expected_candidate,
        )

    def test_persisted_splits_are_create_once_and_exact_replays_are_noops(self) -> None:
        metadata = self._metadata()
        splits = self._construct(metadata, "c" * 64)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._persist(splits, root, metadata, "c" * 64)
            before = {
                path.name: sha256_file(path)
                for path in root.iterdir()
                if path.is_file()
            }
            replay = self._persist(splits, root, metadata, "c" * 64)
            after = {
                path.name: sha256_file(path)
                for path in root.iterdir()
                if path.is_file()
            }
            self.assertEqual(first, replay)
            self.assertEqual(before, after)
            self.assertFalse(any(path.name.endswith(".tmp") for path in root.iterdir()))

    def test_persisted_split_csv_tampering_fails_instead_of_overwriting(self) -> None:
        metadata = self._metadata()
        splits = self._construct(metadata, "d" * 64)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._persist(splits, root, metadata, "d" * 64)
            target = root / "waterbirds100_candidate_train.csv"
            target.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(
                PreflightError, "refusing to overwrite nonidentical"
            ):
                self._persist(splits, root, metadata, "d" * 64)
            self.assertEqual(target.read_text(encoding="utf-8"), "tampered\n")

    def test_persisted_split_manifest_tampering_fails_instead_of_overwriting(self) -> None:
        metadata = self._metadata()
        splits = self._construct(metadata, "e" * 64)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._persist(splits, root, metadata, "e" * 64)
            target = root / "manifest.json"
            target.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                PreflightError, "refusing to overwrite nonidentical"
            ):
                self._persist(splits, root, metadata, "e" * 64)
            self.assertEqual(target.read_text(encoding="utf-8"), "{}\n")


if __name__ == "__main__":
    unittest.main()
