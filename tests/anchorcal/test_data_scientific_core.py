from __future__ import annotations

import hashlib
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
from anchorcal.splits import construct_splits, persist_splits  # noqa: E402


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
        return pd.DataFrame(rows)

    @classmethod
    def _persist(
        cls,
        splits: dict[str, pd.DataFrame],
        output: str | Path,
        metadata: pd.DataFrame,
        metadata_sha256: str,
    ) -> dict[str, object]:
        return persist_splits(
            splits,
            output,
            source_metadata=metadata,
            source_metadata_sha256=metadata_sha256,
            source_release=cls.RELEASE,
        )

    def test_split_ids_and_csv_hashes_ignore_input_row_order(self) -> None:
        metadata = self._metadata()
        first = construct_splits(metadata, "a" * 64)
        shuffled = metadata.sample(frac=1.0, random_state=99).reset_index(drop=True)
        replay = construct_splits(shuffled, "a" * 64)
        for name in first:
            self.assertEqual(
                first[name]["img_id"].tolist(),
                replay[name]["img_id"].tolist(),
                f"{name} ordering depends on input row order",
            )
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            self._persist(first, left, metadata, "a" * 64)
            self._persist(replay, right, shuffled, "a" * 64)
            for name in first:
                left_path = Path(left) / f"waterbirds100_{name}.csv"
                right_path = Path(right) / f"waterbirds100_{name}.csv"
                self.assertEqual(sha256_file(left_path), sha256_file(right_path))

    def test_nested_expert_split_partitions_candidate_train(self) -> None:
        splits = construct_splits(self._metadata(), "b" * 64)
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
        splits = construct_splits(metadata, "f" * 64)
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
            construct_splits(corrupted, "f" * 64)

        for missing_split in (1, 2):
            with self.subTest(missing_split=missing_split), self.assertRaisesRegex(
                PreflightError, "must all be nonempty"
            ):
                construct_splits(
                    metadata.loc[metadata["split"] != missing_split].copy(),
                    "f" * 64,
                )

    def test_split_manifest_rejects_incomplete_source_membership(self) -> None:
        metadata = self._metadata()
        splits = construct_splits(metadata, "9" * 64)
        extra = metadata.iloc[[0]].copy()
        extra["img_id"] = int(metadata["img_id"].max()) + 1
        expanded_source = pd.concat([metadata, extra], ignore_index=True)
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(
            PreflightError, "complete official membership"
        ):
            self._persist(
                splits,
                temporary,
                expanded_source,
                "9" * 64,
            )

    def test_persisted_manifest_records_overlap_union_and_alignment_assertions(self) -> None:
        metadata = self._metadata()
        splits = construct_splits(metadata, "e" * 64)
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self._persist(splits, temporary, metadata, "e" * 64)
        self.assertEqual(manifest["schema_version"], "anchorcal-splits-v3")
        self.assertEqual(manifest["source_release"], self.RELEASE)
        self.assertEqual(manifest["source_metadata_sha256"], "e" * 64)
        membership = manifest["official_membership"]
        self.assertEqual(membership["train"]["rows"], 160)
        self.assertEqual(membership["oracle_val"]["rows"], 12)
        self.assertEqual(membership["test"]["rows"], 12)
        self.assertTrue(membership["train"]["complete_membership_verified"])
        self.assertTrue(membership["train"]["all_y_equal_place"])
        for summary in membership.values():
            self.assertRegex(summary["img_ids_sha256"], r"^[0-9a-f]{64}$")
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
        self.assertTrue(assertions["waterbirds100_alignment"]["passed"])

    def test_persisted_splits_are_create_once_and_exact_replays_are_noops(self) -> None:
        metadata = self._metadata()
        splits = construct_splits(metadata, "c" * 64)
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
        splits = construct_splits(metadata, "d" * 64)
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
        splits = construct_splits(metadata, "e" * 64)
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
