from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import h5py
except (ImportError, ModuleNotFoundError):
    h5py = None

from anchorcal.errors import StorageError
from anchorcal.io import sha256_file
from anchorcal.storage import (
    HIDDEN_FILENAME,
    HIDDEN_SCHEMA_VERSION,
    SELECTOR_FILENAME,
    SELECTOR_SCHEMA_VERSION,
    CandidateStorage,
    HiddenMetricsReader,
    PredictionBatch,
    SampleMetadata,
    SelectorVisibleReader,
    verify_candidate_storage,
)


def _metadata(ids: list[int], *, groups: bool) -> SampleMetadata:
    labels = np.asarray([value % 2 for value in ids], dtype=np.int64)
    return SampleMetadata(
        img_ids=np.asarray(ids, dtype=np.int64),
        labels=labels,
        groups=(labels * 2 + (1 - labels)) if groups else None,
    )


def _predictions(metadata: SampleMetadata, offset: float = 0.0) -> PredictionBatch:
    labels = np.asarray(metadata.labels, dtype=np.int64)
    logits = np.full((len(labels), 2), -1.0 + offset, dtype=np.float32)
    logits[np.arange(len(labels)), labels] = 1.0 + offset
    losses = np.linspace(0.1, 0.2, len(labels), dtype=np.float32)
    return PredictionBatch.from_logits(logits, labels, losses)


@unittest.skipUnless(h5py is not None, "h5py unavailable")
class CandidateStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary.name) / "candidate_a"
        self.selector_metadata = _metadata([10, 11, 12], groups=False)
        self.hidden_metadata = {
            "oracle_val": _metadata([20, 21], groups=True),
            "test": _metadata([30, 31, 32, 33], groups=True),
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _writer(self, capacity: int = 2) -> CandidateStorage:
        return CandidateStorage(
            self.run_dir,
            run_id="candidate_a",
            epoch_capacity=capacity,
            num_classes=2,
            selector_metadata=self.selector_metadata,
            hidden_metadata=self.hidden_metadata,
            selector_metric_names=("ordinary_accuracy", "saliency_harmonic"),
            compression="gzip",
        )

    def _write(self, writer: CandidateStorage, slot: int, epoch: int, offset: float = 0.0) -> None:
        writer.write_epoch(
            slot=slot,
            epoch_number=epoch,
            selector=_predictions(self.selector_metadata, offset),
            hidden={
                name: _predictions(metadata, offset)
                for name, metadata in self.hidden_metadata.items()
            },
            selector_metrics={
                "ordinary_accuracy": 1.0,
                "saliency_harmonic": 0.75 + offset,
            },
        )

    def test_preallocates_compressed_epoch_major_separate_schemas(self) -> None:
        with self._writer() as writer:
            self.assertTrue(writer.selector_partial_path.is_file())
            self.assertTrue(writer.hidden_partial_path.is_file())
            with h5py.File(writer.selector_partial_path, "r") as selector:
                self.assertEqual(selector.attrs["schema_version"], SELECTOR_SCHEMA_VERSION)
                self.assertEqual(selector.attrs["namespace"], "selector_visible")
                self.assertEqual(selector["predictions/logits"].shape, (2, 3, 2))
                self.assertEqual(selector["predictions/logits"].chunks[0], 1)
                self.assertEqual(selector["predictions/logits"].compression, "gzip")
                self.assertNotIn("group", selector["samples"])
                self.assertEqual(selector["epochs/complete"][:].tolist(), [0, 0])
            with h5py.File(writer.hidden_partial_path, "r") as hidden:
                self.assertEqual(hidden.attrs["schema_version"], HIDDEN_SCHEMA_VERSION)
                self.assertIn("group", hidden["splits/oracle_val/samples"])
                self.assertIn("group", hidden["splits/test/samples"])
                self.assertEqual(
                    hidden["splits/test/predictions/logits"].shape, (2, 4, 2)
                )

    def test_rejects_protected_groups_in_selector_metadata(self) -> None:
        selector = _metadata([1, 2], groups=True)
        with self.assertRaisesRegex(ValueError, "must not contain protected"):
            CandidateStorage(
                self.run_dir,
                run_id="candidate_a",
                epoch_capacity=1,
                num_classes=2,
                selector_metadata=selector,
                hidden_metadata=self.hidden_metadata,
            )

    def test_pair_commit_and_recovery_roll_back_one_sided_flag(self) -> None:
        writer = self._writer()
        self._write(writer, 0, 0)
        selector_partial = writer.selector_partial_path
        hidden_partial = writer.hidden_partial_path
        writer.close()

        # Simulate process death after the hidden completion flag but before the
        # selector-visible flag and journal commit.
        with h5py.File(hidden_partial, "r+") as hidden:
            hidden["epochs/number"][1] = 1
            hidden["splits/test/predictions/logits"][1, ...] = 99.0
            hidden.flush()
            hidden["epochs/complete"][1] = 1
            hidden.flush()
        journal_path = self.run_dir / "candidate_storage_journal.json"
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        journal["state"] = "writing"
        journal["current"] = {"slot": 1, "epoch_number": 1}
        journal_path.write_text(json.dumps(journal), encoding="utf-8")

        with self._writer() as recovered:
            self.assertEqual(recovered.committed_slots, (0,))
            self.assertEqual(recovered.next_slot, 1)
            with h5py.File(hidden_partial, "r") as hidden:
                self.assertEqual(int(hidden["epochs/complete"][1]), 0)
                self.assertEqual(int(hidden["epochs/number"][1]), -1)
            self._write(recovered, 1, 1, offset=0.1)
            self.assertEqual(recovered.committed_slots, (0, 1))

        with h5py.File(selector_partial, "r") as selector, h5py.File(
            hidden_partial, "r"
        ) as hidden:
            np.testing.assert_array_equal(selector["epochs/complete"][:], [1, 1])
            np.testing.assert_array_equal(hidden["epochs/complete"][:], [1, 1])
            self.assertFalse(
                np.all(hidden["splits/test/predictions/logits"][1, ...] == 99.0)
            )

    def test_exclusive_writer_lock(self) -> None:
        first = self._writer()
        try:
            with self.assertRaisesRegex(StorageError, "another writer"):
                self._writer()
        finally:
            first.close()

    def test_recovery_rejects_missing_configured_scalar_metric(self) -> None:
        writer = self._writer()
        selector_partial = writer.selector_partial_path
        writer.close()
        with h5py.File(selector_partial, "r+") as selector:
            del selector["metrics/saliency_harmonic"]
        with self.assertRaisesRegex(StorageError, "scalar metric schema"):
            self._writer()

    def test_recovery_rejects_per_example_shape_or_dtype_drift(self) -> None:
        auxiliary = _metadata([40, 41, 42], groups=False)
        writer = CandidateStorage(
            self.run_dir,
            run_id="candidate_a",
            epoch_capacity=2,
            num_classes=2,
            selector_metadata=self.selector_metadata,
            hidden_metadata=self.hidden_metadata,
            selector_metric_names=("ordinary_accuracy",),
            selector_auxiliary_metadata=auxiliary,
            selector_array_shapes={"swap_margin_drop": (3,)},
        )
        selector_partial = writer.selector_partial_path
        writer.close()
        with h5py.File(selector_partial, "r+") as selector:
            del selector["selector_subset/per_example/swap_margin_drop"]
            selector["selector_subset/per_example"].create_dataset(
                "swap_margin_drop", shape=(2, 3), dtype=np.float64
            )
        with self.assertRaisesRegex(StorageError, "shape/dtype"):
            CandidateStorage(
                self.run_dir,
                run_id="candidate_a",
                epoch_capacity=2,
                num_classes=2,
                selector_metadata=self.selector_metadata,
                hidden_metadata=self.hidden_metadata,
                selector_metric_names=("ordinary_accuracy",),
                selector_auxiliary_metadata=auxiliary,
                selector_array_shapes={"swap_margin_drop": (3,)},
            )

    def test_finalize_publishes_pair_hash_manifest_and_readers(self) -> None:
        with self._writer(capacity=1) as writer:
            self._write(writer, 0, 0)
            manifest = writer.finalize()
            self.assertEqual(manifest["completed_slots"], [0])
            self.assertFalse(writer.selector_partial_path.exists())
            self.assertFalse(writer.hidden_partial_path.exists())

        selector_path = self.run_dir / SELECTOR_FILENAME
        hidden_path = self.run_dir / HIDDEN_FILENAME
        self.assertEqual(
            manifest["files"]["selector_visible"]["sha256"],
            sha256_file(selector_path),
        )
        self.assertEqual(
            manifest["files"]["exploratory_hidden_metrics"]["sha256"],
            sha256_file(hidden_path),
        )
        with SelectorVisibleReader(selector_path) as selector_reader:
            self.assertEqual(selector_reader.completed_slots, (0,))
            self.assertEqual(set(selector_reader.sample_metadata()), {"img_id", "label"})
            epoch = selector_reader.read_epoch(0)
            self.assertEqual(epoch["epoch_number"], 0)
            self.assertEqual(epoch["logits"].shape, (3, 2))
            self.assertEqual(epoch["metrics"]["ordinary_accuracy"], 1.0)
        with HiddenMetricsReader(hidden_path) as hidden_reader:
            self.assertEqual(hidden_reader.completed_slots, (0,))
            metadata = hidden_reader.sample_metadata("test")
            self.assertEqual(set(metadata), {"img_id", "label", "group"})
            self.assertEqual(hidden_reader.read_epoch("test", 0)["logits"].shape, (4, 2))
        verified = verify_candidate_storage(self.run_dir)
        self.assertEqual(verified["run_id"], "candidate_a")

    def test_selector_reader_has_no_hidden_file_dependency(self) -> None:
        with self._writer(capacity=1) as writer:
            self._write(writer, 0, 0)
            writer.finalize()
        hidden = self.run_dir / HIDDEN_FILENAME
        hidden.rename(self.run_dir / "hidden.temporarily_unavailable")
        with SelectorVisibleReader(self.run_dir / SELECTOR_FILENAME) as reader:
            self.assertEqual(reader.completed_slots, (0,))
            self.assertEqual(reader.read_epoch(0)["epoch_number"], 0)

    def test_recovers_publication_interrupted_between_pair_renames(self) -> None:
        writer = self._writer(capacity=1)
        self._write(writer, 0, 0)
        selector_partial = writer.selector_partial_path
        hidden_partial = writer.hidden_partial_path
        selector_final = writer.selector_path
        hidden_final = writer.hidden_path
        journal_path = writer.journal_path
        writer.close()

        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        journal["state"] = "publishing"
        journal_path.write_text(json.dumps(journal), encoding="utf-8")
        os.replace(selector_partial, selector_final)
        self.assertTrue(selector_final.exists())
        self.assertTrue(hidden_partial.exists())

        with self._writer(capacity=1) as recovered:
            self.assertEqual(recovered.committed_slots, (0,))
            self.assertEqual(recovered.finalize()["completed_slots"], [0])
        self.assertTrue(selector_final.is_file())
        self.assertTrue(hidden_final.is_file())
        self.assertFalse(hidden_partial.exists())
        self.assertTrue((self.run_dir / "candidate_storage_manifest.json").is_file())
        with SelectorVisibleReader(selector_final) as reader:
            self.assertEqual(reader.completed_slots, (0,))

    def test_recovers_publication_after_manifest_before_journal_publish(self) -> None:
        with self._writer(capacity=1) as writer:
            self._write(writer, 0, 0)
            original_manifest = writer.finalize()

        journal_path = self.run_dir / "candidate_storage_journal.json"
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        journal["state"] = "publishing"
        journal.pop("manifest_sha256", None)
        journal_path.write_text(json.dumps(journal), encoding="utf-8")

        with self._writer(capacity=1) as recovered:
            self.assertEqual(recovered.committed_slots, (0,))
            self.assertEqual(recovered.finalize(), original_manifest)
        repaired = json.loads(journal_path.read_text(encoding="utf-8"))
        self.assertEqual(repaired["state"], "published")
        self.assertEqual(
            repaired["manifest_sha256"],
            sha256_file(self.run_dir / "candidate_storage_manifest.json"),
        )

    def test_verifier_rejects_expected_run_directory_identity_mismatch(self) -> None:
        with self._writer(capacity=1) as writer:
            self._write(writer, 0, 0)
            writer.finalize()
        with self.assertRaisesRegex(StorageError, "directory name"):
            verify_candidate_storage(
                self.run_dir, expected_run_id="candidate_b"
            )

    def test_verifier_rejects_hdf5_run_identity_even_with_rehashed_files(self) -> None:
        with self._writer(capacity=1) as writer:
            self._write(writer, 0, 0)
            writer.finalize()
        selector_path = self.run_dir / SELECTOR_FILENAME
        hidden_path = self.run_dir / HIDDEN_FILENAME
        for path in (selector_path, hidden_path):
            with h5py.File(path, "r+") as archive:
                archive.attrs["run_id"] = "candidate_b"

        manifest_path = self.run_dir / "candidate_storage_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"]["selector_visible"]["size_bytes"] = selector_path.stat().st_size
        manifest["files"]["selector_visible"]["sha256"] = sha256_file(selector_path)
        manifest["files"]["exploratory_hidden_metrics"][
            "size_bytes"
        ] = hidden_path.stat().st_size
        manifest["files"]["exploratory_hidden_metrics"]["sha256"] = sha256_file(
            hidden_path
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        journal_path = self.run_dir / "candidate_storage_journal.json"
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        journal["manifest_sha256"] = sha256_file(manifest_path)
        journal_path.write_text(json.dumps(journal), encoding="utf-8")

        with self.assertRaisesRegex(StorageError, "identity attributes"):
            verify_candidate_storage(
                self.run_dir, expected_run_id="candidate_a"
            )

    def test_verifier_rejects_manifest_journal_completion_drift(self) -> None:
        with self._writer(capacity=1) as writer:
            self._write(writer, 0, 0)
            writer.finalize()
        manifest_path = self.run_dir / "candidate_storage_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["completed_slots"] = []
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        journal_path = self.run_dir / "candidate_storage_journal.json"
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        journal["manifest_sha256"] = sha256_file(manifest_path)
        journal_path.write_text(json.dumps(journal), encoding="utf-8")
        with self.assertRaisesRegex(StorageError, "identity/completion"):
            verify_candidate_storage(
                self.run_dir, expected_run_id="candidate_a"
            )

    def test_verifier_requires_locked_hdf5_filenames(self) -> None:
        with self._writer(capacity=1) as writer:
            self._write(writer, 0, 0)
            writer.finalize()
        selector_path = self.run_dir / SELECTOR_FILENAME
        renamed = self.run_dir / "renamed_selector.h5"
        selector_path.rename(renamed)
        manifest_path = self.run_dir / "candidate_storage_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record = manifest["files"]["selector_visible"]
        record["path"] = renamed.name
        record["size_bytes"] = renamed.stat().st_size
        record["sha256"] = sha256_file(renamed)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        journal_path = self.run_dir / "candidate_storage_journal.json"
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        journal["manifest_sha256"] = sha256_file(manifest_path)
        journal_path.write_text(json.dumps(journal), encoding="utf-8")
        with self.assertRaisesRegex(StorageError, "locked filename"):
            verify_candidate_storage(
                self.run_dir, expected_run_id="candidate_a"
            )

    def test_readers_reject_wrong_expected_run_identity(self) -> None:
        with self._writer(capacity=1) as writer:
            self._write(writer, 0, 0)
            writer.finalize()
        with self.assertRaisesRegex(StorageError, "run ID mismatch"):
            SelectorVisibleReader(
                self.run_dir / SELECTOR_FILENAME,
                expected_run_id="candidate_b",
            )
        with self.assertRaisesRegex(StorageError, "run ID mismatch"):
            HiddenMetricsReader(
                self.run_dir / HIDDEN_FILENAME,
                expected_run_id="candidate_b",
            )

    def test_inconsistent_predictions_fail_before_any_epoch_flag(self) -> None:
        with self._writer(capacity=1) as writer:
            invalid = _predictions(self.selector_metadata)
            invalid = PredictionBatch(
                logits=invalid.logits,
                prediction=np.asarray([1, 1, 1]),
                correct=invalid.correct,
                loss=invalid.loss,
            )
            with self.assertRaisesRegex(ValueError, "argmax"):
                writer.write_epoch(
                    slot=0,
                    epoch_number=0,
                    selector=invalid,
                    hidden={
                        name: _predictions(metadata)
                        for name, metadata in self.hidden_metadata.items()
                    },
                    selector_metrics={
                        "ordinary_accuracy": 1.0,
                        "saliency_harmonic": 0.7,
                    },
                )
            self.assertEqual(writer.committed_slots, ())


if __name__ == "__main__":
    unittest.main()
