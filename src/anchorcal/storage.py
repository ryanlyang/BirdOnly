"""Transactional, namespace-separated candidate output storage.

The candidate trainer writes two physically distinct HDF5 files.  The
selector-visible file contains only biased-validation labels and predictions;
the hidden file contains oracle/test group labels and predictions.  An
external journal makes the two per-epoch completion flags one logical commit.

This module deliberately imports :mod:`h5py` lazily.  Lightweight AnchorCal
utilities therefore remain importable in environments used only for config or
path discovery.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import re
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .errors import StorageError
from .hidden_storage import (
    HIDDEN_FILENAME,
    HIDDEN_SCHEMA_VERSION,
    REQUIRED_HIDDEN_SPLITS,
    HiddenMetricsReader,
)
from .io import atomic_write_json, hash_object, sha256_file
from .selector_storage import (
    SELECTOR_FILENAME,
    SELECTOR_SCHEMA_VERSION,
    SelectorVisibleReader,
)


STORAGE_MANIFEST_FILENAME = "candidate_storage_manifest.json"
STORAGE_JOURNAL_FILENAME = "candidate_storage_journal.json"
STORAGE_LOCK_FILENAME = "candidate_outputs.h5.lock"

JOURNAL_SCHEMA_VERSION = "anchorcal-candidate-pair-journal-v1"
MANIFEST_SCHEMA_VERSION = "anchorcal-candidate-storage-manifest-v1"

_SAFE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


def _require_h5py():
    try:
        import h5py
    except (ImportError, ModuleNotFoundError) as error:
        raise StorageError(
            "candidate storage requires h5py; run the frozen environment preflight"
        ) from error
    return h5py


def _as_numpy(value: Any) -> np.ndarray:
    """Detach tensors without making torch a module-level dependency."""

    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _partial_path(final_path: Path) -> Path:
    if final_path.suffix != ".h5":
        raise ValueError(f"HDF5 output must end in .h5: {final_path}")
    return final_path.with_suffix(".partial.h5")


def _validate_name(value: str, description: str) -> str:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        raise ValueError(f"unsafe {description}: {value!r}")
    return value


@dataclass(frozen=True)
class SampleMetadata:
    """Static sample identity stored once rather than repeated per epoch."""

    img_ids: Any
    labels: Any
    groups: Any | None = None


@dataclass(frozen=True)
class PredictionBatch:
    """Per-example outputs for one model epoch and one fixed split."""

    logits: Any
    prediction: Any
    correct: Any
    loss: Any

    @classmethod
    def from_logits(
        cls, logits: Any, labels: Any, loss: Any
    ) -> "PredictionBatch":
        logits_array = _as_numpy(logits)
        labels_array = _as_numpy(labels)
        prediction = np.argmax(logits_array, axis=1)
        return cls(
            logits=logits_array,
            prediction=prediction,
            correct=prediction == labels_array,
            loss=loss,
        )


class ExclusiveFileLock:
    """A process-scoped, non-blocking Linux advisory lock."""

    def __init__(self, path: str | Path, *, blocking: bool = False) -> None:
        self.path = Path(path)
        self.blocking = bool(blocking)
        self._descriptor: int | None = None

    def acquire(self) -> None:
        if self._descriptor is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            operation = fcntl.LOCK_EX
            if not self.blocking:
                operation |= fcntl.LOCK_NB
            fcntl.flock(descriptor, operation)
        except OSError as error:
            os.close(descriptor)
            if error.errno in (errno.EACCES, errno.EAGAIN):
                raise StorageError(
                    f"another writer holds the exclusive lock: {self.path}"
                ) from error
            raise
        payload = (
            f"pid={os.getpid()} host={socket.gethostname()} "
            f"acquired_unix_ns={time.time_ns()}\n"
        ).encode("utf-8")
        os.ftruncate(descriptor, 0)
        os.write(descriptor, payload)
        os.fsync(descriptor)
        self._descriptor = descriptor

    def release(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __enter__(self) -> "ExclusiveFileLock":
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def _normalize_metadata(
    metadata: SampleMetadata,
    *,
    require_groups: bool,
    namespace: str,
) -> dict[str, np.ndarray]:
    img_ids = _as_numpy(metadata.img_ids)
    labels = _as_numpy(metadata.labels)
    if img_ids.ndim != 1 or labels.ndim != 1 or len(img_ids) != len(labels):
        raise ValueError(f"{namespace} img_ids and labels must be equal-length vectors")
    img_ids = _strict_integral_array(img_ids, f"{namespace} img_ids", np.int64)
    labels = _strict_integral_array(labels, f"{namespace} labels", np.int64)
    if len(img_ids) == 0:
        raise ValueError(f"{namespace} sample metadata must not be empty")
    if len(np.unique(img_ids)) != len(img_ids):
        raise ValueError(f"{namespace} img_ids must be unique")
    if np.any(labels < 0):
        raise ValueError(f"{namespace} labels must be nonnegative")

    result = {"img_id": img_ids, "label": labels}
    if require_groups:
        if metadata.groups is None:
            raise ValueError(f"{namespace} requires protected group labels")
        groups = _as_numpy(metadata.groups)
        if groups.ndim != 1 or len(groups) != len(img_ids):
            raise ValueError(f"{namespace} groups must match img_ids")
        groups = _strict_integral_array(groups, f"{namespace} groups", np.int64)
        if np.any(groups < 0):
            raise ValueError(f"{namespace} groups must be nonnegative")
        result["group"] = groups
    elif metadata.groups is not None:
        raise ValueError(
            "selector-visible metadata must not contain protected group labels"
        )
    return result


def _strict_integral_array(
    value: np.ndarray, description: str, dtype: Any
) -> np.ndarray:
    try:
        numeric = value.astype(np.float64, copy=False)
        converted = value.astype(dtype, copy=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{description} must be integral") from error
    if not np.isfinite(numeric).all() or not np.array_equal(
        numeric, converted.astype(np.float64)
    ):
        raise ValueError(f"{description} must be integral")
    return converted


def _metadata_digest(metadata: Mapping[str, np.ndarray]) -> str:
    payload = {
        name: {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "values": array.tolist(),
        }
        for name, array in sorted(metadata.items())
    }
    return hash_object(payload)


def _coerce_prediction_batch(value: PredictionBatch | Mapping[str, Any]) -> PredictionBatch:
    if isinstance(value, PredictionBatch):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("epoch predictions must be PredictionBatch or a mapping")
    aliases = {
        "prediction": ("prediction", "predictions", "pred"),
        "correct": ("correct", "correctness"),
        "loss": ("loss", "losses"),
    }
    selected: dict[str, Any] = {"logits": value.get("logits")}
    for destination, names in aliases.items():
        for name in names:
            if name in value:
                selected[destination] = value[name]
                break
    missing = [
        name
        for name in ("logits", "prediction", "correct", "loss")
        if selected.get(name) is None
    ]
    if missing:
        raise ValueError(f"epoch predictions are missing fields: {missing}")
    return PredictionBatch(**selected)


def _validate_predictions(
    batch: PredictionBatch | Mapping[str, Any],
    metadata: Mapping[str, np.ndarray],
    num_classes: int,
    namespace: str,
) -> dict[str, np.ndarray]:
    value = _coerce_prediction_batch(batch)
    logits = _as_numpy(value.logits).astype(np.float32, copy=False)
    prediction = _as_numpy(value.prediction)
    correct = _as_numpy(value.correct)
    loss = _as_numpy(value.loss).astype(np.float32, copy=False)
    count = len(metadata["img_id"])
    if logits.shape != (count, num_classes):
        raise ValueError(
            f"{namespace} logits have shape {logits.shape}; expected {(count, num_classes)}"
        )
    for name, array in (
        ("prediction", prediction),
        ("correct", correct),
        ("loss", loss),
    ):
        if array.shape != (count,):
            raise ValueError(f"{namespace} {name} must have shape {(count,)}")
    prediction = _strict_integral_array(
        prediction, f"{namespace} predictions", np.int16
    )
    try:
        correctness_numeric = correct.astype(np.int8, copy=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{namespace} correctness must be boolean") from error
    if not np.isin(correctness_numeric, (0, 1)).all() or not np.array_equal(
        correct.astype(np.bool_, copy=False), correctness_numeric.astype(bool)
    ):
        raise ValueError(f"{namespace} correctness must contain only boolean values")
    correctness = correctness_numeric.astype(np.bool_)
    if not np.isfinite(logits).all() or not np.isfinite(loss).all():
        raise ValueError(f"{namespace} logits and losses must be finite")
    if np.any(loss < -1e-6):
        raise ValueError(f"{namespace} per-example losses must be nonnegative")
    if np.any(prediction < 0) or np.any(prediction >= num_classes):
        raise ValueError(f"{namespace} predictions are outside the class range")
    expected_prediction = np.argmax(logits, axis=1).astype(np.int16)
    if not np.array_equal(prediction, expected_prediction):
        raise ValueError(f"{namespace} predictions do not match argmax(logits)")
    expected_correct = prediction.astype(np.int64) == metadata["label"]
    if not np.array_equal(correctness, expected_correct):
        raise ValueError(f"{namespace} correctness does not match prediction and label")
    return {
        "logits": logits,
        "prediction": prediction,
        "correct": correctness.astype(np.uint8),
        "loss": loss,
    }


def _compression_kwargs(compression: str) -> dict[str, Any]:
    if compression not in {"gzip", "lzf"}:
        raise ValueError("compression must be 'gzip' or 'lzf'")
    result: dict[str, Any] = {"compression": compression, "shuffle": True}
    if compression == "gzip":
        result["compression_opts"] = 4
    return result


def _prediction_group(
    h5: Any,
    parent: Any,
    *,
    epoch_capacity: int,
    sample_count: int,
    num_classes: int,
    compression: str,
) -> None:
    predictions = parent.create_group("predictions")
    compressed = _compression_kwargs(compression)
    sample_chunk = min(sample_count, 256)
    vector_chunk = min(sample_count, 1024)
    predictions.create_dataset(
        "logits",
        shape=(epoch_capacity, sample_count, num_classes),
        dtype=np.float32,
        chunks=(1, sample_chunk, num_classes),
        fillvalue=np.nan,
        **compressed,
    )
    for name, dtype, fillvalue in (
        ("prediction", np.int16, -1),
        ("correct", np.uint8, 0),
        ("loss", np.float32, np.nan),
    ):
        predictions.create_dataset(
            name,
            shape=(epoch_capacity, sample_count),
            dtype=dtype,
            chunks=(1, vector_chunk),
            fillvalue=fillvalue,
            **compressed,
        )


def _create_epoch_group(h5: Any, epoch_capacity: int) -> None:
    epochs = h5.create_group("epochs")
    chunks = (min(epoch_capacity, 64),)
    epochs.create_dataset(
        "number", shape=(epoch_capacity,), dtype=np.int32, chunks=chunks, fillvalue=-1
    )
    epochs.create_dataset(
        "complete", shape=(epoch_capacity,), dtype=np.uint8, chunks=chunks, fillvalue=0
    )
    epochs.create_dataset(
        "committed_unix_ns",
        shape=(epoch_capacity,),
        dtype=np.int64,
        chunks=chunks,
        fillvalue=0,
    )


def _write_static_samples(
    parent: Any, metadata: Mapping[str, np.ndarray], compression: str
) -> None:
    samples = parent.create_group("samples")
    compressed = _compression_kwargs(compression)
    for name, array in metadata.items():
        samples.create_dataset(
            name,
            data=array,
            dtype=np.int64,
            chunks=(min(len(array), 1024),),
            **compressed,
        )


class CandidateStorage:
    """Sole writer for one candidate run's paired HDF5 transaction."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        run_id: str,
        epoch_capacity: int,
        num_classes: int,
        selector_metadata: SampleMetadata,
        hidden_metadata: Mapping[str, SampleMetadata],
        selector_metric_names: Sequence[str] = (),
        selector_auxiliary_metadata: SampleMetadata | None = None,
        selector_array_shapes: Mapping[str, Sequence[int]] | None = None,
        compression: str = "gzip",
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.run_id = _validate_name(run_id, "run_id")
        if epoch_capacity <= 0 or num_classes <= 1:
            raise ValueError("epoch_capacity must be positive and num_classes >= 2")
        self.epoch_capacity = int(epoch_capacity)
        self.num_classes = int(num_classes)
        self.compression = compression
        self.selector_metric_names = tuple(
            _validate_name(name, "selector metric name") for name in selector_metric_names
        )
        if len(set(self.selector_metric_names)) != len(self.selector_metric_names):
            raise ValueError("selector metric names must be unique")
        self.selector_array_shapes = {
            _validate_name(name, "selector per-example array name"): tuple(
                int(value) for value in shape
            )
            for name, shape in dict(selector_array_shapes or {}).items()
        }
        if any(not shape or any(value <= 0 for value in shape) for shape in self.selector_array_shapes.values()):
            raise ValueError("selector per-example array shapes must be nonempty and positive")
        if bool(self.selector_array_shapes) != (selector_auxiliary_metadata is not None):
            raise ValueError(
                "selector auxiliary metadata and array shapes must be supplied together"
            )
        self.selector_auxiliary_metadata = (
            _normalize_metadata(
                selector_auxiliary_metadata,
                require_groups=False,
                namespace="selector_subset",
            )
            if selector_auxiliary_metadata is not None
            else None
        )
        if self.selector_auxiliary_metadata is not None:
            expected_count = len(self.selector_auxiliary_metadata["img_id"])
            for name, shape in self.selector_array_shapes.items():
                if shape[0] != expected_count:
                    raise ValueError(
                        f"selector array {name} first dimension must match selector subset"
                    )
        if set(hidden_metadata) != set(REQUIRED_HIDDEN_SPLITS):
            raise ValueError(
                f"hidden metadata must contain exactly {REQUIRED_HIDDEN_SPLITS}"
            )
        self.selector_metadata = _normalize_metadata(
            selector_metadata, require_groups=False, namespace="biased_val"
        )
        self.hidden_metadata = {
            split: _normalize_metadata(
                hidden_metadata[split], require_groups=True, namespace=split
            )
            for split in REQUIRED_HIDDEN_SPLITS
        }
        for namespace, metadata in (
            ("biased_val", self.selector_metadata),
            *((name, self.hidden_metadata[name]) for name in REQUIRED_HIDDEN_SPLITS),
        ):
            if np.any(metadata["label"] >= self.num_classes):
                raise ValueError(
                    f"{namespace} labels are outside the configured class range"
                )

        self.selector_path = self.run_dir / SELECTOR_FILENAME
        self.hidden_path = self.run_dir / HIDDEN_FILENAME
        self.selector_partial_path = _partial_path(self.selector_path)
        self.hidden_partial_path = _partial_path(self.hidden_path)
        self.journal_path = self.run_dir / STORAGE_JOURNAL_FILENAME
        self.manifest_path = self.run_dir / STORAGE_MANIFEST_FILENAME
        self.lock = ExclusiveFileLock(self.run_dir / STORAGE_LOCK_FILENAME)
        self._selector_h5: Any | None = None
        self._hidden_h5: Any | None = None
        self._closed = False
        self._published = False
        self._committed_slots: list[int] = []

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.lock.acquire()
        try:
            self._open_or_create()
        except BaseException:
            self.lock.release()
            raise

    @property
    def committed_slots(self) -> tuple[int, ...]:
        return tuple(self._committed_slots)

    @property
    def next_slot(self) -> int:
        return len(self._committed_slots)

    def _initial_journal(self) -> dict[str, Any]:
        return {
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "run_id": self.run_id,
            "epoch_capacity": self.epoch_capacity,
            "state": "open",
            "committed_slots": [],
            "current": None,
            "recoveries": [],
            "updated_unix_ns": time.time_ns(),
        }

    def _read_journal(self) -> dict[str, Any]:
        try:
            value = json.loads(self.journal_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise StorageError(
                f"invalid candidate storage journal: {self.journal_path}"
            ) from error
        if (
            value.get("schema_version") != JOURNAL_SCHEMA_VERSION
            or value.get("run_id") != self.run_id
            or int(value.get("epoch_capacity", -1)) != self.epoch_capacity
        ):
            raise StorageError("candidate storage journal does not match this run")
        return value

    def _write_journal(self, value: dict[str, Any]) -> None:
        value = dict(value)
        value["updated_unix_ns"] = time.time_ns()
        atomic_write_json(self.journal_path, value)
        _fsync_directory(self.run_dir)

    def _open_or_create(self) -> None:
        h5py = _require_h5py()
        final_presence = (self.selector_path.exists(), self.hidden_path.exists())
        partial_presence = (
            self.selector_partial_path.exists(),
            self.hidden_partial_path.exists(),
        )
        if any(final_presence):
            recovered = self._recover_interrupted_publication(
                final_presence=final_presence,
            )
            # A process may die after publishing the paired HDF5 transaction
            # but before archiving the restart state or writing completion.json.
            # Reopen that state as a read-only logical completion so the outer
            # pipeline can finish its remaining idempotent bookkeeping.
            if not self.manifest_path.is_file():
                status = "recovered" if recovered else "published"
                raise StorageError(
                    f"candidate run is {status} without a storage manifest: {self.run_dir}"
                )
            journal = self._read_journal()
            if journal.get("state") != "published":
                raise StorageError("published candidate files have a non-published journal")
            try:
                manifest = json.loads(
                    self.manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as error:
                raise StorageError("published candidate manifest is invalid") from error
            self._validate_publication_manifest(manifest)
            if journal.get("manifest_sha256") != sha256_file(self.manifest_path):
                raise StorageError(
                    "published candidate manifest hash does not match its journal"
                )
            committed = [int(value) for value in journal.get("committed_slots", [])]
            if committed != list(range(self.epoch_capacity)):
                raise StorageError("published candidate journal is not fully committed")
            # A published run is still reopened against the caller's complete
            # locked schema.  Hash verification occurs in the outer pipeline;
            # this check additionally rejects a stale run created with missing
            # diagnostics, different selector samples, or drifted array shapes.
            with h5py.File(self.selector_path, "r") as selector, h5py.File(
                self.hidden_path, "r"
            ) as hidden:
                self._selector_h5 = selector
                self._hidden_h5 = hidden
                self._validate_files()
            self._selector_h5 = None
            self._hidden_h5 = None
            self._committed_slots = committed
            self._published = True
            return
        if partial_presence not in {(False, False), (True, True)}:
            raise StorageError("only one candidate partial HDF5 file exists")

        if partial_presence == (False, False):
            if self.journal_path.exists():
                raise StorageError("journal exists without candidate partial files")
            self._selector_h5 = h5py.File(self.selector_partial_path, "w")
            self._hidden_h5 = h5py.File(self.hidden_partial_path, "w")
            self._initialize_selector_file()
            self._initialize_hidden_file()
            self._selector_h5.flush()
            self._hidden_h5.flush()
            _fsync_file(self.selector_partial_path)
            _fsync_file(self.hidden_partial_path)
            self._write_journal(self._initial_journal())
        else:
            if not self.journal_path.is_file():
                raise StorageError("candidate partial files exist without their journal")
            self._selector_h5 = h5py.File(self.selector_partial_path, "r+")
            self._hidden_h5 = h5py.File(self.hidden_partial_path, "r+")
            self._validate_files()
        self._recover_epoch_pair()

    def _recover_interrupted_publication(
        self,
        *,
        final_presence: tuple[bool, bool],
    ) -> bool:
        """Finish a crash-interrupted pair of same-filesystem renames.

        Publishing two files cannot be one filesystem operation.  The journal
        therefore enters ``publishing`` before either rename.  If a process
        dies between renames (or before the manifest), the next sole writer
        validates both complete HDF5 sources and finishes the publication.
        """

        if not self.journal_path.is_file():
            raise StorageError(
                "published or partially published HDF5 exists without its journal"
            )
        journal = self._read_journal()
        if (
            final_presence == (True, True)
            and self.manifest_path.is_file()
            and journal.get("state") == "published"
        ):
            return False
        if journal.get("state") not in {"publishing", "published"}:
            raise StorageError(
                "inconsistent candidate publication outside a publishing journal state"
            )
        selector_sources = [
            path
            for path in (self.selector_path, self.selector_partial_path)
            if path.exists()
        ]
        hidden_sources = [
            path for path in (self.hidden_path, self.hidden_partial_path) if path.exists()
        ]
        if len(selector_sources) != 1 or len(hidden_sources) != 1:
            raise StorageError("interrupted candidate publication has ambiguous files")

        h5py = _require_h5py()
        with h5py.File(selector_sources[0], "r") as selector, h5py.File(
            hidden_sources[0], "r"
        ) as hidden:
            self._selector_h5 = selector
            self._hidden_h5 = hidden
            self._validate_files()
            selector_complete = np.asarray(selector["epochs/complete"][:], dtype=bool)
            hidden_complete = np.asarray(hidden["epochs/complete"][:], dtype=bool)
            selector_number = np.asarray(selector["epochs/number"][:], dtype=np.int64)
            hidden_number = np.asarray(hidden["epochs/number"][:], dtype=np.int64)
            if (
                not selector_complete.all()
                or not hidden_complete.all()
                or not np.array_equal(selector_number, hidden_number)
                or np.any(selector_number < 0)
            ):
                raise StorageError(
                    "interrupted publication does not contain a complete paired run"
                )
        self._selector_h5 = None
        self._hidden_h5 = None

        if self.selector_partial_path.exists():
            os.replace(self.selector_partial_path, self.selector_path)
            _fsync_directory(self.run_dir)
        if self.hidden_partial_path.exists():
            os.replace(self.hidden_partial_path, self.hidden_path)
            _fsync_directory(self.run_dir)
        if self.manifest_path.is_file():
            try:
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise StorageError(
                    "interrupted candidate publication has an invalid manifest"
                ) from error
            self._validate_publication_manifest(manifest)
        else:
            manifest = self._build_manifest(list(range(self.epoch_capacity)))
            atomic_write_json(self.manifest_path, manifest)
            _fsync_directory(self.run_dir)
        journal["state"] = "published"
        journal["committed_slots"] = list(range(self.epoch_capacity))
        journal["current"] = None
        journal["publication_recovered_unix_ns"] = time.time_ns()
        journal["manifest_sha256"] = sha256_file(self.manifest_path)
        self._write_journal(journal)
        return True

    def _validate_publication_manifest(self, manifest: Mapping[str, Any]) -> None:
        """Validate the immutable identity and byte records of a published pair."""

        completed = list(range(self.epoch_capacity))
        if (
            manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
            or manifest.get("run_id") != self.run_id
            or int(manifest.get("epoch_capacity", -1)) != self.epoch_capacity
            or int(manifest.get("num_classes", -1)) != self.num_classes
            or manifest.get("completed_slots") != completed
        ):
            raise StorageError("candidate storage manifest identity is incompatible")
        files = manifest.get("files")
        expected = {
            "selector_visible": (
                self.selector_path,
                SELECTOR_SCHEMA_VERSION,
                SELECTOR_FILENAME,
            ),
            "exploratory_hidden_metrics": (
                self.hidden_path,
                HIDDEN_SCHEMA_VERSION,
                HIDDEN_FILENAME,
            ),
        }
        if not isinstance(files, dict) or set(files) != set(expected):
            raise StorageError("candidate storage manifest file set is incompatible")
        for name, (path, schema, filename) in expected.items():
            record = files[name]
            if (
                not isinstance(record, dict)
                or record.get("path") != filename
                or record.get("schema_version") != schema
                or not path.is_file()
                or int(record.get("size_bytes", -1)) != path.stat().st_size
                or record.get("sha256") != sha256_file(path)
            ):
                raise StorageError(
                    f"candidate storage manifest file record is invalid: {name}"
                )

    def _initialize_selector_file(self) -> None:
        h5 = self._selector_h5
        assert h5 is not None
        h5.attrs.update(
            {
                "schema_version": SELECTOR_SCHEMA_VERSION,
                "namespace": "selector_visible",
                "split": "biased_val",
                "run_id": self.run_id,
                "epoch_capacity": self.epoch_capacity,
                "num_classes": self.num_classes,
                "metadata_sha256": _metadata_digest(self.selector_metadata),
            }
        )
        _create_epoch_group(h5, self.epoch_capacity)
        _write_static_samples(h5, self.selector_metadata, self.compression)
        _prediction_group(
            h5,
            h5,
            epoch_capacity=self.epoch_capacity,
            sample_count=len(self.selector_metadata["img_id"]),
            num_classes=self.num_classes,
            compression=self.compression,
        )
        metrics = h5.create_group("metrics")
        for name in self.selector_metric_names:
            metrics.create_dataset(
                name,
                shape=(self.epoch_capacity,),
                dtype=np.float64,
                chunks=(min(self.epoch_capacity, 64),),
                fillvalue=np.nan,
                **_compression_kwargs(self.compression),
            )
        if self.selector_auxiliary_metadata is not None:
            auxiliary = h5.create_group("selector_subset")
            auxiliary.attrs["metadata_sha256"] = _metadata_digest(
                self.selector_auxiliary_metadata
            )
            _write_static_samples(
                auxiliary, self.selector_auxiliary_metadata, self.compression
            )
            arrays = auxiliary.create_group("per_example")
            compressed = _compression_kwargs(self.compression)
            for name, shape in self.selector_array_shapes.items():
                chunks = (1, *tuple(min(value, 256) for value in shape))
                arrays.create_dataset(
                    name,
                    shape=(self.epoch_capacity, *shape),
                    dtype=np.float32,
                    chunks=chunks,
                    fillvalue=np.nan,
                    **compressed,
                )

    def _initialize_hidden_file(self) -> None:
        h5 = self._hidden_h5
        assert h5 is not None
        h5.attrs.update(
            {
                "schema_version": HIDDEN_SCHEMA_VERSION,
                "namespace": "exploratory_hidden_metrics",
                "run_id": self.run_id,
                "epoch_capacity": self.epoch_capacity,
                "num_classes": self.num_classes,
            }
        )
        _create_epoch_group(h5, self.epoch_capacity)
        splits = h5.create_group("splits")
        for split in REQUIRED_HIDDEN_SPLITS:
            group = splits.create_group(split)
            metadata = self.hidden_metadata[split]
            group.attrs["metadata_sha256"] = _metadata_digest(metadata)
            _write_static_samples(group, metadata, self.compression)
            _prediction_group(
                h5,
                group,
                epoch_capacity=self.epoch_capacity,
                sample_count=len(metadata["img_id"]),
                num_classes=self.num_classes,
                compression=self.compression,
            )

    def _validate_common_attrs(self, h5: Any, schema: str, namespace: str) -> None:
        if h5.attrs.get("schema_version") != schema:
            raise StorageError(f"unexpected HDF5 schema for {namespace}")
        if h5.attrs.get("namespace") != namespace:
            raise StorageError(f"unexpected HDF5 namespace for {namespace}")
        if h5.attrs.get("run_id") != self.run_id:
            raise StorageError(f"HDF5 run_id mismatch for {namespace}")
        if int(h5.attrs.get("epoch_capacity", -1)) != self.epoch_capacity:
            raise StorageError(f"HDF5 epoch capacity mismatch for {namespace}")
        if int(h5.attrs.get("num_classes", -1)) != self.num_classes:
            raise StorageError(f"HDF5 class count mismatch for {namespace}")

    def _validate_files(self) -> None:
        selector = self._selector_h5
        hidden = self._hidden_h5
        assert selector is not None and hidden is not None
        self._validate_common_attrs(selector, SELECTOR_SCHEMA_VERSION, "selector_visible")
        self._validate_common_attrs(
            hidden, HIDDEN_SCHEMA_VERSION, "exploratory_hidden_metrics"
        )
        if selector.attrs.get("metadata_sha256") != _metadata_digest(
            self.selector_metadata
        ):
            raise StorageError("selector sample metadata changed across recovery")
        if "group" in selector["samples"]:
            raise StorageError("selector-visible HDF5 contains protected group labels")
        if "metrics" not in selector or set(selector["metrics"].keys()) != set(
            self.selector_metric_names
        ):
            raise StorageError("selector scalar metric schema changed across recovery")
        for name in self.selector_metric_names:
            dataset = selector[f"metrics/{name}"]
            if dataset.shape != (self.epoch_capacity,) or dataset.dtype != np.dtype(
                np.float64
            ):
                raise StorageError(
                    f"selector scalar metric dataset has invalid shape/dtype: {name}"
                )
        if self.selector_auxiliary_metadata is not None:
            if "selector_subset" not in selector:
                raise StorageError("selector-visible HDF5 lacks selector subset arrays")
            auxiliary = selector["selector_subset"]
            if auxiliary.attrs.get("metadata_sha256") != _metadata_digest(
                self.selector_auxiliary_metadata
            ):
                raise StorageError("selector subset metadata changed across recovery")
            if set(auxiliary["per_example"].keys()) != set(self.selector_array_shapes):
                raise StorageError("selector per-example array schema changed")
            for name, shape in self.selector_array_shapes.items():
                dataset = auxiliary[f"per_example/{name}"]
                if dataset.shape != (self.epoch_capacity, *shape) or dataset.dtype != np.dtype(
                    np.float32
                ):
                    raise StorageError(
                        f"selector per-example dataset has invalid shape/dtype: {name}"
                    )
        elif "selector_subset" in selector:
            raise StorageError("unexpected selector subset exists across recovery")
        for split in REQUIRED_HIDDEN_SPLITS:
            if split not in hidden["splits"]:
                raise StorageError(f"hidden HDF5 is missing split {split}")
            expected = _metadata_digest(self.hidden_metadata[split])
            if hidden["splits"][split].attrs.get("metadata_sha256") != expected:
                raise StorageError(f"hidden sample metadata changed for {split}")

    def _recover_epoch_pair(self) -> None:
        selector = self._selector_h5
        hidden = self._hidden_h5
        assert selector is not None and hidden is not None
        journal = self._read_journal()
        previous_journal_state = journal.get("state")
        previous_current = journal.get("current")
        selector_complete = np.asarray(selector["epochs/complete"][:], dtype=bool)
        hidden_complete = np.asarray(hidden["epochs/complete"][:], dtype=bool)
        selector_number = np.asarray(selector["epochs/number"][:], dtype=np.int64)
        hidden_number = np.asarray(hidden["epochs/number"][:], dtype=np.int64)

        committed: list[int] = []
        recovered: list[dict[str, Any]] = []
        gap_seen = False
        for slot in range(self.epoch_capacity):
            both = bool(selector_complete[slot] and hidden_complete[slot])
            if both:
                if gap_seen:
                    raise StorageError("completed candidate epoch exists after an incomplete gap")
                if selector_number[slot] < 0 or selector_number[slot] != hidden_number[slot]:
                    raise StorageError("paired HDF5 epoch numbers disagree")
                committed.append(slot)
                continue
            gap_seen = True
            if selector_complete[slot] or hidden_complete[slot]:
                recovered.append(
                    {
                        "slot": slot,
                        "selector_was_complete": bool(selector_complete[slot]),
                        "hidden_was_complete": bool(hidden_complete[slot]),
                    }
                )
            selector["epochs/complete"][slot] = 0
            hidden["epochs/complete"][slot] = 0
            selector["epochs/number"][slot] = -1
            hidden["epochs/number"][slot] = -1
            selector["epochs/committed_unix_ns"][slot] = 0
            hidden["epochs/committed_unix_ns"][slot] = 0

        selector.flush()
        hidden.flush()
        if recovered or previous_journal_state in {"writing", "publishing"}:
            _fsync_file(self.selector_partial_path)
            _fsync_file(self.hidden_partial_path)
        self._committed_slots = committed
        journal["state"] = "open"
        journal["committed_slots"] = committed
        journal["current"] = None
        if recovered or previous_journal_state in {"writing", "publishing"}:
            history = list(journal.get("recoveries", []))
            history.append(
                {
                    "recovered_unix_ns": time.time_ns(),
                    "rolled_back": recovered,
                    "previous_current": previous_current,
                }
            )
            journal["recoveries"] = history
        self._write_journal(journal)

    def _write_predictions(self, parent: Any, slot: int, values: Mapping[str, np.ndarray]) -> None:
        for name in ("logits", "prediction", "correct", "loss"):
            parent[f"predictions/{name}"][slot, ...] = values[name]

    def write_epoch(
        self,
        *,
        slot: int,
        epoch_number: int,
        selector: PredictionBatch | Mapping[str, Any],
        hidden: Mapping[str, PredictionBatch | Mapping[str, Any]],
        selector_metrics: Mapping[str, float] | None = None,
        selector_per_example: Mapping[str, Any] | None = None,
    ) -> None:
        """Commit one epoch to both files, with the visible flag written last."""

        if self._closed or self._published:
            raise StorageError("candidate storage writer is closed")
        if slot != self.next_slot or not 0 <= slot < self.epoch_capacity:
            raise StorageError(
                f"epoch slot must be the next incomplete slot {self.next_slot}; got {slot}"
            )
        if (
            isinstance(epoch_number, bool)
            or not isinstance(epoch_number, int)
            or epoch_number < 0
        ):
            raise ValueError("epoch_number must be a nonnegative integer")
        if self._committed_slots:
            selector_h5_for_order = self._selector_h5
            assert selector_h5_for_order is not None
            previous_number = int(
                selector_h5_for_order["epochs/number"][self._committed_slots[-1]]
            )
            if epoch_number <= previous_number:
                raise ValueError(
                    "epoch_number must increase strictly across committed slots"
                )
        if set(hidden) != set(REQUIRED_HIDDEN_SPLITS):
            raise ValueError(f"hidden outputs must contain exactly {REQUIRED_HIDDEN_SPLITS}")
        metrics = dict(selector_metrics or {})
        if set(metrics) != set(self.selector_metric_names):
            raise ValueError(
                f"selector metric keys must equal {self.selector_metric_names}; "
                f"got {tuple(metrics)}"
            )
        for name, value in metrics.items():
            if not np.isfinite(float(value)):
                raise ValueError(f"selector metric {name} must be finite")
        per_example = dict(selector_per_example or {})
        if set(per_example) != set(self.selector_array_shapes):
            raise ValueError(
                "selector per-example keys must equal configured array schema"
            )
        normalized_per_example: dict[str, np.ndarray] = {}
        for name, shape in self.selector_array_shapes.items():
            array = _as_numpy(per_example[name]).astype(np.float32, copy=False)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(
                    f"selector per-example {name} must be finite with shape {shape}"
                )
            normalized_per_example[name] = array

        selector_values = _validate_predictions(
            selector, self.selector_metadata, self.num_classes, "biased_val"
        )
        hidden_values = {
            split: _validate_predictions(
                hidden[split], self.hidden_metadata[split], self.num_classes, split
            )
            for split in REQUIRED_HIDDEN_SPLITS
        }
        selector_h5 = self._selector_h5
        hidden_h5 = self._hidden_h5
        assert selector_h5 is not None and hidden_h5 is not None

        journal = self._read_journal()
        journal["state"] = "writing"
        journal["current"] = {
            "slot": slot,
            "epoch_number": epoch_number,
            "started_unix_ns": time.time_ns(),
        }
        self._write_journal(journal)
        try:
            selector_h5["epochs/number"][slot] = epoch_number
            hidden_h5["epochs/number"][slot] = epoch_number
            self._write_predictions(selector_h5, slot, selector_values)
            for split in REQUIRED_HIDDEN_SPLITS:
                self._write_predictions(
                    hidden_h5[f"splits/{split}"], slot, hidden_values[split]
                )
            for name, value in metrics.items():
                selector_h5[f"metrics/{name}"][slot] = float(value)
            for name, value in normalized_per_example.items():
                selector_h5[f"selector_subset/per_example/{name}"][slot, ...] = value
            committed_ns = time.time_ns()
            selector_h5["epochs/committed_unix_ns"][slot] = committed_ns
            hidden_h5["epochs/committed_unix_ns"][slot] = committed_ns

            # Every epoch payload is flushed before either completion flag.
            selector_h5.flush()
            hidden_h5.flush()
            _fsync_file(self.selector_partial_path)
            _fsync_file(self.hidden_partial_path)

            # Hidden first and selector-visible last: a visible completed epoch
            # can never precede its hidden payload.  Recovery still requires both.
            hidden_h5["epochs/complete"][slot] = 1
            hidden_h5.flush()
            _fsync_file(self.hidden_partial_path)
            selector_h5["epochs/complete"][slot] = 1
            selector_h5.flush()
            _fsync_file(self.selector_partial_path)
        except BaseException:
            selector_h5["epochs/complete"][slot] = 0
            hidden_h5["epochs/complete"][slot] = 0
            selector_h5.flush()
            hidden_h5.flush()
            journal = self._read_journal()
            journal["state"] = "open"
            journal["current"] = None
            journal.setdefault("recoveries", []).append(
                {"recovered_unix_ns": time.time_ns(), "rolled_back": [{"slot": slot}]}
            )
            self._write_journal(journal)
            raise

        self._committed_slots.append(slot)
        journal = self._read_journal()
        journal["state"] = "open"
        journal["current"] = None
        journal["committed_slots"] = list(self._committed_slots)
        self._write_journal(journal)

    def _close_h5(self) -> None:
        for name in ("_selector_h5", "_hidden_h5"):
            handle = getattr(self, name)
            if handle is not None:
                handle.flush()
                handle.close()
                setattr(self, name, None)

    def _build_manifest(self, completed_slots: Sequence[int]) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": self.run_id,
            "epoch_capacity": self.epoch_capacity,
            "num_classes": self.num_classes,
            "completed_slots": [int(slot) for slot in completed_slots],
            "files": {
                "selector_visible": {
                    "path": self.selector_path.name,
                    "schema_version": SELECTOR_SCHEMA_VERSION,
                    "size_bytes": self.selector_path.stat().st_size,
                    "sha256": sha256_file(self.selector_path),
                },
                "exploratory_hidden_metrics": {
                    "path": self.hidden_path.name,
                    "schema_version": HIDDEN_SCHEMA_VERSION,
                    "size_bytes": self.hidden_path.stat().st_size,
                    "sha256": sha256_file(self.hidden_path),
                },
            },
            "published_unix_ns": time.time_ns(),
        }

    def finalize(self) -> dict[str, Any]:
        """Publish both complete files and their SHA-256 manifest."""

        if self._published:
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        if self._closed:
            raise StorageError("candidate storage writer is closed")
        if self.next_slot != self.epoch_capacity:
            raise StorageError(
                f"cannot publish {self.next_slot}/{self.epoch_capacity} committed epochs"
            )
        journal = self._read_journal()
        journal["state"] = "publishing"
        journal["current"] = None
        self._write_journal(journal)
        self._close_h5()
        _fsync_file(self.selector_partial_path)
        _fsync_file(self.hidden_partial_path)
        os.replace(self.selector_partial_path, self.selector_path)
        _fsync_directory(self.run_dir)
        os.replace(self.hidden_partial_path, self.hidden_path)
        _fsync_directory(self.run_dir)
        manifest = self._build_manifest(self._committed_slots)
        atomic_write_json(self.manifest_path, manifest)
        _fsync_directory(self.run_dir)
        journal = self._read_journal()
        journal["state"] = "published"
        journal["manifest_sha256"] = sha256_file(self.manifest_path)
        self._write_journal(journal)
        self._published = True
        return manifest

    def close(self) -> None:
        if self._closed:
            return
        self._close_h5()
        self.lock.release()
        self._closed = True

    def __enter__(self) -> "CandidateStorage":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def verify_candidate_storage(
    run_dir: str | Path, *, expected_run_id: str | None = None
) -> dict[str, Any]:
    """Hash- and schema-verify one fully published paired candidate artifact."""

    root = Path(run_dir).resolve()
    expected_identity = root.name if expected_run_id is None else expected_run_id
    try:
        expected_identity = _validate_name(expected_identity, "expected run_id")
    except ValueError as error:
        raise StorageError("expected candidate run ID is invalid") from error
    if root.name != expected_identity:
        raise StorageError(
            "candidate run directory name does not match the expected run ID"
        )
    manifest_path = root / STORAGE_MANIFEST_FILENAME
    journal_path = root / STORAGE_JOURNAL_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StorageError(f"candidate storage receipt is unreadable under {root}") from error
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise StorageError("candidate storage manifest schema is incompatible")
    if journal.get("schema_version") != JOURNAL_SCHEMA_VERSION:
        raise StorageError("candidate storage journal schema is incompatible")
    if journal.get("state") != "published":
        raise StorageError("candidate storage journal is not published")
    if journal.get("manifest_sha256") != sha256_file(manifest_path):
        raise StorageError("candidate storage manifest hash does not match its journal")
    epoch_capacity = manifest.get("epoch_capacity")
    num_classes = manifest.get("num_classes")
    if (
        manifest.get("run_id") != expected_identity
        or journal.get("run_id") != expected_identity
        or not isinstance(epoch_capacity, int)
        or isinstance(epoch_capacity, bool)
        or epoch_capacity <= 0
        or not isinstance(num_classes, int)
        or isinstance(num_classes, bool)
        or num_classes <= 1
        or journal.get("epoch_capacity") != epoch_capacity
        or manifest.get("completed_slots") != list(range(epoch_capacity))
        or journal.get("committed_slots") != list(range(epoch_capacity))
        or journal.get("current") is not None
    ):
        raise StorageError("candidate storage identity/completion contract is incompatible")

    expected = {
        "selector_visible": (
            SELECTOR_SCHEMA_VERSION,
            "selector_visible",
            SELECTOR_FILENAME,
        ),
        "exploratory_hidden_metrics": (
            HIDDEN_SCHEMA_VERSION,
            "exploratory_hidden_metrics",
            HIDDEN_FILENAME,
        ),
    }
    if not isinstance(manifest.get("files"), dict) or set(
        manifest["files"]
    ) != set(expected):
        raise StorageError("candidate storage manifest file set is incompatible")
    paths: dict[str, Path] = {}
    for name, (schema, _, filename) in expected.items():
        record = manifest.get("files", {}).get(name, {})
        relative = record.get("path")
        if relative != filename:
            raise StorageError(
                f"candidate manifest path is not the locked filename for {name}"
            )
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise StorageError(f"candidate manifest path escapes run root: {relative}") from error
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("size_bytes", -1))
            or sha256_file(path) != record.get("sha256")
            or record.get("schema_version") != schema
        ):
            raise StorageError(f"candidate HDF5 manifest verification failed for {name}")
        paths[name] = path

    h5py = _require_h5py()
    with h5py.File(paths["selector_visible"], "r") as selector, h5py.File(
        paths["exploratory_hidden_metrics"], "r"
    ) as hidden:
        if (
            selector.attrs.get("schema_version") != SELECTOR_SCHEMA_VERSION
            or selector.attrs.get("namespace") != "selector_visible"
            or hidden.attrs.get("schema_version") != HIDDEN_SCHEMA_VERSION
            or hidden.attrs.get("namespace") != "exploratory_hidden_metrics"
        ):
            raise StorageError("published candidate HDF5 namespace/schema mismatch")
        if (
            selector.attrs.get("run_id") != expected_identity
            or hidden.attrs.get("run_id") != expected_identity
            or int(selector.attrs.get("epoch_capacity", -1)) != epoch_capacity
            or int(hidden.attrs.get("epoch_capacity", -1)) != epoch_capacity
            or int(selector.attrs.get("num_classes", -1)) != num_classes
            or int(hidden.attrs.get("num_classes", -1)) != num_classes
        ):
            raise StorageError("published candidate HDF5 identity attributes disagree")
        selector_complete = np.asarray(selector["epochs/complete"][:], dtype=bool)
        hidden_complete = np.asarray(hidden["epochs/complete"][:], dtype=bool)
        if (
            selector_complete.shape != (epoch_capacity,)
            or hidden_complete.shape != (epoch_capacity,)
            or not selector_complete.all()
            or not np.array_equal(selector_complete, hidden_complete)
        ):
            raise StorageError("published candidate HDF5 pair has incomplete epochs")
        selector_numbers = np.asarray(selector["epochs/number"][:], dtype=np.int64)
        hidden_numbers = np.asarray(hidden["epochs/number"][:], dtype=np.int64)
        if (
            selector_numbers.shape != (epoch_capacity,)
            or hidden_numbers.shape != (epoch_capacity,)
            or not np.array_equal(selector_numbers, hidden_numbers)
            or np.any(selector_numbers < 0)
            or (epoch_capacity > 1 and np.any(np.diff(selector_numbers) <= 0))
        ):
            raise StorageError("published candidate HDF5 epoch numbers disagree")
        if "group" in selector["samples"]:
            raise StorageError("published selector-visible HDF5 exposes groups")
    return manifest
