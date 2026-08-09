"""Read-only API for the physically selector-visible candidate namespace.

This module is deliberately self-contained.  In particular, importing it does
not import the paired candidate writer or any reporting-only schema.  That
one-way dependency boundary lets practical selection run in a process which
cannot name or open reporting-only candidate artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .errors import StorageError


SELECTOR_FILENAME = "candidate_outputs.h5"
SELECTOR_SCHEMA_VERSION = "anchorcal-candidate-visible-v1"

_ROOT_KEYS = frozenset({"epochs", "samples", "predictions", "metrics"})
_ROOT_ATTRS = frozenset(
    {
        "schema_version",
        "namespace",
        "split",
        "run_id",
        "epoch_capacity",
        "num_classes",
        "metadata_sha256",
    }
)
_EPOCH_KEYS = frozenset({"number", "complete", "committed_unix_ns"})
_SAMPLE_KEYS = frozenset({"img_id", "label"})
_PREDICTION_KEYS = frozenset({"logits", "prediction", "correct", "loss"})
_SUBSET_KEYS = frozenset({"samples", "per_example"})


def _require_exact_keys(container: Any, expected: frozenset[str], name: str) -> None:
    actual = set(container.keys())
    if actual != expected:
        raise StorageError(
            f"selector-visible HDF5 {name} is not exactly allowlisted: "
            f"expected={sorted(expected)}, actual={sorted(actual)}"
        )


def _validate_visible_layout(h5: Any) -> None:
    """Reject every non-schema field before exposing any selector data."""

    root_keys = set(_ROOT_KEYS)
    if "selector_subset" in h5:
        root_keys.add("selector_subset")
    _require_exact_keys(h5, frozenset(root_keys), "root datasets")
    if set(h5.attrs.keys()) != _ROOT_ATTRS:
        raise StorageError("selector-visible HDF5 root attributes are not allowlisted")
    _require_exact_keys(h5["epochs"], _EPOCH_KEYS, "epoch datasets")
    _require_exact_keys(h5["samples"], _SAMPLE_KEYS, "sample datasets")
    _require_exact_keys(h5["predictions"], _PREDICTION_KEYS, "prediction datasets")
    if "selector_subset" in h5:
        subset = h5["selector_subset"]
        _require_exact_keys(subset, _SUBSET_KEYS, "selector-subset groups")
        if set(subset.attrs.keys()) != {"metadata_sha256"}:
            raise StorageError(
                "selector-visible HDF5 selector-subset attributes are not allowlisted"
            )
        _require_exact_keys(
            subset["samples"], _SAMPLE_KEYS, "selector-subset sample datasets"
        )


def _require_h5py():
    try:
        import h5py
    except (ImportError, ModuleNotFoundError) as error:
        raise StorageError(
            "selector-visible candidate storage requires h5py; "
            "run the frozen environment preflight"
        ) from error
    return h5py


class SelectorVisibleReader:
    """Read exactly one selector-visible HDF5 file.

    The public API exposes only biased-validation sample identity, predictions,
    practical scalar metrics, and the fixed selector-subset diagnostics.
    """

    def __init__(
        self, path: str | Path, *, expected_run_id: str | None = None
    ) -> None:
        h5py = _require_h5py()
        self.path = Path(path)
        self._h5 = h5py.File(self.path, "r")
        try:
            if self._h5.attrs.get("schema_version") != SELECTOR_SCHEMA_VERSION:
                raise StorageError("not an AnchorCal selector-visible HDF5 file")
            if self._h5.attrs.get("namespace") != "selector_visible":
                raise StorageError("candidate HDF5 namespace is not selector-visible")
            if self._h5.attrs.get("split") != "biased_val":
                raise StorageError("selector-visible HDF5 split is not biased_val")
            if (
                expected_run_id is not None
                and self._h5.attrs.get("run_id") != expected_run_id
            ):
                raise StorageError("selector-visible HDF5 run ID mismatch")
            _validate_visible_layout(self._h5)
        except BaseException:
            self._h5.close()
            raise

    @property
    def completed_slots(self) -> tuple[int, ...]:
        flags = np.asarray(self._h5["epochs/complete"][:], dtype=bool)
        return tuple(int(value) for value in np.flatnonzero(flags))

    def sample_metadata(self) -> dict[str, np.ndarray]:
        return {
            "img_id": self._h5["samples/img_id"][:],
            "label": self._h5["samples/label"][:],
        }

    def selector_subset_metadata(self) -> dict[str, np.ndarray] | None:
        if "selector_subset" not in self._h5:
            return None
        return {
            "img_id": self._h5["selector_subset/samples/img_id"][:],
            "label": self._h5["selector_subset/samples/label"][:],
        }

    def validate_candidate_schema(
        self,
        *,
        metric_names: tuple[str, ...],
        per_example_shapes: dict[str, tuple[int, ...]] | Any,
    ) -> None:
        """Fail closed unless this file has the complete locked candidate schema."""

        epoch_capacity = int(self._h5.attrs.get("epoch_capacity", -1))
        if epoch_capacity <= 0 or "metrics" not in self._h5:
            raise StorageError("selector-visible HDF5 lacks its metric schema")
        if set(self._h5["metrics"].keys()) != set(metric_names):
            raise StorageError("selector-visible scalar metric schema is incomplete")
        for name in metric_names:
            dataset = self._h5[f"metrics/{name}"]
            if dataset.shape != (epoch_capacity,) or dataset.dtype != np.dtype(np.float64):
                raise StorageError(
                    f"selector-visible metric has invalid shape/dtype: {name}"
                )
        if "selector_subset" not in self._h5:
            raise StorageError("selector-visible HDF5 lacks the fixed selector subset")
        arrays = self._h5["selector_subset/per_example"]
        if set(arrays.keys()) != set(per_example_shapes):
            raise StorageError("selector-visible per-example schema is incomplete")
        for name, shape in per_example_shapes.items():
            dataset = arrays[name]
            if dataset.shape != (epoch_capacity, *tuple(shape)) or dataset.dtype != np.dtype(
                np.float32
            ):
                raise StorageError(
                    f"selector-visible per-example array has invalid shape/dtype: {name}"
                )

    def read_epoch(self, slot: int) -> dict[str, Any]:
        if not bool(self._h5["epochs/complete"][slot]):
            raise StorageError(f"selector epoch slot {slot} is incomplete")
        result = {
            "epoch_number": int(self._h5["epochs/number"][slot]),
            "logits": self._h5["predictions/logits"][slot, ...],
            "prediction": self._h5["predictions/prediction"][slot, ...],
            "correct": self._h5["predictions/correct"][slot, ...].astype(bool),
            "loss": self._h5["predictions/loss"][slot, ...],
            "metrics": {
                name: float(dataset[slot])
                for name, dataset in self._h5["metrics"].items()
            },
        }
        if "selector_subset" in self._h5:
            result["selector_per_example"] = {
                name: dataset[slot, ...]
                for name, dataset in self._h5[
                    "selector_subset/per_example"
                ].items()
            }
        return result

    def close(self) -> None:
        self._h5.close()

    def __enter__(self) -> "SelectorVisibleReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
