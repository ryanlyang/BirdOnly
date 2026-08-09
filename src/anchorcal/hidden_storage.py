"""Read-only API for AnchorCal reporting-only candidate metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .errors import StorageError


HIDDEN_FILENAME = "exploratory_hidden_metrics.h5"
HIDDEN_SCHEMA_VERSION = "anchorcal-candidate-hidden-v1"
REQUIRED_HIDDEN_SPLITS = ("oracle_val", "test")


def _require_h5py():
    try:
        import h5py
    except (ImportError, ModuleNotFoundError) as error:
        raise StorageError(
            "reporting-only candidate storage requires h5py; "
            "run the frozen environment preflight"
        ) from error
    return h5py


class HiddenMetricsReader:
    """Read one reporting-only HDF5 file after selection has been frozen."""

    def __init__(
        self, path: str | Path, *, expected_run_id: str | None = None
    ) -> None:
        h5py = _require_h5py()
        self.path = Path(path)
        self._h5 = h5py.File(self.path, "r")
        try:
            if self._h5.attrs.get("schema_version") != HIDDEN_SCHEMA_VERSION:
                raise StorageError("not an AnchorCal reporting-only HDF5 file")
            if self._h5.attrs.get("namespace") != "exploratory_hidden_metrics":
                raise StorageError("candidate HDF5 namespace is not reporting-only")
            if (
                expected_run_id is not None
                and self._h5.attrs.get("run_id") != expected_run_id
            ):
                raise StorageError("reporting-only HDF5 run ID mismatch")
        except BaseException:
            self._h5.close()
            raise

    @property
    def completed_slots(self) -> tuple[int, ...]:
        flags = np.asarray(self._h5["epochs/complete"][:], dtype=bool)
        return tuple(int(value) for value in np.flatnonzero(flags))

    def sample_metadata(self, split: str) -> dict[str, np.ndarray]:
        if split not in REQUIRED_HIDDEN_SPLITS:
            raise KeyError(split)
        samples = self._h5[f"splits/{split}/samples"]
        return {name: samples[name][:] for name in ("img_id", "label", "group")}

    def read_epoch(self, split: str, slot: int) -> dict[str, Any]:
        if split not in REQUIRED_HIDDEN_SPLITS:
            raise KeyError(split)
        if not bool(self._h5["epochs/complete"][slot]):
            raise StorageError(f"reporting-only epoch slot {slot} is incomplete")
        group = self._h5[f"splits/{split}/predictions"]
        return {
            "epoch_number": int(self._h5["epochs/number"][slot]),
            "logits": group["logits"][slot, ...],
            "prediction": group["prediction"][slot, ...],
            "correct": group["correct"][slot, ...].astype(bool),
            "loss": group["loss"][slot, ...],
        }

    def close(self) -> None:
        self._h5.close()

    def __enter__(self) -> "HiddenMetricsReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
