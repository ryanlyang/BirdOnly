"""Atomic restart and deduplicated rolling candidate checkpoints.

Only selector-winning, oracle-winning, and final model states are retained.
Oracle selection metadata lives in a physically separate hidden manifest so
ordinary selector code cannot learn its epoch or score.  Weight objects may be
shared by content hash across manifests without exposing why they were saved.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .checkpoint_verification import (
    CHECKPOINT_SCHEMA_VERSION,
    HIDDEN_CHECKPOINT_SCHEMA_VERSION,
    HIDDEN_SELECTORS,
    MODEL_OBJECT_SCHEMA_VERSION,
    RESUME_SCHEMA_VERSION,
    VISIBLE_SELECTORS,
    hash_model_state,
)
from .errors import StorageError
from .io import atomic_write_json, sha256_file
from .storage import ExclusiveFileLock


ROLLING_SELECTORS = VISIBLE_SELECTORS[:-1] + HIDDEN_SELECTORS
ALL_SELECTORS = VISIBLE_SELECTORS + HIDDEN_SELECTORS


def _require_torch():
    try:
        import torch
    except (ImportError, ModuleNotFoundError) as error:
        raise StorageError(
            "checkpoint serialization requires torch in the frozen environment"
        ) from error
    return torch


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_torch_save(path: Path, payload: Any) -> None:
    torch = _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        torch.save(payload, temporary_path)
        file_descriptor = os.open(temporary_path, os.O_RDONLY)
        try:
            os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _json_ranking_key(values: Sequence[int | float]) -> list[int | float]:
    result: list[int | float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("ranking keys must contain only finite numbers")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("ranking keys must contain only finite numbers")
        result.append(value)
    if not result:
        raise ValueError("ranking key must not be empty")
    return result


class CheckpointManager:
    """Maintain one resume state and at most one best state per selector."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        run_id: str,
        verify_hashes_on_open: bool = True,
    ) -> None:
        if not run_id or any(character in run_id for character in "/\\\n\r"):
            raise ValueError(f"unsafe run_id: {run_id!r}")
        self.run_dir = Path(run_dir).resolve()
        self.run_id = run_id
        self.root = self.run_dir / "checkpoints"
        self.weights_dir = self.root / "weights"
        self.hidden_dir = self.root / "exploratory_hidden"
        self.manifest_path = self.root / "manifest.json"
        self.hidden_manifest_path = self.hidden_dir / "oracle_manifest.json"
        self.resume_path = self.root / "resume.pt"
        self.final_state_path = self.root / "final_state.pt"
        self.lock = ExclusiveFileLock(self.root / "checkpoints.lock")
        self._closed = False
        self.root.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.hidden_dir.mkdir(parents=True, exist_ok=True)
        self.lock.acquire()
        try:
            self.visible = self._load_or_initialize_visible()
            self.hidden = self._load_or_initialize_hidden()
            self._validate_manifests(verify_hashes=verify_hashes_on_open)
            self._remove_untracked_weight_files()
        except BaseException:
            self.lock.release()
            raise

    def _load_json(self, path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise StorageError(f"invalid checkpoint manifest: {path}") from error
        if not isinstance(value, dict):
            raise StorageError(f"checkpoint manifest is not a mapping: {path}")
        return value

    def _load_or_initialize_visible(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            return self._load_json(self.manifest_path)
        value = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "selectors": {name: None for name in VISIBLE_SELECTORS},
            "objects": {},
            "resume": None,
            "completion": None,
            "updated_unix_ns": time.time_ns(),
        }
        atomic_write_json(self.manifest_path, value)
        return value

    def _load_or_initialize_hidden(self) -> dict[str, Any]:
        if self.hidden_manifest_path.exists():
            return self._load_json(self.hidden_manifest_path)
        value = {
            "schema_version": HIDDEN_CHECKPOINT_SCHEMA_VERSION,
            "namespace": "exploratory_hidden_metrics",
            "run_id": self.run_id,
            "selectors": {name: None for name in HIDDEN_SELECTORS},
            "objects": {},
            "updated_unix_ns": time.time_ns(),
        }
        atomic_write_json(self.hidden_manifest_path, value)
        return value

    def _validate_manifests(self, *, verify_hashes: bool) -> None:
        if (
            self.visible.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or self.hidden.get("schema_version") != HIDDEN_CHECKPOINT_SCHEMA_VERSION
            or self.visible.get("run_id") != self.run_id
            or self.hidden.get("run_id") != self.run_id
        ):
            raise StorageError("checkpoint manifests do not match this run")
        if set(self.visible.get("selectors", {})) != set(VISIBLE_SELECTORS):
            raise StorageError("visible checkpoint selector set is invalid")
        if set(self.hidden.get("selectors", {})) != set(HIDDEN_SELECTORS):
            raise StorageError("hidden checkpoint selector set is invalid")
        if "oracle" in self.visible["selectors"]:
            raise StorageError("visible checkpoint manifest exposes oracle selection")
        for manifest in (self.visible, self.hidden):
            for model_hash, record in manifest.get("objects", {}).items():
                if record.get("model_hash") != model_hash:
                    raise StorageError("checkpoint object hash key mismatch")
                path = self._resolve_relative(record.get("path"))
                if not path.is_file():
                    raise StorageError(f"checkpoint weight object is missing: {path}")
                if verify_hashes and sha256_file(path) != record.get("sha256"):
                    raise StorageError(f"checkpoint weight object hash mismatch: {path}")
            for selector, selection in manifest["selectors"].items():
                if selection is None:
                    continue
                model_hash = selection.get("model_hash")
                if model_hash not in manifest.get("objects", {}):
                    raise StorageError(
                        f"selector {selector} references an unmanifested model object"
                    )
                object_record = manifest["objects"][model_hash]
                if (
                    selection.get("path") != object_record.get("path")
                    or int(selection.get("epoch", -1))
                    not in object_record.get("epochs", [])
                ):
                    raise StorageError(
                        f"selector {selector} checkpoint reference is inconsistent"
                    )
        resume = self.visible.get("resume")
        if resume is not None:
            path = self._resolve_relative(resume.get("path"))
            if not path.is_file():
                raise StorageError(f"resume checkpoint is missing: {path}")
            if verify_hashes and sha256_file(path) != resume.get("sha256"):
                raise StorageError("resume checkpoint hash mismatch")
        completion = self.visible.get("completion")
        if isinstance(completion, dict) and completion.get("final_state") is not None:
            final_record = completion["final_state"]
            final_path = self._resolve_relative(final_record.get("path"))
            if not final_path.is_file():
                raise StorageError("archived final-state checkpoint is missing")
            if verify_hashes and sha256_file(final_path) != final_record.get("sha256"):
                raise StorageError("archived final-state checkpoint hash mismatch")

    def _resolve_relative(self, relative: Any) -> Path:
        if not isinstance(relative, str):
            raise StorageError("checkpoint manifest path must be a string")
        path = (self.root / relative).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise StorageError(f"checkpoint path escapes its run root: {relative}") from error
        return path

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    def _write_visible(self) -> None:
        self.visible["updated_unix_ns"] = time.time_ns()
        atomic_write_json(self.manifest_path, self.visible)
        _fsync_directory(self.root)

    def _write_hidden(self) -> None:
        self.hidden["updated_unix_ns"] = time.time_ns()
        atomic_write_json(self.hidden_manifest_path, self.hidden)
        _fsync_directory(self.hidden_dir)

    def _referenced_hashes(self) -> set[str]:
        hashes: set[str] = set()
        for manifest in (self.visible, self.hidden):
            for record in manifest["selectors"].values():
                if record is not None:
                    hashes.add(str(record["model_hash"]))
        return hashes

    def _remove_untracked_weight_files(self) -> None:
        referenced = self._referenced_hashes()
        for path in self.weights_dir.glob("model_*.pt"):
            model_hash = path.stem.removeprefix("model_")
            if model_hash not in referenced:
                path.unlink()
        self._prune_object_records()

    def _prune_object_records(self) -> None:
        # Object inventories are namespace-local.  A model retained only by
        # the oracle must not leave an otherwise unreferenced record in the
        # selector-visible manifest (and vice versa), even though the physical
        # weight file is retained by the union of both selector sets.
        visible_referenced = {
            str(record["model_hash"])
            for record in self.visible["selectors"].values()
            if record is not None
        }
        hidden_referenced = {
            str(record["model_hash"])
            for record in self.hidden["selectors"].values()
            if record is not None
        }
        visible_changed = False
        hidden_changed = False
        for model_hash in list(self.visible["objects"]):
            if model_hash not in visible_referenced:
                del self.visible["objects"][model_hash]
                visible_changed = True
        for model_hash in list(self.hidden["objects"]):
            if model_hash not in hidden_referenced:
                del self.hidden["objects"][model_hash]
                hidden_changed = True
        if visible_changed:
            self._write_visible()
        if hidden_changed:
            self._write_hidden()

    def _object_record(self, model_hash: str) -> dict[str, Any] | None:
        return self.visible["objects"].get(model_hash) or self.hidden["objects"].get(
            model_hash
        )

    def _ensure_model_object(
        self,
        model_state: Mapping[str, Any],
        *,
        epoch: int,
    ) -> tuple[str, dict[str, Any]]:
        model_hash = hash_model_state(model_state)
        existing = self._object_record(model_hash)
        if existing is not None:
            path = self._resolve_relative(existing["path"])
            if not path.is_file() or sha256_file(path) != existing["sha256"]:
                raise StorageError("deduplicated model object failed hash verification")
            # Epochs are namespace-local metadata.  In particular, never copy
            # an oracle-selected epoch from the hidden manifest into the
            # selector-visible object inventory merely because weights match.
            return model_hash, {
                "model_hash": model_hash,
                "path": existing["path"],
                "sha256": existing["sha256"],
                "size_bytes": existing["size_bytes"],
                "epochs": [],
            }

        for manifest in (self.visible, self.hidden):
            for other_hash, record in manifest["objects"].items():
                if epoch in record.get("epochs", []) and other_hash != model_hash:
                    raise StorageError(
                        f"epoch {epoch} was already checkpointed with a different model hash"
                    )

        path = self.weights_dir / f"model_{model_hash}.pt"
        payload = {
            "schema_version": MODEL_OBJECT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "model_hash": model_hash,
            "model_state": model_state,
        }
        if path.exists():
            # A crash may leave a valid object before its manifest publication.
            loaded = _require_torch().load(path, map_location="cpu", weights_only=False)
            if loaded.get("model_hash") != model_hash:
                raise StorageError(f"untracked checkpoint object is invalid: {path}")
        else:
            _atomic_torch_save(path, payload)
        record = {
            "model_hash": model_hash,
            "path": self._relative(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "epochs": [int(epoch)],
        }
        return model_hash, record

    def update_selector(
        self,
        selector: str,
        *,
        epoch: int,
        model_state: Mapping[str, Any],
        ranking_key: Sequence[int | float],
        metadata: Mapping[str, Any] | None = None,
        force: bool = False,
    ) -> bool:
        """Update one rolling selector using a caller-defined lexicographic key.

        The caller owns the scientific tie-break construction.  For practical
        criteria it should pass ``(score, biased_accuracy, -cross_entropy,
        -epoch, ...)``; for oracle it should use its locked WGA/group-balanced
        ordering.  This storage layer compares only the resulting key.
        """

        if self._closed:
            raise StorageError("checkpoint manager is closed")
        if selector not in ALL_SELECTORS:
            raise ValueError(f"unknown rolling selector: {selector}")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        key = _json_ranking_key(ranking_key)
        manifest = self.hidden if selector in HIDDEN_SELECTORS else self.visible
        current = manifest["selectors"][selector]
        if not force and current is not None:
            if tuple(key) <= tuple(current["ranking_key"]):
                return False
        if selector == "final" and not force:
            raise ValueError("the final-epoch selector must be updated with force=True")

        model_hash, object_record = self._ensure_model_object(model_state, epoch=epoch)
        object_record = dict(manifest["objects"].get(model_hash, object_record))
        epochs = sorted(set(object_record.get("epochs", [])) | {epoch})
        object_record["epochs"] = epochs
        manifest["objects"][model_hash] = object_record
        manifest["selectors"][selector] = {
            "epoch": epoch,
            "model_hash": model_hash,
            "path": object_record["path"],
            "ranking_key": key,
            "metadata": dict(metadata or {}),
            "updated_unix_ns": time.time_ns(),
        }
        if manifest is self.hidden:
            self._write_hidden()
        else:
            self._write_visible()

        referenced = self._referenced_hashes()
        self._prune_object_records()
        for path in self.weights_dir.glob("model_*.pt"):
            if path.stem.removeprefix("model_") not in referenced:
                path.unlink()
        _fsync_directory(self.weights_dir)
        return True

    def save_final_epoch(
        self,
        *,
        epoch: int,
        model_state: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.update_selector(
            "final",
            epoch=epoch,
            model_state=model_state,
            ranking_key=(epoch,),
            metadata=metadata,
            force=True,
        )

    def save_resume(
        self,
        *,
        epoch: int,
        model_state: Mapping[str, Any],
        optimizer_state: Mapping[str, Any],
        scheduler_state: Mapping[str, Any],
        rng_states: Mapping[str, Any],
        scaler_state: Mapping[str, Any] | None = None,
        dataloader_progress: Mapping[str, Any] | None = None,
        at_epoch_boundary: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise StorageError("checkpoint manager is closed")
        if not at_epoch_boundary:
            raise ValueError("resume checkpoints are permitted only at epoch boundaries")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        model_hash = hash_model_state(model_state)
        payload = {
            "schema_version": RESUME_SCHEMA_VERSION,
            "run_id": self.run_id,
            "epoch": epoch,
            "at_epoch_boundary": True,
            "model_hash": model_hash,
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "rng_states": rng_states,
            "scaler_state": scaler_state,
            "dataloader_progress": dict(dataloader_progress or {}),
            "metadata": dict(metadata or {}),
        }
        _atomic_torch_save(self.resume_path, payload)
        record = {
            "path": self._relative(self.resume_path),
            "sha256": sha256_file(self.resume_path),
            "size_bytes": self.resume_path.stat().st_size,
            "epoch": epoch,
            "model_hash": model_hash,
            "at_epoch_boundary": True,
            "updated_unix_ns": time.time_ns(),
        }
        self.visible["resume"] = record
        self._write_visible()
        return record

    def load_resume(self, *, map_location: str | Any = "cpu") -> dict[str, Any] | None:
        record = self.visible.get("resume")
        if record is None:
            return None
        path = self._resolve_relative(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise StorageError("resume checkpoint hash mismatch")
        payload = _require_torch().load(
            path, map_location=map_location, weights_only=False
        )
        if (
            payload.get("schema_version") != RESUME_SCHEMA_VERSION
            or payload.get("run_id") != self.run_id
            or not payload.get("at_epoch_boundary")
        ):
            raise StorageError("resume checkpoint payload is incompatible")
        try:
            actual_model_hash = hash_model_state(payload.get("model_state", {}))
        except (TypeError, ValueError) as error:
            raise StorageError("resume checkpoint has an invalid model state") from error
        if (
            actual_model_hash != payload.get("model_hash")
            or actual_model_hash != record.get("model_hash")
        ):
            raise StorageError("resume checkpoint model-state hash mismatch")
        return payload

    def complete(self, *, resume_policy: str = "archive") -> None:
        """Finalize restart-state disposition after successful training."""

        if resume_policy not in {"archive", "delete"}:
            raise ValueError("resume_policy must be 'archive' or 'delete'")
        record = self.visible.get("resume")
        completion: dict[str, Any] = {
            "completed_unix_ns": time.time_ns(),
            "resume_policy": resume_policy,
        }
        if record is not None:
            source = self._resolve_relative(record["path"])
            if resume_policy == "archive":
                os.replace(source, self.final_state_path)
                _fsync_directory(self.root)
                completion["final_state"] = {
                    **record,
                    "path": self._relative(self.final_state_path),
                    "sha256": sha256_file(self.final_state_path),
                    "size_bytes": self.final_state_path.stat().st_size,
                }
            else:
                source.unlink()
                _fsync_directory(self.root)
            self.visible["resume"] = None
        self.visible["completion"] = completion
        self._write_visible()

    def close(self) -> None:
        if self._closed:
            return
        self.lock.release()
        self._closed = True

    def __enter__(self) -> "CheckpointManager":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
