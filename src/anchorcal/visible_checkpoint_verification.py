"""Read-only verification for selector-visible candidate checkpoints.

This module and its import closure deliberately know only the practical
selector checkpoint schema.  Selection can therefore verify retained model
states without importing or naming any reporting-only artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import StorageError
from .io import sha256_file


CHECKPOINT_SCHEMA_VERSION = "anchorcal-rolling-checkpoints-v1"
MODEL_OBJECT_SCHEMA_VERSION = "anchorcal-model-object-v1"
RESUME_SCHEMA_VERSION = "anchorcal-resume-checkpoint-v1"

VISIBLE_SELECTORS = ("ordinary", "saliency", "swap", "blur", "final")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StorageError(message)


def _require_torch():
    try:
        import torch
    except (ImportError, ModuleNotFoundError) as error:
        raise StorageError(
            "checkpoint verification requires torch in the frozen environment"
        ) from error
    return torch


def _hash_length_prefixed(digest: Any, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def hash_model_state(model_state: Mapping[str, Any]) -> str:
    """Hash tensor values canonically, independent of ``torch.save`` bytes."""

    torch = _require_torch()
    if not isinstance(model_state, Mapping) or not model_state:
        raise ValueError("model_state must be a non-empty mapping")
    digest = hashlib.sha256()
    for name in sorted(model_state):
        if not isinstance(name, str):
            raise ValueError("model_state keys must be strings")
        _hash_length_prefixed(digest, name.encode("utf-8"))
        value = model_state[name]
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            _hash_length_prefixed(digest, str(tensor.dtype).encode("ascii"))
            _hash_length_prefixed(
                digest, ",".join(str(part) for part in tensor.shape).encode("ascii")
            )
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
            _hash_length_prefixed(digest, raw)
        else:
            _hash_length_prefixed(digest, type(value).__qualname__.encode("utf-8"))
            _hash_length_prefixed(
                digest, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            )
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"checkpoint manifest is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StorageError(f"invalid checkpoint manifest: {path}") from error
    _require(isinstance(value, dict), f"checkpoint manifest is not a mapping: {path}")
    return value


def _safe_path(root: Path, relative_value: Any, expected: str) -> Path:
    _require(isinstance(relative_value, str), "checkpoint path must be a string")
    relative = Path(relative_value)
    _require(
        not relative.is_absolute() and ".." not in relative.parts,
        f"checkpoint path escapes its root: {relative_value}",
    )
    _require(
        relative.as_posix() == expected,
        f"checkpoint path is not the locked path {expected}: {relative_value}",
    )
    root = root.resolve()
    unresolved = root / relative
    _require(
        not unresolved.is_symlink(),
        f"checkpoint artifact must not be a symbolic link: {unresolved}",
    )
    path = unresolved.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise StorageError(f"checkpoint path escapes its root: {relative_value}") from error
    _require(path.is_file(), f"checkpoint artifact is missing: {path}")
    try:
        mode = path.stat().st_mode
    except OSError as error:
        raise StorageError(f"checkpoint artifact cannot be inspected: {path}") from error
    _require(stat.S_ISREG(mode), f"checkpoint artifact is not a regular file: {path}")
    return path


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_file_record(
    root: Path,
    record: Any,
    *,
    expected_relative: str,
) -> tuple[Path, str, int]:
    _require(isinstance(record, dict), "checkpoint file record is not a mapping")
    path = _safe_path(root, record.get("path"), expected_relative)
    expected_hash = record.get("sha256")
    expected_size = record.get("size_bytes")
    _require(_valid_sha256(expected_hash), f"invalid checkpoint SHA-256: {path}")
    _require(
        isinstance(expected_size, int)
        and not isinstance(expected_size, bool)
        and expected_size > 0,
        f"invalid checkpoint byte size: {path}",
    )
    actual_size = path.stat().st_size
    _require(actual_size == expected_size, f"checkpoint byte size mismatch: {path}")
    actual_hash = sha256_file(path)
    _require(actual_hash == expected_hash, f"checkpoint hash mismatch: {path}")
    return path, actual_hash, actual_size


def _torch_load(path: Path) -> Mapping[str, Any]:
    try:
        payload = _require_torch().load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise StorageError(f"checkpoint payload cannot be loaded: {path}") from error
    _require(isinstance(payload, Mapping), f"checkpoint payload is not a mapping: {path}")
    return payload


def _verify_model_object(
    root: Path, run_id: str, model_hash: str, record: Any
) -> dict[str, Any]:
    _require(_valid_sha256(model_hash), "checkpoint model hash key is invalid")
    _require(isinstance(record, dict), "checkpoint model record is not a mapping")
    _require(record.get("model_hash") == model_hash, "checkpoint object hash key mismatch")
    expected_relative = f"weights/model_{model_hash}.pt"
    path, file_hash, size = _verify_file_record(
        root, record, expected_relative=expected_relative
    )
    epochs = record.get("epochs")
    _require(isinstance(epochs, list) and epochs, "checkpoint object epoch list is empty")
    _require(
        all(
            isinstance(epoch, int) and not isinstance(epoch, bool) and epoch >= 0
            for epoch in epochs
        )
        and epochs == sorted(set(epochs)),
        "checkpoint object epochs are invalid",
    )
    payload = _torch_load(path)
    _require(
        payload.get("schema_version") == MODEL_OBJECT_SCHEMA_VERSION
        and payload.get("run_id") == run_id
        and payload.get("model_hash") == model_hash,
        f"checkpoint model payload provenance mismatch: {path}",
    )
    try:
        actual_model_hash = hash_model_state(payload.get("model_state", {}))
    except (TypeError, ValueError) as error:
        raise StorageError(f"checkpoint model state is invalid: {path}") from error
    _require(actual_model_hash == model_hash, f"checkpoint model-state hash mismatch: {path}")
    return {
        "path": str(path),
        "relative_path": expected_relative,
        "sha256": file_hash,
        "size_bytes": size,
        "model_hash": model_hash,
        "epochs": tuple(epochs),
    }


def _verify_resume_payload(
    root: Path,
    run_id: str,
    record: Any,
    *,
    expected_relative: str,
) -> dict[str, Any]:
    path, file_hash, size = _verify_file_record(
        root, record, expected_relative=expected_relative
    )
    epoch = record.get("epoch")
    model_hash = record.get("model_hash")
    _require(
        isinstance(epoch, int) and not isinstance(epoch, bool) and epoch >= 0,
        "resume checkpoint epoch is invalid",
    )
    _require(_valid_sha256(model_hash), "resume checkpoint model hash is invalid")
    _require(record.get("at_epoch_boundary") is True, "resume is not at an epoch boundary")
    payload = _torch_load(path)
    _require(
        payload.get("schema_version") == RESUME_SCHEMA_VERSION
        and payload.get("run_id") == run_id
        and payload.get("epoch") == epoch
        and payload.get("model_hash") == model_hash
        and payload.get("at_epoch_boundary") is True,
        f"resume checkpoint payload provenance mismatch: {path}",
    )
    try:
        actual_model_hash = hash_model_state(payload.get("model_state", {}))
    except (TypeError, ValueError) as error:
        raise StorageError(f"resume checkpoint model state is invalid: {path}") from error
    _require(
        actual_model_hash == model_hash,
        f"resume checkpoint model-state hash mismatch: {path}",
    )
    return {
        "path": str(path),
        "relative_path": expected_relative,
        "sha256": file_hash,
        "size_bytes": size,
        "model_hash": model_hash,
        "epoch": epoch,
    }


def verify_visible_checkpoint_artifacts(
    run_dir: str | Path,
    *,
    expected_run_id: str | None = None,
    require_complete: bool = False,
    required_selectors: Sequence[str] = (),
) -> dict[str, Any]:
    """Verify the selector-visible checkpoint namespace without mutation."""

    run_dir = Path(run_dir).resolve()
    root = run_dir / "checkpoints"
    manifest_path = root / "manifest.json"
    manifest = _json(manifest_path)
    run_id = manifest.get("run_id")
    _require(
        isinstance(run_id, str)
        and run_id
        and not any(character in run_id for character in "/\\\n\r"),
        "checkpoint manifest run ID is invalid",
    )
    if expected_run_id is not None:
        _require(run_id == expected_run_id, "checkpoint manifest run ID mismatch")
    _require(
        manifest.get("schema_version") == CHECKPOINT_SCHEMA_VERSION,
        "checkpoint schema is incompatible",
    )
    selectors = manifest.get("selectors")
    _require(
        isinstance(selectors, dict) and set(selectors) == set(VISIBLE_SELECTORS),
        "checkpoint selector set is invalid",
    )
    _require(
        set(required_selectors).issubset(VISIBLE_SELECTORS),
        "unknown required checkpoint selector",
    )
    for selector in required_selectors:
        _require(
            selectors.get(selector) is not None,
            f"required checkpoint selector is missing: {selector}",
        )

    objects_value = manifest.get("objects")
    _require(isinstance(objects_value, dict), "checkpoint object inventory is invalid")
    objects = {
        model_hash: _verify_model_object(root, run_id, model_hash, record)
        for model_hash, record in objects_value.items()
    }
    referenced: set[str] = set()
    selector_summary: dict[str, Any] = {}
    for selector, selection in selectors.items():
        if selection is None:
            selector_summary[selector] = None
            continue
        _require(isinstance(selection, dict), f"checkpoint selector is invalid: {selector}")
        model_hash = selection.get("model_hash")
        _require(model_hash in objects, f"selector references an unknown model: {selector}")
        record = objects[model_hash]
        epoch = selection.get("epoch")
        _require(
            isinstance(epoch, int)
            and not isinstance(epoch, bool)
            and epoch in record["epochs"],
            f"selector epoch is inconsistent: {selector}",
        )
        _require(
            selection.get("path") == record["relative_path"],
            f"selector path is inconsistent: {selector}",
        )
        ranking_key = selection.get("ranking_key")
        _require(
            isinstance(ranking_key, list)
            and ranking_key
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in ranking_key
            ),
            f"selector ranking key is invalid: {selector}",
        )
        _require(
            isinstance(selection.get("metadata"), dict),
            f"selector metadata is invalid: {selector}",
        )
        referenced.add(model_hash)
        selector_summary[selector] = {
            "epoch": epoch,
            "model_hash": model_hash,
            "path": record["relative_path"],
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
        }
    _require(
        referenced == set(objects),
        "checkpoint manifest contains unreferenced model objects",
    )

    resume_summary = None
    completion_summary = None
    resume = manifest.get("resume")
    completion = manifest.get("completion")
    if resume is not None:
        resume_summary = _verify_resume_payload(
            root, run_id, resume, expected_relative="resume.pt"
        )
    if completion is not None:
        _require(isinstance(completion, dict), "checkpoint completion is invalid")
        policy = completion.get("resume_policy")
        _require(policy in {"archive", "delete"}, "checkpoint completion policy is invalid")
        _require(resume is None, "completed checkpoint manifest still exposes resume state")
        _require(
            not (root / "resume.pt").exists(),
            "completed run retained an unmanifested resume.pt",
        )
        final_record = completion.get("final_state")
        if policy == "archive":
            _require(final_record is not None, "archived completion lacks final state")
            final_summary = _verify_resume_payload(
                root, run_id, final_record, expected_relative="final_state.pt"
            )
        else:
            _require(final_record is None, "delete completion unexpectedly names final state")
            _require(
                not (root / "final_state.pt").exists(),
                "delete completion retained final state",
            )
            final_summary = None
        completion_summary = {"resume_policy": policy, "final_state": final_summary}
    else:
        _require(
            not (root / "final_state.pt").exists(),
            "incomplete checkpoint manifest has an unmanifested final state",
        )
    if require_complete:
        _require(completion_summary is not None, "candidate checkpoint set is incomplete")
        _require(resume_summary is None, "completed candidate still has resume state")

    return {
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
        "selectors": selector_summary,
        "objects": objects,
        "resume": resume_summary,
        "completion": completion_summary,
    }
