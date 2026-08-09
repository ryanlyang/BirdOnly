"""Immutable timestamped hash receipts for campaign chronology."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_hashed_receipt(
    directory: str | Path, prefix: str, payload: dict[str, Any]
) -> tuple[Path, Path]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    stamp = timestamp.strftime("%Y%m%dT%H%M%S.%fZ")
    receipt = root / f"{prefix}_{stamp}.json"
    value = {
        "created_at_utc": timestamp.isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        ),
        **payload,
    }
    serialized = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{receipt.name}.", suffix=".tmp", dir=root
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, receipt)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    digest = hashlib.sha256(serialized).hexdigest()
    sidecar = receipt.with_suffix(receipt.suffix + ".sha256")
    sidecar_payload = f"{digest}  {receipt.name}\n".encode("ascii")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{sidecar.name}.", suffix=".tmp", dir=root
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(sidecar_payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, sidecar)
        temporary.unlink()
        directory_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        receipt.unlink(missing_ok=True)
        raise
    return receipt, sidecar


def verify_hashed_receipt(path: str | Path) -> bool:
    receipt = Path(path)
    sidecar = receipt.with_suffix(receipt.suffix + ".sha256")
    try:
        digest, name = sidecar.read_text(encoding="ascii").split()
        return name == receipt.name and hashlib.sha256(receipt.read_bytes()).hexdigest() == digest
    except (OSError, ValueError):
        return False
