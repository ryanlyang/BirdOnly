"""Read-only verification of the vendored official uLA source."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from setv.errors import DataValidationError
from setv.utils.hashing import sha256_file

OFFICIAL_URL = "https://github.com/tsirif/uLA"
OFFICIAL_COMMIT = "5867fb6e9a8485ed08b4cbe84900f2b5ac4fac5d"
OFFICIAL_TREE_SHA256 = "44004b6f24dffa16e233f9ee62dc8a1acad4ae5397504df4873f04af8e843a07"
REQUIRED_FILES = (
    "README.md",
    "main_pretrain.py",
    "main_train.py",
    "requirements.txt",
    "scripts/mocov2plus.sh",
    "scripts/ula.sh",
    "ula/methods/ula.py",
    "ula/utils/metrics.py",
)


def tree_sha256(root: str | Path) -> tuple[str, dict[str, str]]:
    root = Path(root).expanduser().resolve()
    files: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if ".git" in path.relative_to(root).parts or "__pycache__" in path.parts:
            continue
        files[path.relative_to(root).as_posix()] = sha256_file(path)
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest, files


def audit_official_source(root: str | Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise DataValidationError(f"uLA source directory does not exist: {root}")
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise DataValidationError(f"uLA vendor is incomplete: {missing}")
    digest, files = tree_sha256(root)
    if digest != OFFICIAL_TREE_SHA256:
        raise DataValidationError(
            "uLA source tree does not match the audited official commit: "
            f"expected {OFFICIAL_TREE_SHA256}, got {digest}"
        )
    commit = None
    remote = None
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        remote = subprocess.run(
            ["git", "-C", str(root), "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return {
        "repository": OFFICIAL_URL,
        "required_commit": OFFICIAL_COMMIT,
        "required_tree_sha256": OFFICIAL_TREE_SHA256,
        "resolved_path": str(root),
        "git_checkout": commit is not None,
        "git_commit": commit,
        "git_remote": remote,
        "vendor_tree_sha256": digest,
        "vendor_file_count": len(files),
        "required_file_sha256": {name: files[name] for name in REQUIRED_FILES},
        "commit_identity_source": (
            "git_checkout" if commit is not None else "verified_vendor_tree"
        ),
    }
