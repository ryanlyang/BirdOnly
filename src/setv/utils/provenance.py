"""Runtime and source provenance capture."""

from __future__ import annotations

import importlib.metadata
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git(command: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *command],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def git_provenance(repo: str | Path) -> dict[str, Any]:
    root = Path(repo).resolve()
    top = _git(["rev-parse", "--show-toplevel"], root)
    if top is None:
        return {"available": False, "requested_path": str(root)}
    top_path = Path(top)
    return {
        "available": True,
        "top_level": top,
        "commit": _git(["rev-parse", "HEAD"], top_path),
        "branch": _git(["branch", "--show-current"], top_path),
        "status_short": _git(["status", "--short"], top_path),
        "remote": _git(["remote", "-v"], top_path),
    }


def package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def runtime_provenance(repo: str | Path, storage_path: str | Path) -> dict[str, Any]:
    usage = shutil.disk_usage(Path(storage_path).resolve().parent)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": package_versions(("numpy", "pandas", "Pillow", "PyYAML")),
        "git": git_provenance(repo),
        "storage": {
            "path": str(Path(storage_path).resolve().parent),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        },
    }

