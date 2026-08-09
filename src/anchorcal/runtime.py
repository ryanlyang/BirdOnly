"""Runtime provenance and fail-closed production guards."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .errors import PreflightError
from .io import atomic_write_json, atomic_write_text, sha256_bytes


REQUIRED_PACKAGES = (
    "torch",
    "torchvision",
    "timm",
    "h5py",
    "safetensors",
    "huggingface_hub",
    "numpy",
    "pandas",
    "Pillow",
    "PyYAML",
    "scipy",
    "scikit-learn",
    "matplotlib",
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def git_state(repo_root: str | Path) -> dict[str, Any]:
    repo = Path(repo_root)
    try:
        commit = _git(repo, "rev-parse", "HEAD")
        status = _git(repo, "status", "--porcelain")
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise PreflightError(f"repository is not a usable Git checkout: {repo}") from error
    return {
        "commit": commit,
        "clean": not bool(status),
        "status": status.splitlines(),
        "dirty_tree_diff_sha256": sha256_bytes(diff),
    }


def enforce_immutable_checkout(repo_root: str | Path, expected_commit: str) -> None:
    state = git_state(repo_root)
    if state["commit"] != expected_commit:
        raise PreflightError(
            f"checkout commit {state['commit']} differs from EXPECTED_COMMIT {expected_commit}"
        )
    if not state["clean"]:
        raise PreflightError("production requires a clean Git worktree")


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    missing: list[str] = []
    for package in REQUIRED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
    if missing:
        raise PreflightError(f"required Python packages are missing: {missing}")
    return versions


def environment_manifest(*, require_gh200: bool) -> dict[str, Any]:
    versions = package_versions()
    if versions["timm"] != "1.0.28":
        raise PreflightError(f"timm must be 1.0.28, found {versions['timm']}")
    architecture = platform.machine()
    gpu_name: str | None = None
    torch_details: dict[str, Any] = {}
    import torch

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    torch_details = {
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu_name,
    }
    if require_gh200:
        if architecture != "aarch64":
            raise PreflightError(f"production requires aarch64, found {architecture}")
        if gpu_name is None or "GH200" not in gpu_name.upper():
            raise PreflightError(f"production requires an NVIDIA GH200, found {gpu_name}")
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "architecture": architecture,
        "hostname": platform.node(),
        "packages": versions,
        "torch": torch_details,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "job_receipt": os.environ.get("ANCHORCAL_JOB_RECEIPT"),
        "job_receipt_sha256": os.environ.get("ANCHORCAL_JOB_RECEIPT_SHA256"),
        "determinism_warning_record": (
            "Slurm stderr path and warn-only policy are frozen in the job receipt"
            if os.environ.get("ANCHORCAL_JOB_RECEIPT")
            else "local process stderr"
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }


def write_package_lock(path: str | Path) -> None:
    versions = package_versions()
    text = "".join(f"{name}=={version}\n" for name, version in sorted(versions.items()))
    atomic_write_text(path, text)


def configure_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def save_environment_manifest(path: str | Path, *, require_gh200: bool) -> dict[str, Any]:
    manifest = environment_manifest(require_gh200=require_gh200)
    atomic_write_json(path, manifest)
    return manifest
