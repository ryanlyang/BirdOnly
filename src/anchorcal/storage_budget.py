"""Fail-closed persistent-storage and filesystem-capacity preflight.

The AnchorCal campaign keeps only a bounded set of model states, but six
candidate writers can still create several GiB concurrently.  This module
turns the campaign's conservative storage projection into an executable
contract instead of relying on an informal estimate in a runbook.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Mapping

from .errors import PreflightError


STORAGE_PREFLIGHT_SCHEMA = "anchorcal-storage-preflight-v1"
GIB = 1024**3


def allocated_bytes(root: str | Path) -> int:
    """Return allocated regular-file bytes without following symlinks.

    ``st_blocks`` reflects filesystem allocation more faithfully than logical
    file length for sparse HDF5 files.  Symlinks are deliberately ignored so a
    campaign cannot accidentally charge a referenced dataset or model cache to
    its output budget.
    """

    resolved = Path(root).expanduser().resolve()
    if not resolved.exists():
        return 0
    total = 0
    stack = [resolved]
    while stack:
        current = stack.pop()
        try:
            entries = os.scandir(current)
        except FileNotFoundError:
            continue
        with entries:
            for entry in entries:
                try:
                    stat = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += int(stat.st_blocks) * 512
    return total


def _finite_positive(value: Any, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise PreflightError(f"storage.{name} must be a positive number") from error
    if not 0.0 < converted < float("inf"):
        raise PreflightError(f"storage.{name} must be a positive finite number")
    return converted


def assess_storage_budget(
    config: Mapping[str, Any],
    output_root: str | Path,
    *,
    stage: str,
    filesystem_free_bytes: int | None = None,
    filesystem_total_bytes: int | None = None,
) -> dict[str, Any]:
    """Validate current allocation, projected growth, and free capacity.

    Optional filesystem values make the scientific arithmetic independently
    testable.  Production callers omit them and use ``shutil.disk_usage`` on
    the actual output filesystem.
    """

    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("storage preflight stage must be a non-empty string")
    storage = config.get("storage")
    if not isinstance(storage, Mapping):
        raise PreflightError("locked storage configuration is missing")
    hard = _finite_positive(storage.get("hard_budget_gib"), "hard_budget_gib")
    guard = _finite_positive(storage.get("launch_guard_gib"), "launch_guard_gib")
    minimum_free = _finite_positive(
        storage.get("minimum_filesystem_free_gib"),
        "minimum_filesystem_free_gib",
    )
    concurrent = _finite_positive(
        storage.get("worst_case_concurrent_growth_gib"),
        "worst_case_concurrent_growth_gib",
    )
    components_value = storage.get("projected_full_campaign_components_gib")
    if not isinstance(components_value, Mapping) or not components_value:
        raise PreflightError(
            "storage.projected_full_campaign_components_gib must be a mapping"
        )
    components = {
        str(name): _finite_positive(value, f"projected_full_campaign_components_gib.{name}")
        for name, value in components_value.items()
    }
    projected_growth = sum(components.values())
    if not 0.0 < guard < hard:
        raise PreflightError("storage launch guard must be below the hard budget")
    if concurrent > projected_growth:
        raise PreflightError(
            "concurrent storage reserve exceeds the full-campaign projection"
        )

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    used_bytes = allocated_bytes(root)
    used_gib = used_bytes / GIB
    if filesystem_free_bytes is None or filesystem_total_bytes is None:
        usage = shutil.disk_usage(root)
        free_bytes = int(usage.free)
        total_bytes = int(usage.total)
    else:
        free_bytes = int(filesystem_free_bytes)
        total_bytes = int(filesystem_total_bytes)
    if free_bytes < 0 or total_bytes <= 0 or free_bytes > total_bytes:
        raise PreflightError("filesystem capacity values are invalid")
    free_gib = free_bytes / GIB
    projected_peak_gib = used_gib + projected_growth

    failures: list[str] = []
    if used_gib >= guard:
        failures.append(
            f"allocated {used_gib:.3f} GiB has reached the {guard:.3f} GiB launch guard"
        )
    if used_gib + concurrent > hard:
        failures.append(
            "current allocation plus worst-case concurrent growth exceeds the "
            f"{hard:.3f} GiB hard budget"
        )
    if projected_peak_gib > guard:
        failures.append(
            f"projected campaign peak {projected_peak_gib:.3f} GiB exceeds the "
            f"{guard:.3f} GiB launch guard"
        )
    required_free_gib = max(minimum_free, projected_growth)
    if free_gib < required_free_gib:
        failures.append(
            f"filesystem free space {free_gib:.3f} GiB is below the required "
            f"{required_free_gib:.3f} GiB"
        )
    if failures:
        raise PreflightError(
            f"storage preflight failed before {stage}: " + "; ".join(failures)
        )

    return {
        "schema_version": STORAGE_PREFLIGHT_SCHEMA,
        "status": "passed",
        "stage": stage,
        "output_root": str(root),
        "allocated_bytes": used_bytes,
        "allocated_gib": used_gib,
        "filesystem_total_bytes": total_bytes,
        "filesystem_free_bytes": free_bytes,
        "filesystem_free_gib": free_gib,
        "filesystem_capacity_scope": (
            "filesystem_statvfs_capacity; site-enforced per-user quota may be lower"
        ),
        "hard_budget_gib": hard,
        "launch_guard_gib": guard,
        "minimum_filesystem_free_gib": minimum_free,
        "worst_case_concurrent_growth_gib": concurrent,
        "projected_full_campaign_components_gib": components,
        "projected_full_campaign_growth_gib": projected_growth,
        "projected_peak_gib": projected_peak_gib,
        "required_free_gib": required_free_gib,
    }

