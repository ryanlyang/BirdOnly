"""AnchorCal criterion choice and immutable decision receipts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .statistics import (
    DEFAULT_ANCHOR_SCORE_TOLERANCE,
    AnchorScoreMetrics,
    BootstrapCriterionResult,
)


TIE_BREAK_RULES = (
    "minimum point-estimate ACE",
    "higher adjacent ordering accuracy for ACE ties",
    "higher pairwise ordering accuracy",
    "lower bootstrap ACE standard deviation",
    "lower prespecified computational cost",
    "earlier criterion in the prespecified eligible-criterion order if still identical",
)


@dataclass(frozen=True)
class CriterionDecision:
    eligible_criteria: tuple[str, ...]
    winner: str
    credible_set: tuple[str, ...]
    point_metrics: dict[str, AnchorScoreMetrics]
    bootstrap_ace_standard_deviation: dict[str, float]
    computational_cost: dict[str, float]
    best_point_ace: float
    one_standard_error: float
    credible_set_threshold: float
    tolerance: float
    tie_break_trace: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "eligible_criteria": list(self.eligible_criteria),
            "winner": self.winner,
            "credible_set": list(self.credible_set),
            "point_metrics": {
                name: self.point_metrics[name].to_dict()
                for name in self.eligible_criteria
            },
            "bootstrap_ace_standard_deviation": {
                name: self.bootstrap_ace_standard_deviation[name]
                for name in self.eligible_criteria
            },
            "computational_cost": {
                name: self.computational_cost[name]
                for name in self.eligible_criteria
            },
            "best_point_ace": self.best_point_ace,
            "one_standard_error": self.one_standard_error,
            "credible_set_threshold": self.credible_set_threshold,
            "anchor_score_tolerance": self.tolerance,
            "tie_break_rules": list(TIE_BREAK_RULES),
            "tie_break_trace": list(self.tie_break_trace),
        }


def _narrow_minimum(
    candidates: list[str],
    values: Mapping[str, float],
    tolerance: float,
) -> list[str]:
    best = min(values[name] for name in candidates)
    return [name for name in candidates if values[name] <= best + tolerance]


def _narrow_maximum(
    candidates: list[str],
    values: Mapping[str, float],
    tolerance: float,
) -> list[str]:
    best = max(values[name] for name in candidates)
    return [name for name in candidates if values[name] >= best - tolerance]


def choose_criterion(
    point_metrics: Mapping[str, AnchorScoreMetrics],
    bootstrap_results: Mapping[str, BootstrapCriterionResult],
    *,
    eligible_criteria: Sequence[str],
    computational_cost: Mapping[str, float],
    tolerance: float = DEFAULT_ANCHOR_SCORE_TOLERANCE,
) -> CriterionDecision:
    """Choose the point-estimate ACE winner and its one-SE credible set.

    The one-standard-error quantity is the sample standard deviation of the
    winner's bootstrap ACE estimates.  It is deliberately not divided by the
    square root of the number of replicates.
    """

    eligible = tuple(eligible_criteria)
    if not eligible or len(set(eligible)) != len(eligible):
        raise ValueError("eligible_criteria must be nonempty and unique")
    tolerance = float(tolerance)
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    for name in eligible:
        if name not in point_metrics:
            raise ValueError(f"missing point metrics for eligible criterion {name!r}")
        if name not in bootstrap_results:
            raise ValueError(f"missing bootstrap results for criterion {name!r}")
        if name not in computational_cost:
            raise ValueError(f"missing computational cost for criterion {name!r}")
    ace = {name: float(point_metrics[name].ace) for name in eligible}
    adjacent = {
        name: float(point_metrics[name].adjacent_accuracy) for name in eligible
    }
    pair = {name: float(point_metrics[name].pair_accuracy) for name in eligible}
    bootstrap_standard_deviation = {
        name: float(bootstrap_results[name].ace_standard_deviation)
        for name in eligible
    }
    costs = {name: float(computational_cost[name]) for name in eligible}
    for label, values in (
        ("point ACE", ace),
        ("adjacent accuracy", adjacent),
        ("pair accuracy", pair),
        ("bootstrap ACE standard deviation", bootstrap_standard_deviation),
        ("computational cost", costs),
    ):
        if any(not np_is_finite(value) for value in values.values()):
            raise ValueError(f"{label} must be finite for every eligible criterion")

    candidates = _narrow_minimum(list(eligible), ace, tolerance)
    trace = [f"ACE tie set: {candidates}"]
    if len(candidates) > 1:
        candidates = _narrow_maximum(candidates, adjacent, tolerance)
        trace.append(f"after adjacent accuracy: {candidates}")
    if len(candidates) > 1:
        candidates = _narrow_maximum(candidates, pair, tolerance)
        trace.append(f"after pair accuracy: {candidates}")
    if len(candidates) > 1:
        candidates = _narrow_minimum(
            candidates, bootstrap_standard_deviation, tolerance
        )
        trace.append(f"after bootstrap ACE standard deviation: {candidates}")
    if len(candidates) > 1:
        candidates = _narrow_minimum(candidates, costs, tolerance)
        trace.append(f"after computational cost: {candidates}")
    winner = candidates[0]
    if len(candidates) > 1:
        trace.append(f"prespecified eligible order selected {winner!r}")

    best_point_ace = min(ace.values())
    one_standard_error = bootstrap_standard_deviation[winner]
    threshold = best_point_ace + one_standard_error
    credible = tuple(
        name for name in eligible if ace[name] <= threshold + tolerance
    )
    return CriterionDecision(
        eligible_criteria=eligible,
        winner=winner,
        credible_set=credible,
        point_metrics={name: point_metrics[name] for name in eligible},
        bootstrap_ace_standard_deviation=bootstrap_standard_deviation,
        computational_cost=costs,
        best_point_ace=best_point_ace,
        one_standard_error=one_standard_error,
        credible_set_threshold=threshold,
        tolerance=tolerance,
        tie_break_trace=tuple(trace),
    )


def np_is_finite(value: float) -> bool:
    # Avoid importing numpy into this small provenance module.
    return value == value and value not in (float("inf"), float("-inf"))


@dataclass(frozen=True)
class ReceiptPaths:
    receipt: Path
    sha256: Path


def _atomic_create(path: Path, payload: bytes) -> None:
    """Atomically create an immutable file, refusing to replace an existing one."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard link publishes the fully flushed inode and fails atomically if
        # this timestamped receipt already exists.
        os.link(temporary, path)
        temporary.unlink()
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Directory fsync is unavailable on a few filesystems; the file
            # itself is still atomically published and fully fsynced.
            pass
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _receipt_timestamp(value: datetime | None) -> tuple[str, str]:
    timestamp = datetime.now(timezone.utc) if value is None else value
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("receipt timestamp must be timezone-aware")
    timestamp = timestamp.astimezone(timezone.utc)
    embedded = timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z")
    filename = timestamp.strftime("%Y%m%dT%H%M%S.%fZ")
    return embedded, filename


def write_decision_receipt(
    output_directory: str | Path,
    decision: CriterionDecision,
    *,
    formulas: Mapping[str, str],
    anchor_subset_hash: str,
    anchor_family: Mapping[str, Any] | str,
    branch_hashes: Mapping[str, str],
    config_hashes: Mapping[str, str],
    timestamp: datetime | None = None,
    extra_provenance: Mapping[str, Any] | None = None,
) -> ReceiptPaths:
    """Write a timestamped, immutable JSON decision and exact SHA-256 sidecar."""

    missing_formulas = set(decision.eligible_criteria) - set(formulas)
    if missing_formulas:
        raise ValueError(
            f"missing formulas for eligible criteria: {sorted(missing_formulas)}"
        )
    if not anchor_subset_hash or not branch_hashes or not config_hashes:
        raise ValueError("anchor subset, branch, and config hashes are required")
    hashes = {
        "anchor_subset_sha256": anchor_subset_hash,
        **{f"branch_sha256.{name}": value for name, value in branch_hashes.items()},
        **{f"config_sha256.{name}": value for name, value in config_hashes.items()},
    }
    for name, value in hashes.items():
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    created_at, filename_timestamp = _receipt_timestamp(timestamp)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": "anchorcal_criterion_decision",
        "created_at_utc": created_at,
        "decision": decision.to_dict(),
        "formulas": {
            name: formulas[name] for name in decision.eligible_criteria
        },
        "anchor_subset_sha256": anchor_subset_hash,
        "anchor_family": anchor_family,
        "branch_sha256": dict(sorted(branch_hashes.items())),
        "config_sha256": dict(sorted(config_hashes.items())),
        "provenance": dict(extra_provenance or {}),
    }
    serialized = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    directory = Path(output_directory)
    receipt = directory / f"anchorcal_decision_{filename_timestamp}.json"
    sidecar = receipt.with_suffix(receipt.suffix + ".sha256")
    _atomic_create(receipt, serialized)
    digest = hashlib.sha256(serialized).hexdigest()
    sidecar_payload = f"{digest}  {receipt.name}\n".encode("ascii")
    try:
        _atomic_create(sidecar, sidecar_payload)
    except BaseException:
        # A receipt without its integrity sidecar is incomplete and must not be
        # mistaken for a frozen decision.
        receipt.unlink(missing_ok=True)
        raise
    return ReceiptPaths(receipt=receipt, sha256=sidecar)


def verify_decision_receipt(
    receipt_path: str | Path, sha256_path: str | Path | None = None
) -> bool:
    """Verify the exact bytes and filename recorded by a receipt sidecar."""

    receipt = Path(receipt_path)
    sidecar = Path(sha256_path) if sha256_path is not None else receipt.with_suffix(
        receipt.suffix + ".sha256"
    )
    try:
        fields = sidecar.read_text(encoding="ascii").strip().split()
        if len(fields) != 2 or fields[1] != receipt.name:
            return False
        expected = fields[0]
        if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
            return False
        actual = hashlib.sha256(receipt.read_bytes()).hexdigest()
        return actual == expected
    except (FileNotFoundError, OSError, UnicodeError):
        return False
