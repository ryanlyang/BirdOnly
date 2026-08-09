"""Frozen background token-budget selection and deterministic view banks."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .errors import AuditFailure, PreflightError, StorageError
from .io import atomic_write_json, hash_object, sha256_file
from .seeds import background_view_seed


BACKGROUND_TOKEN_BUDGET_SCHEMA = "anchorcal-background-token-budget-v1"
BACKGROUND_TOKEN_BUDGET_SELECTION_RULE = (
    "largest_prespecified_budget_meeting_minimum_overall_and_per_class_"
    "coverage_and_maximum_biased_val_invalid_fraction"
)
BACKGROUND_TOKEN_BUDGET_KEYS = frozenset(
    {
        "schema_version",
        "token_budget",
        "candidates",
        "minimum_coverage",
        "maximum_biased_val_invalid_fraction",
        "coverage",
        "biased_val_invalidity",
        "candidate_passes",
        "selected_biased_val_invalidity",
        "selection_inputs",
        "selection_rule",
    }
)


@dataclass(frozen=True)
class TokenBudgetDecision:
    token_budget: int
    coverage: dict[str, dict[str, float]]
    biased_val_invalidity: dict[str, dict[str, float | int]]
    candidate_passes: dict[str, bool]


def _coverage(counts: np.ndarray, labels: np.ndarray, budget: int) -> dict[str, float]:
    valid = counts >= budget
    result = {"overall": float(valid.mean())}
    for label in sorted(np.unique(labels).tolist()):
        member = labels == label
        result[f"class_{int(label)}"] = float(valid[member].mean()) if member.any() else 0.0
    return result


def _validated_budget_inputs(
    split_counts: Mapping[str, np.ndarray],
    split_labels: Mapping[str, np.ndarray],
    candidates: tuple[int, ...],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    tuple[int, ...],
]:
    """Normalize and validate the scientific inputs to budget selection."""

    required = {"expert_train", "expert_calibration", "biased_val"}
    if set(split_counts) != required or set(split_labels) != required:
        raise ValueError(f"coverage inputs must have exactly {sorted(required)}")
    if not candidates or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
        for value in candidates
    ):
        raise ValueError(
            "background token-budget candidates must be positive integers"
        )
    normalized_candidates = tuple(int(value) for value in candidates)
    if len(set(normalized_candidates)) != len(normalized_candidates):
        raise ValueError("background token-budget candidates must be unique")
    if normalized_candidates != tuple(
        sorted(normalized_candidates, reverse=True)
    ):
        raise ValueError(
            "background token-budget candidates must be strictly descending"
        )

    counts_by_split: dict[str, np.ndarray] = {}
    labels_by_split: dict[str, np.ndarray] = {}
    for split in sorted(required):
        counts = np.asarray(split_counts[split])
        labels = np.asarray(split_labels[split])
        if (
            counts.ndim != 1
            or labels.ndim != 1
            or counts.size == 0
            or counts.shape != labels.shape
            or not np.issubdtype(counts.dtype, np.integer)
            or not np.issubdtype(labels.dtype, np.integer)
            or np.any(counts < 0)
            or set(np.unique(labels).tolist()) != {0, 1}
        ):
            raise ValueError(
                f"{split} budget inputs require aligned nonempty 1D "
                "nonnegative integer counts and both binary classes"
            )
        counts_by_split[split] = counts.astype(np.int64, copy=False)
        labels_by_split[split] = labels.astype(np.int64, copy=False)
    return counts_by_split, labels_by_split, normalized_candidates


def select_token_budget(
    split_counts: Mapping[str, np.ndarray],
    split_labels: Mapping[str, np.ndarray],
    candidates: tuple[int, ...] = (64, 48, 32),
    minimum_coverage: float = 0.95,
    maximum_biased_val_invalid_fraction: float = 0.01,
) -> TokenBudgetDecision:
    counts_by_split, labels_by_split, candidates = _validated_budget_inputs(
        split_counts,
        split_labels,
        candidates,
    )
    required = set(counts_by_split)
    if (
        isinstance(minimum_coverage, (bool, np.bool_))
        or not isinstance(minimum_coverage, (int, float, np.integer, np.floating))
        or not np.isfinite(float(minimum_coverage))
    ):
        raise ValueError("minimum background coverage must be numeric")
    if not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("minimum background coverage must lie in [0, 1]")
    if (
        isinstance(maximum_biased_val_invalid_fraction, (bool, np.bool_))
        or not isinstance(
            maximum_biased_val_invalid_fraction,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(float(maximum_biased_val_invalid_fraction))
    ):
        raise ValueError("maximum biased-val invalid fraction must be numeric")
    if not 0.0 <= maximum_biased_val_invalid_fraction <= 1.0:
        raise ValueError(
            "maximum biased-val invalid fraction must lie in [0, 1]"
        )
    all_coverage: dict[str, dict[str, float]] = {}
    invalidity: dict[str, dict[str, float | int]] = {}
    candidate_passes: dict[str, bool] = {}
    for budget in candidates:
        coverage = {
            name: _coverage(
                counts_by_split[name],
                labels_by_split[name],
                budget,
            )
            for name in sorted(required)
        }
        key = str(int(budget))
        all_coverage[key] = {
            f"{split}.{metric}": value
            for split, values in coverage.items()
            for metric, value in values.items()
        }
        biased_counts = counts_by_split["biased_val"]
        invalid_count = int(np.sum(biased_counts < int(budget)))
        biased_total = int(len(biased_counts))
        if biased_total <= 0:
            raise ValueError("biased_val token-budget input must be nonempty")
        invalid_fraction = float(invalid_count / biased_total)
        invalidity[key] = {
            "count": invalid_count,
            "total": biased_total,
            "fraction": invalid_fraction,
        }
        coverage_passes = all(
            value >= minimum_coverage
            for split_values in coverage.values()
            for value in split_values.values()
        )
        candidate_passes[key] = bool(
            coverage_passes
            and invalid_fraction <= maximum_biased_val_invalid_fraction
        )
    for budget in candidates:
        key = str(int(budget))
        if candidate_passes[key]:
            return TokenBudgetDecision(
                int(budget),
                all_coverage,
                invalidity,
                candidate_passes,
            )
    raise AuditFailure(
        f"no background token budget in {candidates} meets "
        f"{minimum_coverage:.0%} overall/per-class coverage and at most "
        f"{maximum_biased_val_invalid_fraction:.0%} biased_val invalidity: "
        f"coverage={all_coverage}, biased_val_invalidity={invalidity}"
    )


def require_background_validity(
    valid: np.ndarray, *, maximum_invalid_fraction: float
) -> dict[str, float | int | str]:
    """Apply the locked overall biased-validation invalidity backstop."""

    values = np.asarray(valid, dtype=bool)
    if values.ndim != 1 or values.size <= 0:
        raise ValueError("background validity gate requires a nonempty 1D array")
    if not 0.0 <= maximum_invalid_fraction <= 1.0:
        raise ValueError("maximum invalid fraction must lie in [0, 1]")
    invalid_count = int(np.sum(~values))
    total = int(values.size)
    invalid_fraction = float(invalid_count / total)
    payload: dict[str, float | int | str] = {
        "schema_version": "anchorcal-background-validity-gate-v1",
        "status": (
            "passed"
            if invalid_fraction <= maximum_invalid_fraction
            else "failed"
        ),
        "invalid_count": invalid_count,
        "total": total,
        "invalid_fraction": invalid_fraction,
        "maximum_invalid_fraction": float(maximum_invalid_fraction),
        "scope": "overall_biased_val",
    }
    if payload["status"] != "passed":
        raise AuditFailure(
            "background biased_val invalidity gate failed: "
            f"{invalid_count}/{total}={invalid_fraction:.8f} exceeds "
            f"{maximum_invalid_fraction:.8f}"
        )
    return payload


def token_budget_manifest(
    decision: TokenBudgetDecision, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Serialize the complete combined-gate decision for provenance."""

    selected_key = str(decision.token_budget)
    payload = {
        "schema_version": BACKGROUND_TOKEN_BUDGET_SCHEMA,
        "token_budget": decision.token_budget,
        "candidates": list(
            config["branches"]["background_token_budget_candidates"]
        ),
        "minimum_coverage": float(config["branches"]["background_min_coverage"]),
        "maximum_biased_val_invalid_fraction": float(
            config["branches"]["background_max_biased_val_invalid_fraction"]
        ),
        "coverage": decision.coverage,
        "biased_val_invalidity": decision.biased_val_invalidity,
        "candidate_passes": decision.candidate_passes,
        "selected_biased_val_invalidity": decision.biased_val_invalidity[
            selected_key
        ],
        "selection_inputs": [
            "expert_train",
            "expert_calibration",
            "biased_val",
        ],
        "selection_rule": BACKGROUND_TOKEN_BUDGET_SELECTION_RULE,
    }
    validate_token_budget_manifest(payload, config)
    return payload


def validate_token_budget_manifest(
    payload: Any, config: Mapping[str, Any]
) -> int:
    """Validate the complete, config-bound background budget decision."""

    if not isinstance(payload, dict) or set(payload) != BACKGROUND_TOKEN_BUDGET_KEYS:
        raise PreflightError("background token-budget manifest schema is incompatible")
    candidates = [
        int(value)
        for value in config["branches"]["background_token_budget_candidates"]
    ]
    minimum = float(config["branches"]["background_min_coverage"])
    maximum = float(
        config["branches"]["background_max_biased_val_invalid_fraction"]
    )
    payload_candidates = payload.get("candidates")
    payload_minimum = payload.get("minimum_coverage")
    payload_maximum = payload.get(
        "maximum_biased_val_invalid_fraction"
    )
    if (
        payload.get("schema_version") != BACKGROUND_TOKEN_BUDGET_SCHEMA
        or not isinstance(payload_candidates, list)
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in payload_candidates
        )
        or payload_candidates != candidates
        or not isinstance(payload_minimum, (int, float))
        or isinstance(payload_minimum, bool)
        or not np.isfinite(float(payload_minimum))
        or float(payload_minimum) != minimum
        or not isinstance(payload_maximum, (int, float))
        or isinstance(payload_maximum, bool)
        or not np.isfinite(float(payload_maximum))
        or float(payload_maximum) != maximum
        or payload.get("selection_inputs")
        != ["expert_train", "expert_calibration", "biased_val"]
        or payload.get("selection_rule") != BACKGROUND_TOKEN_BUDGET_SELECTION_RULE
    ):
        raise PreflightError("background token-budget manifest contract is incompatible")
    coverage = payload.get("coverage")
    invalidity = payload.get("biased_val_invalidity")
    recorded_passes = payload.get("candidate_passes")
    expected_keys = {str(value) for value in candidates}
    if (
        not isinstance(coverage, dict)
        or set(coverage) != expected_keys
        or not isinstance(invalidity, dict)
        or set(invalidity) != expected_keys
        or not isinstance(recorded_passes, dict)
        or set(recorded_passes) != expected_keys
        or any(type(value) is not bool for value in recorded_passes.values())
    ):
        raise PreflightError("background token-budget candidate records are incomplete")
    expected_metrics = {
        f"{split}.{metric}"
        for split in ("biased_val", "expert_calibration", "expert_train")
        for metric in ("overall", "class_0", "class_1")
    }
    computed_passes: dict[str, bool] = {}
    for budget in candidates:
        key = str(budget)
        metrics = coverage[key]
        record = invalidity[key]
        if (
            not isinstance(metrics, dict)
            or set(metrics) != expected_metrics
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
                for value in metrics.values()
            )
            or not isinstance(record, dict)
            or set(record) != {"count", "total", "fraction"}
            or not isinstance(record.get("count"), int)
            or isinstance(record.get("count"), bool)
            or not isinstance(record.get("total"), int)
            or isinstance(record.get("total"), bool)
            or int(record["total"]) <= 0
            or not 0 <= int(record["count"]) <= int(record["total"])
            or not isinstance(record.get("fraction"), (int, float))
            or isinstance(record.get("fraction"), bool)
            or not np.isfinite(float(record["fraction"]))
            or not np.isclose(
                float(record["fraction"]),
                int(record["count"]) / int(record["total"]),
                rtol=0.0,
                atol=1.0e-15,
            )
            or not np.isclose(
                float(metrics["biased_val.overall"]),
                1.0 - float(record["fraction"]),
                rtol=0.0,
                atol=1.0e-15,
            )
        ):
            raise PreflightError(
                f"background token-budget diagnostics are malformed for K={budget}"
            )
        computed_passes[key] = bool(
            all(float(value) >= minimum for value in metrics.values())
            and float(record["fraction"]) <= maximum
        )
    invalid_totals = {
        int(invalidity[str(budget)]["total"]) for budget in candidates
    }
    invalid_counts = [
        int(invalidity[str(budget)]["count"]) for budget in candidates
    ]
    if len(invalid_totals) != 1 or any(
        earlier < later
        for earlier, later in zip(invalid_counts, invalid_counts[1:])
    ):
        raise PreflightError(
            "background token-budget invalidity records are not physically monotonic"
        )
    for metric in expected_metrics:
        values = [float(coverage[str(budget)][metric]) for budget in candidates]
        if any(
            earlier > later
            for earlier, later in zip(values, values[1:])
        ):
            raise PreflightError(
                "background token-budget coverage records are not physically monotonic"
            )
    if recorded_passes != computed_passes:
        raise PreflightError("background token-budget gate outcomes are invalid")
    selected = next(
        (budget for budget in candidates if computed_passes[str(budget)]),
        None,
    )
    if selected is None:
        raise PreflightError("background token-budget manifest has no passing candidate")
    if (
        not isinstance(payload.get("token_budget"), int)
        or isinstance(payload.get("token_budget"), bool)
        or payload.get("token_budget") != selected
        or payload.get("selected_biased_val_invalidity")
        != invalidity[str(selected)]
    ):
        raise PreflightError("background token-budget selection is not the largest valid K")
    return int(selected)


def _geometry_budget_inputs(
    artifact_root: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Read the frozen geometry rows used to make the budget decision."""

    counts_by_split: dict[str, np.ndarray] = {}
    labels_by_split: dict[str, np.ndarray] = {}
    required_columns = {"img_id", "y", "safe_background_count"}
    for split in ("expert_train", "expert_calibration", "biased_val"):
        path = artifact_root / f"{split}_geometry.csv"
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None or not required_columns.issubset(
                    reader.fieldnames
                ):
                    raise PreflightError(
                        f"background geometry columns are incomplete: {path}"
                    )
                rows = list(reader)
        except OSError as error:
            raise PreflightError(
                f"background geometry is unavailable: {path}"
            ) from error
        if not rows:
            raise PreflightError(f"background geometry is empty: {path}")
        try:
            ids = [int(row["img_id"]) for row in rows]
            labels = np.asarray([int(row["y"]) for row in rows], dtype=np.int64)
            counts = np.asarray(
                [int(row["safe_background_count"]) for row in rows],
                dtype=np.int64,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise PreflightError(
                f"background geometry values are malformed: {path}"
            ) from error
        if len(set(ids)) != len(ids):
            raise PreflightError(
                f"background geometry contains duplicate img_id values: {path}"
            )
        counts_by_split[split] = counts
        labels_by_split[split] = labels
    return counts_by_split, labels_by_split


def load_token_budget_manifest(
    path: str | Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Load and strictly validate a frozen token-budget decision."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError(
            f"invalid background token-budget manifest: {source}"
        ) from error
    validate_token_budget_manifest(payload, config)
    counts, labels = _geometry_budget_inputs(source.parent)
    try:
        recomputed = select_token_budget(
            counts,
            labels,
            candidates=tuple(
                config["branches"]["background_token_budget_candidates"]
            ),
            minimum_coverage=float(
                config["branches"]["background_min_coverage"]
            ),
            maximum_biased_val_invalid_fraction=float(
                config["branches"][
                    "background_max_biased_val_invalid_fraction"
                ]
            ),
        )
    except (AuditFailure, ValueError) as error:
        raise PreflightError(
            "background token-budget geometry cannot reproduce a valid decision"
        ) from error
    if payload != token_budget_manifest(recomputed, config):
        raise PreflightError(
            "background token-budget manifest does not match frozen geometry"
        )
    return payload


def sample_background_indices(
    eligible_indices: np.ndarray,
    token_budget: int,
    *,
    global_seed: int,
    sample_id: int,
    view_index: int,
    purpose: str,
) -> tuple[np.ndarray, int]:
    eligible = np.asarray(eligible_indices, dtype=np.int64)
    if len(np.unique(eligible)) != len(eligible):
        raise ValueError("eligible background indices contain duplicates")
    if len(eligible) < token_budget:
        raise AuditFailure(
            f"img_id={sample_id} has {len(eligible)} pure patches, needs {token_budget}"
        )
    seed = background_view_seed(global_seed, sample_id, view_index, purpose)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=token_budget, replace=False)
    # choice already randomizes order; assert the no-replacement contract.
    if len(np.unique(chosen)) != token_budget:
        raise AssertionError("background sampling unexpectedly used replacement")
    return chosen.astype(np.int16), seed


def build_view_arrays(
    eligible_by_id: Mapping[int, np.ndarray],
    *,
    token_budget: int,
    views: int,
    global_seed: int,
    purpose: str = "background_branch_eval",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    ids = np.asarray(sorted(eligible_by_id), dtype=np.int64)
    indices = np.full((len(ids), views, token_budget), -1, dtype=np.int16)
    seeds = np.zeros((len(ids), views), dtype=np.uint64)
    invalid: list[int] = []
    for row, img_id in enumerate(ids.tolist()):
        eligible = eligible_by_id[img_id]
        if len(eligible) < token_budget:
            invalid.append(img_id)
            continue
        for view in range(views):
            chosen, seed = sample_background_indices(
                eligible,
                token_budget,
                global_seed=global_seed,
                sample_id=img_id,
                view_index=view,
                purpose=purpose,
            )
            indices[row, view] = chosen
            seeds[row, view] = seed
    return ids, indices, seeds, invalid


def invalid_background_records(
    eligible_by_id: Mapping[int, np.ndarray],
    invalid_ids: list[int],
    token_budget: int,
) -> list[dict[str, Any]]:
    """Describe every invalid example instead of persisting IDs alone."""

    records = []
    for img_id in invalid_ids:
        eligible_count = int(len(np.asarray(eligible_by_id[int(img_id)])))
        if eligible_count >= token_budget:
            raise ValueError("invalid background ID unexpectedly meets token budget")
        records.append(
            {
                "img_id": int(img_id),
                "reason": "insufficient_pure_background_patches",
                "eligible_patch_count": eligible_count,
                "required_token_budget": int(token_budget),
                "shortfall": int(token_budget - eligible_count),
            }
        )
    return records


def persist_view_bank(
    path: str | Path,
    *,
    img_ids: np.ndarray,
    indices: np.ndarray,
    seeds: np.ndarray,
    invalid_ids: list[int],
    invalid_records: list[dict[str, Any]],
    token_budget: int,
    mask_dilation_hash: str,
    purpose: str,
) -> dict[str, object]:
    try:
        import h5py
    except ImportError as error:
        raise StorageError("h5py is required for the background view bank") from error
    target = Path(path)
    if [int(record.get("img_id", -1)) for record in invalid_records] != [
        int(value) for value in invalid_ids
    ] or any(
        record.get("reason") != "insufficient_pure_background_patches"
        for record in invalid_records
    ):
        raise ValueError("invalid background records must exactly explain invalid_ids")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    with h5py.File(partial, "w") as handle:
        handle.attrs["schema_version"] = "anchorcal-background-views-v1"
        handle.attrs["token_budget"] = token_budget
        handle.attrs["mask_dilation_hash"] = mask_dilation_hash
        handle.attrs["purpose"] = purpose
        handle.attrs[
            "invalid_reason_code"
        ] = "insufficient_pure_background_patches"
        handle.create_dataset("img_id", data=img_ids)
        handle.create_dataset("source_patch_indices", data=indices, compression="gzip")
        handle.create_dataset("view_seed", data=seeds, compression="gzip")
        handle.create_dataset("invalid_img_id", data=np.asarray(invalid_ids, dtype=np.int64))
        handle.create_dataset(
            "invalid_eligible_patch_count",
            data=np.asarray(
                [record["eligible_patch_count"] for record in invalid_records],
                dtype=np.int16,
            ),
        )
        handle.flush()
    partial.replace(target)
    manifest = {
        "schema_version": "anchorcal-background-views-v1",
        "path": str(target.resolve()),
        "sha256": sha256_file(target),
        "count": int(len(img_ids)),
        "valid_count": int(np.sum(np.all(indices >= 0, axis=(1, 2)))),
        "invalid_ids": [int(value) for value in invalid_ids],
        "invalid_examples": invalid_records,
        "token_budget": int(token_budget),
        "views": int(indices.shape[1]),
        "mask_dilation_hash": mask_dilation_hash,
        "content_hash": hash_object(
            {
                "ids": img_ids.tolist(),
                "indices": indices.tolist(),
                "seeds": [list(map(int, row)) for row in seeds],
            }
        ),
    }
    atomic_write_json(target.with_suffix(target.suffix + ".manifest.json"), manifest)
    return manifest


def load_view_bank(path: str | Path) -> dict[str, np.ndarray | int | str]:
    import h5py

    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema_version") != "anchorcal-background-views-v1":
            raise StorageError("unsupported background view-bank schema")
        return {
            "img_id": handle["img_id"][:],
            "source_patch_indices": handle["source_patch_indices"][:],
            "view_seed": handle["view_seed"][:],
            "invalid_img_id": handle["invalid_img_id"][:],
            "invalid_eligible_patch_count": handle[
                "invalid_eligible_patch_count"
            ][:],
            "token_budget": int(handle.attrs["token_budget"]),
            "mask_dilation_hash": str(handle.attrs["mask_dilation_hash"]),
            "invalid_reason_code": str(handle.attrs["invalid_reason_code"]),
        }
