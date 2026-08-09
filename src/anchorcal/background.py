"""Frozen background token-budget selection and deterministic view banks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .errors import AuditFailure, StorageError
from .io import atomic_write_json, hash_object, sha256_file
from .seeds import background_view_seed


@dataclass(frozen=True)
class TokenBudgetDecision:
    token_budget: int
    coverage: dict[str, dict[str, float]]


def _coverage(counts: np.ndarray, labels: np.ndarray, budget: int) -> dict[str, float]:
    valid = counts >= budget
    result = {"overall": float(valid.mean())}
    for label in sorted(np.unique(labels).tolist()):
        member = labels == label
        result[f"class_{int(label)}"] = float(valid[member].mean()) if member.any() else 0.0
    return result


def select_token_budget(
    split_counts: Mapping[str, np.ndarray],
    split_labels: Mapping[str, np.ndarray],
    candidates: tuple[int, ...] = (64, 48, 32),
    minimum_coverage: float = 0.95,
) -> TokenBudgetDecision:
    required = {"expert_train", "expert_calibration", "biased_val"}
    if set(split_counts) != required or set(split_labels) != required:
        raise ValueError(f"coverage inputs must have exactly {sorted(required)}")
    all_coverage: dict[str, dict[str, float]] = {}
    for budget in candidates:
        coverage = {
            name: _coverage(np.asarray(split_counts[name]), np.asarray(split_labels[name]), budget)
            for name in sorted(required)
        }
        all_coverage[str(budget)] = {
            f"{split}.{key}": value
            for split, values in coverage.items()
            for key, value in values.items()
        }
        if all(
            value >= minimum_coverage
            for split_values in coverage.values()
            for value in split_values.values()
        ):
            return TokenBudgetDecision(int(budget), all_coverage)
    raise AuditFailure(
        f"no background token budget in {candidates} meets {minimum_coverage:.0%} "
        f"overall and per-class coverage: {all_coverage}"
    )


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
