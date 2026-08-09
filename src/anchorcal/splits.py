"""Deterministic Waterbirds100 and branch-calibration split materialization."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from .errors import PreflightError
from .io import sha256_bytes, sha256_file


SPLIT_COLUMNS = (
    "img_id",
    "img_filename",
    "y",
    "place",
    "split",
    "split_seed",
    "source_metadata_sha256",
)


def _split_stratified(
    frame: pd.DataFrame, test_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    ordered = frame.sort_values("img_id", kind="stable").reset_index(drop=True)
    left, right = train_test_split(
        ordered,
        test_size=test_fraction,
        random_state=seed,
        shuffle=True,
        stratify=ordered["y"],
    )
    return (
        left.sort_values("img_id", kind="stable").reset_index(drop=True),
        right.sort_values("img_id", kind="stable").reset_index(drop=True),
    )


def construct_splits(
    metadata: pd.DataFrame,
    metadata_sha256: str,
    *,
    split_seed: int = 1729,
    calibration_seed: int = 2718,
) -> dict[str, pd.DataFrame]:
    official_train = metadata.loc[metadata["split"] == 0].copy()
    aligned = official_train.loc[official_train["y"] == official_train["place"]].copy()
    if aligned.empty:
        raise PreflightError("Waterbirds100 aligned official-training pool is empty")
    candidate_train, biased_val = _split_stratified(aligned, 0.20, split_seed)
    expert_train, expert_calibration = _split_stratified(
        candidate_train, 0.10, calibration_seed
    )
    result = {
        "candidate_train": candidate_train,
        "biased_val": biased_val,
        "expert_train": expert_train,
        "expert_calibration": expert_calibration,
        "oracle_val": metadata.loc[metadata["split"] == 1].copy(),
        "test": metadata.loc[metadata["split"] == 2].copy(),
    }
    for name, frame in result.items():
        frame = frame.sort_values("img_id", kind="stable").reset_index(drop=True)
        frame["split_seed"] = (
            calibration_seed
            if name.startswith("expert_")
            else (split_seed if name in {"candidate_train", "biased_val"} else -1)
        )
        frame["source_metadata_sha256"] = metadata_sha256
        result[name] = frame.loc[:, SPLIT_COLUMNS]
    _assert_split_contract(result)
    return result


def _assert_split_contract(splits: dict[str, pd.DataFrame]) -> dict[str, object]:
    ids = {name: set(frame["img_id"].astype(int)) for name, frame in splits.items()}
    candidate_biased_overlap = ids["candidate_train"] & ids["biased_val"]
    expert_overlap = ids["expert_train"] & ids["expert_calibration"]
    expert_union = ids["expert_train"] | ids["expert_calibration"]
    top_level_train = ids["candidate_train"] | ids["biased_val"]
    oracle_test_overlap = ids["oracle_val"] & ids["test"]
    train_hidden_overlap = top_level_train & (ids["oracle_val"] | ids["test"])
    if candidate_biased_overlap:
        raise AssertionError("candidate_train and biased_val overlap")
    if expert_overlap:
        raise AssertionError("expert train/calibration overlap")
    if expert_union != ids["candidate_train"]:
        raise AssertionError("expert splits do not partition candidate_train")
    if oracle_test_overlap:
        raise AssertionError("official oracle_val and test splits overlap")
    if train_hidden_overlap:
        raise AssertionError("derived official-train and hidden official splits overlap")
    aligned_passed: dict[str, bool] = {}
    for name, frame in splits.items():
        aligned = bool((frame["y"] == frame["place"]).all())
        if name not in {"oracle_val", "test"} and not aligned:
            raise AssertionError(f"{name} contains a shortcut-misaligned example")
        if name not in {"oracle_val", "test"}:
            aligned_passed[name] = aligned
    return {
        "all_passed": True,
        "candidate_train_biased_val_disjoint": {
            "passed": True,
            "overlap_count": len(candidate_biased_overlap),
        },
        "expert_train_expert_calibration_disjoint": {
            "passed": True,
            "overlap_count": len(expert_overlap),
        },
        "expert_union_equals_candidate_train": {
            "passed": True,
            "expert_union_count": len(expert_union),
            "candidate_train_count": len(ids["candidate_train"]),
        },
        "oracle_val_test_disjoint": {
            "passed": True,
            "overlap_count": len(oracle_test_overlap),
        },
        "derived_train_hidden_disjoint": {
            "passed": True,
            "overlap_count": len(train_hidden_overlap),
        },
        "waterbirds100_alignment": {
            "passed": all(aligned_passed.values()),
            "splits": aligned_passed,
        },
    }


def _describe(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(frame)),
        "class_counts": {str(k): int(v) for k, v in frame["y"].value_counts().sort_index().items()},
        "place_counts": {str(k): int(v) for k, v in frame["place"].value_counts().sort_index().items()},
        "aligned_group_counts": {
            f"{int(y)}_{int(place)}": int(count)
            for (y, place), count in frame.groupby(["y", "place"]).size().items()
        },
        "empirical_correlation": float((frame["y"] == frame["place"]).mean()),
    }


def _atomic_create_or_verify(path: Path, expected: bytes) -> None:
    """Publish ``expected`` once, or verify an identical prior publication.

    A temporary file plus an atomic hard-link avoids the check-then-replace
    race inherent in ``os.replace``: an already published split is never
    overwritten, even when two deterministic preflights race.  Crash debris is
    confined to hidden temporary files and cannot be mistaken for a split.
    """

    if path.is_file():
        actual = path.read_bytes()
        if actual != expected:
            raise PreflightError(
                f"refusing to overwrite nonidentical persisted split artifact {path}: "
                f"expected_sha256={sha256_bytes(expected)}, "
                f"actual_sha256={sha256_bytes(actual)}"
            )
        return
    if path.exists():
        raise PreflightError(f"persisted split artifact is not a regular file: {path}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            # A concurrent preflight won publication.  It is acceptable only
            # when its bytes are the same deterministic replay.
            if not path.is_file() or path.read_bytes() != expected:
                actual_hash = sha256_file(path) if path.is_file() else "not-a-file"
                raise PreflightError(
                    f"concurrent nonidentical split publication at {path}: "
                    f"expected_sha256={sha256_bytes(expected)}, "
                    f"actual_sha256={actual_hash}"
                )
        # Persist the directory entry before returning success.
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def persist_splits(
    splits: dict[str, pd.DataFrame], output_dir: str | Path
) -> dict[str, object]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    payloads: dict[str, bytes] = {}
    for name, frame in splits.items():
        path = root / f"waterbirds100_{name}.csv"
        payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
        _atomic_create_or_verify(path, payload)
        paths[name] = path
        payloads[name] = payload
    contract = _assert_split_contract(splits)
    manifest: dict[str, object] = {
        "schema_version": "anchorcal-splits-v2",
        "contract_assertions": contract,
        "splits": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_bytes(payloads[name]),
                **_describe(splits[name]),
            }
            for name, path in paths.items()
        },
    }
    manifest_payload = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_create_or_verify(root / "manifest.json", manifest_payload)
    return manifest
