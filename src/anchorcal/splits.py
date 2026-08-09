"""Frozen FCV development membership and protected Waterbirds100 splits.

The top-level 80/20 partition is imported from the completed FCV seed-0
study.  AnchorCal never invokes a splitter for that partition.  Protected
context labels are retained only in the physically separate analysis-only
namespace when the resulting frames are persisted.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import validated_waterbirds100_official_splits
from .errors import PreflightError
from .io import sha256_bytes, sha256_file


WATERBIRDS100_RELEASE = "waterbird_1.0_forest2water2"
VISIBLE_SPLIT_SCHEMA = "anchorcal-splits-v4"
ANALYSIS_ONLY_SPLIT_SCHEMA = "anchorcal-analysis-only-splits-v1"

VISIBLE_SPLIT_NAMES = (
    "candidate_train",
    "biased_val",
    "expert_train",
    "expert_calibration",
)
ANALYSIS_ONLY_SPLIT_NAMES = ("oracle_val", "test")

VISIBLE_SPLIT_COLUMNS = (
    "img_id",
    "img_filename",
    "y",
    "split",
    "membership_source",
    "membership_seed",
    "source_metadata_sha256",
    "source_membership_sha256",
)
ANALYSIS_ONLY_SPLIT_COLUMNS = (
    "img_id",
    "metadata_index",
    "img_filename",
    "y",
    "place",
    "group",
    "split",
    "source_metadata_sha256",
)
PROTECTED_COLUMNS = frozenset({"place", "group", "group_name"})


@dataclass(frozen=True)
class FcvMembership:
    candidate_train_metadata_indices: tuple[int, ...]
    biased_validation_metadata_indices: tuple[int, ...]
    provenance: dict[str, Any]


def _require_sha256(value: Any, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PreflightError(f"{description} is not a lowercase SHA-256 digest")
    return value


def _metadata_indices_sha256(values: Any) -> str:
    ordered = [int(value) for value in values]
    return sha256_bytes(",".join(str(value) for value in ordered).encode("ascii"))


def _load_json_mapping(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise PreflightError(f"required {description} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError(f"invalid {description}: {path}") from error
    if not isinstance(payload, dict):
        raise PreflightError(f"{description} must be a JSON mapping: {path}")
    return payload


def _integer_index_list(value: Any, description: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise PreflightError(f"{description} must be a nonempty JSON list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise PreflightError(f"{description} must contain only integer indices")
    result = tuple(int(item) for item in value)
    if any(item < 0 for item in result):
        raise PreflightError(f"{description} contains a negative metadata index")
    if len(set(result)) != len(result):
        raise PreflightError(f"{description} contains duplicate metadata indices")
    if tuple(sorted(result)) != result:
        raise PreflightError(f"{description} must be canonically sorted")
    return result


def load_fcv_membership(
    artifact_root: str | Path,
    reference: dict[str, Any],
    *,
    source_metadata_sha256: str,
) -> FcvMembership:
    """Load and hash-verify the prior FCV membership without regenerating it."""

    root = Path(artifact_root).expanduser().resolve()
    if not root.is_dir():
        raise PreflightError(f"FCV split-manifest root is missing: {root}")

    required_reference = {
        "study_id",
        "protocol_version",
        "source_metadata_sha256",
        "source_train_count",
        "candidate_train_count",
        "biased_val_count",
        "manifest_bundle_sha256",
        "split_indices_sha256",
        "candidate_train_csv_sha256",
        "biased_val_csv_sha256",
        "candidate_train_metadata_indices_sha256",
        "biased_val_metadata_indices_sha256",
    }
    missing = sorted(required_reference - set(reference))
    if missing:
        raise PreflightError(f"FCV reference is missing locked fields: {missing}")

    expected_metadata_hash = _require_sha256(
        reference["source_metadata_sha256"], "FCV source metadata hash"
    )
    if source_metadata_sha256 != expected_metadata_hash:
        raise PreflightError(
            "Waterbirds100 metadata does not match the frozen FCV source: "
            f"expected={expected_metadata_hash}, actual={source_metadata_sha256}"
        )

    paths = {
        "manifest_bundle": root / "manifest_bundle.json",
        "split_indices": root / "split_indices.json",
        "candidate_train_csv": root / "metadata_train.csv",
        "biased_val_csv": root / "metadata_val.csv",
    }
    for key, path in paths.items():
        if not path.is_file():
            raise PreflightError(f"frozen FCV artifact is missing: {path}")
        expected = _require_sha256(reference[f"{key}_sha256"], f"FCV {key} hash")
        actual = sha256_file(path)
        if actual != expected:
            raise PreflightError(
                f"frozen FCV artifact hash mismatch for {path.name}: "
                f"expected={expected}, actual={actual}"
            )

    bundle = _load_json_mapping(paths["manifest_bundle"], "FCV manifest bundle")
    indices = _load_json_mapping(paths["split_indices"], "FCV split indices")
    locked_bundle_values = {
        "artifact_type": "fcv_vit_waterbirds100_manifest_bundle",
        "status": "complete",
        "study_id": str(reference["study_id"]),
        "protocol_version": str(reference["protocol_version"]),
        "original_metadata_sha256": expected_metadata_hash,
        "split_indices_sha256": str(reference["split_indices_sha256"]),
    }
    for key, expected in locked_bundle_values.items():
        if bundle.get(key) != expected:
            raise PreflightError(
                f"FCV manifest bundle field {key!r} differs from the lock: "
                f"expected={expected!r}, actual={bundle.get(key)!r}"
            )
    holdout = bundle.get("holdout")
    if not isinstance(holdout, dict) or any(
        holdout.get(key) != expected
        for key, expected in {
            "split_seed": 0,
            "stratify_by": "y",
            "train_fraction": 0.8,
            "validation_fraction": 0.2,
            "source_split": "train",
            "require_complete_shortcut_correlation": True,
            "reuse_identical_indices_for_all_candidates": True,
        }.items()
    ):
        raise PreflightError("FCV manifest holdout contract differs from seed-0 80/20")

    locked_index_values = {
        "metadata_sha256": expected_metadata_hash,
        "split_seed": 0,
        "stratify_by": "y",
        "train_fraction": 0.8,
        "validation_fraction": 0.2,
    }
    for key, expected in locked_index_values.items():
        if indices.get(key) != expected:
            raise PreflightError(
                f"FCV split-indices field {key!r} differs from the lock"
            )
    candidate = _integer_index_list(
        indices.get("candidate_train_metadata_indices"),
        "candidate-train FCV metadata indices",
    )
    biased = _integer_index_list(
        indices.get("biased_validation_metadata_indices"),
        "biased-validation FCV metadata indices",
    )
    expected_counts = {
        "candidate_train": int(reference["candidate_train_count"]),
        "biased_validation": int(reference["biased_val_count"]),
    }
    if len(candidate) != expected_counts["candidate_train"] or len(biased) != expected_counts[
        "biased_validation"
    ]:
        raise PreflightError(
            "FCV membership row counts differ from the lock: "
            f"candidate={len(candidate)}, biased_val={len(biased)}"
        )
    if set(candidate) & set(biased):
        raise PreflightError("FCV candidate-train and biased-validation memberships overlap")
    if len(candidate) + len(biased) != int(reference["source_train_count"]):
        raise PreflightError("FCV memberships do not have the locked source-train count")

    expected_candidate_hash = _require_sha256(
        reference["candidate_train_metadata_indices_sha256"],
        "candidate-train FCV membership hash",
    )
    expected_biased_hash = _require_sha256(
        reference["biased_val_metadata_indices_sha256"],
        "biased-validation FCV membership hash",
    )
    actual_candidate_hash = _metadata_indices_sha256(candidate)
    actual_biased_hash = _metadata_indices_sha256(biased)
    if (
        indices.get("candidate_train_indices_sha256") != expected_candidate_hash
        or actual_candidate_hash != expected_candidate_hash
        or indices.get("biased_validation_indices_sha256") != expected_biased_hash
        or actual_biased_hash != expected_biased_hash
    ):
        raise PreflightError("FCV metadata-index membership hash verification failed")

    # The canonical CSVs are independently hash-bound above.  Recheck that
    # their metadata-index columns encode the same imported lists so a wrong
    # file cannot be paired with an otherwise valid JSON list.
    for name, path, expected_indices in (
        ("candidate_train", paths["candidate_train_csv"], candidate),
        ("biased_val", paths["biased_val_csv"], biased),
    ):
        try:
            csv = pd.read_csv(path, usecols=["metadata_index"])
        except (OSError, ValueError) as error:
            raise PreflightError(f"invalid frozen FCV {name} CSV: {path}") from error
        observed = tuple(sorted(csv["metadata_index"].astype(int).tolist()))
        if observed != expected_indices:
            raise PreflightError(
                f"frozen FCV {name} CSV membership differs from split_indices.json"
            )

    provenance = {
        "study_id": str(reference["study_id"]),
        "protocol_version": str(reference["protocol_version"]),
        "development_split_seed": 0,
        "reuse_frozen_membership": True,
        "artifact_root": str(root),
        "source_metadata_sha256": expected_metadata_hash,
        "source_train_count": int(reference["source_train_count"]),
        "files": {
            key: {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for key, path in paths.items()
        },
        "candidate_train": {
            "rows": len(candidate),
            "metadata_indices_sha256": actual_candidate_hash,
        },
        "biased_val": {
            "rows": len(biased),
            "metadata_indices_sha256": actual_biased_hash,
        },
        "metadata_index_hash_encoding": "sha256-comma-separated-sorted-ascii-v1",
    }
    return FcvMembership(candidate, biased, provenance)


def _split_stratified(
    frame: pd.DataFrame, test_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create only the nested expert-calibration split (never the FCV split)."""

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
    fcv_split_manifest_root: str | Path,
    fcv_reference: dict[str, Any],
    calibration_seed: int = 2718,
) -> dict[str, pd.DataFrame]:
    if int(calibration_seed) != 2718:
        raise PreflightError("nested expert-calibration seed must remain 2718")
    official = validated_waterbirds100_official_splits(metadata)
    if "metadata_row_index" not in metadata.columns:
        raise PreflightError(
            "authoritative metadata_row_index is missing; load source metadata with "
            "anchorcal.data.load_metadata before importing FCV membership"
        )
    membership = load_fcv_membership(
        fcv_split_manifest_root,
        fcv_reference,
        source_metadata_sha256=metadata_sha256,
    )
    source = metadata.rename(columns={"metadata_row_index": "metadata_index"}).copy()
    if source["metadata_index"].duplicated().any():
        raise PreflightError("source metadata_index values are not unique")
    by_index = source.set_index("metadata_index", drop=False)
    candidate_indices = set(membership.candidate_train_metadata_indices)
    biased_indices = set(membership.biased_validation_metadata_indices)
    source_train_indices = set(
        official["train"]["metadata_row_index"].astype(int).tolist()
    )
    if candidate_indices | biased_indices != source_train_indices:
        missing = sorted(source_train_indices - candidate_indices - biased_indices)[:20]
        unexpected = sorted((candidate_indices | biased_indices) - source_train_indices)[:20]
        raise PreflightError(
            "frozen FCV membership is not the complete official split-0 partition: "
            f"missing={missing}, unexpected={unexpected}"
        )
    try:
        candidate_train = by_index.loc[
            list(membership.candidate_train_metadata_indices)
        ].copy()
        biased_val = by_index.loc[
            list(membership.biased_validation_metadata_indices)
        ].copy()
    except KeyError as error:
        raise PreflightError("FCV membership references absent metadata rows") from error

    expert_train, expert_calibration = _split_stratified(
        candidate_train, 0.10, calibration_seed
    )
    hidden = {
        "oracle_val": official["oracle_val"].rename(
            columns={"metadata_row_index": "metadata_index"}
        ),
        "test": official["test"].rename(
            columns={"metadata_row_index": "metadata_index"}
        ),
    }
    result = {
        "candidate_train": candidate_train,
        "biased_val": biased_val,
        "expert_train": expert_train,
        "expert_calibration": expert_calibration,
        **hidden,
    }
    top_level_hashes = {
        "candidate_train": membership.provenance["candidate_train"][
            "metadata_indices_sha256"
        ],
        "biased_val": membership.provenance["biased_val"][
            "metadata_indices_sha256"
        ],
    }
    candidate_source_hash = top_level_hashes["candidate_train"]
    for name, frame in result.items():
        frame = frame.sort_values("img_id", kind="stable").reset_index(drop=True)
        frame["source_metadata_sha256"] = metadata_sha256
        if name in {"candidate_train", "biased_val"}:
            frame["membership_source"] = (
                f"{membership.provenance['study_id']}/split_indices.json"
            )
            frame["membership_seed"] = 0
            frame["source_membership_sha256"] = top_level_hashes[name]
        elif name in {"expert_train", "expert_calibration"}:
            frame["membership_source"] = "candidate_train/nested_expert_partition"
            frame["membership_seed"] = int(calibration_seed)
            frame["source_membership_sha256"] = candidate_source_hash
        else:
            frame["group"] = frame["y"].astype(np.int64) * 2 + frame[
                "place"
            ].astype(np.int64)
        result[name] = frame
    _assert_split_contract(result)
    return result


def _assert_split_contract(splits: dict[str, pd.DataFrame]) -> dict[str, object]:
    required_names = set(VISIBLE_SPLIT_NAMES) | set(ANALYSIS_ONLY_SPLIT_NAMES)
    if set(splits) != required_names:
        raise AssertionError(
            "split collection has missing or unexpected names: "
            f"found={sorted(splits)}, expected={sorted(required_names)}"
        )
    empty = sorted(name for name, frame in splits.items() if frame.empty)
    if empty:
        raise AssertionError(f"Waterbirds100 persisted splits are empty: {empty}")
    expected_official_split = {
        "candidate_train": 0,
        "biased_val": 0,
        "expert_train": 0,
        "expert_calibration": 0,
        "oracle_val": 1,
        "test": 2,
    }
    for name, frame in splits.items():
        if frame["img_id"].duplicated().any():
            raise AssertionError(f"{name} contains duplicate img_id values")
        observed_split = set(frame["split"].astype(int))
        if observed_split != {expected_official_split[name]}:
            raise AssertionError(
                f"{name} has wrong official split values: {sorted(observed_split)}"
            )
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
    for name in VISIBLE_SPLIT_NAMES:
        frame = splits[name]
        aligned = bool((frame["y"] == frame["place"]).all())
        if not aligned:
            raise AssertionError(f"{name} contains a shortcut-misaligned example")
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
        "derived_train_hidden_disjoint": {
            "passed": True,
            "overlap_count": len(train_hidden_overlap),
        },
        "candidate_train_biased_val_union": {
            "passed": True,
            "rows": len(top_level_train),
        },
        "complete_official_train_alignment_audit": {
            "passed": all(aligned_passed.values()),
            "misaligned_rows": 0,
        },
    }


def _describe_visible(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(frame)),
        "class_counts": {
            str(key): int(value)
            for key, value in frame["y"].value_counts().sort_index().items()
        },
    }


def _describe_protected(frame: pd.DataFrame) -> dict[str, object]:
    return {
        **_describe_visible(frame),
        "place_counts": {
            str(key): int(value)
            for key, value in frame["place"].value_counts().sort_index().items()
        },
        "group_counts": {
            str(key): int(value)
            for key, value in frame["group"].value_counts().sort_index().items()
        },
        "empirical_correlation": float((frame["y"] == frame["place"]).mean()),
    }


def _atomic_create_or_verify(path: Path, expected: bytes) -> None:
    """Publish ``expected`` once, or verify an identical prior publication."""

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
    path.parent.mkdir(parents=True, exist_ok=True)
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
            if not path.is_file() or path.read_bytes() != expected:
                actual_hash = sha256_file(path) if path.is_file() else "not-a-file"
                raise PreflightError(
                    f"concurrent nonidentical split publication at {path}: "
                    f"expected_sha256={sha256_bytes(expected)}, "
                    f"actual_sha256={actual_hash}"
                )
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def persist_splits(
    splits: dict[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    analysis_only_output_dir: str | Path,
    source_metadata: pd.DataFrame,
    source_metadata_sha256: str,
    source_release: str,
    fcv_split_manifest_root: str | Path,
    fcv_reference: dict[str, Any],
) -> dict[str, object]:
    contract = _assert_split_contract(splits)
    if source_release != WATERBIRDS100_RELEASE:
        raise PreflightError(
            f"split source release must be {WATERBIRDS100_RELEASE!r}, "
            f"found {source_release!r}"
        )
    membership = load_fcv_membership(
        fcv_split_manifest_root,
        fcv_reference,
        source_metadata_sha256=source_metadata_sha256,
    )
    official = validated_waterbirds100_official_splits(source_metadata)
    expected_train_indices = set(
        official["train"]["metadata_row_index"].astype(int).tolist()
    )
    observed_train_indices = set(
        splits["candidate_train"]["metadata_index"].astype(int).tolist()
    ) | set(splits["biased_val"]["metadata_index"].astype(int).tolist())
    if expected_train_indices != observed_train_indices:
        raise PreflightError("persisted development splits omit official training rows")
    if set(splits["candidate_train"]["metadata_index"].astype(int)) != set(
        membership.candidate_train_metadata_indices
    ) or set(splits["biased_val"]["metadata_index"].astype(int)) != set(
        membership.biased_validation_metadata_indices
    ):
        raise PreflightError(
            "persisted development membership differs from the frozen FCV lists"
        )
    recorded_metadata_hashes = {
        str(value)
        for frame in splits.values()
        for value in frame["source_metadata_sha256"].unique().tolist()
    }
    if recorded_metadata_hashes != {source_metadata_sha256}:
        raise PreflightError(
            "derived split metadata hashes do not match the authoritative source"
        )

    visible_root = Path(output_dir)
    protected_root = Path(analysis_only_output_dir)
    visible_root.mkdir(parents=True, exist_ok=True)
    protected_root.mkdir(parents=True, exist_ok=True)

    visible_paths: dict[str, Path] = {}
    visible_payloads: dict[str, bytes] = {}
    for name in VISIBLE_SPLIT_NAMES:
        frame = splits[name].loc[:, VISIBLE_SPLIT_COLUMNS].copy()
        forbidden = PROTECTED_COLUMNS.intersection(frame.columns)
        if forbidden:
            raise PreflightError(
                f"selector-visible split {name} contains protected columns: {sorted(forbidden)}"
            )
        path = visible_root / f"waterbirds100_{name}.csv"
        payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
        _atomic_create_or_verify(path, payload)
        visible_paths[name] = path
        visible_payloads[name] = payload

    protected_paths: dict[str, Path] = {}
    protected_payloads: dict[str, bytes] = {}
    for name in ANALYSIS_ONLY_SPLIT_NAMES:
        frame = splits[name].loc[:, ANALYSIS_ONLY_SPLIT_COLUMNS].copy()
        path = protected_root / f"waterbirds100_{name}.csv"
        payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
        _atomic_create_or_verify(path, payload)
        protected_paths[name] = path
        protected_payloads[name] = payload

    visible_manifest: dict[str, object] = {
        "schema_version": VISIBLE_SPLIT_SCHEMA,
        "namespace": "selector_visible",
        "source_release": source_release,
        "source_metadata_sha256": source_metadata_sha256,
        "source_fcv_membership": membership.provenance,
        "complete_official_train": {
            "rows": len(expected_train_indices),
            "metadata_indices_sha256": _metadata_indices_sha256(
                sorted(expected_train_indices)
            ),
            "all_y_equal_protected_context": True,
            "misaligned_rows": 0,
        },
        "contract_assertions": contract,
        "protected_columns_excluded": sorted(PROTECTED_COLUMNS),
        "splits": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_bytes(visible_payloads[name]),
                "columns": list(VISIBLE_SPLIT_COLUMNS),
                **_describe_visible(splits[name]),
            }
            for name, path in visible_paths.items()
        },
    }
    protected_manifest: dict[str, object] = {
        "schema_version": ANALYSIS_ONLY_SPLIT_SCHEMA,
        "namespace": "analysis_only",
        "reporting_only": True,
        "source_release": source_release,
        "source_metadata_sha256": source_metadata_sha256,
        "protected_columns": ["place", "group"],
        "splits": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_bytes(protected_payloads[name]),
                "columns": list(ANALYSIS_ONLY_SPLIT_COLUMNS),
                **_describe_protected(splits[name]),
            }
            for name, path in protected_paths.items()
        },
        "contract_assertions": {
            "oracle_val_test_disjoint": True,
            "development_hidden_disjoint": True,
            "group_equals_two_y_plus_place": True,
        },
    }
    visible_payload = (
        json.dumps(visible_manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    protected_payload = (
        json.dumps(protected_manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_create_or_verify(visible_root / "manifest.json", visible_payload)
    _atomic_create_or_verify(protected_root / "manifest.json", protected_payload)
    return visible_manifest
