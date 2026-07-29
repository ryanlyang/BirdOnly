"""Immutable Waterbirds95 split construction and auditing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from setv.errors import DataValidationError


SAFE_COLUMNS = [
    "sample_id",
    "metadata_index",
    "img_filename",
    "y",
    "official_split",
    "split_name",
    "mask_relative_path",
]


@dataclass
class SplitBuildResult:
    safe_manifests: dict[str, pd.DataFrame]
    protected_labels: pd.DataFrame
    summary: dict[str, Any]


def read_and_validate_metadata(config: dict) -> pd.DataFrame:
    data = config["data"]
    dataset_root = Path(data["dataset_root"]).expanduser().resolve()
    metadata_path = dataset_root / data["metadata_csv"]
    if not dataset_root.is_dir():
        raise DataValidationError(f"Waterbirds95 dataset root is missing: {dataset_root}")
    if not metadata_path.is_file():
        raise DataValidationError(f"Metadata CSV is missing: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    required = {
        data["image_column"],
        data["sample_id_column"],
        data["target_column"],
        data["place_column"],
        data["official_split_column"],
    }
    missing = required - set(metadata.columns)
    if missing:
        raise DataValidationError(
            f"Metadata is missing required columns: {sorted(missing)}"
        )
    if metadata[data["sample_id_column"]].isna().any():
        raise DataValidationError("Metadata contains null sample IDs")
    if metadata[data["sample_id_column"]].duplicated().any():
        duplicate = metadata.loc[
            metadata[data["sample_id_column"]].duplicated(), data["sample_id_column"]
        ].iloc[0]
        raise DataValidationError(f"Duplicate sample ID in metadata: {duplicate}")

    frame = pd.DataFrame(
        {
            "sample_id": metadata[data["sample_id_column"]].astype(str),
            "metadata_index": np.arange(len(metadata), dtype=np.int64),
            "img_filename": metadata[data["image_column"]].astype(str),
            "y": pd.to_numeric(metadata[data["target_column"]], errors="raise").astype(int),
            "place": pd.to_numeric(metadata[data["place_column"]], errors="raise").astype(int),
            "official_split": pd.to_numeric(
                metadata[data["official_split_column"]], errors="raise"
            ).astype(int),
        }
    )
    if set(frame["y"].unique()) != {0, 1}:
        raise DataValidationError(
            f"Expected binary target labels {{0, 1}}, found {sorted(frame['y'].unique())}"
        )
    if set(frame["place"].unique()) != {0, 1}:
        raise DataValidationError(
            f"Expected binary place labels {{0, 1}}, found "
            f"{sorted(frame['place'].unique())}"
        )
    expected_splits = set(data["official_split_values"].values())
    unknown = set(frame["official_split"].unique()) - expected_splits
    if unknown:
        raise DataValidationError(f"Unknown official split values: {sorted(unknown)}")

    missing_images = [
        filename
        for filename in frame["img_filename"]
        if not (dataset_root / filename).is_file()
    ]
    if missing_images:
        raise DataValidationError(
            f"{len(missing_images)} metadata images are missing; first: "
            f"{missing_images[:5]}"
        )
    return frame


def _split_training_groups(
    training: pd.DataFrame,
    candidate_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    candidate_indices: list[int] = []
    validation_indices: list[int] = []
    grouped = training.groupby(["y", "place"], sort=True)
    observed_groups = set(grouped.groups)
    if observed_groups != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise DataValidationError(
            f"Official training split must contain all four groups; got "
            f"{sorted(observed_groups)}"
        )
    for group, group_frame in grouped:
        indices = group_frame.index.to_numpy(copy=True)
        if len(indices) < 2:
            raise DataValidationError(
                f"Group {group} has {len(indices)} sample(s); cannot split both ways"
            )
        shuffled = rng.permutation(indices)
        validation_count = int(round(len(indices) * (1.0 - candidate_fraction)))
        validation_count = max(1, min(validation_count, len(indices) - 1))
        validation_indices.extend(shuffled[:validation_count].tolist())
        candidate_indices.extend(shuffled[validation_count:].tolist())
    candidate = training.loc[sorted(candidate_indices)].copy()
    validation = training.loc[sorted(validation_indices)].copy()
    return candidate, validation


def _summary(frame: pd.DataFrame) -> dict[str, Any]:
    group_counts = (
        frame.groupby(["y", "place"], sort=True)
        .size()
        .rename("count")
        .reset_index()
    )
    class_counts = frame.groupby("y", sort=True).size()
    sample_count = int(len(frame))
    return {
        "sample_count": sample_count,
        "class_counts": {str(int(key)): int(value) for key, value in class_counts.items()},
        "class_proportions": {
            str(int(key)): float(value / sample_count)
            for key, value in class_counts.items()
        },
        "group_counts": {
            f"y={int(row.y)},place={int(row.place)}": int(row["count"])
            for _, row in group_counts.iterrows()
        },
        "group_proportions": {
            f"y={int(row.y)},place={int(row.place)}": float(
                row["count"] / sample_count
            )
            for _, row in group_counts.iterrows()
        },
        "empirical_bird_background_correlation": float(
            (frame["y"] == frame["place"]).mean()
        ),
    }


def build_splits(frame: pd.DataFrame, config: dict) -> SplitBuildResult:
    split_values = config["data"]["official_split_values"]
    training = frame[frame["official_split"] == int(split_values["train"])]
    oracle = frame[frame["official_split"] == int(split_values["oracle_val"])].copy()
    test = frame[frame["official_split"] == int(split_values["test"])].copy()
    if training.empty or oracle.empty or test.empty:
        raise DataValidationError("Official train, validation, and test splits must be nonempty")

    candidate, biased = _split_training_groups(
        training,
        float(config["split"]["candidate_train_fraction"]),
        int(config["split"]["seed"]),
    )
    named = {
        "candidate_train": candidate,
        "biased_val": biased,
        "oracle_val": oracle,
        "test": test,
    }
    safe: dict[str, pd.DataFrame] = {}
    protected_parts: list[pd.DataFrame] = []
    summary: dict[str, Any] = {
        "algorithm": "independent_joint_group_split",
        "rounding": "round validation count per group, clamped to [1, n-1]",
        "seed": int(config["split"]["seed"]),
        "candidate_train_fraction": float(
            config["split"]["candidate_train_fraction"]
        ),
        "source_official_train": _summary(training),
        "splits": {},
    }
    for split_name, split_frame in named.items():
        if set(split_frame["y"].unique()) != {0, 1}:
            raise DataValidationError(f"{split_name} does not contain both classes")
        groups = set(zip(split_frame["y"], split_frame["place"]))
        if groups != {(0, 0), (0, 1), (1, 0), (1, 1)}:
            raise DataValidationError(
                f"{split_name} does not contain all four groups: {sorted(groups)}"
            )
        split_frame = split_frame.copy()
        split_frame["split_name"] = split_name
        split_frame["group"] = 2 * split_frame["y"] + split_frame["place"]
        protected_parts.append(
            split_frame[
                ["sample_id", "metadata_index", "split_name", "y", "place", "group"]
            ]
        )
        selector_safe = split_frame.drop(columns=["place", "group"])
        selector_safe["mask_relative_path"] = ""
        safe[split_name] = selector_safe[SAFE_COLUMNS].reset_index(drop=True)
        summary["splits"][split_name] = _summary(split_frame)

    protected = pd.concat(protected_parts, ignore_index=True).sort_values(
        "metadata_index"
    )
    if protected["sample_id"].duplicated().any():
        raise DataValidationError("A sample appeared in more than one constructed split")
    if len(protected) != len(frame):
        raise DataValidationError(
            f"Constructed splits cover {len(protected)} of {len(frame)} metadata rows"
        )
    return SplitBuildResult(safe, protected.reset_index(drop=True), summary)
