"""Waterbirds metadata and source-resolution image/mask access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .errors import PreflightError


REQUIRED_COLUMNS = ("img_id", "img_filename", "y", "place", "split")


def load_metadata(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise PreflightError(f"metadata is missing required columns: {missing}")
    for column in ("img_id", "y", "place", "split"):
        try:
            numeric = pd.to_numeric(frame[column], errors="raise").to_numpy(
                dtype=np.float64
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise PreflightError(f"metadata column {column!r} is not integral") from error
        if not np.isfinite(numeric).all() or not np.equal(
            numeric, np.trunc(numeric)
        ).all():
            raise PreflightError(f"metadata column {column!r} is not integral")
        frame[column] = numeric.astype(np.int64)
    if frame["img_id"].duplicated().any() or (frame["img_id"] < 0).any():
        duplicates = frame.loc[frame["img_id"].duplicated(False), "img_id"].tolist()
        raise PreflightError(
            "metadata img_id must be nonnegative and unique: "
            f"duplicates={duplicates[:20]}"
        )
    for column in ("y", "place"):
        invalid = sorted(set(frame[column].tolist()) - {0, 1})
        if invalid:
            raise PreflightError(
                f"metadata column {column!r} must be binary {{0,1}}; found {invalid}"
            )
    invalid_splits = sorted(set(frame["split"].tolist()) - {0, 1, 2})
    if invalid_splits:
        raise PreflightError(
            "metadata column 'split' must contain only {0,1,2}; "
            f"found {invalid_splits}"
        )
    if frame["img_filename"].isna().any():
        raise PreflightError("metadata img_filename must be non-null")
    frame["img_filename"] = frame["img_filename"].astype(str)
    if (frame["img_filename"].str.strip().str.len() == 0).any():
        raise PreflightError("metadata img_filename must be nonempty")
    unsafe_filenames = []
    for value in frame["img_filename"].tolist():
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            unsafe_filenames.append(value)
    if unsafe_filenames:
        raise PreflightError(
            "metadata img_filename must be a contained relative path without '..': "
            f"{unsafe_filenames[:20]}"
        )
    return frame.sort_values("img_id", kind="stable").reset_index(drop=True)


def image_path(root: str | Path, image_filename: str) -> Path:
    return Path(root) / image_filename


def mask_relative_path(image_filename: str) -> Path:
    return Path(image_filename).with_suffix(".png")


def mask_path(root: str | Path, image_filename: str) -> Path:
    return Path(root) / mask_relative_path(image_filename)


@dataclass(frozen=True)
class WaterbirdsRecord:
    img_id: int
    img_filename: str
    y: int
    place: int
    split: int


class WaterbirdsDataset:
    """Dataset returning raw PIL image/mask and immutable metadata.

    A transform must explicitly accept both image and mask.  There is no
    candidate-training convenience path that could accidentally feed masks to a
    model.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        image_root: str | Path,
        mask_root: str | Path,
        transform: Any | None = None,
    ) -> None:
        self.frame = frame.sort_values("img_id", kind="stable").reset_index(drop=True)
        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image_file = image_path(self.image_root, str(row.img_filename))
        mask_file = mask_path(self.mask_root, str(row.img_filename))
        with Image.open(image_file) as opened:
            image = opened.convert("RGB")
        with Image.open(mask_file) as opened:
            mask = opened.copy()
        sample: dict[str, Any] = {
            "image": image,
            "mask": mask,
            "img_id": int(row.img_id),
            "y": int(row.y),
            "place": int(row.place),
            "split": int(row.split),
            "img_filename": str(row.img_filename),
        }
        if self.transform is not None:
            transformed = self.transform(
                image=image,
                mask=mask,
                img_id=sample["img_id"],
            )
            sample.update(transformed)
        return sample


class CandidateTrainView:
    """Erase masks from the training-model API while retaining joint geometry."""

    def __init__(self, dataset: WaterbirdsDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Any, int, int]:
        sample = self.dataset[index]
        return sample["image"], sample["y"], sample["img_id"]
