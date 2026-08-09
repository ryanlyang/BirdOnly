"""Waterbirds metadata validation and contained source-image access."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .errors import PreflightError


REQUIRED_COLUMNS = ("img_id", "img_filename", "y", "place", "split")


def load_metadata(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["metadata_row_index"] = np.arange(len(frame), dtype=np.int64)
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
        normalized = str(value).strip().replace("\\", "/")
        relative = Path(normalized)
        windows_absolute = (
            len(normalized) >= 3
            and normalized[0].isalpha()
            and normalized[1:3] == ":/"
        )
        if relative.is_absolute() or windows_absolute or ".." in relative.parts:
            unsafe_filenames.append(value)
    if unsafe_filenames:
        raise PreflightError(
            "metadata img_filename must be a contained relative path without '..': "
            f"{unsafe_filenames[:20]}"
        )
    return frame.sort_values("img_id", kind="stable").reset_index(drop=True)


def validated_waterbirds100_official_splits(
    metadata: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Return all three nonempty official splits or fail the WB100 contract."""

    official = {
        "train": metadata.loc[metadata["split"] == 0].copy(),
        "oracle_val": metadata.loc[metadata["split"] == 1].copy(),
        "test": metadata.loc[metadata["split"] == 2].copy(),
    }
    empty = [name for name, frame in official.items() if frame.empty]
    if empty:
        raise PreflightError(
            "Waterbirds100 official train, validation, and test splits must all "
            f"be nonempty; empty={empty}"
        )
    official_train = official["train"]
    misaligned = official_train.loc[official_train["y"] != official_train["place"]]
    if not misaligned.empty:
        raise PreflightError(
            "Waterbirds100 source training split is not completely correlated: "
            f"{len(misaligned)} of {len(official_train)} rows have y != place; "
            f"example_img_ids={misaligned['img_id'].astype(int).tolist()[:20]}"
        )
    return official


def image_path(root: str | Path, image_filename: str) -> Path:
    resolved_root = Path(root).expanduser().resolve()
    normalized = str(image_filename).strip().replace("\\", "/")
    relative = Path(normalized)
    windows_absolute = (
        len(normalized) >= 3
        and normalized[0].isalpha()
        and normalized[1:3] == ":/"
    )
    if (
        not normalized
        or relative.is_absolute()
        or windows_absolute
        or ".." in relative.parts
    ):
        raise PreflightError(
            f"unsafe Waterbirds img_filename: {image_filename!r}"
        )
    candidates = (
        (resolved_root / relative).resolve(strict=False),
        (resolved_root / "images" / relative).resolve(strict=False),
    )
    for candidate in candidates:
        try:
            candidate.relative_to(resolved_root)
        except ValueError as error:
            raise PreflightError(
                f"Waterbirds image path escapes its root: {image_filename!r}"
            ) from error
    existing = tuple(
        dict.fromkeys(candidate for candidate in candidates if candidate.is_file())
    )
    if len(existing) > 1:
        raise PreflightError(
            f"ambiguous Waterbirds image paths for {image_filename!r}: "
            f"{[str(candidate) for candidate in existing]}"
        )
    if existing:
        return existing[0]
    # Preserve a concrete, deterministic missing-file path for preflight's
    # complete failure report.  Waterbirds releases normally store metadata
    # paths directly under the release root; ``images/`` is the documented
    # alternate layout.
    return candidates[0]
