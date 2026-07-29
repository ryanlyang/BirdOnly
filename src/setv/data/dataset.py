"""Manifest-backed image/mask access without exposing protected group labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image

from setv.data.joint_transforms import binarize_mask
from setv.errors import DataValidationError


SAFE_MANIFEST_COLUMNS = {
    "sample_id",
    "metadata_index",
    "img_filename",
    "y",
    "official_split",
    "split_name",
    "mask_relative_path",
}
FORBIDDEN_SELECTOR_COLUMNS = {"place", "group", "group_id", "background"}


@dataclass
class WaterbirdsExample:
    sample_id: str
    image: Image.Image
    mask: Image.Image
    target: int


class WaterbirdsManifestDataset:
    """Load original RGB images and VLM masks from a selector-safe manifest."""

    def __init__(
        self,
        manifest_csv: str | Path,
        dataset_root: str | Path,
        mask_root: str | Path,
        *,
        threshold_normalized: float = 0.5,
        foreground_is_high: bool = True,
        map_format: str = "threshold",
        foreground_class_ids: tuple[int, ...] = (1,),
        transform: Callable | None = None,
    ):
        self.manifest_path = Path(manifest_csv)
        self.dataset_root = Path(dataset_root)
        self.mask_root = Path(mask_root)
        self.threshold_normalized = threshold_normalized
        self.foreground_is_high = foreground_is_high
        self.map_format = map_format
        self.foreground_class_ids = foreground_class_ids
        self.transform = transform
        self.rows = pd.read_csv(self.manifest_path, dtype={"sample_id": str})
        forbidden = FORBIDDEN_SELECTOR_COLUMNS.intersection(self.rows.columns)
        if forbidden:
            raise DataValidationError(
                f"Selector-safe manifest contains protected columns: {sorted(forbidden)}"
            )
        missing = {"sample_id", "img_filename", "mask_relative_path", "y"} - set(
            self.rows.columns
        )
        if missing:
            raise DataValidationError(
                f"Manifest is missing required columns: {sorted(missing)}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> WaterbirdsExample:
        row = self.rows.iloc[index]
        image_path = self.dataset_root / str(row["img_filename"])
        mask_path = self.mask_root / str(row["mask_relative_path"])
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        with Image.open(mask_path) as opened:
            mask = binarize_mask(
                opened,
                self.threshold_normalized,
                self.foreground_is_high,
                map_format=self.map_format,
                foreground_class_ids=self.foreground_class_ids,
            )
        if image.size != mask.size:
            raise DataValidationError(
                f"Runtime image-mask mismatch for sample_id={row['sample_id']}: "
                f"{image.size} versus {mask.size}"
            )
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return WaterbirdsExample(
            sample_id=str(row["sample_id"]),
            image=image,
            mask=mask,
            target=int(row["y"]),
        )
