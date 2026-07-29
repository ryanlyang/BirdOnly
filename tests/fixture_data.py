"""Generated Waterbirds-like fixtures for Phase 0 integration tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


def create_fixture(root: Path) -> tuple[Path, Path]:
    dataset_root = root / "waterbirds"
    mask_root = root / "prediction_cmap"
    image_root = dataset_root / "images"
    image_root.mkdir(parents=True)
    mask_root.mkdir(parents=True)
    rows = []
    sample_id = 0

    def add_example(y: int, place: int, split: int) -> None:
        nonlocal sample_id
        filename = f"bird_{sample_id:04d}.jpg"
        relative = f"images/{filename}"
        image = Image.new(
            "RGB",
            (40, 32),
            (20 + 80 * place, 120 + 30 * y, 180 - 50 * place),
        )
        draw = ImageDraw.Draw(image)
        left = 5 + (sample_id % 4)
        top = 7
        draw.rectangle((left, top, left + 10, top + 12), fill=(220, 30, 30))
        image.save(dataset_root / relative)

        mask = Image.new("RGB", image.size, (0, 0, 0))
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle((left, top, left + 10, top + 12), fill=(128, 0, 0))
        mask.save(mask_root / f"bird_{sample_id:04d}.png")
        rows.append(
            {
                "img_id": sample_id,
                "img_filename": relative,
                "y": y,
                "place": place,
                "split": split,
            }
        )
        sample_id += 1

    for y in (0, 1):
        for place in (0, 1):
            for _ in range(10):
                add_example(y, place, 0)
            for _ in range(2):
                add_example(y, place, 1)
            for _ in range(2):
                add_example(y, place, 2)
    pd.DataFrame(rows).to_csv(dataset_root / "metadata.csv", index=False)
    return dataset_root, mask_root


def fixture_config(dataset_root: Path, mask_root: Path, output: Path) -> dict:
    return {
        "schema_version": 1,
        "project_name": "fixture_setv",
        "data": {
            "dataset_root": str(dataset_root),
            "metadata_csv": "metadata.csv",
            "image_column": "img_filename",
            "sample_id_column": "img_id",
            "target_column": "y",
            "place_column": "place",
            "official_split_column": "split",
            "official_split_values": {"train": 0, "oracle_val": 1, "test": 2},
        },
        "masks": {
            "root": str(mask_root),
            "allowed_extensions": [".png"],
            "mapping_mode": "relative_stem_then_unique_basename",
            "format": "voc_colormap_class_ids",
            "foreground_class_ids": [1],
            "required_official_splits": [0, 1],
            "optional_official_splits": [2],
            "threshold_normalized": 1.0 / 255.0,
            "foreground_is_high": True,
            "require_same_dimensions": True,
            "minimum_foreground_fraction": 0.001,
            "maximum_foreground_fraction": 0.95,
        },
        "split": {"candidate_train_fraction": 0.8, "seed": 1729},
        "transforms": {
            "image_size": 24,
            "evaluation_resize_shortest": 28,
            "train_random_resized_crop_scale": [0.7, 1.0],
            "train_random_resized_crop_ratio": [0.75, 4.0 / 3.0],
            "train_horizontal_flip_probability": 0.5,
        },
        "audit": {
            "visual_samples_per_split": 2,
            "visual_seed": 1729,
            "contact_sheet_columns": 2,
            "thumbnail_size": 48,
        },
        "output": {"phase0_dir": str(output)},
    }
