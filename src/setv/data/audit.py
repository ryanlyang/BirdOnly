"""Deterministic visual VLM-mask audit artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from setv.data.joint_transforms import binarize_mask
from setv.errors import DataValidationError
from setv.utils.hashing import sha256_file


def select_visual_audit_samples(
    manifests: dict[str, pd.DataFrame],
    samples_per_split: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    selections: list[pd.DataFrame] = []
    for split_name in ("candidate_train", "biased_val", "oracle_val", "test"):
        frame = manifests[split_name]
        if len(frame) < samples_per_split:
            raise DataValidationError(
                f"{split_name} has only {len(frame)} examples; cannot inspect the "
                f"required {samples_per_split}"
            )
        chosen = np.sort(rng.choice(len(frame), size=samples_per_split, replace=False))
        selected = frame.iloc[chosen].copy()
        selected["audit_order"] = np.arange(samples_per_split)
        selections.append(selected)
    return pd.concat(selections, ignore_index=True)


def _overlay(image: Image.Image, mask: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    binary = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    array = np.asarray(rgb, dtype=np.uint8).copy()
    tint = np.zeros_like(array)
    tint[..., 0] = 255
    array[binary] = (
        0.55 * array[binary].astype(np.float32)
        + 0.45 * tint[binary].astype(np.float32)
    ).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def render_contact_sheets(
    audit_samples: pd.DataFrame,
    *,
    dataset_root: str | Path,
    mask_root: str | Path,
    output_dir: str | Path,
    threshold_normalized: float,
    foreground_is_high: bool,
    columns: int,
    thumbnail_size: int,
) -> list[dict[str, Any]]:
    dataset_root = Path(dataset_root)
    mask_root = Path(mask_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, Any]] = []

    for split_name, frame in audit_samples.groupby("split_name", sort=False):
        rows = int(np.ceil(len(frame) / columns))
        label_height = 24
        cell_width = thumbnail_size * 2
        cell_height = thumbnail_size + label_height
        sheet = Image.new(
            "RGB", (columns * cell_width, rows * cell_height), (255, 255, 255)
        )
        draw = ImageDraw.Draw(sheet)
        for position, (_, row) in enumerate(frame.iterrows()):
            image_path = dataset_root / str(row["img_filename"])
            mask_path = mask_root / str(row["mask_relative_path"])
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            with Image.open(mask_path) as opened:
                mask = binarize_mask(
                    opened,
                    threshold_normalized=threshold_normalized,
                    foreground_is_high=foreground_is_high,
                )
            if image.size != mask.size:
                mask = mask.resize(image.size, Image.Resampling.NEAREST)
            original = ImageOps.contain(
                image.copy(),
                (thumbnail_size, thumbnail_size),
                method=Image.Resampling.BICUBIC,
            )
            overlaid = ImageOps.contain(
                _overlay(image, mask),
                (thumbnail_size, thumbnail_size),
                method=Image.Resampling.BICUBIC,
            )

            x = (position % columns) * cell_width
            y = (position // columns) * cell_height
            original_offset = (
                x + (thumbnail_size - original.width) // 2,
                y + label_height + (thumbnail_size - original.height) // 2,
            )
            overlay_offset = (
                x + thumbnail_size + (thumbnail_size - overlaid.width) // 2,
                y + label_height + (thumbnail_size - overlaid.height) // 2,
            )
            sheet.paste(original, original_offset)
            sheet.paste(overlaid, overlay_offset)
            label = f"id={row['sample_id']} y={row['y']}"
            draw.text((x + 3, y + 4), label, fill=(0, 0, 0))

        destination = output_dir / f"{split_name}_contact_sheet.jpg"
        sheet.save(destination, format="JPEG", quality=92, subsampling=0)
        receipts.append(
            {
                "split_name": split_name,
                "path": destination.name,
                "sample_count": int(len(frame)),
                "sha256": sha256_file(destination),
            }
        )
    return receipts
