"""Deterministic visual VLM-mask audit artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from setv.data.joint_transforms import (
    binarize_mask,
    build_eval_transform,
    green_fill,
)
from setv.errors import DataValidationError
from setv.experts.exact_data import dilate_binary_mask
from setv.utils.hashing import sha256_file


def select_visual_audit_samples(
    manifests: dict[str, pd.DataFrame],
    samples_per_split: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    selections: list[pd.DataFrame] = []
    for split_name in ("candidate_train", "biased_val", "oracle_val"):
        frame = manifests[split_name]
        frame = frame[
            frame["mask_relative_path"].fillna("").astype(str).str.len() > 0
        ].reset_index(drop=True)
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
    map_format: str = "threshold",
    foreground_class_ids: tuple[int, ...] = (1,),
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
                    map_format=map_format,
                    foreground_class_ids=foreground_class_ids,
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


def render_preflight_galleries(
    audit_samples: pd.DataFrame,
    *,
    config: dict[str, Any],
    output_dir: str | Path,
    samples_per_page: int,
    thumbnail_size: int,
    exact_dilation_pixels_at_224: int = 8,
) -> list[dict[str, Any]]:
    """Render raw alignment and exact evaluation green-fill views."""
    if samples_per_page < 1 or thumbnail_size < 32:
        raise ValueError("Invalid preflight gallery geometry")
    dataset_root = Path(config["data"]["dataset_root"])
    mask_root = Path(config["masks"]["root"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    threshold = float(config["masks"]["threshold_normalized"])
    foreground_is_high = bool(config["masks"]["foreground_is_high"])
    map_format = str(config["masks"].get("format", "threshold"))
    foreground_class_ids = tuple(
        int(value)
        for value in config["masks"].get("foreground_class_ids", [1])
    )
    eval_transform = build_eval_transform(config)
    view_names = (
        "original",
        "mask overlay (red)",
        "bird + green background",
        "background + green bird",
    )
    header_height = 30
    label_height = 24
    receipts: list[dict[str, Any]] = []

    for split_name, frame in audit_samples.groupby("split_name", sort=False):
        frame = frame.reset_index(drop=True)
        for page_start in range(0, len(frame), samples_per_page):
            page = frame.iloc[page_start : page_start + samples_per_page]
            sheet = Image.new(
                "RGB",
                (
                    len(view_names) * thumbnail_size,
                    header_height + len(page) * (thumbnail_size + label_height),
                ),
                "white",
            )
            draw = ImageDraw.Draw(sheet)
            for column, name in enumerate(view_names):
                draw.text(
                    (column * thumbnail_size + 4, 8),
                    name,
                    fill=(0, 0, 0),
                )

            page_samples = []
            for row_index, (_, row) in enumerate(page.iterrows()):
                image_path = dataset_root / str(row["img_filename"])
                mask_path = mask_root / str(row["mask_relative_path"])
                with Image.open(image_path) as opened:
                    image = opened.convert("RGB")
                with Image.open(mask_path) as opened:
                    mask = binarize_mask(
                        opened,
                        threshold_normalized=threshold,
                        foreground_is_high=foreground_is_high,
                        map_format=map_format,
                        foreground_class_ids=foreground_class_ids,
                    )
                if image.size != mask.size:
                    raise DataValidationError(
                        "Preflight gallery image-mask mismatch for "
                        f"sample_id={row['sample_id']}: "
                        f"{image.size} versus {mask.size}"
                    )
                eval_image, eval_mask = eval_transform(image, mask)
                object_view = green_fill(
                    eval_image,
                    eval_mask,
                    keep_foreground=True,
                    green_rgb=(0, 255, 0),
                )
                dilation_radius = max(
                    1,
                    int(
                        round(
                            exact_dilation_pixels_at_224
                            * eval_image.width
                            / 224.0
                        )
                    ),
                )
                dilated = dilate_binary_mask(
                    eval_mask,
                    dilation_radius,
                    structuring_element="euclidean_disk",
                )
                background_view = green_fill(
                    eval_image,
                    dilated,
                    keep_foreground=False,
                    green_rgb=(0, 255, 0),
                )
                views = (
                    image,
                    _overlay(image, mask),
                    object_view,
                    background_view,
                )
                top = header_height + row_index * (thumbnail_size + label_height)
                for column, view in enumerate(views):
                    thumbnail = ImageOps.contain(
                        view,
                        (thumbnail_size, thumbnail_size),
                        method=Image.Resampling.BICUBIC,
                    )
                    left = (
                        column * thumbnail_size
                        + (thumbnail_size - thumbnail.width) // 2
                    )
                    vertical = top + (thumbnail_size - thumbnail.height) // 2
                    sheet.paste(thumbnail, (left, vertical))
                label = (
                    f"split={split_name} id={row['sample_id']} y={row['y']} "
                    f"mask={row['mask_relative_path']}"
                )
                draw.text(
                    (4, top + thumbnail_size + 4),
                    label[:100],
                    fill=(0, 0, 0),
                )
                page_samples.append(str(row["sample_id"]))

            page_number = page_start // samples_per_page + 1
            destination = output_dir / (
                f"{split_name}_mask_views_page_{page_number:02d}.png"
            )
            sheet.save(destination, format="PNG", compress_level=6)
            receipts.append(
                {
                    "split_name": str(split_name),
                    "page": page_number,
                    "path": str(destination.resolve()),
                    "sha256": sha256_file(destination),
                    "sample_count": len(page_samples),
                    "sample_ids": page_samples,
                    "views": list(view_names),
                    "green_rgb": [0, 255, 0],
                    "exact_dilation_pixels_at_224": (
                        exact_dilation_pixels_at_224
                    ),
                    "rendered_dilation_pixels": dilation_radius,
                }
            )
    return receipts
