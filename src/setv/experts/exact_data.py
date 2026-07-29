"""Dilated exact-mask green-fill background views."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from setv.data.dataset import WaterbirdsManifestDataset
from setv.data.joint_transforms import (
    build_eval_transform,
    build_train_transform,
    green_fill,
)
from setv.experts.object_data import normalized_tensor
from setv.utils.seeds import derive_seed


def dilate_binary_mask(
    mask: Image.Image,
    radius: int,
    *,
    structuring_element: str = "euclidean_disk",
) -> Image.Image:
    """Dilate a binary mask by an explicit Euclidean pixel radius."""
    if radius < 0:
        raise ValueError("radius cannot be negative")
    if structuring_element != "euclidean_disk":
        raise ValueError(f"Unsupported structuring element: {structuring_element}")
    binary = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    if radius == 0:
        return Image.fromarray(binary.astype(np.uint8) * 255, mode="L")
    try:
        from scipy.ndimage import binary_dilation
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for the locked Euclidean-disk dilation"
        ) from exc
    coordinates = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    structure = (xx * xx + yy * yy) <= radius * radius
    dilated = binary_dilation(binary, structure=structure)
    return Image.fromarray(dilated.astype(np.uint8) * 255, mode="L")


class ExactBackgroundDataset:
    def __init__(
        self,
        manifest_csv: str | Path,
        phase0_config: dict[str, Any],
        exact_config: dict[str, Any],
        *,
        training: bool,
    ):
        masks = phase0_config["masks"]
        self.base = WaterbirdsManifestDataset(
            manifest_csv,
            phase0_config["data"]["dataset_root"],
            masks["root"],
            threshold_normalized=float(masks["threshold_normalized"]),
            foreground_is_high=bool(masks["foreground_is_high"]),
            map_format=str(masks.get("format", "threshold")),
            foreground_class_ids=tuple(
                int(value) for value in masks.get("foreground_class_ids", [1])
            ),
        )
        self.phase0_config = phase0_config
        self.training = training
        self.base_seed = int(exact_config["training"]["seed"])
        self.epoch = 0
        self.green_rgb = tuple(int(value) for value in exact_config["input"]["green_rgb"])
        self.radius_at_224 = int(exact_config["input"]["dilation_pixels_at_224"])
        self.structuring_element = exact_config["input"]["dilation_structuring_element"]
        self.mean = tuple(
            float(value) for value in exact_config["model"]["normalization_mean"]
        )
        self.std = tuple(
            float(value) for value in exact_config["model"]["normalization_std"]
        )
        self.eval_transform = build_eval_transform(phase0_config)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.base[index]
        if self.training:
            sample_seed = derive_seed(
                self.base_seed,
                f"background_exact:epoch={self.epoch}:sample={example.sample_id}",
            )
            transform = build_train_transform(self.phase0_config, sample_seed)
        else:
            transform = self.eval_transform
        image, mask = transform(example.image, example.mask)
        if image.width != image.height:
            raise ValueError(
                f"Exact expert expects a square transformed image, got {image.size}"
            )
        radius = max(1, int(round(self.radius_at_224 * image.width / 224.0)))
        dilated = dilate_binary_mask(
            mask, radius, structuring_element=self.structuring_element
        )
        background_view = green_fill(
            image,
            dilated,
            keep_foreground=False,
            green_rgb=self.green_rgb,
        )
        return {
            "image": normalized_tensor(background_view, self.mean, self.std),
            "target": int(example.target),
            "sample_id": example.sample_id,
        }
