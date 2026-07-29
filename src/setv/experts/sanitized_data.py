"""Datasets for paired training and eight-view sanitized evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from setv.data.dataset import WaterbirdsManifestDataset
from setv.data.joint_transforms import build_eval_transform, green_fill
from setv.errors import DataValidationError
from setv.experts.object_data import normalized_tensor
from setv.experts.sanitized_bank import load_bank
from setv.experts.sanitized_masks import unpack_masks
from setv.utils.seeds import derive_seed


class SanitizedBackgroundDataset:
    def __init__(
        self,
        manifest_csv: str | Path,
        split: str,
        phase0_config: dict[str, Any],
        expert_config: dict[str, Any],
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
        self.bank = load_bank(expert_config["mask_bank_dir"], split)
        expected_ids = self.base.rows["sample_id"].astype(str).to_numpy()
        if not np.array_equal(self.bank["sample_id"].astype(str), expected_ids):
            raise DataValidationError(
                f"Sanitized {split} bank does not align with its manifest"
            )
        self.training = training
        self.base_seed = int(expert_config["training"]["seed"])
        self.epoch = 0
        self.green_rgb = tuple(int(value) for value in expert_config["input"]["green_rgb"])
        self.mean = tuple(
            float(value) for value in expert_config["model"]["normalization_mean"]
        )
        self.std = tuple(
            float(value) for value in expert_config["model"]["normalization_std"]
        )
        self.transform = build_eval_transform(phase0_config)
        self.width = int(np.asarray(self.bank["width"]).item())
        self.height = int(np.asarray(self.bank["height"]).item())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.base)

    def _views(self, index: int, image: Image.Image, selected=None) -> list:
        masks = unpack_masks(self.bank["packed_masks"][index], self.width)
        if selected is not None:
            masks = masks[np.asarray(selected, dtype=np.int64)]
        return [
            normalized_tensor(
                green_fill(
                    image,
                    Image.fromarray(mask.astype(np.uint8) * 255, mode="L"),
                    keep_foreground=False,
                    green_rgb=self.green_rgb,
                ),
                self.mean,
                self.std,
            )
            for mask in masks
        ]

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        example = self.base[index]
        image, _ = self.transform(example.image, example.mask)
        if image.size != (self.width, self.height):
            raise DataValidationError(
                f"Canonical image {image.size} differs from bank {(self.width, self.height)}"
            )
        if self.training:
            generator = np.random.default_rng(
                derive_seed(
                    self.base_seed,
                    f"sanitized_pair:epoch={self.epoch}:sample={example.sample_id}",
                )
            )
            selected = generator.choice(8, size=2, replace=False)
            views = self._views(index, image, selected)
            return {
                "image_a": views[0],
                "image_b": views[1],
                "view_index_a": int(selected[0]),
                "view_index_b": int(selected[1]),
                "target": int(example.target),
                "sample_id": example.sample_id,
            }
        views = self._views(index, image)
        return {
            "images": torch.stack(views),
            "target": int(example.target),
            "sample_id": example.sample_id,
        }
