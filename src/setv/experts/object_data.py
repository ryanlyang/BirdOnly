"""Object-only Waterbirds views backed by approved Phase 0 manifests."""

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
from setv.utils.seeds import derive_seed


def normalized_tensor(
    image: Image.Image,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
):
    """Convert a composed raw-RGB PIL image to a normalized CHW torch tensor."""
    import torch

    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean_tensor) / std_tensor


class ObjectExpertDataset:
    """Create original-position bird-on-green views deterministically."""

    def __init__(
        self,
        manifest_csv: str | Path,
        phase0_config: dict[str, Any],
        object_config: dict[str, Any],
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
        )
        self.phase0_config = phase0_config
        self.training = training
        self.base_seed = int(object_config["training"]["seed"])
        self.epoch = 0
        self.green_rgb = tuple(int(value) for value in object_config["input"]["green_rgb"])
        self.mean = tuple(
            float(value) for value in object_config["model"]["normalization_mean"]
        )
        self.std = tuple(
            float(value) for value in object_config["model"]["normalization_std"]
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
                f"object_train:epoch={self.epoch}:sample={example.sample_id}",
            )
            transform = build_train_transform(self.phase0_config, sample_seed)
        else:
            transform = self.eval_transform
        image, mask = transform(example.image, example.mask)
        object_view = green_fill(
            image,
            mask,
            keep_foreground=True,
            green_rgb=self.green_rgb,
        )
        return {
            "image": normalized_tensor(object_view, self.mean, self.std),
            "target": int(example.target),
            "sample_id": example.sample_id,
        }

