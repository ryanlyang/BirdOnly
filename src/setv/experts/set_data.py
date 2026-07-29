"""Background-patch selection and deterministic token-dropout datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from setv.data.dataset import WaterbirdsManifestDataset
from setv.data.joint_transforms import build_eval_transform, build_train_transform
from setv.errors import DataValidationError
from setv.experts.exact_data import dilate_binary_mask
from setv.experts.object_data import normalized_tensor
from setv.utils.seeds import derive_seed


def select_background_patch_tokens(
    mask,
    config: dict[str, Any],
    *,
    selection_seed: int,
    dropout_seed: int,
    apply_dropout: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    input_config = config["input"]
    training = config["training"]
    patch_size = int(input_config["patch_size"])
    width, height = mask.size
    if width != height or width % patch_size != 0:
        raise DataValidationError(
            f"Set expert requires square patch-divisible masks, got {mask.size}"
        )
    dilation_radius = max(
        1,
        int(
            round(
                int(input_config["dilation_pixels_at_224"]) * width / 224.0
            )
        ),
    )
    dilated = np.asarray(
        dilate_binary_mask(
            mask,
            dilation_radius,
            structuring_element=input_config["dilation_structuring_element"],
        )
    ) > 0
    grid = width // patch_size
    foreground_fraction = dilated.reshape(
        grid, patch_size, grid, patch_size
    ).mean(axis=(1, 3))
    valid = np.flatnonzero(
        foreground_fraction.reshape(-1)
        <= float(input_config["maximum_foreground_fraction"])
    )
    minimum = int(input_config["min_background_tokens"])
    if len(valid) < minimum:
        raise DataValidationError(
            f"Only {len(valid)} valid background patches remain; minimum is {minimum}"
        )
    maximum = int(input_config["max_background_tokens"])
    if len(valid) > maximum:
        generator = np.random.default_rng(selection_seed)
        valid = np.sort(generator.choice(valid, size=maximum, replace=False))
    before_dropout = len(valid)
    if apply_dropout:
        retained = max(
            minimum,
            int(round(before_dropout * (1.0 - float(training["token_dropout"])))),
        )
        retained = min(before_dropout, retained)
        generator = np.random.default_rng(dropout_seed)
        valid = np.sort(generator.choice(valid, size=retained, replace=False))
    token_mask = np.zeros(grid * grid, dtype=bool)
    token_mask[valid] = True
    return token_mask, {
        "valid_before_cap": int(
            (
                foreground_fraction.reshape(-1)
                <= float(input_config["maximum_foreground_fraction"])
            ).sum()
        ),
        "valid_after_cap": before_dropout,
        "retained_after_dropout": int(token_mask.sum()),
        "dilation_radius": dilation_radius,
    }


class SetBackgroundDataset:
    def __init__(
        self,
        manifest_csv: str | Path,
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
        self.phase0_config = phase0_config
        self.config = expert_config
        self.training = training
        self.base_seed = int(expert_config["training"]["seed"])
        self.epoch = 0
        self.mean = tuple(
            float(value) for value in expert_config["model"]["normalization_mean"]
        )
        self.std = tuple(
            float(value) for value in expert_config["model"]["normalization_std"]
        )
        self.eval_transform = build_eval_transform(phase0_config)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        example = self.base[index]
        if self.training:
            transform = build_train_transform(
                self.phase0_config,
                derive_seed(
                    self.base_seed,
                    f"set_transform:epoch={self.epoch}:sample={example.sample_id}",
                ),
            )
        else:
            transform = self.eval_transform
        image, mask = transform(example.image, example.mask)
        image_tensor = normalized_tensor(image, self.mean, self.std)
        if self.training:
            token_mask, counts = select_background_patch_tokens(
                mask,
                self.config,
                selection_seed=derive_seed(
                    self.base_seed,
                    f"set_cap:epoch={self.epoch}:sample={example.sample_id}",
                ),
                dropout_seed=derive_seed(
                    self.base_seed,
                    f"set_dropout:epoch={self.epoch}:sample={example.sample_id}",
                ),
                apply_dropout=True,
            )
            return {
                "image": image_tensor,
                "token_mask": torch.from_numpy(token_mask),
                "target": int(example.target),
                "sample_id": example.sample_id,
                **counts,
            }
        view_masks = []
        view_counts = []
        for view in range(int(self.config["training"]["validation_views"])):
            token_mask, counts = select_background_patch_tokens(
                mask,
                self.config,
                selection_seed=derive_seed(
                    self.base_seed, f"set_cap:sample={example.sample_id}"
                ),
                dropout_seed=derive_seed(
                    self.base_seed,
                    f"set_dropout:sample={example.sample_id}:view={view}",
                ),
                apply_dropout=True,
            )
            view_masks.append(torch.from_numpy(token_mask))
            view_counts.append(counts["retained_after_dropout"])
        return {
            "image": image_tensor,
            "token_masks": torch.stack(view_masks),
            "target": int(example.target),
            "sample_id": example.sample_id,
            "retained_token_counts": torch.tensor(view_counts, dtype=torch.int64),
        }
