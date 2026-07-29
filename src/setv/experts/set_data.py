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


class InsufficientBackgroundTokensError(DataValidationError):
    """A transformed view cannot satisfy the locked set-cardinality floor."""

    def __init__(self, available: int, minimum: int):
        self.available = int(available)
        self.minimum = int(minimum)
        super().__init__(
            f"Only {self.available} valid background patches remain; "
            f"minimum is {self.minimum}"
        )


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
        raise InsufficientBackgroundTokensError(len(valid), minimum)
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


def select_training_background_view(
    image,
    mask,
    phase0_config: dict[str, Any],
    expert_config: dict[str, Any],
    *,
    sample_id: str,
    epoch: int,
    base_seed: int,
    eval_transform,
):
    """Select a deterministic valid crop without weakening patch eligibility."""

    maximum_attempts = int(
        expert_config["input"]["training_crop_max_attempts"]
    )
    rejected_counts: list[int] = []
    for attempt in range(maximum_attempts):
        namespace = f"epoch={epoch}:sample={sample_id}"
        if attempt:
            namespace = f"{namespace}:retry={attempt}"
        transform = build_train_transform(
            phase0_config,
            derive_seed(base_seed, f"set_transform:{namespace}"),
        )
        transformed_image, transformed_mask = transform(image, mask)
        try:
            token_mask, counts = select_background_patch_tokens(
                transformed_mask,
                expert_config,
                selection_seed=derive_seed(
                    base_seed, f"set_cap:{namespace}"
                ),
                dropout_seed=derive_seed(
                    base_seed, f"set_dropout:{namespace}"
                ),
                apply_dropout=True,
            )
        except InsufficientBackgroundTokensError as exc:
            rejected_counts.append(exc.available)
            continue
        return transformed_image, token_mask, counts, {
            "training_crop_attempt_count": attempt + 1,
            "training_crop_rejected_count": attempt,
            "training_crop_fallback_used": 0,
        }

    if (
        expert_config["input"]["training_crop_fallback"]
        != "canonical_evaluation_transform"
    ):
        raise DataValidationError("Unsupported set-expert crop fallback")
    transformed_image, transformed_mask = eval_transform(image, mask)
    try:
        token_mask, counts = select_background_patch_tokens(
            transformed_mask,
            expert_config,
            selection_seed=derive_seed(
                base_seed,
                f"set_cap:epoch={epoch}:sample={sample_id}:fallback",
            ),
            dropout_seed=derive_seed(
                base_seed,
                f"set_dropout:epoch={epoch}:sample={sample_id}:fallback",
            ),
            apply_dropout=True,
        )
    except InsufficientBackgroundTokensError as exc:
        raise DataValidationError(
            "Set-expert canonical fallback cannot satisfy the locked token "
            f"minimum for sample {sample_id}: "
            f"rejected_crop_counts={rejected_counts}, "
            f"fallback_available={exc.available}, minimum={exc.minimum}"
        ) from exc
    return transformed_image, token_mask, counts, {
        "training_crop_attempt_count": maximum_attempts,
        "training_crop_rejected_count": maximum_attempts,
        "training_crop_fallback_used": 1,
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

    def audit_canonical_background_capacity(self) -> dict[str, Any]:
        available_counts: list[int] = []
        minimum_sample_id: str | None = None
        minimum_available: int | None = None
        for index in range(len(self.base)):
            example = self.base[index]
            _, mask = self.eval_transform(example.image, example.mask)
            try:
                _, counts = select_background_patch_tokens(
                    mask,
                    self.config,
                    selection_seed=derive_seed(
                        self.base_seed,
                        f"set_fallback_audit:sample={example.sample_id}",
                    ),
                    dropout_seed=0,
                    apply_dropout=False,
                )
            except InsufficientBackgroundTokensError as exc:
                raise DataValidationError(
                    "Set-expert canonical fallback fails the locked token "
                    f"minimum for sample {example.sample_id}: "
                    f"available={exc.available}, minimum={exc.minimum}"
                ) from exc
            available = int(counts["valid_before_cap"])
            available_counts.append(available)
            if minimum_available is None or available < minimum_available:
                minimum_available = available
                minimum_sample_id = str(example.sample_id)
        return {
            "sample_count": len(available_counts),
            "minimum_valid_background_tokens": int(minimum_available),
            "maximum_valid_background_tokens": int(max(available_counts)),
            "mean_valid_background_tokens": float(np.mean(available_counts)),
            "minimum_sample_id": minimum_sample_id,
            "locked_minimum_tokens": int(
                self.config["input"]["min_background_tokens"]
            ),
            "status": "passed",
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        example = self.base[index]
        if self.training:
            image, token_mask, counts, crop_metadata = (
                select_training_background_view(
                    example.image,
                    example.mask,
                    self.phase0_config,
                    self.config,
                    sample_id=str(example.sample_id),
                    epoch=self.epoch,
                    base_seed=self.base_seed,
                    eval_transform=self.eval_transform,
                )
            )
            image_tensor = normalized_tensor(image, self.mean, self.std)
            return {
                "image": image_tensor,
                "token_mask": torch.from_numpy(token_mask),
                "target": int(example.target),
                "sample_id": example.sample_id,
                **counts,
                **crop_metadata,
            }
        image, mask = self.eval_transform(example.image, example.mask)
        image_tensor = normalized_tensor(image, self.mean, self.std)
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
