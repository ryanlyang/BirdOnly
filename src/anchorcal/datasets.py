"""Torch datasets with epoch carried explicitly through persistent workers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from PIL import Image

from .data import image_path
from .preprocessing import EvaluationPreprocessing
from .seeds import stateless_rng
from .transforms import (
    criterion_patch_eligibility,
    deterministic_eval_transform,
    deterministic_image_eval_transform,
    foreground_eval_transform,
    foreground_train_transform,
    normalize_image,
    stateless_image_train_transform,
    stateless_train_transform,
)
from .vlm_masks import VlmMaskBank


class EpochSampler:
    """Yield ``(epoch,index)`` so worker-resident datasets see epoch changes."""

    def __init__(self, size: int, seed: int, *, shuffle: bool = True) -> None:
        self.size = int(size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        indices = np.arange(self.size)
        if self.shuffle:
            stateless_rng(self.seed, self.epoch, "training_sampler").shuffle(indices)
        return iter((self.epoch, int(index)) for index in indices.tolist())

    def __len__(self) -> int:
        return self.size


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return opened.convert("RGB")


def _tensor(array: np.ndarray):
    import torch

    return torch.from_numpy(np.ascontiguousarray(array))


class CandidateTrainingDataset:
    """Mask-free candidate training dataset API."""

    def __init__(
        self,
        frame: pd.DataFrame,
        image_root: str | Path,
        run_seed: int,
        *,
        preprocessing: EvaluationPreprocessing,
    ) -> None:
        self.frame = frame.sort_values("img_id", kind="stable").reset_index(drop=True)
        self.image_root = Path(image_root)
        self.run_seed = int(run_seed)
        self.preprocessing = preprocessing

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, key: tuple[int, int]) -> dict[str, Any]:
        epoch, index = key
        row = self.frame.iloc[index]
        image = _load_image(image_path(self.image_root, str(row.img_filename)))
        transformed = stateless_image_train_transform(
            image,
            run_seed=self.run_seed,
            epoch=epoch,
            img_id=int(row.img_id),
            purpose="candidate_train",
            image_size=self.preprocessing.image_size,
        )
        return {
            "image": _tensor(
                normalize_image(
                    transformed.image,
                    mean=self.preprocessing.mean,
                    std=self.preprocessing.std,
                )
            ),
            "y": int(row.y),
            "img_id": int(row.img_id),
        }


class PlainEvaluationDataset:
    """Mask-free ordinary evaluation for oracle/test namespace computation."""

    def __init__(
        self,
        frame: pd.DataFrame,
        image_root: str | Path,
        *,
        preprocessing: EvaluationPreprocessing,
    ) -> None:
        self.frame = frame.sort_values("img_id", kind="stable").reset_index(drop=True)
        self.image_root = Path(image_root)
        self.preprocessing = preprocessing

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image = _load_image(image_path(self.image_root, str(row.img_filename)))
        transformed = deterministic_image_eval_transform(
            image,
            image_size=self.preprocessing.image_size,
            resize_shortest=self.preprocessing.effective_resize_shortest,
        )
        sample = {
            "image": _tensor(
                normalize_image(
                    transformed.image,
                    mean=self.preprocessing.mean,
                    std=self.preprocessing.std,
                )
            ),
            "y": int(row.y),
            "img_id": int(row.img_id),
        }
        if "place" in self.frame.columns:
            sample["place"] = int(row.place)
        return sample


class BranchTrainingDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        image_root: str | Path,
        mask_bank: VlmMaskBank,
        *,
        branch: str,
        run_seed: int,
        token_budget: int | None = None,
        background_seed: int = 6003,
        preprocessing: EvaluationPreprocessing,
    ) -> None:
        if branch not in {"foreground", "background"}:
            raise ValueError("branch must be foreground or background")
        self.frame = frame.sort_values("img_id", kind="stable").reset_index(drop=True)
        self.image_root = Path(image_root)
        self.mask_bank = mask_bank
        self.branch = branch
        self.run_seed = int(run_seed)
        self.token_budget = token_budget
        self.background_seed = int(background_seed)
        self.preprocessing = preprocessing
        if branch == "background" and token_budget is None:
            raise ValueError("background dataset requires a token budget")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, key: tuple[int, int]) -> dict[str, Any]:
        epoch, index = key
        row = self.frame.iloc[index]
        image = _load_image(image_path(self.image_root, str(row.img_filename)))
        mask = self.mask_bank.load(int(row.img_id), str(row.img_filename))
        if self.branch == "foreground":
            transformed = foreground_train_transform(
                image,
                mask,
                run_seed=self.run_seed,
                epoch=epoch,
                img_id=int(row.img_id),
                image_size=self.preprocessing.image_size,
                resize_shortest=self.preprocessing.effective_resize_shortest,
            )
        else:
            transformed = stateless_train_transform(
                image,
                mask,
                run_seed=self.run_seed,
                epoch=epoch,
                img_id=int(row.img_id),
                purpose="background_branch_train_geometry",
                require_visible_foreground=True,
                image_size=self.preprocessing.image_size,
                resize_shortest=self.preprocessing.effective_resize_shortest,
            )
        sample: dict[str, Any] = {
            "image": _tensor(
                normalize_image(
                    transformed.image,
                    mean=self.preprocessing.mean,
                    std=self.preprocessing.std,
                )
            ),
            "mask": _tensor(transformed.mask.astype(bool)),
            "y": int(row.y),
            "img_id": int(row.img_id),
            "fallback": bool(transformed.fallback_used),
            "attempt_count": int(transformed.attempt_count),
            "epoch": int(epoch),
        }
        if self.branch == "background":
            eligibility = criterion_patch_eligibility(transformed.mask)
            eligible = np.asarray(eligibility.background_indices, dtype=np.int64)
            valid = len(eligible) >= int(self.token_budget)
            sample["background_valid"] = valid
            if valid:
                rng = stateless_rng(
                    self.background_seed,
                    epoch,
                    int(row.img_id),
                    "background_branch_train",
                )
                chosen = rng.choice(eligible, int(self.token_budget), replace=False)
                sample["source_indices"] = _tensor(chosen.astype(np.int64))
            else:
                sample["source_indices"] = _tensor(
                    np.full(int(self.token_budget), -1, dtype=np.int64)
                )
        return sample


class EvaluationDataset:
    """Deterministic ordinary, foreground, and mask geometry evaluation views."""

    def __init__(
        self,
        frame: pd.DataFrame,
        image_root: str | Path,
        mask_bank: VlmMaskBank,
        *,
        preprocessing: EvaluationPreprocessing,
    ) -> None:
        self.frame = frame.sort_values("img_id", kind="stable").reset_index(drop=True)
        self.image_root = Path(image_root)
        self.mask_bank = mask_bank
        self.preprocessing = preprocessing

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image = _load_image(image_path(self.image_root, str(row.img_filename)))
        mask = self.mask_bank.load(int(row.img_id), str(row.img_filename))
        transform_options = {
            "image_size": self.preprocessing.image_size,
            "resize_shortest": self.preprocessing.effective_resize_shortest,
        }
        ordinary = deterministic_eval_transform(image, mask, **transform_options)
        foreground = foreground_eval_transform(image, mask, **transform_options)
        eligibility = criterion_patch_eligibility(ordinary.mask)
        return {
            "image": _tensor(
                normalize_image(
                    ordinary.image,
                    mean=self.preprocessing.mean,
                    std=self.preprocessing.std,
                )
            ),
            "foreground_image": _tensor(
                normalize_image(
                    foreground.image,
                    mean=self.preprocessing.mean,
                    std=self.preprocessing.std,
                )
            ),
            "unnormalized_image": np.asarray(ordinary.image, dtype=np.uint8),
            "mask": _tensor(ordinary.mask.astype(bool)),
            "foreground_indices": _tensor(
                np.asarray(eligibility.foreground_indices, dtype=np.int64)
            ),
            "background_indices": _tensor(
                np.asarray(eligibility.background_indices, dtype=np.int64)
            ),
            "criterion_eligible": eligibility.eligible,
            "criterion_exclusion_reasons": eligibility.exclusion_reasons,
            "y": int(row.y),
            "img_id": int(row.img_id),
            "img_filename": str(row.img_filename),
        }


def collate_background_training(samples: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    valid = [sample for sample in samples if sample["background_valid"]]
    fallback_events = [
        {
            "img_id": int(sample["img_id"]),
            "epoch": int(sample["epoch"]),
            "attempt_count": int(sample["attempt_count"]),
            "fallback": True,
        }
        for sample in samples
        if sample["fallback"]
    ]
    if not valid:
        return {
            "empty": True,
            "fallback_count": sum(item["fallback"] for item in samples),
            "fallback_events": fallback_events,
            "sample_count": len(samples),
            "invalid_count": len(samples),
        }
    return {
        "empty": False,
        "image": torch.stack([sample["image"] for sample in valid]),
        "mask": torch.stack([sample["mask"] for sample in valid]),
        "source_indices": torch.stack([sample["source_indices"] for sample in valid]),
        "y": torch.tensor([sample["y"] for sample in valid], dtype=torch.long),
        "img_id": torch.tensor([sample["img_id"] for sample in valid], dtype=torch.long),
        "fallback_count": sum(item["fallback"] for item in samples),
        "fallback_events": fallback_events,
        "sample_count": len(samples),
        "invalid_count": len(samples) - len(valid),
    }


def collate_evaluation(samples: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    return {
        "image": torch.stack([sample["image"] for sample in samples]),
        "foreground_image": torch.stack([sample["foreground_image"] for sample in samples]),
        "unnormalized_image": [sample["unnormalized_image"] for sample in samples],
        "mask": torch.stack([sample["mask"] for sample in samples]),
        "foreground_indices": [sample["foreground_indices"] for sample in samples],
        "background_indices": [sample["background_indices"] for sample in samples],
        "criterion_eligible": torch.tensor(
            [sample["criterion_eligible"] for sample in samples], dtype=torch.bool
        ),
        "criterion_exclusion_reasons": [
            sample["criterion_exclusion_reasons"] for sample in samples
        ],
        "y": torch.tensor([sample["y"] for sample in samples], dtype=torch.long),
        "img_id": torch.tensor([sample["img_id"] for sample in samples], dtype=torch.long),
        "img_filename": [sample["img_filename"] for sample in samples],
    }
