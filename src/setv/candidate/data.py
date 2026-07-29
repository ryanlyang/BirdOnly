"""Original-image datasets for the Phase 5 ERM candidate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from setv.data.dataset import FORBIDDEN_SELECTOR_COLUMNS
from setv.data.joint_transforms import build_eval_transform, build_train_transform
from setv.errors import DataValidationError
from setv.experts.object_data import normalized_tensor
from setv.utils.seeds import derive_seed


class CandidateDataset:
    """Load only selector-safe labels and original RGB images."""

    def __init__(
        self,
        manifest_csv: str | Path,
        phase0_config: dict[str, Any],
        candidate_config: dict[str, Any],
        *,
        training: bool,
    ):
        self.rows = pd.read_csv(manifest_csv, dtype={"sample_id": str})
        forbidden = FORBIDDEN_SELECTOR_COLUMNS.intersection(self.rows.columns)
        if forbidden:
            raise DataValidationError(
                f"Candidate manifest exposes protected columns: {sorted(forbidden)}"
            )
        required = {"sample_id", "img_filename", "y"}
        if required - set(self.rows):
            raise DataValidationError("Candidate manifest is missing safe columns")
        self.dataset_root = Path(phase0_config["data"]["dataset_root"])
        self.phase0_config = phase0_config
        self.training = training
        self.seed = int(candidate_config["training"]["seed"])
        self.epoch = 0
        self.mean = tuple(candidate_config["model"]["normalization_mean"])
        self.std = tuple(candidate_config["model"]["normalization_std"])
        self.eval_transform = build_eval_transform(phase0_config)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        with Image.open(self.dataset_root / str(row["img_filename"])) as opened:
            image = opened.convert("RGB")
        # Joint transforms are reused with a blank geometry carrier. The mask
        # is discarded and never shown to the candidate.
        blank = Image.new("L", image.size, 0)
        if self.training:
            transform = build_train_transform(
                self.phase0_config,
                derive_seed(
                    self.seed,
                    f"candidate_transform:epoch={self.epoch}:sample={row['sample_id']}",
                ),
            )
        else:
            transform = self.eval_transform
        image, _ = transform(image, blank)
        return {
            "image": normalized_tensor(image, self.mean, self.std),
            "target": int(row["y"]),
            "sample_id": str(row["sample_id"]),
        }
