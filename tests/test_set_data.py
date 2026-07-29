from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.errors import DataValidationError
from setv.experts.set_config import load_set_expert_config
from setv.experts.set_data import (
    select_background_patch_tokens,
    select_training_background_view,
)


class SetDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_set_expert_config(
            ROOT / "configs" / "expert_background_set.yaml", seed=51
        )

    def test_patch_selection_cap_and_dropout_are_deterministic(self) -> None:
        mask = Image.new("L", (224, 224), 0)
        first, counts = select_background_patch_tokens(
            mask,
            self.config,
            selection_seed=1,
            dropout_seed=2,
            apply_dropout=True,
        )
        second, repeated = select_background_patch_tokens(
            mask,
            self.config,
            selection_seed=1,
            dropout_seed=2,
            apply_dropout=True,
        )
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(counts, repeated)
        self.assertEqual(counts["valid_before_cap"], 196)
        self.assertEqual(counts["valid_after_cap"], 180)
        self.assertEqual(counts["retained_after_dropout"], 144)

    def test_foreground_fraction_and_minimum_are_enforced(self) -> None:
        mask = Image.new("L", (224, 224), 255)
        with self.assertRaises(DataValidationError):
            select_background_patch_tokens(
                mask,
                self.config,
                selection_seed=1,
                dropout_seed=2,
                apply_dropout=False,
            )

        sparse = Image.new("L", (224, 224), 0)
        draw = ImageDraw.Draw(sparse)
        draw.rectangle((96, 96, 112, 112), fill=255)
        selected, counts = select_background_patch_tokens(
            sparse,
            self.config,
            selection_seed=4,
            dropout_seed=5,
            apply_dropout=False,
        )
        self.assertEqual(selected.dtype, np.bool_)
        self.assertGreaterEqual(counts["retained_after_dropout"], 16)
        self.assertLess(counts["valid_before_cap"], 196)

    def test_training_crop_retries_before_using_valid_view(self) -> None:
        image = Image.new("RGB", (224, 224), "white")
        source_mask = Image.new("L", (224, 224), 0)
        invalid_mask = Image.new("L", (224, 224), 255)
        valid_mask = Image.new("L", (224, 224), 0)

        def invalid_transform(unused_image, unused_mask):
            return image, invalid_mask

        def valid_transform(unused_image, unused_mask):
            return image, valid_mask

        with patch(
            "setv.experts.set_data.build_train_transform",
            side_effect=[invalid_transform, valid_transform],
        ):
            _, token_mask, counts, metadata = select_training_background_view(
                image,
                source_mask,
                {},
                self.config,
                sample_id="retry-fixture",
                epoch=3,
                base_seed=51,
                fallback_transform=valid_transform,
            )
        self.assertEqual(metadata["training_crop_attempt_count"], 2)
        self.assertEqual(metadata["training_crop_rejected_count"], 1)
        self.assertEqual(metadata["training_crop_fallback_used"], 0)
        self.assertGreaterEqual(counts["retained_after_dropout"], 16)
        self.assertEqual(int(token_mask.sum()), counts["retained_after_dropout"])

    def test_training_crop_uses_audited_fallback_after_retry_limit(self) -> None:
        config = {
            **self.config,
            "input": {
                **self.config["input"],
                "training_crop_max_attempts": 2,
            },
        }
        image = Image.new("RGB", (224, 224), "white")
        source_mask = Image.new("L", (224, 224), 0)
        invalid_mask = Image.new("L", (224, 224), 255)
        valid_mask = Image.new("L", (224, 224), 0)

        def invalid_transform(unused_image, unused_mask):
            return image, invalid_mask

        def valid_transform(unused_image, unused_mask):
            return image, valid_mask

        with patch(
            "setv.experts.set_data.build_train_transform",
            side_effect=[invalid_transform, invalid_transform],
        ):
            _, _, counts, metadata = select_training_background_view(
                image,
                source_mask,
                {},
                config,
                sample_id="fallback-fixture",
                epoch=4,
                base_seed=51,
                fallback_transform=valid_transform,
            )
        self.assertEqual(metadata["training_crop_attempt_count"], 2)
        self.assertEqual(metadata["training_crop_rejected_count"], 2)
        self.assertEqual(metadata["training_crop_fallback_used"], 1)
        self.assertGreaterEqual(counts["retained_after_dropout"], 16)

    def test_full_frame_padding_cannot_be_selected_as_background(self) -> None:
        from setv.data.joint_transforms import (
            JointFitLongestWithExcludedPadding,
        )

        image = Image.new("RGB", (40, 80), "white")
        source_mask = Image.new("L", image.size, 0)
        _, transformed_mask = JointFitLongestWithExcludedPadding(224)(
            image, source_mask
        )
        selected, counts = select_background_patch_tokens(
            transformed_mask,
            self.config,
            selection_seed=7,
            dropout_seed=8,
            apply_dropout=False,
        )
        selected_grid = selected.reshape(14, 14)
        self.assertFalse(selected_grid[:, :4].any())
        self.assertFalse(selected_grid[:, 10:].any())
        self.assertGreaterEqual(counts["retained_after_dropout"], 16)


if __name__ == "__main__":
    unittest.main()
