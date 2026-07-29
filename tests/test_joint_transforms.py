from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.data.joint_transforms import (
    JointCompose,
    JointFitLongestWithExcludedPadding,
    JointRandomHorizontalFlip,
    JointRandomResizedCrop,
    binarize_mask,
    green_fill,
)


class JointTransformTests(unittest.TestCase):
    def _pair(self) -> tuple[Image.Image, Image.Image]:
        image = Image.new("RGB", (64, 48), (0, 0, 0))
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(image).rectangle((8, 10, 28, 34), fill=(255, 255, 255))
        ImageDraw.Draw(mask).rectangle((8, 10, 28, 34), fill=255)
        return image, mask

    def test_horizontal_flip_uses_same_geometry(self) -> None:
        image, mask = self._pair()
        transformed_image, transformed_mask = JointRandomHorizontalFlip(
            probability=1.0, rng=random.Random(4)
        )(image, mask)
        bright = np.asarray(transformed_image)[..., 0] > 127
        foreground = np.asarray(transformed_mask) > 0
        np.testing.assert_array_equal(bright, foreground)

    def test_random_crop_keeps_alignment_and_binary_mask(self) -> None:
        image, mask = self._pair()
        transform = JointCompose(
            [
                JointRandomResizedCrop(32, rng=random.Random(1729)),
                JointRandomHorizontalFlip(0.5, rng=random.Random(1729)),
            ]
        )
        transformed_image, transformed_mask = transform(image, mask)
        self.assertEqual(transformed_image.size, (32, 32))
        self.assertEqual(transformed_mask.size, (32, 32))
        self.assertTrue(set(np.unique(transformed_mask)).issubset({0, 255}))

    def test_green_composition_occurs_in_raw_rgb(self) -> None:
        image, mask = self._pair()
        binary = binarize_mask(mask)
        object_view = np.asarray(green_fill(image, binary, keep_foreground=True))
        background = np.asarray(binary) == 0
        self.assertTrue(np.all(object_view[background] == np.array([0, 255, 0])))

    def test_full_frame_fit_marks_padding_ineligible(self) -> None:
        image = Image.new("RGB", (40, 80), (12, 34, 56))
        mask = Image.new("L", image.size, 0)
        transformed_image, transformed_mask = (
            JointFitLongestWithExcludedPadding(224)(image, mask)
        )
        self.assertEqual(transformed_image.size, (224, 224))
        mask_array = np.asarray(transformed_mask)
        self.assertTrue(np.all(mask_array[:, :56] == 255))
        self.assertTrue(np.all(mask_array[:, 56:168] == 0))
        self.assertTrue(np.all(mask_array[:, 168:] == 255))


if __name__ == "__main__":
    unittest.main()
