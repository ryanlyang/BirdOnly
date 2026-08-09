from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from anchorcal.errors import AuditFailure
from anchorcal.transforms import (
    Geometry,
    check_fallback_rate,
    criterion_patch_eligibility,
    deterministic_eval_transform,
    evaluation_resize_shortest,
    final_coordinate_dilated_mask,
    foreground_eval_transform,
    gaussian_kernel_1d,
    mask_normalized_background_blur,
    normalize_image,
    stateless_train_transform,
)


class AnchorCalTransformTests(unittest.TestCase):
    @staticmethod
    def _source_pair(background_rgb: tuple[int, int, int]) -> tuple[Image.Image, np.ndarray]:
        height, width = 96, 128
        array = np.empty((height, width, 3), dtype=np.uint8)
        array[...] = np.asarray(background_rgb, dtype=np.uint8)
        mask = np.zeros((height, width), dtype=bool)
        mask[20:78, 38:92] = True
        # Foreground content is identical regardless of hidden background.
        yy, xx = np.indices((height, width))
        foreground_pattern = np.stack(
            [
                (3 * xx + yy) % 256,
                (xx + 5 * yy) % 256,
                (2 * xx + 7 * yy) % 256,
            ],
            axis=-1,
        ).astype(np.uint8)
        array[mask] = foreground_pattern[mask]
        return Image.fromarray(array, mode="RGB"), mask

    def test_source_green_screen_prevents_hidden_background_edge_leak(self) -> None:
        first, mask = self._source_pair((255, 0, 0))
        second, repeated_mask = self._source_pair((0, 0, 255))
        first_view = foreground_eval_transform(first, mask)
        second_view = foreground_eval_transform(second, repeated_mask)
        np.testing.assert_array_equal(
            np.asarray(first_view.image), np.asarray(second_view.image)
        )
        np.testing.assert_array_equal(first_view.mask, second_view.mask)

    def test_stateless_crop_and_flip_replay_exactly(self) -> None:
        image, mask = self._source_pair((30, 60, 90))
        arguments = {
            "run_seed": 6001,
            "epoch": 7,
            "img_id": 412,
            "purpose": "foreground_branch_train",
            "require_visible_foreground": True,
        }
        first = stateless_train_transform(image, mask, **arguments)
        second = stateless_train_transform(image, mask, **arguments)
        self.assertEqual(first.geometry, second.geometry)
        self.assertEqual(first.attempt_count, second.attempt_count)
        self.assertEqual(first.fallback_used, second.fallback_used)
        np.testing.assert_array_equal(np.asarray(first.image), np.asarray(second.image))
        np.testing.assert_array_equal(first.mask, second.mask)

    def test_eval_geometry_image_interpolation_and_binary_mask(self) -> None:
        image, mask = self._source_pair((20, 40, 80))
        result = deterministic_eval_transform(image, mask)
        self.assertEqual(evaluation_resize_shortest(), 248)
        self.assertEqual(result.image.size, (224, 224))
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.mask.shape, (224, 224))
        self.assertEqual(result.mask.dtype, np.bool_)
        self.assertEqual(set(np.unique(result.mask).tolist()), {False, True})
        normalized = normalize_image(result.image)
        self.assertEqual(normalized.shape, (3, 224, 224))
        self.assertEqual(normalized.dtype, np.float32)

    def test_empty_random_crops_use_deterministic_fallback_and_gate(self) -> None:
        array = np.full((100, 100, 3), 120, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=bool)
        mask[45:55, 45:55] = True
        image = Image.fromarray(array, mode="RGB")
        corner = Geometry(0, 0, 20, 20, 224, False)
        with patch(
            "anchorcal.transforms._random_resized_crop_geometry",
            return_value=corner,
        ) as generator:
            result = stateless_train_transform(
                image,
                mask,
                run_seed=6001,
                epoch=2,
                img_id="fallback",
                purpose="foreground_branch_train",
                require_visible_foreground=True,
                resize_shortest=260,
            )
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.attempt_count, 10)
        self.assertEqual(generator.call_count, 10)
        self.assertTrue(result.mask.any())
        self.assertEqual(result.geometry.kind, "evaluation_center_crop_after_resize")
        locked_fallback = deterministic_eval_transform(
            image, mask, resize_shortest=260
        )
        np.testing.assert_array_equal(
            np.asarray(result.image), np.asarray(locked_fallback.image)
        )
        self.assertEqual(check_fallback_rate(1, 1000), 0.001)
        with self.assertRaises(AuditFailure):
            check_fallback_rate(2, 1000)

    def test_final_disk_dilation_and_pure_patch_eligibility(self) -> None:
        mask = np.zeros((224, 224), dtype=bool)
        mask[96:128, 96:128] = True
        dilated = final_coordinate_dilated_mask(mask)
        self.assertTrue(dilated[95, 96])
        self.assertTrue(dilated[88, 96])
        self.assertFalse(dilated[87, 96])
        eligibility = criterion_patch_eligibility(mask)
        self.assertTrue(eligibility.eligible)
        self.assertGreater(len(eligibility.foreground_indices), 0)
        self.assertGreater(len(eligibility.background_indices), 0)

    def test_mask_normalized_blur_has_no_foreground_color_bleed(self) -> None:
        height = width = 64
        mask = np.zeros((height, width), dtype=bool)
        mask[20:44, 20:44] = True
        first = np.full((height, width, 3), 0.25, dtype=np.float32)
        second = first.copy()
        first[mask] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        second[mask] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        first_blur = mask_normalized_background_blur(first, mask, sigma=4)
        second_blur = mask_normalized_background_blur(second, mask, sigma=4)
        np.testing.assert_allclose(
            first_blur[~mask], second_blur[~mask], atol=1e-7, rtol=0
        )
        np.testing.assert_allclose(first_blur[mask], first[mask], atol=0, rtol=0)
        np.testing.assert_allclose(second_blur[mask], second[mask], atol=0, rtol=0)
        np.testing.assert_allclose(first_blur[~mask], 0.25, atol=1e-6, rtol=0)
        self.assertEqual(len(gaussian_kernel_1d(4)), 25)


if __name__ == "__main__":
    unittest.main()
