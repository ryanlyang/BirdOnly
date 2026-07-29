from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.data.masks import MaskResolver, inspect_mask
from setv.errors import DataValidationError


class MaskTests(unittest.TestCase):
    def test_unique_basename_fallback_and_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mask_root = root / "masks"
            mask_root.mkdir()
            mask = Image.new("L", (20, 20), 0)
            for x in range(5, 15):
                for y in range(5, 15):
                    mask.putpixel((x, y), 255)
            mask.save(mask_root / "example.png")
            image = Image.new("RGB", (20, 20), (10, 20, 30))
            image_path = root / "example.jpg"
            image.save(image_path)

            resolver = MaskResolver(
                mask_root,
                [".png"],
                "relative_stem_then_unique_basename",
            )
            resolved = resolver.resolve("images/example.jpg", "123")
            self.assertEqual(resolved.mapping_rule, "unique_basename")
            details = inspect_mask(
                image_path,
                resolved.path,
                threshold_normalized=0.5,
                foreground_is_high=True,
                require_same_dimensions=True,
                minimum_foreground_fraction=0.001,
                maximum_foreground_fraction=0.95,
            )
            self.assertAlmostEqual(details["foreground_fraction"], 0.25)
            self.assertTrue(details["source_mask_binary"])

    def test_ambiguous_basename_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in (root / "a", root / "b"):
                directory.mkdir(parents=True)
                Image.new("L", (4, 4), 255).save(directory / "same.png")
            resolver = MaskResolver(root, [".png"], "unique_basename")
            with self.assertRaisesRegex(DataValidationError, "Ambiguous"):
                resolver.resolve("images/same.jpg", "0")

    def test_missing_mask_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("L", (4, 4), 255).save(root / "other.png")
            resolver = MaskResolver(root, [".png"], "unique_basename")
            with self.assertRaisesRegex(DataValidationError, "No VLM mask"):
                resolver.resolve("images/missing.jpg", "0")

    def test_waterbirds_weclip_flattened_relative_stem(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mask_name = (
                "001_Black_footed_Albatross_"
                "Black_Footed_Albatross_0046_18.png"
            )
            Image.new("L", (4, 4), 255).save(root / mask_name)
            resolver = MaskResolver(
                root,
                [".png"],
                "weclip_flattened_relative_stem",
            )
            resolved = resolver.resolve(
                (
                    "001.Black_footed_Albatross/"
                    "Black_Footed_Albatross_0046_18.jpg"
                ),
                "1",
            )
            self.assertEqual(
                resolved.mapping_rule,
                "weclip_flattened_relative_stem",
            )
            self.assertEqual(resolved.relative_path, mask_name)
            self.assertEqual(
                MaskResolver.weclip_flattened_stem(
                    r"001.Black_footed_Albatross\Black_Footed_Albatross_0046_18.jpg"
                ),
                (
                    "001_Black_footed_Albatross_"
                    "Black_Footed_Albatross_0046_18"
                ),
            )

    def test_pascal_colorized_bird_label_uses_exact_voc_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "bird.jpg"
            mask_path = root / "bird.png"
            Image.new("RGB", (4, 4), (20, 30, 40)).save(image_path)
            mask = Image.new("RGB", (4, 4), (0, 0, 0))
            for x in range(2):
                for y in range(2):
                    mask.putpixel((x, y), (128, 0, 0))
            mask.save(mask_path)
            details = inspect_mask(
                image_path,
                mask_path,
                threshold_normalized=1.0 / 255.0,
                foreground_is_high=True,
                require_same_dimensions=True,
                minimum_foreground_fraction=0.001,
                maximum_foreground_fraction=0.95,
                map_format="voc_colormap_class_ids",
                foreground_class_ids=(1,),
            )
            self.assertAlmostEqual(details["foreground_fraction"], 0.25)
            self.assertFalse(details["source_mask_binary"])
            self.assertEqual(
                details["source_rgb_color_counts"],
                {"0,0,0": 12, "128,0,0": 4},
            )

    def test_pascal_decoder_rejects_binary_white_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "bird.jpg"
            mask_path = root / "bird.png"
            Image.new("RGB", (4, 4), (20, 30, 40)).save(image_path)
            Image.new("L", (4, 4), 255).save(mask_path)
            with self.assertRaisesRegex(DataValidationError, "Unexpected VOC colors"):
                inspect_mask(
                    image_path,
                    mask_path,
                    threshold_normalized=1.0 / 255.0,
                    foreground_is_high=True,
                    require_same_dimensions=True,
                    minimum_foreground_fraction=0.001,
                    maximum_foreground_fraction=0.95,
                    map_format="voc_colormap_class_ids",
                    foreground_class_ids=(1,),
                )


if __name__ == "__main__":
    unittest.main()
