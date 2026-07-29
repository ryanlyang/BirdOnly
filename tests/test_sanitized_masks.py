from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.experts.exact_data import dilate_binary_mask
from setv.experts.sanitized_config import load_sanitized_bank_config
from setv.experts.sanitized_masks import (
    FAMILIES,
    generate_sanitized_bank,
    globally_balanced_short_families,
    pack_masks,
    unpack_masks,
)


class SanitizedMaskTests(unittest.TestCase):
    def test_global_allocation_and_all_families_contain_dilated_mask(self) -> None:
        ids = [f"sample-{index}" for index in range(31)]
        assignment = globally_balanced_short_families(ids)
        counts = [sum(value == family for value in assignment.values()) for family in FAMILIES]
        self.assertLessEqual(max(counts) - min(counts), 1)
        config = load_sanitized_bank_config(
            ROOT / "configs" / "sanitized_mask_bank.yaml", seed=123
        )
        source = Image.new("L", (64, 64), 0)
        draw = ImageDraw.Draw(source)
        draw.ellipse((23, 18, 41, 45), fill=255)
        first = generate_sanitized_bank(
            source,
            sample_id=ids[0],
            short_family=assignment[ids[0]],
            base_seed=123,
            config=config["mask_bank"],
            dilation_radius=3,
        )
        second = generate_sanitized_bank(
            source,
            sample_id=ids[0],
            short_family=assignment[ids[0]],
            base_seed=123,
            config=config["mask_bank"],
            dilation_radius=3,
        )
        masks, family_ids, seeds, _ = first
        np.testing.assert_array_equal(masks, second[0])
        np.testing.assert_array_equal(seeds, second[2])
        self.assertEqual(sorted(np.bincount(family_ids, minlength=3).tolist()), [2, 3, 3])
        dilated = np.asarray(dilate_binary_mask(source, 3)) > 0
        self.assertTrue(np.all(masks[:, dilated]))
        np.testing.assert_array_equal(unpack_masks(pack_masks(masks[None]), 64)[0], masks)


if __name__ == "__main__":
    unittest.main()

