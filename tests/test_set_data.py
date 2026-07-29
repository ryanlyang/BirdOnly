from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.errors import DataValidationError
from setv.experts.set_config import load_set_expert_config
from setv.experts.set_data import select_background_patch_tokens


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


if __name__ == "__main__":
    unittest.main()
