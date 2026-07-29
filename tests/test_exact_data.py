from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.experts.exact_data import dilate_binary_mask


class ExactDataTests(unittest.TestCase):
    def test_euclidean_disk_radius(self) -> None:
        mask = Image.new("L", (25, 25), 0)
        mask.putpixel((12, 12), 255)
        dilated = np.asarray(dilate_binary_mask(mask, 8)) > 0
        self.assertTrue(dilated[12, 20])
        self.assertTrue(dilated[20, 12])
        self.assertFalse(dilated[20, 20])
        self.assertEqual(int(dilated.sum()), 197)


if __name__ == "__main__":
    unittest.main()

