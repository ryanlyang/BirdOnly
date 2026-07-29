from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.utils.seeds import derive_seed


class SeedTests(unittest.TestCase):
    def test_seed_derivation_is_stable_and_namespaced(self) -> None:
        self.assertEqual(derive_seed(1729, "split"), derive_seed(1729, "split"))
        self.assertNotEqual(derive_seed(1729, "split"), derive_seed(1729, "mask"))


if __name__ == "__main__":
    unittest.main()

