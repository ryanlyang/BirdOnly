from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.errors import ConfigurationError
from setv.experts.exact_config import load_exact_expert_config


class ExactConfigTests(unittest.TestCase):
    def test_seed_is_explicit_and_dilation_is_locked(self) -> None:
        path = ROOT / "configs" / "expert_background_exact.yaml"
        with self.assertRaisesRegex(ConfigurationError, "seed is not locked"):
            load_exact_expert_config(path)
        config = load_exact_expert_config(path, seed=42)
        self.assertEqual(config["input"]["dilation_pixels_at_224"], 8)
        self.assertEqual(
            config["input"]["dilation_structuring_element"], "euclidean_disk"
        )
        self.assertFalse(config["storage"]["save_intermediate_checkpoints"])


if __name__ == "__main__":
    unittest.main()
