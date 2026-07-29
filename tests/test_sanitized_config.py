from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.errors import ConfigurationError
from setv.experts.sanitized_config import (
    load_sanitized_bank_config,
    load_sanitized_expert_config,
)


class SanitizedConfigTests(unittest.TestCase):
    def test_bank_and_expert_seeds_and_paths_are_explicit(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "seed must be explicit"):
            load_sanitized_bank_config(ROOT / "configs" / "sanitized_mask_bank.yaml")
        bank = load_sanitized_bank_config(
            ROOT / "configs" / "sanitized_mask_bank.yaml", seed=11
        )
        self.assertEqual(bank["mask_bank"]["masks_per_image"], 8)
        with self.assertRaisesRegex(ConfigurationError, "seed must be explicit"):
            load_sanitized_expert_config(
                ROOT / "configs" / "expert_background_sanitized.yaml",
                mask_bank_dir="/tmp/bank",
            )
        expert = load_sanitized_expert_config(
            ROOT / "configs" / "expert_background_sanitized.yaml",
            seed=12,
            mask_bank_dir="/tmp/bank",
        )
        self.assertEqual(expert["training"]["lambda_consistency"], 0.5)
        self.assertEqual(expert["input"]["validation_masks_per_image"], 8)


if __name__ == "__main__":
    unittest.main()

