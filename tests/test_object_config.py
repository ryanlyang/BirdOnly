from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.errors import ConfigurationError
from setv.experts.config import load_object_expert_config


class ObjectConfigTests(unittest.TestCase):
    def test_seed_must_be_explicit(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "seed is not locked"):
            load_object_expert_config(ROOT / "configs" / "expert_object_green.yaml")

    def test_locked_config_accepts_explicit_seed(self) -> None:
        config = load_object_expert_config(
            ROOT / "configs" / "expert_object_green.yaml", seed=31415
        )
        self.assertEqual(config["training"]["seed"], 31415)
        self.assertEqual(config["training"]["epochs"], 20)
        self.assertFalse(config["storage"]["save_intermediate_checkpoints"])


if __name__ == "__main__":
    unittest.main()

