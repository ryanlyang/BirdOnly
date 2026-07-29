from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.errors import ConfigurationError
from setv.experts.set_config import load_set_expert_config, validate_set_expert_config
from setv.fusion.set_config import load_set_fusion_config


class SetConfigTests(unittest.TestCase):
    def test_locked_expert_config_and_explicit_seed(self) -> None:
        config = load_set_expert_config(
            ROOT / "configs" / "expert_background_set.yaml", seed=41
        )
        self.assertEqual(config["training"]["seed"], 41)
        self.assertFalse(config["input"]["use_dense_position_embeddings"])
        self.assertEqual(config["training"]["validation_views"], 8)
        changed = deepcopy(config)
        changed["input"]["maximum_foreground_fraction"] = 0.02
        with self.assertRaises(ConfigurationError):
            validate_set_expert_config(changed)

    def test_fusion_config_uses_set_expert_path(self) -> None:
        config = load_set_fusion_config(
            ROOT / "configs" / "fusion_set.yaml",
            seed=43,
            object_expert_dir="/tmp/object",
            set_expert_dir="/tmp/set",
        )
        self.assertEqual(config["phase"], "set_fusion")
        self.assertEqual(config["fusion"]["logistic"]["n_repeats"], 5)
        self.assertEqual(config["set_expert_dir"], "/tmp/set")


if __name__ == "__main__":
    unittest.main()
