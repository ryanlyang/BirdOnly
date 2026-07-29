from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.model_contracts import (
    VIT_SMALL_PATCH16_224_MEAN,
    VIT_SMALL_PATCH16_224_STD,
    vit_small_normalization_matches,
)


class ModelContractTests(unittest.TestCase):
    def test_every_vit_config_uses_pretrained_metadata_normalization(self) -> None:
        paths = (
            "expert_object_green.yaml",
            "expert_background_exact.yaml",
            "expert_background_sanitized.yaml",
            "expert_background_set.yaml",
            "candidate_erm.yaml",
        )
        for name in paths:
            with self.subTest(config=name):
                config = yaml.safe_load((ROOT / "configs" / name).read_text())
                self.assertTrue(vit_small_normalization_matches(config["model"]))
                self.assertEqual(
                    config["model"]["normalization_mean"],
                    VIT_SMALL_PATCH16_224_MEAN,
                )
                self.assertEqual(
                    config["model"]["normalization_std"],
                    VIT_SMALL_PATCH16_224_STD,
                )


if __name__ == "__main__":
    unittest.main()
