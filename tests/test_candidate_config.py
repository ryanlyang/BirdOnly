from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.candidate.config import load_candidate_config, validate_candidate_config
from setv.errors import ConfigurationError


class CandidateConfigTests(unittest.TestCase):
    def test_locked_fifty_epoch_configuration(self) -> None:
        config = load_candidate_config(
            ROOT / "configs" / "candidate_erm.yaml",
            seed=501,
            exact_fusion_dir="/tmp/exact",
            sanitized_fusion_dir="/tmp/sanitized",
            set_fusion_dir="/tmp/set",
        )
        self.assertEqual(config["training"]["epochs"], 50)
        self.assertFalse(config["storage"]["save_all_candidate_checkpoints"])
        self.assertTrue(config["storage"]["hide_test_until_selection_receipt"])
        changed = deepcopy(config)
        changed["training"]["warmup_epochs"] = 4
        with self.assertRaises(ConfigurationError):
            validate_candidate_config(changed)

    def test_all_three_fusion_paths_are_required(self) -> None:
        with self.assertRaises(ConfigurationError):
            load_candidate_config(
                ROOT / "configs" / "candidate_erm.yaml",
                seed=502,
                exact_fusion_dir="/tmp/exact",
            )


if __name__ == "__main__":
    unittest.main()
