from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.ula.analysis import _derive_expert_metadata


class Phase6ReadinessTests(unittest.TestCase):
    def test_secondary_metrics_are_derived_and_missing_values_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank = root / "bank"
            expert = root / "sanitized_expert"
            bank.mkdir()
            expert.mkdir()
            (bank / "leakage_audit.json").write_text(
                json.dumps(
                    {
                        "auditors": {
                            "linear": {
                                "heldout_mask_balanced_accuracy": 0.51
                            },
                            "cnn": {
                                "heldout_mask_balanced_accuracy": 0.53
                            },
                        }
                    }
                )
            )
            (bank / "sanitized_mask_bank_receipt.json").write_text(
                json.dumps(
                    {"leakage_audit": {"path": "leakage_audit.json"}}
                )
            )
            (expert / "phase3_sanitized_receipt.json").write_text(
                json.dumps({"mask_bank_dir": str(bank)})
            )
            inputs = {
                "exact": SimpleNamespace(
                    expert_dir=root / "exact",
                    background_scores={},
                ),
                "sanitized": SimpleNamespace(
                    expert_dir=expert,
                    background_scores={
                        "background_sanitized_margin_std": np.asarray(
                            [0.1, 0.3]
                        )
                    },
                ),
                "set": SimpleNamespace(
                    expert_dir=root / "set",
                    background_scores={
                        "background_set_margin_std": np.asarray([0.2, 0.4])
                    },
                ),
            }
            config = {
                "expert_metadata": {
                    "simplicity_rank": {
                        "exact": 1,
                        "sanitized": 2,
                        "set": 3,
                    }
                }
            }
            result = _derive_expert_metadata(inputs, config)
            self.assertIsNone(result["exact"]["leakage_balanced_accuracy"])
            self.assertIsNotNone(
                result["exact"]["leakage_unavailable_reason"]
            )
            self.assertAlmostEqual(
                result["sanitized"]["leakage_balanced_accuracy"], 0.53
            )
            self.assertAlmostEqual(
                result["sanitized"]["view_margin_std"], 0.2
            )
            self.assertAlmostEqual(result["set"]["view_margin_std"], 0.3)
            self.assertIsNone(result["set"]["leakage_balanced_accuracy"])

    def test_phase0_launchers_fail_closed_on_git_provenance(self):
        submit = (ROOT / "scripts" / "submit_phase0.sh").read_text()
        runtime = (ROOT / "slurm" / "phase0_preflight.sbatch").read_text()
        self.assertIn("status --porcelain", submit)
        self.assertIn("SETV_EXPECTED_COMMIT", submit)
        self.assertIn("SETV_EXPECTED_COMMIT", runtime)
        self.assertIn("worktree became dirty", runtime)


if __name__ == "__main__":
    unittest.main()
