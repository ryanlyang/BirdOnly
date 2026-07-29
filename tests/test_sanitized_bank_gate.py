from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from fixture_data import create_fixture, fixture_config
from setv.errors import DataValidationError
from setv.experts.sanitized_bank import (
    build_sanitized_mask_bank,
    verify_sanitized_mask_bank,
)
from setv.experts.sanitized_config import load_sanitized_bank_config
from setv.phase0 import approve_visual_audit, build_phase0


def rejected_audit(
    packed_masks, family_ids, width, sample_ids, labels, *, seed, config
):
    return (
        {
            "accepted": False,
            "acceptance_rule": "fixture rejection",
            "split_unit": "sample_id/image",
            "train_image_count": len(labels) - 2,
            "heldout_image_count": 2,
            "feature_names": [],
            "auditors": {
                "fixture": {
                    "heldout_mask_balanced_accuracy": 0.75,
                    "cluster_bootstrap_confidence_interval": [0.65, 0.85],
                    "accepted": False,
                }
            },
        },
        {
            "train_image_index": np.arange(2, len(labels), dtype=np.int64),
            "heldout_image_index": np.asarray([0, 1], dtype=np.int64),
        },
    )


class SanitizedBankGateTests(unittest.TestCase):
    def test_rejected_bank_is_retained_but_cannot_train(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, masks = create_fixture(root)
            phase0 = root / "phase0"
            build_phase0(fixture_config(dataset, masks, phase0))
            approve_visual_audit(phase0, reviewer="bank gate fixture", confirmation=True)
            config = load_sanitized_bank_config(
                ROOT / "configs" / "sanitized_mask_bank.yaml",
                seed=808,
                phase0_dir=str(phase0),
                output_root=str(root / "banks"),
                auditor_device="cpu",
            )
            bank = build_sanitized_mask_bank(config, auditor_runner=rejected_audit)
            result = verify_sanitized_mask_bank(
                bank, require_accepted=False, verify_containment=True
            )
            self.assertEqual(result["status"], "rejected")
            with self.assertRaisesRegex(DataValidationError, "leakage gate"):
                verify_sanitized_mask_bank(
                    bank, require_accepted=True, verify_containment=False
                )


if __name__ == "__main__":
    unittest.main()
