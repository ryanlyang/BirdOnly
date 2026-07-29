from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

from fixture_data import create_fixture, fixture_config
from setv.experts.config import load_object_expert_config
from setv.experts.sanitized_bank import (
    build_sanitized_mask_bank,
    verify_sanitized_mask_bank,
)
from setv.experts.sanitized_config import (
    load_sanitized_bank_config,
    load_sanitized_expert_config,
)
from setv.experts.train_object import train_object_expert
from setv.experts.train_sanitized import (
    run_sanitized_expert_smoke,
    train_sanitized_expert,
    verify_sanitized_expert,
)
from setv.fusion.artifacts import load_fusion_artifacts
from setv.fusion.sanitized_artifacts import (
    build_sanitized_fusion_artifacts,
    score_candidate_sanitized_file,
    verify_sanitized_fusion_artifacts,
)
from setv.fusion.sanitized_config import load_sanitized_fusion_config
from setv.phase0 import approve_visual_audit, build_phase0


def accepted_fixture_audit(
    packed_masks, family_ids, width, sample_ids, labels, *, seed, config
):
    heldout = np.array([0, 1], dtype=np.int64)
    return (
        {
            "accepted": True,
            "acceptance_rule": "fixture auditor injection",
            "split_unit": "sample_id/image",
            "train_image_count": len(labels) - 2,
            "heldout_image_count": 2,
            "feature_names": [],
            "auditors": {},
        },
        {
            "train_image_index": np.arange(2, len(labels), dtype=np.int64),
            "heldout_image_index": heldout,
            "heldout_sample_id": np.asarray(sample_ids)[heldout].astype(np.str_),
            "heldout_true_label": np.asarray(labels)[heldout],
            "heldout_family_id": np.asarray(family_ids)[heldout],
        },
    )


if TORCH_AVAILABLE:

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 4, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.head = torch.nn.Linear(4, 2)

        def forward(self, inputs):
            return self.head(self.features(inputs).flatten(1))


@unittest.skipUnless(TORCH_AVAILABLE, "torch unavailable")
class Phase3IntegrationTests(unittest.TestCase):
    def test_bank_training_fusion_and_candidate_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, masks = create_fixture(root)
            phase0 = root / "phase0"
            build_phase0(fixture_config(dataset, masks, phase0))
            approve_visual_audit(phase0, reviewer="phase3 fixture", confirmation=True)
            bank_config = load_sanitized_bank_config(
                ROOT / "configs" / "sanitized_mask_bank.yaml",
                seed=301,
                phase0_dir=str(phase0),
                output_root=str(root / "banks"),
                auditor_device="cpu",
            )
            bank_dir = build_sanitized_mask_bank(
                bank_config, auditor_runner=accepted_fixture_audit
            )
            self.assertTrue(verify_sanitized_mask_bank(bank_dir)["accepted"])

            object_config = load_object_expert_config(
                ROOT / "configs" / "expert_object_green.yaml",
                seed=302,
                phase0_dir=str(phase0),
                output_root=str(root / "object"),
                device="cpu",
            )
            object_config["training"].update(
                {"num_workers": 0, "evaluation_batch_size": 8, "mixed_precision": False}
            )
            object_dir = train_object_expert(
                object_config, model_factory=lambda unused: TinyModel()
            )

            expert_config = load_sanitized_expert_config(
                ROOT / "configs" / "expert_background_sanitized.yaml",
                seed=303,
                phase0_dir=str(phase0),
                mask_bank_dir=str(bank_dir),
                output_root=str(root / "sanitized"),
                device="cpu",
            )
            expert_config["training"].update(
                {"num_workers": 0, "evaluation_batch_size": 4, "mixed_precision": False}
            )
            smoke = run_sanitized_expert_smoke(
                expert_config,
                root / "sanitized_smoke.json",
                model_factory=lambda unused: TinyModel(),
            )
            self.assertEqual(smoke["status"], "passed")
            sanitized_dir = train_sanitized_expert(
                expert_config, model_factory=lambda unused: TinyModel()
            )
            self.assertEqual(
                verify_sanitized_expert(sanitized_dir)["status"], "complete"
            )

            fusion_config = load_sanitized_fusion_config(
                ROOT / "configs" / "fusion_sanitized.yaml",
                seed=304,
                phase0_dir=str(phase0),
                object_expert_dir=str(object_dir),
                sanitized_expert_dir=str(sanitized_dir),
                output_root=str(root / "fusion"),
                allow_expert_sanity_warnings=True,
            )
            fusion_dir = build_sanitized_fusion_artifacts(fusion_config)
            verified = verify_sanitized_fusion_artifacts(fusion_dir)
            self.assertEqual(verified["status"], "complete")
            fusion = load_fusion_artifacts(fusion_dir)
            candidate_path = root / "candidate.npz"
            logits = np.zeros((len(fusion["true_label"]), 2), dtype=np.float32)
            logits[np.arange(len(logits)), fusion["true_label"]] = 2
            np.savez_compressed(
                candidate_path,
                sample_id=fusion["sample_id"],
                true_label=fusion["true_label"],
                candidate_logits=logits,
            )
            scores = score_candidate_sanitized_file(fusion_dir, candidate_path)
            self.assertAlmostEqual(scores["rank"]["setv_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
