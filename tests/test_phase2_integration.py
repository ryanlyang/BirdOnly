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
from setv.experts.exact_config import load_exact_expert_config
from setv.experts.train_exact import (
    run_exact_expert_smoke,
    train_exact_expert,
    verify_exact_expert,
)
from setv.experts.train_object import train_object_expert
from setv.fusion.artifacts import (
    build_exact_fusion_artifacts,
    load_fusion_artifacts,
    score_candidate_file,
    verify_fusion_artifacts,
)
from setv.fusion.config import load_fusion_config
from setv.phase0 import approve_visual_audit, build_phase0


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
class Phase2IntegrationTests(unittest.TestCase):
    def test_exact_training_and_fusion_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, masks = create_fixture(root)
            phase0 = root / "phase0"
            build_phase0(fixture_config(dataset, masks, phase0))
            approve_visual_audit(phase0, reviewer="phase2 fixture", confirmation=True)

            object_config = load_object_expert_config(
                ROOT / "configs" / "expert_object_green.yaml",
                seed=101,
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

            exact_config = load_exact_expert_config(
                ROOT / "configs" / "expert_background_exact.yaml",
                seed=202,
                phase0_dir=str(phase0),
                output_root=str(root / "exact"),
                device="cpu",
            )
            exact_config["training"].update(
                {"num_workers": 0, "evaluation_batch_size": 8, "mixed_precision": False}
            )
            smoke = run_exact_expert_smoke(
                exact_config,
                root / "exact_smoke.json",
                model_factory=lambda unused: TinyModel(),
            )
            self.assertEqual(smoke["status"], "passed")
            self.assertGreater(smoke["train"]["train_optimizer_step_count"], 0)
            self.assertEqual(smoke["train"]["train_amp_skipped_step_count"], 0)
            exact_dir = train_exact_expert(
                exact_config, model_factory=lambda unused: TinyModel()
            )
            self.assertEqual(verify_exact_expert(exact_dir)["status"], "complete")

            fusion_config = load_fusion_config(
                ROOT / "configs" / "fusion_exact.yaml",
                seed=303,
                phase0_dir=str(phase0),
                object_expert_dir=str(object_dir),
                exact_expert_dir=str(exact_dir),
                output_root=str(root / "fusion"),
                allow_expert_sanity_warnings=True,
            )
            fusion_dir = build_exact_fusion_artifacts(fusion_config)
            verified = verify_fusion_artifacts(fusion_dir)
            self.assertEqual(verified["status"], "complete")
            self.assertEqual(verified["sample_count"], 8)
            fusion = load_fusion_artifacts(fusion_dir)
            candidate_path = root / "candidate_logits.npz"
            candidate_logits = np.zeros((8, 2), dtype=np.float32)
            candidate_logits[np.arange(8), fusion["true_label"]] = 2.0
            np.savez_compressed(
                candidate_path,
                sample_id=fusion["sample_id"],
                true_label=fusion["true_label"],
                candidate_logits=candidate_logits,
            )
            candidate_scores = score_candidate_file(fusion_dir, candidate_path)
            self.assertAlmostEqual(candidate_scores["rank"]["setv_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
