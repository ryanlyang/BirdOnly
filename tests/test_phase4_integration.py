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
from setv.experts.set_config import load_set_expert_config
from setv.experts.train_object import train_object_expert
from setv.experts.train_set import (
    run_set_expert_smoke,
    train_set_expert,
    verify_set_expert,
)
from setv.fusion.artifacts import load_fusion_artifacts
from setv.fusion.set_artifacts import (
    build_set_fusion_artifacts,
    score_candidate_set_file,
    verify_set_fusion_artifacts,
)
from setv.fusion.set_config import load_set_fusion_config
from setv.phase0 import approve_visual_audit, build_phase0


if TORCH_AVAILABLE:

    class TinyImageModel(torch.nn.Module):
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


    class TinySetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 4, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.head = torch.nn.Linear(5, 2)
            self.initialization_report = {
                "available": True,
                "fixture": True,
                "dense_position_embeddings_discarded": True,
                "second_attention_pooling_module": False,
            }

        def forward(self, inputs, token_mask):
            image_features = self.features(inputs).flatten(1)
            if token_mask.ndim == 2:
                fraction = token_mask.float().mean(dim=1, keepdim=True)
                return self.head(torch.cat([image_features, fraction], dim=1))
            batch, views, _ = token_mask.shape
            repeated = image_features[:, None, :].expand(batch, views, 4)
            fraction = token_mask.float().mean(dim=2, keepdim=True)
            return self.head(torch.cat([repeated, fraction], dim=2))


@unittest.skipUnless(TORCH_AVAILABLE, "torch unavailable")
class Phase4IntegrationTests(unittest.TestCase):
    def test_set_training_fusion_and_candidate_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, masks = create_fixture(root)
            phase0 = root / "phase0"
            phase0_config = fixture_config(dataset, masks, phase0)
            phase0_config["transforms"].update(
                {
                    "image_size": 96,
                    "evaluation_resize_shortest": 112,
                }
            )
            build_phase0(phase0_config)
            approve_visual_audit(phase0, reviewer="phase4 fixture", confirmation=True)

            object_config = load_object_expert_config(
                ROOT / "configs" / "expert_object_green.yaml",
                seed=401,
                phase0_dir=str(phase0),
                output_root=str(root / "object"),
                device="cpu",
            )
            object_config["training"].update(
                {"num_workers": 0, "evaluation_batch_size": 8, "mixed_precision": False}
            )
            object_dir = train_object_expert(
                object_config, model_factory=lambda unused: TinyImageModel()
            )

            set_config = load_set_expert_config(
                ROOT / "configs" / "expert_background_set.yaml",
                seed=402,
                phase0_dir=str(phase0),
                output_root=str(root / "set"),
                device="cpu",
            )
            set_config["training"].update(
                {"num_workers": 0, "evaluation_batch_size": 4, "mixed_precision": False}
            )
            # This synthetic fixture has no census exclusions. Production
            # retains the locked job-22266 expectation of 4887 and 6285.
            set_config["input"]["training_capacity_expected_exclusions"] = []
            smoke = run_set_expert_smoke(
                set_config,
                root / "set_smoke.json",
                model_factory=lambda unused: TinySetModel(),
            )
            self.assertEqual(smoke["status"], "passed")
            self.assertGreater(smoke["train"]["train_optimizer_step_count"], 0)
            self.assertEqual(smoke["train"]["train_amp_skipped_step_count"], 0)
            self.assertIn("train_crop_fallback_count", smoke["train"])
            self.assertEqual(
                smoke["background_view_capacity_audit"][
                    "candidate_train"
                ]["status"],
                "passed",
            )
            self.assertGreaterEqual(
                smoke["background_view_capacity_audit"][
                    "candidate_train"
                ]["fixed_view_fallback_fraction"],
                0.0,
            )
            self.assertEqual(
                smoke["training_capacity_eligibility"][
                    "original_sample_count"
                ],
                32,
            )
            set_dir = train_set_expert(
                set_config, model_factory=lambda unused: TinySetModel()
            )
            self.assertEqual(verify_set_expert(set_dir)["status"], "complete")

            fusion_config = load_set_fusion_config(
                ROOT / "configs" / "fusion_set.yaml",
                seed=403,
                phase0_dir=str(phase0),
                object_expert_dir=str(object_dir),
                set_expert_dir=str(set_dir),
                output_root=str(root / "fusion"),
                allow_expert_sanity_warnings=True,
            )
            fusion_dir = build_set_fusion_artifacts(fusion_config)
            verified = verify_set_fusion_artifacts(fusion_dir)
            self.assertEqual(verified["status"], "complete")
            self.assertEqual(verified["sample_count"], 8)
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
            score = score_candidate_set_file(fusion_dir, candidate_path)
            self.assertAlmostEqual(score["rank"]["setv_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
