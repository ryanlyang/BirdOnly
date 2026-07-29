from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

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
from setv.experts.scores import load_object_scores
from setv.experts.train_object import (
    run_object_expert_smoke,
    train_object_expert,
    verify_object_expert,
)
from setv.phase0 import approve_visual_audit, build_phase0


if TORCH_AVAILABLE:

    class TinyObjectModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = torch.nn.Linear(4, 2)

        def forward(self, images):
            features = self.features(images).flatten(1)
            return self.classifier(features)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is unavailable in the local test interpreter")
class ObjectTrainingTests(unittest.TestCase):
    def test_twenty_epoch_training_publishes_final_state_and_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, masks = create_fixture(root)
            phase0_dir = root / "phase0"
            build_phase0(fixture_config(dataset, masks, phase0_dir))
            approve_visual_audit(
                phase0_dir, reviewer="object fixture reviewer", confirmation=True
            )

            output_root = root / "object_expert"
            config = load_object_expert_config(
                ROOT / "configs" / "expert_object_green.yaml",
                seed=2718,
                phase0_dir=str(phase0_dir),
                output_root=str(output_root),
                device="cpu",
            )
            config["training"]["num_workers"] = 0
            config["training"]["evaluation_batch_size"] = 8
            config["training"]["mixed_precision"] = False

            smoke_report = root / "object_smoke.json"
            smoke = run_object_expert_smoke(
                config,
                smoke_report,
                model_factory=lambda unused: TinyObjectModel(),
            )
            self.assertEqual(smoke["status"], "passed")
            self.assertEqual(smoke["train"]["train_sample_count"], 32)
            self.assertEqual(smoke["biased_val"]["biased_val_sample_count"], 8)
            self.assertTrue(smoke_report.is_file())

            output = train_object_expert(
                config, model_factory=lambda unused: TinyObjectModel()
            )
            verified = verify_object_expert(output, load_checkpoint=True)
            self.assertEqual(verified["status"], "complete")
            self.assertTrue(
                (output / "checkpoints" / "object_expert_final.pt").is_file()
            )
            self.assertEqual(
                len(pd.read_csv(output / "metrics" / "epoch_metrics.csv")), 20
            )
            scores = load_object_scores(output / "scores" / "object_val_scores.npz")
            self.assertEqual(len(scores["sample_id"]), 8)
            self.assertEqual(
                set(scores),
                {
                    "sample_id",
                    "true_label",
                    "object_logits",
                    "object_true_class_margin",
                    "object_predicted_class",
                    "object_correct",
                },
            )


if __name__ == "__main__":
    unittest.main()
