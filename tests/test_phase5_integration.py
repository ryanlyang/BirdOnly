from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
from setv.candidate.config import load_candidate_config
from setv.candidate.selectors import ExpertFusionInput
from setv.candidate.train import (
    run_candidate_smoke,
    train_candidate,
    verify_candidate,
)
from setv.phase0 import approve_visual_audit, build_phase0


if TORCH_AVAILABLE:

    class TinyCandidate(torch.nn.Module):
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


def fake_selector_inputs(
    ids: np.ndarray, labels: np.ndarray, fusion_paths: dict[str, str]
):
    hard = np.zeros(len(labels), dtype=np.uint8)
    u = np.empty(len(labels), dtype=np.float64)
    for class_id in (0, 1):
        indices = np.flatnonzero(labels == class_id)
        u[indices] = np.linspace(0.25, 1.0, len(indices))
    fusion = {
        "sample_id": ids,
        "true_label": labels,
        "hard_target": hard,
        "hard_valid": False,
        "rank": {"u_rank": u},
        "logistic": {"available": False, "reason": "fixture degenerate target"},
    }
    return {
        name: ExpertFusionInput(
            name=name,
            fusion_dir=Path(fusion_paths[name]),
            fusion=fusion,
            background_prediction=(1 - labels).astype(np.int64),
        )
        for name in ("exact", "sanitized", "set")
    }


@unittest.skipUnless(TORCH_AVAILABLE, "torch unavailable")
class Phase5IntegrationTests(unittest.TestCase):
    def test_trajectory_selection_checkpoints_and_hidden_test_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, masks = create_fixture(root)
            phase0 = root / "phase0"
            build_phase0(fixture_config(dataset, masks, phase0))
            approve_visual_audit(phase0, reviewer="phase5 fixture", confirmation=True)
            biased = np.genfromtxt(
                phase0 / "splits" / "waterbirds95_biased_val.csv",
                delimiter=",",
                names=True,
                dtype=None,
                encoding="utf-8",
            )
            ids = np.asarray([str(value) for value in biased["sample_id"]])
            labels = np.asarray(biased["y"], dtype=np.int64)
            fusion_paths = {}
            for name in ("exact", "sanitized", "set"):
                path = root / f"{name}_fusion"
                path.mkdir()
                (path / "fusion_receipt.json").write_text(
                    json.dumps(
                        {
                            "kind": f"fixture_{name}",
                            "phase0_dir": str(phase0),
                        }
                    ),
                    encoding="utf-8",
                )
                fusion_paths[name] = str(path)
            synthetic = fake_selector_inputs(ids, labels, fusion_paths)
            config = load_candidate_config(
                ROOT / "configs" / "candidate_erm.yaml",
                seed=505,
                phase0_dir=str(phase0),
                exact_fusion_dir=fusion_paths["exact"],
                sanitized_fusion_dir=fusion_paths["sanitized"],
                set_fusion_dir=fusion_paths["set"],
                output_root=str(root / "candidate"),
                device="cpu",
            )
            config["training"].update(
                {"num_workers": 0, "evaluation_batch_size": 8, "mixed_precision": False}
            )
            with patch(
                "setv.candidate.train.load_selector_inputs",
                return_value=synthetic,
            ):
                smoke = run_candidate_smoke(
                    config,
                    root / "smoke.json",
                    model_factory=lambda unused: TinyCandidate(),
                )
                self.assertIsNone(smoke["test_metrics"])
                candidate_dir = train_candidate(
                    config, model_factory=lambda unused: TinyCandidate()
                )
            verified = verify_candidate(candidate_dir)
            self.assertEqual(verified["completed_epochs"], 50)
            self.assertLess(verified["rolling_checkpoint_count"], 50)
            with np.load(
                candidate_dir / "biased_val" / "epoch_predictions.npz",
                allow_pickle=False,
            ) as archive:
                self.assertEqual(archive["candidate_logits"].shape, (50, 8, 2))
            selection = json.loads(
                (candidate_dir / "selection" / "selection_receipt.json").read_text()
            )
            self.assertFalse(selection["test_metrics_seen"])
            self.assertFalse(selection["test_metrics_used"])
            test_report = json.loads(
                (candidate_dir / "reporting_only" / "test_metrics.json").read_text()
            )
            self.assertEqual(test_report["namespace"], "reporting_only")
            self.assertEqual(len(test_report["epochs"]), 50)
            training_log = (candidate_dir / "training.jsonl").read_text()
            self.assertNotIn('"worst_group_accuracy"', training_log)


if __name__ == "__main__":
    unittest.main()
