from __future__ import annotations

import platform
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fixture_data import create_fixture, fixture_config
from prepare_ula_official_shadow import _smoke_subset
from setv.phase0 import approve_visual_audit, build_phase0
from setv.ula.compatibility import probe_ula_environment
from setv.ula.config import load_ula_proxy_config
from setv.ula.proxy import run_ula_proxy_smoke

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ULACompatibilityTests(unittest.TestCase):
    def test_external_adapter_probe_records_version_drift(self) -> None:
        report = probe_ula_environment(
            ROOT / "uLA",
            mode="external_checkpoint",
            require_gpu=False,
            expected_architecture=platform.machine(),
        )
        imports_usable = all(
            report["imports"][name]["ok"] for name in ("torch", "torchvision")
        )
        self.assertEqual(report["accepted"], imports_usable)
        if not imports_usable:
            self.assertTrue(
                any(
                    reason.startswith("import_failed:")
                    for reason in report["blocking_reasons"]
                )
            )
        self.assertEqual(report["mode"], "external_checkpoint")
        self.assertEqual(
            report["official_source"]["required_commit"],
            "5867fb6e9a8485ed08b4cbe84900f2b5ac4fac5d",
        )
        self.assertIn("official_requirements_exact_match", report)

    def test_smoke_shadow_is_deterministic_and_class_balanced(self) -> None:
        frame = pd.DataFrame(
            {
                "sample_id": [f"id-{index:02d}" for index in range(20)],
                "y": np.asarray([0, 1] * 10),
            }
        )
        first = _smoke_subset(frame.sample(frac=1, random_state=3), 8)
        second = _smoke_subset(frame.sample(frac=1, random_state=9), 8)
        self.assertEqual(first["sample_id"].tolist(), second["sample_id"].tolist())
        self.assertEqual(first["y"].value_counts().to_dict(), {0: 4, 1: 4})

    @unittest.skipUnless(TORCH_AVAILABLE, "torch unavailable")
    def test_proxy_smoke_updates_only_linear_head(self) -> None:
        class TinyProxy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Sequential(
                    torch.nn.AdaptiveAvgPool2d(1),
                    torch.nn.Flatten(),
                    torch.nn.Linear(3, 4),
                )
                self.encoder.requires_grad_(False)
                self.head = torch.nn.Linear(4, 2)

            def train(self, mode: bool = True):
                super().train(mode)
                self.encoder.eval()
                return self

            def forward(self, inputs):
                self.encoder.eval()
                with torch.no_grad():
                    features = self.encoder(inputs)
                return self.head(features)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, masks = create_fixture(root)
            phase0 = root / "phase0"
            build_phase0(fixture_config(dataset, masks, phase0))
            approve_visual_audit(
                phase0, reviewer="uLA smoke fixture", confirmation=True
            )
            checkpoint = root / "official.ckpt"
            checkpoint.write_bytes(b"fixture official checkpoint")
            config = load_ula_proxy_config(
                ROOT / "configs" / "ula_proxy_waterbirds95.yaml",
                phase0_dir=str(phase0),
                official_repo=str(ROOT / "uLA"),
                ssl_checkpoint=str(checkpoint),
                output_root=str(root / "unused"),
                seed=17,
                device="cpu",
            )
            config["training"]["num_workers"] = 0
            report = run_ula_proxy_smoke(
                config,
                root / "smoke.json",
                model_factory=lambda _: TinyProxy(),
            )
            self.assertTrue(report["accepted"])
            self.assertEqual(
                report["train_batch"]["trainable_parameters"],
                ["head.weight", "head.bias"],
            )
            self.assertFalse(report["information_boundary"]["test_used"])
            self.assertFalse(report["information_boundary"]["oracle_used"])

    def test_phase6_submission_is_smoke_gated(self) -> None:
        launcher = (ROOT / "scripts" / "submit_phase6.sh").read_text()
        smoke = (ROOT / "slurm" / "phase6_ula_smoke.sbatch").read_text()
        self.assertIn("phase6_ula_smoke.sbatch", launcher)
        self.assertIn('afterok:${smoke_id}', launcher)
        self.assertIn("smoke_ula_proxy.py", smoke)
        self.assertIn("--smoke-samples-per-split 8", smoke)


if __name__ == "__main__":
    unittest.main()
