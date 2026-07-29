from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.errors import ConfigurationError
from setv.ula.config import load_phase6_config, load_ula_proxy_config
from setv.ula.provenance import audit_official_source


class ULAConfigProvenanceTests(unittest.TestCase):
    def test_vendored_official_source_is_bound(self) -> None:
        audit = audit_official_source(ROOT / "uLA")
        self.assertEqual(
            audit["required_commit"],
            "5867fb6e9a8485ed08b4cbe84900f2b5ac4fac5d",
        )
        self.assertGreater(audit["vendor_file_count"], 20)
        self.assertEqual(audit["repository"], "https://github.com/tsirif/uLA")

    def test_proxy_config_locks_method_details(self) -> None:
        config = load_ula_proxy_config(
            ROOT / "configs" / "ula_proxy_waterbirds95.yaml",
            phase0_dir="/tmp/p0",
            official_repo=str(ROOT / "uLA"),
            ssl_checkpoint="/tmp/ssl.ckpt",
            output_root="/tmp/out",
            seed=17,
            device="cpu",
        )
        self.assertEqual(config["method"]["ssl_method"], "mocov2plus")
        broken = Path(tempfile.mkdtemp()) / "broken.yaml"
        text = (ROOT / "configs" / "ula_proxy_waterbirds95.yaml").read_text()
        broken.write_text(text.replace("calibration: none", "calibration: temperature"))
        with self.assertRaises(ConfigurationError):
            load_ula_proxy_config(
                broken,
                phase0_dir="/tmp/p0",
                official_repo=str(ROOT / "uLA"),
                ssl_checkpoint="/tmp/ssl.ckpt",
                output_root="/tmp/out",
                seed=17,
                device="cpu",
            )

    def test_analysis_requires_three_unique_candidate_seeds(self) -> None:
        common = {
            "phase0_dir": "/tmp/p0",
            "candidate_root": "/tmp/candidates",
            "ula_proxy_dir": "/tmp/ula",
            "exact_fusion_dir": "/tmp/exact",
            "sanitized_fusion_dir": "/tmp/sanitized",
            "set_fusion_dir": "/tmp/set",
            "output_dir": "/tmp/phase6",
        }
        with self.assertRaises(ConfigurationError):
            load_phase6_config(
                ROOT / "configs" / "phase6_analysis.yaml",
                candidate_seeds=[1, 2],
                **common,
            )
        config = load_phase6_config(
            ROOT / "configs" / "phase6_analysis.yaml",
            candidate_seeds=[1, 2, 3],
            **common,
        )
        self.assertTrue(config["selection"]["choose_expert_and_fusion_jointly"])


if __name__ == "__main__":
    unittest.main()
