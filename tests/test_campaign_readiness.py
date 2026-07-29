from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.campaign import (
    STAGES,
    _verify_artifact,
    campaign_environment,
    expected_artifact_paths,
    load_campaign_manifest,
    run_campaign_preflight,
    validate_campaign_manifest,
)
from setv.errors import ConfigurationError


def _inventory(manifest: dict, *, target: str) -> dict:
    result = {}
    for stage, paths in expected_artifact_paths(manifest).items():
        state = "absent" if stage == target else "verified"
        result[stage] = [
            {
                "path": str(path),
                "exists": state != "absent",
                "receipt": "fixture.json",
                "receipt_exists": state != "absent",
                "state": state,
                "verification": {"status": "fixture"} if state == "verified" else None,
            }
            for path in paths
        ]
    return result


class CampaignReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_campaign_manifest(
            ROOT / "configs" / "campaign_waterbirds95.yaml"
        )

    def test_manifest_exports_the_frozen_campaign_namespace(self) -> None:
        environment = campaign_environment(self.manifest)
        self.assertEqual(
            environment["SETV_CAMPAIGN_ROOT"],
            "/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95",
        )
        self.assertEqual(environment["SETV_OBJECT_SEED"], "1101")
        self.assertEqual(environment["SETV_EXACT_FUSION_SEED"], "2101")
        self.assertEqual(environment["SETV_SANITIZED_FUSION_SEED"], "2201")
        self.assertEqual(environment["SETV_SET_FUSION_SEED"], "2301")
        self.assertEqual(environment["SETV_CANDIDATE_SEEDS"], "3101,3102,3103")
        self.assertEqual(
            environment["SETV_ULA_REPO"],
            "/home/ryreu/guided_cnn/BirdOnly/uLA",
        )
        self.assertNotIn("SETV_ULA_ENV", environment)
        self.assertNotIn("SETV_ULA_SSL_CHECKPOINT", environment)

    def test_manifest_rejects_a_two_seed_private_pilot(self) -> None:
        broken = deepcopy(self.manifest)
        broken["seeds"]["candidate_erm"] = [1, 2]
        with self.assertRaises(ConfigurationError):
            validate_campaign_manifest(broken)

    def test_sourceable_loader_matches_python_manifest(self) -> None:
        result = subprocess.run(
            [
                "bash",
                "-c",
                (
                    "source scripts/load_campaign_env.sh; "
                    "printf '%s|%s|%s\\n' "
                    "\"$SETV_OBJECT_SEED\" "
                    "\"$SETV_CANDIDATE_SEEDS\" "
                    "\"$SETV_CAMPAIGN_ROOT\""
                ),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual(
            result.stdout.strip(),
            (
                "1101|3101,3102,3103|"
                "/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95"
            ),
        )

    def test_every_production_launcher_is_manifest_and_preflight_gated(self) -> None:
        launchers = {
            "phase0": "submit_phase0.sh",
            "phase1": "submit_phase1_object.sh",
            "phase2": "submit_phase2_exact.sh",
            "phase3": "submit_phase3_sanitized.sh",
            "phase4": "submit_phase4_set.sh",
            "phase5": "submit_phase5_candidate.sh",
            "phase6": "submit_phase6.sh",
        }
        for stage, filename in launchers.items():
            with self.subTest(stage=stage):
                text = (ROOT / "scripts" / filename).read_text()
                self.assertIn('source "${SCRIPT_DIR}/load_campaign_env.sh"', text)
                self.assertIn(
                    f'run_submission_preflight.sh" {stage} "$preflight_report"',
                    text,
                )
                self.assertIn("preflight_sha256=", text)
        searched = [
            ROOT / "scripts" / "submit_phase2_exact.sh",
            ROOT / "scripts" / "submit_phase3_sanitized.sh",
            ROOT / "scripts" / "submit_phase4_set.sh",
            ROOT / "slurm" / "phase2_exact_fusion.sbatch",
            ROOT / "slurm" / "phase3_sanitized_fusion.sbatch",
            ROOT / "slurm" / "phase4_set_fusion.sbatch",
        ]
        self.assertNotIn(
            "SETV_FUSION_SEED",
            "\n".join(path.read_text() for path in searched),
        )

    def test_preflight_receipt_is_reporting_safe(self) -> None:
        inventory = _inventory(self.manifest, target="phase0")
        with (
            patch(
                "setv.campaign._git",
                side_effect=[(0, "a" * 40), (0, "")],
            ),
            patch("setv.campaign._artifact_inventory", return_value=inventory),
        ):
            report = run_campaign_preflight(
                self.manifest,
                stage="phase0",
                repository=ROOT,
                check_tigris_filesystem=False,
            )
        self.assertTrue(report["ready"])
        self.assertFalse(report["reporting_only_metric_values_included"])
        self.assertFalse(report["preflight_used_for_method_selection"])
        self.assertNotIn("test_results", report)
        self.assertEqual(report["artifact_inventory"]["phase0"][0]["state"], "absent")

    def test_phase6_requires_a_real_explicit_ula_execution_source(self) -> None:
        inventory = _inventory(self.manifest, target="phase6")
        with tempfile.TemporaryDirectory() as temporary:
            environment = Path(temporary) / "legacy"
            python = environment / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.touch()
            python.chmod(0o755)
            with (
                patch.dict(
                    os.environ,
                    {
                        "SETV_ULA_ENV": str(environment),
                        "SETV_ULA_SSL_CHECKPOINT": "",
                    },
                    clear=False,
                ),
                patch(
                    "setv.campaign._git",
                    side_effect=[(0, "b" * 40), (0, "")],
                ),
                patch("setv.campaign._artifact_inventory", return_value=inventory),
            ):
                report = run_campaign_preflight(
                    self.manifest,
                    stage="phase6",
                    repository=ROOT,
                    check_tigris_filesystem=False,
                )
        self.assertTrue(report["ready"])
        check = next(
            item for item in report["checks"] if item["name"] == "ula_execution_source"
        )
        self.assertTrue(check["details"]["runtime_override_used"])
        self.assertTrue(check["details"]["legacy_environment_usable"])

    def test_invalid_existing_phase0_approval_is_not_treated_as_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            phase0 = Path(temporary)
            approval = phase0 / "mask_audit" / "visual_review_approval.json"
            approval.parent.mkdir(parents=True)
            approval.write_text("{}")
            with patch(
                "setv.phase0.verify_phase0",
                side_effect=ValueError("approval hash mismatch"),
            ):
                with self.assertRaisesRegex(ValueError, "approval hash mismatch"):
                    _verify_artifact("phase0", 0, phase0)

    def test_stage_names_are_complete_and_ordered(self) -> None:
        self.assertEqual(STAGES, tuple(f"phase{index}" for index in range(7)))


if __name__ == "__main__":
    unittest.main()
