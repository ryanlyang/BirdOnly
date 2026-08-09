from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from anchorcal.errors import PreflightError
from anchorcal.runtime import REQUIRED_PACKAGES, write_package_lock


REPOSITORY = Path(__file__).resolve().parents[2]
LAUNCHER = REPOSITORY / "scripts" / "anchorcal" / "submit_campaign.sh"
RUNTIME_COMMON = REPOSITORY / "slurm" / "anchorcal" / "runtime_common.sh"
PREFLIGHT_JOB = REPOSITORY / "slurm" / "anchorcal" / "preflight.sbatch"


def _failure_function_source() -> str:
    source = LAUNCHER.read_text(encoding="utf-8")
    start = source.index("write_partial_receipt() {")
    end = source.index("trap handle_submission_error ERR")
    return source[start:end]


class PackageLockTests(unittest.TestCase):
    def test_package_lock_queries_only_required_packages(self) -> None:
        versions = {name: f"version-for-{index}" for index, name in enumerate(REQUIRED_PACKAGES)}
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "package-lock.txt"
            with mock.patch(
                "anchorcal.runtime.importlib.metadata.version",
                side_effect=lambda name: versions[name],
            ) as lookup:
                write_package_lock(target)

            self.assertEqual(
                target.read_text(encoding="utf-8").splitlines(),
                [f"{name}=={versions[name]}" for name in sorted(REQUIRED_PACKAGES)],
            )
            self.assertEqual(
                {call.args[0] for call in lookup.call_args_list}, set(REQUIRED_PACKAGES)
            )

    def test_package_lock_fails_when_a_required_package_is_missing(self) -> None:
        def version(name: str) -> str:
            if name == "timm":
                raise __import__("importlib").metadata.PackageNotFoundError(name)
            return "1.0"

        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "package-lock.txt"
            with mock.patch(
                "anchorcal.runtime.importlib.metadata.version", side_effect=version
            ):
                with self.assertRaisesRegex(PreflightError, "timm"):
                    write_package_lock(target)
            self.assertFalse(target.exists())


class LauncherContractTests(unittest.TestCase):
    def test_frozen_fcv_membership_root_is_a_required_exact_tigris_path(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")
        preflight = (REPOSITORY / "src" / "anchorcal" / "preflight.py").read_text(
            encoding="utf-8"
        )
        expected = (
            "/home/ryreu/guided_cnn/logsWaterbird/"
            "fcv_vit_waterbirds100_first_study/split_manifests"
        )
        self.assertIn('"fcv_split_manifest_root"', launcher)
        self.assertIn(expected, launcher)
        self.assertIn('"fcv_split_manifest_root"', preflight)
        self.assertIn("fcv_vit_waterbirds100_first_study/split_manifests", preflight)
        for filename in (
            "manifest_bundle.json",
            "split_indices.json",
            "metadata_train.csv",
            "metadata_val.csv",
        ):
            self.assertIn(filename, launcher)

    def test_launcher_and_jobs_share_fast_required_package_lock(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")
        runtime = RUNTIME_COMMON.read_text(encoding="utf-8")
        self.assertNotIn("metadata.distributions", launcher)
        self.assertNotIn("metadata.distributions", runtime)
        self.assertIn("from anchorcal.runtime import write_package_lock", launcher)
        self.assertIn("from anchorcal.runtime import write_package_lock", runtime)

    def test_preflight_checksum_binds_public_receipt_and_protected_mask_audit(self) -> None:
        source = PREFLIGHT_JOB.read_text(encoding="utf-8")
        self.assertIn("preflight/selector_mask_receipt.json", source)
        self.assertIn(
            "analysis_only/masks/waterbirds100_oracle_val_mask_audit.json",
            source,
        )

    def test_interrupt_traps_are_installed_before_frozen_input_work(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        trap_position = source.index("trap handle_submission_error ERR")
        self.assertLess(trap_position, source.index('progress "freezing campaign inputs'))
        self.assertLess(trap_position, source.index("from anchorcal.runtime import write_package_lock"))
        self.assertLess(trap_position, source.index('submit_job "$REPO/slurm/anchorcal/preflight.sbatch"'))
        self.assertIn("trap 'handle_submission_signal INT' INT", source)
        self.assertIn("trap 'handle_submission_signal TERM' TERM", source)

    def _run_failure_handler(
        self, *, submitted_jobs: list[int], scheduler_attempted: bool | None = None
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name) / "output"
        manifest = root / "manifests" / "campaign_test"
        (root / "submission_receipts").mkdir(parents=True)
        manifest.mkdir(parents=True)
        (manifest / "partial.txt").write_text("partial\n", encoding="utf-8")
        jobs = " ".join(str(value) for value in submitted_jobs)
        attempted = bool(submitted_jobs) if scheduler_attempted is None else scheduler_attempted
        shell = f"""
set -Eeuo pipefail
OUTPUT_ROOT={root!s}
campaign_id=test
manifest_dir="$OUTPUT_ROOT/manifests/campaign_test"
ANCHORCAL_EXPECTED_COMMIT={'a' * 40}
submission_phase=freezing_inputs
INPUT_RECEIPT=
input_receipt_sha256=
PARTIAL_MANIFEST_ARCHIVE=not_applicable
submitted_jobs=({jobs})
scheduler_submission_attempted={str(attempted).lower()}
SCANCEL_LOG="$OUTPUT_ROOT/scancel.log"
sha256_of() {{ sha256sum -- "$1" | cut -d' ' -f1; }}
scancel() {{ printf '%s\n' "$*" > "$SCANCEL_LOG"; }}
{_failure_function_source()}
handle_submission_failure 130 signal_INT
"""
        result = subprocess.run(
            ["bash", "-c", shell], text=True, capture_output=True, check=False
        )
        return result, root

    def test_presubmission_interrupt_archives_partial_manifest(self) -> None:
        result, root = self._run_failure_handler(submitted_jobs=[])
        self.assertEqual(result.returncode, 130, result.stderr)
        self.assertFalse((root / "manifests" / "campaign_test").exists())
        archive = (
            root
            / "submission_receipts"
            / "aborted_manifests"
            / "campaign_test"
        )
        self.assertEqual((archive / "partial.txt").read_text(encoding="utf-8"), "partial\n")
        receipt = root / "submission_receipts" / "campaign_test_submission_interrupted.txt"
        payload = receipt.read_text(encoding="utf-8")
        self.assertIn("submitted_job_ids=none", payload)
        self.assertIn("rollback_status=no_jobs_submitted_manifest_archived", payload)
        self.assertIn(f"partial_manifest_archive={archive}", payload)

    def test_postsubmission_interrupt_cancels_jobs_and_preserves_manifest(self) -> None:
        result, root = self._run_failure_handler(submitted_jobs=[101, 102])
        self.assertEqual(result.returncode, 130, result.stderr)
        self.assertTrue((root / "manifests" / "campaign_test" / "partial.txt").is_file())
        self.assertEqual((root / "scancel.log").read_text(encoding="utf-8"), "101 102\n")
        receipt = root / "submission_receipts" / "campaign_test_submission_interrupted.txt"
        payload = receipt.read_text(encoding="utf-8")
        self.assertIn("submitted_job_ids=101 102", payload)
        self.assertIn("rollback_status=scancel_requested", payload)
        self.assertIn("partial_manifest_archive=preserved_for_submitted_jobs", payload)

    def test_unparseable_scheduler_response_preserves_manifest_for_manual_audit(self) -> None:
        result, root = self._run_failure_handler(
            submitted_jobs=[], scheduler_attempted=True
        )
        self.assertEqual(result.returncode, 130, result.stderr)
        self.assertTrue((root / "manifests" / "campaign_test" / "partial.txt").is_file())
        self.assertFalse((root / "submission_receipts" / "aborted_manifests").exists())
        receipt = root / "submission_receipts" / "campaign_test_submission_interrupted.txt"
        payload = receipt.read_text(encoding="utf-8")
        self.assertIn("scheduler_submission_attempted=true", payload)
        self.assertIn(
            "rollback_status=unknown_job_id_manual_scheduler_check_required", payload
        )
        self.assertIn("partial_manifest_archive=preserved_after_scheduler_attempt", payload)


if __name__ == "__main__":
    unittest.main()
