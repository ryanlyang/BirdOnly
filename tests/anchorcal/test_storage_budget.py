"""Storage projection and free-capacity preflight tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from anchorcal.errors import PreflightError
from anchorcal.storage_budget import GIB, allocated_bytes, assess_storage_budget


def _config() -> dict[str, object]:
    return {
        "storage": {
            "hard_budget_gib": 40.0,
            "launch_guard_gib": 35.0,
            "minimum_filesystem_free_gib": 16.0,
            "worst_case_concurrent_growth_gib": 6.0,
            "projected_full_campaign_components_gib": {
                "candidate_checkpoints": 6.0,
                "restart_and_atomic_staging": 2.0,
                "candidate_hdf5_and_analysis": 1.0,
                "branches_and_anchors": 1.0,
                "manifests_galleries_and_reserve": 2.0,
            },
        }
    }


class StorageBudgetTests(unittest.TestCase):
    def test_projection_and_capacity_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "small.bin").write_bytes(b"x" * 4097)
            report = assess_storage_budget(
                _config(),
                root,
                stage="production_preflight",
                filesystem_free_bytes=100 * GIB,
                filesystem_total_bytes=200 * GIB,
            )
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["projected_full_campaign_growth_gib"], 12.0)
            self.assertEqual(report["required_free_gib"], 16.0)
            self.assertEqual(report["allocated_bytes"], allocated_bytes(root))
            self.assertGreater(report["allocated_bytes"], 0)

    def test_insufficient_free_space_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(PreflightError, "filesystem free space"):
                assess_storage_budget(
                    _config(),
                    temporary,
                    stage="production_preflight",
                    filesystem_free_bytes=15 * GIB,
                    filesystem_total_bytes=200 * GIB,
                )

    def test_projection_must_fit_launch_guard(self) -> None:
        config = _config()
        config["storage"]["projected_full_campaign_components_gib"] = {
            "unbounded": 36.0
        }
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(PreflightError, "projected campaign peak"):
                assess_storage_budget(
                    config,
                    temporary,
                    stage="production_preflight",
                    filesystem_free_bytes=100 * GIB,
                    filesystem_total_bytes=200 * GIB,
                )

    def test_concurrent_reserve_must_not_exceed_projection(self) -> None:
        config = _config()
        config["storage"]["projected_full_campaign_components_gib"] = {
            "too_small": 5.0
        }
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(PreflightError, "concurrent storage reserve"):
                assess_storage_budget(
                    config,
                    temporary,
                    stage="production_preflight",
                    filesystem_free_bytes=100 * GIB,
                    filesystem_total_bytes=200 * GIB,
                )


if __name__ == "__main__":
    unittest.main()
