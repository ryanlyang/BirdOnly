from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None

from anchorcal.checkpoint_verification import (
    verify_candidate_checkpoint_artifacts,
)
from anchorcal.checkpoints import CheckpointManager, hash_model_state
from anchorcal.errors import StorageError
from anchorcal.visible_checkpoint_verification import (
    verify_visible_checkpoint_artifacts,
)


def _state(value: float):
    assert torch is not None
    return {
        "linear.weight": torch.tensor([[value, value + 1]], dtype=torch.float32),
        "linear.bias": torch.tensor([value], dtype=torch.bfloat16),
        "counter": torch.tensor(3, dtype=torch.int64),
    }


@unittest.skipUnless(torch is not None, "torch unavailable")
class CheckpointManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary.name) / "candidate_a"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_complete_candidate(self) -> None:
        state = _state(9.0)
        with CheckpointManager(self.run_dir, run_id="candidate_a") as manager:
            for selector in ("ordinary", "saliency", "swap", "blur"):
                manager.update_selector(
                    selector,
                    epoch=3,
                    model_state=state,
                    ranking_key=(0.8, 0.7, -0.2, -3),
                )
            manager.update_selector(
                "oracle", epoch=3, model_state=state, ranking_key=(0.9, 0.8, 0.7, -3)
            )
            manager.save_final_epoch(epoch=3, model_state=state)
            manager.save_resume(
                epoch=3,
                model_state=state,
                optimizer_state={"step": 3},
                scheduler_state={"last_epoch": 3},
                rng_states={"torch": torch.get_rng_state()},
            )
            manager.complete(resume_policy="archive")

    def test_model_hash_is_value_based_and_supports_bfloat16(self) -> None:
        first = _state(1.0)
        second = {name: tensor.clone() for name, tensor in first.items()}
        self.assertEqual(hash_model_state(first), hash_model_state(second))
        second["linear.weight"][0, 0] += 1
        self.assertNotEqual(hash_model_state(first), hash_model_state(second))

    def test_visible_and_oracle_manifests_are_physically_separate(self) -> None:
        with CheckpointManager(self.run_dir, run_id="candidate_a") as manager:
            manager.update_selector(
                "ordinary",
                epoch=2,
                model_state=_state(1.0),
                ranking_key=(0.8, 0.9, -0.2, -2),
            )
            manager.update_selector(
                "oracle",
                epoch=3,
                model_state=_state(2.0),
                ranking_key=(0.7, 0.75, 0.8, -3),
            )
            visible_path = manager.manifest_path
            hidden_path = manager.hidden_manifest_path

        visible = json.loads(visible_path.read_text(encoding="utf-8"))
        hidden = json.loads(hidden_path.read_text(encoding="utf-8"))
        self.assertNotIn("oracle", visible["selectors"])
        self.assertNotIn("oracle", visible_path.read_text(encoding="utf-8"))
        self.assertEqual(hidden["namespace"], "exploratory_hidden_metrics")
        self.assertEqual(hidden["selectors"]["oracle"]["epoch"], 3)

    def test_shared_weight_object_does_not_leak_oracle_epoch_or_metadata(self) -> None:
        shared = _state(7.0)
        with CheckpointManager(self.run_dir, run_id="candidate_a") as manager:
            manager.update_selector(
                "oracle",
                epoch=17,
                model_state=shared,
                ranking_key=(0.9,),
                metadata={"oracle_wga": 0.9},
            )
            manager.update_selector(
                "ordinary",
                epoch=4,
                model_state=shared,
                ranking_key=(0.8,),
                metadata={"biased_accuracy": 0.8},
            )
            visible_text = manager.manifest_path.read_text(encoding="utf-8")
            self.assertNotIn("oracle_wga", visible_text)
            model_hash = manager.visible["selectors"]["ordinary"]["model_hash"]
            self.assertEqual(manager.visible["objects"][model_hash]["epochs"], [4])
            self.assertNotIn(
                17,
                [
                    epoch
                    for record in manager.visible["objects"].values()
                    for epoch in record["epochs"]
                ],
            )
            object_path = manager._resolve_relative(
                manager.visible["objects"][model_hash]["path"]
            )
            payload = torch.load(object_path, map_location="cpu", weights_only=False)
            self.assertNotIn("first_epoch", payload)
            self.assertNotIn("metadata", payload)

    def test_namespace_object_inventories_prune_cross_namespace_orphans(self) -> None:
        with CheckpointManager(self.run_dir, run_id="candidate_a") as manager:
            manager.update_selector(
                "ordinary", epoch=1, model_state=_state(1.0), ranking_key=(0.7,)
            )
            manager.update_selector(
                "oracle", epoch=1, model_state=_state(1.0), ranking_key=(0.8,)
            )
            oracle_hash = manager.hidden["selectors"]["oracle"]["model_hash"]
            manager.update_selector(
                "ordinary", epoch=2, model_state=_state(2.0), ranking_key=(0.9,)
            )
            self.assertNotIn(oracle_hash, manager.visible["objects"])
            self.assertIn(oracle_hash, manager.hidden["objects"])
            self.assertTrue(
                (manager.weights_dir / f"model_{oracle_hash}.pt").is_file()
            )

    def test_deduplicates_selectors_by_model_hash_and_prunes_orphans(self) -> None:
        shared = _state(1.0)
        newer = _state(2.0)
        with CheckpointManager(self.run_dir, run_id="candidate_a") as manager:
            self.assertTrue(
                manager.update_selector(
                    "ordinary", epoch=2, model_state=shared, ranking_key=(0.7,)
                )
            )
            self.assertTrue(
                manager.update_selector(
                    "saliency", epoch=2, model_state=shared, ranking_key=(0.6,)
                )
            )
            self.assertEqual(len(list(manager.weights_dir.glob("model_*.pt"))), 1)
            self.assertFalse(
                manager.update_selector(
                    "ordinary", epoch=3, model_state=newer, ranking_key=(0.65,)
                )
            )
            self.assertTrue(
                manager.update_selector(
                    "ordinary", epoch=3, model_state=newer, ranking_key=(0.8,)
                )
            )
            self.assertEqual(len(list(manager.weights_dir.glob("model_*.pt"))), 2)
            self.assertTrue(
                manager.update_selector(
                    "saliency", epoch=3, model_state=newer, ranking_key=(0.9,)
                )
            )
            # The epoch-2 object is no longer selected by any criterion.
            self.assertEqual(len(list(manager.weights_dir.glob("model_*.pt"))), 1)
            ordinary = manager.visible["selectors"]["ordinary"]
            saliency = manager.visible["selectors"]["saliency"]
            self.assertEqual(ordinary["model_hash"], saliency["model_hash"])

    def test_identical_weights_across_epochs_reuse_one_object(self) -> None:
        shared = _state(4.0)
        with CheckpointManager(self.run_dir, run_id="candidate_a") as manager:
            manager.update_selector(
                "ordinary", epoch=2, model_state=shared, ranking_key=(0.7,)
            )
            manager.update_selector(
                "blur", epoch=5, model_state=shared, ranking_key=(0.8,)
            )
            self.assertEqual(len(list(manager.weights_dir.glob("model_*.pt"))), 1)
            model_hash = manager.visible["selectors"]["ordinary"]["model_hash"]
            self.assertEqual(manager.visible["objects"][model_hash]["epochs"], [2, 5])

    def test_atomic_resume_round_trip_and_archive(self) -> None:
        with CheckpointManager(self.run_dir, run_id="candidate_a") as manager:
            first = manager.save_resume(
                epoch=4,
                model_state=_state(1.0),
                optimizer_state={"step": 10},
                scheduler_state={"last_epoch": 10},
                rng_states={"python": (1, 2), "torch": torch.get_rng_state()},
                scaler_state=None,
                dataloader_progress={"next_epoch": 5},
            )
            second = manager.save_resume(
                epoch=5,
                model_state=_state(2.0),
                optimizer_state={"step": 20},
                scheduler_state={"last_epoch": 20},
                rng_states={"python": (3, 4), "torch": torch.get_rng_state()},
                dataloader_progress={"next_epoch": 6},
            )
            self.assertNotEqual(first["sha256"], second["sha256"])
            payload = manager.load_resume()
            assert payload is not None
            self.assertEqual(payload["epoch"], 5)
            self.assertEqual(payload["optimizer_state"]["step"], 20)
            manager.complete(resume_policy="archive")
            self.assertFalse(manager.resume_path.exists())
            self.assertTrue(manager.final_state_path.is_file())
            self.assertIsNone(manager.visible["resume"])
            self.assertEqual(
                manager.visible["completion"]["resume_policy"], "archive"
            )

        with CheckpointManager(self.run_dir, run_id="candidate_a") as reopened:
            self.assertIsNone(reopened.load_resume())
            self.assertTrue(reopened.final_state_path.is_file())

    def test_resume_rejects_mid_epoch_state(self) -> None:
        with CheckpointManager(self.run_dir, run_id="candidate_a") as manager:
            with self.assertRaisesRegex(ValueError, "epoch boundaries"):
                manager.save_resume(
                    epoch=1,
                    model_state=_state(1.0),
                    optimizer_state={},
                    scheduler_state={},
                    rng_states={},
                    at_epoch_boundary=False,
                )
            self.assertFalse(manager.resume_path.exists())

    def test_final_epoch_is_forced_and_manifested(self) -> None:
        with CheckpointManager(self.run_dir, run_id="candidate_a") as manager:
            manager.save_final_epoch(epoch=40, model_state=_state(4.0))
            final = manager.visible["selectors"]["final"]
            self.assertEqual(final["epoch"], 40)
            self.assertTrue(manager._resolve_relative(final["path"]).is_file())

    def test_checkpoint_writer_lock(self) -> None:
        first = CheckpointManager(self.run_dir, run_id="candidate_a")
        try:
            with self.assertRaisesRegex(StorageError, "another writer"):
                CheckpointManager(self.run_dir, run_id="candidate_a")
        finally:
            first.close()

    def test_read_only_verifier_covers_visible_hidden_weights_and_final_state(self) -> None:
        self._write_complete_candidate()
        visible_path = self.run_dir / "checkpoints" / "manifest.json"
        hidden_path = (
            self.run_dir
            / "checkpoints"
            / "exploratory_hidden"
            / "oracle_manifest.json"
        )
        before = {
            path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in (visible_path, hidden_path)
        }
        verified = verify_candidate_checkpoint_artifacts(
            self.run_dir,
            expected_run_id="candidate_a",
            require_complete=True,
            required_visible_selectors=(
                "ordinary",
                "saliency",
                "swap",
                "blur",
                "final",
            ),
            required_hidden_selectors=("oracle",),
        )
        self.assertEqual(verified["retained_weight_count"], 1)
        self.assertIsNotNone(verified["visible"]["completion"]["final_state"])
        for path, snapshot in before.items():
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), snapshot)

    def test_verifier_rejects_tampered_weight_and_deleted_final_state(self) -> None:
        self._write_complete_candidate()
        weight = next((self.run_dir / "checkpoints" / "weights").glob("model_*.pt"))
        with weight.open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaisesRegex(StorageError, "byte size mismatch"):
            verify_candidate_checkpoint_artifacts(
                self.run_dir, expected_run_id="candidate_a", require_complete=True
            )

        self.temporary.cleanup()
        self.temporary = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary.name) / "candidate_a"
        self._write_complete_candidate()
        (self.run_dir / "checkpoints" / "final_state.pt").unlink()
        with self.assertRaisesRegex(StorageError, "missing"):
            verify_visible_checkpoint_artifacts(
                self.run_dir, expected_run_id="candidate_a", require_complete=True
            )

    def test_verifier_rejects_selector_reference_and_resume_tampering(self) -> None:
        self._write_complete_candidate()
        manifest_path = self.run_dir / "checkpoints" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["selectors"]["ordinary"]["epoch"] = 999
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(StorageError, "selector epoch"):
            verify_visible_checkpoint_artifacts(
                self.run_dir, expected_run_id="candidate_a", require_complete=True
            )

        self.temporary.cleanup()
        self.temporary = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary.name) / "candidate_a"
        with CheckpointManager(self.run_dir, run_id="candidate_a") as manager:
            manager.save_resume(
                epoch=1,
                model_state=_state(1.0),
                optimizer_state={},
                scheduler_state={},
                rng_states={"torch": torch.get_rng_state()},
            )
        resume = self.run_dir / "checkpoints" / "resume.pt"
        with resume.open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaisesRegex(StorageError, "byte size mismatch"):
            verify_visible_checkpoint_artifacts(
                self.run_dir, expected_run_id="candidate_a"
            )

    def test_full_verifier_rejects_hidden_reference_and_untracked_weight(self) -> None:
        self._write_complete_candidate()
        hidden_path = (
            self.run_dir
            / "checkpoints"
            / "exploratory_hidden"
            / "oracle_manifest.json"
        )
        hidden = json.loads(hidden_path.read_text(encoding="utf-8"))
        hidden["selectors"]["oracle"]["epoch"] = 999
        hidden_path.write_text(json.dumps(hidden) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(StorageError, "selector epoch"):
            verify_candidate_checkpoint_artifacts(
                self.run_dir, expected_run_id="candidate_a", require_complete=True
            )

        self.temporary.cleanup()
        self.temporary = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary.name) / "candidate_a"
        self._write_complete_candidate()
        weight = next((self.run_dir / "checkpoints" / "weights").glob("model_*.pt"))
        shutil.copyfile(weight, weight.parent / "model_untracked.pt")
        with self.assertRaisesRegex(StorageError, "exactly match"):
            verify_candidate_checkpoint_artifacts(
                self.run_dir, expected_run_id="candidate_a", require_complete=True
            )


if __name__ == "__main__":
    unittest.main()
