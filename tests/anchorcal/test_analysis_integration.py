from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from anchorcal.analysis import run_final_analysis  # noqa: E402
from anchorcal.anchor_artifacts import verify_anchor_artifacts  # noqa: E402
from anchorcal.branch_provenance import verify_branch_artifacts  # noqa: E402
from anchorcal.candidate_schema import (  # noqa: E402
    CANDIDATE_SCALAR_METRICS,
    candidate_per_example_shapes,
)
from anchorcal.campaign_verification import (  # noqa: E402
    _verify_checksum_bundle,
    verify_campaign_artifacts,
)
from anchorcal.checkpoints import CheckpointManager  # noqa: E402
from anchorcal.errors import AuditFailure, PreflightError, StorageError  # noqa: E402
from anchorcal.io import atomic_write_json, hash_object, sha256_file  # noqa: E402
from anchorcal.hidden_analysis import (  # noqa: E402
    REAL_QUALITY_TARGETS,
    _correlation_columns,
    _python,
)
from anchorcal.selector_analysis import run_selector_stage  # noqa: E402
from anchorcal.storage import (  # noqa: E402
    CandidateStorage,
    PredictionBatch,
    SampleMetadata,
)


def _prediction(labels: np.ndarray, correct: np.ndarray) -> PredictionBatch:
    prediction = np.where(correct, labels, 1 - labels).astype(np.int16)
    logits = np.full((len(labels), 2), -2.0, dtype=np.float32)
    logits[np.arange(len(labels)), prediction] = 2.0
    loss = np.where(correct, 0.02, 4.02).astype(np.float32)
    return PredictionBatch(logits, prediction, correct.astype(bool), loss)


class RealQualityCorrelationTests(unittest.TestCase):
    def test_preflight_checksum_bundle_rejects_incomplete_membership(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "preflight").mkdir()
            member = root / "preflight" / "report.json"
            member.write_text("{}\n", encoding="utf-8")
            (root / "preflight" / "preflight_artifacts.sha256").write_text(
                f"{sha256_file(member)}  preflight/report.json\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AuditFailure, "membership mismatch"):
                _verify_checksum_bundle(root)

    def test_full_correlation_matrix_is_computed_for_every_target(self) -> None:
        frame = pd.DataFrame(
            {
                "criterion": [0.0, 1.0, 2.0, 3.0],
                "test_wga": [0.0, 1.0, 2.0, 3.0],
                "test_accuracy": [3.0, 2.0, 1.0, 0.0],
                "oracle_wga": [0.0, 0.5, 0.5, 1.0],
            }
        )
        columns, nested = _correlation_columns(
            frame, "criterion", prefix="competent_pool_"
        )
        self.assertEqual(len(columns), 9)
        self.assertEqual(set(nested), set(REAL_QUALITY_TARGETS))
        self.assertEqual(
            set(nested["test_wga"]), {"pearson", "spearman", "kendall_tau_b"}
        )
        for method in ("pearson", "spearman", "kendall_tau_b"):
            self.assertAlmostEqual(nested["test_wga"][method], 1.0)
            self.assertAlmostEqual(nested["test_accuracy"][method], -1.0)

    def test_constant_correlations_are_null_and_strict_json_safe(self) -> None:
        frame = pd.DataFrame(
            {
                "criterion": [1.0, 1.0, 1.0],
                "test_wga": [0.1, 0.2, 0.3],
                "test_accuracy": [0.9, 0.8, 0.7],
                "oracle_wga": [0.2, 0.4, 0.6],
            }
        )
        columns, nested = _correlation_columns(frame, "criterion")
        self.assertTrue(all(value is None for value in columns.values()))
        serialized = json.dumps(_python(nested), allow_nan=False, sort_keys=True)
        self.assertNotIn("NaN", serialized)
        self.assertEqual(serialized.count("null"), 9)
        csv_text = pd.DataFrame([columns]).to_csv(index=False, lineterminator="\n")
        self.assertNotIn("nan", csv_text.lower())


class AnalysisIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import h5py  # noqa: F401
            import matplotlib  # noqa: F401
        except ImportError as error:
            self.skipTest(str(error))
        self.temporary = tempfile.TemporaryDirectory()
        self.output = Path(self.temporary.name)
        self.config_hash = "a" * 64
        self.config = {
            "paths": {"output_root": str(self.output)},
            "runtime": {"debug": True},
            "data": {"crop_fallback_rate_gate": 0.001},
            "resolved_config_sha256": self.config_hash,
            "candidate_grid": {
                "learning_rates": [3.0e-5],
                "weight_decays": [0.05],
                "seed": 1234,
                "epochs": 3,
            },
            "anchorcal": {
                "candidate_score_tolerance": 1.0e-8,
                "final_metric_bootstrap_replicates": 20,
                "lambdas": np.linspace(0.0, 1.0, 5).tolist(),
            },
            "criteria": {"swap_donors": 2, "blur_sigmas": [1.0, 2.0]},
            "branches": {
                "frozen_epoch": 30,
                "foreground_position_mode": "object_relative",
            },
            "seeds": {"final_metric_bootstrap": 7003},
        }
        self._write_preflight()
        self._write_splits()
        self._write_anchor_artifacts()
        self._write_candidate()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _resign_receipt(receipt: Path, payload: dict) -> None:
        receipt.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
        receipt.with_suffix(receipt.suffix + ".sha256").write_text(
            f"{digest}  {receipt.name}\n", encoding="ascii"
        )

    def _write_preflight(self) -> None:
        preflight_root = self.output / "preflight"
        preflight_root.mkdir(parents=True)
        preprocessing = preflight_root / "preprocessing_manifest.json"
        atomic_write_json(preprocessing, {"schema_version": "synthetic"})
        atomic_write_json(
            preflight_root / "report.json",
            {
                "status": "passed",
                "metadata_sha256": "b" * 64,
                "mask_bank_sha256": "c" * 64,
                "preprocessing": {"manifest_sha256": sha256_file(preprocessing)},
                "git": {"commit": "d" * 40},
            },
        )

    def _write_splits(self) -> None:
        root = self.output / "splits"
        root.mkdir(parents=True)
        biased = pd.DataFrame(
            {
                "img_id": [1, 2, 3, 4],
                "img_filename": ["a", "b", "c", "d"],
                "y": [0, 0, 1, 1],
                "place": [0, 0, 1, 1],
                "split": [0, 0, 0, 0],
            }
        )
        biased.to_csv(root / "waterbirds100_biased_val.csv", index=False)
        geometry = self.output / "debug" / "preflight" / "geometry"
        geometry.mkdir(parents=True)
        biased.to_csv(geometry / "selector_eval_subset.csv", index=False)
        atomic_write_json(geometry / "background_token_budget.json", {"token_budget": 64})
        hidden = pd.DataFrame(
            {
                "img_id": np.arange(10, 18),
                "img_filename": [str(value) for value in range(8)],
                "y": [0, 0, 0, 0, 1, 1, 1, 1],
                "place": [0, 0, 1, 1, 0, 0, 1, 1],
                "split": [1] * 8,
            }
        )
        hidden.to_csv(root / "waterbirds100_oracle_val.csv", index=False)
        hidden.assign(split=2, img_id=np.arange(20, 28)).to_csv(
            root / "waterbirds100_test.csv", index=False
        )

    def _write_anchor_artifacts(self) -> None:
        receipt_root = self.output / "debug" / "receipt"
        receipt_root.mkdir(parents=True)
        receipt = receipt_root / "anchorcal_decision_20260101T000000.000000Z.json"
        payload = {
            "schema_version": 1,
            "receipt_type": "anchorcal_criterion_decision",
            "config_sha256": {"resolved": self.config_hash},
            "decision": {
                "winner": "saliency_harmonic",
                "credible_set": ["saliency_harmonic", "token_swap_harmonic"],
            },
        }
        receipt.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
        receipt.with_suffix(receipt.suffix + ".sha256").write_text(
            f"{digest}  {receipt.name}\n", encoding="ascii"
        )
        anchors = self.output / "debug" / "anchors"
        anchors.mkdir(parents=True)
        lambdas = np.linspace(0, 1, 5)
        criteria = {
            "ordinary_accuracy": np.ones(5),
            "saliency_harmonic": np.linspace(0.55, 0.95, 5),
            "token_swap_harmonic": np.linspace(0.50, 0.85, 5),
            "background_blur_harmonic": np.linspace(0.50, 0.70, 5),
            "foreground_only_harmonic": np.linspace(0.45, 0.90, 5),
            "saliency_product": np.linspace(0.55, 0.95, 5),
            "swap_product": np.linspace(0.50, 0.85, 5),
            "blur_product": np.linspace(0.50, 0.70, 5),
        }
        result = {}
        for offset, (name, scores) in enumerate(criteria.items()):
            result[name] = {
                "scores": scores.tolist(),
                "metrics": {
                    "ace": 0.04 + offset * 0.01,
                    "kendall_tau_b": None if name == "ordinary_accuracy" else 1.0,
                    "spearman": None if name == "ordinary_accuracy" else 1.0,
                    "pair_accuracy": 0.5 if name == "ordinary_accuracy" else 1.0,
                    "adjacent_accuracy": 0.5 if name == "ordinary_accuracy" else 1.0,
                    "violations": 0,
                    "perfect_order": name != "ordinary_accuracy",
                    "ace_predictions": lambdas.tolist(),
                },
                "bootstrap": {"perfect_order_rate": {"mean": 0.8}},
            }
        atomic_write_json(anchors / "criterion_results.json", result)
        payload["decision"]["point_metrics"] = {
            name: result[name]["metrics"]
            for name in (
                "ordinary_accuracy",
                "saliency_harmonic",
                "token_swap_harmonic",
                "background_blur_harmonic",
            )
        }
        payload["provenance"] = {
            "criterion_results": str((anchors / "criterion_results.json").resolve()),
            "criterion_results_sha256": sha256_file(
                anchors / "criterion_results.json"
            ),
        }
        receipt.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
        receipt.with_suffix(receipt.suffix + ".sha256").write_text(
            f"{digest}  {receipt.name}\n", encoding="ascii"
        )
        np.savez_compressed(
            anchors / "anchor_bootstrap_score_vectors.npz",
            lambdas=lambdas,
            **{name: np.tile(scores, (20, 1)) for name, scores in criteria.items()},
        )
        atomic_write_json(
            anchors / "competence_intersection_manifest.json",
            {
                "count": 120,
                "retained_fraction": 0.6,
                "per_class_count": {"0": 60, "1": 60},
            },
        )
        for branch, accuracy in (("foreground", 0.8), ("background", 0.75)):
            branch_root = self.output / "debug" / "branches" / branch
            branch_root.mkdir(parents=True)
            checkpoint = branch_root / "epoch_final.pt"
            checkpoint.write_bytes(b"synthetic checkpoint\n")
            calibration = branch_root / "expert_calibration_outputs.npz"
            biased_output = branch_root / "biased_val_outputs.npz"
            np.savez_compressed(calibration, logits=np.zeros((2, 2)))
            np.savez_compressed(biased_output, logits=np.zeros((4, 2)))
            fallback = branch_root / "crop_fallback_events.json"
            atomic_write_json(
                fallback,
                {
                    "schema_version": "anchorcal-branch-crop-fallback-events-v1",
                    "branch": branch,
                    "event_count": 0,
                    "events": [],
                },
            )
            fallback_gate = branch_root / "crop_fallback_gate.json"
            atomic_write_json(
                fallback_gate,
                {
                    "schema_version": "anchorcal-branch-crop-fallback-gate-v1",
                    "branch": branch,
                    "maximum_rate": 0.001,
                    "status": "passed",
                    "fallback_count": 0,
                    "sampled_examples": 100,
                    "fallback_rate": 0.0,
                    "gate_scope": "all_sampled_training_examples_across_all_epochs",
                },
            )
            history = [
                {
                    "epoch": epoch,
                    "fallback_count": 0,
                    "sampled_examples": 100 if epoch == 1 else 0,
                    "fallback_events": [],
                }
                for epoch in range(1, 31)
            ]
            history_path = branch_root / "history.json"
            atomic_write_json(history_path, history)
            optimizer = {
                "decay_names": ["weight"],
                "no_decay_names": ["bias"],
                "decay_parameter_count": 1,
                "no_decay_parameter_count": 1,
            }
            optimizer_path = branch_root / "optimizer_groups.json"
            atomic_write_json(optimizer_path, optimizer)
            preflight = self.output / "preflight" / "report.json"
            preflight_payload = json.loads(preflight.read_text(encoding="utf-8"))
            preprocessing = self.output / "preflight" / "preprocessing_manifest.json"
            atomic_write_json(
                branch_root / "manifest.json",
                {
                    "schema_version": "anchorcal-branch-manifest-v3",
                    "branch": branch,
                    "fixed_epoch": 30,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "checkpoint_size_bytes": checkpoint.stat().st_size,
                    "token_budget": 64 if branch == "background" else None,
                    "foreground_position_mode": (
                        "object_relative" if branch == "foreground" else None
                    ),
                    "outputs": {
                        "expert_calibration": {
                            "path": str(calibration.resolve()),
                            "sha256": sha256_file(calibration),
                            "size_bytes": calibration.stat().st_size,
                        },
                        "biased_val": {
                            "path": str(biased_output.resolve()),
                            "sha256": sha256_file(biased_output),
                            "size_bytes": biased_output.stat().st_size,
                        },
                    },
                    "resolved_config_sha256": self.config_hash,
                    "preflight_report": str(preflight.resolve()),
                    "preflight_report_sha256": sha256_file(preflight),
                    "metadata_sha256": preflight_payload["metadata_sha256"],
                    "mask_bank_sha256": preflight_payload["mask_bank_sha256"],
                    "preprocessing_manifest_sha256": sha256_file(preprocessing),
                    "git_commit": preflight_payload["git"]["commit"],
                    "paths": self.config["paths"],
                    "optimizer": optimizer,
                    "training_artifacts": {
                        "history": {
                            "path": str(history_path.resolve()),
                            "sha256": sha256_file(history_path),
                            "size_bytes": history_path.stat().st_size,
                        },
                        "optimizer_groups": {
                            "path": str(optimizer_path.resolve()),
                            "sha256": sha256_file(optimizer_path),
                            "size_bytes": optimizer_path.stat().st_size,
                        },
                    },
                    "crop_fallback_events": str(fallback.resolve()),
                    "crop_fallback_events_sha256": sha256_file(fallback),
                    "crop_fallback_events_size_bytes": fallback.stat().st_size,
                    "crop_fallback_event_count": 0,
                    "crop_fallback_aggregate": {
                        "fallback_count": 0,
                        "sampled_examples": 100,
                        "fallback_rate": 0.0,
                        "gate_scope": "all_sampled_training_examples_across_all_epochs",
                    },
                    "crop_fallback_gate": str(fallback_gate.resolve()),
                    "crop_fallback_gate_sha256": sha256_file(fallback_gate),
                    "crop_fallback_gate_size_bytes": fallback_gate.stat().st_size,
                    "biased_val_competence": {
                        "point": accuracy,
                        "lower": accuracy - 0.05,
                        "upper": accuracy + 0.05,
                    },
                    "temperature": {
                        "nll_before": 0.6,
                        "nll_after": 0.5,
                        "temperature": 1.2,
                    },
                    "invalid_biased_val_fraction": 0.0,
                },
            )
        audits = self.output / "debug" / "audits"
        audits.mkdir(parents=True)
        atomic_write_json(audits / "branch_audits.json", {"synthetic": True})
        (anchors / "competence_intersection.csv").write_text(
            "img_id,y\n1,0\n2,1\n", encoding="utf-8"
        )
        (anchors / "criterion_subset.csv").write_text(
            "img_id,y\n1,0\n2,1\n", encoding="utf-8"
        )
        atomic_write_json(
            anchors / "margin_scales.json",
            {
                "foreground": 1.0,
                "background": 1.0,
                "normalization_epsilon": 1.0e-8,
                "criterion_subset_hash": hash_object([1, 2]),
            },
        )
        np.savez_compressed(anchors / "anchor_per_image_outputs.npz", labels=[0, 1])
        (anchors / "anchor_scores.csv").write_text(
            "criterion,lambda,score\n", encoding="utf-8"
        )
        (anchors / "anchor_bootstrap_metrics.csv").write_text(
            "criterion,metric,mean\n", encoding="utf-8"
        )
        (anchors / "anchor_intervention_diagnostics.csv").write_text(
            "lambda,swap_mean_true_class_margin_drop\n", encoding="utf-8"
        )
        atomic_write_json(anchors / "cache_parity.json", {"cases": {}})
        atomic_write_json(
            anchors / "foreground_stream_intervention_audit.json",
            {"all_passed": True},
        )
        artifact_paths = {
            "branch_audits": audits / "branch_audits.json",
            "competence_intersection": anchors / "competence_intersection.csv",
            "competence_intersection_manifest": anchors
            / "competence_intersection_manifest.json",
            "criterion_subset": anchors / "criterion_subset.csv",
            "margin_scales": anchors / "margin_scales.json",
            "anchor_per_image_outputs": anchors / "anchor_per_image_outputs.npz",
            "anchor_bootstrap_score_vectors": anchors
            / "anchor_bootstrap_score_vectors.npz",
            "criterion_results": anchors / "criterion_results.json",
            "anchor_scores": anchors / "anchor_scores.csv",
            "anchor_bootstrap_metrics": anchors / "anchor_bootstrap_metrics.csv",
            "anchor_intervention_diagnostics": anchors
            / "anchor_intervention_diagnostics.csv",
            "cache_parity": anchors / "cache_parity.json",
            "foreground_stream_intervention_audit": anchors
            / "foreground_stream_intervention_audit.json",
        }
        artifact_manifest = anchors / "artifact_manifest.json"
        atomic_write_json(
            artifact_manifest,
            {
                "schema_version": "anchorcal-anchor-artifacts-v1",
                "resolved_config_sha256": self.config_hash,
                "criterion_result_keys": sorted(criteria),
                "definitions": {
                    "margin_normalization_epsilon": 1.0e-8,
                    "harmonic_mean_epsilon": 1.0e-8,
                },
                "files": {
                    name: {
                        "path": str(path.resolve().relative_to(self.output.resolve())),
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for name, path in artifact_paths.items()
                },
            },
        )
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        payload.update(
            {
                "anchor_subset_sha256": hash_object([1, 2]),
                "anchor_family": {
                    "type": "normalized_centered_raw_logit_reliance",
                    "lambdas": self.config["anchorcal"]["lambdas"],
                    "foreground_scale": 1.0,
                    "background_scale": 1.0,
                    "normalization_epsilon": 1.0e-8,
                    "harmonic_mean_epsilon": 1.0e-8,
                },
                "branch_sha256": {
                    branch: sha256_file(
                        self.output
                        / "debug"
                        / "branches"
                        / branch
                        / "epoch_final.pt"
                    )
                    for branch in ("background", "foreground")
                },
            }
        )
        payload.setdefault("provenance", {}).update(
            {
                "anchor_artifact_manifest": str(artifact_manifest.resolve()),
                "anchor_artifact_manifest_sha256": sha256_file(artifact_manifest),
                "criterion_results_sha256": sha256_file(
                    anchors / "criterion_results.json"
                ),
            }
        )
        receipt.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
        receipt.with_suffix(receipt.suffix + ".sha256").write_text(
            f"{digest}  {receipt.name}\n", encoding="ascii"
        )

    def _write_candidate(self) -> None:
        run_id = "lr3e-5_wd0.05_seed1234"
        run = self.output / "debug" / "candidates" / run_id
        run.mkdir(parents=True)
        preflight = self.output / "preflight" / "report.json"
        anchor_receipt = next((self.output / "debug" / "receipt").glob("anchorcal_decision_*.json"))
        atomic_write_json(
            run / "run_manifest.json",
            {
                "schema_version": "anchorcal-candidate-run-v2",
                "run_id": run_id,
                "learning_rate": 3.0e-5,
                "weight_decay": 0.05,
                "seed": 1234,
                "decision_receipt": str(anchor_receipt.resolve()),
                "decision_receipt_sha256": sha256_file(anchor_receipt),
                "resolved_config_sha256": self.config_hash,
                "preflight_report": str(preflight.resolve()),
                "preflight_report_sha256": sha256_file(preflight),
            },
        )
        biased_labels = np.asarray([0, 0, 1, 1])
        hidden_labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
        places = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
        groups = hidden_labels * 2 + places
        selector_shapes = candidate_per_example_shapes(
            4, swap_donors=2, blur_sigmas=2
        )
        with CandidateStorage(
            run,
            run_id=run_id,
            epoch_capacity=3,
            num_classes=2,
            selector_metadata=SampleMetadata(np.arange(1, 5), biased_labels),
            hidden_metadata={
                "oracle_val": SampleMetadata(np.arange(10, 18), hidden_labels, groups),
                "test": SampleMetadata(np.arange(20, 28), hidden_labels, groups),
            },
            selector_metric_names=CANDIDATE_SCALAR_METRICS,
            selector_auxiliary_metadata=SampleMetadata(
                np.arange(1, 5), biased_labels
            ),
            selector_array_shapes=selector_shapes,
        ) as storage:
            patterns = [
                np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=bool),
                np.asarray([1, 1, 1, 0, 1, 1, 1, 0], dtype=bool),
                np.ones(8, dtype=bool),
            ]
            for slot, correct in enumerate(patterns):
                selector_correct = np.asarray([1, 0, 1, slot > 0], dtype=bool)
                scalar = 0.55 + 0.1 * slot
                storage.write_epoch(
                    slot=slot,
                    epoch_number=slot + 1,
                    selector=_prediction(biased_labels, selector_correct),
                    hidden={
                        "oracle_val": _prediction(hidden_labels, correct),
                        "test": _prediction(hidden_labels, correct),
                    },
                    selector_metrics={
                        "ordinary_accuracy": scalar,
                        "saliency_harmonic": scalar + 0.03,
                        "token_swap_harmonic": scalar + 0.02,
                        "background_blur_harmonic": scalar + 0.01,
                        "foreground_only_harmonic": scalar,
                        "saliency_alignment": scalar,
                        "swap_accuracy": scalar,
                        "blur_accuracy": scalar,
                        "foreground_only_accuracy": scalar,
                        "saliency_product": scalar * scalar,
                        "swap_product": scalar * scalar,
                        "blur_product": scalar * scalar,
                        "swap_mean_true_class_margin_drop": 0.1,
                        "swap_prediction_flip_rate": 0.1,
                        "swap_donor_margin_variance": 0.01,
                        "biased_mean_loss": 1.0 - scalar,
                    },
                    selector_per_example={
                        name: np.zeros(shape, dtype=np.float32)
                        for name, shape in selector_shapes.items()
                    },
                )
            storage.finalize()
        state = {"weight": torch.tensor([3.0], dtype=torch.float32)}
        with CheckpointManager(run, run_id=run_id) as manager:
            for selector in ("ordinary", "saliency", "swap", "blur"):
                manager.update_selector(
                    selector,
                    epoch=3,
                    model_state=state,
                    ranking_key=(0.8, 0.7, -0.2, -3),
                )
            manager.update_selector(
                "oracle", epoch=3, model_state=state, ranking_key=(1.0, 1.0, 1.0, -3)
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
        visible_checkpoint = run / "checkpoints" / "manifest.json"
        hidden_checkpoint = (
            run / "checkpoints" / "exploratory_hidden" / "oracle_manifest.json"
        )
        atomic_write_json(
            run / "completion.json",
            {
                "run_id": run_id,
                "epochs": 3,
                "checkpoint_manifest_sha256": sha256_file(visible_checkpoint),
                "hidden_checkpoint_manifest_sha256": sha256_file(hidden_checkpoint),
            },
        )

    def test_end_to_end_freeze_join_reports_and_null_correlations(self) -> None:
        summary = run_final_analysis(self.config)
        self.assertEqual(summary["candidate_count"], 3)
        self.assertEqual(summary["run_count"], 1)
        analysis = self.output / "debug" / "analysis"
        for relative in (
            "anchorcal_summary.csv",
            "selected_candidates.csv",
            "criterion_real_quality.csv",
            "tables/branch_table.csv",
            "tables/anchorcal_decision_table.csv",
            "figures/predicted_lambda_vs_true.png",
            "figures/competence_subset_diagnostics.png",
            "manifest.json",
        ):
            self.assertTrue((analysis / relative).is_file(), relative)
        selected = pd.read_csv(analysis / "selected_candidates.csv")
        self.assertIn("oracle_validation", set(selected["criterion"]))
        self.assertIn("oracle_wga_ci_low", selected)
        self.assertIn("paired_test_regret_ci_high", selected)
        quality = pd.read_csv(analysis / "criterion_real_quality.csv")
        for target in REAL_QUALITY_TARGETS:
            for method in ("pearson", "spearman", "kendall_tau_b"):
                self.assertIn(f"competent_pool_{method}_{target}", quality)
        self.assertTrue((quality["within_run_competent_valid"] == 0).all())
        self.assertTrue((quality["within_run_competent_na"] == 1).all())
        competent_summary = json.loads(
            (analysis / "competent_pool_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            set(competent_summary["correlations"]),
            {
                "ordinary_accuracy",
                "saliency_harmonic",
                "token_swap_harmonic",
                "background_blur_harmonic",
            },
        )
        self.assertTrue(
            all(
                value is None
                for criterion in competent_summary["correlations"].values()
                for target in criterion.values()
                for value in target.values()
            )
        )
        self.assertTrue(
            all(
                list(by_run.values()) == [None]
                for by_run in competent_summary[
                    "secondary_within_run_correlations_by_run"
                ].values()
            )
        )
        serialized = (analysis / "summary.json").read_text(encoding="utf-8")
        self.assertNotIn("NaN", serialized)
        # A reporting-stage retry must reuse the already frozen selector
        # receipt rather than either failing or publishing a second choice.
        repeated = run_final_analysis(self.config)
        self.assertEqual(repeated["candidate_count"], 3)
        self.assertEqual(
            len(list((self.output / "debug" / "receipt").glob("candidate_selection_*.json"))),
            1,
        )
        verified = verify_campaign_artifacts(self.config)
        self.assertEqual(verified["status"], "complete")
        self.assertEqual(verified["candidate_count"], 3)

    def test_hidden_join_rejects_anchor_table_changed_after_decision(self) -> None:
        run_selector_stage(self.config)
        path = self.output / "debug" / "anchors" / "criterion_results.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["saliency_harmonic"]["metrics"]["ace"] += 0.01
        atomic_write_json(path, payload)
        with self.assertRaisesRegex(PreflightError, "changed after|failed verification"):
            run_final_analysis(self.config)

    def test_selector_rejects_deleted_retained_checkpoint(self) -> None:
        weight = next(
            (self.output / "debug" / "candidates").glob("*/checkpoints/weights/model_*.pt")
        )
        weight.unlink()
        with self.assertRaisesRegex(StorageError, "missing"):
            run_selector_stage(self.config)

    def test_hidden_stage_rejects_checkpoint_manifest_changed_after_receipt(self) -> None:
        run_selector_stage(self.config)
        manifest = next(
            (self.output / "debug" / "candidates").glob("*/checkpoints/manifest.json")
        )
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["updated_unix_ns"] += 1
        atomic_write_json(manifest, payload)
        with self.assertRaisesRegex(PreflightError, "changed after selection"):
            run_final_analysis(self.config)

    def test_branch_position_mode_tamper_is_rejected(self) -> None:
        path = self.output / "debug" / "branches" / "foreground" / "manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["foreground_position_mode"] = None
        atomic_write_json(path, payload)
        with self.assertRaisesRegex(PreflightError, "provenance mismatch"):
            verify_branch_artifacts(self.config, "foreground")

    def test_branch_history_and_optimizer_group_hashes_are_required(self) -> None:
        branch_root = self.output / "debug" / "branches" / "foreground"
        history = branch_root / "history.json"
        history.write_text("[]\n", encoding="utf-8")
        with self.assertRaisesRegex(PreflightError, "training history"):
            verify_branch_artifacts(self.config, "foreground")

    def test_decision_receipt_binds_branches_subset_lambdas_and_scales(self) -> None:
        receipt = next(
            (self.output / "debug" / "receipt").glob("anchorcal_decision_*.json")
        )
        original = json.loads(receipt.read_text(encoding="utf-8"))

        def wrong_branch(payload: dict) -> None:
            payload["branch_sha256"]["foreground"] = "0" * 64

        def wrong_subset(payload: dict) -> None:
            payload["anchor_subset_sha256"] = "1" * 64

        def wrong_lambdas(payload: dict) -> None:
            payload["anchor_family"]["lambdas"][1] = 0.123

        def wrong_scale(payload: dict) -> None:
            payload["anchor_family"]["foreground_scale"] = 9.0

        for name, mutate in (
            ("branch", wrong_branch),
            ("subset", wrong_subset),
            ("lambdas", wrong_lambdas),
            ("scale", wrong_scale),
        ):
            with self.subTest(name=name):
                payload = json.loads(json.dumps(original))
                mutate(payload)
                self._resign_receipt(receipt, payload)
                with self.assertRaisesRegex(PreflightError, "frozen decision receipt"):
                    verify_anchor_artifacts(self.config, decision_receipt=receipt)
        self._resign_receipt(receipt, original)


if __name__ == "__main__":
    unittest.main()
