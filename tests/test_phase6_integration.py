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

from fixture_data import create_fixture, fixture_config
from setv.candidate.selectors import ExpertFusionInput
from setv.phase0 import approve_visual_audit, build_phase0
from setv.ula.analysis import build_phase6_analysis, verify_phase6
from setv.ula.config import load_phase6_config
from setv.utils.hashing import sha256_file


def _metric(name: str, score: float) -> dict:
    if name == "ordinary":
        return {"available": True, "accuracy": score, "loss": 1.0 - score}
    if name.startswith("hard_pseudogroup"):
        return {
            "available": True,
            "worst_nonempty_proxy_group_accuracy": score,
            "group_balanced_proxy_accuracy": score,
        }
    if name.endswith(".hard"):
        return {
            "available": True,
            "valid": True,
            "class_balanced_accuracy": score,
            "class_balanced_cross_entropy": 1.0 - score,
        }
    return {
        "available": True,
        "setv_score": score,
        "setv_loss": 1.0 - score,
    }


class Phase6IntegrationTests(unittest.TestCase):
    def test_freeze_precedes_reporting_only_test_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, masks = create_fixture(root)
            phase0 = root / "phase0"
            build_phase0(fixture_config(dataset, masks, phase0))
            approve_visual_audit(phase0, reviewer="phase6 fixture", confirmation=True)
            biased = np.genfromtxt(
                phase0 / "splits" / "waterbirds95_biased_val.csv",
                delimiter=",",
                names=True,
                dtype=None,
                encoding="utf-8",
            )
            ids = np.asarray([str(value) for value in biased["sample_id"]])
            labels = np.asarray(biased["y"], dtype=np.int64)
            proxy_root = root / "ula_proxy"
            (proxy_root / "scores").mkdir(parents=True)
            np.savez_compressed(
                proxy_root / "scores" / "proxy.npz",
                sample_id=ids.astype(np.str_),
                true_label=labels,
                ula_proxy_logits=np.eye(2, dtype=np.float32)[labels],
                ula_proxy_predicted_class=labels,
                ula_proxy_correct=np.ones(len(labels), dtype=np.uint8),
            )
            (proxy_root / "phase6_ula_proxy_receipt.json").write_text(
                json.dumps(
                    {
                        "scores": {"path": "scores/proxy.npz"},
                        "official_source": {
                            "repository": "https://github.com/tsirif/uLA",
                            "required_commit": "5867fb6e9a8485ed08b4cbe84900f2b5ac4fac5d",
                        },
                    }
                )
            )
            names = [
                "ordinary",
                "hard_pseudogroup.exact",
                *[
                    f"setv.{expert}.{fusion}"
                    for expert in ("exact", "sanitized", "set")
                    for fusion in ("hard", "rank", "logistic")
                ],
            ]
            candidate_root = root / "candidate"
            for seed_index, seed in enumerate((101, 202, 303)):
                candidate = candidate_root / f"seed_{seed}"
                for directory in (
                    "biased_val",
                    "selection",
                    "analysis_only",
                    "reporting_only",
                ):
                    (candidate / directory).mkdir(parents=True, exist_ok=True)
                oracle_curve = np.linspace(0.4, 0.9, 50)
                logits = np.zeros((50, len(ids), 2), dtype=np.float32)
                correct = np.zeros((50, len(ids)), dtype=np.uint8)
                for epoch in range(50):
                    predicted = labels if epoch >= 20 + seed_index else 1 - labels
                    logits[epoch, np.arange(len(ids)), predicted] = 2.0
                    correct[epoch] = predicted == labels
                np.savez_compressed(
                    candidate / "biased_val" / "epoch_predictions.npz",
                    epoch=np.arange(1, 51),
                    sample_id=ids.astype(np.str_),
                    true_label=labels,
                    candidate_logits=logits,
                    candidate_correct=correct,
                )
                selector_epochs = []
                selected = {}
                for epoch in range(1, 51):
                    selector_metrics = {}
                    for offset, name in enumerate(names):
                        score = 0.3 + 0.01 * epoch - 0.0001 * offset
                        selector_metrics[name] = _metric(name, score)
                    selector_epochs.append(
                        {
                            "epoch": epoch,
                            "ordinary": selector_metrics["ordinary"],
                            "selectors": selector_metrics,
                        }
                    )
                for name in names:
                    selected[name] = {"epoch": 50}
                (candidate / "biased_val" / "selector_metrics.json").write_text(
                    json.dumps({"epochs": selector_epochs})
                )
                (candidate / "selection" / "selection_receipt.json").write_text(
                    json.dumps({"realistic_selectors": selected})
                )
                oracle_epochs = [
                    {
                        "epoch": epoch,
                        "average_accuracy": float(value),
                        "group_balanced_accuracy": float(value),
                        "worst_group_accuracy": float(value),
                    }
                    for epoch, value in enumerate(oracle_curve, start=1)
                ]
                (candidate / "analysis_only" / "oracle_metrics.json").write_text(
                    json.dumps({"epochs": oracle_epochs})
                )
                (candidate / "analysis_only" / "oracle_selector.json").write_text(
                    json.dumps({"best": {"epoch": 50}})
                )
                (candidate / "reporting_only" / "test_metrics.json").write_text(
                    json.dumps(
                        {
                            "namespace": "reporting_only",
                            "epochs": oracle_epochs,
                        }
                    )
                )
                (candidate / "phase5_receipt.json").write_text(
                    json.dumps(
                        {
                            "seed": seed,
                            "phase0_dir": str(phase0),
                            "biased_val_predictions": {
                                "path": "biased_val/epoch_predictions.npz"
                            },
                        }
                    )
                )
            fusion_dirs = {}
            fusion_inputs = {}
            hard = np.asarray([index % 2 for index in range(len(ids))], dtype=np.uint8)
            u = np.linspace(0.1, 1.0, len(ids))
            for name in ("exact", "sanitized", "set"):
                path = root / f"fusion_{name}"
                path.mkdir()
                fusion_dirs[name] = str(path)
                fusion = {
                    "sample_id": ids,
                    "true_label": labels,
                    "hard_target": hard,
                    "hard_valid": True,
                    "rank": {
                        "object_percentile": u,
                        "background_percentile": u[::-1],
                        "q_rank": u,
                    },
                    "logistic": {
                        "available": True,
                        "q_logistic": u[::-1],
                        "diagnostics": {},
                    },
                }
                fusion_inputs[name] = ExpertFusionInput(
                    name=name,
                    fusion_dir=path,
                    fusion=fusion,
                    background_prediction=labels,
                )
            config = load_phase6_config(
                ROOT / "configs" / "phase6_analysis.yaml",
                phase0_dir=str(phase0),
                candidate_root=str(candidate_root),
                ula_proxy_dir=str(proxy_root),
                exact_fusion_dir=fusion_dirs["exact"],
                sanitized_fusion_dir=fusion_dirs["sanitized"],
                set_fusion_dir=fusion_dirs["set"],
                output_dir=str(root / "phase6"),
                candidate_seeds=[101, 202, 303],
            )

            def fake_materializer(candidate_dir, epoch, output_dir):
                output_dir.mkdir(parents=True)
                checkpoint = output_dir / f"ula_selected_epoch_{epoch:03d}.pt"
                checkpoint.write_bytes(b"fixture checkpoint")
                return {
                    "path": checkpoint.relative_to(output_dir.parents[1]).as_posix(),
                    "sha256": sha256_file(checkpoint),
                    "epoch": epoch,
                    "materialization_method": "fixture",
                }

            with patch("setv.ula.analysis.verify_candidate"), patch(
                "setv.ula.analysis.verify_ula_proxy"
            ), patch(
                "setv.ula.analysis.load_selector_inputs",
                return_value=fusion_inputs,
            ):
                output = build_phase6_analysis(
                    config, checkpoint_materializer=fake_materializer
                )
            verified = verify_phase6(output)
            self.assertEqual(verified["candidate_seed_count"], 3)
            freeze = json.loads(
                (output / "selection" / "method_freeze_receipt.json").read_text()
            )
            self.assertFalse(freeze["test_metrics_seen"])
            report = json.loads(
                (output / "reporting_only" / "test_results.json").read_text()
            )
            self.assertTrue(report["loaded_after_method_freeze"])
            self.assertEqual(
                report["method_freeze_receipt_sha256"],
                sha256_file(output / "selection" / "method_freeze_receipt.json"),
            )
            self.assertTrue(
                (output / "tables" / "primary_results_aggregate.csv").is_file()
            )
            self.assertTrue((output / "plots" / "mean_selection_regret.svg").is_file())
            self.assertTrue(
                (
                    output
                    / "plots"
                    / "candidate_disagreement_by_fusion_decile.svg"
                ).is_file()
            )
            self.assertTrue(
                (output / "plots" / "representative_setv_alpha_curves.svg").is_file()
            )
            kill = json.loads(
                (output / "analysis_only" / "kill_criteria.json").read_text()
            )
            self.assertEqual(kill["schema_version"], 1)
            self.assertIsInstance(
                kill["object_expert_near_chance"]["triggered"], bool
            )
            self.assertIsInstance(
                kill[
                    "rank_and_logistic_do_not_outperform_background_confidence_alone"
                ]["triggered_for_selected_expert"],
                bool,
            )
            self.assertTrue(
                (
                    output
                    / "analysis_only"
                    / "background_confidence_baselines.json"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
