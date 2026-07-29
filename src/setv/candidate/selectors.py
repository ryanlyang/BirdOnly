"""Realistic Phase 5 selectors and their isolated online tie-breaking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from setv.candidate.metrics import ordinary_metrics, proxy_group_metrics
from setv.errors import DataValidationError
from setv.experts.exact_scores import load_exact_scores
from setv.experts.sanitized_scores import load_sanitized_scores
from setv.experts.set_scores import load_set_scores
from setv.fusion.artifacts import load_fusion_artifacts, verify_fusion_artifacts
from setv.fusion.core import score_candidate
from setv.fusion.sanitized_artifacts import verify_sanitized_fusion_artifacts
from setv.fusion.set_artifacts import verify_set_fusion_artifacts


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class ExpertFusionInput:
    name: str
    fusion_dir: Path
    fusion: dict[str, Any]
    background_prediction: np.ndarray
    # Production loaders retain the verified source artifact so Phase 6 can
    # derive tie-break evidence without accepting manually entered metrics.
    # Optional defaults keep small unit-test fixtures lightweight.
    expert_dir: Path | None = None
    background_scores: dict[str, np.ndarray] | None = None


def load_selector_inputs(
    fusion_dirs: dict[str, str], expected_ids: np.ndarray, expected_labels: np.ndarray
) -> dict[str, ExpertFusionInput]:
    loaders = {
        "exact": (
            verify_fusion_artifacts,
            "setv_exact_fusion",
            "exact_expert_dir",
            "phase2_exact_receipt.json",
            load_exact_scores,
            "background_exact_predicted_class",
        ),
        "sanitized": (
            verify_sanitized_fusion_artifacts,
            "setv_sanitized_fusion",
            "sanitized_expert_dir",
            "phase3_sanitized_receipt.json",
            load_sanitized_scores,
            "background_sanitized_predicted_class",
        ),
        "set": (
            verify_set_fusion_artifacts,
            "setv_set_fusion",
            "set_expert_dir",
            "phase4_set_receipt.json",
            load_set_scores,
            "background_set_predicted_class",
        ),
    }
    output = {}
    for name, (
        verifier,
        kind,
        expert_key,
        expert_receipt_name,
        score_loader,
        prediction_key,
    ) in loaders.items():
        root = Path(fusion_dirs[name]).expanduser().resolve()
        verifier(root)
        receipt = _json(root / "fusion_receipt.json")
        if receipt.get("kind") != kind:
            raise DataValidationError(f"{name} fusion has wrong artifact kind")
        fusion = load_fusion_artifacts(root)
        if not np.array_equal(fusion["sample_id"], expected_ids) or not np.array_equal(
            fusion["true_label"], expected_labels
        ):
            raise DataValidationError(f"{name} fusion does not align with biased_val")
        expert_dir = Path(receipt[expert_key])
        expert_receipt = _json(expert_dir / expert_receipt_name)
        scores = score_loader(expert_dir / expert_receipt["scores"]["path"])
        if not np.array_equal(scores["sample_id"].astype(str), expected_ids):
            raise DataValidationError(f"{name} background scores do not align")
        output[name] = ExpertFusionInput(
            name=name,
            fusion_dir=root,
            fusion=fusion,
            background_prediction=scores[prediction_key].astype(np.int64),
            expert_dir=expert_dir,
            background_scores=scores,
        )
    return output


def compute_realistic_selectors(
    candidate_payload: dict[str, np.ndarray],
    inputs: dict[str, ExpertFusionInput],
) -> dict[str, dict[str, Any]]:
    """Compute realistic selectors without accepting oracle/test inputs."""
    ordinary = ordinary_metrics(candidate_payload)
    logits = candidate_payload["logits"]
    labels = candidate_payload["true_label"]
    metrics: dict[str, dict[str, Any]] = {
        "ordinary": {"available": True, **ordinary}
    }
    for name in ("exact", "sanitized", "set"):
        source = inputs[name]
        proxy = proxy_group_metrics(candidate_payload, source.background_prediction)
        metrics[f"hard_pseudogroup.{name}"] = {"available": True, **proxy}
        scores = score_candidate(logits, labels, source.fusion)
        metrics[f"setv.{name}.hard"] = {
            "available": bool(scores["hard"]["valid"]),
            **scores["hard"],
        }
        metrics[f"setv.{name}.rank"] = {"available": True, **scores["rank"]}
        metrics[f"setv.{name}.logistic"] = scores["logistic"]
    return metrics


def _better_scalar(
    candidate: float, incumbent: float, *, maximize: bool, tolerance: float
) -> int:
    difference = candidate - incumbent
    if abs(difference) <= tolerance:
        return 0
    if maximize:
        return 1 if difference > 0 else -1
    return 1 if difference < 0 else -1


def selector_kind(name: str) -> str:
    if name == "ordinary":
        return "ordinary"
    if name.startswith("hard_pseudogroup."):
        return "pseudogroup"
    if name.endswith(".hard"):
        return "hard"
    return "continuous"


def is_better(
    name: str,
    candidate: dict[str, Any],
    incumbent: dict[str, Any] | None,
    *,
    tolerance: float,
) -> bool:
    if incumbent is None:
        return True
    kind = selector_kind(name)
    ordinary = candidate["ordinary"]
    old_ordinary = incumbent["ordinary"]
    if kind == "ordinary":
        criteria = [
            (ordinary["accuracy"], old_ordinary["accuracy"], True),
            (ordinary["loss"], old_ordinary["loss"], False),
        ]
    elif kind == "continuous":
        criteria = [
            (candidate["metrics"]["setv_score"], incumbent["metrics"]["setv_score"], True),
            (candidate["metrics"]["setv_loss"], incumbent["metrics"]["setv_loss"], False),
            (ordinary["accuracy"], old_ordinary["accuracy"], True),
            (ordinary["loss"], old_ordinary["loss"], False),
        ]
    elif kind == "hard":
        criteria = [
            (
                candidate["metrics"]["class_balanced_accuracy"],
                incumbent["metrics"]["class_balanced_accuracy"],
                True,
            ),
            (
                candidate["metrics"]["class_balanced_cross_entropy"],
                incumbent["metrics"]["class_balanced_cross_entropy"],
                False,
            ),
            (ordinary["accuracy"], old_ordinary["accuracy"], True),
        ]
    else:
        # The plan locks the proxy-group objective but not its ties. Phase 5
        # explicitly records this deterministic ordinary-validation fallback.
        criteria = [
            (
                candidate["metrics"]["worst_nonempty_proxy_group_accuracy"],
                incumbent["metrics"]["worst_nonempty_proxy_group_accuracy"],
                True,
            ),
            (ordinary["accuracy"], old_ordinary["accuracy"], True),
            (ordinary["loss"], old_ordinary["loss"], False),
        ]
    for value, old_value, maximize in criteria:
        decision = _better_scalar(
            float(value), float(old_value), maximize=maximize, tolerance=tolerance
        )
        if decision:
            return decision > 0
    return int(candidate["epoch"]) < int(incumbent["epoch"])


class RealisticSelectionTracker:
    def __init__(self, tolerance: float):
        self.tolerance = float(tolerance)
        self.best: dict[str, dict[str, Any]] = {}
        self.unavailable: dict[str, dict[str, Any]] = {}

    def update(
        self,
        epoch: int,
        metrics: dict[str, dict[str, Any]],
    ) -> list[str]:
        ordinary = metrics["ordinary"]
        changed = []
        for name, values in metrics.items():
            if not values.get("available", True):
                self.unavailable[name] = {
                    "available": False,
                    "reason": values.get("reason", "selector input unavailable"),
                }
                continue
            candidate = {
                "epoch": int(epoch),
                "metrics": values,
                "ordinary": {
                    "accuracy": ordinary["accuracy"],
                    "loss": ordinary["loss"],
                },
            }
            if is_better(
                name,
                candidate,
                self.best.get(name),
                tolerance=self.tolerance,
            ):
                self.best[name] = candidate
                changed.append(name)
        return changed


def oracle_is_better(
    candidate: dict[str, Any],
    incumbent: dict[str, Any] | None,
    *,
    tolerance: float,
) -> bool:
    if incumbent is None:
        return True
    for key in (
        "worst_group_accuracy",
        "group_balanced_accuracy",
        "average_accuracy",
    ):
        decision = _better_scalar(
            float(candidate["metrics"][key]),
            float(incumbent["metrics"][key]),
            maximize=True,
            tolerance=tolerance,
        )
        if decision:
            return decision > 0
    return int(candidate["epoch"]) < int(incumbent["epoch"])
