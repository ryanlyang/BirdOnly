"""Build, verify, and consume Phase 4 set-background fusion artifacts."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from setv.errors import ArtifactExistsError, DataValidationError
from setv.experts.scores import load_object_scores
from setv.experts.set_scores import load_set_scores
from setv.experts.train_object import _artifact_manifest, verify_object_expert
from setv.experts.train_set import verify_set_expert
from setv.fusion.artifacts import _atomic_npz, _load_json, load_fusion_artifacts
from setv.fusion.core import build_fusions, score_candidate
from setv.fusion.set_config import resolved_set_fusion_config
from setv.phase0 import BASE_ARTIFACT_MANIFEST, verify_phase0
from setv.utils.hashing import sha256_file, sha256_json
from setv.utils.io import write_json


def _sources(config: dict[str, Any]):
    phase0_dir = Path(config["phase0_dir"]).expanduser().resolve()
    object_dir = Path(config["object_expert_dir"]).expanduser().resolve()
    set_dir = Path(config["set_expert_dir"]).expanduser().resolve()
    verify_phase0(phase0_dir, require_approval=True)
    verify_object_expert(object_dir, load_checkpoint=False)
    verify_set_expert(set_dir, load_checkpoint=False)
    object_receipt = _load_json(object_dir / "phase1_receipt.json")
    set_receipt = _load_json(set_dir / "phase4_set_receipt.json")
    for name, receipt in (("object", object_receipt), ("set", set_receipt)):
        if Path(receipt["phase0_dir"]).resolve() != phase0_dir:
            raise DataValidationError(
                f"{name} expert references a different Phase 0"
            )
        warnings = receipt.get("scientific_warnings", [])
        if warnings and not config["allow_expert_sanity_warnings"]:
            raise DataValidationError(
                f"{name} expert has unresolved scientific warnings: {warnings}"
            )
    return phase0_dir, object_dir, set_dir, object_receipt, set_receipt


def build_set_fusion_artifacts(config: dict[str, Any]) -> Path:
    (
        phase0_dir,
        object_dir,
        set_dir,
        object_receipt,
        set_receipt,
    ) = _sources(config)
    object_scores = load_object_scores(object_dir / object_receipt["scores"]["path"])
    set_scores = load_set_scores(set_dir / set_receipt["scores"]["path"])
    manifest = pd.read_csv(
        phase0_dir / "splits" / "waterbirds95_biased_val.csv",
        dtype={"sample_id": str},
    )
    ids = manifest["sample_id"].astype(str).to_numpy()
    labels = manifest["y"].to_numpy(dtype=np.int64)
    if not np.array_equal(object_scores["sample_id"].astype(str), ids):
        raise DataValidationError("Object scores do not align with biased_val")
    if not np.array_equal(set_scores["sample_id"].astype(str), ids):
        raise DataValidationError("Set-background scores do not align with biased_val")
    if not np.array_equal(object_scores["true_label"], labels) or not np.array_equal(
        set_scores["true_label"], labels
    ):
        raise DataValidationError("Expert labels do not align with biased_val")

    fusion = build_fusions(
        object_scores["object_true_class_margin"],
        set_scores["background_set_true_class_margin"],
        labels,
        seed=int(config["fusion"]["seed"]),
        min_hard_examples_per_class=int(
            config["fusion"]["hard"]["min_examples_per_class"]
        ),
        clip_minimum=float(config["fusion"]["rank"]["clip_minimum"]),
    )
    object_seed = int(object_receipt["seed"])
    set_seed = int(set_receipt["seed"])
    fusion_seed = int(config["fusion"]["seed"])
    destination = (
        Path(config["output_root"]).expanduser().resolve()
        / f"object_{object_seed}_set_{set_seed}_fusion_{fusion_seed}"
    )
    if destination.exists():
        raise ArtifactExistsError(f"Set fusion output exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("SLURM_JOB_ID", uuid.uuid4().hex[:12])
    staging = destination.parent / f".{destination.name}.building.{token}"
    staging.mkdir()

    arrays = {
        "sample_id": ids.astype(np.str_),
        "true_label": labels,
        "object_true_class_margin": object_scores[
            "object_true_class_margin"
        ].astype(np.float32),
        "background_set_true_class_margin": set_scores[
            "background_set_true_class_margin"
        ].astype(np.float32),
        "hard_target": fusion["hard_target"].astype(np.uint8),
        **{key: value.astype(np.float32) for key, value in fusion["rank"].items()},
    }
    logistic = fusion["logistic"]
    if logistic["available"]:
        arrays.update(
            {
                "q_logistic": logistic["q_logistic"].astype(np.float32),
                "q_logistic_repeats": logistic["q_logistic_repeats"].astype(
                    np.float32
                ),
                "u_logistic": logistic["u_logistic"].astype(np.float32),
                "logistic_fold_assignments": logistic["fold_assignments"].astype(
                    np.int16
                ),
            }
        )
    score_path = staging / "set_fusion_scores.npz"
    _atomic_npz(score_path, arrays)
    diagnostics = {
        "hard_counts_per_class": fusion["hard_counts_per_class"],
        "hard_valid": fusion["hard_valid"],
        "rank": fusion["rank_diagnostics"],
        "logistic": {
            key: value
            for key, value in logistic.items()
            if key
            not in {
                "q_logistic",
                "q_logistic_repeats",
                "u_logistic",
                "fold_assignments",
            }
        },
        "logistic_auc_interpretation": (
            "ROC/PR AUC only diagnose approximation of the margin-defined hard "
            "target; robust-selection utility is evaluated later using Oracle "
            "regret and candidate-epoch ranking metrics."
        ),
    }
    write_json(staging / "fusion_diagnostics.json", diagnostics)
    resolved = resolved_set_fusion_config(config)
    with (staging / "resolved_fusion.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=True)
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "kind": "setv_set_fusion",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "object_seed": object_seed,
        "set_seed": set_seed,
        "fusion_seed": fusion_seed,
        "phase0_dir": str(phase0_dir),
        "phase0_artifact_manifest_sha256": sha256_file(
            phase0_dir / BASE_ARTIFACT_MANIFEST
        ),
        "object_expert_dir": str(object_dir),
        "object_scores_sha256": sha256_file(
            object_dir / object_receipt["scores"]["path"]
        ),
        "set_expert_dir": str(set_dir),
        "set_scores_sha256": sha256_file(set_dir / set_receipt["scores"]["path"]),
        "config_sha256": sha256_json(resolved),
        "fusion_scores": {
            "path": score_path.relative_to(staging).as_posix(),
            "sha256": sha256_file(score_path),
        },
        "hard_valid": fusion["hard_valid"],
        "logistic_available": logistic["available"],
        "calibration_holdout": False,
        "temperature_scaling": False,
        "candidate_input_contract": (
            "raw logits on untouched biased-validation images in Phase 0 order"
        ),
    }
    write_json(staging / "fusion_receipt.json", receipt)
    write_json(staging / "artifact_manifest.json", _artifact_manifest(staging))
    os.rename(staging, destination)
    return destination


def verify_set_fusion_artifacts(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    artifact = _load_json(root / "artifact_manifest.json")
    if sha256_json(artifact["files"]) != artifact["manifest_digest"]:
        raise DataValidationError("Set fusion manifest digest is invalid")
    for relative, expected in artifact["files"].items():
        item = root / relative
        if not item.is_file() or item.stat().st_size != expected["size_bytes"]:
            raise DataValidationError(f"Missing/changed set fusion: {relative}")
        if sha256_file(item) != expected["sha256"]:
            raise DataValidationError(f"Changed set fusion hash: {relative}")

    receipt = _load_json(root / "fusion_receipt.json")
    if receipt.get("kind") != "setv_set_fusion":
        raise DataValidationError("Set fusion receipt has wrong kind")
    phase0_dir = Path(receipt["phase0_dir"])
    verify_phase0(phase0_dir, require_approval=True)
    if receipt["phase0_artifact_manifest_sha256"] != sha256_file(
        phase0_dir / BASE_ARTIFACT_MANIFEST
    ):
        raise DataValidationError("Set fusion Phase 0 binding changed")
    object_dir = Path(receipt["object_expert_dir"])
    set_dir = Path(receipt["set_expert_dir"])
    verify_object_expert(object_dir, load_checkpoint=False)
    verify_set_expert(set_dir, load_checkpoint=False)
    object_receipt = _load_json(object_dir / "phase1_receipt.json")
    set_receipt = _load_json(set_dir / "phase4_set_receipt.json")
    if receipt["object_scores_sha256"] != sha256_file(
        object_dir / object_receipt["scores"]["path"]
    ):
        raise DataValidationError("Set fusion object-score source changed")
    if receipt["set_scores_sha256"] != sha256_file(
        set_dir / set_receipt["scores"]["path"]
    ):
        raise DataValidationError("Set fusion background-score source changed")

    fusion = load_fusion_artifacts(root)
    manifest = pd.read_csv(
        phase0_dir / "splits" / "waterbirds95_biased_val.csv",
        dtype={"sample_id": str},
    )
    if not np.array_equal(
        fusion["sample_id"], manifest["sample_id"].astype(str).to_numpy()
    ) or not np.array_equal(
        fusion["true_label"], manifest["y"].to_numpy(dtype=np.int64)
    ):
        raise DataValidationError("Set fusion does not align with biased_val")
    return {
        "status": "complete",
        "sample_count": len(fusion["sample_id"]),
        "hard_valid": fusion["hard_valid"],
        "logistic_available": fusion["logistic"]["available"],
        "artifact_count": len(artifact["files"]),
    }


def score_candidate_set_file(
    fusion_dir: str | Path, candidate_npz: str | Path
) -> dict[str, Any]:
    fusion = load_fusion_artifacts(fusion_dir)
    with np.load(candidate_npz, allow_pickle=False) as archive:
        required = {"sample_id", "true_label", "candidate_logits"}
        if set(archive.files) != required:
            raise DataValidationError(
                f"Candidate NPZ keys must be exactly {sorted(required)}"
            )
        ids = archive["sample_id"].astype(str)
        labels = archive["true_label"].astype(np.int64)
        logits = archive["candidate_logits"].astype(np.float64)
    if not np.array_equal(ids, fusion["sample_id"]) or not np.array_equal(
        labels, fusion["true_label"]
    ):
        raise DataValidationError("Candidate logits do not align with set fusion")
    return score_candidate(logits, labels, fusion)
