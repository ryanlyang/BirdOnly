"""Phase 2 exact-fill background expert training and verification."""

from __future__ import annotations

import json
import os
import random
import uuid
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from setv.errors import ArtifactExistsError, DataValidationError
from setv.experts.exact_config import resolved_exact_config
from setv.experts.exact_data import ExactBackgroundDataset
from setv.experts.exact_scores import (
    build_exact_score_payload,
    load_exact_scores,
    save_exact_scores,
    validate_exact_score_payload,
)
from setv.experts.object_model import create_object_expert_model
from setv.experts.train_object import (
    _artifact_manifest,
    _atomic_torch_save,
    _build_optimizer_scheduler,
    _device,
    _evaluate_logits,
    _grad_scaler,
    _load_phase0_config,
    _source_root,
    _train_epoch,
    _write_csv_atomic,
)
from setv.phase0 import APPROVAL_RECEIPT, BASE_ARTIFACT_MANIFEST, verify_phase0
from setv.utils.hashing import sha256_file, sha256_json
from setv.utils.io import write_json
from setv.utils.logging import EventLogger
from setv.utils.provenance import runtime_provenance
from setv.utils.seeds import derive_seed, seed_python_numpy, seed_torch_if_available


ModelFactory = Callable[[dict[str, Any]], Any]


def _worker_init(worker_id: int, *, base_seed: int) -> None:
    import torch

    seed = derive_seed(base_seed, f"exact_worker={worker_id}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_loaders(phase0_dir: Path, phase0_config: dict, config: dict):
    import torch

    split_dir = phase0_dir / "splits"
    train_dataset = ExactBackgroundDataset(
        split_dir / "waterbirds95_candidate_train.csv",
        phase0_config,
        config,
        training=True,
    )
    validation_dataset = ExactBackgroundDataset(
        split_dir / "waterbirds95_biased_val.csv",
        phase0_config,
        config,
        training=False,
    )
    training = config["training"]
    seed = int(training["seed"])
    generator = torch.Generator()
    generator.manual_seed(derive_seed(seed, "exact_train_shuffle"))
    common = {
        "num_workers": int(training["num_workers"]),
        "pin_memory": training["device"] == "cuda",
        "persistent_workers": False,
        "worker_init_fn": partial(_worker_init, base_seed=seed),
    }
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
        drop_last=False,
        **common,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=int(training["evaluation_batch_size"]),
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_dataset, train_loader, validation_loader


def _prepare(config: dict, model_factory: ModelFactory | None):
    import torch

    phase0_dir = Path(config["phase0_dir"]).expanduser().resolve()
    verify_phase0(phase0_dir, require_approval=True)
    phase0_config = _load_phase0_config(phase0_dir)
    seed = int(config["training"]["seed"])
    seed_receipt = seed_python_numpy(seed)
    seed_receipt.update(seed_torch_if_available(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = _device(config)
    train_dataset, train_loader, validation_loader = _make_loaders(
        phase0_dir, phase0_config, config
    )
    model = (model_factory or create_object_expert_model)(config).to(device)
    pretrained_cfg = getattr(model, "pretrained_cfg", None)
    if isinstance(pretrained_cfg, dict):
        expected_mean = tuple(float(v) for v in config["model"]["normalization_mean"])
        expected_std = tuple(float(v) for v in config["model"]["normalization_std"])
        actual_mean = tuple(float(v) for v in pretrained_cfg.get("mean", expected_mean))
        actual_std = tuple(float(v) for v in pretrained_cfg.get("std", expected_std))
        if actual_mean != expected_mean or actual_std != expected_std:
            raise DataValidationError(
                "Exact-expert normalization differs from pretrained metadata"
            )
    pretrained_provenance = (
        {
            key: pretrained_cfg.get(key)
            for key in (
                "architecture",
                "tag",
                "hf_hub_id",
                "url",
                "input_size",
                "mean",
                "std",
            )
            if key in pretrained_cfg
        }
        if isinstance(pretrained_cfg, dict)
        else {"available": False, "reason": "injected model has no pretrained_cfg"}
    )
    return {
        "phase0_dir": phase0_dir,
        "phase0_config": phase0_config,
        "seed_receipt": seed_receipt,
        "device": device,
        "train_dataset": train_dataset,
        "train_loader": train_loader,
        "validation_loader": validation_loader,
        "model": model,
        "pretrained_provenance": pretrained_provenance,
    }


def _evaluate_exact(model, loader, device):
    raw = _evaluate_logits(model, loader, device)
    payload = build_exact_score_payload(raw["sample_id"], raw["labels"], raw["logits"])
    summary = validate_exact_score_payload(payload)
    metrics = {
        "biased_val_loss": raw["loss_sum"] / len(raw["sample_id"]),
        "biased_val_accuracy": summary["accuracy"],
        "biased_val_class_0_accuracy": summary["per_class_accuracy"]["0"],
        "biased_val_class_1_accuracy": summary["per_class_accuracy"]["1"],
        "biased_val_sample_count": len(raw["sample_id"]),
        "evaluation_seconds": raw["evaluation_seconds"],
    }
    return metrics, payload


def _warnings(summary: dict[str, Any]) -> list[str]:
    warnings = []
    if float(summary["accuracy"]) <= 0.5:
        warnings.append("background_exact_accuracy_not_above_binary_chance")
    if float(summary["margin"]["standard_deviation"]) <= 1e-8:
        warnings.append("background_exact_margin_has_negligible_variation")
    return warnings


def run_exact_expert_smoke(
    config: dict,
    report_path: str | Path,
    *,
    model_factory: ModelFactory | None = None,
) -> dict[str, Any]:
    import torch

    prepared = _prepare(config, model_factory)
    optimizer, scheduler = _build_optimizer_scheduler(
        prepared["model"], config, len(prepared["train_loader"])
    )
    mixed = bool(config["training"]["mixed_precision"]) and prepared["device"].type == "cuda"
    scaler = _grad_scaler(prepared["device"], mixed)
    train_metrics = _train_epoch(
        prepared["model"],
        prepared["train_loader"],
        prepared["train_dataset"],
        epoch=0,
        device=prepared["device"],
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        mixed_precision=mixed,
    )
    validation_metrics, payload = _evaluate_exact(
        prepared["model"], prepared["validation_loader"], prepared["device"]
    )
    expected = pd.read_csv(
        prepared["phase0_dir"] / "splits" / "waterbirds95_biased_val.csv",
        dtype={"sample_id": str},
    )
    summary = validate_exact_score_payload(payload, expected)
    report = {
        "schema_version": 1,
        "status": "passed",
        "kind": "background_exact_real_one_epoch_smoke",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(config["training"]["seed"]),
        "device": str(prepared["device"]),
        "cuda_device": (
            torch.cuda.get_device_name(0) if prepared["device"].type == "cuda" else None
        ),
        "train": train_metrics,
        "biased_val": validation_metrics,
        "score_summary": summary,
        "scientific_warnings": _warnings(summary),
        "projected_twenty_epoch_seconds": float(train_metrics["epoch_seconds"] * 20),
        "phase0_artifact_manifest_sha256": sha256_file(
            prepared["phase0_dir"] / BASE_ARTIFACT_MANIFEST
        ),
    }
    write_json(report_path, report)
    return report


def train_exact_expert(
    config: dict,
    *,
    model_factory: ModelFactory | None = None,
) -> Path:
    import torch

    seed = int(config["training"]["seed"])
    output_root = Path(config["output_root"]).expanduser().resolve()
    destination = output_root / f"seed_{seed}"
    if destination.exists():
        raise ArtifactExistsError(f"Exact-expert output exists: {destination}")
    output_root.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("SLURM_JOB_ID", uuid.uuid4().hex[:12])
    staging = output_root / f".seed_{seed}.building.{token}"
    if staging.exists():
        raise ArtifactExistsError(f"Exact-expert staging path exists: {staging}")
    for directory in ("metrics", "scores", "checkpoints", "config", "provenance"):
        (staging / directory).mkdir(parents=True, exist_ok=True)
    logger = EventLogger(staging / "training.jsonl", echo=True)
    logger.log("background_exact_training_started", seed=seed)
    try:
        prepared = _prepare(config, model_factory)
        optimizer, scheduler = _build_optimizer_scheduler(
            prepared["model"], config, len(prepared["train_loader"])
        )
        mixed = (
            bool(config["training"]["mixed_precision"])
            and prepared["device"].type == "cuda"
        )
        scaler = _grad_scaler(prepared["device"], mixed)
        rows = []
        final_payload = None
        for epoch in range(int(config["training"]["epochs"])):
            train_metrics = _train_epoch(
                prepared["model"],
                prepared["train_loader"],
                prepared["train_dataset"],
                epoch=epoch,
                device=prepared["device"],
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                mixed_precision=mixed,
            )
            validation_metrics, final_payload = _evaluate_exact(
                prepared["model"], prepared["validation_loader"], prepared["device"]
            )
            row = {"epoch": epoch + 1, **train_metrics, **validation_metrics}
            rows.append(row)
            _write_csv_atomic(
                staging / "metrics" / "epoch_metrics.csv", pd.DataFrame(rows)
            )
            logger.log("background_exact_epoch_complete", **row)
        if final_payload is None:
            raise DataValidationError("Exact expert produced no validation payload")
        expected = pd.read_csv(
            prepared["phase0_dir"] / "splits" / "waterbirds95_biased_val.csv",
            dtype={"sample_id": str},
        )
        summary = validate_exact_score_payload(final_payload, expected)
        score_path = staging / "scores" / config["storage"]["scores_filename"]
        save_exact_scores(score_path, final_payload)
        write_json(staging / "scores" / "background_exact_score_summary.json", summary)

        checkpoint_path = (
            staging / "checkpoints" / config["storage"]["checkpoint_filename"]
        )
        checkpoint = {
            "format_version": 1,
            "kind": "setv_background_exact",
            "architecture": config["model"]["architecture"],
            "seed": seed,
            "completed_epochs": int(config["training"]["epochs"]),
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in prepared["model"].state_dict().items()
            },
            "green_rgb": config["input"]["green_rgb"],
            "dilation_pixels_at_224": config["input"]["dilation_pixels_at_224"],
            "dilation_structuring_element": config["input"][
                "dilation_structuring_element"
            ],
            "pretrained_provenance": prepared["pretrained_provenance"],
            "phase0_artifact_manifest_sha256": sha256_file(
                prepared["phase0_dir"] / BASE_ARTIFACT_MANIFEST
            ),
        }
        _atomic_torch_save(checkpoint, checkpoint_path)

        resolved = resolved_exact_config(config)
        with (staging / "config" / "resolved_background_exact.yaml").open(
            "w", encoding="utf-8"
        ) as handle:
            yaml.safe_dump(resolved, handle, sort_keys=True)
        provenance = runtime_provenance(_source_root(), destination)
        provenance["seed_receipt"] = prepared["seed_receipt"]
        provenance["torch"] = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "device": str(prepared["device"]),
            "device_name": (
                torch.cuda.get_device_name(0)
                if prepared["device"].type == "cuda"
                else None
            ),
        }
        try:
            import timm

            provenance["timm_version"] = timm.__version__
        except ImportError:
            provenance["timm_version"] = None
        write_json(staging / "provenance" / "runtime.json", provenance)
        warnings = _warnings(summary)
        receipt = {
            "schema_version": 1,
            "status": "complete",
            "kind": "setv_background_exact",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "trained_on": "full candidate_train",
            "calibration_holdout": False,
            "temperature_scaling": False,
            "config_sha256": sha256_json(resolved),
            "phase0_dir": str(prepared["phase0_dir"]),
            "phase0_artifact_manifest_sha256": sha256_file(
                prepared["phase0_dir"] / BASE_ARTIFACT_MANIFEST
            ),
            "phase0_visual_approval_sha256": sha256_file(
                prepared["phase0_dir"] / APPROVAL_RECEIPT
            ),
            "checkpoint": {
                "path": checkpoint_path.relative_to(staging).as_posix(),
                "sha256": sha256_file(checkpoint_path),
            },
            "scores": {
                "path": score_path.relative_to(staging).as_posix(),
                "sha256": sha256_file(score_path),
                "summary": summary,
            },
            "scientific_warnings": warnings,
            "scientific_gate": "review_required" if warnings else "no_automatic_warning",
        }
        write_json(staging / "phase2_exact_receipt.json", receipt)
        logger.log("background_exact_training_complete", summary=summary)
        write_json(staging / "artifact_manifest.json", _artifact_manifest(staging))
        os.rename(staging, destination)
        return destination
    except Exception as exc:
        write_json(
            staging / "failure.json",
            {
                "status": "failed",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        logger.log(
            "background_exact_training_failed",
            exception_type=type(exc).__name__,
            message=str(exc),
        )
        raise


def verify_exact_expert(
    output_dir: str | Path, *, load_checkpoint: bool = True
) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "artifact_manifest.json"
    receipt_path = root / "phase2_exact_receipt.json"
    if not manifest_path.is_file() or not receipt_path.is_file():
        raise DataValidationError(f"Incomplete exact-expert output: {root}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if sha256_json(manifest["files"]) != manifest["manifest_digest"]:
        raise DataValidationError("Exact-expert artifact manifest digest is invalid")
    for relative, expected in manifest["files"].items():
        path = root / relative
        if not path.is_file() or path.stat().st_size != expected["size_bytes"]:
            raise DataValidationError(f"Missing/changed exact artifact: {relative}")
        if sha256_file(path) != expected["sha256"]:
            raise DataValidationError(f"Changed exact artifact hash: {relative}")
    with receipt_path.open("r", encoding="utf-8") as handle:
        receipt = json.load(handle)
    phase0_dir = Path(receipt["phase0_dir"])
    verify_phase0(phase0_dir, require_approval=True)
    if receipt["phase0_artifact_manifest_sha256"] != sha256_file(
        phase0_dir / BASE_ARTIFACT_MANIFEST
    ):
        raise DataValidationError("Exact expert Phase 0 binding changed")
    expected = pd.read_csv(
        phase0_dir / "splits" / "waterbirds95_biased_val.csv",
        dtype={"sample_id": str},
    )
    payload = load_exact_scores(root / receipt["scores"]["path"])
    summary = validate_exact_score_payload(payload, expected)
    if load_checkpoint:
        import torch

        checkpoint = torch.load(
            root / receipt["checkpoint"]["path"],
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint.get("kind") != "setv_background_exact":
            raise DataValidationError("Exact final checkpoint has wrong kind")
        if not checkpoint.get("model_state_dict"):
            raise DataValidationError("Exact final checkpoint state_dict is empty")
    return {
        "status": "complete",
        "seed": receipt["seed"],
        "artifact_count": len(manifest["files"]),
        "scores": summary,
        "checkpoint_loaded": load_checkpoint,
        "scientific_warnings": receipt["scientific_warnings"],
    }
