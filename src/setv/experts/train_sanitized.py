"""Phase 3 paired-view sanitized-background expert training."""

from __future__ import annotations

import json
import os
import random
import time
import uuid
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from setv.errors import ArtifactExistsError, DataValidationError
from setv.experts.object_model import create_object_expert_model
from setv.experts.sanitized_bank import verify_sanitized_mask_bank
from setv.experts.sanitized_config import resolved_config
from setv.experts.sanitized_data import SanitizedBackgroundDataset
from setv.experts.sanitized_scores import (
    build_sanitized_score_payload,
    load_sanitized_scores,
    save_sanitized_scores,
    validate_sanitized_score_payload,
)
from setv.experts.train_object import (
    _artifact_manifest,
    _atomic_torch_save,
    _autocast,
    _build_optimizer_scheduler,
    _device,
    _grad_scaler,
    _load_phase0_config,
    _step_optimizer_and_scheduler,
    _source_root,
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

    seed = derive_seed(base_seed, f"sanitized_worker={worker_id}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_loaders(phase0_dir: Path, phase0_config: dict, config: dict):
    import torch

    split_dir = phase0_dir / "splits"
    train_dataset = SanitizedBackgroundDataset(
        split_dir / "waterbirds95_candidate_train.csv",
        "candidate_train",
        phase0_config,
        config,
        training=True,
    )
    validation_dataset = SanitizedBackgroundDataset(
        split_dir / "waterbirds95_biased_val.csv",
        "biased_val",
        phase0_config,
        config,
        training=False,
    )
    training = config["training"]
    seed = int(training["seed"])
    generator = torch.Generator()
    generator.manual_seed(derive_seed(seed, "sanitized_train_shuffle"))
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
    mask_bank_dir = Path(config["mask_bank_dir"]).expanduser().resolve()
    verify_phase0(phase0_dir, require_approval=True)
    bank_verification = verify_sanitized_mask_bank(
        mask_bank_dir, require_accepted=True, verify_containment=True
    )
    with (mask_bank_dir / "sanitized_mask_bank_receipt.json").open(
        "r", encoding="utf-8"
    ) as handle:
        bank_receipt = json.load(handle)
    if Path(bank_receipt["phase0_dir"]).resolve() != phase0_dir:
        raise DataValidationError("Sanitized bank references a different Phase 0")
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
                "Sanitized-expert normalization differs from pretrained metadata"
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
        "mask_bank_dir": mask_bank_dir,
        "bank_receipt": bank_receipt,
        "bank_verification": bank_verification,
        "phase0_config": phase0_config,
        "seed_receipt": seed_receipt,
        "device": device,
        "train_dataset": train_dataset,
        "train_loader": train_loader,
        "validation_loader": validation_loader,
        "model": model,
        "pretrained_provenance": pretrained_provenance,
    }


def _train_epoch(
    model,
    loader,
    dataset,
    *,
    epoch: int,
    device,
    optimizer,
    scheduler,
    scaler,
    mixed_precision: bool,
    lambda_consistency: float,
) -> dict[str, float]:
    import torch

    dataset.set_epoch(epoch)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    total_sum = 0.0
    ce_sum = 0.0
    consistency_sum = 0.0
    correct = 0
    view_count = 0
    sample_count = 0
    optimizer_step_count = 0
    amp_skipped_step_count = 0
    started = time.monotonic()
    for batch in loader:
        image_a = batch["image_a"].to(device, non_blocking=True)
        image_b = batch["image_b"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, mixed_precision):
            joined = model(torch.cat([image_a, image_b], dim=0))
            logits_a, logits_b = joined.chunk(2, dim=0)
            ce = 0.5 * (criterion(logits_a, targets) + criterion(logits_b, targets))
            log_a = torch.log_softmax(logits_a, dim=1)
            log_b = torch.log_softmax(logits_b, dim=1)
            probability_a = log_a.exp()
            probability_b = log_b.exp()
            symmetric_kl = 0.5 * (
                (probability_a * (log_a - log_b)).sum(dim=1).mean()
                + (probability_b * (log_b - log_a)).sum(dim=1).mean()
            )
            loss = ce + lambda_consistency * symmetric_kl
        scaler.scale(loss).backward()
        if _step_optimizer_and_scheduler(
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
        ):
            optimizer_step_count += 1
        else:
            amp_skipped_step_count += 1
        batch_size = int(targets.numel())
        total_sum += float(loss.detach()) * batch_size
        ce_sum += float(ce.detach()) * batch_size
        consistency_sum += float(symmetric_kl.detach()) * batch_size
        correct += int((logits_a.detach().argmax(dim=1) == targets).sum())
        correct += int((logits_b.detach().argmax(dim=1) == targets).sum())
        view_count += 2 * batch_size
        sample_count += batch_size
    if optimizer_step_count == 0:
        raise DataValidationError("AMP skipped every optimizer update in the epoch")
    return {
        "train_loss": total_sum / sample_count,
        "train_cross_entropy": ce_sum / sample_count,
        "train_symmetric_kl": consistency_sum / sample_count,
        "train_view_accuracy": correct / view_count,
        "train_sample_count": sample_count,
        "train_optimizer_step_count": optimizer_step_count,
        "train_amp_skipped_step_count": amp_skipped_step_count,
        "epoch_seconds": time.monotonic() - started,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }


def _evaluate(model, loader, device):
    import torch

    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    sample_ids = []
    labels = []
    view_logits_parts = []
    loss_sum = 0.0
    started = time.monotonic()
    with torch.inference_mode():
        for batch in loader:
            images = batch["images"]
            batch_size, views = images.shape[:2]
            if views != 8:
                raise DataValidationError("Sanitized validation requires all eight masks")
            targets = batch["target"].to(device, non_blocking=True)
            logits = model(
                images.reshape(batch_size * views, *images.shape[2:]).to(
                    device, non_blocking=True
                )
            ).reshape(batch_size, views, 2)
            mean_logits = logits.mean(dim=1)
            loss_sum += float(criterion(mean_logits, targets))
            sample_ids.extend(str(value) for value in batch["sample_id"])
            labels.append(targets.cpu().numpy())
            view_logits_parts.append(logits.float().cpu().numpy())
    labels_array = np.concatenate(labels)
    view_logits = np.concatenate(view_logits_parts)
    payload = build_sanitized_score_payload(sample_ids, labels_array, view_logits)
    summary = validate_sanitized_score_payload(payload)
    metrics = {
        "biased_val_loss": loss_sum / len(sample_ids),
        "biased_val_accuracy": summary["accuracy"],
        "biased_val_class_0_accuracy": summary["per_class_accuracy"]["0"],
        "biased_val_class_1_accuracy": summary["per_class_accuracy"]["1"],
        "biased_val_sample_count": len(sample_ids),
        "evaluation_seconds": time.monotonic() - started,
    }
    return metrics, payload


def _warnings(summary: dict[str, Any]) -> list[str]:
    warnings = []
    if float(summary["accuracy"]) <= 0.5:
        warnings.append("background_sanitized_accuracy_not_above_binary_chance")
    if float(summary["margin"]["standard_deviation"]) <= 1e-8:
        warnings.append("background_sanitized_margin_has_negligible_variation")
    return warnings


def run_sanitized_expert_smoke(
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
        lambda_consistency=float(config["training"]["lambda_consistency"]),
    )
    validation_metrics, payload = _evaluate(
        prepared["model"], prepared["validation_loader"], prepared["device"]
    )
    expected = pd.read_csv(
        prepared["phase0_dir"] / "splits" / "waterbirds95_biased_val.csv",
        dtype={"sample_id": str},
    )
    summary = validate_sanitized_score_payload(payload, expected)
    report = {
        "schema_version": 1,
        "status": "passed",
        "kind": "background_sanitized_real_one_epoch_smoke",
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
        "mask_bank_artifact_manifest_sha256": sha256_file(
            prepared["mask_bank_dir"] / "artifact_manifest.json"
        ),
    }
    write_json(report_path, report)
    return report


def train_sanitized_expert(
    config: dict,
    *,
    model_factory: ModelFactory | None = None,
) -> Path:
    import torch

    seed = int(config["training"]["seed"])
    output_root = Path(config["output_root"]).expanduser().resolve()
    destination = output_root / f"seed_{seed}"
    if destination.exists():
        raise ArtifactExistsError(f"Sanitized-expert output exists: {destination}")
    output_root.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("SLURM_JOB_ID", uuid.uuid4().hex[:12])
    staging = output_root / f".seed_{seed}.building.{token}"
    for directory in ("metrics", "scores", "checkpoints", "config", "provenance"):
        (staging / directory).mkdir(parents=True, exist_ok=True)
    logger = EventLogger(staging / "training.jsonl", echo=True)
    logger.log("background_sanitized_training_started", seed=seed)
    try:
        prepared = _prepare(config, model_factory)
        optimizer, scheduler = _build_optimizer_scheduler(
            prepared["model"], config, len(prepared["train_loader"])
        )
        mixed = bool(config["training"]["mixed_precision"]) and prepared["device"].type == "cuda"
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
                lambda_consistency=float(config["training"]["lambda_consistency"]),
            )
            validation_metrics, final_payload = _evaluate(
                prepared["model"], prepared["validation_loader"], prepared["device"]
            )
            row = {"epoch": epoch + 1, **train_metrics, **validation_metrics}
            rows.append(row)
            _write_csv_atomic(
                staging / "metrics" / "epoch_metrics.csv", pd.DataFrame(rows)
            )
            logger.log("background_sanitized_epoch_complete", **row)
        if final_payload is None:
            raise DataValidationError("Sanitized expert produced no validation payload")
        expected = pd.read_csv(
            prepared["phase0_dir"] / "splits" / "waterbirds95_biased_val.csv",
            dtype={"sample_id": str},
        )
        summary = validate_sanitized_score_payload(final_payload, expected)
        score_path = staging / "scores" / config["storage"]["scores_filename"]
        save_sanitized_scores(score_path, final_payload)
        write_json(
            staging / "scores" / "background_sanitized_score_summary.json", summary
        )
        checkpoint_path = (
            staging / "checkpoints" / config["storage"]["checkpoint_filename"]
        )
        checkpoint = {
            "format_version": 1,
            "kind": "setv_background_sanitized",
            "architecture": config["model"]["architecture"],
            "seed": seed,
            "completed_epochs": int(config["training"]["epochs"]),
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in prepared["model"].state_dict().items()
            },
            "lambda_consistency": config["training"]["lambda_consistency"],
            "validation_view_count": 8,
            "mask_bank_artifact_manifest_sha256": sha256_file(
                prepared["mask_bank_dir"] / "artifact_manifest.json"
            ),
            "pretrained_provenance": prepared["pretrained_provenance"],
        }
        _atomic_torch_save(checkpoint, checkpoint_path)
        resolved = resolved_config(config)
        with (staging / "config" / "resolved_background_sanitized.yaml").open(
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
            "kind": "setv_background_sanitized",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "trained_on": "full candidate_train",
            "training_views": "two_distinct_masks_per_image",
            "validation_views": 8,
            "validation_aggregation": "mean_raw_logits_once",
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
            "mask_bank_dir": str(prepared["mask_bank_dir"]),
            "mask_bank_seed": prepared["bank_receipt"]["seed"],
            "mask_bank_artifact_manifest_sha256": sha256_file(
                prepared["mask_bank_dir"] / "artifact_manifest.json"
            ),
            "leakage_audit_accepted": True,
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
        write_json(staging / "phase3_sanitized_receipt.json", receipt)
        logger.log("background_sanitized_training_complete", summary=summary)
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
            "background_sanitized_training_failed",
            exception_type=type(exc).__name__,
            message=str(exc),
        )
        raise


def verify_sanitized_expert(
    output_dir: str | Path, *, load_checkpoint: bool = True
) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "artifact_manifest.json"
    receipt_path = root / "phase3_sanitized_receipt.json"
    if not manifest_path.is_file() or not receipt_path.is_file():
        raise DataValidationError(f"Incomplete sanitized-expert output: {root}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if sha256_json(manifest["files"]) != manifest["manifest_digest"]:
        raise DataValidationError("Sanitized-expert manifest digest is invalid")
    for relative, expected_hash in manifest["files"].items():
        path = root / relative
        if not path.is_file() or path.stat().st_size != expected_hash["size_bytes"]:
            raise DataValidationError(f"Missing/changed sanitized artifact: {relative}")
        if sha256_file(path) != expected_hash["sha256"]:
            raise DataValidationError(f"Changed sanitized artifact hash: {relative}")
    with receipt_path.open("r", encoding="utf-8") as handle:
        receipt = json.load(handle)
    phase0_dir = Path(receipt["phase0_dir"])
    verify_phase0(phase0_dir, require_approval=True)
    if receipt["phase0_artifact_manifest_sha256"] != sha256_file(
        phase0_dir / BASE_ARTIFACT_MANIFEST
    ):
        raise DataValidationError("Sanitized expert Phase 0 binding changed")
    mask_bank_dir = Path(receipt["mask_bank_dir"])
    verify_sanitized_mask_bank(
        mask_bank_dir, require_accepted=True, verify_containment=False
    )
    if receipt["mask_bank_artifact_manifest_sha256"] != sha256_file(
        mask_bank_dir / "artifact_manifest.json"
    ):
        raise DataValidationError("Sanitized expert mask-bank binding changed")
    expected = pd.read_csv(
        phase0_dir / "splits" / "waterbirds95_biased_val.csv",
        dtype={"sample_id": str},
    )
    payload = load_sanitized_scores(root / receipt["scores"]["path"])
    summary = validate_sanitized_score_payload(payload, expected)
    if load_checkpoint:
        import torch

        checkpoint = torch.load(
            root / receipt["checkpoint"]["path"],
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint.get("kind") != "setv_background_sanitized":
            raise DataValidationError("Sanitized final checkpoint has wrong kind")
        if not checkpoint.get("model_state_dict"):
            raise DataValidationError("Sanitized final checkpoint state_dict is empty")
    return {
        "status": "complete",
        "seed": receipt["seed"],
        "artifact_count": len(manifest["files"]),
        "scores": summary,
        "checkpoint_loaded": load_checkpoint,
        "scientific_warnings": receipt["scientific_warnings"],
    }
