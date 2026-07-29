"""Phase 4 background set-transformer training and verification."""

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
from setv.experts.set_config import resolved_set_config
from setv.experts.set_data import SetBackgroundDataset
from setv.experts.set_model import create_set_expert_model
from setv.experts.set_scores import (
    build_set_score_payload,
    load_set_scores,
    save_set_scores,
    validate_set_score_payload,
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

    seed = derive_seed(base_seed, f"set_worker={worker_id}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_loaders(phase0_dir: Path, phase0_config: dict, config: dict):
    import torch

    split_dir = phase0_dir / "splits"
    train_dataset = SetBackgroundDataset(
        split_dir / "waterbirds95_candidate_train.csv",
        phase0_config,
        config,
        training=True,
    )
    validation_dataset = SetBackgroundDataset(
        split_dir / "waterbirds95_biased_val.csv",
        phase0_config,
        config,
        training=False,
    )
    fallback_capacity_audit = {
        "candidate_train": train_dataset.audit_canonical_background_capacity(),
        "biased_val": validation_dataset.audit_canonical_background_capacity(),
    }
    training = config["training"]
    seed = int(training["seed"])
    generator = torch.Generator()
    generator.manual_seed(derive_seed(seed, "set_train_shuffle"))
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
    return (
        train_dataset,
        train_loader,
        validation_loader,
        fallback_capacity_audit,
    )


def _prepare(config: dict, model_factory: ModelFactory | None):
    import torch

    phase0_dir = Path(config["phase0_dir"]).expanduser().resolve()
    verify_phase0(phase0_dir, require_approval=True)
    phase0_config = _load_phase0_config(phase0_dir)
    if model_factory is None and int(phase0_config["transforms"]["image_size"]) != 224:
        raise DataValidationError(
            "Production ViT-S/16 set expert requires Phase 0 image_size=224"
        )
    seed = int(config["training"]["seed"])
    seed_receipt = seed_python_numpy(seed)
    seed_receipt.update(seed_torch_if_available(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = _device(config)
    (
        train_dataset,
        train_loader,
        validation_loader,
        fallback_capacity_audit,
    ) = _make_loaders(phase0_dir, phase0_config, config)
    model = (model_factory or create_set_expert_model)(config).to(device)
    initialization_report = getattr(
        model,
        "initialization_report",
        {
            "available": False,
            "reason": "injected model does not expose pretrained initialization",
        },
    )
    if model_factory is None:
        if not initialization_report.get("dense_position_embeddings_discarded"):
            raise DataValidationError("Set expert retained dense positional embeddings")
        if initialization_report.get("second_attention_pooling_module"):
            raise DataValidationError("Set expert unexpectedly added a second pooling module")
    pretrained_cfg = getattr(model, "pretrained_cfg", None)
    if isinstance(pretrained_cfg, dict):
        expected_mean = tuple(float(v) for v in config["model"]["normalization_mean"])
        expected_std = tuple(float(v) for v in config["model"]["normalization_std"])
        actual_mean = tuple(float(v) for v in pretrained_cfg.get("mean", expected_mean))
        actual_std = tuple(float(v) for v in pretrained_cfg.get("std", expected_std))
        if actual_mean != expected_mean or actual_std != expected_std:
            raise DataValidationError(
                "Set-expert normalization differs from pretrained metadata"
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
        else {"available": False, "reason": "model has no pretrained_cfg"}
    )
    return {
        "phase0_dir": phase0_dir,
        "phase0_config": phase0_config,
        "seed_receipt": seed_receipt,
        "device": device,
        "train_dataset": train_dataset,
        "train_loader": train_loader,
        "validation_loader": validation_loader,
        "fallback_capacity_audit": fallback_capacity_audit,
        "model": model,
        "initialization_report": initialization_report,
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
) -> dict[str, float]:
    import torch

    dataset.set_epoch(epoch)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    sample_count = 0
    optimizer_step_count = 0
    amp_skipped_step_count = 0
    retained_counts = []
    valid_counts = []
    crop_attempt_counts = []
    crop_rejected_counts = []
    crop_fallback_count = 0
    started = time.monotonic()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        token_mask = batch["token_mask"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, mixed_precision):
            logits = model(images, token_mask)
            loss = criterion(logits, targets)
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
        loss_sum += float(loss.detach()) * batch_size
        correct += int((logits.detach().argmax(dim=1) == targets).sum())
        sample_count += batch_size
        retained_counts.extend(batch["retained_after_dropout"].numpy().tolist())
        valid_counts.extend(batch["valid_after_cap"].numpy().tolist())
        crop_attempt_counts.extend(
            batch["training_crop_attempt_count"].numpy().tolist()
        )
        crop_rejected_counts.extend(
            batch["training_crop_rejected_count"].numpy().tolist()
        )
        crop_fallback_count += int(
            batch["training_crop_fallback_used"].sum()
        )
    if optimizer_step_count == 0:
        raise DataValidationError("AMP skipped every optimizer update in the epoch")
    return {
        "train_loss": loss_sum / sample_count,
        "train_accuracy": correct / sample_count,
        "train_sample_count": sample_count,
        "train_optimizer_step_count": optimizer_step_count,
        "train_amp_skipped_step_count": amp_skipped_step_count,
        "train_valid_tokens_after_cap_mean": float(np.mean(valid_counts)),
        "train_retained_tokens_mean": float(np.mean(retained_counts)),
        "train_retained_tokens_minimum": int(np.min(retained_counts)),
        "train_retained_tokens_maximum": int(np.max(retained_counts)),
        "train_crop_random_attempts_mean": float(
            np.mean(crop_attempt_counts)
        ),
        "train_crop_random_attempts_maximum": int(
            np.max(crop_attempt_counts)
        ),
        "train_crop_rejected_count": int(np.sum(crop_rejected_counts)),
        "train_crop_fallback_count": crop_fallback_count,
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
    retained_counts = []
    loss_sum = 0.0
    started = time.monotonic()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            token_masks = batch["token_masks"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            logits = model(images, token_masks)
            if logits.shape[1:] != (8, 2):
                raise DataValidationError(
                    f"Set validation logits must have shape [B,8,2], got {logits.shape}"
                )
            mean_logits = logits.mean(dim=1)
            loss_sum += float(criterion(mean_logits, targets))
            sample_ids.extend(str(value) for value in batch["sample_id"])
            labels.append(targets.cpu().numpy())
            view_logits_parts.append(logits.float().cpu().numpy())
            retained_counts.extend(
                batch["retained_token_counts"].reshape(-1).numpy().tolist()
            )
    labels_array = np.concatenate(labels)
    payload = build_set_score_payload(
        sample_ids, labels_array, np.concatenate(view_logits_parts)
    )
    summary = validate_set_score_payload(payload)
    metrics = {
        "biased_val_loss": loss_sum / len(sample_ids),
        "biased_val_accuracy": summary["accuracy"],
        "biased_val_class_0_accuracy": summary["per_class_accuracy"]["0"],
        "biased_val_class_1_accuracy": summary["per_class_accuracy"]["1"],
        "biased_val_sample_count": len(sample_ids),
        "validation_retained_tokens_mean": float(np.mean(retained_counts)),
        "validation_retained_tokens_minimum": int(np.min(retained_counts)),
        "validation_retained_tokens_maximum": int(np.max(retained_counts)),
        "evaluation_seconds": time.monotonic() - started,
    }
    return metrics, payload


def _warnings(summary: dict[str, Any]) -> list[str]:
    warnings = []
    if float(summary["accuracy"]) <= 0.5:
        warnings.append("background_set_accuracy_not_above_binary_chance")
    if float(summary["margin"]["standard_deviation"]) <= 1e-8:
        warnings.append("background_set_margin_has_negligible_variation")
    return warnings


def run_set_expert_smoke(
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
    validation_metrics, payload = _evaluate(
        prepared["model"], prepared["validation_loader"], prepared["device"]
    )
    expected = pd.read_csv(
        prepared["phase0_dir"] / "splits" / "waterbirds95_biased_val.csv",
        dtype={"sample_id": str},
    )
    summary = validate_set_score_payload(payload, expected)
    report = {
        "schema_version": 1,
        "status": "passed",
        "kind": "background_set_real_one_epoch_smoke",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(config["training"]["seed"]),
        "device": str(prepared["device"]),
        "cuda_device": (
            torch.cuda.get_device_name(0) if prepared["device"].type == "cuda" else None
        ),
        "model_parameter_count": int(
            sum(parameter.numel() for parameter in prepared["model"].parameters())
        ),
        "train": train_metrics,
        "biased_val": validation_metrics,
        "score_summary": summary,
        "scientific_warnings": _warnings(summary),
        "initialization": prepared["initialization_report"],
        "canonical_fallback_capacity_audit": prepared[
            "fallback_capacity_audit"
        ],
        "projected_thirty_epoch_seconds": float(
            (
                train_metrics["epoch_seconds"]
                + validation_metrics["evaluation_seconds"]
            )
            * 30
        ),
        "phase0_artifact_manifest_sha256": sha256_file(
            prepared["phase0_dir"] / BASE_ARTIFACT_MANIFEST
        ),
    }
    write_json(report_path, report)
    return report


def train_set_expert(
    config: dict,
    *,
    model_factory: ModelFactory | None = None,
) -> Path:
    import torch

    seed = int(config["training"]["seed"])
    output_root = Path(config["output_root"]).expanduser().resolve()
    destination = output_root / f"seed_{seed}"
    if destination.exists():
        raise ArtifactExistsError(f"Set-expert output exists: {destination}")
    output_root.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("SLURM_JOB_ID", uuid.uuid4().hex[:12])
    staging = output_root / f".seed_{seed}.building.{token}"
    for directory in ("metrics", "scores", "checkpoints", "config", "provenance"):
        (staging / directory).mkdir(parents=True, exist_ok=True)
    logger = EventLogger(staging / "training.jsonl", echo=True)
    logger.log("background_set_training_started", seed=seed)
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
            )
            validation_metrics, final_payload = _evaluate(
                prepared["model"], prepared["validation_loader"], prepared["device"]
            )
            row = {"epoch": epoch + 1, **train_metrics, **validation_metrics}
            rows.append(row)
            _write_csv_atomic(
                staging / "metrics" / "epoch_metrics.csv", pd.DataFrame(rows)
            )
            logger.log("background_set_epoch_complete", **row)
        if final_payload is None:
            raise DataValidationError("Set expert produced no validation payload")
        expected = pd.read_csv(
            prepared["phase0_dir"] / "splits" / "waterbirds95_biased_val.csv",
            dtype={"sample_id": str},
        )
        summary = validate_set_score_payload(final_payload, expected)
        score_path = staging / "scores" / config["storage"]["scores_filename"]
        save_set_scores(score_path, final_payload)
        write_json(staging / "scores" / "background_set_score_summary.json", summary)
        checkpoint_path = (
            staging / "checkpoints" / config["storage"]["checkpoint_filename"]
        )
        checkpoint = {
            "format_version": 1,
            "kind": "setv_background_set",
            "architecture": config["model"]["architecture"],
            "seed": seed,
            "completed_epochs": int(config["training"]["epochs"]),
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in prepared["model"].state_dict().items()
            },
            "initialization_report": prepared["initialization_report"],
            "dense_position_embeddings_discarded": True,
            "pooling": "pretrained_cls_through_four_pretrained_blocks",
            "validation_view_count": 8,
            "pretrained_provenance": prepared["pretrained_provenance"],
            "phase0_artifact_manifest_sha256": sha256_file(
                prepared["phase0_dir"] / BASE_ARTIFACT_MANIFEST
            ),
        }
        _atomic_torch_save(checkpoint, checkpoint_path)
        resolved = resolved_set_config(config)
        with (staging / "config" / "resolved_background_set.yaml").open(
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
            "kind": "setv_background_set",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "trained_on": "full candidate_train",
            "token_dropout": config["training"]["token_dropout"],
            "validation_views": 8,
            "validation_aggregation": "mean_raw_logits_once",
            "calibration_holdout": False,
            "temperature_scaling": False,
            "candidate_training_data_withheld": False,
            "background_pixels": "original_rgb_retained_patches",
            "group_or_place_labels_used": False,
            "patch_selection": {
                "foreground_mask": "vlm_authoritative_then_dilated",
                "maximum_foreground_fraction": config["input"][
                    "maximum_foreground_fraction"
                ],
                "deterministic_maximum_token_cap": config["input"][
                    "max_background_tokens"
                ],
                "minimum_tokens": config["input"]["min_background_tokens"],
                "training_dropout": "deterministic_fixed_count_without_replacement",
                "training_crop_max_attempts": config["input"][
                    "training_crop_max_attempts"
                ],
                "training_crop_fallback": config["input"][
                    "training_crop_fallback"
                ],
                "canonical_fallback_capacity_audit": prepared[
                    "fallback_capacity_audit"
                ],
            },
            "config_sha256": sha256_json(resolved),
            "phase0_dir": str(prepared["phase0_dir"]),
            "phase0_artifact_manifest_sha256": sha256_file(
                prepared["phase0_dir"] / BASE_ARTIFACT_MANIFEST
            ),
            "phase0_visual_approval_sha256": sha256_file(
                prepared["phase0_dir"] / APPROVAL_RECEIPT
            ),
            "initialization": prepared["initialization_report"],
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
        write_json(staging / "phase4_set_receipt.json", receipt)
        logger.log("background_set_training_complete", summary=summary)
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
            "background_set_training_failed",
            exception_type=type(exc).__name__,
            message=str(exc),
        )
        raise


def verify_set_expert(
    output_dir: str | Path, *, load_checkpoint: bool = True
) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "artifact_manifest.json"
    receipt_path = root / "phase4_set_receipt.json"
    if not manifest_path.is_file() or not receipt_path.is_file():
        raise DataValidationError(f"Incomplete set-expert output: {root}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if sha256_json(manifest["files"]) != manifest["manifest_digest"]:
        raise DataValidationError("Set-expert manifest digest is invalid")
    for relative, expected_hash in manifest["files"].items():
        path = root / relative
        if not path.is_file() or path.stat().st_size != expected_hash["size_bytes"]:
            raise DataValidationError(f"Missing/changed set artifact: {relative}")
        if sha256_file(path) != expected_hash["sha256"]:
            raise DataValidationError(f"Changed set artifact hash: {relative}")
    with receipt_path.open("r", encoding="utf-8") as handle:
        receipt = json.load(handle)
    phase0_dir = Path(receipt["phase0_dir"])
    verify_phase0(phase0_dir, require_approval=True)
    if receipt["phase0_artifact_manifest_sha256"] != sha256_file(
        phase0_dir / BASE_ARTIFACT_MANIFEST
    ):
        raise DataValidationError("Set expert Phase 0 binding changed")
    initialization = receipt["initialization"]
    if initialization.get("dense_position_embeddings_discarded") is not True:
        raise DataValidationError("Set expert initialization retained dense positions")
    if initialization.get("second_attention_pooling_module") is not False:
        raise DataValidationError("Set expert initialization added a second pooler")
    expected = pd.read_csv(
        phase0_dir / "splits" / "waterbirds95_biased_val.csv",
        dtype={"sample_id": str},
    )
    payload = load_set_scores(root / receipt["scores"]["path"])
    summary = validate_set_score_payload(payload, expected)
    if load_checkpoint:
        import torch

        checkpoint = torch.load(
            root / receipt["checkpoint"]["path"],
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint.get("kind") != "setv_background_set":
            raise DataValidationError("Set final checkpoint has wrong kind")
        if not checkpoint.get("model_state_dict"):
            raise DataValidationError("Set final checkpoint state_dict is empty")
    return {
        "status": "complete",
        "seed": receipt["seed"],
        "artifact_count": len(manifest["files"]),
        "scores": summary,
        "checkpoint_loaded": load_checkpoint,
        "scientific_warnings": receipt["scientific_warnings"],
    }
