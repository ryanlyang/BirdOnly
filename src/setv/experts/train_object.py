"""Training, smoke testing, and verification for the Phase 1 object expert."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
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
from setv.experts.config import resolved_config
from setv.experts.object_data import ObjectExpertDataset
from setv.experts.object_model import create_object_expert_model
from setv.experts.scores import (
    build_object_score_payload,
    load_object_scores,
    object_sanity_warnings,
    save_object_scores,
    validate_object_score_payload,
)
from setv.phase0 import APPROVAL_RECEIPT, BASE_ARTIFACT_MANIFEST, verify_phase0
from setv.utils.hashing import sha256_file, sha256_json
from setv.utils.io import write_json
from setv.utils.logging import EventLogger
from setv.utils.provenance import runtime_provenance
from setv.utils.seeds import derive_seed, seed_python_numpy, seed_torch_if_available


PHASE1_MANIFEST = "artifact_manifest.json"
ModelFactory = Callable[[dict[str, Any]], Any]


def _source_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_phase0_config(phase0_dir: Path) -> dict[str, Any]:
    path = phase0_dir / "config" / "resolved_phase0.yaml"
    if not path.is_file():
        raise DataValidationError(f"Resolved Phase 0 config is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise DataValidationError("Resolved Phase 0 config is invalid")
    return config


def _worker_init(worker_id: int, *, base_seed: int) -> None:
    import torch

    seed = derive_seed(base_seed, f"dataloader_worker={worker_id}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_loaders(
    phase0_dir: Path,
    phase0_config: dict[str, Any],
    config: dict[str, Any],
):
    import torch

    split_dir = phase0_dir / "splits"
    train_dataset = ObjectExpertDataset(
        split_dir / "waterbirds95_candidate_train.csv",
        phase0_config,
        config,
        training=True,
    )
    validation_dataset = ObjectExpertDataset(
        split_dir / "waterbirds95_biased_val.csv",
        phase0_config,
        config,
        training=False,
    )
    training = config["training"]
    seed = int(training["seed"])
    generator = torch.Generator()
    generator.manual_seed(derive_seed(seed, "object_train_shuffle"))
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
    return train_dataset, validation_dataset, train_loader, validation_loader


def _device(config: dict[str, Any]):
    import torch

    requested = config["training"]["device"]
    if requested == "cuda" and not torch.cuda.is_available():
        raise DataValidationError(
            "Object-expert production config requires CUDA, but CUDA is unavailable"
        )
    return torch.device(requested)


def _autocast(device, enabled: bool):
    import torch

    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=enabled,
    )


def _grad_scaler(device, enabled: bool):
    import torch

    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _step_optimizer_and_scheduler(*, scaler, optimizer, scheduler) -> bool:
    """Advance the scheduler only when GradScaler applies the optimizer update.

    GradScaler silently skips ``optimizer.step()`` when it detects non-finite
    gradients. A scale decrease after ``update()`` is the public indication
    that this happened. Advancing the scheduler on such a batch puts the
    learning-rate schedule ahead of the model updates and triggers PyTorch's
    scheduler-order warning when the first update is skipped.
    """

    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    optimizer_step_applied = float(scaler.get_scale()) >= scale_before
    if optimizer_step_applied:
        scheduler.step()
    return optimizer_step_applied


def _build_optimizer_scheduler(model, config: dict[str, Any], steps_per_epoch: int):
    import torch

    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    total_steps = int(training["epochs"]) * steps_per_epoch
    warmup_steps = int(training["warmup_epochs"]) * steps_per_epoch

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        denominator = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / denominator))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    return optimizer, scheduler


def _train_epoch(
    model,
    loader,
    dataset: ObjectExpertDataset,
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
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
    loss_sum = 0.0
    correct = 0
    sample_count = 0
    optimizer_step_count = 0
    amp_skipped_step_count = 0
    started = time.monotonic()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, mixed_precision):
            logits = model(images)
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
    if optimizer_step_count == 0:
        raise DataValidationError("AMP skipped every optimizer update in the epoch")
    return {
        "train_loss": loss_sum / sample_count,
        "train_accuracy": correct / sample_count,
        "train_sample_count": sample_count,
        "train_optimizer_step_count": optimizer_step_count,
        "train_amp_skipped_step_count": amp_skipped_step_count,
        "epoch_seconds": time.monotonic() - started,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }


def _evaluate_logits(model, loader, device) -> dict[str, Any]:
    import torch

    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    sample_ids: list[str] = []
    labels: list[np.ndarray] = []
    logits_parts: list[np.ndarray] = []
    loss_sum = 0.0
    started = time.monotonic()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            logits = model(images)
            loss_sum += float(criterion(logits, targets))
            sample_ids.extend(str(value) for value in batch["sample_id"])
            labels.append(targets.cpu().numpy())
            logits_parts.append(logits.float().cpu().numpy())
    label_array = np.concatenate(labels)
    logit_array = np.concatenate(logits_parts)
    return {
        "sample_id": sample_ids,
        "labels": label_array,
        "logits": logit_array,
        "loss_sum": loss_sum,
        "evaluation_seconds": time.monotonic() - started,
    }


def _evaluate(model, loader, device) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    raw = _evaluate_logits(model, loader, device)
    payload = build_object_score_payload(
        raw["sample_id"], raw["labels"], raw["logits"]
    )
    summary = validate_object_score_payload(payload)
    metrics = {
        "biased_val_loss": raw["loss_sum"] / len(raw["sample_id"]),
        "biased_val_accuracy": summary["accuracy"],
        "biased_val_class_0_accuracy": summary["per_class_accuracy"].get("0", float("nan")),
        "biased_val_class_1_accuracy": summary["per_class_accuracy"].get("1", float("nan")),
        "biased_val_sample_count": len(raw["sample_id"]),
        "evaluation_seconds": raw["evaluation_seconds"],
    }
    return metrics, payload


def _prepare(
    config: dict[str, Any],
    model_factory: ModelFactory | None,
):
    import torch

    phase0_dir = Path(config["phase0_dir"]).expanduser().resolve()
    phase0_verification = verify_phase0(phase0_dir, require_approval=True)
    phase0_config = _load_phase0_config(phase0_dir)
    seed = int(config["training"]["seed"])
    seed_receipt = seed_python_numpy(seed)
    seed_receipt.update(seed_torch_if_available(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = _device(config)
    train_dataset, validation_dataset, train_loader, validation_loader = _make_loaders(
        phase0_dir, phase0_config, config
    )
    factory = model_factory or create_object_expert_model
    model = factory(config).to(device)
    pretrained_cfg = getattr(model, "pretrained_cfg", None)
    pretrained_provenance: dict[str, Any]
    if isinstance(pretrained_cfg, dict):
        expected_mean = tuple(float(value) for value in config["model"]["normalization_mean"])
        expected_std = tuple(float(value) for value in config["model"]["normalization_std"])
        actual_mean = tuple(float(value) for value in pretrained_cfg.get("mean", expected_mean))
        actual_std = tuple(float(value) for value in pretrained_cfg.get("std", expected_std))
        if actual_mean != expected_mean or actual_std != expected_std:
            raise DataValidationError(
                "Configured normalization does not match timm pretrained metadata: "
                f"configured mean/std={expected_mean}/{expected_std}, "
                f"pretrained mean/std={actual_mean}/{actual_std}"
            )
        pretrained_provenance = {
            key: pretrained_cfg.get(key)
            for key in (
                "architecture",
                "tag",
                "hf_hub_id",
                "url",
                "input_size",
                "num_classes",
                "crop_pct",
                "interpolation",
                "mean",
                "std",
            )
            if key in pretrained_cfg
        }
    else:
        pretrained_provenance = {
            "available": False,
            "reason": "model factory did not expose pretrained_cfg",
        }
    return {
        "phase0_dir": phase0_dir,
        "phase0_config": phase0_config,
        "phase0_verification": phase0_verification,
        "seed_receipt": seed_receipt,
        "device": device,
        "train_dataset": train_dataset,
        "validation_dataset": validation_dataset,
        "train_loader": train_loader,
        "validation_loader": validation_loader,
        "model": model,
        "pretrained_provenance": pretrained_provenance,
    }


def run_object_expert_smoke(
    config: dict[str, Any],
    report_path: str | Path,
    *,
    model_factory: ModelFactory | None = None,
) -> dict[str, Any]:
    """Run one complete real-data train/evaluate epoch without saving weights."""
    import torch

    prepared = _prepare(config, model_factory)
    model = prepared["model"]
    train_loader = prepared["train_loader"]
    optimizer, scheduler = _build_optimizer_scheduler(model, config, len(train_loader))
    mixed = bool(config["training"]["mixed_precision"]) and prepared["device"].type == "cuda"
    scaler = _grad_scaler(prepared["device"], mixed)
    train_metrics = _train_epoch(
        model,
        train_loader,
        prepared["train_dataset"],
        epoch=0,
        device=prepared["device"],
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        mixed_precision=mixed,
    )
    validation_metrics, payload = _evaluate(
        model, prepared["validation_loader"], prepared["device"]
    )
    expected = pd.read_csv(
        prepared["phase0_dir"] / "splits" / "waterbirds95_biased_val.csv",
        dtype={"sample_id": str},
    )
    score_summary = validate_object_score_payload(payload, expected)
    report = {
        "schema_version": 1,
        "status": "passed",
        "kind": "object_expert_real_one_epoch_smoke",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(config["training"]["seed"]),
        "device": str(prepared["device"]),
        "cuda_device": (
            torch.cuda.get_device_name(0) if prepared["device"].type == "cuda" else None
        ),
        "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "train": train_metrics,
        "biased_val": validation_metrics,
        "score_summary": score_summary,
        "scientific_warnings": object_sanity_warnings(score_summary),
        "pretrained_provenance": prepared["pretrained_provenance"],
        "projected_twenty_epoch_seconds": float(train_metrics["epoch_seconds"] * 20),
        "phase0_artifact_manifest_sha256": sha256_file(
            prepared["phase0_dir"] / BASE_ARTIFACT_MANIFEST
        ),
        "phase0_visual_approval_sha256": sha256_file(
            prepared["phase0_dir"] / APPROVAL_RECEIPT
        ),
    }
    write_json(report_path, report)
    return report


def _atomic_torch_save(value: Any, destination: Path) -> None:
    import torch

    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        torch.save(value, temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        frame.to_csv(temporary_path, index=False, lineterminator="\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _artifact_manifest(root: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == PHASE1_MANIFEST:
            continue
        files[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return {
        "schema_version": 1,
        "files": files,
        "manifest_digest": sha256_json(files),
    }


def train_object_expert(
    config: dict[str, Any],
    *,
    model_factory: ModelFactory | None = None,
) -> Path:
    """Train 20 epochs and atomically publish final Phase 1 artifacts."""
    import torch

    seed = int(config["training"]["seed"])
    output_root = Path(config["output_root"]).expanduser().resolve()
    destination = output_root / f"seed_{seed}"
    if destination.exists():
        raise ArtifactExistsError(
            f"Object-expert output already exists and will not be overwritten: "
            f"{destination}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    job_token = os.environ.get("SLURM_JOB_ID", uuid.uuid4().hex[:12])
    staging = output_root / f".seed_{seed}.building.{job_token}"
    if staging.exists():
        raise ArtifactExistsError(f"Object-expert staging path exists: {staging}")
    staging.mkdir()
    (staging / "metrics").mkdir()
    (staging / "scores").mkdir()
    (staging / "checkpoints").mkdir()
    (staging / "config").mkdir()
    (staging / "provenance").mkdir()
    logger = EventLogger(staging / "training.jsonl", echo=True)
    logger.log("object_training_started", seed=seed, destination=str(destination))
    try:
        prepared = _prepare(config, model_factory)
        model = prepared["model"]
        train_loader = prepared["train_loader"]
        optimizer, scheduler = _build_optimizer_scheduler(
            model, config, len(train_loader)
        )
        mixed = (
            bool(config["training"]["mixed_precision"])
            and prepared["device"].type == "cuda"
        )
        scaler = _grad_scaler(prepared["device"], mixed)
        epoch_rows: list[dict[str, Any]] = []
        final_payload: dict[str, np.ndarray] | None = None
        for epoch in range(int(config["training"]["epochs"])):
            train_metrics = _train_epoch(
                model,
                train_loader,
                prepared["train_dataset"],
                epoch=epoch,
                device=prepared["device"],
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                mixed_precision=mixed,
            )
            validation_metrics, final_payload = _evaluate(
                model, prepared["validation_loader"], prepared["device"]
            )
            row = {"epoch": epoch + 1, **train_metrics, **validation_metrics}
            epoch_rows.append(row)
            _write_csv_atomic(
                staging / "metrics" / "epoch_metrics.csv", pd.DataFrame(epoch_rows)
            )
            logger.log("object_epoch_complete", **row)

        if final_payload is None:
            raise DataValidationError("Object expert produced no final validation scores")
        expected = pd.read_csv(
            prepared["phase0_dir"] / "splits" / "waterbirds95_biased_val.csv",
            dtype={"sample_id": str},
        )
        score_summary = validate_object_score_payload(final_payload, expected)
        sanity_warnings = object_sanity_warnings(score_summary)
        score_path = staging / "scores" / config["storage"]["scores_filename"]
        save_object_scores(score_path, final_payload)
        write_json(staging / "scores" / "object_score_summary.json", score_summary)

        state_dict = {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        }
        checkpoint = {
            "format_version": 1,
            "kind": "setv_object_expert",
            "architecture": config["model"]["architecture"],
            "num_classes": int(config["model"]["num_classes"]),
            "seed": seed,
            "completed_epochs": int(config["training"]["epochs"]),
            "model_state_dict": state_dict,
            "normalization_mean": config["model"]["normalization_mean"],
            "normalization_std": config["model"]["normalization_std"],
            "green_rgb": config["input"]["green_rgb"],
            "phase0_artifact_manifest_sha256": sha256_file(
                prepared["phase0_dir"] / BASE_ARTIFACT_MANIFEST
            ),
            "pretrained_provenance": prepared["pretrained_provenance"],
        }
        checkpoint_path = (
            staging / "checkpoints" / config["storage"]["checkpoint_filename"]
        )
        _atomic_torch_save(checkpoint, checkpoint_path)

        resolved = resolved_config(config)
        with (staging / "config" / "resolved_object_expert.yaml").open(
            "w", encoding="utf-8"
        ) as handle:
            yaml.safe_dump(resolved, handle, sort_keys=True)
        provenance = runtime_provenance(_source_root(), destination)
        provenance["seed_receipt"] = prepared["seed_receipt"]
        provenance["torch"] = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(prepared["device"]),
            "device_name": (
                torch.cuda.get_device_name(0)
                if prepared["device"].type == "cuda"
                else None
            ),
            "cudnn": torch.backends.cudnn.version(),
        }
        try:
            import timm

            provenance["timm_version"] = timm.__version__
        except ImportError:
            provenance["timm_version"] = None
        provenance["source_documents"] = {
            name: sha256_file(_source_root() / name)
            for name in (
                "SETV_Waterbirds95_Implementation_Plan_v2.md",
                "TIGRIS_RESEARCH_COMPUTE_HANDOFF.md",
            )
            if (_source_root() / name).is_file()
        }
        write_json(staging / "provenance" / "runtime.json", provenance)

        receipt = {
            "schema_version": 1,
            "status": "complete",
            "kind": "setv_object_expert",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "trained_on": "full candidate_train",
            "calibration_holdout": False,
            "temperature_scaling": False,
            "pretrained_provenance": prepared["pretrained_provenance"],
            "completed_epochs": int(config["training"]["epochs"]),
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
                "summary": score_summary,
            },
            "scientific_warnings": sanity_warnings,
            "scientific_gate": (
                "review_required" if sanity_warnings else "no_automatic_warning"
            ),
        }
        write_json(staging / "phase1_receipt.json", receipt)
        logger.log("object_training_complete", score_summary=score_summary)
        write_json(staging / PHASE1_MANIFEST, _artifact_manifest(staging))
        os.rename(staging, destination)
        return destination
    except Exception as exc:
        write_json(
            staging / "failure.json",
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        logger.log(
            "object_training_failed",
            exception_type=type(exc).__name__,
            message=str(exc),
        )
        raise


def verify_object_expert(
    output_dir: str | Path,
    *,
    load_checkpoint: bool = True,
) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / PHASE1_MANIFEST
    receipt_path = root / "phase1_receipt.json"
    if not manifest_path.is_file() or not receipt_path.is_file():
        raise DataValidationError(f"Incomplete object-expert output: {root}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if sha256_json(manifest["files"]) != manifest["manifest_digest"]:
        raise DataValidationError("Object-expert artifact manifest digest is invalid")
    for relative, expected in manifest["files"].items():
        path = root / relative
        if not path.is_file():
            raise DataValidationError(f"Missing object-expert artifact: {relative}")
        if path.stat().st_size != expected["size_bytes"]:
            raise DataValidationError(f"Changed object-expert artifact size: {relative}")
        if sha256_file(path) != expected["sha256"]:
            raise DataValidationError(f"Changed object-expert artifact hash: {relative}")
    with receipt_path.open("r", encoding="utf-8") as handle:
        receipt = json.load(handle)
    phase0_dir = Path(receipt["phase0_dir"])
    verify_phase0(phase0_dir, require_approval=True)
    if receipt["phase0_artifact_manifest_sha256"] != sha256_file(
        phase0_dir / BASE_ARTIFACT_MANIFEST
    ):
        raise DataValidationError("Object expert is bound to a different Phase 0 manifest")
    if receipt["phase0_visual_approval_sha256"] != sha256_file(
        phase0_dir / APPROVAL_RECEIPT
    ):
        raise DataValidationError("Object expert is bound to a different mask approval")

    scores = load_object_scores(root / receipt["scores"]["path"])
    expected = pd.read_csv(
        phase0_dir / "splits" / "waterbirds95_biased_val.csv",
        dtype={"sample_id": str},
    )
    summary = validate_object_score_payload(scores, expected)
    if load_checkpoint:
        import torch

        checkpoint = torch.load(
            root / receipt["checkpoint"]["path"],
            map_location="cpu",
            weights_only=False,
        )
        required = {
            "format_version",
            "kind",
            "architecture",
            "seed",
            "completed_epochs",
            "model_state_dict",
        }
        missing = required - set(checkpoint)
        if missing:
            raise DataValidationError(
                f"Final object checkpoint is missing keys: {sorted(missing)}"
            )
        if checkpoint["kind"] != "setv_object_expert":
            raise DataValidationError("Final checkpoint has the wrong kind")
        if not checkpoint["model_state_dict"]:
            raise DataValidationError("Final checkpoint state_dict is empty")
    return {
        "status": "complete",
        "seed": receipt["seed"],
        "artifact_count": len(manifest["files"]),
        "scores": summary,
        "checkpoint_loaded": load_checkpoint,
    }
