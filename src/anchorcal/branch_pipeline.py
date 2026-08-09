"""Fixed-epoch training, calibration, evaluation, and artifacts for both branches."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .audits import require_branch_above_chance
from .background import load_view_bank
from .calibration import calibration_diagnostics, fit_temperature
from .datasets import (
    BranchTrainingDataset,
    EpochSampler,
    EvaluationDataset,
    collate_background_training,
    collate_evaluation,
)
from .errors import AuditFailure, PreflightError
from .io import atomic_write_json, hash_object, sha256_file
from .models.branches import BackgroundBranch, ForegroundBranch
from .pretrained import create_pretrained_vit
from .preprocessing import load_preprocessing_manifest
from .paths import geometry_artifact_root
from .precision import evaluation_inference
from .seeds import seed_everything, seeded_worker_init
from .training import UpdateScheduler, make_adamw
from .transforms import check_fallback_rate
from .vlm_masks import VlmMaskBank, load_vlm_mask_bank


def _load_frame(output: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(output / "splits" / f"waterbirds100_{name}.csv")


def _load_pretrained_path(output: Path) -> str:
    path = output / "preflight" / "pretrained_manifest.json"
    if not path.is_file():
        raise PreflightError("pretrained preflight manifest is missing")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return str(manifest["weights_path"])


def _atomic_torch_save(path: Path, value: Any) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Some distributed filesystems do not expose directory fsync. The
            # checkpoint file itself has still been flushed before publication.
            pass
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _provenance_file(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise PreflightError(f"branch provenance is missing {description}: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _branch_restart_provenance(
    config: dict[str, Any],
    branch: str,
    *,
    token_budget: int | None,
    optimizer_groups_path: Path,
) -> dict[str, Any]:
    """Resolve the complete immutable identity accepted by a branch restart."""

    if branch not in {"foreground", "background"}:
        raise ValueError("branch must be foreground or background")
    output = Path(config["paths"]["output_root"])
    preflight_root = output / "preflight"
    split_root = output / "splits"
    pretrained_path = preflight_root / "pretrained_manifest.json"
    try:
        pretrained = json.loads(pretrained_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError("branch pretrained manifest is invalid") from error
    weights_path = Path(str(pretrained.get("weights_path", "")))
    weights = _provenance_file(weights_path, "pretrained weights")
    if pretrained.get("weights_sha256") != weights["sha256"]:
        raise PreflightError("branch pretrained weights differ from preflight")

    data_files = {
        "mask_manifest": _provenance_file(
            preflight_root / "mask_manifest.json", "mask manifest"
        ),
        "split_manifest": _provenance_file(
            split_root / "manifest.json", "split manifest"
        ),
        "expert_train_split": _provenance_file(
            split_root / "waterbirds100_expert_train.csv", "expert-train split"
        ),
        "preprocessing_manifest": _provenance_file(
            preflight_root / "preprocessing_manifest.json",
            "preprocessing manifest",
        ),
        "geometry_manifest": _provenance_file(
            geometry_artifact_root(config) / "manifest.json", "geometry manifest"
        ),
    }
    model_binding: dict[str, Any] = {
        "implementation": (
            "anchorcal.models.branches.ForegroundBranch"
            if branch == "foreground"
            else "anchorcal.models.branches.BackgroundBranch"
        ),
        "token_budget": token_budget,
        "foreground_position_mode": (
            config["branches"]["foreground_position_mode"]
            if branch == "foreground"
            else None
        ),
        "copied_blocks": config["branches"].get("copied_blocks"),
        "optimizer_groups": _provenance_file(
            optimizer_groups_path, "optimizer groups"
        ),
    }
    if branch == "background":
        model_binding["background_token_budget"] = _provenance_file(
            geometry_artifact_root(config) / "background_token_budget.json",
            "background token-budget decision",
        )
    return {
        "schema_version": "anchorcal-branch-restart-provenance-v1",
        "branch": branch,
        "resolved_config_sha256": config["resolved_config_sha256"],
        "resolved_paths_sha256": hash_object(config["paths"]),
        "training_seed": int(config["seeds"][f"{branch}_branch_train"]),
        "preflight_report": _provenance_file(
            preflight_root / "report.json", "preflight report"
        ),
        "pretrained_manifest": _provenance_file(
            pretrained_path, "pretrained manifest"
        ),
        "pretrained_weights": weights,
        "pretrained_identity": {
            "repository": pretrained.get("repository"),
            "revision": pretrained.get("revision"),
            "weights_sha256": pretrained.get("weights_sha256"),
        },
        "data_identity": {
            "metadata_path": config["paths"]["metadata_path"],
            "waterbirds_root": config["paths"]["waterbirds_root"],
            "vlm_mask_root": config["paths"]["vlm_mask_root"],
            "vlm_mask_contract": dict(config["masks"]),
            **data_files,
        },
        "model_identity": model_binding,
    }


def _validate_branch_restart(
    state: dict[str, Any],
    expected_provenance: dict[str, Any],
    *,
    branch: str,
    epochs: int,
    history_path: Path,
) -> tuple[list[dict[str, Any]], int]:
    """Fail closed before loading any mutable state from a restart file."""

    if (
        not isinstance(state, dict)
        or state.get("schema_version") != "anchorcal-branch-restart-v2"
        or state.get("branch") != branch
        or state.get("provenance") != expected_provenance
    ):
        raise PreflightError("branch restart provenance mismatch")
    try:
        epoch = int(state["epoch"])
        history = state["history"]
    except (KeyError, TypeError, ValueError) as error:
        raise PreflightError("branch restart progress metadata is invalid") from error
    if (
        not 1 <= epoch <= epochs
        or not isinstance(history, list)
        or len(history) != epoch
        or [int(item.get("epoch", -1)) for item in history]
        != list(range(1, epoch + 1))
    ):
        raise PreflightError("branch restart epoch/history sequence is invalid")
    try:
        persisted_history = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError("branch restart history artifact is invalid") from error
    if persisted_history != history:
        raise PreflightError("branch restart history differs from restart.pt")
    return history, epoch + 1


def _rng_state() -> dict[str, Any]:
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _train_epoch(model, loader, optimizer, scheduler, branch: str, device) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_valid = 0
    sampled = 0
    fallback = 0
    invalid = 0
    fallback_events: list[dict[str, Any]] = []
    for batch in loader:
        if "fallback_count" in batch:
            fallback += int(batch["fallback_count"])
            fallback_events.extend(batch.get("fallback_events", []))
        else:
            fallback += int(batch["fallback"].sum().item())
            fallback_mask = batch["fallback"].bool()
            for img_id, epoch, attempts in zip(
                batch["img_id"][fallback_mask].tolist(),
                batch["epoch"][fallback_mask].tolist(),
                batch["attempt_count"][fallback_mask].tolist(),
                strict=True,
            ):
                fallback_events.append(
                    {
                        "img_id": int(img_id),
                        "epoch": int(epoch),
                        "attempt_count": int(attempts),
                        "fallback": True,
                    }
                )
        sampled += int(
            batch.get(
                "sample_count",
                len(batch["y"]) if "y" in batch else 0,
            )
        )
        invalid += int(batch.get("invalid_count", 0))
        # Invalid examples are excluded by lock; a chance all-invalid batch is
        # not itself a redesign trigger when the epoch still has valid data.
        if branch == "background" and batch.get("empty", False):
            continue
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        labels = batch["y"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            if branch == "foreground":
                logits = model(images, masks).logits
            else:
                indices = batch["source_indices"].to(device, non_blocking=True)
                logits = model(images, indices).logits
            loss = functional.cross_entropy(logits, labels)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise AuditFailure("non-finite branch gradient norm")
        optimizer.step()
        scheduler.step_after_optimizer()
        total_loss += float(loss.detach().item()) * len(labels)
        total_correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
        total_valid += len(labels)
    if not total_valid:
        raise AuditFailure("branch epoch has no valid examples")
    return {
        "loss": total_loss / total_valid,
        "accuracy": total_correct / total_valid,
        "valid_examples": total_valid,
        "sampled_examples": sampled,
        "invalid_examples": invalid,
        "fallback_count": fallback,
        "fallback_rate": fallback / sampled,
        "fallback_events": fallback_events,
    }


def _persist_crop_fallback_events(
    branch_root: Path, branch: str, history: list[dict[str, Any]]
) -> tuple[Path, list[dict[str, Any]]]:
    events = [
        {"branch": branch, **event}
        for epoch_record in history
        for event in epoch_record.get("fallback_events", [])
    ]
    path = branch_root / "crop_fallback_events.json"
    atomic_write_json(
        path,
        {
            "schema_version": "anchorcal-branch-crop-fallback-events-v1",
            "branch": branch,
            "event_count": len(events),
            "events": events,
        },
    )
    return path, events


def _aggregate_crop_fallback_gate(
    history: list[dict[str, Any]],
    *,
    branch: str | None = None,
    report_path: Path | None = None,
    maximum_rate: float = 0.001,
) -> dict[str, int | float | str]:
    fallback_count = sum(int(record["fallback_count"]) for record in history)
    sampled_examples = sum(int(record["sampled_examples"]) for record in history)
    if sampled_examples <= 0:
        raise ValueError("aggregate sampled training example count must be positive")
    fallback_rate = fallback_count / sampled_examples
    result: dict[str, int | float | str] = {
        "fallback_count": fallback_count,
        "sampled_examples": sampled_examples,
        "fallback_rate": fallback_rate,
        "gate_scope": "all_sampled_training_examples_across_all_epochs",
    }
    if report_path is not None:
        atomic_write_json(
            report_path,
            {
                "schema_version": "anchorcal-branch-crop-fallback-gate-v1",
                "branch": branch,
                "maximum_rate": maximum_rate,
                "status": "passed" if fallback_rate <= maximum_rate else "failed",
                **result,
            },
        )
    check_fallback_rate(
        fallback_count, sampled_examples, maximum_rate=maximum_rate
    )
    return result


def evaluate_branch(
    model,
    frame: pd.DataFrame,
    config: dict[str, Any],
    branch: str,
    device,
    *,
    mask_bank: VlmMaskBank | None = None,
) -> dict[str, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader

    output = Path(config["paths"]["output_root"])
    preprocessing = load_preprocessing_manifest(output)
    if mask_bank is None:
        mask_bank = load_vlm_mask_bank(config)
    dataset = EvaluationDataset(
        frame,
        config["paths"]["waterbirds_root"],
        mask_bank,
        preprocessing=preprocessing,
    )
    workers = int(config["optimization"]["num_workers"])
    loader_options: dict[str, Any] = {
        "batch_size": int(config["branches"]["batch_size"]),
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": bool(config["optimization"]["pin_memory"]),
        "collate_fn": collate_evaluation,
    }
    if workers:
        loader_options.update(
            persistent_workers=bool(config["optimization"]["persistent_workers"]),
            prefetch_factor=int(config["optimization"]["prefetch_factor"]),
        )
    loader = DataLoader(dataset, **loader_options)
    view_lookup: dict[int, np.ndarray] = {}
    if branch == "background":
        bank = load_view_bank(
            geometry_artifact_root(config) / "fixed_background_views.h5"
        )
        view_lookup = {
            int(img_id): bank["source_patch_indices"][index]
            for index, img_id in enumerate(bank["img_id"])
        }
    model.eval()
    ids: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    places: list[np.ndarray] = []
    logits_list: list[np.ndarray] = []
    valid_list: list[np.ndarray] = []
    source_metadata_list: list[np.ndarray] = []
    with evaluation_inference(device):
        for batch in loader:
            ids_array = batch["img_id"].numpy()
            if branch == "foreground":
                images = batch["foreground_image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)
                branch_output = model(images, masks)
                logits = branch_output.logits.float().cpu().numpy()
                valid = np.ones(len(ids_array), dtype=bool)
                source_metadata = np.zeros((len(ids_array), 196), dtype=np.uint8)
                source_indices = branch_output.source_indices.cpu().numpy()
                source_valid = branch_output.patch_valid.cpu().numpy()
                for row_index in range(len(ids_array)):
                    source_metadata[
                        row_index, source_indices[row_index, source_valid[row_index]]
                    ] = 1
            else:
                images = batch["image"].to(device, non_blocking=True)
                valid = np.asarray(
                    [
                        value in view_lookup and np.all(view_lookup[value] >= 0)
                        for value in ids_array.tolist()
                    ],
                    dtype=bool,
                )
                logits = np.full((len(ids_array), 2), np.nan, dtype=np.float32)
                source_metadata = np.full(
                    (
                        len(ids_array),
                        int(config["branches"]["background_eval_views"]),
                        int(model.token_budget),
                    ),
                    -1,
                    dtype=np.int16,
                )
                if valid.any():
                    selected_images = images[torch.from_numpy(valid).to(device)]
                    selected_ids = ids_array[valid]
                    view_indices = np.stack([view_lookup[int(value)] for value in selected_ids])
                    view_tensor = torch.from_numpy(view_indices.astype(np.int64)).to(device)
                    source_metadata[valid] = view_indices.astype(np.int16)
                    projected = model.project(selected_images)
                    view_logits = [
                        model.forward_from_projected(projected, view_tensor[:, view]).logits
                        for view in range(view_tensor.shape[1])
                    ]
                    logits[valid] = (
                        torch.stack(view_logits).mean(dim=0).float().cpu().numpy()
                    )
            ids.append(ids_array)
            labels.append(batch["y"].numpy())
            places.append(batch["place"].numpy())
            logits_list.append(logits)
            valid_list.append(valid)
            source_metadata_list.append(source_metadata)
    return {
        "img_id": np.concatenate(ids),
        "sample_id": np.concatenate(ids),
        "labels": np.concatenate(labels),
        "label": np.concatenate(labels),
        "places": np.concatenate(places),
        "logits": np.concatenate(logits_list),
        "valid": np.concatenate(valid_list),
        "source_patch_metadata": np.concatenate(source_metadata_list),
    }


def _attach_calibrated_outputs(
    outputs: dict[str, np.ndarray], temperature: float
) -> dict[str, np.ndarray]:
    raw = np.asarray(outputs["logits"], dtype=np.float32)
    valid = np.asarray(outputs["valid"], dtype=bool)
    calibrated = np.full_like(raw, np.nan)
    calibrated[valid] = raw[valid] / float(temperature)
    prediction = np.full(len(raw), -1, dtype=np.int16)
    prediction[valid] = raw[valid].argmax(axis=1).astype(np.int16)
    correct = np.zeros(len(raw), dtype=bool)
    correct[valid] = prediction[valid] == np.asarray(outputs["labels"])[valid]
    return {
        **outputs,
        "raw_logits": raw,
        "calibrated_logits": calibrated,
        "prediction": prediction,
        "correct": correct,
    }


def train_branch(config: dict[str, Any], branch: str) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    if branch not in {"foreground", "background"}:
        raise ValueError("branch must be foreground or background")
    output = Path(config["paths"]["output_root"])
    preprocessing = load_preprocessing_manifest(output)
    preflight = output / "preflight" / "report.json"
    if not preflight.is_file():
        raise PreflightError("successful preflight report is required before training")
    with preflight.open("r", encoding="utf-8") as handle:
        preflight_report = json.load(handle)
    if preflight_report.get("status") != "passed":
        raise PreflightError("preflight report is not successful")
    if (
        not config["runtime"]["debug"]
        and preflight_report.get("resolved_config_sha256")
        != config["resolved_config_sha256"]
    ):
        raise PreflightError("production branch config differs from preflight")
    if preflight_report.get("resolved_paths") != config["paths"]:
        raise PreflightError("resolved branch paths differ from preflight")
    seed = int(config["seeds"][f"{branch}_branch_train"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = create_pretrained_vit(_load_pretrained_path(output))
    token_budget = None
    if branch == "background":
        with (geometry_artifact_root(config) / "background_token_budget.json").open(
            "r", encoding="utf-8"
        ) as handle:
            token_budget = int(json.load(handle)["token_budget"])
        model = BackgroundBranch(source, token_budget=token_budget)
    else:
        model = ForegroundBranch(
            source, position_mode=config["branches"]["foreground_position_mode"]
        )
    model.to(device)
    train_frame = _load_frame(output, "expert_train")
    mask_bank = load_vlm_mask_bank(config)
    dataset = BranchTrainingDataset(
        train_frame,
        config["paths"]["waterbirds_root"],
        mask_bank,
        branch=branch,
        run_seed=seed,
        token_budget=token_budget,
        background_seed=int(config["seeds"]["background_sampling"]),
        preprocessing=preprocessing,
    )
    sampler = EpochSampler(len(dataset), seed)
    workers = int(config["optimization"]["num_workers"])
    loader_options: dict[str, Any] = {
        "batch_size": int(config["branches"]["batch_size"]),
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": bool(config["optimization"]["pin_memory"]),
        "drop_last": False,
        "worker_init_fn": seeded_worker_init(seed),
    }
    if workers:
        loader_options.update(
            persistent_workers=bool(config["optimization"]["persistent_workers"]),
            prefetch_factor=int(config["optimization"]["prefetch_factor"]),
        )
    if branch == "background":
        loader_options["collate_fn"] = collate_background_training
    loader = DataLoader(dataset, **loader_options)
    optimizer, optimizer_manifest = make_adamw(
        model,
        learning_rate=float(config["branches"]["learning_rate"]),
        weight_decay=float(config["branches"]["weight_decay"]),
    )
    epochs = int(config["branches"]["epochs"])
    updates_per_epoch = math.ceil(len(dataset) / int(config["branches"]["batch_size"]))
    scheduler = UpdateScheduler(
        optimizer,
        base_lr=float(config["branches"]["learning_rate"]),
        total_updates=epochs * updates_per_epoch,
        warmup_updates=int(config["branches"]["warmup_epochs"]) * updates_per_epoch,
    )
    namespace = "debug/branches" if config["runtime"]["debug"] else "branches"
    branch_root = output / namespace / branch
    branch_root.mkdir(parents=True, exist_ok=True)
    optimizer_groups_path = branch_root / "optimizer_groups.json"
    atomic_write_json(optimizer_groups_path, optimizer_manifest)
    restart_provenance = _branch_restart_provenance(
        config,
        branch,
        token_budget=token_budget,
        optimizer_groups_path=optimizer_groups_path,
    )
    restart = branch_root / "restart.pt"
    history_path = branch_root / "history.json"
    start_epoch = 1
    history: list[dict[str, Any]] = []
    if restart.is_file():
        state = torch.load(restart, map_location="cpu", weights_only=False)
        history, start_epoch = _validate_branch_restart(
            state,
            restart_provenance,
            branch=branch,
            epochs=epochs,
            history_path=history_path,
        )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        _restore_rng(state["rng"])
    for epoch in range(start_epoch, epochs + 1):
        sampler.set_epoch(epoch)
        result = _train_epoch(model, loader, optimizer, scheduler, branch, device)
        result["epoch"] = epoch
        history.append(result)
        atomic_write_json(history_path, history)
        # Persist individual events before applying the hard aggregate gate so
        # a gate failure remains fully auditable.
        _persist_crop_fallback_events(branch_root, branch, history)
        _atomic_torch_save(
            restart,
            {
                "schema_version": "anchorcal-branch-restart-v2",
                "branch": branch,
                "provenance": restart_provenance,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "history": history,
                "rng": _rng_state(),
            },
        )
    fallback_gate_path = branch_root / "crop_fallback_gate.json"
    aggregate_fallback = _aggregate_crop_fallback_gate(
        history,
        branch=branch,
        report_path=fallback_gate_path,
        maximum_rate=float(config["data"]["crop_fallback_rate_gate"]),
    )
    if epochs != int(config["branches"]["frozen_epoch"]):
        raise AssertionError("the configured fixed final branch epoch was not reached")
    checkpoint = branch_root / "epoch_final.pt"
    _atomic_torch_save(
        checkpoint,
        {
            "schema_version": "anchorcal-branch-checkpoint-v2",
            "model": model.state_dict(),
            "epoch": epochs,
            "branch": branch,
            "token_budget": token_budget,
            "foreground_position_mode": (
                config["branches"]["foreground_position_mode"]
                if branch == "foreground"
                else None
            ),
            "resolved_config_sha256": config["resolved_config_sha256"],
        },
    )
    calibration = evaluate_branch(
        model,
        _load_frame(output, "expert_calibration"),
        config,
        branch,
        device,
        mask_bank=mask_bank,
    )
    calibration_valid = calibration["valid"]
    temperature = fit_temperature(
        calibration["logits"][calibration_valid], calibration["labels"][calibration_valid]
    )
    temperature_diagnostics = calibration_diagnostics(
        calibration["logits"][calibration_valid],
        calibration["labels"][calibration_valid],
        temperature.temperature,
    )
    calibration = _attach_calibrated_outputs(calibration, temperature.temperature)
    biased = evaluate_branch(
        model,
        _load_frame(output, "biased_val"),
        config,
        branch,
        device,
        mask_bank=mask_bank,
    )
    biased = _attach_calibrated_outputs(biased, temperature.temperature)
    valid = biased["valid"]
    correct = biased["logits"][valid].argmax(axis=1) == biased["labels"][valid]
    competence_interval = require_branch_above_chance(
        correct,
        biased["labels"][valid],
        seed=int(config["seeds"]["branch_audit_bootstrap"]),
        replicates=2000,
    )
    calibration_path = branch_root / "expert_calibration_outputs.npz"
    biased_path = branch_root / "biased_val_outputs.npz"
    np.savez_compressed(calibration_path, **calibration)
    np.savez_compressed(biased_path, **biased)
    fallback_event_path, fallback_events = _persist_crop_fallback_events(
        branch_root, branch, history
    )
    manifest = {
        "schema_version": "anchorcal-branch-manifest-v4",
        "branch": branch,
        "fixed_epoch": epochs,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "temperature": temperature.__dict__,
        "calibration_diagnostics": temperature_diagnostics,
        "biased_val_competence": competence_interval.__dict__,
        "token_budget": token_budget,
        "foreground_position_mode": (
            config["branches"]["foreground_position_mode"]
            if branch == "foreground"
            else None
        ),
        "outputs": {
            "expert_calibration": {
                "path": str(calibration_path.resolve()),
                "sha256": sha256_file(calibration_path),
                "size_bytes": calibration_path.stat().st_size,
            },
            "biased_val": {
                "path": str(biased_path.resolve()),
                "sha256": sha256_file(biased_path),
                "size_bytes": biased_path.stat().st_size,
            },
        },
        "invalid_biased_val_fraction": float((~valid).mean()),
        "resolved_config_sha256": config["resolved_config_sha256"],
        "preflight_report": str(preflight.resolve()),
        "preflight_report_sha256": sha256_file(preflight),
        "metadata_path": config["paths"]["metadata_path"],
        "metadata_sha256": preflight_report["metadata_sha256"],
        "mask_bank_sha256": preflight_report["mask_bank_sha256"],
        "mask_manifest_sha256": preflight_report["mask_manifest_sha256"],
        "mask_source": preflight_report["mask_source"],
        "mask_contract": dict(config["masks"]),
        "preprocessing_manifest_sha256": preflight_report["preprocessing"][
            "manifest_sha256"
        ],
        "git_commit": preflight_report["git"]["commit"],
        "paths": dict(config["paths"]),
        "seeds": dict(config["seeds"]),
        "optimizer": optimizer_manifest,
        "training_artifacts": {
            "history": {
                "path": str(history_path.resolve()),
                "sha256": sha256_file(history_path),
                "size_bytes": history_path.stat().st_size,
            },
            "optimizer_groups": {
                "path": str(optimizer_groups_path.resolve()),
                "sha256": sha256_file(optimizer_groups_path),
                "size_bytes": optimizer_groups_path.stat().st_size,
            },
        },
        "scheduler": {
            "type": "per_optimizer_update_linear_warmup_then_cosine",
            "warmup_updates": int(config["branches"]["warmup_epochs"])
            * updates_per_epoch,
            "total_updates": epochs * updates_per_epoch,
        },
        "rng_policy": "stateless_sha256_geometry_and_sampling_plus_strongly_seeded_torch",
        "crop_fallback_events": str(fallback_event_path.resolve()),
        "crop_fallback_events_sha256": sha256_file(fallback_event_path),
        "crop_fallback_events_size_bytes": fallback_event_path.stat().st_size,
        "crop_fallback_event_count": len(fallback_events),
        "crop_fallback_aggregate": aggregate_fallback,
        "crop_fallback_gate": str(fallback_gate_path.resolve()),
        "crop_fallback_gate_sha256": sha256_file(fallback_gate_path),
        "crop_fallback_gate_size_bytes": fallback_gate_path.stat().st_size,
        "runtime_job_receipt": os.environ.get("ANCHORCAL_JOB_RECEIPT"),
        "runtime_job_receipt_sha256": os.environ.get("ANCHORCAL_JOB_RECEIPT_SHA256"),
        "nondeterminism_warning_record": (
            "job-receipt stderr_log" if os.environ.get("ANCHORCAL_JOB_RECEIPT") else "process stderr"
        ),
    }
    if branch == "background" and manifest["invalid_biased_val_fraction"] > 0.01:
        raise AuditFailure("more than 1% of biased_val is invalid for background branch")
    atomic_write_json(branch_root / "manifest.json", manifest)
    return manifest
