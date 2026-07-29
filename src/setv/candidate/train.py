"""Phase 5 ERM trajectory, online selection, and reporting-only evaluation."""

from __future__ import annotations

import json
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

from setv.candidate.config import resolved_candidate_config
from setv.candidate.data import CandidateDataset
from setv.candidate.metrics import (
    grouped_metrics,
    ordinary_metrics,
    prediction_payload,
)
from setv.candidate.model import create_candidate_model
from setv.candidate.selectors import (
    RealisticSelectionTracker,
    compute_realistic_selectors,
    load_selector_inputs,
    oracle_is_better,
)
from setv.errors import ArtifactExistsError, DataValidationError
from setv.experts.train_object import (
    _artifact_manifest,
    _atomic_torch_save,
    _autocast,
    _build_optimizer_scheduler,
    _grad_scaler,
    _load_phase0_config,
    _source_root,
)
from setv.phase0 import APPROVAL_RECEIPT, BASE_ARTIFACT_MANIFEST, verify_phase0
from setv.utils.hashing import sha256_file, sha256_json
from setv.utils.io import write_json
from setv.utils.logging import EventLogger
from setv.utils.provenance import runtime_provenance
from setv.utils.seeds import derive_seed, seed_python_numpy, seed_torch_if_available


ModelFactory = Callable[[dict[str, Any]], Any]


def _candidate_device(config: dict[str, Any]):
    import torch

    requested = config["training"]["device"]
    if requested == "cuda" and not torch.cuda.is_available():
        raise DataValidationError(
            "Candidate production config requires CUDA, but CUDA is unavailable"
        )
    return torch.device(requested)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _worker_init(worker_id: int, *, base_seed: int) -> None:
    import torch

    seed = derive_seed(base_seed, f"candidate_worker={worker_id}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_loaders(phase0_dir: Path, phase0_config: dict, config: dict):
    import torch

    split_dir = phase0_dir / "splits"
    datasets = {
        name: CandidateDataset(
            split_dir / f"waterbirds95_{name}.csv",
            phase0_config,
            config,
            training=name == "candidate_train",
        )
        for name in ("candidate_train", "biased_val", "oracle_val", "test")
    }
    training = config["training"]
    seed = int(training["seed"])
    common = {
        "num_workers": int(training["num_workers"]),
        "pin_memory": training["device"] == "cuda",
        "persistent_workers": False,
        "worker_init_fn": partial(_worker_init, base_seed=seed),
    }
    generator = torch.Generator()
    generator.manual_seed(derive_seed(seed, "candidate_train_shuffle"))
    loaders = {
        "candidate_train": torch.utils.data.DataLoader(
            datasets["candidate_train"],
            batch_size=int(training["batch_size"]),
            shuffle=True,
            generator=generator,
            drop_last=False,
            **common,
        )
    }
    for name in ("biased_val", "oracle_val", "test"):
        loaders[name] = torch.utils.data.DataLoader(
            datasets[name],
            batch_size=int(training["evaluation_batch_size"]),
            shuffle=False,
            drop_last=False,
            **common,
        )
    return datasets, loaders


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
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
    total_loss = 0.0
    correct = 0
    count = 0
    started = time.monotonic()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, mixed_precision):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        size = int(targets.numel())
        total_loss += float(loss.detach()) * size
        correct += int((logits.detach().argmax(1) == targets).sum())
        count += size
    return {
        "train_loss": total_loss / count,
        "train_accuracy": correct / count,
        "train_sample_count": count,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "epoch_seconds": time.monotonic() - started,
    }


def _evaluate(model, loader, device) -> dict[str, np.ndarray]:
    model.eval()
    sample_ids = []
    labels = []
    logits = []
    import torch

    with torch.inference_mode():
        for batch in loader:
            output = model(batch["image"].to(device, non_blocking=True))
            sample_ids.extend(str(value) for value in batch["sample_id"])
            labels.append(batch["target"].numpy())
            logits.append(output.float().cpu().numpy())
    return prediction_payload(sample_ids, np.concatenate(labels), np.concatenate(logits))


def _protected_groups(
    phase0_dir: Path, split_name: str, sample_ids: np.ndarray
) -> np.ndarray:
    protected = pd.read_csv(
        phase0_dir / "private_analysis" / "protected_group_labels.csv",
        dtype={"sample_id": str},
    )
    subset = protected[protected["split_name"] == split_name].set_index("sample_id")
    if set(sample_ids.tolist()) != set(subset.index.astype(str)):
        raise DataValidationError(f"Protected {split_name} IDs do not align")
    return subset.loc[sample_ids.astype(str), "group"].to_numpy(dtype=np.int64)


def _prepare(config: dict, model_factory: ModelFactory | None):
    import torch

    phase0_dir = Path(config["phase0_dir"]).expanduser().resolve()
    verify_phase0(phase0_dir, require_approval=True)
    phase0_config = _load_phase0_config(phase0_dir)
    if model_factory is None and int(phase0_config["transforms"]["image_size"]) != 224:
        raise DataValidationError("Production candidate ViT requires image_size=224")
    seed = int(config["training"]["seed"])
    seed_receipt = seed_python_numpy(seed)
    seed_receipt.update(seed_torch_if_available(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = _candidate_device(config)
    datasets, loaders = _make_loaders(phase0_dir, phase0_config, config)
    model = (model_factory or create_candidate_model)(config).to(device)
    pretrained_cfg = getattr(model, "pretrained_cfg", None)
    if isinstance(pretrained_cfg, dict):
        expected_mean = tuple(config["model"]["normalization_mean"])
        expected_std = tuple(config["model"]["normalization_std"])
        if tuple(pretrained_cfg.get("mean", expected_mean)) != expected_mean or tuple(
            pretrained_cfg.get("std", expected_std)
        ) != expected_std:
            raise DataValidationError("Candidate normalization differs from pretrained model")
    biased_manifest = pd.read_csv(
        phase0_dir / "splits" / "waterbirds95_biased_val.csv",
        dtype={"sample_id": str},
    )
    ids = biased_manifest["sample_id"].astype(str).to_numpy()
    labels = biased_manifest["y"].to_numpy(dtype=np.int64)
    selector_inputs = load_selector_inputs(config["fusion_dirs"], ids, labels)
    for source in selector_inputs.values():
        receipt = json.loads((source.fusion_dir / "fusion_receipt.json").read_text())
        if Path(receipt["phase0_dir"]).resolve() != phase0_dir:
            raise DataValidationError("A fusion source references a different Phase 0")
    return {
        "phase0_dir": phase0_dir,
        "phase0_config": phase0_config,
        "seed_receipt": seed_receipt,
        "device": device,
        "datasets": datasets,
        "loaders": loaders,
        "model": model,
        "selector_inputs": selector_inputs,
        "biased_ids": ids,
        "biased_labels": labels,
        "pretrained_cfg": pretrained_cfg,
    }


def _checkpoint(
    model,
    path: Path,
    *,
    relative_path: str,
    epoch: int,
    config: dict,
    phase0_sha256: str,
) -> dict[str, Any]:
    if not path.exists():
        _atomic_torch_save(
            {
                "format_version": 1,
                "kind": "setv_candidate_erm_selected",
                "architecture": config["model"]["architecture"],
                "seed": int(config["training"]["seed"]),
                "epoch": int(epoch),
                "phase0_artifact_manifest_sha256": phase0_sha256,
                "normalization_mean": config["model"]["normalization_mean"],
                "normalization_std": config["model"]["normalization_std"],
                "model_state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
            },
            path,
        )
    return {
        "path": relative_path,
        "sha256": sha256_file(path),
        "epoch": int(epoch),
    }


def _bind_and_collect_checkpoints(
    model,
    realistic_checkpoint_root: Path,
    oracle_checkpoint_root: Path,
    tracker: RealisticSelectionTracker,
    realistic_changed: list[str],
    oracle_best: dict[str, Any],
    oracle_changed: bool,
    *,
    epoch: int,
    config: dict,
    phase0_sha256: str,
) -> None:
    if not realistic_changed and not oracle_changed:
        return
    if realistic_changed:
        path = realistic_checkpoint_root / f"epoch_{epoch:03d}.pt"
        reference = _checkpoint(
            model,
            path,
            relative_path=f"selection/checkpoints/{path.name}",
            epoch=epoch,
            config=config,
            phase0_sha256=phase0_sha256,
        )
        for name in realistic_changed:
            tracker.best[name]["checkpoint"] = reference
        referenced = {
            int(item["checkpoint"]["epoch"])
            for item in tracker.best.values()
            if "checkpoint" in item
        }
        for candidate in realistic_checkpoint_root.glob("epoch_*.pt"):
            candidate_epoch = int(candidate.stem.split("_")[1])
            if candidate_epoch not in referenced:
                candidate.unlink()
    if oracle_changed:
        path = oracle_checkpoint_root / f"epoch_{epoch:03d}.pt"
        reference = _checkpoint(
            model,
            path,
            relative_path=f"analysis_only/oracle_checkpoints/{path.name}",
            epoch=epoch,
            config=config,
            phase0_sha256=phase0_sha256,
        )
        oracle_best["checkpoint"] = reference
        for candidate in oracle_checkpoint_root.glob("epoch_*.pt"):
            if candidate != path:
                candidate.unlink()


def _selector_snapshot(
    tracker: RealisticSelectionTracker, *, completed_epochs: int
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "rolling" if completed_epochs < 50 else "frozen",
        "completed_epochs": int(completed_epochs),
        "selectors": tracker.best,
        "unavailable": tracker.unavailable,
        "ula": {
            "status": "deferred_to_phase6",
            "reason": "uLA implementation and validation procedure are Phase 6",
        },
        "oracle_or_test_metrics_present": False,
    }


def run_candidate_smoke(
    config: dict,
    report_path: str | Path,
    *,
    model_factory: ModelFactory | None = None,
) -> dict[str, Any]:
    prepared = _prepare(config, model_factory)
    optimizer, scheduler = _build_optimizer_scheduler(
        prepared["model"], config, len(prepared["loaders"]["candidate_train"])
    )
    mixed = bool(config["training"]["mixed_precision"]) and prepared["device"].type == "cuda"
    scaler = _grad_scaler(prepared["device"], mixed)
    train = _train_epoch(
        prepared["model"],
        prepared["loaders"]["candidate_train"],
        prepared["datasets"]["candidate_train"],
        epoch=0,
        device=prepared["device"],
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        mixed_precision=mixed,
    )
    biased = _evaluate(
        prepared["model"], prepared["loaders"]["biased_val"], prepared["device"]
    )
    realistic = compute_realistic_selectors(biased, prepared["selector_inputs"])
    oracle = _evaluate(
        prepared["model"], prepared["loaders"]["oracle_val"], prepared["device"]
    )
    test = _evaluate(
        prepared["model"], prepared["loaders"]["test"], prepared["device"]
    )
    report = {
        "schema_version": 1,
        "status": "passed",
        "kind": "candidate_real_one_epoch_smoke",
        "seed": int(config["training"]["seed"]),
        "train": train,
        "biased_val": ordinary_metrics(biased),
        "realistic_selector_count": len(realistic),
        "oracle_evaluated_in_isolated_path": len(oracle["sample_id"]),
        "test_evaluated_but_metrics_hidden": len(test["sample_id"]),
        "test_metrics": None,
        "phase0_artifact_manifest_sha256": sha256_file(
            prepared["phase0_dir"] / BASE_ARTIFACT_MANIFEST
        ),
    }
    write_json(report_path, report)
    return report


def train_candidate(
    config: dict, *, model_factory: ModelFactory | None = None
) -> Path:
    import torch

    seed = int(config["training"]["seed"])
    output_root = Path(config["output_root"]).expanduser().resolve()
    destination = output_root / f"seed_{seed}"
    if destination.exists():
        raise ArtifactExistsError(f"Candidate output exists: {destination}")
    output_root.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("SLURM_JOB_ID", uuid.uuid4().hex[:12])
    staging = output_root / f".seed_{seed}.building.{token}"
    for directory in (
        "biased_val",
        "biased_val/epoch_predictions",
        "selection/checkpoints",
        "analysis_only",
        "analysis_only/oracle_checkpoints",
        "reporting_only",
        "config",
        "provenance",
    ):
        (staging / directory).mkdir(parents=True, exist_ok=True)
    logger = EventLogger(staging / "training.jsonl", echo=True)
    logger.log("candidate_training_started", seed=seed)
    try:
        prepared = _prepare(config, model_factory)
        optimizer, scheduler = _build_optimizer_scheduler(
            prepared["model"], config, len(prepared["loaders"]["candidate_train"])
        )
        mixed = (
            bool(config["training"]["mixed_precision"])
            and prepared["device"].type == "cuda"
        )
        scaler = _grad_scaler(prepared["device"], mixed)
        tolerance = float(config["selectors"]["score_tolerance"])
        tracker = RealisticSelectionTracker(tolerance)
        oracle_best = None
        phase0_sha = sha256_file(prepared["phase0_dir"] / BASE_ARTIFACT_MANIFEST)
        biased_predictions = []
        biased_metric_rows = []
        oracle_rows = []
        test_rows = []

        oracle_ids = (
            prepared["datasets"]["oracle_val"]
            .rows["sample_id"]
            .astype(str)
            .to_numpy()
        )
        test_ids = prepared["datasets"]["test"].rows["sample_id"].astype(str).to_numpy()
        oracle_groups = _protected_groups(
            prepared["phase0_dir"], "oracle_val", oracle_ids
        )
        test_groups = _protected_groups(prepared["phase0_dir"], "test", test_ids)

        for epoch_index in range(int(config["training"]["epochs"])):
            epoch = epoch_index + 1
            train_metrics = _train_epoch(
                prepared["model"],
                prepared["loaders"]["candidate_train"],
                prepared["datasets"]["candidate_train"],
                epoch=epoch_index,
                device=prepared["device"],
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                mixed_precision=mixed,
            )
            biased = _evaluate(
                prepared["model"],
                prepared["loaders"]["biased_val"],
                prepared["device"],
            )
            if not np.array_equal(biased["sample_id"], prepared["biased_ids"]):
                raise DataValidationError("Candidate biased-val ordering changed")
            realistic = compute_realistic_selectors(
                biased, prepared["selector_inputs"]
            )
            changed = tracker.update(epoch, realistic)
            ordinary = ordinary_metrics(biased)
            biased_metric_rows.append(
                {
                    "epoch": epoch,
                    "ordinary": ordinary,
                    "selectors": realistic,
                }
            )
            biased_predictions.append(biased)
            _atomic_npz(
                staging
                / "biased_val"
                / "epoch_predictions"
                / f"epoch_{epoch:03d}.npz",
                {
                    "sample_id": biased["sample_id"].astype(np.str_),
                    "true_label": biased["true_label"].astype(np.int64),
                    "candidate_logits": biased["logits"].astype(np.float32),
                    "candidate_predicted_class": biased[
                        "predicted_class"
                    ].astype(np.int64),
                    "candidate_cross_entropy": biased[
                        "cross_entropy"
                    ].astype(np.float32),
                    "candidate_correct": biased["correct"].astype(np.uint8),
                },
            )

            # Protected labels never enter compute_realistic_selectors.
            oracle_payload = _evaluate(
                prepared["model"],
                prepared["loaders"]["oracle_val"],
                prepared["device"],
            )
            oracle_metrics = grouped_metrics(oracle_payload, oracle_groups)
            oracle_candidate = {"epoch": epoch, "metrics": oracle_metrics}
            oracle_changed = oracle_is_better(
                oracle_candidate, oracle_best, tolerance=tolerance
            )
            if oracle_changed:
                oracle_best = oracle_candidate
            oracle_rows.append({"epoch": epoch, **oracle_metrics})

            # Reporting-only values remain in memory, outside all logs and
            # selector calls, until the selection receipt is frozen.
            test_payload = _evaluate(
                prepared["model"], prepared["loaders"]["test"], prepared["device"]
            )
            test_rows.append(
                {"epoch": epoch, **grouped_metrics(test_payload, test_groups)}
            )
            _bind_and_collect_checkpoints(
                prepared["model"],
                staging / "selection" / "checkpoints",
                staging / "analysis_only" / "oracle_checkpoints",
                tracker,
                changed,
                oracle_best,
                oracle_changed,
                epoch=epoch,
                config=config,
                phase0_sha256=phase0_sha,
            )
            write_json(
                staging / "selection" / "rolling_selectors.json",
                _selector_snapshot(tracker, completed_epochs=epoch),
            )
            write_json(
                staging / "analysis_only" / "oracle_selector.json",
                {
                    "namespace": "analysis_only",
                    "best": oracle_best,
                    "test_metrics_present": False,
                },
            )
            write_json(
                staging / "biased_val" / "selector_metrics.json",
                {"epochs": biased_metric_rows},
            )
            write_json(
                staging / "analysis_only" / "oracle_metrics.json",
                {"namespace": "analysis_only", "epochs": oracle_rows},
            )
            logger.log(
                "candidate_epoch_complete",
                epoch=epoch,
                **train_metrics,
                biased_val_accuracy=ordinary["accuracy"],
                biased_val_loss=ordinary["loss"],
                realistic_selectors_updated=changed,
                oracle_metrics_hidden_from_training_log=True,
                test_metrics_hidden_from_training_log=True,
            )

        arrays = {
            "epoch": np.arange(1, 51, dtype=np.int16),
            "sample_id": prepared["biased_ids"].astype(np.str_),
            "true_label": prepared["biased_labels"].astype(np.int64),
            "candidate_logits": np.stack(
                [item["logits"] for item in biased_predictions]
            ).astype(np.float32),
            "candidate_predicted_class": np.stack(
                [item["predicted_class"] for item in biased_predictions]
            ).astype(np.int64),
            "candidate_cross_entropy": np.stack(
                [item["cross_entropy"] for item in biased_predictions]
            ).astype(np.float32),
            "candidate_correct": np.stack(
                [item["correct"] for item in biased_predictions]
            ).astype(np.uint8),
        }
        prediction_path = staging / "biased_val" / "epoch_predictions.npz"
        _atomic_npz(prediction_path, arrays)
        final_snapshot = _selector_snapshot(tracker, completed_epochs=50)
        write_json(staging / "selection" / "rolling_selectors.json", final_snapshot)
        selection_receipt = {
            "schema_version": 1,
            "status": "frozen",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_seed": seed,
            "completed_epochs": 50,
            "score_tolerance": tolerance,
            "realistic_selectors": tracker.best,
            "unavailable_selectors": tracker.unavailable,
            "ula": final_snapshot["ula"],
            "oracle_selector_path": "analysis_only/oracle_selector.json",
            "oracle_excluded_from_realistic_selection": True,
            "test_metrics_seen": False,
            "test_metrics_used": False,
        }
        selection_path = staging / "selection" / "selection_receipt.json"
        write_json(selection_path, selection_receipt)
        selection_hash = sha256_file(selection_path)
        write_json(
            staging / "selection" / "selection_frozen.json",
            {
                "status": "frozen_before_test_publication",
                "selection_receipt_sha256": selection_hash,
            },
        )
        # This is the first point at which test values are written to disk.
        write_json(
            staging / "reporting_only" / "test_metrics.json",
            {
                "namespace": "reporting_only",
                "selection_receipt_sha256": selection_hash,
                "selection_frozen_before_publication": True,
                "epochs": test_rows,
            },
        )
        resolved = resolved_candidate_config(config)
        with (staging / "config" / "resolved_candidate.yaml").open(
            "w", encoding="utf-8"
        ) as handle:
            yaml.safe_dump(resolved, handle, sort_keys=True)
        provenance = runtime_provenance(_source_root(), destination)
        provenance["seed_receipt"] = prepared["seed_receipt"]
        provenance["torch"] = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "device": str(prepared["device"]),
        }
        write_json(staging / "provenance" / "runtime.json", provenance)
        source_hashes = {
            name: sha256_file(Path(path) / "fusion_receipt.json")
            for name, path in config["fusion_dirs"].items()
        }
        receipt = {
            "schema_version": 1,
            "status": "complete",
            "kind": "setv_candidate_erm_trajectory",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "completed_epochs": 50,
            "trained_on": "full candidate_train",
            "candidate_inputs": "original_rgb_images_only",
            "calibration_holdout": False,
            "temperature_scaling": False,
            "candidate_training_data_withheld": False,
            "phase0_dir": str(prepared["phase0_dir"]),
            "phase0_artifact_manifest_sha256": phase0_sha,
            "phase0_visual_approval_sha256": sha256_file(
                prepared["phase0_dir"] / APPROVAL_RECEIPT
            ),
            "fusion_receipt_sha256": source_hashes,
            "pretrained_provenance": (
                {
                    key: prepared["pretrained_cfg"].get(key)
                    for key in (
                        "architecture",
                        "tag",
                        "hf_hub_id",
                        "url",
                        "input_size",
                        "mean",
                        "std",
                    )
                    if key in prepared["pretrained_cfg"]
                }
                if isinstance(prepared["pretrained_cfg"], dict)
                else {
                    "available": False,
                    "reason": "injected model has no pretrained_cfg",
                }
            ),
            "config_sha256": sha256_json(resolved),
            "biased_val_predictions": {
                "path": prediction_path.relative_to(staging).as_posix(),
                "sha256": sha256_file(prediction_path),
            },
            "selection_receipt": {
                "path": selection_path.relative_to(staging).as_posix(),
                "sha256": selection_hash,
            },
            "rolling_checkpoint_count": len(
                list((staging / "selection" / "checkpoints").glob("*.pt"))
            ),
            "oracle_checkpoint_count": len(
                list(
                    (staging / "analysis_only" / "oracle_checkpoints").glob(
                        "*.pt"
                    )
                )
            ),
            "all_epoch_checkpoints_saved": False,
            "test_policy": {
                "namespace": "reporting_only",
                "not_logged": True,
                "not_used_by_selectors": True,
                "published_after_selection_hash": True,
            },
        }
        write_json(staging / "phase5_receipt.json", receipt)
        logger.log(
            "candidate_training_complete",
            completed_epochs=50,
            selection_receipt_sha256=selection_hash,
            test_metrics_not_printed=True,
        )
        write_json(staging / "artifact_manifest.json", _artifact_manifest(staging))
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
            "candidate_training_failed",
            exception_type=type(exc).__name__,
            message=str(exc),
        )
        raise


def verify_candidate(
    output_dir: str | Path, *, load_checkpoints: bool = True
) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    if sha256_json(manifest["files"]) != manifest["manifest_digest"]:
        raise DataValidationError("Candidate artifact manifest digest is invalid")
    for relative, expected in manifest["files"].items():
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != expected["size_bytes"]
            or sha256_file(path) != expected["sha256"]
        ):
            raise DataValidationError(f"Missing/changed candidate artifact: {relative}")
    receipt = json.loads((root / "phase5_receipt.json").read_text())
    if receipt.get("kind") != "setv_candidate_erm_trajectory":
        raise DataValidationError("Candidate receipt has wrong kind")
    phase0_dir = Path(receipt["phase0_dir"])
    verify_phase0(phase0_dir, require_approval=True)
    if receipt["phase0_artifact_manifest_sha256"] != sha256_file(
        phase0_dir / BASE_ARTIFACT_MANIFEST
    ):
        raise DataValidationError("Candidate Phase 0 binding changed")
    selection_path = root / receipt["selection_receipt"]["path"]
    selection_hash = sha256_file(selection_path)
    if selection_hash != receipt["selection_receipt"]["sha256"]:
        raise DataValidationError("Candidate selection receipt hash changed")
    frozen = json.loads((root / "selection" / "selection_frozen.json").read_text())
    reporting = json.loads((root / "reporting_only" / "test_metrics.json").read_text())
    if frozen["selection_receipt_sha256"] != selection_hash or reporting[
        "selection_receipt_sha256"
    ] != selection_hash:
        raise DataValidationError("Test report predates or mismatches frozen selection")
    with np.load(
        root / receipt["biased_val_predictions"]["path"], allow_pickle=False
    ) as archive:
        if archive["candidate_logits"].shape[0] != 50 or not np.array_equal(
            archive["epoch"], np.arange(1, 51)
        ):
            raise DataValidationError("Candidate epoch prediction trajectory is incomplete")
    selection = json.loads(selection_path.read_text())
    if selection["test_metrics_seen"] or selection["test_metrics_used"]:
        raise DataValidationError("Selection receipt indicates test leakage")
    checkpoints = list((root / "selection" / "checkpoints").glob("*.pt"))
    if len(checkpoints) != receipt["rolling_checkpoint_count"]:
        raise DataValidationError("Rolling checkpoint count changed")
    oracle_checkpoints = list(
        (root / "analysis_only" / "oracle_checkpoints").glob("*.pt")
    )
    if len(oracle_checkpoints) != receipt["oracle_checkpoint_count"]:
        raise DataValidationError("Oracle checkpoint count changed")
    if load_checkpoints:
        import torch

        for path in checkpoints + oracle_checkpoints:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint.get("kind") != "setv_candidate_erm_selected":
                raise DataValidationError(f"Wrong candidate checkpoint kind: {path}")
    referenced = []
    referenced.extend(selection["realistic_selectors"].values())
    oracle = json.loads((root / "analysis_only" / "oracle_selector.json").read_text())
    referenced.append(oracle["best"])
    for item in referenced:
        reference = item.get("checkpoint")
        if not reference:
            raise DataValidationError("A selected candidate has no rolling checkpoint")
        path = root / reference["path"]
        if not path.is_file() or sha256_file(path) != reference["sha256"]:
            raise DataValidationError("A selected candidate checkpoint binding changed")
    epoch_files = sorted(
        (root / "biased_val" / "epoch_predictions").glob("epoch_*.npz")
    )
    if len(epoch_files) != 50:
        raise DataValidationError("Per-epoch biased-validation files are incomplete")
    metrics = json.loads((root / "biased_val" / "selector_metrics.json").read_text())
    oracle_metrics = json.loads(
        (root / "analysis_only" / "oracle_metrics.json").read_text()
    )
    if len(metrics["epochs"]) != 50 or len(oracle_metrics["epochs"]) != 50:
        raise DataValidationError("Candidate scalar metric curves are incomplete")
    if len(reporting["epochs"]) != 50:
        raise DataValidationError("Reporting-only test curve is incomplete")
    resolved_path = root / "config" / "resolved_candidate.yaml"
    with resolved_path.open("r", encoding="utf-8") as handle:
        resolved = yaml.safe_load(handle)
    for name, fusion_dir in resolved["fusion_dirs"].items():
        current = sha256_file(Path(fusion_dir) / "fusion_receipt.json")
        if current != receipt["fusion_receipt_sha256"][name]:
            raise DataValidationError(f"Candidate {name} fusion binding changed")
    return {
        "status": "complete",
        "seed": receipt["seed"],
        "completed_epochs": receipt["completed_epochs"],
        "realistic_selector_count": len(selection["realistic_selectors"]),
        "rolling_checkpoint_count": len(checkpoints),
        "test_namespace": reporting["namespace"],
        "selection_receipt_sha256": selection_hash,
        "artifact_count": len(manifest["files"]),
    }
