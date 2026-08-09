"""Six-run ordinary ViT training with live criteria, HDF5, and rolling states."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .anchor_artifacts import verify_anchor_artifacts
from .candidate_evaluation import evaluate_plain, evaluate_practical_criteria
from .candidate_schema import CANDIDATE_SCALAR_METRICS, candidate_per_example_shapes
from .checkpoint_verification import verify_candidate_checkpoint_artifacts
from .checkpoints import CheckpointManager
from .datasets import CandidateTrainingDataset, EpochSampler
from .decision import verify_decision_receipt
from .errors import AuditFailure, PreflightError
from .io import atomic_write_json, sha256_file
from .metrics import per_example_cross_entropy
from .models.candidate import CandidateViT
from .pretrained import create_pretrained_vit
from .preprocessing import load_preprocessing_manifest
from .paths import geometry_artifact_root
from .seeds import seed_everything, seeded_worker_init
from .storage import (
    CandidateStorage,
    ExclusiveFileLock,
    PredictionBatch,
    SampleMetadata,
    verify_candidate_storage,
)
from .training import UpdateScheduler, make_adamw
from .vlm_masks import VlmMaskBank, load_vlm_mask_bank


ELIGIBLE_SCORE_KEYS = {
    "ordinary": "ordinary_accuracy",
    "saliency": "saliency_harmonic",
    "swap": "token_swap_harmonic",
    "blur": "background_blur_harmonic",
}


def candidate_run_id(learning_rate: float, weight_decay: float, seed: int) -> str:
    lr = f"{learning_rate:.0e}".replace("+", "").replace("-0", "-")
    wd = f"{weight_decay:.2f}"
    return f"lr{lr}_wd{wd}_seed{seed}"


def validate_grid_member(config: dict[str, Any], learning_rate: float, weight_decay: float) -> None:
    if learning_rate not in [float(value) for value in config["candidate_grid"]["learning_rates"]]:
        raise ValueError("learning rate is not in the locked candidate grid")
    if weight_decay not in [float(value) for value in config["candidate_grid"]["weight_decays"]]:
        raise ValueError("weight decay is not in the locked candidate grid")


def _frame(output: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(output / "splits" / f"waterbirds100_{name}.csv").sort_values(
        "img_id", kind="stable"
    ).reset_index(drop=True)


def _model_weights(output: Path) -> str:
    with (output / "preflight" / "pretrained_manifest.json").open("r", encoding="utf-8") as handle:
        return str(json.load(handle)["weights_path"])


def _require_receipt(config: dict[str, Any]) -> Path:
    output = Path(config["paths"]["output_root"])
    debug = bool(config["runtime"]["debug"])
    receipt_root = output / ("debug/receipt" if debug else "receipt")
    receipts = sorted(receipt_root.glob("anchorcal_decision_*.json"))
    if len(receipts) != 1 or not verify_decision_receipt(receipts[0]):
        raise PreflightError("exactly one valid AnchorCal decision receipt is required")
    verify_anchor_artifacts(config, decision_receipt=receipts[0])
    return receipts[0]


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


def _prediction_batch(values: dict[str, Any]) -> PredictionBatch:
    return PredictionBatch(
        logits=values["logits"],
        prediction=values["prediction"],
        correct=values["correct"],
        loss=values["loss"],
    )


def _train_epoch(model, loader, optimizer, scheduler, device) -> dict[str, float]:
    import torch
    import torch.nn.functional as functional

    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["y"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(images)
            loss = functional.cross_entropy(logits, labels)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise AuditFailure("candidate gradient norm is non-finite")
        optimizer.step()
        scheduler.step_after_optimizer()
        count = len(labels)
        total += count
        total_loss += float(loss.detach().item()) * count
        total_correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
    if not total:
        raise AuditFailure("candidate training loader is empty")
    return {"loss": total_loss / total, "accuracy": total_correct / total}


def _practical_should_update(
    current: dict[str, Any] | None,
    *,
    score: float,
    biased_accuracy: float,
    biased_loss: float,
    epoch: int,
    tolerance: float,
) -> bool:
    if current is None:
        return True
    metadata = current["metadata"]
    comparisons = (
        (score, float(metadata["score"]), 1),
        (biased_accuracy, float(metadata["biased_accuracy"]), 1),
        (biased_loss, float(metadata["biased_loss"]), -1),
    )
    for new, old, direction in comparisons:
        if abs(new - old) <= tolerance:
            continue
        return direction * new > direction * old
    return epoch < int(current["epoch"])


def _oracle_should_update(
    current: dict[str, Any] | None, metrics: dict[str, Any], epoch: int
) -> bool:
    if current is None:
        return True
    metadata = current["metadata"]
    for key in ("worst_group_accuracy", "group_balanced_accuracy", "accuracy"):
        new = float(metrics[key])
        old = float(metadata[key])
        if new != old:
            return new > old
    return epoch < int(current["epoch"])


def _atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _save_epoch_zero(
    root: Path,
    practical: dict[str, Any],
    oracle: dict[str, Any],
    test: dict[str, Any],
) -> None:
    visible = root / "epoch_zero_visible.npz"
    hidden = root / "exploratory_hidden_metrics" / "epoch_zero_hidden.npz"
    visible.parent.mkdir(parents=True, exist_ok=True)
    hidden.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(
        visible,
        img_id=practical["ordinary"]["img_id"],
        labels=practical["ordinary"]["labels"],
        logits=practical["ordinary"]["logits"],
        prediction=practical["ordinary"]["prediction"],
        correct=practical["ordinary"]["correct"],
        loss=practical["ordinary"]["loss"],
        scores_json=np.asarray(json.dumps(practical["scores"], sort_keys=True)),
    )
    _atomic_savez(
        hidden,
        oracle_img_id=oracle["img_id"],
        oracle_labels=oracle["labels"],
        oracle_groups=oracle["groups"],
        oracle_logits=oracle["logits"],
        oracle_prediction=oracle["prediction"],
        oracle_correct=oracle["correct"],
        oracle_loss=oracle["loss"],
        test_img_id=test["img_id"],
        test_labels=test["labels"],
        test_groups=test["groups"],
        test_logits=test["logits"],
        test_prediction=test["prediction"],
        test_correct=test["correct"],
        test_loss=test["loss"],
    )


def _ensure_shared_epoch_zero(
    output: Path,
    config: dict[str, Any],
    model,
    biased_val: pd.DataFrame,
    oracle_val: pd.DataFrame,
    test: pd.DataFrame,
    device,
    mask_bank: VlmMaskBank,
) -> dict[str, Any]:
    namespace = "debug/diagnostics/shared_epoch_zero" if config["runtime"]["debug"] else "diagnostics/shared_epoch_zero"
    root = output / namespace
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    visible = root / "epoch_zero_visible.npz"
    hidden = root / "exploratory_hidden_metrics" / "epoch_zero_hidden.npz"
    expected = {
        "schema_version": "anchorcal-shared-epoch-zero-v1",
        "architecture": config["candidate_grid"]["architecture"],
        "initialization_seed": int(config["candidate_grid"]["seed"]),
        "resolved_config_sha256": config["resolved_config_sha256"],
    }
    with ExclusiveFileLock(root / "epoch_zero.lock", blocking=True):
        if manifest_path.is_file():
            with manifest_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if any(existing.get(key) != value for key, value in expected.items()):
                raise PreflightError("shared epoch-zero manifest does not match this campaign")
            if (
                not visible.is_file()
                or not hidden.is_file()
                or existing.get("visible_sha256") != sha256_file(visible)
                or existing.get("hidden_sha256") != sha256_file(hidden)
            ):
                raise PreflightError("shared epoch-zero artifacts failed hash verification")
            return existing
        practical = evaluate_practical_criteria(
            model, biased_val, config, device, mask_bank=mask_bank
        )
        oracle = evaluate_plain(model, oracle_val, config, device)
        hidden_test = evaluate_plain(model, test, config, device)
        _save_epoch_zero(root, practical, oracle, hidden_test)
        manifest = {
            **expected,
            "visible_path": str(visible.resolve()),
            "visible_sha256": sha256_file(visible),
            "hidden_path": str(hidden.resolve()),
            "hidden_sha256": sha256_file(hidden),
            "primary_pool_eligible": False,
        }
        atomic_write_json(manifest_path, manifest)
        return manifest


def train_candidate_run(
    config: dict[str, Any], *, learning_rate: float, weight_decay: float
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    validate_grid_member(config, learning_rate, weight_decay)
    output = Path(config["paths"]["output_root"])
    receipt = _require_receipt(config)
    seed = int(config["candidate_grid"]["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CandidateViT(create_pretrained_vit(_model_weights(output))).to(device)
    candidate_train = _frame(output, "candidate_train")
    biased_val = _frame(output, "biased_val")
    oracle_val = _frame(output, "oracle_val")
    test = _frame(output, "test")
    selector_table = pd.read_csv(
        geometry_artifact_root(config) / "selector_eval_subset.csv"
    ).sort_values("img_id")
    run_id = candidate_run_id(learning_rate, weight_decay, seed)
    namespace = "debug/candidates" if config["runtime"]["debug"] else "candidates"
    run_dir = output / namespace / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = output / "preflight" / "report.json"
    if not preflight_path.is_file():
        raise PreflightError("successful preflight report is required before candidate training")
    with preflight_path.open("r", encoding="utf-8") as handle:
        preflight_report = json.load(handle)
    if preflight_report.get("status") != "passed":
        raise PreflightError("preflight report is not successful")
    if (
        not config["runtime"]["debug"]
        and preflight_report.get("resolved_config_sha256")
        != config["resolved_config_sha256"]
    ):
        raise PreflightError("production candidate config differs from preflight")
    if preflight_report.get("resolved_paths") != config["paths"]:
        raise PreflightError("resolved candidate paths differ from preflight")
    # This manifest is the immutable *scientific* identity of a candidate
    # trajectory.  Scheduler-attempt identity deliberately lives in the
    # append-only submission_receipts/jobs records instead: a preempted run
    # must be resumable by a replacement Slurm job with a different job ID.
    run_manifest = {
        "schema_version": "anchorcal-candidate-run-v3",
        "run_id": run_id,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "decision_receipt": str(receipt.resolve()),
        "decision_receipt_sha256": sha256_file(receipt),
        "resolved_config_sha256": config["resolved_config_sha256"],
        "preflight_report": str(preflight_path.resolve()),
        "preflight_report_sha256": sha256_file(preflight_path),
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
        "rng_policy": "stateless_sha256_geometry_and_epoch_sampler_plus_strongly_seeded_torch",
        "runtime_attempt_provenance": {
            "policy": "immutable_external_receipt_per_scheduler_attempt",
            "receipt_directory": str(
                (output / "submission_receipts" / "jobs").resolve()
            ),
            "final_successful_attempt_recorded_in": "completion.json",
        },
        "nondeterminism_warning_policy": (
            "scheduler_attempt_receipt_stderr_when_available_else_process_stderr"
        ),
    }
    run_manifest_path = run_dir / "run_manifest.json"
    if run_manifest_path.is_file():
        with run_manifest_path.open("r", encoding="utf-8") as handle:
            existing_manifest = json.load(handle)
        if existing_manifest != run_manifest:
            raise PreflightError(
                "candidate run manifest differs from the requested resume; "
                "the original provenance was preserved"
            )
    else:
        atomic_write_json(run_manifest_path, run_manifest)
    # A completed trajectory may be resumed long after its original job.  Do
    # not return its receipt until the authoritative source bank still matches
    # the frozen manifest byte-for-byte.
    mask_bank = load_vlm_mask_bank(config)
    completion_path = run_dir / "completion.json"
    if completion_path.is_file():
        storage_manifest = verify_candidate_storage(
            run_dir, expected_run_id=run_id
        )
        with completion_path.open("r", encoding="utf-8") as handle:
            completed = json.load(handle)
        checkpoint_verification = verify_candidate_checkpoint_artifacts(
            run_dir,
            expected_run_id=run_id,
            require_complete=True,
            required_visible_selectors=(
                "ordinary",
                "saliency",
                "swap",
                "blur",
                "final",
            ),
            required_hidden_selectors=("oracle",),
        )
        if (
            completed.get("run_id") != run_id
            or int(completed.get("epochs", -1))
            != int(config["candidate_grid"]["epochs"])
            or completed.get("checkpoint_manifest_sha256")
            != checkpoint_verification["visible"]["manifest_sha256"]
            or completed.get("hidden_checkpoint_manifest_sha256")
            != checkpoint_verification["hidden"]["manifest_sha256"]
        ):
            raise PreflightError("candidate completion receipt does not match this run")
        completed["storage_manifest"] = storage_manifest
        return completed
    preprocessing = load_preprocessing_manifest(output)
    dataset = CandidateTrainingDataset(
        candidate_train,
        config["paths"]["waterbirds_root"],
        seed,
        preprocessing=preprocessing,
    )
    sampler = EpochSampler(len(dataset), seed)
    workers = int(config["optimization"]["num_workers"])
    loader_options: dict[str, Any] = {
        "batch_size": int(config["candidate_grid"]["batch_size"]),
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
    loader = DataLoader(dataset, **loader_options)
    optimizer, optimizer_groups = make_adamw(
        model, learning_rate=learning_rate, weight_decay=weight_decay
    )
    atomic_write_json(run_dir / "optimizer_groups.json", optimizer_groups)
    epochs = int(config["candidate_grid"]["epochs"])
    updates_per_epoch = math.ceil(len(dataset) / int(config["candidate_grid"]["batch_size"]))
    scheduler = UpdateScheduler(
        optimizer,
        base_lr=learning_rate,
        total_updates=epochs * updates_per_epoch,
        warmup_updates=int(config["candidate_grid"]["warmup_epochs"]) * updates_per_epoch,
    )
    visible_metadata = SampleMetadata(
        biased_val["img_id"].to_numpy(), biased_val["y"].to_numpy()
    )
    hidden_metadata = {
        name: SampleMetadata(
            frame["img_id"].to_numpy(),
            frame["y"].to_numpy(),
            (frame["y"] * 2 + frame["place"]).to_numpy(),
        )
        for name, frame in (("oracle_val", oracle_val), ("test", test))
    }
    auxiliary_metadata = SampleMetadata(
        selector_table["img_id"].to_numpy(), selector_table["y"].to_numpy()
    )
    donors = int(config["criteria"]["swap_donors"])
    sigmas = len(config["criteria"]["blur_sigmas"])
    selector_shapes = candidate_per_example_shapes(
        len(selector_table), swap_donors=donors, blur_sigmas=sigmas
    )
    metric_names = CANDIDATE_SCALAR_METRICS
    epoch_zero_manifest = _ensure_shared_epoch_zero(
        output,
        config,
        model,
        biased_val,
        oracle_val,
        test,
        device,
        mask_bank,
    )
    history: list[dict[str, Any]] = []
    with CandidateStorage(
        run_dir,
        run_id=run_id,
        epoch_capacity=epochs,
        num_classes=2,
        selector_metadata=visible_metadata,
        hidden_metadata=hidden_metadata,
        selector_metric_names=metric_names,
        selector_auxiliary_metadata=auxiliary_metadata,
        selector_array_shapes=selector_shapes,
    ) as storage, CheckpointManager(run_dir, run_id=run_id) as checkpoints:
        resume = checkpoints.load_resume(map_location="cpu")
        if (
            storage.committed_slots == tuple(range(epochs))
            and resume is None
            and checkpoints.visible.get("completion") is not None
        ):
            storage.finalize()
            storage_manifest = verify_candidate_storage(
                run_dir, expected_run_id=run_id
            )
            checkpoint_verification = verify_candidate_checkpoint_artifacts(
                run_dir,
                expected_run_id=run_id,
                require_complete=True,
                required_visible_selectors=(
                    "ordinary",
                    "saliency",
                    "swap",
                    "blur",
                    "final",
                ),
                required_hidden_selectors=("oracle",),
            )
            result = {
                "run_id": run_id,
                "epochs": epochs,
                "storage_manifest": storage_manifest,
                "checkpoint_manifest": checkpoint_verification["visible"][
                    "manifest_path"
                ],
                "checkpoint_manifest_sha256": checkpoint_verification["visible"][
                    "manifest_sha256"
                ],
                "hidden_checkpoint_manifest": checkpoint_verification["hidden"][
                    "manifest_path"
                ],
                "hidden_checkpoint_manifest_sha256": checkpoint_verification[
                    "hidden"
                ]["manifest_sha256"],
                "retained_checkpoint_weight_count": checkpoint_verification[
                    "retained_weight_count"
                ],
                "epoch_zero_manifest": epoch_zero_manifest,
                "final_runtime_job_receipt": os.environ.get("ANCHORCAL_JOB_RECEIPT"),
                "final_runtime_job_receipt_sha256": os.environ.get(
                    "ANCHORCAL_JOB_RECEIPT_SHA256"
                ),
            }
            atomic_write_json(completion_path, result)
            return result
        start_epoch = 1
        pending_epoch: int | None = None
        pending_training: dict[str, float] | None = None
        if resume is not None:
            if resume["metadata"].get("resolved_config_sha256") != config["resolved_config_sha256"]:
                raise PreflightError("candidate resume resolved-config hash mismatch")
            model.load_state_dict(resume["model_state"])
            optimizer.load_state_dict(resume["optimizer_state"])
            scheduler.load_state_dict(resume["scheduler_state"])
            _restore_rng(resume["rng_states"])
            phase = resume["metadata"].get("phase", "complete")
            if phase == "pending_evaluation":
                pending_epoch = int(resume["epoch"])
                pending_training = dict(resume["metadata"]["training"])
                start_epoch = pending_epoch
            elif phase == "complete":
                start_epoch = int(resume["epoch"]) + 1
            else:
                raise PreflightError(f"unknown candidate resume phase: {phase}")
            history = list(resume["metadata"].get("history", []))
        allowed_counts = {start_epoch - 1}
        if pending_epoch is not None:
            # A crash can occur after the paired HDF5 commit but before the
            # post-evaluation resume publication. Re-evaluate selectors and
            # continue without rewriting that committed slot.
            allowed_counts.add(start_epoch)
        if len(storage.committed_slots) not in allowed_counts:
            raise PreflightError(
                "resume checkpoint epoch and paired HDF5 completion count disagree"
            )
        for epoch in range(start_epoch, epochs + 1):
            if pending_epoch == epoch:
                assert pending_training is not None
                training = pending_training
            else:
                sampler.set_epoch(epoch)
                training = _train_epoch(model, loader, optimizer, scheduler, device)
                pending_state = {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                }
                checkpoints.save_resume(
                    epoch=epoch,
                    model_state=pending_state,
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict(),
                    rng_states=_rng_state(),
                    scaler_state=None,
                    dataloader_progress={"at_epoch_boundary": True},
                    metadata={
                        "resolved_config_sha256": config["resolved_config_sha256"],
                        "history": history,
                        "phase": "pending_evaluation",
                        "training": training,
                    },
                )
            practical = evaluate_practical_criteria(
                model, biased_val, config, device, mask_bank=mask_bank
            )
            oracle = evaluate_plain(model, oracle_val, config, device)
            hidden_test = evaluate_plain(model, test, config, device)
            biased_loss = float(np.mean(practical["ordinary"]["loss"]))
            scalar_metrics = {
                "ordinary_accuracy": float(practical["scores"]["ordinary_accuracy"]),
                "saliency_harmonic": float(practical["scores"]["saliency_harmonic"]),
                "token_swap_harmonic": float(practical["scores"]["token_swap_harmonic"]),
                "background_blur_harmonic": float(
                    practical["scores"]["background_blur_harmonic"]
                ),
                "foreground_only_harmonic": float(
                    practical["scores"]["foreground_only_harmonic"]
                ),
                **{
                    name: float(practical["scores"][name])
                    for name in (
                        "saliency_alignment",
                        "swap_accuracy",
                        "blur_accuracy",
                        "foreground_only_accuracy",
                        "saliency_product",
                        "swap_product",
                        "blur_product",
                        "swap_mean_true_class_margin_drop",
                        "swap_prediction_flip_rate",
                        "swap_donor_margin_variance",
                    )
                },
                "biased_mean_loss": biased_loss,
            }
            selector_per_example = {
                name: practical[name].astype(np.float32)
                for name in selector_shapes
            }
            if (epoch - 1) not in storage.committed_slots:
                storage.write_epoch(
                    slot=epoch - 1,
                    epoch_number=epoch,
                    selector=_prediction_batch(practical["ordinary"]),
                    hidden={
                        "oracle_val": _prediction_batch(oracle),
                        "test": _prediction_batch(hidden_test),
                    },
                    selector_metrics=scalar_metrics,
                    selector_per_example=selector_per_example,
                )
            model_state = {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            }
            tolerance = float(config["anchorcal"]["candidate_score_tolerance"])
            for selector, score_key in ELIGIBLE_SCORE_KEYS.items():
                current = checkpoints.visible["selectors"][selector]
                score = scalar_metrics[score_key]
                if _practical_should_update(
                    current,
                    score=score,
                    biased_accuracy=scalar_metrics["ordinary_accuracy"],
                    biased_loss=biased_loss,
                    epoch=epoch,
                    tolerance=tolerance,
                ):
                    checkpoints.update_selector(
                        selector,
                        epoch=epoch,
                        model_state=model_state,
                        ranking_key=(score, scalar_metrics["ordinary_accuracy"], -biased_loss, -epoch),
                        metadata={
                            "score": score,
                            "biased_accuracy": scalar_metrics["ordinary_accuracy"],
                            "biased_loss": biased_loss,
                        },
                        force=True,
                    )
            oracle_current = checkpoints.hidden["selectors"]["oracle"]
            if _oracle_should_update(oracle_current, oracle["metrics"], epoch):
                checkpoints.update_selector(
                    "oracle",
                    epoch=epoch,
                    model_state=model_state,
                    ranking_key=(
                        oracle["metrics"]["worst_group_accuracy"],
                        oracle["metrics"]["group_balanced_accuracy"],
                        oracle["metrics"]["accuracy"],
                        -epoch,
                    ),
                    metadata={
                        key: oracle["metrics"][key]
                        for key in (
                            "worst_group_accuracy",
                            "group_balanced_accuracy",
                            "accuracy",
                        )
                    },
                    force=True,
                )
            if epoch == epochs:
                checkpoints.save_final_epoch(epoch=epoch, model_state=model_state)
            epoch_history = {
                "epoch": epoch,
                "training": training,
                "selector_metrics": scalar_metrics,
                "oracle_metrics_stored_hidden": True,
                "test_metrics_stored_hidden": True,
            }
            history = [item for item in history if int(item["epoch"]) != epoch]
            history.append(epoch_history)
            history.sort(key=lambda item: int(item["epoch"]))
            atomic_write_json(run_dir / "training_history.json", history)
            checkpoints.save_resume(
                epoch=epoch,
                model_state=model_state,
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                rng_states=_rng_state(),
                scaler_state=None,
                dataloader_progress={"at_epoch_boundary": True},
                metadata={
                    "resolved_config_sha256": config["resolved_config_sha256"],
                    "history": history,
                    "phase": "complete",
                },
            )
            pending_epoch = None
            pending_training = None
        storage.finalize()
        checkpoints.complete(resume_policy="archive")
    storage_manifest = verify_candidate_storage(run_dir, expected_run_id=run_id)
    checkpoint_verification = verify_candidate_checkpoint_artifacts(
        run_dir,
        expected_run_id=run_id,
        require_complete=True,
        required_visible_selectors=("ordinary", "saliency", "swap", "blur", "final"),
        required_hidden_selectors=("oracle",),
    )
    result = {
        "run_id": run_id,
        "epochs": epochs,
        "storage_manifest": storage_manifest,
        "checkpoint_manifest": checkpoint_verification["visible"]["manifest_path"],
        "checkpoint_manifest_sha256": checkpoint_verification["visible"][
            "manifest_sha256"
        ],
        "hidden_checkpoint_manifest": checkpoint_verification["hidden"][
            "manifest_path"
        ],
        "hidden_checkpoint_manifest_sha256": checkpoint_verification["hidden"][
            "manifest_sha256"
        ],
        "retained_checkpoint_weight_count": checkpoint_verification[
            "retained_weight_count"
        ],
        "epoch_zero_manifest": epoch_zero_manifest,
        "final_runtime_job_receipt": os.environ.get("ANCHORCAL_JOB_RECEIPT"),
        "final_runtime_job_receipt_sha256": os.environ.get(
            "ANCHORCAL_JOB_RECEIPT_SHA256"
        ),
    }
    atomic_write_json(completion_path, result)
    print(
        "EXPLORATORY_TEST_ONLY: per-epoch test outputs were stored in the hidden HDF5; "
        "no test values were used for selection."
    )
    return result
