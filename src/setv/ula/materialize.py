"""Recover the uLA-selected candidate state without retaining all 50 epochs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from setv.candidate.train import (
    _evaluate,
    _prepare,
    _train_epoch,
    verify_candidate,
)
from setv.errors import ArtifactExistsError, DataValidationError
from setv.experts.train_object import (
    _atomic_torch_save,
    _build_optimizer_scheduler,
    _grad_scaler,
)
from setv.utils.hashing import sha256_file
from setv.utils.io import write_json


def _relative_to_phase6_staging(output_dir: Path, path: Path) -> str:
    # output_dir is <phase6-staging>/selected_candidate_checkpoints/seed_N
    return path.relative_to(output_dir.parents[1]).as_posix()


def materialize_ula_selected_checkpoint(
    candidate_dir: Path, selected_epoch: int, output_dir: Path
) -> dict[str, Any]:
    """Copy a deduplicated state or deterministically replay to the selected epoch."""
    import torch

    candidate_dir = Path(candidate_dir).expanduser().resolve()
    output_dir = Path(output_dir).resolve()
    verify_candidate(candidate_dir, load_checkpoints=False)
    if not 1 <= int(selected_epoch) <= 50:
        raise DataValidationError("uLA selected epoch must lie in [1,50]")
    if output_dir.exists():
        raise ArtifactExistsError(f"uLA checkpoint output exists: {output_dir}")
    output_dir.mkdir(parents=True)
    destination = output_dir / f"ula_selected_epoch_{selected_epoch:03d}.pt"
    existing = (
        candidate_dir
        / "selection"
        / "checkpoints"
        / f"epoch_{selected_epoch:03d}.pt"
    )
    if not existing.is_file():
        oracle = (
            candidate_dir
            / "analysis_only"
            / "oracle_checkpoints"
            / f"epoch_{selected_epoch:03d}.pt"
        )
        existing = oracle if oracle.is_file() else existing
    if existing.is_file():
        shutil.copy2(existing, destination)
        method = "copied_deduplicated_phase5_checkpoint"
        replay = None
    else:
        with (
            candidate_dir / "config" / "resolved_candidate.yaml"
        ).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        prepared = _prepare(config, model_factory=None)
        optimizer, scheduler = _build_optimizer_scheduler(
            prepared["model"],
            config,
            len(prepared["loaders"]["candidate_train"]),
        )
        mixed = bool(config["training"]["mixed_precision"]) and (
            prepared["device"].type == "cuda"
        )
        scaler = _grad_scaler(prepared["device"], mixed)
        for epoch_index in range(int(selected_epoch)):
            _train_epoch(
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
        reproduced = _evaluate(
            prepared["model"],
            prepared["loaders"]["biased_val"],
            prepared["device"],
        )
        epoch_path = (
            candidate_dir
            / "biased_val"
            / "epoch_predictions"
            / f"epoch_{selected_epoch:03d}.npz"
        )
        with np.load(epoch_path, allow_pickle=False) as archive:
            expected_ids = archive["sample_id"].astype(str)
            expected_logits = archive["candidate_logits"]
        if not np.array_equal(reproduced["sample_id"].astype(str), expected_ids):
            raise DataValidationError("Candidate replay sample ordering changed")
        maximum_error = float(
            np.max(np.abs(reproduced["logits"] - expected_logits))
        )
        if not np.allclose(
            reproduced["logits"], expected_logits, rtol=1e-5, atol=1e-5
        ):
            raise DataValidationError(
                "Candidate replay did not reproduce the selected epoch; "
                f"maximum logit error={maximum_error}"
            )
        phase5 = json.loads((candidate_dir / "phase5_receipt.json").read_text())
        _atomic_torch_save(
            {
                "format_version": 1,
                "kind": "setv_candidate_erm_selected",
                "selector": "uLA-style",
                "architecture": config["model"]["architecture"],
                "seed": int(config["training"]["seed"]),
                "epoch": int(selected_epoch),
                "phase0_artifact_manifest_sha256": phase5[
                    "phase0_artifact_manifest_sha256"
                ],
                "normalization_mean": config["model"]["normalization_mean"],
                "normalization_std": config["model"]["normalization_std"],
                "model_state_dict": {
                    key: value.detach().cpu()
                    for key, value in prepared["model"].state_dict().items()
                },
            },
            destination,
        )
        method = "deterministic_training_replay"
        replay = {
            "verified_against_epoch_logits": True,
            "maximum_absolute_logit_error": maximum_error,
            "rtol": 1e-5,
            "atol": 1e-5,
        }
    checkpoint = torch.load(destination, map_location="cpu", weights_only=False)
    if int(checkpoint.get("epoch", -1)) != int(selected_epoch):
        raise DataValidationError("Materialized candidate checkpoint has wrong epoch")
    receipt = {
        "path": _relative_to_phase6_staging(output_dir, destination),
        "sha256": sha256_file(destination),
        "epoch": int(selected_epoch),
        "source_candidate_dir": str(candidate_dir),
        "materialization_method": method,
        "replay": replay,
    }
    write_json(output_dir / "materialization_receipt.json", receipt)
    return receipt
