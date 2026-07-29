"""uLA-style frozen-SSL linear bias proxy training and artifacts."""

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
from PIL import Image

from setv.errors import ArtifactExistsError, DataValidationError
from setv.experts.object_data import normalized_tensor
from setv.experts.train_object import (
    _artifact_manifest,
    _atomic_torch_save,
    _autocast,
    _grad_scaler,
    _load_phase0_config,
)
from setv.phase0 import APPROVAL_RECEIPT, BASE_ARTIFACT_MANIFEST, verify_phase0
from setv.ula.config import resolved_config
from setv.ula.provenance import audit_official_source
from setv.utils.hashing import sha256_file, sha256_json
from setv.utils.io import write_json
from setv.utils.logging import EventLogger
from setv.utils.seeds import derive_seed, seed_python_numpy, seed_torch_if_available

ModelFactory = Callable[[dict[str, Any]], Any]


class ULAOriginalImageDataset:
    """Selector-safe original-image data with the official Waterbirds geometry."""

    def __init__(self, manifest: Path, phase0_config: dict, config: dict, training: bool):
        from setv.data.joint_transforms import build_eval_transform, build_train_transform

        self.rows = pd.read_csv(manifest, dtype={"sample_id": str})
        if set(self.rows) < {"sample_id", "img_filename", "y"}:
            raise DataValidationError("uLA manifest lacks safe ID/image/target columns")
        forbidden = {"place", "group", "group_id", "background", "confounder"}
        if forbidden.intersection(self.rows.columns):
            raise DataValidationError("uLA proxy input manifest exposes protected labels")
        self.dataset_root = Path(phase0_config["data"]["dataset_root"])
        self.phase0_config = phase0_config
        self.training = bool(training)
        self.seed = int(config["training"]["seed"])
        self.epoch = 0
        self.train_builder = build_train_transform
        self.eval_transform = build_eval_transform(phase0_config)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        with Image.open(self.dataset_root / str(row["img_filename"])) as opened:
            image = opened.convert("RGB")
        blank = Image.new("L", image.size, 0)
        if self.training:
            transform = self.train_builder(
                self.phase0_config,
                derive_seed(
                    self.seed,
                    f"ula_proxy_transform:epoch={self.epoch}:sample={row['sample_id']}",
                ),
            )
        else:
            transform = self.eval_transform
        image, _ = transform(image, blank)
        return {
            "image": normalized_tensor(
                image,
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
            "target": int(row["y"]),
            "sample_id": str(row["sample_id"]),
        }


def _extract_backbone_state(checkpoint: dict[str, Any]) -> dict[str, Any]:
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise DataValidationError("Official SSL checkpoint has no state dictionary")
    prefixes = (
        "backbone.",
        "momentum_backbone.",
        "encoder_q.",
        "module.encoder_q.",
    )
    best: dict[str, Any] = {}
    for prefix in prefixes:
        selected = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
            and not key[len(prefix) :].startswith(("fc.", "classifier."))
        }
        if len(selected) > len(best):
            best = selected
    if not best:
        raise DataValidationError(
            "Could not find a ResNet backbone in the official SSL checkpoint"
        )
    return best


def create_proxy_model(config: dict[str, Any]):
    import torch
    import torch.nn as nn
    import torchvision

    checkpoint = torch.load(
        Path(config["ssl_checkpoint"]).expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    encoder = torchvision.models.resnet50(weights=None)
    encoder.fc = nn.Identity()
    missing, unexpected = encoder.load_state_dict(
        _extract_backbone_state(checkpoint), strict=False
    )
    material_missing = [
        key for key in missing if not key.startswith(("fc.", "num_batches_tracked"))
    ]
    material_unexpected = [key for key in unexpected if not key.startswith("fc.")]
    if material_missing or material_unexpected:
        raise DataValidationError(
            "Official SSL backbone is incompatible with torchvision ResNet-50: "
            f"missing={material_missing[:8]}, unexpected={material_unexpected[:8]}"
        )
    encoder.requires_grad_(False)

    class FrozenLinearProxy(nn.Module):
        def __init__(self, feature_extractor):
            super().__init__()
            self.encoder = feature_extractor
            self.head = nn.Linear(2048, 2)

        def train(self, mode: bool = True):
            super().train(mode)
            self.encoder.eval()
            return self

        def forward(self, inputs):
            self.encoder.eval()
            with torch.no_grad():
                features = self.encoder(inputs)
            return self.head(features)

    return FrozenLinearProxy(encoder)


def _worker_init(worker_id: int, *, seed: int) -> None:
    import torch

    value = derive_seed(seed, f"ula_proxy_worker={worker_id}")
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def _loaders(phase0_dir: Path, phase0_config: dict, config: dict):
    import torch

    split_dir = phase0_dir / "splits"
    train = ULAOriginalImageDataset(
        split_dir / "waterbirds95_candidate_train.csv",
        phase0_config,
        config,
        True,
    )
    valid = ULAOriginalImageDataset(
        split_dir / "waterbirds95_biased_val.csv",
        phase0_config,
        config,
        False,
    )
    settings = config["training"]
    generator = torch.Generator()
    generator.manual_seed(derive_seed(int(settings["seed"]), "ula_proxy_shuffle"))
    common = {
        "num_workers": int(settings["num_workers"]),
        "pin_memory": settings["device"] == "cuda",
        "persistent_workers": False,
        "worker_init_fn": partial(_worker_init, seed=int(settings["seed"])),
    }
    return train, valid, torch.utils.data.DataLoader(
        train,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        generator=generator,
        drop_last=False,
        **common,
    ), torch.utils.data.DataLoader(
        valid,
        batch_size=int(settings["evaluation_batch_size"]),
        shuffle=False,
        drop_last=False,
        **common,
    )


def _evaluate(model, loader, device) -> dict[str, np.ndarray]:
    import torch

    model.eval()
    ids, labels, logits = [], [], []
    with torch.inference_mode():
        for batch in loader:
            ids.extend(str(value) for value in batch["sample_id"])
            labels.append(batch["target"].numpy())
            logits.append(model(batch["image"].to(device)).float().cpu().numpy())
    labels_array = np.concatenate(labels).astype(np.int64)
    logits_array = np.concatenate(logits).astype(np.float32)
    predicted = logits_array.argmax(1).astype(np.int64)
    return {
        "sample_id": np.asarray(ids, dtype=np.str_),
        "true_label": labels_array,
        "ula_proxy_logits": logits_array,
        "ula_proxy_predicted_class": predicted,
        "ula_proxy_correct": (predicted == labels_array).astype(np.uint8),
    }


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def train_ula_proxy(
    config: dict[str, Any], *, model_factory: ModelFactory | None = None
) -> Path:
    import torch

    phase0_dir = Path(config["phase0_dir"]).expanduser().resolve()
    verify_phase0(phase0_dir, require_approval=True)
    ssl_path = Path(config["ssl_checkpoint"]).expanduser().resolve()
    if not ssl_path.is_file() and model_factory is None:
        raise DataValidationError(f"Official SSL checkpoint is missing: {ssl_path}")
    official = audit_official_source(config["official_repo"])
    seed = int(config["training"]["seed"])
    destination = Path(config["output_root"]).expanduser().resolve() / f"seed_{seed}"
    if destination.exists():
        raise ArtifactExistsError(f"uLA proxy output exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / (
        f".seed_{seed}.building."
        f"{os.environ.get('SLURM_JOB_ID', uuid.uuid4().hex[:12])}"
    )
    for directory in ("scores", "checkpoints", "config", "provenance"):
        (staging / directory).mkdir(parents=True, exist_ok=True)
    logger = EventLogger(staging / "training.jsonl", echo=True)
    try:
        seed_receipt = seed_python_numpy(seed)
        seed_receipt.update(seed_torch_if_available(seed))
        device_name = config["training"]["device"]
        if device_name == "cuda" and not torch.cuda.is_available():
            raise DataValidationError("uLA proxy config requires unavailable CUDA")
        device = torch.device(device_name)
        phase0_config = _load_phase0_config(phase0_dir)
        train_set, _, train_loader, valid_loader = _loaders(
            phase0_dir, phase0_config, config
        )
        model = (model_factory or create_proxy_model)(config).to(device)
        parameters = [value for value in model.parameters() if value.requires_grad]
        optimizer = torch.optim.SGD(
            parameters,
            lr=float(config["training"]["learning_rate"]),
            momentum=float(config["training"]["momentum"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        scaler = _grad_scaler(
            device,
            bool(config["training"]["mixed_precision"]) and device.type == "cuda",
        )
        criterion = torch.nn.CrossEntropyLoss()
        history = []
        for epoch in range(int(config["training"]["epochs"])):
            train_set.set_epoch(epoch)
            model.train()
            total_loss = correct = count = 0.0
            started = time.monotonic()
            for batch in train_loader:
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["target"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device, scaler.is_enabled()):
                    logits = model(images)
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                size = int(targets.numel())
                total_loss += float(loss.detach()) * size
                correct += int((logits.detach().argmax(1) == targets).sum())
                count += size
            values = {
                "epoch": epoch + 1,
                "loss": total_loss / count,
                "accuracy": correct / count,
                "sample_count": int(count),
                "seconds": time.monotonic() - started,
            }
            history.append(values)
            logger.log("ula_proxy_epoch_complete", **values)
        scores = _evaluate(model, valid_loader, device)
        score_path = staging / "scores" / "biased_val_proxy_scores.npz"
        _atomic_npz(score_path, scores)
        checkpoint_path = staging / "checkpoints" / "final_proxy.pt"
        _atomic_torch_save(
            {
                "format_version": 1,
                "kind": "setv_ula_style_proxy",
                "seed": seed,
                "official_commit": official["required_commit"],
                "ssl_checkpoint_sha256": (
                    sha256_file(ssl_path) if ssl_path.is_file() else "injected-test-model"
                ),
                "model_state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
            },
            checkpoint_path,
        )
        resolved = resolved_config(config)
        with (staging / "config" / "resolved_ula_proxy.yaml").open(
            "w", encoding="utf-8"
        ) as handle:
            yaml.safe_dump(resolved, handle, sort_keys=True)
        write_json(staging / "provenance" / "official_source.json", official)
        write_json(
            staging / "provenance" / "seed.json",
            seed_receipt,
        )
        receipt = {
            "schema_version": 1,
            "status": "complete",
            "kind": "setv_ula_style_bias_proxy",
            "label": "uLA-style",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "reason_not_exact_ula": (
                "The official uLA validation rule is applied to independently trained "
                "SETV ViT candidate epochs rather than to uLA's own debiased ResNet trajectory."
            ),
            "seed": seed,
            "phase0_dir": str(phase0_dir),
            "phase0_artifact_manifest_sha256": sha256_file(
                phase0_dir / BASE_ARTIFACT_MANIFEST
            ),
            "phase0_visual_approval_sha256": sha256_file(
                phase0_dir / APPROVAL_RECEIPT
            ),
            "official_source": official,
            "ssl_encoder": {
                "method": "MoCoV2+",
                "backbone": "ResNet-50",
                "checkpoint_path": str(ssl_path),
                "checkpoint_sha256": (
                    sha256_file(ssl_path) if ssl_path.is_file() else "injected-test-model"
                ),
            },
            "linear_bias_classifier": {
                "target": "bird class y",
                "encoder_frozen": True,
                "epochs": 50,
                "optimizer": "SGD",
                "learning_rate": 1e-4,
                "momentum": 0.9,
                "weight_decay": 0.0,
            },
            "calibration": "none",
            "proxy_group_construction": "(argmax(proxy_logits), true_label)",
            "validation_formula": (
                "mean accuracy over nonempty (proxy_prediction,true_label) groups"
            ),
            "trained_on": "full candidate_train",
            "candidate_training_data_withheld": False,
            "protected_group_labels_used": False,
            "scores": {
                "path": score_path.relative_to(staging).as_posix(),
                "sha256": sha256_file(score_path),
                "sample_count": len(scores["sample_id"]),
                "accuracy_diagnostic_only": float(scores["ula_proxy_correct"].mean()),
            },
            "checkpoint": {
                "path": checkpoint_path.relative_to(staging).as_posix(),
                "sha256": sha256_file(checkpoint_path),
            },
            "config_sha256": sha256_json(resolved),
            "history": history,
        }
        write_json(staging / "phase6_ula_proxy_receipt.json", receipt)
        write_json(staging / "artifact_manifest.json", _artifact_manifest(staging))
        os.rename(staging, destination)
        return destination
    except Exception:
        raise


def load_ula_proxy_scores(root: str | Path) -> dict[str, np.ndarray]:
    root = Path(root).expanduser().resolve()
    receipt = json.loads((root / "phase6_ula_proxy_receipt.json").read_text())
    with np.load(root / receipt["scores"]["path"], allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    expected = {
        "sample_id",
        "true_label",
        "ula_proxy_logits",
        "ula_proxy_predicted_class",
        "ula_proxy_correct",
    }
    if set(payload) != expected:
        raise DataValidationError("uLA proxy score schema changed")
    if payload["ula_proxy_logits"].shape != (len(payload["sample_id"]), 2):
        raise DataValidationError("uLA proxy logit shape is invalid")
    if not np.array_equal(
        payload["ula_proxy_logits"].argmax(1),
        payload["ula_proxy_predicted_class"],
    ):
        raise DataValidationError("uLA proxy predictions disagree with logits")
    return payload


def verify_ula_proxy(root: str | Path, *, load_checkpoint: bool = True) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    if sha256_json(manifest["files"]) != manifest["manifest_digest"]:
        raise DataValidationError("uLA proxy manifest digest is invalid")
    for relative, expected in manifest["files"].items():
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != expected["size_bytes"]
            or sha256_file(path) != expected["sha256"]
        ):
            raise DataValidationError(f"Missing/changed uLA proxy artifact: {relative}")
    receipt = json.loads((root / "phase6_ula_proxy_receipt.json").read_text())
    if receipt.get("kind") != "setv_ula_style_bias_proxy":
        raise DataValidationError("Wrong uLA proxy artifact kind")
    if receipt["official_source"]["required_commit"] != (
        "5867fb6e9a8485ed08b4cbe84900f2b5ac4fac5d"
    ):
        raise DataValidationError("uLA upstream commit binding changed")
    phase0_dir = Path(receipt["phase0_dir"])
    verify_phase0(phase0_dir, require_approval=True)
    if receipt["phase0_artifact_manifest_sha256"] != sha256_file(
        phase0_dir / BASE_ARTIFACT_MANIFEST
    ):
        raise DataValidationError("uLA proxy Phase 0 binding changed")
    scores = load_ula_proxy_scores(root)
    checkpoint = root / receipt["checkpoint"]["path"]
    if sha256_file(checkpoint) != receipt["checkpoint"]["sha256"]:
        raise DataValidationError("uLA proxy checkpoint hash changed")
    if load_checkpoint:
        import torch

        value = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if value.get("kind") != "setv_ula_style_proxy":
            raise DataValidationError("Wrong uLA proxy checkpoint kind")
    return {
        "status": "complete",
        "label": receipt["label"],
        "sample_count": len(scores["sample_id"]),
        "official_commit": receipt["official_source"]["required_commit"],
        "artifact_count": len(manifest["files"]),
    }
