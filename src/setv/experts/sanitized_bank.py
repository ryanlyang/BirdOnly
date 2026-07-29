"""Build and verify immutable Phase 3 sanitized-mask banks."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from setv.data.dataset import WaterbirdsManifestDataset
from setv.data.joint_transforms import build_eval_transform
from setv.errors import ArtifactExistsError, DataValidationError
from setv.experts.exact_data import dilate_binary_mask
from setv.experts.sanitized_audit import run_mask_auditors
from setv.experts.sanitized_config import resolved_config
from setv.experts.sanitized_masks import (
    FAMILIES,
    generate_sanitized_bank,
    globally_balanced_short_families,
    pack_masks,
    unpack_masks,
)
from setv.experts.train_object import _artifact_manifest, _load_phase0_config
from setv.phase0 import APPROVAL_RECEIPT, BASE_ARTIFACT_MANIFEST, verify_phase0
from setv.utils.hashing import sha256_file, sha256_json
from setv.utils.io import write_json


BANK_FILENAMES = {
    "candidate_train": "candidate_train_mask_bank.npz",
    "biased_val": "biased_val_mask_bank.npz",
}
AuditorRunner = Callable[..., tuple[dict[str, Any], dict[str, np.ndarray]]]


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


def _dataset(phase0_dir: Path, phase0_config: dict, split: str):
    masks = phase0_config["masks"]
    return WaterbirdsManifestDataset(
        phase0_dir / "splits" / f"waterbirds95_{split}.csv",
        phase0_config["data"]["dataset_root"],
        masks["root"],
        threshold_normalized=float(masks["threshold_normalized"]),
        foreground_is_high=bool(masks["foreground_is_high"]),
    )


def _create_split_bank(
    phase0_dir: Path,
    phase0_config: dict,
    split: str,
    short_family: dict[str, str],
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    dataset = _dataset(phase0_dir, phase0_config, split)
    transform = build_eval_transform(phase0_config)
    image_size = int(phase0_config["transforms"]["image_size"])
    dilation_radius = max(
        1,
        int(
            round(
                int(config["mask_bank"]["dilation_pixels_at_224"])
                * image_size
                / 224.0
            )
        ),
    )
    packed_rows = []
    family_rows = []
    seed_rows = []
    sample_ids = []
    area_by_family = {family: [] for family in FAMILIES}
    for index in range(len(dataset)):
        example = dataset[index]
        _, transformed_mask = transform(example.image, example.mask)
        generated, families, seeds, parameters = generate_sanitized_bank(
            transformed_mask,
            sample_id=example.sample_id,
            short_family=short_family[example.sample_id],
            base_seed=int(config["mask_bank"]["seed"]),
            config=config["mask_bank"],
            dilation_radius=dilation_radius,
        )
        packed_rows.append(pack_masks(generated[None, ...])[0])
        family_rows.append(families)
        seed_rows.append(seeds)
        sample_ids.append(example.sample_id)
        for parameter in parameters:
            area_by_family[parameter["family"]].append(parameter["area_fraction"])
    payload = {
        "sample_id": np.asarray(sample_ids, dtype=np.str_),
        "packed_masks": np.stack(packed_rows).astype(np.uint8),
        "family_id": np.stack(family_rows).astype(np.uint8),
        "mask_seed": np.stack(seed_rows).astype(np.uint32),
        "height": np.asarray(image_size, dtype=np.int64),
        "width": np.asarray(image_size, dtype=np.int64),
        "dilation_radius": np.asarray(dilation_radius, dtype=np.int64),
    }
    validate_bank_payload(payload)
    summary = {
        "split": split,
        "sample_count": len(sample_ids),
        "masks_per_image": 8,
        "image_size": image_size,
        "dilation_radius": dilation_radius,
        "family_mask_counts": {
            family: int((payload["family_id"] == family_id).sum())
            for family_id, family in enumerate(FAMILIES)
        },
        "family_area_fraction": {
            family: {
                "minimum": float(np.min(values)),
                "mean": float(np.mean(values)),
                "maximum": float(np.max(values)),
            }
            for family, values in area_by_family.items()
        },
    }
    return payload, summary


def validate_bank_payload(payload: dict[str, np.ndarray]) -> None:
    expected = {
        "sample_id",
        "packed_masks",
        "family_id",
        "mask_seed",
        "height",
        "width",
        "dilation_radius",
    }
    if set(payload) != expected:
        raise DataValidationError(
            f"Sanitized bank keys must be {sorted(expected)}, got {sorted(payload)}"
        )
    ids = np.asarray(payload["sample_id"]).astype(str)
    packed = np.asarray(payload["packed_masks"], dtype=np.uint8)
    families = np.asarray(payload["family_id"], dtype=np.uint8)
    seeds = np.asarray(payload["mask_seed"], dtype=np.uint32)
    height = int(np.asarray(payload["height"]).item())
    width = int(np.asarray(payload["width"]).item())
    if len(set(ids.tolist())) != len(ids):
        raise DataValidationError("Sanitized bank sample IDs are not unique")
    if packed.shape != (len(ids), 8, height, (width + 7) // 8):
        raise DataValidationError(f"Invalid packed sanitized-mask shape: {packed.shape}")
    if families.shape != (len(ids), 8) or seeds.shape != (len(ids), 8):
        raise DataValidationError("Sanitized family/seed arrays are misaligned")
    if not set(np.unique(families)).issubset({0, 1, 2}):
        raise DataValidationError("Sanitized bank contains an invalid family ID")
    for row in families:
        counts = sorted(int((row == family_id).sum()) for family_id in range(3))
        if counts != [2, 3, 3]:
            raise DataValidationError(f"Invalid per-image family allocation: {counts}")
    masks = unpack_masks(packed[: min(16, len(packed))], width)
    if not np.all(masks.any(axis=(-2, -1))):
        raise DataValidationError("Sanitized bank contains an empty mask")


def save_bank(path: Path, payload: dict[str, np.ndarray]) -> None:
    validate_bank_payload(payload)
    _atomic_npz(path, payload)


def load_bank(root: str | Path, split: str) -> dict[str, np.ndarray]:
    if split not in BANK_FILENAMES:
        raise DataValidationError(f"Unknown sanitized bank split: {split}")
    with np.load(Path(root) / BANK_FILENAMES[split], allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    validate_bank_payload(payload)
    return payload


def build_sanitized_mask_bank(
    config: dict[str, Any],
    *,
    auditor_runner: AuditorRunner = run_mask_auditors,
) -> Path:
    phase0_dir = Path(config["phase0_dir"]).expanduser().resolve()
    verify_phase0(phase0_dir, require_approval=True)
    phase0_config = _load_phase0_config(phase0_dir)
    seed = int(config["mask_bank"]["seed"])
    destination = Path(config["output_root"]).expanduser().resolve() / f"seed_{seed}"
    if destination.exists():
        raise ArtifactExistsError(f"Sanitized mask-bank output exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("SLURM_JOB_ID", uuid.uuid4().hex[:12])
    staging = destination.parent / f".seed_{seed}.building.{token}"
    staging.mkdir()
    try:
        manifests = {
            split: pd.read_csv(
                phase0_dir / "splits" / f"waterbirds95_{split}.csv",
                dtype={"sample_id": str},
            )
            for split in BANK_FILENAMES
        }
        all_ids = [
            sample_id
            for split in BANK_FILENAMES
            for sample_id in manifests[split]["sample_id"].astype(str)
        ]
        short_family = globally_balanced_short_families(all_ids)
        summaries = {}
        payloads = {}
        for split, filename in BANK_FILENAMES.items():
            payload, summary = _create_split_bank(
                phase0_dir, phase0_config, split, short_family, config
            )
            expected_ids = manifests[split]["sample_id"].astype(str).to_numpy()
            if not np.array_equal(payload["sample_id"].astype(str), expected_ids):
                raise DataValidationError(f"{split} mask bank changed manifest order")
            save_bank(staging / filename, payload)
            payloads[split] = payload
            summaries[split] = summary
        short_counts = {
            family: int(sum(value == family for value in short_family.values()))
            for family in FAMILIES
        }
        if max(short_counts.values()) - min(short_counts.values()) > 1:
            raise DataValidationError("Global two-mask family allocation is imbalanced")
        labels = manifests["candidate_train"]["y"].to_numpy(dtype=np.int64)
        audit_report, audit_arrays = auditor_runner(
            payloads["candidate_train"]["packed_masks"],
            payloads["candidate_train"]["family_id"],
            int(payloads["candidate_train"]["width"]),
            payloads["candidate_train"]["sample_id"],
            labels,
            seed=seed,
            config=config["auditor"],
        )
        _atomic_npz(staging / "auditor_split_and_predictions.npz", audit_arrays)
        write_json(staging / "leakage_audit.json", audit_report)
        write_json(
            staging / "mask_bank_summary.json",
            {
                "global_short_family_counts": short_counts,
                "splits": summaries,
            },
        )
        resolved = resolved_config(config)
        with (staging / "resolved_sanitized_mask_bank.yaml").open(
            "w", encoding="utf-8"
        ) as handle:
            yaml.safe_dump(resolved, handle, sort_keys=True)
        receipt = {
            "schema_version": 1,
            "status": "accepted" if audit_report["accepted"] else "rejected",
            "kind": "setv_sanitized_mask_bank",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "algorithm_version": config["mask_bank"]["algorithm_version"],
            "phase0_dir": str(phase0_dir),
            "phase0_artifact_manifest_sha256": sha256_file(
                phase0_dir / BASE_ARTIFACT_MANIFEST
            ),
            "phase0_visual_approval_sha256": sha256_file(
                phase0_dir / APPROVAL_RECEIPT
            ),
            "config_sha256": sha256_json(resolved),
            "banks": {
                split: {
                    "path": filename,
                    "sha256": sha256_file(staging / filename),
                }
                for split, filename in BANK_FILENAMES.items()
            },
            "leakage_audit": {
                "path": "leakage_audit.json",
                "sha256": sha256_file(staging / "leakage_audit.json"),
                "accepted": bool(audit_report["accepted"]),
            },
            "auditor_predictions_sha256": sha256_file(
                staging / "auditor_split_and_predictions.npz"
            ),
        }
        write_json(staging / "sanitized_mask_bank_receipt.json", receipt)
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
        raise


def verify_sanitized_mask_bank(
    root: str | Path,
    *,
    require_accepted: bool = True,
    verify_containment: bool = True,
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    manifest_path = root / "artifact_manifest.json"
    receipt_path = root / "sanitized_mask_bank_receipt.json"
    if not manifest_path.is_file() or not receipt_path.is_file():
        raise DataValidationError(f"Incomplete sanitized mask bank: {root}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    if sha256_json(artifact["files"]) != artifact["manifest_digest"]:
        raise DataValidationError("Sanitized bank manifest digest is invalid")
    for relative, expected in artifact["files"].items():
        item = root / relative
        if not item.is_file() or item.stat().st_size != expected["size_bytes"]:
            raise DataValidationError(f"Missing/changed sanitized artifact: {relative}")
        if sha256_file(item) != expected["sha256"]:
            raise DataValidationError(f"Changed sanitized artifact hash: {relative}")
    with receipt_path.open("r", encoding="utf-8") as handle:
        receipt = json.load(handle)
    if require_accepted and receipt["status"] != "accepted":
        raise DataValidationError("Sanitized mask bank failed the leakage gate")
    phase0_dir = Path(receipt["phase0_dir"])
    verify_phase0(phase0_dir, require_approval=True)
    if receipt["phase0_artifact_manifest_sha256"] != sha256_file(
        phase0_dir / BASE_ARTIFACT_MANIFEST
    ):
        raise DataValidationError("Sanitized mask bank Phase 0 binding changed")
    phase0_config = _load_phase0_config(phase0_dir)
    banks = {split: load_bank(root, split) for split in BANK_FILENAMES}
    all_ids = [
        sample_id
        for split in BANK_FILENAMES
        for sample_id in banks[split]["sample_id"].astype(str)
    ]
    allocation = globally_balanced_short_families(all_ids)
    for split, bank in banks.items():
        manifest = pd.read_csv(
            phase0_dir / "splits" / f"waterbirds95_{split}.csv",
            dtype={"sample_id": str},
        )
        if not np.array_equal(
            bank["sample_id"].astype(str),
            manifest["sample_id"].astype(str).to_numpy(),
        ):
            raise DataValidationError(f"{split} sanitized IDs do not align")
        for sample_id, row in zip(bank["sample_id"].astype(str), bank["family_id"]):
            short_id = FAMILIES.index(allocation[sample_id])
            if int((row == short_id).sum()) != 2:
                raise DataValidationError("Sanitized short-family assignment changed")
        if verify_containment:
            dataset = _dataset(phase0_dir, phase0_config, split)
            transform = build_eval_transform(phase0_config)
            for index in range(len(dataset)):
                example = dataset[index]
                _, source_mask = transform(example.image, example.mask)
                dilated = np.asarray(
                    dilate_binary_mask(
                        source_mask,
                        int(bank["dilation_radius"]),
                        structuring_element="euclidean_disk",
                    )
                ) > 0
                masks = unpack_masks(bank["packed_masks"][index], int(bank["width"]))
                if not np.all(masks[:, dilated]):
                    raise DataValidationError(
                        f"Sanitized containment failed for {split}:{example.sample_id}"
                    )
    return {
        "status": receipt["status"],
        "seed": receipt["seed"],
        "accepted": receipt["status"] == "accepted",
        "sample_counts": {split: len(bank["sample_id"]) for split, bank in banks.items()},
        "artifact_count": len(artifact["files"]),
        "containment_verified": verify_containment,
    }
