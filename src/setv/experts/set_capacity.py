"""Read-only Stage 4 background-token capacity census."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from setv.data.dataset import WaterbirdsManifestDataset
from setv.data.joint_transforms import (
    build_eval_transform,
    build_full_frame_background_transform,
)
from setv.errors import ArtifactExistsError, DataValidationError
from setv.experts.set_data import background_patch_capacity
from setv.phase0 import APPROVAL_RECEIPT, BASE_ARTIFACT_MANIFEST, verify_phase0
from setv.utils.hashing import sha256_file, sha256_json
from setv.utils.io import write_json


def _dataset(
    manifest: Path,
    phase0_config: dict[str, Any],
) -> WaterbirdsManifestDataset:
    masks = phase0_config["masks"]
    return WaterbirdsManifestDataset(
        manifest,
        phase0_config["data"]["dataset_root"],
        masks["root"],
        threshold_normalized=float(masks["threshold_normalized"]),
        foreground_is_high=bool(masks["foreground_is_high"]),
        map_format=str(masks.get("format", "threshold")),
        foreground_class_ids=tuple(
            int(value) for value in masks.get("foreground_class_ids", [1])
        ),
    )


def _distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        raise DataValidationError("Capacity census cannot summarize an empty split")
    array = np.asarray(values, dtype=np.int64)
    quantiles = {
        name: int(np.quantile(array, probability, method="nearest"))
        for name, probability in (
            ("q01", 0.01),
            ("q05", 0.05),
            ("q10", 0.10),
            ("q25", 0.25),
            ("median", 0.50),
            ("q75", 0.75),
            ("q90", 0.90),
            ("q95", 0.95),
            ("q99", 0.99),
        )
    }
    return {
        "sample_count": int(array.size),
        "minimum": int(array.min()),
        **quantiles,
        "maximum": int(array.max()),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std()),
    }


def _support_by_floor(
    capacities: list[int],
    maximum_floor: int,
) -> dict[str, dict[str, Any]]:
    total = len(capacities)
    return {
        str(floor): {
            "supported_sample_count": int(
                sum(capacity >= floor for capacity in capacities)
            ),
            "unsupported_sample_count": int(
                sum(capacity < floor for capacity in capacities)
            ),
            "supported_fraction": float(
                sum(capacity >= floor for capacity in capacities) / total
            ),
        }
        for floor in range(1, maximum_floor + 1)
    }


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False, lineterminator="\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def run_set_capacity_census(
    config: dict[str, Any],
    json_report: str | Path,
    csv_report: str | Path,
) -> dict[str, Any]:
    """Measure capacity under both fixed transforms without training a model."""

    json_path = Path(json_report).expanduser().resolve()
    csv_path = Path(csv_report).expanduser().resolve()
    if json_path.exists() or csv_path.exists():
        raise ArtifactExistsError(
            "Capacity census refuses to overwrite an existing report"
        )
    if json_path == csv_path:
        raise DataValidationError("JSON and CSV census paths must differ")

    phase0_dir = Path(config["phase0_dir"]).expanduser().resolve()
    verify_phase0(phase0_dir, require_approval=True)
    with (phase0_dir / "config" / "resolved_phase0.yaml").open(
        encoding="utf-8"
    ) as handle:
        phase0_config = yaml.safe_load(handle)
    image_size = int(phase0_config["transforms"]["image_size"])
    patch_size = int(config["input"]["patch_size"])
    if image_size != 224 or image_size % patch_size:
        raise DataValidationError(
            "Production capacity census requires a 224px patch-divisible view"
        )

    canonical_transform = build_eval_transform(phase0_config)
    full_frame_transform = build_full_frame_background_transform(
        phase0_config
    )
    rows: list[dict[str, Any]] = []
    processed = 0
    for split_name in ("candidate_train", "biased_val"):
        manifest = (
            phase0_dir
            / "splits"
            / f"waterbirds95_{split_name}.csv"
        )
        dataset = _dataset(manifest, phase0_config)
        for index in range(len(dataset)):
            example = dataset[index]
            row = dataset.rows.iloc[index]
            _, canonical_mask = canonical_transform(
                example.image, example.mask
            )
            _, full_frame_mask = full_frame_transform(
                example.image, example.mask
            )
            canonical = background_patch_capacity(
                canonical_mask, config
            )
            full_frame = background_patch_capacity(
                full_frame_mask, config
            )
            canonical_count = int(canonical["eligible_count"])
            full_frame_count = int(full_frame["eligible_count"])
            rows.append(
                {
                    "sample_id": str(example.sample_id),
                    "split_name": split_name,
                    "target": int(example.target),
                    "img_filename": str(row["img_filename"]),
                    "mask_relative_path": str(row["mask_relative_path"]),
                    "source_width": int(example.image.width),
                    "source_height": int(example.image.height),
                    "canonical_capacity": canonical_count,
                    "full_frame_capacity": full_frame_count,
                    "best_fixed_view_capacity": max(
                        canonical_count, full_frame_count
                    ),
                    "best_fixed_view": (
                        "canonical"
                        if canonical_count >= full_frame_count
                        else "full_frame"
                    ),
                }
            )
            processed += 1
            if processed % 500 == 0:
                print(
                    "capacity_census_progress "
                    f"processed={processed} split={split_name}",
                    flush=True,
                )

    frame = pd.DataFrame(rows).sort_values(
        ["split_name", "sample_id"], kind="stable"
    )
    locked_floor = int(config["input"]["min_background_tokens"])
    combined_best = [
        int(value) for value in frame["best_fixed_view_capacity"].tolist()
    ]
    split_summaries: dict[str, Any] = {}
    for split_name, split_frame in frame.groupby(
        "split_name", sort=True
    ):
        canonical_values = [
            int(value) for value in split_frame["canonical_capacity"]
        ]
        full_frame_values = [
            int(value) for value in split_frame["full_frame_capacity"]
        ]
        best_values = [
            int(value) for value in split_frame["best_fixed_view_capacity"]
        ]
        failures = split_frame[
            split_frame["best_fixed_view_capacity"] < locked_floor
        ]
        split_summaries[str(split_name)] = {
            "canonical_capacity": _distribution(canonical_values),
            "full_frame_capacity": _distribution(full_frame_values),
            "best_fixed_view_capacity": _distribution(best_values),
            "largest_universally_supported_floor": int(min(best_values)),
            "locked_floor_unsupported_sample_count": int(len(failures)),
            "locked_floor_unsupported_sample_ids": [
                str(value) for value in failures["sample_id"].tolist()
            ],
        }

    combined_failures = frame[
        frame["best_fixed_view_capacity"] < locked_floor
    ]
    report = {
        "schema_version": 1,
        "audit_type": "stage4_background_patch_capacity_census",
        "status": "complete",
        "diagnostic_only": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "information_boundary": {
            "protected_group_columns_loaded": False,
            "oracle_or_test_split_loaded": False,
            "splits": ["candidate_train", "biased_val"],
        },
        "locked_patch_policy": {
            "image_size": image_size,
            "patch_size": patch_size,
            "dilation_pixels_at_224": int(
                config["input"]["dilation_pixels_at_224"]
            ),
            "dilation_structuring_element": config["input"][
                "dilation_structuring_element"
            ],
            "maximum_foreground_fraction": float(
                config["input"]["maximum_foreground_fraction"]
            ),
            "configured_minimum_background_tokens": locked_floor,
            "transforms_measured": [
                "canonical_evaluation_center_crop",
                "full_frame_aspect_fit_excluding_padding",
            ],
        },
        "split_summaries": split_summaries,
        "combined": {
            "sample_count": int(len(frame)),
            "best_fixed_view_capacity": _distribution(combined_best),
            "largest_universally_supported_floor": int(min(combined_best)),
            "configured_floor_is_universally_feasible": bool(
                min(combined_best) >= locked_floor
            ),
            "configured_floor_unsupported_sample_count": int(
                len(combined_failures)
            ),
            "configured_floor_unsupported_sample_ids": [
                str(value)
                for value in combined_failures["sample_id"].tolist()
            ],
            "support_by_floor": _support_by_floor(
                combined_best, locked_floor
            ),
        },
        "provenance": {
            "phase0_dir": str(phase0_dir),
            "phase0_artifact_manifest_sha256": sha256_file(
                phase0_dir / BASE_ARTIFACT_MANIFEST
            ),
            "phase0_visual_approval_sha256": sha256_file(
                phase0_dir / APPROVAL_RECEIPT
            ),
            "resolved_expert_config_sha256": sha256_json(config),
            "csv_report": str(csv_path),
        },
    }
    _write_csv_atomic(csv_path, frame)
    report["provenance"]["csv_report_sha256"] = sha256_file(csv_path)
    write_json(json_path, report)
    return report
