#!/usr/bin/env python3
"""Audit complete metadata-to-VLM-mask coverage before Phase 0 construction."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.config import load_config
from setv.data.audit import render_preflight_galleries
from setv.data.masks import MaskResolver, inspect_mask
from setv.data.splits import build_splits, read_and_validate_metadata
from setv.errors import DataValidationError
from setv.utils.hashing import sha256_file
from setv.utils.io import write_json


def _gallery_samples(
    metadata,
    config: dict,
    resolved_by_id: dict[str, str],
    *,
    samples_per_split: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    split_result = build_splits(metadata, config)
    rng = np.random.default_rng(int(config["audit"]["visual_seed"]))
    selections = []
    available_counts = {}
    for split_name in ("candidate_train", "biased_val", "oracle_val"):
        frame = split_result.safe_manifests[split_name].copy()
        frame["mask_relative_path"] = frame["sample_id"].map(resolved_by_id)
        frame = frame[frame["mask_relative_path"].notna()].reset_index(drop=True)
        available_counts[split_name] = int(len(frame))
        count = min(samples_per_split, len(frame))
        if count == 0:
            continue
        chosen = np.sort(rng.choice(len(frame), size=count, replace=False))
        selections.append(frame.iloc[chosen].copy())
    if not selections:
        return pd.DataFrame(), available_counts
    return pd.concat(selections, ignore_index=True), available_counts


def audit_mapping(
    config_path: str | Path,
    *,
    gallery_dir: str | Path | None = None,
    samples_per_split: int = 24,
    samples_per_page: int = 8,
    thumbnail_size: int = 160,
) -> dict:
    config = load_config(config_path)
    metadata = read_and_validate_metadata(config)
    mask_config = config["masks"]
    resolver = MaskResolver(
        mask_config["root"],
        mask_config["allowed_extensions"],
        mask_config["mapping_mode"],
    )
    mapped_by_split: Counter[int] = Counter()
    failures_by_split: Counter[int] = Counter()
    rules: Counter[str] = Counter()
    path_to_samples: dict[str, list[str]] = defaultdict(list)
    resolved_by_id: dict[str, str] = {}
    failures = []
    optional_missing = []
    required_official_splits = {
        int(value) for value in mask_config["required_official_splits"]
    }
    dataset_root = Path(config["data"]["dataset_root"]).expanduser().resolve()
    for row in metadata.itertuples(index=False):
        required = int(row.official_split) in required_official_splits
        try:
            resolved = resolver.resolve(row.img_filename, row.sample_id)
        except DataValidationError as exc:
            failures_by_split[int(row.official_split)] += 1
            record = (
                {
                    "sample_id": str(row.sample_id),
                    "metadata_index": int(row.metadata_index),
                    "img_filename": str(row.img_filename),
                    "official_split": int(row.official_split),
                    "error": str(exc),
                }
            )
            if required:
                failures.append(record)
            elif len(optional_missing) < 20:
                optional_missing.append(record)
            continue
        if required:
            try:
                inspect_mask(
                    dataset_root / row.img_filename,
                    resolved.path,
                    threshold_normalized=float(
                        mask_config["threshold_normalized"]
                    ),
                    foreground_is_high=bool(
                        mask_config["foreground_is_high"]
                    ),
                    require_same_dimensions=bool(
                        mask_config["require_same_dimensions"]
                    ),
                    minimum_foreground_fraction=float(
                        mask_config["minimum_foreground_fraction"]
                    ),
                    maximum_foreground_fraction=float(
                        mask_config["maximum_foreground_fraction"]
                    ),
                    map_format=str(mask_config.get("format", "threshold")),
                    foreground_class_ids=tuple(
                        int(value)
                        for value in mask_config.get(
                            "foreground_class_ids", [1]
                        )
                    ),
                )
            except DataValidationError as exc:
                failures_by_split[int(row.official_split)] += 1
                failures.append(
                    {
                        "sample_id": str(row.sample_id),
                        "metadata_index": int(row.metadata_index),
                        "img_filename": str(row.img_filename),
                        "official_split": int(row.official_split),
                        "error": str(exc),
                    }
                )
                continue
        mapped_by_split[int(row.official_split)] += 1
        rules[resolved.mapping_rule] += 1
        path_to_samples[resolved.relative_path].append(str(row.sample_id))
        resolved_by_id[str(row.sample_id)] = resolved.relative_path

    collisions = [
        {"mask_relative_path": path, "sample_ids": sample_ids}
        for path, sample_ids in sorted(path_to_samples.items())
        if len(sample_ids) != 1
    ]
    gallery_receipts = []
    gallery_available_counts = {}
    if gallery_dir is not None:
        samples, gallery_available_counts = _gallery_samples(
            metadata,
            config,
            resolved_by_id,
            samples_per_split=samples_per_split,
        )
        if len(samples):
            gallery_receipts = render_preflight_galleries(
                samples,
                config=config,
                output_dir=gallery_dir,
                samples_per_page=samples_per_page,
                thumbnail_size=thumbnail_size,
            )
    required_rows = int(
        metadata["official_split"].isin(required_official_splits).sum()
    )
    required_mapped = sum(
        count
        for split, count in mapped_by_split.items()
        if split in required_official_splits
    )
    complete = not failures and not collisions and required_mapped == required_rows
    return {
        "schema_version": 1,
        "status": "complete" if complete else "incomplete",
        "accepted": complete,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "path": str(Path(config_path).expanduser().resolve()),
            "sha256": sha256_file(Path(config_path).expanduser().resolve()),
        },
        "dataset_root": str(Path(config["data"]["dataset_root"]).resolve()),
        "mask_root": str(Path(mask_config["root"]).resolve()),
        "mapping_mode": mask_config["mapping_mode"],
        "metadata_row_count": int(len(metadata)),
        "required_official_splits": sorted(required_official_splits),
        "required_row_count": required_rows,
        "required_mapped_row_count": int(required_mapped),
        "optional_test_masks_required": False,
        "indexed_mask_file_count": resolver.total_mask_files,
        "mapped_row_count": int(sum(mapped_by_split.values())),
        "mapped_by_official_split": {
            str(key): int(value) for key, value in sorted(mapped_by_split.items())
        },
        "failures_by_official_split": {
            str(key): int(value)
            for key, value in sorted(failures_by_split.items())
        },
        "mapping_rule_counts": {
            key: int(value) for key, value in sorted(rules.items())
        },
        "missing_or_ambiguous_count": len(failures),
        "missing_or_ambiguous_samples": failures,
        "optional_missing_count": int(
            failures_by_split[
                int(mask_config["optional_official_splits"][0])
            ]
        ),
        "optional_missing_examples": optional_missing,
        "mask_reuse_collision_count": len(collisions),
        "mask_reuse_collisions": collisions,
        "visual_gallery": {
            "directory": (
                str(Path(gallery_dir).expanduser().resolve())
                if gallery_dir is not None
                else None
            ),
            "seed": int(config["audit"]["visual_seed"]),
            "requested_samples_per_split": samples_per_split,
            "available_mapped_samples_by_split": gallery_available_counts,
            "samples_per_page": samples_per_page,
            "thumbnail_size": thumbnail_size,
            "page_count": len(gallery_receipts),
            "pages": gallery_receipts,
        },
        "reporting_only_metric_values_included": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--gallery-dir")
    parser.add_argument("--samples-per-split", type=int, default=24)
    parser.add_argument("--samples-per-page", type=int, default=8)
    parser.add_argument("--thumbnail-size", type=int, default=160)
    args = parser.parse_args()
    destination = Path(args.report).expanduser().resolve()
    if destination.exists():
        print(f"Refusing to overwrite mask-mapping audit: {destination}", file=sys.stderr)
        return 2
    gallery_dir = (
        Path(args.gallery_dir).expanduser().resolve()
        if args.gallery_dir
        else destination.with_suffix("").with_name(destination.stem + "_galleries")
    )
    if gallery_dir.exists():
        print(f"Refusing to overwrite mask galleries: {gallery_dir}", file=sys.stderr)
        return 2
    report = audit_mapping(
        args.config,
        gallery_dir=gallery_dir,
        samples_per_split=args.samples_per_split,
        samples_per_page=args.samples_per_page,
        thumbnail_size=args.thumbnail_size,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, report)
    print(
        f"mask_mapping_status={report['status']} "
        f"required_mapped={report['required_mapped_row_count']}/"
        f"{report['required_row_count']} "
        f"optional_test_missing={report['optional_missing_count']} "
        f"gallery_pages={report['visual_gallery']['page_count']} "
        f"report={destination}"
    )
    if not report["accepted"]:
        print(
            "Required train/validation VLM-mask mapping or decoding is "
            "incomplete; inspect the JSON report and do not build Phase 0.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
