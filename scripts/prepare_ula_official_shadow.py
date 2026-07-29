#!/usr/bin/env python3
"""Build protected-label-free metadata for official uLA SSL pretraining."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.phase0 import verify_phase0
from setv.utils.hashing import sha256_file
from setv.utils.io import write_json


def _smoke_subset(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if maximum < 2 or maximum % 2:
        raise ValueError(
            "Smoke shadow requires an even maximum of at least two per split"
        )
    selected = []
    per_class = maximum // 2
    for class_id in (0, 1):
        rows = frame[frame["y"] == class_id].sort_values(
            "sample_id", kind="stable"
        )
        if len(rows) < per_class:
            raise ValueError(
                f"Smoke shadow lacks {per_class} class-{class_id} samples"
            )
        selected.append(rows.iloc[:per_class])
    result = pd.concat(selected, ignore_index=True)
    return result.sort_values("sample_id", kind="stable").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase0-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke-samples-per-split", type=int)
    args = parser.parse_args()
    phase0 = Path(args.phase0_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    verify_phase0(phase0, require_approval=True)
    if output.exists():
        raise FileExistsError(f"Shadow dataset already exists: {output}")
    output.mkdir(parents=True)
    with (phase0 / "config" / "resolved_phase0.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        phase0_config = yaml.safe_load(handle)
    dataset_root = Path(phase0_config["data"]["dataset_root"])
    frames = []
    for split_name, split_value in (("candidate_train", 0), ("biased_val", 1)):
        frame = pd.read_csv(
            phase0 / "splits" / f"waterbirds95_{split_name}.csv",
            dtype={"sample_id": str},
        )
        if args.smoke_samples_per_split is not None:
            frame = _smoke_subset(frame, args.smoke_samples_per_split)
        frame = frame[["sample_id", "img_filename", "y"]].copy()
        frame["img_filename"] = frame["img_filename"].map(
            lambda value: str((dataset_root / str(value)).resolve())
        )
        frame["place"] = 0
        frame["split"] = split_value
        frames.append(frame)
    # Official Waterbirds code unconditionally instantiates a test loader.
    # Duplicate biased_val structurally; no test or oracle sample is exposed.
    duplicate = frames[1].copy()
    duplicate["sample_id"] = duplicate["sample_id"].map(lambda value: f"dup-{value}")
    duplicate["split"] = 2
    frames.append(duplicate)
    metadata = pd.concat(frames, ignore_index=True)
    metadata.to_csv(output / "metadata.csv", index=False, lineterminator="\n")
    receipt = {
        "status": "complete",
        "purpose": "official_uLA_SSL_pretraining_adapter",
        "protected_place_labels_redacted_to_constant": True,
        "oracle_samples_present": False,
        "test_samples_present": False,
        "official_test_loader_uses_duplicated_biased_val": True,
        "candidate_train_count": len(frames[0]),
        "biased_val_count": len(frames[1]),
        "smoke_subset": args.smoke_samples_per_split is not None,
        "smoke_samples_per_split_requested": args.smoke_samples_per_split,
        "smoke_sample_ids": (
            {
                "candidate_train": frames[0]["sample_id"].astype(str).tolist(),
                "biased_val": frames[1]["sample_id"].astype(str).tolist(),
            }
            if args.smoke_samples_per_split is not None
            else None
        ),
        "metadata_sha256": sha256_file(output / "metadata.csv"),
    }
    write_json(output / "shadow_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
