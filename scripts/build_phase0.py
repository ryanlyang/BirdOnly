#!/usr/bin/env python3
"""Build immutable Waterbirds95 Phase 0 artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.config import apply_overrides, load_config
from setv.phase0 import build_phase0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--mask-root")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = apply_overrides(
        load_config(args.config),
        dataset_root=args.dataset_root,
        mask_root=args.mask_root,
        output_dir=args.output_dir,
    )
    destination = build_phase0(config)
    print(
        json.dumps(
            {
                "status": "automated_checks_passed_visual_review_pending",
                "phase0_dir": str(destination),
                "next": (
                    f"Inspect {destination / 'mask_audit'} and run "
                    "scripts/approve_mask_audit.py with --confirm"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

