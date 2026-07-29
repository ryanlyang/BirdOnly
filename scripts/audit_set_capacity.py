#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.experts.set_capacity import run_set_capacity_census
from setv.experts.set_config import load_set_expert_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure Stage 4 background-patch capacity without training"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--phase0-dir")
    parser.add_argument("--json-report", required=True)
    parser.add_argument("--csv-report", required=True)
    args = parser.parse_args()
    config = load_set_expert_config(
        args.config,
        seed=args.seed,
        phase0_dir=args.phase0_dir,
        device="cpu",
    )
    report = run_set_capacity_census(
        config, args.json_report, args.csv_report
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
