#!/usr/bin/env python3
"""Train and publish the Phase 1 object expert."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.experts.config import load_object_expert_config
from setv.experts.train_object import train_object_expert


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--phase0-dir")
    parser.add_argument("--output-root")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    args = parser.parse_args()
    config = load_object_expert_config(
        args.config,
        seed=args.seed,
        phase0_dir=args.phase0_dir,
        output_root=args.output_root,
        device=args.device,
    )
    destination = train_object_expert(config)
    print(
        json.dumps(
            {"status": "complete", "object_expert_dir": str(destination)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

