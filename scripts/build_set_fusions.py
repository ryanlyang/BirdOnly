#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.fusion.set_artifacts import build_set_fusion_artifacts
from setv.fusion.set_config import load_set_fusion_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--object-expert-dir", required=True)
    parser.add_argument("--set-expert-dir", required=True)
    parser.add_argument("--phase0-dir")
    parser.add_argument("--output-root")
    parser.add_argument("--allow-expert-sanity-warnings", action="store_true")
    args = parser.parse_args()
    config = load_set_fusion_config(
        args.config,
        seed=args.seed,
        phase0_dir=args.phase0_dir,
        object_expert_dir=args.object_expert_dir,
        set_expert_dir=args.set_expert_dir,
        output_root=args.output_root,
        allow_expert_sanity_warnings=args.allow_expert_sanity_warnings,
    )
    destination = build_set_fusion_artifacts(config)
    print(json.dumps({"status": "complete", "fusion_dir": str(destination)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
