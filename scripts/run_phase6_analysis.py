#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.ula.analysis import build_phase6_analysis
from setv.ula.config import load_phase6_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate-seeds", required=True)
    parser.add_argument("--ula-proxy-dir", required=True)
    parser.add_argument("--exact-fusion-dir", required=True)
    parser.add_argument("--sanitized-fusion-dir", required=True)
    parser.add_argument("--set-fusion-dir", required=True)
    parser.add_argument("--phase0-dir")
    parser.add_argument("--candidate-root")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    seeds = [int(value) for value in args.candidate_seeds.split(",") if value]
    config = load_phase6_config(
        args.config,
        phase0_dir=args.phase0_dir,
        candidate_root=args.candidate_root,
        ula_proxy_dir=args.ula_proxy_dir,
        exact_fusion_dir=args.exact_fusion_dir,
        sanitized_fusion_dir=args.sanitized_fusion_dir,
        set_fusion_dir=args.set_fusion_dir,
        output_dir=args.output_dir,
        candidate_seeds=seeds,
    )
    destination = build_phase6_analysis(config)
    print(json.dumps({"status": "complete", "phase6_dir": str(destination)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
