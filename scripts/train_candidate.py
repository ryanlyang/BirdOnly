#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.candidate.config import load_candidate_config
from setv.candidate.train import train_candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--exact-fusion-dir", required=True)
    parser.add_argument("--sanitized-fusion-dir", required=True)
    parser.add_argument("--set-fusion-dir", required=True)
    parser.add_argument("--phase0-dir")
    parser.add_argument("--output-root")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    args = parser.parse_args()
    config = load_candidate_config(
        args.config,
        seed=args.seed,
        phase0_dir=args.phase0_dir,
        exact_fusion_dir=args.exact_fusion_dir,
        sanitized_fusion_dir=args.sanitized_fusion_dir,
        set_fusion_dir=args.set_fusion_dir,
        output_root=args.output_root,
        device=args.device,
    )
    destination = train_candidate(config)
    print(json.dumps({"status": "complete", "candidate_dir": str(destination)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
