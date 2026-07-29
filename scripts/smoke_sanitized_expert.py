#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.experts.sanitized_config import load_sanitized_expert_config
from setv.experts.train_sanitized import run_sanitized_expert_smoke


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--mask-bank-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--phase0-dir")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    args = parser.parse_args()
    config = load_sanitized_expert_config(
        args.config,
        seed=args.seed,
        phase0_dir=args.phase0_dir,
        mask_bank_dir=args.mask_bank_dir,
        device=args.device,
    )
    print(json.dumps(run_sanitized_expert_smoke(config, args.report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

