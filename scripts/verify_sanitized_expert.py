#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.experts.train_sanitized import verify_sanitized_expert


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-checkpoint-load", action="store_true")
    args = parser.parse_args()
    result = verify_sanitized_expert(
        args.output_dir, load_checkpoint=not args.skip_checkpoint_load
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

