#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.fusion.sanitized_artifacts import score_candidate_sanitized_file
from setv.utils.io import write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion-dir", required=True)
    parser.add_argument("--candidate-npz", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    result = score_candidate_sanitized_file(args.fusion_dir, args.candidate_npz)
    if args.output_json:
        write_json(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
