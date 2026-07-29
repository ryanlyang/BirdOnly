#!/usr/bin/env python3
"""Explicitly reveal the reporting-only test curve after verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.candidate.train import verify_candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.output_dir).expanduser().resolve()
    verify_candidate(root, load_checkpoints=False)
    report = json.loads((root / "reporting_only" / "test_metrics.json").read_text())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
