#!/usr/bin/env python3
"""Verify Phase 0 artifact hashes and human visual approval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.phase0 import verify_phase0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase0-dir", required=True)
    parser.add_argument(
        "--allow-pending-visual-review",
        action="store_true",
        help="Verify automated artifacts without requiring human approval.",
    )
    args = parser.parse_args()
    result = verify_phase0(
        args.phase0_dir, require_approval=not args.allow_pending_visual_review
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

