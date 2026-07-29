#!/usr/bin/env python3
"""Record hash-bound human approval of Phase 0 mask contact sheets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.phase0 import approve_visual_audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase0-dir", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    receipt = approve_visual_audit(
        args.phase0_dir, reviewer=args.reviewer, confirmation=args.confirm
    )
    print(json.dumps({"status": "approved", "receipt": str(receipt)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

