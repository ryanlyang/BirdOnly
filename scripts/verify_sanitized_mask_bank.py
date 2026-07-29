#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.experts.sanitized_bank import verify_sanitized_mask_bank


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-bank-dir", required=True)
    parser.add_argument("--allow-rejected", action="store_true")
    parser.add_argument("--skip-containment", action="store_true")
    args = parser.parse_args()
    result = verify_sanitized_mask_bank(
        args.mask_bank_dir,
        require_accepted=not args.allow_rejected,
        verify_containment=not args.skip_containment,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

