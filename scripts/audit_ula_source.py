#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.ula.provenance import audit_official_source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", required=True)
    args = parser.parse_args()
    print(json.dumps(audit_official_source(args.official_repo), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
