#!/usr/bin/env python3
"""Write a machine-readable uLA environment compatibility receipt."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.ula.compatibility import probe_ula_environment
from setv.utils.io import write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("official_ssl", "external_checkpoint"),
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--expected-architecture", default="aarch64")
    args = parser.parse_args()
    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = probe_ula_environment(
            args.official_repo,
            mode=args.mode,
            require_gpu=args.require_gpu,
            expected_architecture=args.expected_architecture,
        )
    except Exception as exc:
        report = {
            "schema_version": 1,
            "status": "probe_failed",
            "accepted": False,
            "mode": args.mode,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("accepted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
