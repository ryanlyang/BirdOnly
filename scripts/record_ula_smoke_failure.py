#!/usr/bin/env python3
"""Persist an actionable failure receipt for a rejected uLA Slurm smoke."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.utils.hashing import sha256_file
from setv.utils.io import write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--line", required=True, type=int)
    parser.add_argument("--compatibility-report")
    parser.add_argument("--proxy-report")
    parser.add_argument("--slurm-job-id")
    args = parser.parse_args()
    evidence = {}
    for name, value in (
        ("compatibility", args.compatibility_report),
        ("proxy_smoke", args.proxy_report),
    ):
        if value and Path(value).is_file():
            evidence[name] = {
                "path": str(Path(value).resolve()),
                "sha256": sha256_file(value),
            }
    report = {
        "schema_version": 1,
        "status": "rejected",
        "accepted": False,
        "kind": "setv_ula_tigris_smoke_failure",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "exit_code": args.exit_code,
        "failed_shell_line": args.line,
        "slurm_job_id": args.slurm_job_id,
        "available_evidence": evidence,
        "next_action": (
            "Inspect compatibility.json and the Slurm error log. Do not "
            "continue the campaign or silently replace the SSL encoder."
        ),
    }
    write_json(Path(args.output).resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
