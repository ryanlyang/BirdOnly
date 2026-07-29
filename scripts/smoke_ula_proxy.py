#!/usr/bin/env python3
"""Run one real uLA proxy update against an official SSL checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.ula.config import load_ula_proxy_config
from setv.ula.proxy import run_ula_proxy_smoke
from setv.utils.io import write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--ssl-checkpoint", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args()
    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        config = load_ula_proxy_config(
            args.config,
            official_repo=args.official_repo,
            ssl_checkpoint=args.ssl_checkpoint,
            seed=args.seed,
            device=args.device,
        )
        report = run_ula_proxy_smoke(config, report_path)
    except Exception as exc:
        report = {
            "schema_version": 1,
            "status": "rejected",
            "accepted": False,
            "kind": "setv_ula_proxy_checkpoint_smoke",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
        write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
