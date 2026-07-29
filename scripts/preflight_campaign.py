#!/usr/bin/env python3
"""Run the fail-closed, reporting-safe campaign stage preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.campaign import load_campaign_manifest, run_campaign_preflight
from setv.utils.io import write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "campaign_waterbirds95.yaml"),
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=tuple(f"phase{index}" for index in range(7)),
    )
    parser.add_argument("--repository", default=str(ROOT))
    parser.add_argument("--report", required=True)
    parser.add_argument("--skip-tigris-filesystem", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument(
        "--resume-rejected-sanitized",
        action="store_true",
        help="Validate the locked diagnostic resume from an existing rejected bank.",
    )
    args = parser.parse_args()
    manifest = load_campaign_manifest(args.config)
    report = run_campaign_preflight(
        manifest,
        stage=args.stage,
        repository=args.repository,
        check_tigris_filesystem=not args.skip_tigris_filesystem,
        resume_rejected_sanitized=args.resume_rejected_sanitized,
    )
    destination = Path(args.report).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not args.status_only:
        print(f"Refusing to overwrite preflight receipt: {destination}", file=sys.stderr)
        return 2
    write_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.status_only:
        return 0
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
