#!/usr/bin/env python3
"""Validate the frozen campaign manifest or emit safe shell assignments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.campaign import campaign_environment, load_campaign_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "campaign_waterbirds95.yaml"),
    )
    parser.add_argument("--emit-env", action="store_true")
    args = parser.parse_args()
    manifest = load_campaign_manifest(args.config)
    if args.emit_env:
        for name, value in sorted(campaign_environment(manifest).items()):
            print(f"{name}={value}")
    else:
        print(
            json.dumps(
                {
                    "status": "valid",
                    "campaign_id": manifest["campaign_id"],
                    "manifest": manifest["_manifest_path"],
                    "environment": campaign_environment(manifest),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
