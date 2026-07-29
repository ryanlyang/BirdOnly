#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.ula.config import load_ula_proxy_config
from setv.ula.proxy import train_ula_proxy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--ssl-checkpoint", required=True)
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--phase0-dir")
    parser.add_argument("--output-root")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    config = load_ula_proxy_config(
        args.config,
        phase0_dir=args.phase0_dir,
        official_repo=args.official_repo,
        ssl_checkpoint=args.ssl_checkpoint,
        output_root=args.output_root,
        seed=args.seed,
        device=args.device,
    )
    destination = train_ula_proxy(config)
    print(json.dumps({"status": "complete", "ula_proxy_dir": str(destination)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
