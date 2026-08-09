#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from _common import DEFAULT_CONFIG, DEFAULT_PATHS, resolved_config
from anchorcal.preflight import run_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fail-closed AnchorCal preflight.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--paths", default=str(DEFAULT_PATHS))
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--require-gh200", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    config = resolved_config(args.config, args.paths, debug=args.debug)
    report = run_preflight(
        config,
        allow_download=args.allow_download,
        require_gh200=args.require_gh200,
    )
    print(json.dumps({"status": report["status"], "output": config["paths"]["output_root"]}))


if __name__ == "__main__":
    main()

