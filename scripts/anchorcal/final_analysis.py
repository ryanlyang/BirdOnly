#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from _common import DEFAULT_CONFIG, DEFAULT_PATHS, resolved_config
from anchorcal.analysis import run_final_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze selection, then join reporting-only metrics.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--paths", default=str(DEFAULT_PATHS))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    config = resolved_config(args.config, args.paths, debug=args.debug)
    print(json.dumps(run_final_analysis(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

