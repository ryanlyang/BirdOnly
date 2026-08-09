#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from _common import DEFAULT_CONFIG, DEFAULT_PATHS, resolved_config
from anchorcal.pretrained import resolve_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate and verify the one pinned HF snapshot.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--paths", default=str(DEFAULT_PATHS))
    args = parser.parse_args()
    config = resolved_config(args.config, args.paths)
    print(json.dumps(resolve_snapshot(config["paths"]["hf_home"], allow_download=True), indent=2))


if __name__ == "__main__":
    main()

