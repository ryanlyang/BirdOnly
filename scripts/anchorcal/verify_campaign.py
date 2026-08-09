#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from _common import DEFAULT_CONFIG, DEFAULT_PATHS, resolved_config
from anchorcal.campaign_verification import verify_campaign_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify immutable AnchorCal campaign artifacts.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--paths", default=str(DEFAULT_PATHS))
    args = parser.parse_args()
    config = resolved_config(args.config, args.paths)
    print(json.dumps(verify_campaign_artifacts(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
