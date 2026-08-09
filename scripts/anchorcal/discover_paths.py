#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from anchorcal.paths import discover_candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print AnchorCal path candidates without choosing among ambiguities."
    )
    parser.add_argument("roots", nargs="+", help="Accessible roots to search")
    args = parser.parse_args()
    print(json.dumps(discover_candidates(args.roots), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

