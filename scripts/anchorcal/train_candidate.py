#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from _common import DEFAULT_CONFIG, DEFAULT_PATHS, resolved_config
from anchorcal.candidate_pipeline import train_candidate_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate one restartable candidate-grid run.")
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--paths", default=str(DEFAULT_PATHS))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    config = resolved_config(args.config, args.paths, debug=args.debug)
    result = train_candidate_run(
        config, learning_rate=args.learning_rate, weight_decay=args.weight_decay
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

