#!/usr/bin/env python3
"""Standalone miniature end-to-end run isolated below outputs/.../debug."""

from __future__ import annotations

import argparse
import json

from _common import DEFAULT_CONFIG, DEFAULT_PATHS, resolved_config
from anchorcal.analysis import run_final_analysis
from anchorcal.anchor_pipeline import evaluate_anchor_ladder
from anchorcal.branch_pipeline import train_branch
from anchorcal.candidate_pipeline import train_candidate_run
from anchorcal.preflight import run_preflight
from anchorcal.prepare import prepare_geometry_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every AnchorCal stage with locked debug sizes.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--paths", default=str(DEFAULT_PATHS))
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Reuse a verified production preflight/cache while keeping all trained artifacts isolated.",
    )
    args = parser.parse_args()
    production_config = resolved_config(args.config, args.paths, debug=False)
    if not args.skip_preflight:
        run_preflight(
            production_config, allow_download=False, require_gh200=True
        )
    config = resolved_config(args.config, args.paths, debug=True)
    prepare_geometry_artifacts(config)
    stages = {
        "foreground": train_branch(config, "foreground"),
        "background": train_branch(config, "background"),
    }
    stages["anchors"] = evaluate_anchor_ladder(config)
    stages["candidate"] = train_candidate_run(
        config, learning_rate=3e-5, weight_decay=0.05
    )
    stages["analysis"] = run_final_analysis(config)
    print(json.dumps(stages, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
