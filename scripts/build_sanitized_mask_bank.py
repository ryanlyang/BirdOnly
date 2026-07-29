#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.experts.sanitized_bank import build_sanitized_mask_bank
from setv.experts.sanitized_config import load_sanitized_bank_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--phase0-dir")
    parser.add_argument("--output-root")
    parser.add_argument("--auditor-device", choices=("cuda", "cpu"))
    args = parser.parse_args()
    config = load_sanitized_bank_config(
        args.config,
        seed=args.seed,
        phase0_dir=args.phase0_dir,
        output_root=args.output_root,
        auditor_device=args.auditor_device,
    )
    destination = build_sanitized_mask_bank(config)
    with (destination / "sanitized_mask_bank_receipt.json").open(
        "r", encoding="utf-8"
    ) as handle:
        receipt = json.load(handle)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "mask_bank_dir": str(destination),
                "leakage_audit_accepted": receipt["leakage_audit"]["accepted"],
            },
            indent=2,
        )
    )
    return 0 if receipt["status"] == "accepted" else 3


if __name__ == "__main__":
    raise SystemExit(main())

