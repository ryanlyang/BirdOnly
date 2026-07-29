#!/usr/bin/env python3
"""Bind uLA environment and proxy smoke evidence into one acceptance receipt."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from setv.ula.provenance import audit_official_source
from setv.utils.hashing import sha256_file
from setv.utils.io import write_json


def _json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=("official_ssl", "external_checkpoint"),
    )
    parser.add_argument("--compatibility-report", required=True)
    parser.add_argument("--proxy-smoke-report", required=True)
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--ssl-checkpoint", required=True)
    parser.add_argument("--setv-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    compatibility_path = Path(args.compatibility_report).resolve()
    proxy_path = Path(args.proxy_smoke_report).resolve()
    checkpoint = Path(args.ssl_checkpoint).resolve()
    compatibility = _json(compatibility_path)
    proxy = _json(proxy_path)
    if not compatibility.get("accepted"):
        raise RuntimeError("uLA compatibility probe was not accepted")
    if compatibility.get("mode") != args.mode:
        raise RuntimeError("uLA compatibility report mode changed")
    if not proxy.get("accepted"):
        raise RuntimeError("uLA proxy checkpoint smoke was not accepted")
    if proxy.get("kind") != "setv_ula_proxy_checkpoint_smoke":
        raise RuntimeError("Wrong uLA proxy smoke artifact kind")
    checkpoint_hash = sha256_file(checkpoint)
    if proxy["ssl_checkpoint"]["sha256"] != checkpoint_hash:
        raise RuntimeError("Proxy smoke is not bound to the selected checkpoint")
    official = audit_official_source(args.official_repo)
    exact = bool(compatibility["official_requirements_exact_match"])
    if args.mode == "official_ssl":
        reproduction = (
            "official_source_and_exact_pinned_environment"
            if exact
            else "official_source_with_recorded_dependency_version_deviation"
        )
    else:
        reproduction = "verified_external_official_ssl_checkpoint_adapter"
    receipt = {
        "schema_version": 1,
        "status": "accepted",
        "accepted": True,
        "kind": "setv_ula_tigris_smoke",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "reproduction_status": reproduction,
        "setv_commit": args.setv_commit,
        "official_source": official,
        "official_requirements_exact_match": exact,
        "compatibility_report": {
            "path": str(compatibility_path),
            "sha256": sha256_file(compatibility_path),
        },
        "proxy_smoke_report": {
            "path": str(proxy_path),
            "sha256": sha256_file(proxy_path),
        },
        "ssl_checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_hash,
        },
        "information_boundary": proxy["information_boundary"],
        "campaign_gate": (
            "Production Phase 6 jobs may run only through an afterok dependency "
            "on the Slurm job that produced this receipt."
        ),
        "method_label": "uLA-style",
    }
    write_json(Path(args.output).resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
