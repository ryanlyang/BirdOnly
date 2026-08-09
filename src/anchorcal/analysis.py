"""Two-stage final analysis orchestration with a frozen selection boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import PreflightError
from .provenance import verify_hashed_receipt
from .selector_analysis import run_selector_stage


def _existing_selection_receipt(config: dict[str, Any]) -> Path | None:
    output = Path(config["paths"]["output_root"])
    debug = bool(config["runtime"]["debug"])
    receipt_root = output / ("debug/receipt" if debug else "receipt")
    receipts = sorted(receipt_root.glob("candidate_selection_*.json"))
    if not receipts:
        return None
    if len(receipts) != 1 or not verify_hashed_receipt(receipts[0]):
        raise PreflightError(
            "final-analysis restart requires zero or one valid candidate-selection receipt"
        )
    try:
        payload = json.loads(receipts[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError("existing candidate-selection receipt is invalid") from error
    if (
        payload.get("schema_version") != "anchorcal-candidate-selection-v2"
        or payload.get("resolved_config_sha256") != config["resolved_config_sha256"]
        or payload.get("hidden_namespace_opened") is not False
    ):
        raise PreflightError(
            "existing candidate-selection receipt is not bound to this campaign"
        )
    return receipts[0]


def run_final_analysis(config: dict[str, Any]) -> dict[str, Any]:
    receipt = _existing_selection_receipt(config)
    if receipt is None:
        selector = run_selector_stage(config)
        receipt = Path(selector["candidate_selection_receipt"])
    # Import reporting-only schemas only after the immutable candidate receipt
    # exists. The selector module itself has no hidden path/schema dependency.
    from .hidden_analysis import run_hidden_stage

    return run_hidden_stage(config, receipt)
