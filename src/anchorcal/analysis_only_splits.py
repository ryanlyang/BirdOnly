"""Fail-closed loader for protected oracle/test metadata.

Only hidden candidate evaluation and post-receipt reporting modules import this
module.  Practical selector modules have no path, schema, or import dependency
on this namespace.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .errors import PreflightError
from .io import sha256_file
from .splits import (
    ANALYSIS_ONLY_SPLIT_COLUMNS,
    ANALYSIS_ONLY_SPLIT_NAMES,
    ANALYSIS_ONLY_SPLIT_SCHEMA,
    WATERBIRDS100_RELEASE,
)


def _root(config: dict[str, Any]) -> Path:
    relative = Path(str(config["data"]["protected_split_root"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise PreflightError("protected split root must be output-root relative")
    return Path(config["paths"]["output_root"]) / relative


def load_analysis_only_splits(
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    root = _root(config)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise PreflightError(f"analysis-only split manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError("analysis-only split manifest is invalid") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != ANALYSIS_ONLY_SPLIT_SCHEMA
        or manifest.get("namespace") != "analysis_only"
        or manifest.get("reporting_only") is not True
        or manifest.get("source_release") != WATERBIRDS100_RELEASE
        or manifest.get("protected_columns") != ["place", "group"]
        or set(manifest.get("splits", {})) != set(ANALYSIS_ONLY_SPLIT_NAMES)
    ):
        raise PreflightError("analysis-only split manifest contract is invalid")
    assertions = manifest.get("contract_assertions")
    if not isinstance(assertions, dict) or any(
        assertions.get(name) is not True
        for name in (
            "oracle_val_test_disjoint",
            "development_hidden_disjoint",
            "group_equals_two_y_plus_place",
        )
    ):
        raise PreflightError("analysis-only split assertions are invalid")

    expected_metadata_hash = str(manifest.get("source_metadata_sha256", ""))
    preflight_path = Path(config["paths"]["output_root"]) / "preflight" / "report.json"
    if preflight_path.is_file():
        try:
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PreflightError("preflight report is invalid") from error
        if preflight.get("metadata_sha256") != expected_metadata_hash:
            raise PreflightError(
                "analysis-only split manifest metadata hash differs from preflight"
            )

    result: dict[str, pd.DataFrame] = {}
    for name in ANALYSIS_ONLY_SPLIT_NAMES:
        path = root / f"waterbirds100_{name}.csv"
        record = manifest["splits"].get(name)
        if not isinstance(record, dict) or not path.is_file():
            raise PreflightError(f"analysis-only split is missing: {name}")
        if (
            Path(str(record.get("path", ""))).resolve() != path.resolve()
            or record.get("sha256") != sha256_file(path)
            or record.get("columns") != list(ANALYSIS_ONLY_SPLIT_COLUMNS)
        ):
            raise PreflightError(f"analysis-only split provenance mismatch: {name}")
        frame = pd.read_csv(path)
        if tuple(frame.columns) != ANALYSIS_ONLY_SPLIT_COLUMNS:
            raise PreflightError(f"analysis-only split columns are invalid: {name}")
        for column in ("img_id", "metadata_index", "y", "place", "group", "split"):
            try:
                numeric = pd.to_numeric(frame[column], errors="raise").to_numpy(
                    dtype=np.float64
                )
            except (TypeError, ValueError, OverflowError) as error:
                raise PreflightError(
                    f"analysis-only split column is not integral: {name}.{column}"
                ) from error
            if not np.isfinite(numeric).all() or not np.equal(
                numeric, np.trunc(numeric)
            ).all():
                raise PreflightError(
                    f"analysis-only split column is not integral: {name}.{column}"
                )
            frame[column] = numeric.astype(np.int64)
        if frame.empty or frame["img_id"].duplicated().any():
            raise PreflightError(f"analysis-only split IDs are invalid: {name}")
        if not frame["y"].isin([0, 1]).all() or not frame["place"].isin([0, 1]).all():
            raise PreflightError(f"analysis-only labels are not binary: {name}")
        if not np.array_equal(
            frame["group"].to_numpy(),
            frame["y"].to_numpy() * 2 + frame["place"].to_numpy(),
        ):
            raise PreflightError(f"analysis-only group encoding is invalid: {name}")
        expected_split = 1 if name == "oracle_val" else 2
        if set(frame["split"].tolist()) != {expected_split}:
            raise PreflightError(f"analysis-only official split is invalid: {name}")
        if set(frame["source_metadata_sha256"].astype(str)) != {
            expected_metadata_hash
        }:
            raise PreflightError(f"analysis-only metadata hash column is invalid: {name}")
        if int(record.get("rows", -1)) != len(frame):
            raise PreflightError(f"analysis-only row count is invalid: {name}")
        expected_counts = {
            "class_counts": {
                str(key): int(value)
                for key, value in frame["y"].value_counts().sort_index().items()
            },
            "place_counts": {
                str(key): int(value)
                for key, value in frame["place"].value_counts().sort_index().items()
            },
            "group_counts": {
                str(key): int(value)
                for key, value in frame["group"].value_counts().sort_index().items()
            },
        }
        if any(record.get(key) != value for key, value in expected_counts.items()):
            raise PreflightError(f"analysis-only count summary is invalid: {name}")
        observed_correlation = float((frame["y"] == frame["place"]).mean())
        if float(record.get("empirical_correlation", -1.0)) != observed_correlation:
            raise PreflightError(
                f"analysis-only correlation summary is invalid: {name}"
            )
        result[name] = frame.sort_values("img_id", kind="stable").reset_index(drop=True)
    if set(result["oracle_val"]["img_id"]) & set(result["test"]["img_id"]):
        raise PreflightError("analysis-only oracle/test memberships overlap")
    return result


def load_analysis_only_split(
    config: dict[str, Any], name: str
) -> pd.DataFrame:
    if name not in ANALYSIS_ONLY_SPLIT_NAMES:
        raise ValueError(f"unknown analysis-only split: {name}")
    return load_analysis_only_splits(config)[name]
