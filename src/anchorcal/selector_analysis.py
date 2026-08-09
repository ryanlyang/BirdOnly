"""Selector-only final stage; this module has no hidden-schema dependency."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .anchor_artifacts import verify_anchor_artifacts
from .candidate_provenance import (
    load_candidate_preflight_binding,
    require_candidate_run_manifest,
)
from .candidate_schema import CANDIDATE_SCALAR_METRICS, candidate_per_example_shapes
from .decision import verify_decision_receipt
from .errors import AuditFailure, PreflightError
from .io import atomic_write_json, atomic_write_text, hash_object, sha256_file
from .provenance import write_hashed_receipt
from .paths import geometry_artifact_root
from .selector_storage import SELECTOR_FILENAME, SelectorVisibleReader
from .visible_checkpoint_verification import verify_visible_checkpoint_artifacts


PRIMARY_CRITERIA = (
    "ordinary_accuracy",
    "saliency_harmonic",
    "token_swap_harmonic",
    "background_blur_harmonic",
)

DIAGNOSTIC_METRICS = (
    "foreground_only_harmonic",
    "saliency_alignment",
    "swap_accuracy",
    "blur_accuracy",
    "foreground_only_accuracy",
    "saliency_product",
    "swap_product",
    "blur_product",
    "swap_mean_true_class_margin_drop",
    "swap_prediction_flip_rate",
    "swap_donor_margin_variance",
)


def _run_id(learning_rate: float, weight_decay: float, seed: int) -> str:
    lr = f"{learning_rate:.0e}".replace("+", "").replace("-0", "-")
    return f"lr{lr}_wd{weight_decay:.2f}_seed{seed}"


def _row_is_better(
    candidate: pd.Series,
    incumbent: pd.Series,
    criterion: str,
    tolerance: float,
) -> bool:
    for column, direction in (
        (criterion, 1.0),
        ("biased_accuracy", 1.0),
        ("biased_mean_loss", -1.0),
    ):
        difference = direction * (
            float(candidate[column]) - float(incumbent[column])
        )
        if abs(difference) > tolerance:
            return difference > 0.0
    for column in ("epoch", "learning_rate", "weight_decay"):
        left = float(candidate[column])
        right = float(incumbent[column])
        if left != right:
            return left < right
    return str(candidate["candidate_id"]) < str(incumbent["candidate_id"])


def select_candidate(table: pd.DataFrame, criterion: str, tolerance: float) -> pd.Series:
    if table.empty or criterion not in table:
        raise ValueError("candidate table is empty or lacks the requested criterion")
    iterator = (row for _, row in table.iterrows())
    incumbent = next(iterator)
    for candidate in iterator:
        if _row_is_better(candidate, incumbent, criterion, tolerance):
            incumbent = candidate
    return incumbent


def run_selector_stage(config: dict[str, Any]) -> dict[str, Any]:
    output = Path(config["paths"]["output_root"])
    debug = bool(config["runtime"]["debug"])
    receipt_root = output / ("debug/receipt" if debug else "receipt")
    anchor_receipts = sorted(receipt_root.glob("anchorcal_decision_*.json"))
    if len(anchor_receipts) != 1 or not verify_decision_receipt(anchor_receipts[0]):
        raise PreflightError("a unique verified AnchorCal decision receipt is required")
    anchor_payload = json.loads(anchor_receipts[0].read_text(encoding="utf-8"))
    if (
        anchor_payload.get("schema_version") != 1
        or anchor_payload.get("receipt_type") != "anchorcal_criterion_decision"
        or anchor_payload.get("config_sha256", {}).get("resolved")
        != config["resolved_config_sha256"]
    ):
        raise PreflightError("AnchorCal decision receipt is not bound to this config")
    verify_anchor_artifacts(config, decision_receipt=anchor_receipts[0])
    preflight = load_candidate_preflight_binding(config)
    candidate_root = output / ("debug/candidates" if debug else "candidates")
    expected_grid = {
        _run_id(float(learning_rate), float(weight_decay), int(config["candidate_grid"]["seed"])): (
            float(learning_rate),
            float(weight_decay),
        )
        for learning_rate in config["candidate_grid"]["learning_rates"]
        for weight_decay in config["candidate_grid"]["weight_decays"]
    }
    expected_runs = len(expected_grid)
    run_dirs = sorted(path.parent for path in candidate_root.glob(f"*/{SELECTOR_FILENAME}"))
    found_run_ids = {path.name for path in run_dirs}
    if len(run_dirs) != expected_runs or found_run_ids != set(expected_grid):
        raise AuditFailure(
            "candidate grid mismatch; "
            f"missing={sorted(set(expected_grid) - found_run_ids)}, "
            f"unexpected={sorted(found_run_ids - set(expected_grid))}"
        )
    biased_split = pd.read_csv(
        output / "splits" / "waterbirds100_biased_val.csv"
    ).sort_values("img_id", kind="stable")
    expected_img_ids = biased_split["img_id"].to_numpy(dtype=np.int64)
    expected_labels = biased_split["y"].to_numpy(dtype=np.int64)
    fixed_selector = pd.read_csv(
        geometry_artifact_root(config) / "selector_eval_subset.csv"
    ).sort_values("img_id", kind="stable").reset_index(drop=True)
    expected_selector_img_ids = fixed_selector["img_id"].to_numpy(dtype=np.int64)
    expected_selector_labels = fixed_selector["y"].to_numpy(dtype=np.int64)
    expected_selector_shapes = candidate_per_example_shapes(
        len(fixed_selector),
        swap_donors=int(config["criteria"]["swap_donors"]),
        blur_sigmas=len(config["criteria"]["blur_sigmas"]),
    )
    rows: list[dict[str, Any]] = []
    visible_hashes: dict[str, str] = {}
    visible_checkpoint_hashes: dict[str, str] = {}
    checkpoint_verifications: dict[str, dict[str, Any]] = {}
    for run_dir in run_dirs:
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        expected_lr, expected_wd = expected_grid[run_dir.name]
        try:
            require_candidate_run_manifest(
                manifest,
                config,
                preflight,
                expected_run_id=run_dir.name,
                expected_decision_sha256=sha256_file(anchor_receipts[0]),
            )
        except PreflightError as error:
            raise AuditFailure(str(error)) from error
        if (
            float(manifest.get("learning_rate", -1)) != expected_lr
            or float(manifest.get("weight_decay", -1)) != expected_wd
            or int(manifest.get("seed", -1)) != int(config["candidate_grid"]["seed"])
            or Path(manifest.get("decision_receipt", "")).resolve()
            != anchor_receipts[0].resolve()
        ):
            raise AuditFailure(f"candidate run provenance mismatch: {run_dir.name}")
        completion_path = run_dir / "completion.json"
        if not completion_path.is_file():
            raise AuditFailure(f"candidate completion/checkpoint receipt is missing: {run_dir.name}")
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        checkpoint_verification = verify_visible_checkpoint_artifacts(
            run_dir,
            expected_run_id=run_dir.name,
            require_complete=True,
            required_selectors=("ordinary", "saliency", "swap", "blur", "final"),
        )
        checkpoint_verifications[run_dir.name] = checkpoint_verification
        if (
            completion.get("run_id") != run_dir.name
            or int(completion.get("epochs", -1)) != int(config["candidate_grid"]["epochs"])
            or completion.get("checkpoint_manifest_sha256")
            != checkpoint_verification["manifest_sha256"]
        ):
            raise AuditFailure(f"candidate rolling checkpoint set is incomplete: {run_dir.name}")
        visible_checkpoint_hashes[run_dir.name] = checkpoint_verification[
            "manifest_sha256"
        ]
        visible_hashes[manifest["run_id"]] = sha256_file(run_dir / SELECTOR_FILENAME)
        with SelectorVisibleReader(
            run_dir / SELECTOR_FILENAME, expected_run_id=run_dir.name
        ) as reader:
            reader.validate_candidate_schema(
                metric_names=CANDIDATE_SCALAR_METRICS,
                per_example_shapes=expected_selector_shapes,
            )
            completed_slots = reader.completed_slots
            epochs = int(config["candidate_grid"]["epochs"])
            if completed_slots != tuple(range(epochs)):
                raise AuditFailure(f"candidate run is incomplete: {run_dir.name}")
            sample_metadata = reader.sample_metadata()
            if not np.array_equal(sample_metadata["img_id"], expected_img_ids) or not np.array_equal(
                sample_metadata["label"], expected_labels
            ):
                raise AuditFailure(f"biased-val sample metadata differs: {run_dir.name}")
            selector_metadata = reader.selector_subset_metadata()
            if selector_metadata is None or not np.array_equal(
                selector_metadata["img_id"], expected_selector_img_ids
            ) or not np.array_equal(
                selector_metadata["label"], expected_selector_labels
            ):
                raise AuditFailure(
                    f"fixed selector-subset metadata differs: {run_dir.name}"
                )
            observed_epoch_numbers: list[int] = []
            for within_run_id, slot in enumerate(completed_slots, start=1):
                epoch = reader.read_epoch(slot)
                observed_epoch_numbers.append(int(epoch["epoch_number"]))
                metrics = epoch["metrics"]
                rows.append(
                    {
                        "global_candidate_id": -1,
                        "within_run_candidate_id": within_run_id,
                        "candidate_id": f"{manifest['run_id']}_epoch_{epoch['epoch_number']}",
                        "run_id": manifest["run_id"],
                        "epoch": epoch["epoch_number"],
                        "learning_rate": manifest["learning_rate"],
                        "weight_decay": manifest["weight_decay"],
                        "seed": manifest["seed"],
                        "biased_accuracy": metrics["ordinary_accuracy"],
                        "biased_mean_loss": metrics["biased_mean_loss"],
                        **{criterion: metrics[criterion] for criterion in PRIMARY_CRITERIA},
                        **{name: metrics[name] for name in DIAGNOSTIC_METRICS},
                    }
                )
            if observed_epoch_numbers != list(range(1, epochs + 1)):
                raise AuditFailure(
                    f"candidate epoch numbers are not exactly 1..{epochs}: {run_dir.name}"
                )
    table = pd.DataFrame(rows).sort_values(
        ["learning_rate", "weight_decay", "epoch"], kind="stable"
    ).reset_index(drop=True)
    table["global_candidate_id"] = np.arange(1, len(table) + 1)
    expected_candidates = expected_runs * int(config["candidate_grid"]["epochs"])
    if len(table) != expected_candidates or table["candidate_id"].duplicated().any():
        raise AuditFailure(
            f"candidate completeness requires {expected_candidates} unique states, found {len(table)}"
        )
    analysis_root = output / ("debug/analysis" if debug else "analysis")
    analysis_root.mkdir(parents=True, exist_ok=True)
    selector_table = analysis_root / "all_candidates_selector_only.csv"
    atomic_write_text(selector_table, table.to_csv(index=False, lineterminator="\n"))
    tolerance = float(config["anchorcal"]["candidate_score_tolerance"])
    selected = {
        criterion: select_candidate(table, criterion, tolerance)[
            [
                "candidate_id",
                "run_id",
                "epoch",
                "learning_rate",
                "weight_decay",
                "seed",
                "biased_accuracy",
                "biased_mean_loss",
                criterion,
            ]
        ].to_dict()
        for criterion in PRIMARY_CRITERIA
    }
    checkpoint_selector = {
        "ordinary_accuracy": "ordinary",
        "saliency_harmonic": "saliency",
        "token_swap_harmonic": "swap",
        "background_blur_harmonic": "blur",
    }
    for criterion, selected_record in selected.items():
        rolling = checkpoint_verifications[str(selected_record["run_id"])][
            "selectors"
        ][checkpoint_selector[criterion]]
        if int(rolling["epoch"]) != int(selected_record["epoch"]):
            raise AuditFailure(
                f"global {criterion} selection is absent from its rolling checkpoint"
            )
    winner = anchor_payload["decision"]["winner"]
    if winner not in selected:
        raise AuditFailure(f"AnchorCal receipt winner is not eligible: {winner}")
    receipt, sidecar = write_hashed_receipt(
        receipt_root,
        "candidate_selection",
        {
            "schema_version": "anchorcal-candidate-selection-v2",
            "resolved_config_sha256": config["resolved_config_sha256"],
            "anchorcal_decision_receipt": str(anchor_receipts[0].resolve()),
            "anchorcal_decision_sha256": sha256_file(anchor_receipts[0]),
            "anchorcal_winner": winner,
            "selected_by_criterion": selected,
            "anchorcal_selected_candidate": selected[winner],
            "selector_table": str(selector_table.resolve()),
            "selector_table_sha256": sha256_file(selector_table),
            "visible_hdf5_sha256": visible_hashes,
            "visible_checkpoint_manifest_sha256": visible_checkpoint_hashes,
            "expected_grid": {
                run_id: {"learning_rate": values[0], "weight_decay": values[1]}
                for run_id, values in sorted(expected_grid.items())
            },
            "biased_val_sample_sha256": hash_object(
                {"img_id": expected_img_ids.tolist(), "label": expected_labels.tolist()}
            ),
            "hidden_namespace_opened": False,
        },
    )
    for run_dir in run_dirs:
        verified = verify_visible_checkpoint_artifacts(
            run_dir,
            expected_run_id=run_dir.name,
            require_complete=True,
            required_selectors=("ordinary", "saliency", "swap", "blur", "final"),
        )
        if verified["manifest_sha256"] != visible_checkpoint_hashes[run_dir.name]:
            raise AuditFailure(
                f"candidate checkpoint manifest changed while freezing selection: {run_dir.name}"
            )
    result = {
        "candidate_selection_receipt": str(receipt),
        "candidate_selection_sidecar": str(sidecar),
        "selector_table": str(selector_table),
        "anchorcal_winner": winner,
        "selected_by_criterion": selected,
    }
    atomic_write_json(analysis_root / "selector_stage.json", result)
    return result
