"""Reporting-only analysis opened after practical selection is immutably frozen."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

from .checkpoint_verification import verify_candidate_checkpoint_artifacts
from .candidate_provenance import (
    load_candidate_preflight_binding,
    require_candidate_run_manifest,
)
from .errors import AuditFailure, PreflightError
from .anchor_artifacts import verify_anchor_artifacts
from .hidden_storage import HIDDEN_FILENAME, HiddenMetricsReader
from .io import atomic_write_json, atomic_write_text, sha256_file
from .metrics import classification_metrics
from .provenance import verify_hashed_receipt
from .seeds import stable_seed
from .storage import verify_candidate_storage


PRIMARY_CRITERIA = (
    "ordinary_accuracy",
    "saliency_harmonic",
    "token_swap_harmonic",
    "background_blur_harmonic",
)

REAL_QUALITY_TARGETS = (
    "test_wga",
    "test_accuracy",
    "oracle_wga",
)


def _python(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_python(item) for item in value]
    if isinstance(value, np.ndarray):
        return _python(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA or (value is not None and not isinstance(value, str) and pd.isna(value)):
        return None
    return value


def _safe_correlation(
    function: Callable[..., Any], first: Any, second: Any, **kwargs: Any
) -> float | None:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if (
        len(left) < 2
        or np.ptp(left) <= 0.0
        or np.ptp(right) <= 0.0
    ):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = function(left, right, **kwargs).statistic
    return float(value) if np.isfinite(value) else None


def _correlation_triplet(
    scores: Any, targets: Any
) -> dict[str, float | None]:
    """Return the three locked real-quality correlations with JSON-safe nulls.

    A correlation is undefined when fewer than two finite pairs remain or either
    vector is constant.  Representing that case as ``None`` here, rather than a
    floating-point NaN, keeps both JSON artifacts and the corresponding blank
    CSV fields portable and standards compliant.
    """

    return {
        "pearson": _safe_correlation(pearsonr, scores, targets),
        "spearman": _safe_correlation(spearmanr, scores, targets),
        "kendall_tau_b": _safe_correlation(
            kendalltau, scores, targets, variant="b"
        ),
    }


def _correlation_columns(
    frame: pd.DataFrame,
    criterion: str,
    *,
    prefix: str = "",
) -> tuple[dict[str, float | None], dict[str, dict[str, float | None]]]:
    """Flatten and retain a structured 3-by-3 criterion/target report."""

    nested: dict[str, dict[str, float | None]] = {}
    columns: dict[str, float | None] = {}
    for target in REAL_QUALITY_TARGETS:
        correlations = _correlation_triplet(frame[criterion], frame[target])
        nested[target] = correlations
        for method, value in correlations.items():
            columns[f"{prefix}{method}_{target}"] = value
    return columns, nested


def _summary(values: list[float | None]) -> tuple[float | None, float | None, int, int]:
    finite = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if not len(finite):
        return None, None, 0, len(values)
    standard_deviation = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    return float(finite.mean()), standard_deviation, int(len(finite)), int(len(values) - len(finite))


def _ranking_pair_accuracy(
    scores: np.ndarray, targets: np.ndarray, tolerance: float
) -> float:
    credit = 0.0
    count = 0
    for first in range(len(scores) - 1):
        for second in range(first + 1, len(scores)):
            target_delta = targets[second] - targets[first]
            if abs(target_delta) <= tolerance:
                continue
            score_delta = scores[second] - scores[first]
            credit += 0.5 if abs(score_delta) <= tolerance else float(
                np.sign(score_delta) == np.sign(target_delta)
            )
            count += 1
    return credit / count if count else 0.5


def _metric_from_reader(
    reader: HiddenMetricsReader, split: str, slot: int
) -> dict[str, Any]:
    metadata = reader.sample_metadata(split)
    epoch = reader.read_epoch(split, slot)
    metrics = classification_metrics(
        epoch["logits"], metadata["label"], metadata["group"] % 2
    )
    return {**metrics, "epoch": epoch, "metadata": metadata}


def _stratified_indices(groups: np.ndarray, replicates: int, seed: int) -> np.ndarray:
    groups = np.asarray(groups, dtype=np.int64)
    unique = np.unique(groups)
    if set(unique.tolist()) != {0, 1, 2, 3}:
        raise AuditFailure("final bootstrap requires all four Waterbirds groups")
    by_group = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(seed)
    result = np.empty((replicates, len(groups)), dtype=np.int64)
    for replicate in range(replicates):
        cursor = 0
        for indices in by_group:
            sampled = rng.choice(indices, len(indices), replace=True)
            result[replicate, cursor : cursor + len(indices)] = sampled
            cursor += len(indices)
    return result


def _bootstrap_accuracy_wga(
    correct: np.ndarray, groups: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(correct, dtype=np.float64)
    group_array = np.asarray(groups, dtype=np.int64)
    accuracy = np.empty(len(indices), dtype=np.float64)
    wga = np.empty(len(indices), dtype=np.float64)
    for replicate, sampled in enumerate(indices):
        accuracy[replicate] = float(values[sampled].mean())
        wga[replicate] = min(
            float(values[sampled][group_array[sampled] == group].mean())
            for group in (0, 1, 2, 3)
        )
    return accuracy, wga


def _interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def _best_finite(mapping: dict[str, float | None], *, maximize: bool) -> str | None:
    finite = {name: value for name, value in mapping.items() if value is not None}
    if not finite:
        return None
    return (max if maximize else min)(finite, key=lambda name: float(finite[name]))


def _require_receipt_binding(
    selection: dict[str, Any], config: dict[str, Any], receipt: Path
) -> None:
    if (
        selection.get("schema_version") != "anchorcal-candidate-selection-v2"
        or selection.get("resolved_config_sha256") != config["resolved_config_sha256"]
        or selection.get("hidden_namespace_opened") is not False
    ):
        raise PreflightError("candidate selection receipt is not bound to this campaign")
    anchor = Path(selection.get("anchorcal_decision_receipt", ""))
    if (
        not anchor.is_file()
        or sha256_file(anchor) != selection.get("anchorcal_decision_sha256")
    ):
        raise PreflightError("AnchorCal decision changed after practical selection")
    if not receipt.resolve().is_relative_to(anchor.parent.resolve()):
        raise PreflightError("candidate selection receipt is outside the decision namespace")


def run_hidden_stage(
    config: dict[str, Any], selection_receipt: str | Path
) -> dict[str, Any]:
    receipt = Path(selection_receipt)
    if not verify_hashed_receipt(receipt):
        raise PreflightError("candidate selection receipt is invalid")
    selection = json.loads(receipt.read_text(encoding="utf-8"))
    _require_receipt_binding(selection, config, receipt)
    output = Path(config["paths"]["output_root"])
    debug = bool(config["runtime"]["debug"])
    analysis_root = output / ("debug/analysis" if debug else "analysis")
    analysis_root.mkdir(parents=True, exist_ok=True)
    selector_path = Path(selection["selector_table"])
    if (
        not selector_path.resolve().is_relative_to(analysis_root.resolve())
        or sha256_file(selector_path) != selection["selector_table_sha256"]
    ):
        raise PreflightError("selector-only table changed or escaped its namespace")
    table = pd.read_csv(selector_path)
    candidate_root = output / ("debug/candidates" if debug else "candidates")
    preflight = load_candidate_preflight_binding(config)
    expected_visible_checkpoint_hashes = selection.get(
        "visible_checkpoint_manifest_sha256"
    )
    observed_run_ids = {str(value) for value in table["run_id"].unique()}
    if (
        not isinstance(expected_visible_checkpoint_hashes, dict)
        or set(expected_visible_checkpoint_hashes) != observed_run_ids
    ):
        raise PreflightError(
            "candidate selection receipt does not bind the exact checkpoint grid"
        )
    expected_splits = {
        split: pd.read_csv(output / "splits" / f"waterbirds100_{split}.csv")
        .sort_values("img_id", kind="stable")
        .reset_index(drop=True)
        for split in ("oracle_val", "test")
    }
    reference_metadata = {
        split: {
            "img_id": frame["img_id"].to_numpy(dtype=np.int64),
            "label": frame["y"].to_numpy(dtype=np.int64),
            "group": (frame["y"] * 2 + frame["place"]).to_numpy(dtype=np.int64),
        }
        for split, frame in expected_splits.items()
    }
    hidden_rows: list[dict[str, Any]] = []
    correctness: dict[str, dict[str, np.ndarray]] = {}
    hidden_hdf5_sha256: dict[str, str] = {}
    epochs = int(config["candidate_grid"]["epochs"])
    for run_id, run_table in table.groupby("run_id", sort=False):
        run_dir = candidate_root / str(run_id)
        storage_manifest = verify_candidate_storage(
            run_dir, expected_run_id=str(run_id)
        )
        visible_record = storage_manifest["files"]["selector_visible"]
        hidden_record = storage_manifest["files"]["exploratory_hidden_metrics"]
        if visible_record["sha256"] != selection["visible_hdf5_sha256"].get(str(run_id)):
            raise PreflightError(f"selector-visible HDF5 changed after receipt: {run_id}")
        run_manifest = json.loads(
            (run_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        require_candidate_run_manifest(
            run_manifest,
            config,
            preflight,
            expected_run_id=str(run_id),
            expected_decision_sha256=selection["anchorcal_decision_sha256"],
        )
        checkpoint_verification = verify_candidate_checkpoint_artifacts(
            run_dir,
            expected_run_id=str(run_id),
            require_complete=True,
            required_visible_selectors=(
                "ordinary",
                "saliency",
                "swap",
                "blur",
                "final",
            ),
            required_hidden_selectors=("oracle",),
        )
        if (
            checkpoint_verification["visible"]["manifest_sha256"]
            != expected_visible_checkpoint_hashes.get(str(run_id))
        ):
            raise PreflightError(
                f"visible checkpoint manifest changed after selection: {run_id}"
            )
        completion = json.loads((run_dir / "completion.json").read_text(encoding="utf-8"))
        if (
            completion.get("checkpoint_manifest_sha256")
            != checkpoint_verification["visible"]["manifest_sha256"]
            or completion.get("hidden_checkpoint_manifest_sha256")
            != checkpoint_verification["hidden"]["manifest_sha256"]
        ):
            raise PreflightError(
                f"candidate completion/checkpoint binding is invalid: {run_id}"
            )
        hidden_path = run_dir / HIDDEN_FILENAME
        if hidden_record.get("sha256") != sha256_file(hidden_path):
            raise PreflightError(f"hidden HDF5 provenance mismatch: {run_id}")
        hidden_hdf5_sha256[str(run_id)] = str(hidden_record["sha256"])
        with HiddenMetricsReader(
            hidden_path, expected_run_id=str(run_id)
        ) as reader:
            if reader.completed_slots != tuple(range(epochs)):
                raise AuditFailure(f"hidden candidate epochs are incomplete: {run_id}")
            for split in ("oracle_val", "test"):
                observed = reader.sample_metadata(split)
                expected = reference_metadata[split]
                if any(
                    not np.array_equal(observed[name], expected[name])
                    for name in ("img_id", "label", "group")
                ):
                    raise AuditFailure(
                        f"hidden {split} sample metadata differs for run {run_id}"
                    )
            observed_epochs: list[int] = []
            for _, row in run_table.sort_values("epoch", kind="stable").iterrows():
                slot = int(row["epoch"]) - 1
                oracle = _metric_from_reader(reader, "oracle_val", slot)
                test = _metric_from_reader(reader, "test", slot)
                if oracle["epoch"]["epoch_number"] != test["epoch"]["epoch_number"]:
                    raise AuditFailure(f"hidden split epoch mismatch: {row['candidate_id']}")
                observed_epochs.append(int(oracle["epoch"]["epoch_number"]))
                if observed_epochs[-1] != int(row["epoch"]):
                    raise AuditFailure(f"selector/hidden epoch mismatch: {row['candidate_id']}")
                candidate_id = str(row["candidate_id"])
                hidden_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "oracle_wga": oracle["worst_group_accuracy"],
                        "oracle_group_balanced_accuracy": oracle[
                            "group_balanced_accuracy"
                        ],
                        "oracle_accuracy": oracle["accuracy"],
                        "test_wga": test["worst_group_accuracy"],
                        "test_group_balanced_accuracy": test[
                            "group_balanced_accuracy"
                        ],
                        "test_accuracy": test["accuracy"],
                    }
                )
                correctness[candidate_id] = {
                    "oracle_val": oracle["epoch"]["correct"].astype(bool),
                    "test": test["epoch"]["correct"].astype(bool),
                }
        if observed_epochs != list(range(1, epochs + 1)):
            raise AuditFailure(f"hidden epochs are not exactly 1..{epochs}: {run_id}")
    joined = table.merge(
        pd.DataFrame(hidden_rows), on="candidate_id", validate="one_to_one"
    )
    joined_path = analysis_root / "all_candidates.csv"
    atomic_write_text(joined_path, joined.to_csv(index=False, lineterminator="\n"))
    test_range = float(joined["test_wga"].max() - joined["test_wga"].min())
    oracle_range = float(joined["oracle_wga"].max() - joined["oracle_wga"].min())
    competent = joined.loc[
        joined["biased_accuracy"] >= joined["biased_accuracy"].max() - 0.01
    ].copy()
    tolerance = float(config["anchorcal"]["candidate_score_tolerance"])
    quality_rows: list[dict[str, Any]] = []
    competent_within: dict[str, list[float | None]] = {}
    competent_within_by_run: dict[str, dict[str, float | None]] = {}
    competent_correlations: dict[
        str, dict[str, dict[str, float | None]]
    ] = {}
    run_ids = joined["run_id"].astype(str).drop_duplicates().tolist()
    for criterion in PRIMARY_CRITERIA:
        scores = joined[criterion].to_numpy(dtype=np.float64)
        test_wga = joined["test_wga"].to_numpy(dtype=np.float64)
        global_columns, _ = _correlation_columns(joined, criterion)
        competent_columns, competent_nested = _correlation_columns(
            competent, criterion, prefix="competent_pool_"
        )
        competent_correlations[criterion] = competent_nested
        within = [
            _safe_correlation(spearmanr, group[criterion], group["test_wga"])
            for _, group in joined.groupby("run_id")
        ]
        within_mean, within_sd, within_valid, within_na = _summary(within)
        competent_by_run = {
            run_id: _safe_correlation(
                spearmanr,
                competent.loc[
                    competent["run_id"].astype(str) == run_id, criterion
                ],
                competent.loc[
                    competent["run_id"].astype(str) == run_id, "test_wga"
                ],
            )
            for run_id in run_ids
        }
        competent_values = list(competent_by_run.values())
        competent_within[criterion] = competent_values
        competent_within_by_run[criterion] = competent_by_run
        comp_mean, comp_sd, comp_valid, comp_na = _summary(competent_values)
        quality_rows.append(
            {
                "criterion": criterion,
                **global_columns,
                # Backward-compatible aliases used by the original report
                # tables before all target names were made explicit.
                "kendall_tau_b": global_columns["kendall_tau_b_test_wga"],
                "kendall_test_accuracy": global_columns[
                    "kendall_tau_b_test_accuracy"
                ],
                "kendall_oracle_wga": global_columns[
                    "kendall_tau_b_oracle_wga"
                ],
                "mean_within_run_spearman": within_mean,
                "std_within_run_spearman": within_sd,
                "within_run_valid": within_valid,
                "within_run_na": within_na,
                **competent_columns,
                # Preserve the primary-target shorthand consumed by the
                # existing publication table.
                "competent_pool_spearman": competent_columns[
                    "competent_pool_spearman_test_wga"
                ],
                "mean_within_run_competent_spearman": comp_mean,
                "std_within_run_competent_spearman": comp_sd,
                "within_run_competent_valid": comp_valid,
                "within_run_competent_na": comp_na,
                "pair_accuracy": _ranking_pair_accuracy(scores, test_wga, tolerance),
            }
        )
    quality = pd.DataFrame(quality_rows)
    quality_path = analysis_root / "criterion_real_quality.csv"
    atomic_write_text(quality_path, quality.to_csv(index=False, lineterminator="\n"))
    atomic_write_text(
        analysis_root / "competent_pool.csv",
        competent.to_csv(index=False, lineterminator="\n"),
    )
    competent_summary = {
        "definition": "biased_accuracy_at_least_global_max_minus_0.01",
        "count": int(len(competent)),
        "underpowered": bool(len(competent) < 20),
        "epochs_represented": sorted(competent["epoch"].astype(int).unique().tolist()),
        "run_ids_represented": sorted(competent["run_id"].astype(str).unique().tolist()),
        "configuration_count": int(
            competent[["learning_rate", "weight_decay"]].drop_duplicates().shape[0]
        ),
        "configuration_coverage": float(
            competent[["learning_rate", "weight_decay"]].drop_duplicates().shape[0]
            / max(1, joined[["learning_rate", "weight_decay"]].drop_duplicates().shape[0])
        ),
        "correlations": competent_correlations,
        "secondary_within_run_correlations": competent_within,
        "secondary_within_run_correlations_by_run": competent_within_by_run,
    }
    atomic_write_json(analysis_root / "competent_pool_summary.json", _python(competent_summary))

    selected_rows: list[dict[str, Any]] = []
    for criterion, selected in selection["selected_by_criterion"].items():
        row = joined.loc[joined["candidate_id"] == selected["candidate_id"]]
        if len(row) != 1:
            raise AuditFailure(f"frozen selection is absent or duplicated: {criterion}")
        value = row.iloc[0]
        selected_rows.append(
            {
                "criterion": criterion,
                "selection_source": "practical",
                "candidate_id": value["candidate_id"],
                "run_id": value["run_id"],
                "learning_rate": value["learning_rate"],
                "weight_decay": value["weight_decay"],
                "seed": value["seed"],
                "epoch": value["epoch"],
                "biased_accuracy": value["biased_accuracy"],
                "biased_mean_loss": value["biased_mean_loss"],
                "oracle_wga": value["oracle_wga"],
                "oracle_accuracy": value["oracle_accuracy"],
                "test_wga": value["test_wga"],
                "test_accuracy": value["test_accuracy"],
                "test_selection_regret": float(joined["test_wga"].max() - value["test_wga"]),
                "oracle_selection_regret": float(
                    joined["oracle_wga"].max() - value["oracle_wga"]
                ),
            }
        )
    oracle_selected = joined.sort_values(
        ["oracle_wga", "oracle_group_balanced_accuracy", "oracle_accuracy", "epoch"],
        ascending=[False, False, False, True],
        kind="stable",
    ).iloc[0]
    test_hindsight = joined.sort_values(
        ["test_wga", "test_group_balanced_accuracy", "test_accuracy", "epoch"],
        ascending=[False, False, False, True],
        kind="stable",
    ).iloc[0]
    selected_rows.append(
        {
            "criterion": "oracle_validation",
            "selection_source": "reporting_only_oracle",
            "candidate_id": oracle_selected["candidate_id"],
            "run_id": oracle_selected["run_id"],
            "learning_rate": oracle_selected["learning_rate"],
            "weight_decay": oracle_selected["weight_decay"],
            "seed": oracle_selected["seed"],
            "epoch": oracle_selected["epoch"],
            "biased_accuracy": oracle_selected["biased_accuracy"],
            "biased_mean_loss": oracle_selected["biased_mean_loss"],
            "oracle_wga": oracle_selected["oracle_wga"],
            "oracle_accuracy": oracle_selected["oracle_accuracy"],
            "test_wga": oracle_selected["test_wga"],
            "test_accuracy": oracle_selected["test_accuracy"],
            "test_selection_regret": float(
                joined["test_wga"].max() - oracle_selected["test_wga"]
            ),
            "oracle_selection_regret": 0.0,
        }
    )
    selected_table = pd.DataFrame(selected_rows)

    replicates = int(config["anchorcal"]["final_metric_bootstrap_replicates"])
    base_seed = int(config["seeds"]["final_metric_bootstrap"])
    bootstrap_indices = {
        split: _stratified_indices(
            reference_metadata[split]["group"],
            replicates,
            stable_seed(base_seed, split, "final_selected_metrics"),
        )
        for split in ("oracle_val", "test")
    }
    selected_ids = set(selected_table["candidate_id"].astype(str)) | {
        str(test_hindsight["candidate_id"]),
        str(oracle_selected["candidate_id"]),
    }
    draws: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    intervals: dict[str, Any] = {}
    for candidate_id in sorted(selected_ids):
        draws[candidate_id] = {}
        intervals[candidate_id] = {}
        for split in ("oracle_val", "test"):
            accuracy_draws, wga_draws = _bootstrap_accuracy_wga(
                correctness[candidate_id][split],
                reference_metadata[split]["group"],
                bootstrap_indices[split],
            )
            draws[candidate_id][split] = (accuracy_draws, wga_draws)
            intervals[candidate_id][split] = {
                "accuracy_95": _interval(accuracy_draws),
                "wga_95": _interval(wga_draws),
                "replicates": replicates,
            }
    test_reference_id = str(test_hindsight["candidate_id"])
    oracle_reference_id = str(oracle_selected["candidate_id"])
    for row_index, row in selected_table.iterrows():
        candidate_id = str(row["candidate_id"])
        selected_table.loc[row_index, "test_accuracy_ci_low"] = intervals[candidate_id][
            "test"
        ]["accuracy_95"][0]
        selected_table.loc[row_index, "test_accuracy_ci_high"] = intervals[candidate_id][
            "test"
        ]["accuracy_95"][1]
        selected_table.loc[row_index, "test_wga_ci_low"] = intervals[candidate_id]["test"][
            "wga_95"
        ][0]
        selected_table.loc[row_index, "test_wga_ci_high"] = intervals[candidate_id]["test"][
            "wga_95"
        ][1]
        selected_table.loc[row_index, "oracle_wga_ci_low"] = intervals[candidate_id][
            "oracle_val"
        ]["wga_95"][0]
        selected_table.loc[row_index, "oracle_wga_ci_high"] = intervals[candidate_id][
            "oracle_val"
        ]["wga_95"][1]
        test_regret_draws = (
            draws[test_reference_id]["test"][1] - draws[candidate_id]["test"][1]
        )
        oracle_regret_draws = (
            draws[oracle_reference_id]["oracle_val"][1]
            - draws[candidate_id]["oracle_val"][1]
        )
        selected_table.loc[row_index, "paired_test_regret_ci_low"] = _interval(
            test_regret_draws
        )[0]
        selected_table.loc[row_index, "paired_test_regret_ci_high"] = _interval(
            test_regret_draws
        )[1]
        selected_table.loc[row_index, "paired_oracle_regret_ci_low"] = _interval(
            oracle_regret_draws
        )[0]
        selected_table.loc[row_index, "paired_oracle_regret_ci_high"] = _interval(
            oracle_regret_draws
        )[1]
    selected_path = analysis_root / "selected_candidates.csv"
    atomic_write_text(
        selected_path, selected_table.to_csv(index=False, lineterminator="\n")
    )

    anchor_winner = str(selection["anchorcal_winner"])
    anchor_selected = selected_table.loc[
        selected_table["criterion"] == anchor_winner
    ].iloc[0]
    anchor_root = output / ("debug/anchors" if debug else "anchors")
    criterion_results_path = anchor_root / "criterion_results.json"
    anchor_receipt = json.loads(
        Path(selection["anchorcal_decision_receipt"]).read_text(encoding="utf-8")
    )
    _, anchor_results = verify_anchor_artifacts(
        config,
        decision_receipt=selection["anchorcal_decision_receipt"],
    )
    frozen_point_metrics = anchor_receipt.get("decision", {}).get(
        "point_metrics", {}
    )
    if any(
        anchor_results.get(name, {}).get("metrics")
        != frozen_point_metrics.get(name)
        for name in PRIMARY_CRITERIA
    ):
        raise PreflightError(
            "anchor point metrics do not match the frozen AnchorCal decision"
        )
    credible_set = set(anchor_receipt["decision"]["credible_set"])
    quality_lookup = quality.set_index("criterion")
    selected_lookup = selected_table.loc[
        selected_table["criterion"].isin(PRIMARY_CRITERIA)
    ].set_index("criterion")
    spearman_map = {
        name: _python(quality_lookup.loc[name, "spearman_test_wga"])
        for name in PRIMARY_CRITERIA
    }
    regret_map = {
        name: float(selected_lookup.loc[name, "test_selection_regret"])
        for name in PRIMARY_CRITERIA
    }
    best_spearman = _best_finite(spearman_map, maximize=True)
    lowest_regret = _best_finite(regret_map, maximize=False)
    anchor_ace = np.asarray(
        [anchor_results[name]["metrics"]["ace"] for name in PRIMARY_CRITERIA],
        dtype=np.float64,
    )
    real_spearman = np.asarray(
        [np.nan if spearman_map[name] is None else spearman_map[name] for name in PRIMARY_CRITERIA],
        dtype=np.float64,
    )
    real_regret = np.asarray([regret_map[name] for name in PRIMARY_CRITERIA])
    meta_spearman = _safe_correlation(spearmanr, -anchor_ace, real_spearman)
    meta_regret = _safe_correlation(spearmanr, -anchor_ace, -real_regret)
    valid_ranked = sorted(
        (name for name in PRIMARY_CRITERIA if spearman_map[name] is not None),
        key=lambda name: float(spearman_map[name]),
        reverse=True,
    )
    spearman_rank = (
        valid_ranked.index(anchor_winner) + 1 if anchor_winner in valid_ranked else None
    )
    regret_ranked = sorted(PRIMARY_CRITERIA, key=lambda name: regret_map[name])
    regret_rank = regret_ranked.index(anchor_winner) + 1
    score_vectors = [
        np.asarray(anchor_results[name]["scores"], dtype=np.float64)
        for name in PRIMARY_CRITERIA
    ]
    maximum_score_gap = max(
        float(np.max(np.abs(score_vectors[first] - score_vectors[second])))
        for first in range(len(score_vectors) - 1)
        for second in range(first + 1, len(score_vectors))
    )
    substantial_threshold = 0.01
    level1 = (
        any(
            anchor_results[name]["metrics"]["kendall_tau_b"] is not None
            and anchor_results[name]["metrics"]["kendall_tau_b"] >= 0.8
            and anchor_results[name]["metrics"]["ace"] <= 0.10
            for name in PRIMARY_CRITERIA
        )
        and maximum_score_gap >= substantial_threshold
    )
    level2_best_spearman = bool(
        best_spearman is not None and best_spearman in credible_set
    )
    level2_lowest_regret = bool(
        lowest_regret is not None and lowest_regret in credible_set
    )
    level2_both = bool(
        best_spearman is not None
        and lowest_regret is not None
        and level2_best_spearman
        and level2_lowest_regret
    )
    practical_best_wga = float(
        selected_lookup["test_wga"].max()
    )
    anchor_regret_gap = float(
        selected_lookup.loc[anchor_winner, "test_selection_regret"]
        - selected_lookup["test_selection_regret"].min()
    )
    level3 = bool(spearman_rank == 1 and anchor_regret_gap <= 0.01)
    oracle_selected_gap = abs(
        float(anchor_selected["test_wga"]) - float(oracle_selected["test_wga"])
    )
    level4 = oracle_selected_gap <= 0.01
    anchorcal_quality = {
        "anchorcal_winner": anchor_winner,
        "anchorcal_winner_real_spearman_rank": spearman_rank,
        "anchorcal_winner_test_regret_rank": regret_rank,
        "best_real_spearman_criterion": best_spearman,
        "lowest_test_regret_criterion": lowest_regret,
        "credible_contains_best_spearman": bool(
            best_spearman is not None and best_spearman in credible_set
        ),
        "credible_contains_lowest_regret": bool(
            lowest_regret is not None and lowest_regret in credible_set
        ),
        "meta_spearman_negative_ace_vs_real_spearman": meta_spearman,
        "meta_spearman_negative_ace_vs_negative_regret": meta_regret,
        "distance_to_hindsight_test_best_wga": float(
            joined["test_wga"].max() - anchor_selected["test_wga"]
        ),
        "distance_to_best_practical_selected_wga": float(
            practical_best_wga - anchor_selected["test_wga"]
        ),
        "distance_to_oracle_selected_test_wga": oracle_selected_gap,
        "level_1_substantial_score_gap": maximum_score_gap,
        "level_1_substantial_score_gap_threshold": substantial_threshold,
        "success_levels": {
            "level_1_anchor_discrimination": bool(level1),
            "level_2_best_spearman_coverage": level2_best_spearman,
            "level_2_lowest_regret_coverage": level2_lowest_regret,
            "level_2_both_hindsight_definitions": level2_both,
            "level_3_correct_or_near_best_winner": bool(level3),
            "level_4_near_oracle_candidate_selection": bool(level4),
        },
    }
    atomic_write_json(analysis_root / "anchorcal_quality.json", _python(anchorcal_quality))
    summary = {
        "schema_version": "anchorcal-final-analysis-v2",
        "candidate_count": int(len(joined)),
        "run_count": int(joined["run_id"].nunique()),
        "test_wga_range": test_range,
        "oracle_wga_range": oracle_range,
        "competent_pool": competent_summary,
        "anchorcal_winner": anchor_winner,
        "anchorcal_selected": _python(anchor_selected.to_dict()),
        "oracle_selected_candidate": str(oracle_selected["candidate_id"]),
        "oracle_selected_test_wga": float(oracle_selected["test_wga"]),
        "test_posthoc_best_candidate": str(test_hindsight["candidate_id"]),
        "test_posthoc_best_wga": float(test_hindsight["test_wga"]),
        "selected_metric_intervals": intervals,
        "uncertainty_scope": {
            "sample_uncertainty": "four-group-stratified paired bootstrap",
            "training_seed_uncertainty": "not estimated; pilot uses one candidate-training seed",
            "replicates": replicates,
        },
        "anchorcal_quality": anchorcal_quality,
        "near_oracle_success": bool(level4),
        "selection_receipt": str(receipt.resolve()),
        "selection_receipt_sha256": sha256_file(receipt),
        "hidden_hdf5_sha256": dict(sorted(hidden_hdf5_sha256.items())),
    }
    atomic_write_json(analysis_root / "summary.json", _python(summary))
    _write_primary_tables(
        config,
        anchor_results,
        quality,
        selected_table,
        anchorcal_quality,
        oracle_selected,
        credible_set,
        analysis_root,
    )
    _make_figures(
        config,
        joined,
        anchor_results,
        selected_table,
        anchor_ace,
        real_spearman,
        real_regret,
        analysis_root,
    )
    summary_csv = {
        key: json.dumps(_python(value), sort_keys=True)
        if isinstance(value, (dict, list, tuple))
        else _python(value)
        for key, value in summary.items()
    }
    atomic_write_text(
        analysis_root / "anchorcal_summary.csv",
        pd.DataFrame([summary_csv]).to_csv(index=False, lineterminator="\n"),
    )
    diversity_failed = test_range < 0.02
    if diversity_failed:
        atomic_write_json(
            analysis_root / "candidate_diversity_failure.json",
            {"test_wga_range": test_range, "required": 0.02},
        )
    manifest_files = sorted(
        path
        for path in analysis_root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    atomic_write_json(
        analysis_root / "manifest.json",
        {
            "schema_version": "anchorcal-analysis-manifest-v1",
            "selection_receipt_sha256": sha256_file(receipt),
            "hidden_hdf5_sha256": dict(sorted(hidden_hdf5_sha256.items())),
            "files": {
                str(path.relative_to(analysis_root)): sha256_file(path)
                for path in manifest_files
            },
        },
    )
    if diversity_failed:
        if not debug:
            raise AuditFailure(
                f"candidate diversity gate failed: test-WGA range {test_range:.4f} < 0.0200"
            )
    return _python(summary)


def _write_primary_tables(
    config: dict[str, Any],
    anchor_results: dict[str, Any],
    quality: pd.DataFrame,
    selected: pd.DataFrame,
    anchorcal_quality: dict[str, Any],
    oracle_selected: pd.Series,
    credible_set: set[str],
    root: Path,
) -> None:
    output = Path(config["paths"]["output_root"])
    debug = bool(config["runtime"]["debug"])
    namespace = "debug/branches" if debug else "branches"
    anchor_namespace = "debug/anchors" if debug else "anchors"
    competence = json.loads(
        (output / anchor_namespace / "competence_intersection_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    branch_rows = []
    for branch in ("foreground", "background"):
        manifest = json.loads(
            (output / namespace / branch / "manifest.json").read_text(encoding="utf-8")
        )
        branch_rows.append(
            {
                "branch": branch,
                "biased_val_balanced_accuracy": manifest["biased_val_competence"]["point"],
                "biased_val_accuracy_ci_low": manifest["biased_val_competence"]["lower"],
                "biased_val_accuracy_ci_high": manifest["biased_val_competence"]["upper"],
                "calibration_nll_before": manifest["temperature"]["nll_before"],
                "calibration_nll_after": manifest["temperature"]["nll_after"],
                "temperature": manifest["temperature"]["temperature"],
                "intersection_contribution_count": competence["count"],
                "intersection_retained_fraction": competence["retained_fraction"],
                "notes": (
                    f"invalid_biased_val_fraction={manifest['invalid_biased_val_fraction']:.6f}; "
                    "temperature diagnostic only; raw logits define anchors"
                ),
            }
        )
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        tables / "branch_table.csv",
        pd.DataFrame(branch_rows).to_csv(index=False, lineterminator="\n"),
    )
    primary_anchor_names = [
        name for name in anchor_results if not name.endswith("_product")
    ]
    anchor_table = pd.DataFrame(
        [
            {
                "criterion": name,
                "eligible_for_decision": name in PRIMARY_CRITERIA,
                "winner": name == anchorcal_quality["anchorcal_winner"],
                "credible_set_member": name in credible_set,
                "ace_endpoint_clipping_floor_retained": True,
                **{
                    key: value
                    for key, value in anchor_results[name]["metrics"].items()
                    if key
                    in {
                        "ace",
                        "kendall_tau_b",
                        "spearman",
                        "pair_accuracy",
                        "adjacent_accuracy",
                        "violations",
                        "perfect_order",
                    }
                },
                "perfect_order_rate": anchor_results[name]["bootstrap"][
                    "perfect_order_rate"
                ]["mean"],
            }
            for name in primary_anchor_names
        ]
    )
    atomic_write_text(
        tables / "anchorcal_criterion_table.csv",
        anchor_table.to_csv(index=False, lineterminator="\n"),
    )
    product_table = pd.DataFrame(
        [
            {
                "criterion": name,
                "eligible_for_decision": False,
                **{
                    key: value
                    for key, value in anchor_results[name]["metrics"].items()
                    if key
                    in {
                        "ace",
                        "kendall_tau_b",
                        "spearman",
                        "pair_accuracy",
                        "adjacent_accuracy",
                        "violations",
                        "perfect_order",
                    }
                },
            }
            for name in anchor_results
            if name.endswith("_product")
        ]
    )
    atomic_write_text(
        tables / "anchorcal_product_diagnostics.csv",
        product_table.to_csv(index=False, lineterminator="\n"),
    )
    real = quality.merge(selected, on="criterion", how="outer")
    atomic_write_text(
        tables / "real_candidate_criterion_table.csv",
        real.to_csv(index=False, lineterminator="\n"),
    )
    decision_table = pd.DataFrame(
        [
            {
                "anchorcal_choice": anchorcal_quality["anchorcal_winner"],
                "credible_set": ",".join(sorted(credible_set)),
                "best_real_correlation_criterion": anchorcal_quality[
                    "best_real_spearman_criterion"
                ],
                "lowest_regret_criterion": anchorcal_quality[
                    "lowest_test_regret_criterion"
                ],
                "chosen_model_test_wga": selected.loc[
                    selected["criterion"] == anchorcal_quality["anchorcal_winner"],
                    "test_wga",
                ].iloc[0],
                "oracle_selected_candidate": oracle_selected["candidate_id"],
                "oracle_selected_test_wga": oracle_selected["test_wga"],
            }
        ]
    )
    atomic_write_text(
        tables / "anchorcal_decision_table.csv",
        decision_table.to_csv(index=False, lineterminator="\n"),
    )


def _make_figures(
    config: dict[str, Any],
    joined: pd.DataFrame,
    anchor_results: dict[str, Any],
    selected: pd.DataFrame,
    anchor_ace: np.ndarray,
    real_spearman: np.ndarray,
    real_regret: np.ndarray,
    root: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root = root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    for criterion in PRIMARY_CRITERIA:
        figure, axis = plt.subplots(figsize=(5, 4))
        axis.scatter(joined[criterion], joined["test_wga"], s=12, alpha=0.65)
        axis.set(xlabel=criterion, ylabel="Exploratory test WGA", title="Candidate criterion quality")
        figure.tight_layout()
        figure.savefig(figure_root / f"{criterion}_vs_test_wga.png", dpi=160)
        plt.close(figure)
    anchor_root = Path(config["paths"]["output_root"]) / (
        "debug/anchors" if config["runtime"]["debug"] else "anchors"
    )
    with np.load(anchor_root / "anchor_bootstrap_score_vectors.npz") as archive:
        lambdas = archive["lambdas"]
        figure, axis = plt.subplots(figsize=(6, 4))
        for criterion in anchor_results:
            scores = np.asarray(anchor_results[criterion]["scores"])
            draws = archive[criterion]
            low, high = np.percentile(draws, [2.5, 97.5], axis=0)
            axis.plot(lambdas, scores, marker="o", label=criterion)
            axis.fill_between(lambdas, low, high, alpha=0.12)
        axis.set(xlabel="Known foreground reliance lambda", ylabel="Criterion score")
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(figure_root / "anchor_scores_vs_lambda.png", dpi=160)
        plt.close(figure)
    criteria = list(anchor_results)
    figure, axes = plt.subplots(1, len(criteria), figsize=(4 * len(criteria), 3.5), sharex=True, sharey=True)
    if len(criteria) == 1:
        axes = [axes]
    for axis, criterion in zip(axes, criteria, strict=True):
        predicted = anchor_results[criterion]["metrics"]["ace_predictions"]
        axis.scatter(lambdas, predicted, s=20)
        axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
        axis.set(title=criterion, xlabel="True lambda", ylabel="Cross-fitted predicted lambda")
    figure.tight_layout()
    figure.savefig(figure_root / "predicted_lambda_vs_true.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(selected["criterion"], selected["test_wga"])
    axis.tick_params(axis="x", rotation=30)
    axis.set_ylabel("Selected exploratory test WGA")
    figure.tight_layout()
    figure.savefig(figure_root / "selected_model_comparison.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    valid = np.isfinite(real_spearman)
    axes[0].scatter(-anchor_ace[valid], real_spearman[valid])
    axes[0].set(xlabel="-Anchor ACE", ylabel="Real test-WGA Spearman")
    axes[1].scatter(-anchor_ace, -real_regret)
    axes[1].set(xlabel="-Anchor ACE", ylabel="-Test selection regret")
    for index, name in enumerate(PRIMARY_CRITERIA):
        if valid[index]:
            axes[0].annotate(name, (-anchor_ace[index], real_spearman[index]), fontsize=6)
        axes[1].annotate(name, (-anchor_ace[index], -real_regret[index]), fontsize=6)
    figure.tight_layout()
    figure.savefig(figure_root / "anchorcal_vs_real_quality.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 4))
    for run_id, group in joined.groupby("run_id"):
        axis.plot(group["epoch"], group["test_wga"], label=run_id, alpha=0.75)
    markers = ("o", "s", "^", "D")
    for marker, criterion in zip(markers, PRIMARY_CRITERIA, strict=True):
        row = selected.loc[selected["criterion"] == criterion].iloc[0]
        axis.scatter(row["epoch"], row["test_wga"], marker=marker, s=55, label=f"selected:{criterion}")
    axis.set(xlabel="Epoch", ylabel="Exploratory test WGA")
    axis.legend(fontsize=5, ncol=2)
    figure.tight_layout()
    figure.savefig(figure_root / "candidate_trajectories.png", dpi=160)
    plt.close(figure)
    competence = json.loads(
        (anchor_root / "competence_intersection_manifest.json").read_text(encoding="utf-8")
    )
    branch_namespace = "debug/branches" if config["runtime"]["debug"] else "branches"
    branch_accuracy = []
    for branch in ("foreground", "background"):
        manifest = json.loads(
            (
                Path(config["paths"]["output_root"])
                / branch_namespace
                / branch
                / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        branch_accuracy.append(manifest["biased_val_competence"]["point"])
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].bar(["foreground", "background"], branch_accuracy)
    axes[0].axhline(0.5, linestyle="--", color="black", linewidth=1)
    axes[0].set(ylabel="Biased-val balanced accuracy", ylim=(0, 1))
    axes[1].bar(list(competence["per_class_count"]), list(competence["per_class_count"].values()))
    axes[1].set(xlabel="Class", ylabel="Competence intersection count", title=f"Total={competence['count']}")
    figure.tight_layout()
    figure.savefig(figure_root / "competence_subset_diagnostics.png", dpi=160)
    plt.close(figure)
