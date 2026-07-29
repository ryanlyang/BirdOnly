"""Leakage-isolated Phase 6 multi-seed selection, diagnostics, and freeze."""

from __future__ import annotations

import csv
import json
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw

from setv.candidate.selectors import load_selector_inputs
from setv.candidate.train import verify_candidate
from setv.errors import ArtifactExistsError, DataValidationError
from setv.experts.train_object import _artifact_manifest, _load_phase0_config
from setv.phase0 import BASE_ARTIFACT_MANIFEST, verify_phase0
from setv.fusion.core import continuous_setv, within_class_percentile
from setv.ula.config import resolved_config
from setv.ula.proxy import load_ula_proxy_scores, verify_ula_proxy
from setv.ula.validation import select_ula_epoch
from setv.utils.hashing import sha256_file, sha256_json
from setv.utils.io import write_json

CheckpointMaterializer = Callable[[Path, int, Path], dict[str, Any]]


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    return ranks


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x, y = _average_ranks(left), _average_ranks(right)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def kendall_tau_b(left: np.ndarray, right: np.ndarray, tolerance: float = 1e-12) -> float:
    concordant = discordant = tie_left = tie_right = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            dx = float(left[first]) - float(left[second])
            dy = float(right[first]) - float(right[second])
            x_tie, y_tie = abs(dx) <= tolerance, abs(dy) <= tolerance
            if x_tie and y_tie:
                continue
            if x_tie:
                tie_left += 1
            elif y_tie:
                tie_right += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + tie_left)
        * (concordant + discordant + tie_right)
    )
    return float((concordant - discordant) / denominator) if denominator else 0.0


def pairwise_ranking_accuracy(
    selector: np.ndarray, oracle: np.ndarray, tolerance: float = 1e-8
) -> dict[str, Any]:
    agreeing = comparable = selector_ties = oracle_ties = 0
    for first in range(len(selector)):
        for second in range(first + 1, len(selector)):
            ds = float(selector[first]) - float(selector[second])
            do = float(oracle[first]) - float(oracle[second])
            if abs(do) <= tolerance:
                oracle_ties += 1
                continue
            if abs(ds) <= tolerance:
                selector_ties += 1
                continue
            comparable += 1
            agreeing += int(ds * do > 0)
    return {
        "accuracy": float(agreeing / comparable) if comparable else 0.0,
        "agreeing_pairs": agreeing,
        "comparable_pairs": comparable,
        "selector_ties_excluded": selector_ties,
        "oracle_ties_excluded": oracle_ties,
    }


def _selector_scalar(name: str, metrics: dict[str, Any]) -> float:
    if name == "ordinary":
        return float(metrics["accuracy"])
    if name == "ula":
        return float(metrics["u_balanced_accuracy"])
    if name.startswith("hard_pseudogroup."):
        return float(metrics["worst_nonempty_proxy_group_accuracy"])
    if name.endswith(".hard"):
        return float(metrics["class_balanced_accuracy"])
    return float(metrics["setv_score"])


def _load_seed(
    candidate_dir: Path, proxy: dict[str, np.ndarray], tolerance: float
) -> dict[str, Any]:
    verify_candidate(candidate_dir, load_checkpoints=False)
    receipt = _json(candidate_dir / "phase5_receipt.json")
    selection = _json(candidate_dir / "selection" / "selection_receipt.json")
    scalar_epochs = _json(candidate_dir / "biased_val" / "selector_metrics.json")[
        "epochs"
    ]
    oracle_epochs = _json(candidate_dir / "analysis_only" / "oracle_metrics.json")[
        "epochs"
    ]
    oracle_best = _json(
        candidate_dir / "analysis_only" / "oracle_selector.json"
    )["best"]
    with np.load(
        candidate_dir / receipt["biased_val_predictions"]["path"], allow_pickle=False
    ) as archive:
        trajectory = {key: archive[key] for key in archive.files}
    if not np.array_equal(
        trajectory["sample_id"].astype(str), proxy["sample_id"].astype(str)
    ) or not np.array_equal(trajectory["true_label"], proxy["true_label"]):
        raise DataValidationError("uLA proxy does not align with candidate biased_val")
    ordinary_accuracy = np.asarray(
        [item["ordinary"]["accuracy"] for item in scalar_epochs], dtype=np.float64
    )
    ordinary_loss = np.asarray(
        [item["ordinary"]["loss"] for item in scalar_epochs], dtype=np.float64
    )
    ula = select_ula_epoch(
        trajectory["candidate_correct"],
        proxy["ula_proxy_predicted_class"],
        trajectory["true_label"],
        ordinary_accuracy,
        ordinary_loss,
        tolerance=tolerance,
    )
    curves: dict[str, np.ndarray] = {
        "ordinary": ordinary_accuracy,
        "ula": np.asarray(
            [item["u_balanced_accuracy"] for item in ula["curve"]],
            dtype=np.float64,
        ),
    }
    available_names = set(selection["realistic_selectors"])
    for name in sorted(available_names - {"ordinary"}):
        curves[name] = np.asarray(
            [_selector_scalar(name, item["selectors"][name]) for item in scalar_epochs],
            dtype=np.float64,
        )
    oracle_curve = np.asarray(
        [item["worst_group_accuracy"] for item in oracle_epochs], dtype=np.float64
    )
    selected = {
        name: int(value["epoch"])
        for name, value in selection["realistic_selectors"].items()
    }
    selected["ula"] = int(ula["best"]["epoch"])
    selected["oracle"] = int(oracle_best["epoch"])
    metrics = {}
    oracle_max = float(oracle_curve.max())
    for name, epoch in selected.items():
        score_curve = oracle_curve if name == "oracle" else curves[name]
        pairwise = pairwise_ranking_accuracy(score_curve, oracle_curve, tolerance)
        metrics[name] = {
            "selected_epoch": epoch,
            "oracle_val_wga": float(oracle_curve[epoch - 1]),
            "oracle_selection_regret": float(oracle_max - oracle_curve[epoch - 1]),
            "spearman": spearman_correlation(score_curve, oracle_curve),
            "kendall_tau_b": kendall_tau_b(score_curve, oracle_curve),
            "pairwise_epoch_ranking_accuracy": pairwise["accuracy"],
            "pairwise_details": pairwise,
        }
    return {
        "seed": int(receipt["seed"]),
        "candidate_dir": str(candidate_dir),
        "phase0_dir": receipt["phase0_dir"],
        "selection_receipt_sha256": sha256_file(
            candidate_dir / "selection" / "selection_receipt.json"
        ),
        "ula": ula,
        "selected": selected,
        "metrics": metrics,
        "curves": curves,
        "oracle_curve": oracle_curve,
        "oracle_epochs": oracle_epochs,
        "candidate_logits": trajectory["candidate_logits"].astype(np.float32),
        "candidate_correct": trajectory["candidate_correct"].astype(np.uint8),
    }


def _aggregate(seeds: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    names = sorted(set.intersection(*(set(item["metrics"]) for item in seeds)))
    output = {}
    for name in names:
        values = [item["metrics"][name] for item in seeds]
        output[name] = {
            "candidate_seed_count": len(values),
            "mean_oracle_selection_regret": float(
                np.mean([item["oracle_selection_regret"] for item in values])
            ),
            "worst_oracle_selection_regret": float(
                np.max([item["oracle_selection_regret"] for item in values])
            ),
            "mean_spearman": float(np.mean([item["spearman"] for item in values])),
            "mean_kendall": float(
                np.mean([item["kendall_tau_b"] for item in values])
            ),
            "mean_pairwise_ranking_accuracy": float(
                np.mean(
                    [item["pairwise_epoch_ranking_accuracy"] for item in values]
                )
            ),
            "selected_epochs": [item["selected_epoch"] for item in values],
        }
    return output


def _curve_metrics(
    curve: np.ndarray, oracle_curve: np.ndarray, tolerance: float
) -> dict[str, Any]:
    best_epoch = int(np.flatnonzero(curve >= curve.max() - tolerance)[0] + 1)
    oracle_max = float(oracle_curve.max())
    pairwise = pairwise_ranking_accuracy(curve, oracle_curve, tolerance)
    return {
        "selected_epoch": best_epoch,
        "oracle_val_wga": float(oracle_curve[best_epoch - 1]),
        "oracle_selection_regret": float(
            oracle_max - oracle_curve[best_epoch - 1]
        ),
        "spearman": spearman_correlation(curve, oracle_curve),
        "kendall_tau_b": kendall_tau_b(curve, oracle_curve),
        "pairwise_epoch_ranking_accuracy": pairwise["accuracy"],
        "pairwise_details": pairwise,
    }


def _background_confidence_baselines(
    seeds: list[dict[str, Any]],
    selector_inputs: dict[str, Any],
    labels: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    """Evaluate a diagnostic background-confidence-only weighting baseline."""

    per_seed: dict[str, dict[str, Any]] = {}
    aggregate: dict[str, Any] = {}
    for expert, source in selector_inputs.items():
        background_margin = np.asarray(
            source.fusion.get(
                "background_true_class_margin",
                source.fusion["rank"]["background_percentile"],
            ),
            dtype=np.float64,
        )
        # Low true-class background margin means stronger background conflict.
        weights = within_class_percentile(-background_margin, labels)
        expert_metrics = []
        for seed in seeds:
            curve = np.asarray(
                [
                    continuous_setv(logits, labels, weights)["setv_score"]
                    for logits in seed["candidate_logits"]
                ],
                dtype=np.float64,
            )
            metrics = _curve_metrics(curve, seed["oracle_curve"], tolerance)
            metrics["curve"] = curve.tolist()
            per_seed.setdefault(str(seed["seed"]), {})[expert] = metrics
            expert_metrics.append(metrics)
        aggregate[expert] = {
            "mean_oracle_selection_regret": float(
                np.mean(
                    [value["oracle_selection_regret"] for value in expert_metrics]
                )
            ),
            "worst_oracle_selection_regret": float(
                np.max(
                    [value["oracle_selection_regret"] for value in expert_metrics]
                )
            ),
            "mean_spearman": float(
                np.mean([value["spearman"] for value in expert_metrics])
            ),
            "selected_epochs": [
                value["selected_epoch"] for value in expert_metrics
            ],
            "definition": (
                "within-class percentile of negative background true-class "
                "margin, scored with the locked SETV alpha ensemble"
            ),
        }
    return {"per_seed": per_seed, "aggregate": aggregate}


def _joint_choice(
    aggregate: dict[str, dict[str, Any]],
    expert_metadata: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    names = [
        f"setv.{expert}.{fusion}"
        for expert in ("exact", "sanitized", "set")
        for fusion in ("hard", "rank", "logistic")
        if f"setv.{expert}.{fusion}" in aggregate
    ]
    if not names:
        raise DataValidationError("No available SETV expert-fusion configurations")
    ranked = sorted(
        names,
        key=lambda name: (
            aggregate[name]["mean_oracle_selection_regret"],
            aggregate[name]["worst_oracle_selection_regret"],
            -aggregate[name]["mean_spearman"],
            -aggregate[name]["mean_kendall"],
            -aggregate[name]["mean_pairwise_ranking_accuracy"],
            name,
        ),
    )
    best = ranked[0]
    regret_threshold = float(config["selection"]["regret_tie_threshold"])
    spearman_threshold = float(config["selection"]["spearman_tie_threshold"])
    tied = [
        name
        for name in ranked
        if abs(
            aggregate[name]["mean_oracle_selection_regret"]
            - aggregate[best]["mean_oracle_selection_regret"]
        )
        <= regret_threshold
        and abs(
            aggregate[name]["mean_spearman"] - aggregate[best]["mean_spearman"]
        )
        <= spearman_threshold
    ]
    metadata = expert_metadata
    tied_experts = sorted({name.split(".")[1] for name in tied})
    leakage_comparable = all(
        metadata[expert].get("leakage_balanced_accuracy") is not None
        for expert in tied_experts
    )
    stability_comparable = all(
        metadata[expert].get("view_margin_std") is not None
        for expert in tied_experts
    )

    def secondary(name: str):
        expert = name.split(".")[1]
        item = metadata[expert]
        leakage = item.get("leakage_balanced_accuracy")
        stability = item.get("view_margin_std")
        return (
            float(leakage) if leakage_comparable else 0.0,
            float(stability) if stability_comparable else 0.0,
            int(item["simplicity_rank"]),
            0 if expert == "sanitized" else 1,
            ranked.index(name),
        )

    winner = min(tied, key=secondary) if len(tied) > 1 else best
    return {
        "selected_configuration": winner,
        "background_expert": winner.split(".")[1],
        "fusion": winner.split(".")[2],
        "initial_metric_ranking": ranked,
        "effectively_tied_finalists": tied,
        "tie_rule": {
            "regret_threshold": regret_threshold,
            "spearman_threshold": spearman_threshold,
            "secondary_order": [
                "lower_leakage",
                "greater_view_stability",
                "simpler_implementation",
                "prefer_sanitized",
            ],
            "expert_metadata": metadata,
            "leakage_metric_applied": leakage_comparable,
            "stability_metric_applied": stability_comparable,
        },
        "selected_metrics": aggregate[winner],
    }


def _derive_expert_metadata(
    selector_inputs: dict[str, Any], config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Derive secondary tie evidence from already verified production artifacts.

    Exact fill and the set expert do not have the held-out sanitized-mask auditor,
    so that leakage number is deliberately unavailable for them. Exact fill is a
    deterministic single view, so an eight-view stability statistic would not be
    comparable either. Missing values are recorded and cause the corresponding
    secondary criterion to be skipped.
    """

    ranks = config["expert_metadata"]["simplicity_rank"]
    output: dict[str, dict[str, Any]] = {}
    for name in ("exact", "sanitized", "set"):
        source = selector_inputs[name]
        scores = source.background_scores
        item: dict[str, Any] = {
            "leakage_balanced_accuracy": None,
            "leakage_source": None,
            "leakage_unavailable_reason": (
                "no comparable held-out mask-geometry auditor artifact"
            ),
            "view_margin_std": None,
            "stability_source": None,
            "stability_unavailable_reason": (
                "verified source score payload unavailable"
                if scores is None
                else "expert has no comparable stochastic eight-view statistic"
            ),
            "simplicity_rank": int(ranks[name]),
            "derivation": "verified_artifacts",
        }
        if scores is not None:
            stability_key = {
                "sanitized": "background_sanitized_margin_std",
                "set": "background_set_margin_std",
            }.get(name)
            if stability_key is not None and stability_key in scores:
                values = np.asarray(scores[stability_key], dtype=np.float64)
                item["view_margin_std"] = float(values.mean())
                item["stability_source"] = stability_key
                item["stability_unavailable_reason"] = None
        if name == "sanitized" and source.expert_dir is not None:
            receipt_path = source.expert_dir / "phase3_sanitized_receipt.json"
            if receipt_path.is_file():
                expert_receipt = _json(receipt_path)
                bank_dir = Path(expert_receipt["mask_bank_dir"])
                bank_receipt_path = bank_dir / "sanitized_mask_bank_receipt.json"
                if bank_receipt_path.is_file():
                    bank_receipt = _json(bank_receipt_path)
                    audit_path = bank_dir / bank_receipt["leakage_audit"]["path"]
                    audit = _json(audit_path)
                    measured = [
                        float(value["heldout_mask_balanced_accuracy"])
                        for value in audit.get("auditors", {}).values()
                        if value.get("heldout_mask_balanced_accuracy") is not None
                    ]
                    if measured:
                        item["leakage_balanced_accuracy"] = max(measured)
                        item["leakage_source"] = {
                            "path": str(audit_path),
                            "sha256": sha256_file(audit_path),
                            "aggregation": "maximum_across_heldout_mask_auditors",
                            "auditor_count": len(measured),
                        }
                        item["leakage_unavailable_reason"] = None
        output[name] = item
    return output


def _expert_dominance(
    seeds: list[dict[str, Any]],
) -> dict[str, Any]:
    experts = ("exact", "sanitized", "set")
    fusions = ("hard", "rank", "logistic")
    output = {}
    for candidate in experts:
        dominates = []
        for other in experts:
            if candidate == other:
                continue
            comparisons = []
            for seed in seeds:
                for fusion in fusions:
                    left = seed["metrics"].get(f"setv.{candidate}.{fusion}")
                    right = seed["metrics"].get(f"setv.{other}.{fusion}")
                    if left is not None and right is not None:
                        comparisons.append(
                            left["oracle_selection_regret"]
                            <= right["oracle_selection_regret"] + 1e-12
                        )
            if comparisons and all(comparisons):
                dominates.append(other)
        output[candidate] = {
            "dominates_across_every_available_fusion_and_seed": dominates,
            "general_superiority_claim_supported": len(dominates) == len(experts) - 1,
        }
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _svg_bar(path: Path, labels: list[str], values: list[float], title: str) -> None:
    width, height, margin = 1000, 500, 70
    maximum = max(values) if values else 1.0
    maximum = maximum if maximum > 0 else 1.0
    bar_width = (width - 2 * margin) / max(1, len(values))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="18">{title}</text>',
    ]
    for index, (label, value) in enumerate(zip(labels, values)):
        x = margin + index * bar_width + 4
        bar_height = (height - 2 * margin) * value / maximum
        y = height - margin - bar_height
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(1,bar_width-8):.1f}" '
            f'height="{bar_height:.1f}" fill="#3973ac"/>'
        )
        parts.append(
            f'<text x="{x + bar_width/2 - 4:.1f}" y="{height-margin+14}" '
            f'text-anchor="end" transform="rotate(-45 {x + bar_width/2 - 4:.1f},'
            f'{height-margin+14})" font-size="10">{label}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _svg_scatter(
    path: Path, rows: list[dict[str, Any]], *, title: str
) -> None:
    width, height, margin = 900, 600, 60
    x_values = [float(row["selector_score"]) for row in rows]
    y_values = [float(row["oracle_wga"]) for row in rows]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_span, y_span = max(1e-12, x_max - x_min), max(1e-12, y_max - y_min)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="18">{title}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" '
        f'y2="{height-margin}" stroke="black"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" '
        f'y2="{height-margin}" stroke="black"/>',
    ]
    for row in rows:
        x = margin + (float(row["selector_score"]) - x_min) / x_span * (
            width - 2 * margin
        )
        y = height - margin - (float(row["oracle_wga"]) - y_min) / y_span * (
            height - 2 * margin
        )
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.5" fill="#3973ac" '
            f'fill-opacity="0.55"/>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _svg_series(
    path: Path,
    series: dict[str, list[tuple[float, float]]],
    *,
    title: str,
    x_label: str,
    y_label: str,
    points: bool = False,
) -> None:
    usable = {
        label: values for label, values in series.items() if len(values) > 0
    }
    if not usable:
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="300">'
            '<rect width="100%" height="100%" fill="white"/>'
            f'<text x="400" y="35" text-anchor="middle" font-size="18">{title}</text>'
            '<text x="400" y="155" text-anchor="middle" font-size="15">'
            "No diagnostic values were available</text></svg>",
            encoding="utf-8",
        )
        return
    width, height, margin = 1000, 620, 75
    colors = ("#3973ac", "#c94c4c", "#3a923a", "#8e5ea2", "#d48806", "#444444")
    all_x = [x for values in usable.values() for x, _ in values]
    all_y = [y for values in usable.values() for _, y in values]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_span = max(1e-12, x_max - x_min)
    y_span = max(1e-12, y_max - y_min)

    def project(value_x: float, value_y: float) -> tuple[float, float]:
        return (
            margin + (value_x - x_min) / x_span * (width - 2 * margin),
            height - margin - (value_y - y_min) / y_span * (height - 2 * margin),
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="18">{title}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" '
        f'y2="{height-margin}" stroke="black"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" '
        f'y2="{height-margin}" stroke="black"/>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle">{x_label}</text>',
        f'<text x="20" y="{height/2}" text-anchor="middle" '
        f'transform="rotate(-90 20,{height/2})">{y_label}</text>',
    ]
    for index, (label, values) in enumerate(usable.items()):
        color = colors[index % len(colors)]
        projected = [project(x, y) for x, y in sorted(values)]
        if not points:
            coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in projected)
            parts.append(
                f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
                'stroke-width="2"/>'
            )
        for x, y in projected:
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.8" fill="{color}" '
                'fill-opacity="0.65"/>'
            )
        legend_y = 52 + 18 * index
        parts.append(
            f'<rect x="{width-245}" y="{legend_y-10}" width="12" height="12" '
            f'fill="{color}"/><text x="{width-226}" y="{legend_y}" '
            f'font-size="12">{label}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _histogram_series(values: np.ndarray, bins: int = 20) -> list[tuple[float, float]]:
    counts, edges = np.histogram(np.asarray(values, dtype=np.float64), bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return [(float(x), float(y)) for x, y in zip(centers, counts)]


def _contact_sheet(
    path: Path,
    phase0_dir: Path,
    sample_ids: np.ndarray,
    scores: np.ndarray,
    *,
    count: int = 16,
) -> None:
    phase0_config = _load_phase0_config(phase0_dir)
    dataset_root = Path(phase0_config["data"]["dataset_root"])
    manifest = pd.read_csv(
        phase0_dir / "splits" / "waterbirds95_biased_val.csv",
        dtype={"sample_id": str},
    ).set_index("sample_id")
    chosen = np.argsort(scores)[::-1][: min(count, len(scores))]
    thumb, label_height, columns = 160, 24, 4
    rows = math.ceil(len(chosen) / columns)
    sheet = Image.new("RGB", (columns * thumb, rows * (thumb + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for slot, index in enumerate(chosen):
        sample_id = str(sample_ids[index])
        with Image.open(dataset_root / str(manifest.loc[sample_id, "img_filename"])) as source:
            image = source.convert("RGB")
            image.thumbnail((thumb, thumb))
        left = (slot % columns) * thumb
        top = (slot // columns) * (thumb + label_height)
        sheet.paste(image, (left + (thumb - image.width) // 2, top))
        draw.text(
            (left + 3, top + thumb + 3),
            f"{sample_id} q={float(scores[index]):.3f}",
            fill="black",
        )
    sheet.save(path)


def _diagnostics(
    staging: Path,
    config: dict[str, Any],
    seeds: list[dict[str, Any]],
    selector_inputs: dict[str, Any],
) -> dict[str, Any]:
    diagnostic_root = staging / "diagnostics"
    gallery_root = diagnostic_root / "galleries"
    gallery_root.mkdir(parents=True, exist_ok=True)
    expert_summary = {}
    for name, source in selector_inputs.items():
        fusion = source.fusion
        rank = fusion["rank"]
        object_margin = np.asarray(
            fusion.get("object_true_class_margin", rank["object_percentile"]),
            dtype=np.float64,
        )
        background_margin = np.asarray(
            fusion.get(
                "background_true_class_margin", rank["background_percentile"]
            ),
            dtype=np.float64,
        )
        rows = [
            {
                "sample_id": sample_id,
                "true_label": int(label),
                "hard_target": int(hard),
                "object_margin": float(object_margin[index]),
                "background_margin": float(background_margin[index]),
                "object_percentile": float(object_rank),
                "background_percentile": float(background_rank),
                "q_rank": float(q_rank),
                "q_logistic": (
                    float(fusion["logistic"]["q_logistic"][index])
                    if fusion["logistic"]["available"]
                    else ""
                ),
            }
            for index, (
                sample_id,
                label,
                hard,
                object_rank,
                background_rank,
                q_rank,
            ) in enumerate(
                zip(
                    fusion["sample_id"],
                    fusion["true_label"],
                    fusion["hard_target"],
                    rank["object_percentile"],
                    rank["background_percentile"],
                    rank["q_rank"],
                )
            )
        ]
        _write_csv(diagnostic_root / f"{name}_fusion_examples.csv", rows)
        histogram_rows = []
        for margin_name, margin_values in (
            ("object", object_margin),
            ("background", background_margin),
        ):
            counts, edges = np.histogram(margin_values, bins=20)
            histogram_rows.extend(
                {
                    "expert": margin_name,
                    "bin_left": float(edges[index]),
                    "bin_right": float(edges[index + 1]),
                    "count": int(count),
                }
                for index, count in enumerate(counts)
            )
        _write_csv(diagnostic_root / f"{name}_margin_histograms.csv", histogram_rows)
        _svg_series(
            staging / "plots" / f"{name}_expert_margin_histograms.svg",
            {
                "object margin": _histogram_series(object_margin),
                "background margin": _histogram_series(background_margin),
            },
            title=f"{name.capitalize()} expert margin distributions",
            x_label="true-class margin",
            y_label="count",
        )
        _svg_series(
            staging / "plots" / f"{name}_object_vs_background_margin.svg",
            {
                "biased-validation images": [
                    (float(x), float(y))
                    for x, y in zip(object_margin, background_margin)
                ]
            },
            title=f"{name.capitalize()}: object versus background margin",
            x_label="object true-class margin",
            y_label="background true-class margin",
            points=True,
        )
        fusion_distributions = {
            "rank": _histogram_series(rank["q_rank"]),
        }
        if fusion["logistic"]["available"]:
            fusion_distributions["logistic OOF"] = _histogram_series(
                fusion["logistic"]["q_logistic"]
            )
        _svg_series(
            staging / "plots" / f"{name}_fusion_score_distributions.svg",
            fusion_distributions,
            title=f"{name.capitalize()} fusion-score distributions",
            x_label="fusion score",
            y_label="count",
        )
        _contact_sheet(
            gallery_root / f"{name}_top_rank.png",
            Path(config["phase0_dir"]),
            fusion["sample_id"],
            rank["q_rank"],
        )
        if fusion["logistic"]["available"]:
            _contact_sheet(
                gallery_root / f"{name}_top_logistic.png",
                Path(config["phase0_dir"]),
                fusion["sample_id"],
                fusion["logistic"]["q_logistic"],
            )
        hard = fusion["hard_target"].astype(bool)
        labels = fusion["true_label"]
        expert_summary[name] = {
            "hard_counts_by_class": {
                str(label): int(hard[labels == label].sum()) for label in (0, 1)
            },
            "hard_valid": bool(fusion["hard_valid"]),
            "correctness_contingency": {
                f"object_correct={int(object_ok)},background_correct={int(background_ok)}": int(
                    np.logical_and(
                        (object_margin > 0) == object_ok,
                        (background_margin > 0) == background_ok,
                    ).sum()
                )
                for object_ok in (False, True)
                for background_ok in (False, True)
            },
            "margin_standard_deviation": {
                "object": float(object_margin.std()),
                "background": float(background_margin.std()),
            },
            "rank_distribution": {
                "minimum": float(rank["q_rank"].min()),
                "mean": float(rank["q_rank"].mean()),
                "maximum": float(rank["q_rank"].max()),
            },
            "logistic": {
                key: value
                for key, value in fusion["logistic"].items()
                if key
                in {
                    "available",
                    "diagnostics",
                    "reason",
                    "target_counts",
                    "n_folds_used",
                    "n_repeats",
                }
            },
        }
    write_json(diagnostic_root / "expert_fusion_summary.json", expert_summary)
    curve_rows = []
    for seed in seeds:
        for name, curve in seed["curves"].items():
            for epoch, (score, oracle) in enumerate(
                zip(curve, seed["oracle_curve"]), start=1
            ):
                curve_rows.append(
                    {
                        "candidate_seed": seed["seed"],
                        "selector": name,
                        "epoch": epoch,
                        "selector_score": float(score),
                        "oracle_wga": float(oracle),
                    }
                )
    _write_csv(diagnostic_root / "selector_oracle_curves.csv", curve_rows)
    _svg_scatter(
        staging / "plots" / "selector_score_vs_oracle_wga.svg",
        curve_rows,
        title="Selector score versus oracle validation WGA",
    )
    decile_rows = []
    alpha_rows = []
    hardness_rows = []
    high_quantile = float(config["kill_criteria"]["high_score_quantile"])
    high_tolerance = float(
        config["kill_criteria"]["high_score_accuracy_tolerance"]
    )
    for seed in seeds:
        candidate_dir = Path(seed["candidate_dir"])
        phase5 = _json(candidate_dir / "phase5_receipt.json")
        with np.load(
            candidate_dir / phase5["biased_val_predictions"]["path"],
            allow_pickle=False,
        ) as archive:
            correct = archive["candidate_correct"].astype(np.float64)
            predictions = archive["candidate_logits"].argmax(-1)
        disagreement = 1.0 - np.maximum(
            (predictions == 0).mean(axis=0), (predictions == 1).mean(axis=0)
        )
        mean_accuracy = correct.mean(axis=0)
        for expert, source in selector_inputs.items():
            score_items = [("rank", source.fusion["rank"]["q_rank"])]
            if source.fusion["logistic"]["available"]:
                score_items.append(
                    ("logistic", source.fusion["logistic"]["q_logistic"])
                )
            for fusion_name, score_values in score_items:
                threshold = float(np.quantile(score_values, high_quantile))
                high = np.asarray(score_values) >= threshold
                per_epoch_high = correct[:, high].mean(axis=1)
                per_epoch_all = correct.mean(axis=1)
                comparisons = per_epoch_high <= per_epoch_all + high_tolerance
                hardness_rows.append(
                    {
                        "candidate_seed": seed["seed"],
                        "expert": expert,
                        "fusion": fusion_name,
                        "high_score_quantile": high_quantile,
                        "high_score_threshold": threshold,
                        "high_score_sample_count": int(high.sum()),
                        "epochs_high_score_at_or_below_overall": int(
                            comparisons.sum()
                        ),
                        "candidate_epoch_count": int(len(comparisons)),
                        "fraction_epochs_high_score_at_or_below_overall": float(
                            comparisons.mean()
                        ),
                        "simply_hard_for_every_candidate": bool(
                            comparisons.all()
                        ),
                    }
                )
                order = np.argsort(score_values, kind="mergesort")
                deciles = np.empty(len(order), dtype=np.int64)
                deciles[order] = np.minimum(
                    9, np.floor(np.arange(len(order)) * 10 / len(order)).astype(int)
                )
                for decile in range(10):
                    indices = deciles == decile
                    if indices.any():
                        decile_rows.append(
                            {
                                "candidate_seed": seed["seed"],
                                "expert": expert,
                                "fusion": fusion_name,
                                "decile": decile + 1,
                                "sample_count": int(indices.sum()),
                                "mean_candidate_disagreement": float(
                                    disagreement[indices].mean()
                                ),
                                "mean_candidate_accuracy": float(
                                    mean_accuracy[indices].mean()
                                ),
                            }
                        )
        scalar_epochs = _json(
            candidate_dir / "biased_val" / "selector_metrics.json"
        )["epochs"]
        representative = sorted(
            {
                1,
                50,
                int(seed["selected"]["ordinary"]),
                int(seed["selected"]["oracle"]),
            }
        )
        for epoch in representative:
            selectors = scalar_epochs[epoch - 1]["selectors"]
            for name, metrics in selectors.items():
                if name.startswith("setv.") and not name.endswith(".hard"):
                    for alpha, accuracy in metrics.get(
                        "alpha_accuracy", {}
                    ).items():
                        alpha_rows.append(
                            {
                                "candidate_seed": seed["seed"],
                                "epoch": epoch,
                                "selector": name,
                                "alpha": float(alpha),
                                "weighted_accuracy": float(accuracy),
                                "weighted_loss": float(
                                    metrics["alpha_loss"][alpha]
                                ),
                            }
                        )
    _write_csv(diagnostic_root / "candidate_fusion_deciles.csv", decile_rows)
    _write_csv(diagnostic_root / "representative_alpha_curves.csv", alpha_rows)
    _write_csv(diagnostic_root / "high_score_hardness.csv", hardness_rows)
    decile_series_disagreement: dict[str, list[tuple[float, float]]] = {}
    decile_series_accuracy: dict[str, list[tuple[float, float]]] = {}
    for expert in ("exact", "sanitized", "set"):
        for fusion_name in ("rank", "logistic"):
            selected_rows = [
                row
                for row in decile_rows
                if row["expert"] == expert and row["fusion"] == fusion_name
            ]
            if not selected_rows:
                continue
            label = f"{expert}.{fusion_name}"
            available_deciles = sorted(
                {int(row["decile"]) for row in selected_rows}
            )
            decile_series_disagreement[label] = [
                (
                    float(decile),
                    float(
                        np.mean(
                            [
                                row["mean_candidate_disagreement"]
                                for row in selected_rows
                                if row["decile"] == decile
                            ]
                        )
                    ),
                )
                for decile in available_deciles
            ]
            decile_series_accuracy[label] = [
                (
                    float(decile),
                    float(
                        np.mean(
                            [
                                row["mean_candidate_accuracy"]
                                for row in selected_rows
                                if row["decile"] == decile
                            ]
                        )
                    ),
                )
                for decile in available_deciles
            ]
    _svg_series(
        staging / "plots" / "candidate_disagreement_by_fusion_decile.svg",
        decile_series_disagreement,
        title="Candidate disagreement by fusion-score decile",
        x_label="fusion-score decile",
        y_label="mean disagreement",
    )
    _svg_series(
        staging / "plots" / "candidate_accuracy_by_fusion_decile.svg",
        decile_series_accuracy,
        title="Candidate accuracy by fusion-score decile",
        x_label="fusion-score decile",
        y_label="mean accuracy",
    )
    alpha_series: dict[str, list[tuple[float, float]]] = {}
    for row in alpha_rows:
        label = (
            f"seed={row['candidate_seed']} epoch={row['epoch']} "
            f"{row['selector']}"
        )
        alpha_series.setdefault(label, []).append(
            (float(row["alpha"]), float(row["weighted_accuracy"]))
        )
    _svg_series(
        staging / "plots" / "representative_setv_alpha_curves.svg",
        alpha_series,
        title="SETV alpha curves at representative epochs",
        x_label="alpha",
        y_label="weighted accuracy",
    )
    return {
        "experts": expert_summary,
        "high_score_hardness": hardness_rows,
    }


def build_phase6_analysis(
    config: dict[str, Any],
    *,
    checkpoint_materializer: CheckpointMaterializer | None = None,
) -> Path:
    phase0_dir = Path(config["phase0_dir"]).expanduser().resolve()
    verify_phase0(phase0_dir, require_approval=True)
    verify_ula_proxy(config["ula_proxy_dir"], load_checkpoint=False)
    proxy = load_ula_proxy_scores(config["ula_proxy_dir"])
    output = Path(config["output_dir"]).expanduser().resolve()
    if output.exists():
        raise ArtifactExistsError(f"Phase 6 output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / (
        f".{output.name}.building."
        f"{os.environ.get('SLURM_JOB_ID', uuid.uuid4().hex[:12])}"
    )
    for directory in (
        "selection",
        "analysis_only",
        "reporting_only",
        "tables",
        "plots",
        "selected_candidate_checkpoints",
        "config",
        "provenance",
    ):
        (staging / directory).mkdir(parents=True, exist_ok=True)
    tolerance = float(config["selection"]["score_tolerance"])
    seeds = []
    for seed in config["candidate_seeds"]:
        candidate_dir = (
            Path(config["candidate_root"]).expanduser().resolve() / f"seed_{seed}"
        )
        value = _load_seed(candidate_dir, proxy, tolerance)
        if Path(value["phase0_dir"]).resolve() != phase0_dir:
            raise DataValidationError("Candidate seed references another Phase 0")
        seeds.append(value)
    aggregate = _aggregate(seeds)
    biased_ids = proxy["sample_id"].astype(str)
    labels = proxy["true_label"].astype(np.int64)
    selector_inputs = load_selector_inputs(config["fusion_dirs"], biased_ids, labels)
    expert_metadata = _derive_expert_metadata(selector_inputs, config)
    choice = _joint_choice(aggregate, expert_metadata, config)
    dominance = _expert_dominance(seeds)
    background_baselines = _background_confidence_baselines(
        seeds, selector_inputs, labels, tolerance
    )

    # Resolve the uLA-selected model state before any test report is opened.
    checkpoint_receipts = {}
    if checkpoint_materializer is None:
        from setv.ula.materialize import materialize_ula_selected_checkpoint

        checkpoint_materializer = materialize_ula_selected_checkpoint
    for seed in seeds:
        checkpoint_receipts[str(seed["seed"])] = checkpoint_materializer(
            Path(seed["candidate_dir"]),
            int(seed["selected"]["ula"]),
            staging / "selected_candidate_checkpoints" / f"seed_{seed['seed']}",
        )

    diagnostics = _diagnostics(staging, config, seeds, selector_inputs)
    write_json(
        staging / "analysis_only" / "background_confidence_baselines.json",
        background_baselines,
    )
    selection_only = {
        "schema_version": 1,
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_seed_count": len(seeds),
        "candidate_seeds": [item["seed"] for item in seeds],
        "per_seed": {
            str(item["seed"]): {
                "selection_receipt_sha256": item["selection_receipt_sha256"],
                "selected_epochs": item["selected"],
                "metrics": item["metrics"],
                "ula": {
                    "label": item["ula"]["label"],
                    "official_primary_metric": item["ula"][
                        "official_primary_metric"
                    ],
                    "tie_break": item["ula"]["tie_break"],
                    "best": item["ula"]["best"],
                },
                "ula_selected_checkpoint": checkpoint_receipts[str(item["seed"])],
            }
            for item in seeds
        },
        "aggregate_selector_metrics": aggregate,
        "joint_expert_fusion_choice": choice,
        "derived_expert_metadata": expert_metadata,
        "background_confidence_baseline": background_baselines["aggregate"],
        "expert_dominance_audit": dominance,
        "test_metrics_seen": False,
        "test_metrics_used": False,
        "oracle_used_for_private_method_design": True,
        "test_used_for_method_design": False,
    }
    selection_path = staging / "selection" / "method_freeze_receipt.json"
    write_json(selection_path, selection_only)
    freeze_hash = sha256_file(selection_path)
    write_json(
        staging / "selection" / "method_frozen.json",
        {
            "status": "frozen_before_test_load",
            "method_freeze_receipt_sha256": freeze_hash,
            "selected_configuration": choice["selected_configuration"],
        },
    )
    # The next line is the first Phase 6 access to reporting-only test values.
    test_reports = {
        item["seed"]: _json(
            Path(item["candidate_dir"]) / "reporting_only" / "test_metrics.json"
        )
        for item in seeds
    }
    primary_rows = []
    row_names = [
        "ordinary",
        "ula",
        "hard_pseudogroup.exact",
        *[
            f"setv.{expert}.{fusion}"
            for expert in ("exact", "sanitized", "set")
            for fusion in ("hard", "rank", "logistic")
        ],
        "oracle",
    ]
    for seed in seeds:
        test_by_epoch = {
            int(item["epoch"]): item for item in test_reports[seed["seed"]]["epochs"]
        }
        for name in row_names:
            if name not in seed["metrics"]:
                continue
            selected = seed["metrics"][name]
            test = test_by_epoch[selected["selected_epoch"]]
            parts = name.split(".")
            primary_rows.append(
                {
                    "candidate_seed": seed["seed"],
                    "selector": (
                        "Ordinary validation"
                        if name == "ordinary"
                        else "uLA-style"
                        if name == "ula"
                        else "Oracle"
                        if name == "oracle"
                        else "Hard pseudo-groups"
                        if name.startswith("hard_pseudogroup")
                        else "SETV"
                    ),
                    "background_expert": (
                        "N/A"
                        if name in {"ordinary", "oracle"}
                        else "uLA proxy"
                        if name == "ula"
                        else parts[1].capitalize()
                    ),
                    "fusion": (
                        "N/A"
                        if name == "ordinary"
                        else "true groups"
                        if name == "oracle"
                        else "hard proxy groups"
                        if name == "ula" or name.startswith("hard_pseudogroup")
                        else parts[2]
                    ),
                    "selector_key": name,
                    "selected_epoch": selected["selected_epoch"],
                    "oracle_val_wga": selected["oracle_val_wga"],
                    "test_average_accuracy": test["average_accuracy"],
                    "test_wga": test["worst_group_accuracy"],
                    "oracle_selection_regret": selected[
                        "oracle_selection_regret"
                    ],
                }
            )
    _write_csv(staging / "tables" / "primary_results_by_seed.csv", primary_rows)
    aggregate_rows = []
    for name in row_names:
        rows = [row for row in primary_rows if row["selector_key"] == name]
        if not rows:
            continue
        aggregate_rows.append(
            {
                "selector_key": name,
                "candidate_seed_count": len(rows),
                "mean_selected_epoch": float(
                    np.mean([row["selected_epoch"] for row in rows])
                ),
                "mean_oracle_val_wga": float(
                    np.mean([row["oracle_val_wga"] for row in rows])
                ),
                "mean_test_average_accuracy": float(
                    np.mean([row["test_average_accuracy"] for row in rows])
                ),
                "mean_test_wga": float(np.mean([row["test_wga"] for row in rows])),
                "mean_oracle_selection_regret": aggregate[name][
                    "mean_oracle_selection_regret"
                ],
                "worst_oracle_selection_regret": aggregate[name][
                    "worst_oracle_selection_regret"
                ],
                "mean_spearman": aggregate[name]["mean_spearman"],
                "mean_kendall": aggregate[name]["mean_kendall"],
                "mean_pairwise_ranking_accuracy": aggregate[name][
                    "mean_pairwise_ranking_accuracy"
                ],
            }
        )
    _write_csv(staging / "tables" / "primary_results_aggregate.csv", aggregate_rows)
    write_json(
        staging / "reporting_only" / "test_results.json",
        {
            "namespace": "reporting_only",
            "method_freeze_receipt_sha256": freeze_hash,
            "loaded_after_method_freeze": True,
            "rows": primary_rows,
        },
    )
    _svg_bar(
        staging / "plots" / "mean_selection_regret.svg",
        [row["selector_key"] for row in aggregate_rows],
        [row["mean_oracle_selection_regret"] for row in aggregate_rows],
        "Mean oracle selection regret across candidate seeds",
    )
    thresholds = config["kill_criteria"]
    object_margins = np.asarray(
        selector_inputs["exact"].fusion.get(
            "object_true_class_margin",
            selector_inputs["exact"].fusion["rank"]["object_percentile"],
        ),
        dtype=np.float64,
    )
    for name in ("sanitized", "set"):
        other = np.asarray(
            selector_inputs[name].fusion.get(
                "object_true_class_margin",
                selector_inputs[name].fusion["rank"]["object_percentile"],
            ),
            dtype=np.float64,
        )
        if not np.allclose(object_margins, other, rtol=0, atol=1e-6):
            raise DataValidationError(
                "Fusion artifacts do not share one object-expert margin vector"
            )
    object_correct = object_margins > 0
    object_balanced_accuracy = float(
        np.mean(
            [
                object_correct[labels == class_id].mean()
                for class_id in (0, 1)
            ]
        )
    )
    background_variation = {
        name: float(
            np.asarray(
                source.fusion.get(
                    "background_true_class_margin",
                    source.fusion["rank"]["background_percentile"],
                ),
                dtype=np.float64,
            ).std()
        )
        for name, source in selector_inputs.items()
    }
    selected_expert = choice["background_expert"]
    selected_fusion = choice["fusion"]
    selected_hardness_rows = [
        row
        for row in diagnostics["high_score_hardness"]
        if row["expert"] == selected_expert
        and (
            row["fusion"] == selected_fusion
            if selected_fusion in {"rank", "logistic"}
            else row["fusion"] == "rank"
        )
    ]
    high_score_triggered = bool(selected_hardness_rows) and all(
        row["simply_hard_for_every_candidate"]
        for row in selected_hardness_rows
    )
    regret_tolerance = float(
        thresholds["selector_regret_improvement_tolerance"]
    )
    baseline_comparisons = {}
    for expert in ("exact", "sanitized", "set"):
        baseline_regret = background_baselines["aggregate"][expert][
            "mean_oracle_selection_regret"
        ]
        comparison = {}
        for fusion_name in ("rank", "logistic"):
            key = f"setv.{expert}.{fusion_name}"
            if key not in aggregate:
                comparison[fusion_name] = {
                    "available": False,
                    "outperforms_background_confidence": False,
                }
            else:
                regret = aggregate[key]["mean_oracle_selection_regret"]
                comparison[fusion_name] = {
                    "available": True,
                    "mean_oracle_selection_regret": regret,
                    "outperforms_background_confidence": bool(
                        regret < baseline_regret - regret_tolerance
                    ),
                }
        baseline_comparisons[expert] = {
            "background_confidence_mean_oracle_selection_regret": baseline_regret,
            "fusion": comparison,
            "neither_rank_nor_logistic_outperforms": not any(
                value["outperforms_background_confidence"]
                for value in comparison.values()
            ),
        }
    epoch_tolerance = int(
        thresholds["essentially_identical_epoch_tolerance"]
    )
    per_seed_epoch_spread = {}
    for seed_index, seed in enumerate(seeds):
        epochs = [
            value["selected_epochs"][seed_index]
            for key, value in aggregate.items()
            if key != "oracle"
        ]
        per_seed_epoch_spread[str(seed["seed"])] = {
            "minimum": int(min(epochs)),
            "maximum": int(max(epochs)),
            "spread": int(max(epochs) - min(epochs)),
            "within_tolerance": bool(max(epochs) - min(epochs) <= epoch_tolerance),
        }
    selected_regret = aggregate[choice["selected_configuration"]][
        "mean_oracle_selection_regret"
    ]
    ordinary_regret = aggregate["ordinary"]["mean_oracle_selection_regret"]
    kill = {
        "schema_version": 1,
        "thresholds": thresholds,
        "object_expert_near_chance": {
            "triggered": bool(
                object_balanced_accuracy
                <= float(
                    thresholds[
                        "object_balanced_accuracy_near_chance_ceiling"
                    ]
                )
            ),
            "balanced_accuracy": object_balanced_accuracy,
        },
        "background_margin_negligible_variation": {
            "triggered_for_selected_expert": bool(
                background_variation[selected_expert]
                < float(
                    thresholds[
                        "background_margin_standard_deviation_floor"
                    ]
                )
            ),
            "selected_expert": selected_expert,
            "standard_deviation_by_expert": background_variation,
        },
        "hard_informative_examples_absent": {
            "triggered_for_selected_expert": not diagnostics["experts"][
                selected_expert
            ]["hard_valid"],
            "hard_valid_by_expert": {
                name: value["hard_valid"]
                for name, value in diagnostics["experts"].items()
            },
        },
        "logistic_cross_fitting_degenerate": {
            "triggered_for_selected_expert": not diagnostics["experts"][
                selected_expert
            ]["logistic"].get("available", False),
            "available_by_expert": {
                name: value["logistic"].get("available", False)
                for name, value in diagnostics["experts"].items()
            },
        },
        "high_fusion_score_images_simply_hard_for_every_candidate": {
            "triggered": high_score_triggered,
            "selected_configuration": choice["selected_configuration"],
            "evaluated_fusion": (
                selected_fusion
                if selected_fusion in {"rank", "logistic"}
                else "rank_diagnostic_for_selected_hard_configuration"
            ),
            "per_seed": selected_hardness_rows,
        },
        "rank_and_logistic_do_not_outperform_background_confidence_alone": {
            "triggered_for_selected_expert": baseline_comparisons[selected_expert][
                "neither_rank_nor_logistic_outperforms"
            ],
            "selected_expert": selected_expert,
            "by_expert": baseline_comparisons,
        },
        "all_realistic_selectors_choose_essentially_identical_epochs": {
            "triggered": all(
                value["within_tolerance"]
                for value in per_seed_epoch_spread.values()
            ),
            "per_seed_epoch_spread": per_seed_epoch_spread,
        },
        "setv_fails_to_improve_mean_regret_over_ordinary": {
            "triggered": bool(
                selected_regret >= ordinary_regret - regret_tolerance
            ),
            "selected_setv_mean_regret": selected_regret,
            "ordinary_mean_regret": ordinary_regret,
        },
        "interpretation": (
            "Flags are computed diagnostic outcomes, not authorization to "
            "redesign the locked method."
        ),
    }
    write_json(staging / "analysis_only" / "kill_criteria.json", kill)
    resolved = resolved_config(config)
    with (staging / "config" / "resolved_phase6.yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(resolved, handle, sort_keys=True)
    ula_receipt = _json(
        Path(config["ula_proxy_dir"]) / "phase6_ula_proxy_receipt.json"
    )
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "kind": "setv_phase6_frozen_analysis",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase0_dir": str(phase0_dir),
        "phase0_artifact_manifest_sha256": sha256_file(
            phase0_dir / BASE_ARTIFACT_MANIFEST
        ),
        "ula_label": "uLA-style",
        "ula_official_repository": ula_receipt["official_source"]["repository"],
        "ula_official_commit": ula_receipt["official_source"]["required_commit"],
        "ula_proxy_receipt_sha256": sha256_file(
            Path(config["ula_proxy_dir"]) / "phase6_ula_proxy_receipt.json"
        ),
        "candidate_seed_count": len(seeds),
        "candidate_selection_receipt_sha256": {
            str(item["seed"]): item["selection_receipt_sha256"] for item in seeds
        },
        "method_freeze_receipt": {
            "path": selection_path.relative_to(staging).as_posix(),
            "sha256": freeze_hash,
        },
        "selected_configuration": choice["selected_configuration"],
        "test_policy": {
            "reporting_only": True,
            "loaded_after_hashed_freeze": True,
            "used_for_selection": False,
        },
        "config_sha256": sha256_json(resolved),
    }
    write_json(staging / "phase6_receipt.json", receipt)
    write_json(staging / "artifact_manifest.json", _artifact_manifest(staging))
    os.rename(staging, output)
    return output


def verify_phase6(root: str | Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    manifest = _json(root / "artifact_manifest.json")
    if sha256_json(manifest["files"]) != manifest["manifest_digest"]:
        raise DataValidationError("Phase 6 manifest digest is invalid")
    for relative, expected in manifest["files"].items():
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != expected["size_bytes"]
            or sha256_file(path) != expected["sha256"]
        ):
            raise DataValidationError(f"Missing/changed Phase 6 artifact: {relative}")
    receipt = _json(root / "phase6_receipt.json")
    if receipt.get("kind") != "setv_phase6_frozen_analysis":
        raise DataValidationError("Wrong Phase 6 artifact kind")
    freeze_path = root / receipt["method_freeze_receipt"]["path"]
    freeze_hash = sha256_file(freeze_path)
    if freeze_hash != receipt["method_freeze_receipt"]["sha256"]:
        raise DataValidationError("Phase 6 freeze receipt hash changed")
    frozen = _json(root / "selection" / "method_frozen.json")
    reporting = _json(root / "reporting_only" / "test_results.json")
    if (
        frozen["method_freeze_receipt_sha256"] != freeze_hash
        or reporting["method_freeze_receipt_sha256"] != freeze_hash
        or not reporting["loaded_after_method_freeze"]
    ):
        raise DataValidationError("Phase 6 test report is not bound to the freeze")
    selection = _json(freeze_path)
    if selection["test_metrics_seen"] or selection["test_metrics_used"]:
        raise DataValidationError("Phase 6 selection receipt indicates test leakage")
    if selection["candidate_seed_count"] < 3:
        raise DataValidationError("Phase 6 result has fewer than three candidate seeds")
    for seed, checkpoint in selection["per_seed"].items():
        path = root / checkpoint["ula_selected_checkpoint"]["path"]
        if not path.is_file() or sha256_file(path) != checkpoint[
            "ula_selected_checkpoint"
        ]["sha256"]:
            raise DataValidationError(f"uLA-selected checkpoint is invalid for seed {seed}")
    return {
        "status": "complete",
        "candidate_seed_count": selection["candidate_seed_count"],
        "selected_configuration": receipt["selected_configuration"],
        "ula_label": receipt["ula_label"],
        "test_namespace": reporting["namespace"],
        "method_freeze_receipt_sha256": freeze_hash,
        "artifact_count": len(manifest["files"]),
    }
