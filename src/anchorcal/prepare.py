"""Model-independent geometry, subset, token-budget, view, and donor artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .background import (
    build_view_arrays,
    invalid_background_records,
    persist_view_bank,
    select_token_budget,
)
from .data import image_path, mask_path
from .errors import AuditFailure
from .interventions import assign_candidate_donor_patches, assign_donors
from .io import atomic_write_json, hash_object
from .masks import load_binary_mask
from .paths import geometry_artifact_root
from .preprocessing import EvaluationPreprocessing, load_preprocessing_manifest
from .seeds import stateless_rng
from .transforms import criterion_patch_eligibility, deterministic_eval_transform
from PIL import Image


def _read_split(root: Path, name: str) -> pd.DataFrame:
    path = root / f"waterbirds100_{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"preflight split is missing: {path}")
    return pd.read_csv(path).sort_values("img_id", kind="stable").reset_index(drop=True)


def geometry_table(
    frame: pd.DataFrame,
    image_root: Path,
    mask_root: Path,
    preprocessing: EvaluationPreprocessing,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    eligible_by_id: dict[int, np.ndarray] = {}
    for row in frame.itertuples(index=False):
        with Image.open(image_path(image_root, str(row.img_filename))) as opened:
            image = opened.convert("RGB")
        mask = load_binary_mask(mask_path(mask_root, str(row.img_filename)))
        transformed = deterministic_eval_transform(
            image,
            mask,
            image_size=preprocessing.image_size,
            resize_shortest=preprocessing.effective_resize_shortest,
        )
        eligibility = criterion_patch_eligibility(transformed.mask)
        background = np.asarray(eligibility.background_indices, dtype=np.int64)
        eligible_by_id[int(row.img_id)] = background
        rows.append(
            {
                "img_id": int(row.img_id),
                "y": int(row.y),
                "place": int(row.place),
                "pure_foreground_count": len(eligibility.foreground_indices),
                "safe_background_count": len(background),
                "common_eligible": eligibility.eligible,
                "exclusion_reasons": ";".join(eligibility.exclusion_reasons),
                "foreground_indices": ",".join(map(str, eligibility.foreground_indices)),
                "background_indices": ",".join(map(str, eligibility.background_indices)),
            }
        )
    return pd.DataFrame(rows).sort_values("img_id", kind="stable"), eligible_by_id


def fixed_class_subset(
    table: pd.DataFrame, *, per_class: int, seed: int
) -> np.ndarray:
    selected: list[np.ndarray] = []
    for label in (0, 1):
        ids = np.sort(
            table.loc[(table["y"] == label) & table["common_eligible"], "img_id"]
            .astype(int)
            .to_numpy()
        )
        if len(ids) > per_class:
            ids = np.sort(
                stateless_rng(seed, label, "selector_eval_subset").choice(
                    ids, per_class, replace=False
                )
            )
        selected.append(ids)
    return np.sort(np.concatenate(selected))


def prepare_geometry_artifacts(
    config: dict[str, Any],
    *,
    preprocessing: EvaluationPreprocessing | None = None,
) -> dict[str, Any]:
    output = Path(config["paths"]["output_root"])
    if preprocessing is None:
        preprocessing = load_preprocessing_manifest(output)
    splits_root = output / "splits"
    artifact_root = geometry_artifact_root(config)
    artifact_root.mkdir(parents=True, exist_ok=True)
    image_root = Path(config["paths"]["waterbirds_root"])
    mask_root = Path(config["paths"]["cub_waterbirds_mask_root"])
    frames = {
        name: _read_split(splits_root, name)
        for name in ("expert_train", "expert_calibration", "biased_val")
    }
    tables: dict[str, pd.DataFrame] = {}
    backgrounds: dict[str, dict[int, np.ndarray]] = {}
    for name, frame in frames.items():
        tables[name], backgrounds[name] = geometry_table(
            frame, image_root, mask_root, preprocessing
        )
        tables[name].to_csv(artifact_root / f"{name}_geometry.csv", index=False)
    decision = select_token_budget(
        {
            name: table["safe_background_count"].to_numpy()
            for name, table in tables.items()
        },
        {name: table["y"].to_numpy() for name, table in tables.items()},
        candidates=tuple(config["branches"]["background_token_budget_candidates"]),
        minimum_coverage=float(config["branches"]["background_min_coverage"]),
    )
    token_manifest = {
        "token_budget": decision.token_budget,
        "coverage": decision.coverage,
        "selection_inputs": ["expert_train", "expert_calibration", "biased_val"],
        "selection_rule": "largest_64_48_32_meeting_95pct_overall_and_per_class",
    }
    atomic_write_json(artifact_root / "background_token_budget.json", token_manifest)
    eval_backgrounds = dict(backgrounds["expert_calibration"])
    eval_backgrounds.update(backgrounds["biased_val"])
    ids, indices, seeds, invalid = build_view_arrays(
        eval_backgrounds,
        token_budget=decision.token_budget,
        views=int(config["branches"]["background_eval_views"]),
        global_seed=int(config["seeds"]["background_sampling"]),
    )
    view_manifest = persist_view_bank(
        artifact_root / "fixed_background_views.h5",
        img_ids=ids,
        indices=indices,
        seeds=seeds,
        invalid_ids=invalid,
        invalid_records=invalid_background_records(
            eval_backgrounds, invalid, decision.token_budget
        ),
        token_budget=decision.token_budget,
        mask_dilation_hash=hash_object({"implementation": "euclidean_disk", "radius": 8}),
        purpose="background_branch_eval",
    )
    biased_table = tables["biased_val"]
    common_counts = {
        str(label): int(
            np.sum(
                (biased_table["y"].to_numpy(dtype=np.int64) == label)
                & biased_table["common_eligible"].to_numpy(dtype=bool)
            )
        )
        for label in (0, 1)
    }
    for label in (0, 1):
        if common_counts[str(label)] < 50:
            raise AuditFailure(
                f"common criterion eligibility class {label} has "
                f"{common_counts[str(label)]}; requires 50"
            )
    selector_ids = fixed_class_subset(
        biased_table,
        per_class=int(config["criteria"]["selector_eval_per_class"]),
        seed=int(config["seeds"]["selector_eval"]),
    )
    subset = biased_table.loc[biased_table["img_id"].isin(selector_ids)].copy()
    subset.to_csv(artifact_root / "selector_eval_subset.csv", index=False)
    valid = biased_table["common_eligible"].to_numpy(dtype=bool)
    # Donor IDs are shared by ordinary candidates and position-free anchors.
    # Every donor must therefore have a valid fixed-K background view; merely
    # having one safe patch is insufficient and would yield NaN branch logits.
    donor_valid = valid & (
        biased_table["safe_background_count"].to_numpy(dtype=np.int64)
        >= decision.token_budget
    )
    donor_assignments = assign_donors(
        biased_table["img_id"].to_numpy(),
        biased_table["y"].to_numpy(),
        valid,
        donor_eligible=donor_valid,
        donors_per_recipient=int(config["criteria"]["swap_donors"]),
        seed=int(config["seeds"]["donor_assignment"]),
    )
    background_lookup = backgrounds["biased_val"]
    donor_payload: list[dict[str, Any]] = []
    for assignment in donor_assignments:
        per_donor = []
        for donor_id in assignment.donor_ids:
            patches = assign_candidate_donor_patches(
                assignment.recipient_id,
                donor_id,
                background_lookup[assignment.recipient_id],
                background_lookup[donor_id],
                seed=int(config["seeds"]["donor_assignment"]),
            )
            per_donor.append(
                {
                    "donor_id": donor_id,
                    "patch_assignments": [patch.__dict__ for patch in patches],
                }
            )
        donor_payload.append(
            {"recipient_id": assignment.recipient_id, "donors": per_donor}
        )
    atomic_write_json(artifact_root / "donor_assignments.json", donor_payload)
    manifest = {
        "schema_version": "anchorcal-geometry-artifacts-v1",
        "preprocessing": preprocessing.to_dict(),
        "token_budget": token_manifest,
        "view_bank": view_manifest,
        "selector_subset": {
            "count": int(len(selector_ids)),
            "img_ids": selector_ids.tolist(),
            "seed": int(config["seeds"]["selector_eval"]),
            "cap_per_class": int(config["criteria"]["selector_eval_per_class"]),
            "common_source_per_class": common_counts,
            "source_pool_hash": hash_object(
                biased_table.loc[biased_table["common_eligible"], "img_id"].tolist()
            ),
        },
        "donor_assignment_hash": hash_object(donor_payload),
        "donor_pool": {
            "rule": "common_eligible_and_fixed_background_token_budget_valid",
            "count": int(donor_valid.sum()),
            "per_class": {
                str(label): int(np.sum(donor_valid & (biased_table["y"].to_numpy() == label)))
                for label in (0, 1)
            },
        },
        "criterion_geometry_coverage": {
            "common_eligible_count": int(biased_table["common_eligible"].sum()),
            "common_eligible_per_class": common_counts,
            "zero_pure_foreground_count": int(
                (biased_table["pure_foreground_count"] == 0).sum()
            ),
            "zero_safe_background_count": int(
                (biased_table["safe_background_count"] == 0).sum()
            ),
            "pure_foreground_patch_count_percentiles": {
                str(percentile): float(
                    np.percentile(biased_table["pure_foreground_count"], percentile)
                )
                for percentile in (0, 25, 50, 75, 100)
            },
            "safe_background_patch_count_percentiles": {
                str(percentile): float(
                    np.percentile(biased_table["safe_background_count"], percentile)
                )
                for percentile in (0, 25, 50, 75, 100)
            },
        },
    }
    atomic_write_json(artifact_root / "manifest.json", manifest)
    return manifest
