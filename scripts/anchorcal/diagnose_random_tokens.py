#!/usr/bin/env python3
"""Run a read-only repeated random-token diagnostic on trained branches."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from _common import DEFAULT_CONFIG, DEFAULT_PATHS, resolved_config
from anchorcal.anchor_pipeline import _load_branch_models
from anchorcal.datasets import EvaluationDataset
from anchorcal.io import atomic_write_json, sha256_file
from anchorcal.paths import geometry_artifact_root
from anchorcal.precision import evaluation_inference
from anchorcal.preprocessing import load_preprocessing_manifest
from anchorcal.random_token_diagnostic import (
    DIAGNOSTIC_MODES,
    diagnostic_seeds,
    random_token_draw_indices,
    summarize_random_token_predictions,
)
from anchorcal.seeds import seed_everything
from anchorcal.vlm_masks import load_vlm_mask_bank


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose a borderline random-token audit without retraining."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--paths", default=str(DEFAULT_PATHS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    import torch

    config = resolved_config(args.config, args.paths, debug=False)
    campaign_root = Path(config["paths"]["output_root"]).resolve()
    target = Path(args.output).resolve()
    try:
        target.relative_to(campaign_root)
    except ValueError:
        pass
    else:
        raise ValueError("diagnostic output must remain outside the frozen campaign root")
    if args.repeats < 2 or args.batch_size < 1:
        raise ValueError("repeats must be at least two and batch size must be positive")

    base_seed = int(config["seeds"]["random_token_audit"])
    seed_everything(base_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("the random-token diagnostic requires a GPU")
    _, background, _, background_checkpoint = _load_branch_models(config, device)
    preprocessing = load_preprocessing_manifest(campaign_root)
    mask_bank = load_vlm_mask_bank(config)
    expert_frame = pd.read_csv(
        campaign_root / "splits" / "waterbirds100_expert_train.csv"
    )
    expert_dataset = EvaluationDataset(
        expert_frame,
        config["paths"]["waterbirds_root"],
        mask_bank,
        preprocessing=preprocessing,
    )

    maximum_per_class = 12000
    pool_parts: dict[int, list[torch.Tensor]] = {0: [], 1: []}
    source_images: dict[int, set[int]] = {0: set(), 1: set()}
    patch_counts: dict[int, int] = {0: 0, 1: 0}
    with evaluation_inference(device):
        for index in range(len(expert_dataset)):
            sample = expert_dataset[index]
            label = int(sample["y"])
            if patch_counts[label] >= maximum_per_class:
                if all(patch_counts[value] >= maximum_per_class for value in (0, 1)):
                    break
                continue
            projected = background.project(sample["image"].unsqueeze(0).to(device))[0]
            indices = sample["background_indices"].to(device)
            if len(indices):
                selected = projected[indices].cpu()
                pool_parts[label].append(selected)
                patch_counts[label] += len(selected)
                source_images[label].add(int(sample["img_id"]))
    count = min(patch_counts[0], patch_counts[1], maximum_per_class)
    if count < background.token_budget:
        raise RuntimeError("diagnostic source pool is too small")
    class_pools = [torch.cat(pool_parts[label])[:count] for label in (0, 1)]
    balanced_pool = torch.cat(class_pools, dim=0).to(device)

    table = pd.read_csv(geometry_artifact_root(config) / "biased_val_geometry.csv")
    valid = table["safe_background_count"] >= background.token_budget
    recipients = table.loc[valid].copy()
    img_ids = recipients["img_id"].to_numpy(dtype=np.int64)
    labels = recipients["y"].to_numpy(dtype=np.int64)
    if any(set(img_ids) & source_images[label] for label in (0, 1)):
        raise RuntimeError("diagnostic source and recipient images overlap")
    seeds = diagnostic_seeds(base_seed, args.repeats)
    results: dict[str, object] = {}
    for mode in DIAGNOSTIC_MODES:
        prediction_rows: list[np.ndarray] = []
        for seed in seeds:
            draw_indices = random_token_draw_indices(
                img_ids,
                patches_per_class=count,
                token_budget=background.token_budget,
                seed=seed,
                mode=mode,
            )
            predictions: list[np.ndarray] = []
            # Preserve the production audit's batch-one numerical path for the
            # original pooled seed. The remaining diagnostic realizations use
            # safe batching so the extra analysis stays quick.
            evaluation_batch_size = (
                1 if mode == "pooled" and seed == base_seed else args.batch_size
            )
            with evaluation_inference(device):
                for start in range(0, len(img_ids), evaluation_batch_size):
                    indices = torch.from_numpy(
                        draw_indices[start : start + evaluation_batch_size]
                    ).to(device)
                    tokens = balanced_pool[indices]
                    token_valid = torch.ones(
                        tokens.shape[:2], dtype=torch.bool, device=device
                    )
                    logits, _ = background.encoder.forward_tokens(tokens, token_valid)
                    predictions.append(logits.argmax(dim=1).cpu().numpy())
            prediction_rows.append(np.concatenate(predictions))
        results[mode] = summarize_random_token_predictions(
            np.stack(prediction_rows),
            labels,
            seeds,
            bootstrap_replicates=2000,
            permutation_replicates=2000,
            summary_seed=base_seed,
        )

    repo = Path(config["paths"]["repo_root"])
    with (campaign_root / "preflight" / "report.json").open(
        "r", encoding="utf-8"
    ) as handle:
        source_preflight = json.load(handle)
    report = {
        "schema_version": "anchorcal-random-token-diagnostic-v1",
        "status": "diagnostic_only_no_gate_or_override",
        "source_campaign_root": str(campaign_root),
        "diagnostic_commit": _git_commit(repo),
        "source_campaign_commit": source_preflight["git"]["commit"],
        "source_preflight_report_sha256": sha256_file(
            campaign_root / "preflight" / "report.json"
        ),
        "source_resolved_config_sha256": config["resolved_config_sha256"],
        "source_background_checkpoint": str(background_checkpoint),
        "source_background_checkpoint_sha256": sha256_file(background_checkpoint),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "device": str(device),
        "token_budget": int(background.token_budget),
        "source_patches_per_class": int(count),
        "source_unique_images_by_class": {
            str(label): len(source_images[label]) for label in (0, 1)
        },
        "recipient_count": len(img_ids),
        "recipient_class_counts": {
            str(label): int(np.sum(labels == label)) for label in (0, 1)
        },
        "source_recipient_overlap_count": 0,
        "seeds": seeds,
        "evaluation_batch_policy": {
            "original_pooled_seed": 1,
            "additional_diagnostic_realizations": args.batch_size,
        },
        "results": results,
        "interpretation_contract": (
            "Diagnostic only. It neither changes nor overrides the original hard gate."
        ),
    }
    atomic_write_json(target, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
