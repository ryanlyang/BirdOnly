"""Per-epoch practical criteria and physically separate hidden evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .criteria import build_scores
from .datasets import EvaluationDataset, PlainEvaluationDataset
from .interventions import (
    InterventionType,
    PatchAssignment,
    apply_candidate_token_swap,
    require_candidate_intervention,
)
from .metrics import class_balanced_mean, classification_metrics, per_example_cross_entropy
from .paths import geometry_artifact_root
from .precision import evaluation_inference, saliency_evaluation
from .preprocessing import load_preprocessing_manifest
from .saliency import image_alignment, signed_gradient_activation
from .transforms import mask_normalized_background_blur, normalize_image
from .vlm_masks import VlmMaskBank, load_vlm_mask_bank


def evaluate_plain(
    model,
    frame: pd.DataFrame,
    config: dict[str, Any],
    device,
    *,
    require_all_groups: bool = True,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    preprocessing = load_preprocessing_manifest(config["paths"]["output_root"])
    dataset = PlainEvaluationDataset(
        frame,
        config["paths"]["waterbirds_root"],
        preprocessing=preprocessing,
    )
    workers = int(config["optimization"]["num_workers"])
    options: dict[str, Any] = {
        "batch_size": int(config["candidate_grid"]["batch_size"]),
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": bool(config["optimization"]["pin_memory"]),
    }
    if workers:
        options.update(
            persistent_workers=bool(config["optimization"]["persistent_workers"]),
            prefetch_factor=int(config["optimization"]["prefetch_factor"]),
        )
    loader = DataLoader(dataset, **options)
    ids: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    places: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    model.eval()
    for intervention in (
        InterventionType.TOKEN_SWAP_BACKGROUND,
        InterventionType.BLUR_BACKGROUND,
        InterventionType.FOREGROUND_ONLY_GREENSCREEN,
    ):
        require_candidate_intervention(intervention)
    with evaluation_inference(device):
        for batch in loader:
            scores = model(batch["image"].to(device, non_blocking=True))
            ids.append(batch["img_id"].numpy())
            labels.append(batch["y"].numpy())
            places.append(batch["place"].numpy())
            logits.append(scores.float().cpu().numpy())
    ids_array = np.concatenate(ids)
    label_array = np.concatenate(labels)
    place_array = np.concatenate(places)
    score_array = np.concatenate(logits)
    predictions = score_array.argmax(axis=1)
    result = {
        "img_id": ids_array,
        "labels": label_array,
        "logits": score_array,
        "prediction": predictions,
        "correct": predictions == label_array,
        "loss": per_example_cross_entropy(score_array, label_array),
        "metrics": classification_metrics(
            score_array, label_array, place_array if require_all_groups else None
        ),
    }
    if require_all_groups:
        result["groups"] = label_array * 2 + place_array
        result["places"] = place_array
    return result


def _load_donors(config: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    path = geometry_artifact_root(config) / "donor_assignments.json"
    with path.open("r", encoding="utf-8") as handle:
        return {
            int(item["recipient_id"]): list(item["donors"])
            for item in json.load(handle)
        }


def evaluate_practical_criteria(
    model,
    biased_frame: pd.DataFrame,
    config: dict[str, Any],
    device,
    *,
    mask_bank: VlmMaskBank | None = None,
) -> dict[str, Any]:
    """Evaluate the fixed full biased set and fixed expensive subset."""

    import torch

    ordinary = evaluate_plain(
        model, biased_frame, config, device, require_all_groups=False
    )
    output = Path(config["paths"]["output_root"])
    preprocessing = load_preprocessing_manifest(output)
    if mask_bank is None:
        mask_bank = load_vlm_mask_bank(config)
    selector_table = pd.read_csv(
        geometry_artifact_root(config) / "selector_eval_subset.csv"
    )
    selector_ids = set(selector_table["img_id"].astype(int).tolist())
    subset_frame = biased_frame.loc[biased_frame["img_id"].isin(selector_ids)].sort_values(
        "img_id"
    )
    subset_dataset = EvaluationDataset(
        subset_frame,
        config["paths"]["waterbirds_root"],
        mask_bank,
        preprocessing=preprocessing,
    )
    # Donors can include common-eligible biased images outside the capped subset.
    ordered_biased_frame = biased_frame.sort_values(
        "img_id", kind="stable"
    ).reset_index(drop=True)
    all_dataset = EvaluationDataset(
        ordered_biased_frame,
        config["paths"]["waterbirds_root"],
        mask_bank,
        preprocessing=preprocessing,
    )
    all_index = {
        int(row.img_id): index
        for index, row in enumerate(ordered_biased_frame.itertuples(index=False))
    }
    donor_payload = _load_donors(config)
    donor_cache: dict[int, Any] = {}
    selector_labels = subset_frame["y"].to_numpy(dtype=np.int64)
    selector_img_ids = subset_frame["img_id"].to_numpy(dtype=np.int64)
    saliency = np.empty(len(subset_dataset), dtype=np.float64)
    swap_correct = np.empty(
        (len(subset_dataset), int(config["criteria"]["swap_donors"])), dtype=np.float64
    )
    swap_logits_cache = np.empty(
        (
            len(subset_dataset),
            int(config["criteria"]["swap_donors"]),
            2,
        ),
        dtype=np.float32,
    )
    blur_correct = np.empty(
        (len(subset_dataset), len(config["criteria"]["blur_sigmas"])), dtype=np.float64
    )
    blur_logits_cache = np.empty(
        (len(subset_dataset), len(config["criteria"]["blur_sigmas"]), 2),
        dtype=np.float32,
    )
    foreground_correct = np.empty(len(subset_dataset), dtype=np.float64)
    fallback_absolute = np.zeros(len(subset_dataset), dtype=bool)
    zero_attribution = np.zeros(len(subset_dataset), dtype=bool)
    model.eval()
    ordinary_index = {
        int(img_id): index for index, img_id in enumerate(ordinary["img_id"].tolist())
    }
    clean_subset_logits = np.stack(
        [ordinary["logits"][ordinary_index[int(img_id)]] for img_id in selector_img_ids]
    )
    for index in range(len(subset_dataset)):
        sample = subset_dataset[index]
        label = int(sample["y"])
        label_tensor = torch.tensor([label], dtype=torch.long, device=device)
        image = sample["image"].unsqueeze(0).to(device)
        with saliency_evaluation(device):
            logits, activations = model.forward_with_patch_leaf(image)
            signed = signed_gradient_activation(logits, label_tensor, activations)[0]
        alignment = image_alignment(
            signed.detach().float().cpu().numpy(),
            np.arange(196, dtype=np.int64),
            sample["foreground_indices"].numpy(),
            sample["background_indices"].numpy(),
        )
        saliency[index] = alignment.alignment
        fallback_absolute[index] = alignment.fallback_absolute
        zero_attribution[index] = alignment.zero_scored_attribution
        with evaluation_inference(device):
            recipient_tokens = model.project(image)
            for donor_offset, donor_item in enumerate(
                donor_payload[int(sample["img_id"])]
            ):
                donor_id = int(donor_item["donor_id"])
                if donor_id not in donor_cache:
                    donor_sample = all_dataset[all_index[donor_id]]
                    donor_cache[donor_id] = model.project(
                        donor_sample["image"].unsqueeze(0).to(device)
                    )
                assignments = [
                    PatchAssignment(**assignment)
                    for assignment in donor_item["patch_assignments"]
                ]
                swapped = apply_candidate_token_swap(
                    recipient_tokens, donor_cache[donor_id], assignments
                )
                swapped_logits = model.forward_from_projected(swapped)
                swap_correct[index, donor_offset] = float(
                    int(swapped_logits.argmax(dim=1).item()) == label
                )
                swap_logits_cache[index, donor_offset] = (
                    swapped_logits.float().cpu().numpy()[0]
                )
            for sigma_offset, sigma in enumerate(config["criteria"]["blur_sigmas"]):
                blurred = mask_normalized_background_blur(
                    sample["unnormalized_image"],
                    sample["mask"].numpy(),
                    sigma=float(sigma),
                )
                tensor = torch.from_numpy(
                    normalize_image(
                        blurred,
                        mean=preprocessing.mean,
                        std=preprocessing.std,
                    )
                ).unsqueeze(0).to(device)
                blurred_logits = model(tensor)
                blur_correct[index, sigma_offset] = float(
                    int(blurred_logits.argmax(dim=1).item()) == label
                )
                blur_logits_cache[index, sigma_offset] = (
                    blurred_logits.float().cpu().numpy()[0]
                )
            green_logits = model(
                sample["foreground_image"].unsqueeze(0).to(device)
            )
        foreground_correct[index] = float(int(green_logits.argmax(dim=1).item()) == label)
    label_column = selector_labels[:, None]
    clean_margin = (
        clean_subset_logits[np.arange(len(selector_labels)), selector_labels]
        - clean_subset_logits[np.arange(len(selector_labels)), 1 - selector_labels]
    )
    donor_true = np.take_along_axis(swap_logits_cache, label_column[:, :, None], axis=2)[
        :, :, 0
    ]
    donor_other = np.take_along_axis(
        swap_logits_cache, (1 - label_column)[:, :, None], axis=2
    )[:, :, 0]
    donor_margins = donor_true - donor_other
    swap_margin_drop = clean_margin - donor_margins.mean(axis=1)
    swap_prediction_flip = (
        swap_logits_cache.argmax(axis=2)
        != clean_subset_logits.argmax(axis=1)[:, None]
    ).mean(axis=1)
    swap_donor_margin_variance = donor_margins.var(axis=1)
    scores = build_scores(
        full_biased_correct=ordinary["correct"],
        full_biased_labels=ordinary["labels"],
        selector_labels=selector_labels,
        saliency_alignment=saliency,
        donor_correct=swap_correct,
        blur_correct=blur_correct,
        foreground_only_correct=foreground_correct,
    )
    score_payload = {
        "ordinary_accuracy": scores.ordinary_accuracy,
        "saliency_harmonic": scores.saliency_harmonic,
        "token_swap_harmonic": scores.token_swap_harmonic,
        "background_blur_harmonic": scores.background_blur_harmonic,
        "foreground_only_harmonic": scores.foreground_only_harmonic,
        **scores.diagnostics,
        "swap_mean_true_class_margin_drop": class_balanced_mean(
            swap_margin_drop, selector_labels
        ),
        "swap_prediction_flip_rate": class_balanced_mean(
            swap_prediction_flip, selector_labels
        ),
        "swap_donor_margin_variance": class_balanced_mean(
            swap_donor_margin_variance, selector_labels
        ),
    }
    return {
        "ordinary": ordinary,
        "selector_img_id": selector_img_ids,
        "selector_labels": selector_labels,
        "saliency_alignment": saliency,
        "saliency_fallback_absolute": fallback_absolute,
        "saliency_zero_attribution": zero_attribution,
        "swap_donor_correct": swap_correct,
        "swap_donor_logits": swap_logits_cache,
        "swap_margin_drop": swap_margin_drop,
        "swap_prediction_flip": swap_prediction_flip,
        "swap_donor_margin_variance": swap_donor_margin_variance,
        "blur_sigma_correct": blur_correct,
        "blur_sigma_logits": blur_logits_cache,
        "foreground_only_correct": foreground_correct,
        "scores": score_payload,
        "intervention_types": [
            InterventionType.TOKEN_SWAP_BACKGROUND.value,
            InterventionType.BLUR_BACKGROUND.value,
            InterventionType.FOREGROUND_ONLY_GREENSCREEN.value,
        ],
    }
