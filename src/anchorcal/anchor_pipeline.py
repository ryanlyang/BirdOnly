"""Branch audits, competence construction, anchor ladder, and decision receipt."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .anchor_cache import (
    MARGIN_SCALE_EPSILON,
    ExtremeCache,
    cached_logits,
    require_cache_parity,
)
from .audits import (
    assert_background_patch_purity,
    fit_geometry_auditors,
    require_foreground_invariance,
    require_random_token_collapse,
    stratified_bootstrap_interval,
)
from .background import load_view_bank
from .branch_provenance import verify_branch_artifacts
from .competence import (
    cap_anchor_subset,
    construct_competence_intersection,
    true_margin,
)
from .criteria import build_scores
from .data import image_path
from .datasets import EvaluationDataset
from .decision import choose_criterion, write_decision_receipt
from .errors import AuditFailure, PreflightError
from .interventions import (
    InterventionType,
    assert_anchor_intervention_contract,
)
from .io import atomic_write_json, atomic_write_text, hash_object, sha256_file
from .metrics import HARMONIC_EPSILON
from .masks import dilate_mask, mask_geometry, patch_fractions
from .models.anchor import centered_logits, mix_anchor_logits
from .models.branches import BackgroundBranch, ForegroundBranch, assert_independent_branches
from .pretrained import create_pretrained_vit
from .preprocessing import load_preprocessing_manifest
from .paths import geometry_artifact_root
from .precision import evaluation_inference, saliency_evaluation
from .saliency import anchor_image_alignment, image_alignment
from .seeds import seed_everything, stable_seed, stateless_rng
from .statistics import (
    class_balanced_mean,
    evaluate_anchor_scores,
    paired_class_stratified_bootstrap,
)
from .transforms import (
    PATCH_SIZE,
    deterministic_eval_transform,
    foreground_eval_transform,
    green_screen_source,
    mask_normalized_background_blur,
    normalize_image,
)
from .vlm_masks import VlmMaskBank, load_vlm_mask_bank


def _load_branch_models(config: dict[str, Any], device):
    import torch

    output = Path(config["paths"]["output_root"])
    namespace = "debug/branches" if config["runtime"]["debug"] else "branches"
    with (output / "preflight" / "pretrained_manifest.json").open("r", encoding="utf-8") as handle:
        weights = json.load(handle)["weights_path"]
    with (geometry_artifact_root(config) / "background_token_budget.json").open(
        "r", encoding="utf-8"
    ) as handle:
        token_budget = int(json.load(handle)["token_budget"])
    foreground = ForegroundBranch(
        create_pretrained_vit(weights),
        position_mode=config["branches"]["foreground_position_mode"],
    )
    background = BackgroundBranch(create_pretrained_vit(weights), token_budget)
    foreground_checkpoint = output / namespace / "foreground" / "epoch_final.pt"
    background_checkpoint = output / namespace / "background" / "epoch_final.pt"
    foreground_manifest = verify_branch_artifacts(config, "foreground")
    background_manifest = verify_branch_artifacts(config, "background")
    foreground_payload = torch.load(
        foreground_checkpoint, map_location="cpu", weights_only=False
    )
    background_payload = torch.load(
        background_checkpoint, map_location="cpu", weights_only=False
    )
    for branch, payload, manifest in (
        ("foreground", foreground_payload, foreground_manifest),
        ("background", background_payload, background_manifest),
    ):
        expected_budget = token_budget if branch == "background" else None
        expected_mode = (
            config["branches"]["foreground_position_mode"]
            if branch == "foreground"
            else None
        )
        if (
            payload.get("schema_version") != "anchorcal-branch-checkpoint-v2"
            or payload.get("branch") != branch
            or int(payload.get("epoch", -1))
            != int(config["branches"]["frozen_epoch"])
            or payload.get("token_budget") != expected_budget
            or payload.get("foreground_position_mode") != expected_mode
            or payload.get("resolved_config_sha256")
            != config["resolved_config_sha256"]
            or manifest.get("foreground_position_mode") != expected_mode
        ):
            raise PreflightError(f"{branch} checkpoint metadata is incompatible")
    foreground.load_state_dict(foreground_payload["model"])
    background.load_state_dict(background_payload["model"])
    assert_independent_branches(foreground, background)
    foreground.to(device).eval()
    background.to(device).eval()
    for model in (foreground, background):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return foreground, background, foreground_checkpoint, background_checkpoint


def _load_outputs(config: dict[str, Any], branch: str) -> dict[str, np.ndarray]:
    output = Path(config["paths"]["output_root"])
    namespace = "debug/branches" if config["runtime"]["debug"] else "branches"
    path = output / namespace / branch / "biased_val_outputs.npz"
    # Recheck the complete branch binding immediately before consuming logits;
    # this prevents a stale/mutable NPZ from being mixed with valid weights.
    verify_branch_artifacts(config, branch)
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def _join_branch_outputs(foreground: dict[str, np.ndarray], background: dict[str, np.ndarray]):
    fg_order = np.argsort(foreground["img_id"])
    bg_order = np.argsort(background["img_id"])
    if not np.array_equal(foreground["img_id"][fg_order], background["img_id"][bg_order]):
        raise AuditFailure("foreground/background biased_val IDs differ")
    if not np.array_equal(foreground["labels"][fg_order], background["labels"][bg_order]):
        raise AuditFailure("foreground/background labels differ")
    return {
        "img_id": foreground["img_id"][fg_order],
        "labels": foreground["labels"][fg_order],
        "foreground_logits": foreground["logits"][fg_order],
        "background_logits": background["logits"][bg_order],
        "background_valid": background["valid"][bg_order],
    }


def _view_lookup(config: dict[str, Any]) -> dict[int, np.ndarray]:
    output = Path(config["paths"]["output_root"])
    bank = load_view_bank(geometry_artifact_root(config) / "fixed_background_views.h5")
    return {
        int(img_id): bank["source_patch_indices"][row]
        for row, img_id in enumerate(bank["img_id"])
    }


def _foreground_invariance_audit(
    model,
    frame: pd.DataFrame,
    config,
    device,
    mask_bank: VlmMaskBank,
) -> dict[str, Any]:
    import torch

    preprocessing = load_preprocessing_manifest(config["paths"]["output_root"])
    eval_options = {
        "image_size": preprocessing.image_size,
        "resize_shortest": preprocessing.effective_resize_shortest,
    }
    image_root = Path(config["paths"]["waterbirds_root"])
    count = min(100, len(frame))
    selected = np.sort(
        stateless_rng(7001, "foreground_invariance_samples").choice(
            np.arange(len(frame)), size=count, replace=False
        )
    )
    differences = []
    shade_differences: dict[str, list[float]] = {
        "0_255_0": [],
        "0_200_0": [],
        "32_255_32": [],
    }
    shade_prediction_changes = {key: 0 for key in shade_differences}
    for offset, index in enumerate(selected.tolist()):
        row = frame.iloc[index]
        with Image.open(image_path(image_root, str(row.img_filename))) as opened:
            image = opened.convert("RGB")
        mask = mask_bank.load(int(row.img_id), str(row.img_filename))
        first = foreground_eval_transform(image, mask, **eval_options)
        first_image = torch.from_numpy(
            normalize_image(first.image, mean=preprocessing.mean, std=preprocessing.std)
        ).unsqueeze(0).to(device)
        mask_tensor = torch.from_numpy(first.mask).unsqueeze(0).to(device)
        with evaluation_inference(device):
            first_output = model(first_image, mask_tensor)
        replacement_choices = np.delete(np.arange(len(frame)), index)
        replacement_indices = stateless_rng(
            int(config["seeds"]["branch_audit_bootstrap"]),
            int(row.img_id),
            "foreground_replacement_backgrounds",
        ).choice(replacement_choices, size=3, replace=False)
        for donor_index in replacement_indices.tolist():
            donor = frame.iloc[int(donor_index)]
            with Image.open(image_path(image_root, str(donor.img_filename))) as opened:
                unrelated = opened.convert("RGB").resize(
                    image.size, Image.Resampling.BICUBIC
                )
            source = np.asarray(image).copy()
            replacement = np.asarray(unrelated)
            source[~mask] = replacement[~mask]
            replaced = Image.fromarray(source, mode="RGB")
            second = foreground_eval_transform(replaced, mask, **eval_options)
            if not np.array_equal(first.mask, second.mask):
                raise AuditFailure("source replacement changed final foreground mask")
            if not np.array_equal(np.asarray(first.image), np.asarray(second.image)):
                raise AuditFailure("source replacement changed final foreground input")
            second_image = (
                torch.from_numpy(
                    normalize_image(
                        second.image,
                        mean=preprocessing.mean,
                        std=preprocessing.std,
                    )
                ).unsqueeze(0).to(device)
            )
            with evaluation_inference(device):
                second_output = model(second_image, mask_tensor)
            differences.append(
                require_foreground_invariance(
                    first_output.logits.cpu().numpy(),
                    second_output.logits.cpu().numpy(),
                    first_tokens=first_output.patch_activations.cpu().numpy(),
                    second_tokens=second_output.patch_activations.cpu().numpy(),
                    first_source=first_output.source_indices.cpu().numpy(),
                    second_source=second_output.source_indices.cpu().numpy(),
                )
            )
            if not torch.equal(first_output.patch_valid, second_output.patch_valid):
                raise AuditFailure("source replacement changed foreground padding metadata")
        for shade in ((0, 255, 0), (0, 200, 0), (32, 255, 32)):
            view = foreground_eval_transform(
                image, mask, green_rgb=shade, **eval_options
            )
            tensor = torch.from_numpy(
                normalize_image(view.image, mean=preprocessing.mean, std=preprocessing.std)
            ).unsqueeze(0).to(device)
            with evaluation_inference(device):
                logits = model(tensor, mask_tensor).logits
            key = "_".join(map(str, shade))
            shade_differences[key].append(
                float((logits - first_output.logits).abs().max().item())
            )
            shade_prediction_changes[key] += int(
                logits.argmax(dim=1).item()
                != first_output.logits.argmax(dim=1).item()
            )
    return {
        "samples": count,
        "replacements_per_sample": 3,
        "max_replacement_logit_difference": max(differences, default=0.0),
        "green_shade_diagnostic_max_difference": {
            key: max(values, default=0.0) for key, values in shade_differences.items()
        },
        "green_shade_diagnostic_prediction_changes": shade_prediction_changes,
    }


def _background_purity_audit(
    config: dict[str, Any], mask_bank: VlmMaskBank | None = None
) -> dict[str, Any]:
    """Independently prove every retained fixed-view patch is background-only.

    This deliberately does not trust the geometry CSV's stored eligibility
    indices.  It reloads each source mask, replays the locked evaluation
    geometry, performs the locked disk dilation, and measures every retained
    source-patch occurrence in the HDF5 view bank against those pixels.
    """

    from scipy.ndimage import distance_transform_edt

    output = Path(config["paths"]["output_root"])
    artifact_root = geometry_artifact_root(config)
    bank = load_view_bank(artifact_root / "fixed_background_views.h5")
    preprocessing = load_preprocessing_manifest(output)
    image_root = Path(config["paths"]["waterbirds_root"])
    if mask_bank is None:
        mask_bank = load_vlm_mask_bank(config)
    dilation_radius = int(config["data"]["dilation_radius"])
    patch_size = PATCH_SIZE
    expected_dilation_hash = hash_object(
        {"implementation": "euclidean_disk", "radius": dilation_radius}
    )
    if bank["mask_dilation_hash"] != expected_dilation_hash:
        raise AuditFailure(
            "background view bank was generated with a different dilation rule"
        )

    split_frames = [
        pd.read_csv(output / "splits" / f"waterbirds100_{name}.csv")
        for name in ("expert_calibration", "biased_val")
    ]
    frame = pd.concat(split_frames, ignore_index=True)
    if frame["img_id"].duplicated().any():
        raise AuditFailure("background-purity source splits contain duplicate img_id values")
    records = {
        int(row.img_id): row for row in frame.itertuples(index=False)
    }
    bank_ids = np.asarray(bank["img_id"], dtype=np.int64)
    if len(np.unique(bank_ids)) != len(bank_ids):
        raise AuditFailure("background view bank contains duplicate img_id rows")
    expected_ids = set(records)
    actual_ids = set(map(int, bank_ids.tolist()))
    if actual_ids != expected_ids:
        raise AuditFailure(
            "background view bank IDs do not exactly match expert_calibration + "
            f"biased_val: missing={sorted(expected_ids - actual_ids)[:20]}, "
            f"unexpected={sorted(actual_ids - expected_ids)[:20]}"
        )

    source_views = np.asarray(bank["source_patch_indices"], dtype=np.int64)
    if source_views.ndim != 3 or source_views.shape[0] != len(bank_ids):
        raise AuditFailure("background view bank source-index tensor is malformed")
    token_budget = int(bank["token_budget"])
    if source_views.shape[2] != token_budget:
        raise AuditFailure("background view-bank token dimension differs from its budget")
    declared_invalid = set(
        map(int, np.asarray(bank["invalid_img_id"], dtype=np.int64).tolist())
    )
    declared_invalid_counts = {
        int(img_id): int(count)
        for img_id, count in zip(
            np.asarray(bank["invalid_img_id"], dtype=np.int64).tolist(),
            np.asarray(
                bank["invalid_eligible_patch_count"], dtype=np.int64
            ).tolist(),
            strict=True,
        )
    }
    if bank.get("invalid_reason_code") != "insufficient_pure_background_patches":
        raise AuditFailure("background view bank invalid-reason code is incompatible")
    if not declared_invalid.issubset(actual_ids):
        raise AuditFailure("background view bank declares unknown invalid img_id values")

    minimum_distance = float("inf")
    maximum_raw_fraction = 0.0
    maximum_dilated_fraction = 0.0
    retained_occurrences = 0
    retained_unique_pairs: set[tuple[int, int]] = set()
    valid_sample_count = 0
    invalid_examples: list[dict[str, Any]] = []

    for row_index, img_id_value in enumerate(bank_ids.tolist()):
        img_id = int(img_id_value)
        row = records[img_id]
        with Image.open(image_path(image_root, str(row.img_filename))) as opened:
            image = opened.convert("RGB")
        raw_mask = mask_bank.load(img_id, str(row.img_filename))
        transformed = deterministic_eval_transform(
            image,
            raw_mask,
            image_size=preprocessing.image_size,
            resize_shortest=preprocessing.effective_resize_shortest,
        )
        if transformed.mask.shape[0] % patch_size or transformed.mask.shape[1] % patch_size:
            raise AuditFailure(
                f"img_id={img_id} transformed mask is not patch-grid divisible"
            )
        raw_fractions = patch_fractions(transformed.mask, patch_size).reshape(-1)
        dilated = dilate_mask(transformed.mask, radius=dilation_radius)
        dilated_fractions = patch_fractions(dilated, patch_size).reshape(-1)
        safe_count = int(np.sum(dilated_fractions == 0.0))
        retained = source_views[row_index]
        has_negative = bool(np.any(retained < 0))

        if has_negative:
            if not np.all(retained == -1):
                raise AuditFailure(
                    f"img_id={img_id} has a partially populated invalid view bank row"
                )
            if img_id not in declared_invalid:
                raise AuditFailure(
                    f"img_id={img_id} has no retained views but is absent from invalid IDs"
                )
            if safe_count >= token_budget:
                raise AuditFailure(
                    f"img_id={img_id} is marked invalid despite independently "
                    f"recomputed safe patch count {safe_count} >= {token_budget}"
                )
            if declared_invalid_counts.get(img_id) != safe_count:
                raise AuditFailure(
                    f"img_id={img_id} declared invalid eligible-patch count "
                    f"{declared_invalid_counts.get(img_id)} differs from independently "
                    f"recomputed {safe_count}"
                )
            reasons = ["insufficient_pure_background_patches"]
            invalid_examples.append(
                {
                    "img_id": img_id,
                    "safe_background_patch_count": safe_count,
                    "required_token_budget": token_budget,
                    "reasons": reasons,
                }
            )
            continue

        if img_id in declared_invalid:
            raise AuditFailure(
                f"img_id={img_id} has retained views but is declared invalid"
            )
        flattened = retained.reshape(-1)
        # The hard purity assertion consumes fractions independently derived
        # from mask pixels, not a mask synthesized from stored source indices.
        assert_background_patch_purity(flattened, dilated_fractions)
        if np.any(raw_fractions[flattened] != 0.0):
            raise AuditFailure(
                f"img_id={img_id} retained a patch containing raw foreground pixels"
            )

        distance_to_raw_foreground = distance_transform_edt(~transformed.mask)
        patch_distance = distance_to_raw_foreground.reshape(
            transformed.mask.shape[0] // patch_size,
            patch_size,
            transformed.mask.shape[1] // patch_size,
            patch_size,
        ).min(axis=(1, 3)).reshape(-1)
        selected_distances = patch_distance[flattened]
        if not np.isfinite(selected_distances).all():
            raise AuditFailure(f"img_id={img_id} produced non-finite mask distances")
        minimum_distance = min(minimum_distance, float(selected_distances.min()))
        maximum_raw_fraction = max(
            maximum_raw_fraction, float(raw_fractions[flattened].max())
        )
        maximum_dilated_fraction = max(
            maximum_dilated_fraction, float(dilated_fractions[flattened].max())
        )
        retained_occurrences += int(flattened.size)
        retained_unique_pairs.update((img_id, int(value)) for value in flattened)
        valid_sample_count += 1

    observed_invalid = {int(item["img_id"]) for item in invalid_examples}
    if observed_invalid != declared_invalid:
        raise AuditFailure(
            "recomputed invalid IDs differ from view bank declarations: "
            f"observed={sorted(observed_invalid)}, declared={sorted(declared_invalid)}"
        )
    if retained_occurrences == 0 or not np.isfinite(minimum_distance):
        raise AuditFailure("background-purity audit found no retained patch occurrences")
    if maximum_dilated_fraction != 0.0:
        raise AuditFailure(
            "background-purity audit retained nonzero dilated foreground fraction"
        )
    return {
        "schema_version": "anchorcal-background-purity-audit-v2",
        "method": "reload_mask_replay_eval_transform_disk_dilate_measure_every_view_patch",
        "dilation_radius_pixels": dilation_radius,
        "patch_size_pixels": patch_size,
        "view_count": int(source_views.shape[1]),
        "token_budget": token_budget,
        "source_sample_count": int(len(bank_ids)),
        "valid_sample_count": valid_sample_count,
        "retained_patch_count": retained_occurrences,
        "retained_unique_image_source_patch_count": len(retained_unique_pairs),
        "minimum_raw_mask_distance_pixels": minimum_distance,
        "maximum_raw_foreground_fraction": maximum_raw_fraction,
        "maximum_dilated_foreground_fraction": maximum_dilated_fraction,
        "invalid_count": len(invalid_examples),
        "invalid_ids": sorted(observed_invalid),
        "invalid_examples": invalid_examples,
        "failures": 0,
    }


def _geometry_audit_seeds(config: dict[str, Any]) -> dict[str, Any]:
    """Materialize and document every geometry-audit stochastic seed."""

    split_seed = int(config["seeds"]["geometry_auditor_split"])
    model_seed = int(config["seeds"]["geometry_auditor_model"])
    return {
        "auditor_split": split_seed,
        "auditor_model": model_seed,
        "logistic_bootstrap": split_seed,
        "mlp_bootstrap": stable_seed(
            split_seed, "geometry_auditor_mlp_bootstrap"
        ),
        "mlp_bootstrap_derivation": (
            "sha256_first_8_bytes_uint32(split_seed|"
            "geometry_auditor_mlp_bootstrap)"
        ),
    }


def _geometry_audit(
    frame: pd.DataFrame,
    config: dict[str, Any],
    mask_bank: VlmMaskBank,
) -> dict[str, Any]:
    preprocessing = load_preprocessing_manifest(config["paths"]["output_root"])
    image_root = Path(config["paths"]["waterbirds_root"])
    features = []
    for row in frame.itertuples(index=False):
        with Image.open(image_path(image_root, str(row.img_filename))) as opened:
            image = opened.convert("RGB")
        mask = mask_bank.load(int(row.img_id), str(row.img_filename))
        transformed = deterministic_eval_transform(
            image,
            mask,
            image_size=preprocessing.image_size,
            resize_shortest=preprocessing.effective_resize_shortest,
        )
        features.append(list(mask_geometry(transformed.mask).values()))
    seed_manifest = _geometry_audit_seeds(config)
    split_seed = int(seed_manifest["auditor_split"])
    model_seed = int(seed_manifest["auditor_model"])
    mlp_bootstrap_seed = int(seed_manifest["mlp_bootstrap"])
    result = fit_geometry_auditors(
        np.asarray(features, dtype=np.float32),
        frame["y"].to_numpy(),
        frame["img_id"].to_numpy(),
        split_seed=split_seed,
        model_seed=model_seed,
    )
    truth = result["labels"]
    logistic_interval = stratified_bootstrap_interval(
        result["logistic_predictions"] == truth,
        truth,
        replicates=2000,
        seed=split_seed,
    )
    mlp_interval = stratified_bootstrap_interval(
        result["mlp_predictions"] == truth,
        truth,
        replicates=2000,
        seed=mlp_bootstrap_seed,
    )
    return {
        "logistic": logistic_interval.__dict__,
        "mlp": mlp_interval.__dict__,
        "test_count": int(len(truth)),
        "protocol": "image_disjoint_standardized_logistic_and_64_gelu_32_gelu_mlp",
        "seeds": seed_manifest,
    }


def _random_token_audit(
    background,
    biased_frame,
    config,
    device,
    mask_bank: VlmMaskBank,
) -> dict[str, Any]:
    import torch

    output = Path(config["paths"]["output_root"])
    preprocessing = load_preprocessing_manifest(output)
    expert_frame = pd.read_csv(output / "splits" / "waterbirds100_expert_train.csv")
    expert_dataset = EvaluationDataset(
        expert_frame,
        config["paths"]["waterbirds_root"],
        mask_bank,
        preprocessing=preprocessing,
    )
    pools: dict[int, list[torch.Tensor]] = {0: [], 1: []}
    maximum_per_class = 12000
    with evaluation_inference(device):
        for index in range(len(expert_dataset)):
            sample = expert_dataset[index]
            label = int(sample["y"])
            if sum(item.shape[0] for item in pools[label]) >= maximum_per_class:
                if all(sum(item.shape[0] for item in pools[key]) >= maximum_per_class for key in (0, 1)):
                    break
                continue
            image = sample["image"].unsqueeze(0).to(device)
            projected = background.project(image)[0]
            indices = sample["background_indices"].to(device)
            if len(indices):
                pools[label].append(projected[indices].cpu())
    count = min(sum(item.shape[0] for item in pools[0]), sum(item.shape[0] for item in pools[1]), maximum_per_class)
    if count < background.token_budget:
        raise AuditFailure("random-token audit has too few class-balanced source patches")
    balanced_pool = torch.cat(
        [torch.cat(pools[label])[:count] for label in (0, 1)], dim=0
    ).to(device)
    table = pd.read_csv(geometry_artifact_root(config) / "biased_val_geometry.csv")
    valid = table["safe_background_count"] >= background.token_budget
    labels = table.loc[valid, "y"].to_numpy(dtype=np.int64)
    predictions = []
    with evaluation_inference(device):
        for row in table.loc[valid].itertuples(index=False):
            rng = stateless_rng(
                int(config["seeds"]["random_token_audit"]),
                int(row.img_id),
                "random_token_audit",
            )
            indices = rng.choice(len(balanced_pool), background.token_budget, replace=False)
            tokens = balanced_pool[torch.from_numpy(indices).to(device)].unsqueeze(0)
            token_valid = torch.ones(
                (1, background.token_budget), dtype=torch.bool, device=device
            )
            logits, _ = background.encoder.forward_tokens(tokens, token_valid)
            predictions.append(int(logits.argmax(dim=1).item()))
    correct = np.asarray(predictions) == labels
    interval = require_random_token_collapse(
        correct,
        labels,
        seed=int(config["seeds"]["random_token_audit"]),
        replicates=2000,
    )
    return {**interval.__dict__, "recipient_count": int(len(labels)), "source_patches_per_class": int(count)}


def run_branch_audits(
    foreground,
    background,
    frame,
    config,
    device,
    mask_bank: VlmMaskBank,
) -> dict[str, Any]:
    report = {
        "foreground_invariance": _foreground_invariance_audit(
            foreground, frame, config, device, mask_bank
        ),
        "background_purity": _background_purity_audit(config, mask_bank),
        "random_token": _random_token_audit(
            background, frame, config, device, mask_bank
        ),
        "geometry_diagnostic": _geometry_audit(frame, config, mask_bank),
    }
    output = Path(config["paths"]["output_root"])
    namespace = "debug/audits" if config["runtime"]["debug"] else "audits"
    atomic_write_json(output / namespace / "branch_audits.json", report)
    return report


def _branch_extremes_and_saliency(
    foreground,
    background,
    sample,
    views: np.ndarray,
    labels: int,
    foreground_scale: float,
    background_scale: float,
    device,
):
    import torch

    foreground_image = sample["foreground_image"].unsqueeze(0).to(device)
    image = sample["image"].unsqueeze(0).to(device)
    mask = sample["mask"].unsqueeze(0).to(device)
    label = torch.tensor([labels], dtype=torch.long, device=device)
    with saliency_evaluation(device):
        foreground_output = foreground(foreground_image, mask, activation_leaf=True)
        foreground_score = (
            centered_logits(foreground_output.logits)
            / (foreground_scale + MARGIN_SCALE_EPSILON)
        ).gather(1, label[:, None]).sum()
        foreground_gradient = torch.autograd.grad(
            foreground_score, foreground_output.patch_activations, retain_graph=False
        )[0]
        foreground_signed = (
            foreground_gradient * foreground_output.patch_activations
        ).sum(dim=-1)[0, foreground_output.patch_valid[0]]
        foreground_source = foreground_output.source_indices[0, foreground_output.patch_valid[0]]
        projected = background.project(image)
        background_outputs = [
            background.forward_from_projected(
                projected,
                torch.from_numpy(views[view].astype(np.int64)).unsqueeze(0).to(device),
                activation_leaf=True,
            )
            for view in range(len(views))
        ]
        background_logits = torch.stack(
            [output.logits for output in background_outputs]
        ).mean(dim=0)
        background_score = (
            centered_logits(background_logits)
            / (background_scale + MARGIN_SCALE_EPSILON)
        ).gather(
            1, label[:, None]
        ).sum()
        gradients = torch.autograd.grad(
            background_score,
            [output.patch_activations for output in background_outputs],
            retain_graph=False,
        )
        background_signed = torch.cat(
            [
                (gradient * output.patch_activations).sum(dim=-1).flatten()
                for gradient, output in zip(gradients, background_outputs, strict=True)
            ]
        )
        background_source = torch.cat(
            [output.source_indices.flatten() for output in background_outputs]
        )
    return {
        "foreground_input_image": foreground_image.detach().float().cpu().numpy(),
        "foreground_input_mask": mask.detach().cpu().numpy(),
        "foreground_logits": foreground_output.logits.detach().float().cpu().numpy()[0],
        "foreground_patch_activations": (
            foreground_output.patch_activations.detach().float().cpu().numpy()
        ),
        "foreground_patch_valid": foreground_output.patch_valid.detach().cpu().numpy(),
        "foreground_source_indices": (
            foreground_output.source_indices.detach().cpu().numpy()
        ),
        "background_logits": background_logits.detach().float().cpu().numpy()[0],
        "foreground_signed": foreground_signed.detach().float().cpu().numpy(),
        "foreground_source": foreground_source.detach().cpu().numpy(),
        "background_signed": background_signed.detach().float().cpu().numpy(),
        "background_source": background_source.detach().cpu().numpy(),
    }


def _evaluate_foreground_stream(foreground, image, mask, device) -> dict[str, np.ndarray]:
    """Run an explicit, independently materialized foreground intervention path."""

    image_tensor = image.detach().clone().unsqueeze(0).to(device)
    mask_tensor = mask.detach().clone().unsqueeze(0).to(device)
    with evaluation_inference(device):
        result = foreground(image_tensor, mask_tensor)
    return {
        "input_image": image_tensor.detach().float().cpu().numpy(),
        "input_mask": mask_tensor.detach().cpu().numpy(),
        "logits": result.logits.detach().float().cpu().numpy()[0],
        "patch_activations": (
            result.patch_activations.detach().float().cpu().numpy()
        ),
        "patch_valid": result.patch_valid.detach().cpu().numpy(),
        "source_indices": result.source_indices.detach().cpu().numpy(),
    }


def _assert_explicit_foreground_intervention(
    intervention: InterventionType,
    clean: dict[str, np.ndarray],
    intervened: dict[str, np.ndarray],
) -> dict[str, float]:
    """Bind the typed intervention contract to complete foreground state."""

    return assert_anchor_intervention_contract(
        intervention,
        clean["logits"],
        intervened["logits"],
        clean_input_image=clean["input_image"],
        intervened_input_image=intervened["input_image"],
        clean_input_mask=clean["input_mask"],
        intervened_input_mask=intervened["input_mask"],
        clean_patch_activations=clean["patch_activations"],
        intervened_patch_activations=intervened["patch_activations"],
        clean_patch_valid=clean["patch_valid"],
        intervened_patch_valid=intervened["patch_valid"],
        clean_source_indices=clean["source_indices"],
        intervened_source_indices=intervened["source_indices"],
    )


def _direct_lambda_saliency(
    foreground,
    background,
    sample,
    views: np.ndarray,
    label_value: int,
    foreground_scale: float,
    background_scale: float,
    reliance_lambda: float,
    device,
) -> dict[str, Any]:
    """Differentiate the final mixed logit directly for cache parity."""

    import torch

    label = torch.tensor([label_value], dtype=torch.long, device=device)
    with saliency_evaluation(device):
        foreground_output = foreground(
            sample["foreground_image"].unsqueeze(0).to(device),
            sample["mask"].unsqueeze(0).to(device),
            activation_leaf=True,
        )
        projected = background.project(sample["image"].unsqueeze(0).to(device))
        background_outputs = [
            background.forward_from_projected(
                projected,
                torch.from_numpy(views[view].astype(np.int64)).unsqueeze(0).to(device),
                activation_leaf=True,
            )
            for view in range(len(views))
        ]
        background_logits = torch.stack(
            [output.logits for output in background_outputs]
        ).mean(dim=0)
        logits = mix_anchor_logits(
            foreground_output.logits,
            background_logits,
            reliance_lambda,
            foreground_scale,
            background_scale,
        )
        score = logits.gather(1, label[:, None]).sum()
        activation_list = [foreground_output.patch_activations] + [
            output.patch_activations for output in background_outputs
        ]
        gradients = torch.autograd.grad(score, activation_list)
        foreground_valid = foreground_output.patch_valid[0]
        foreground_signed = (
            gradients[0] * foreground_output.patch_activations
        ).sum(dim=-1)[0, foreground_valid]
        background_signed = torch.cat(
            [
                (gradient * output.patch_activations).sum(dim=-1).flatten()
                for gradient, output in zip(
                    gradients[1:], background_outputs, strict=True
                )
            ]
        )
    return {
        "logits": logits.detach().float().cpu().numpy()[0],
        "foreground_signed": foreground_signed.detach().float().cpu().numpy(),
        "foreground_source": foreground_output.source_indices[
            0, foreground_valid
        ].detach().cpu().numpy(),
        "background_signed": background_signed.detach().float().cpu().numpy(),
        "background_source": torch.cat(
            [output.source_indices.flatten() for output in background_outputs]
        ).detach().cpu().numpy(),
    }


def evaluate_anchor_ladder(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    seed_everything(int(config["seeds"]["anchor_bootstrap"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(config["paths"]["output_root"])
    preprocessing = load_preprocessing_manifest(output)
    preflight_path = output / "preflight" / "report.json"
    if not preflight_path.is_file():
        raise PreflightError("successful preflight report is required before anchors")
    with preflight_path.open("r", encoding="utf-8") as handle:
        preflight_report = json.load(handle)
    if (
        preflight_report.get("status") != "passed"
        or preflight_report.get("resolved_paths") != config["paths"]
        or (
            not config["runtime"]["debug"]
            and preflight_report.get("resolved_config_sha256")
            != config["resolved_config_sha256"]
        )
    ):
        raise PreflightError("anchor config/path provenance differs from preflight")
    mask_bank = load_vlm_mask_bank(config)
    foreground, background, foreground_checkpoint, background_checkpoint = _load_branch_models(
        config, device
    )
    biased_frame = pd.read_csv(output / "splits" / "waterbirds100_biased_val.csv")
    run_branch_audits(
        foreground, background, biased_frame, config, device, mask_bank
    )
    joined = _join_branch_outputs(_load_outputs(config, "foreground"), _load_outputs(config, "background"))
    competence = construct_competence_intersection(
        joined["img_id"],
        joined["labels"],
        joined["foreground_logits"],
        joined["background_logits"],
        joined["background_valid"],
        minimum_per_class=int(config["anchorcal"]["minimum_intersection_per_class"]),
    )
    geometry = pd.read_csv(geometry_artifact_root(config) / "biased_val_geometry.csv")
    eligible_ids = set(
        geometry.loc[geometry["common_eligible"], "img_id"].astype(int).tolist()
    )
    subset_ids = cap_anchor_subset(
        competence,
        eligible_ids,
        per_class=int(config["anchorcal"]["anchor_subset_per_class"]),
        seed=int(config["seeds"]["anchor_subset"]),
        minimum_per_class=int(config["anchorcal"]["minimum_intersection_per_class"]),
    )
    namespace = "debug/anchors" if config["runtime"]["debug"] else "anchors"
    anchor_root = output / namespace
    anchor_root.mkdir(parents=True, exist_ok=True)
    foreground_margins = true_margin(joined["foreground_logits"], joined["labels"])
    background_margins = true_margin(joined["background_logits"], joined["labels"])
    competence_member = np.isin(joined["img_id"], competence.img_ids)
    competence_frame = pd.DataFrame(
        {
            "img_id": joined["img_id"][competence_member],
            "y": joined["labels"][competence_member],
            "foreground_raw_margin": foreground_margins[competence_member],
            "background_raw_margin": background_margins[competence_member],
        }
    ).sort_values("img_id", kind="stable")
    competence_path = anchor_root / "competence_intersection.csv"
    atomic_write_text(
        competence_path,
        competence_frame.to_csv(index=False, lineterminator="\n"),
    )
    class_counts = {
        str(label): int((competence_frame["y"] == label).sum()) for label in (0, 1)
    }
    atomic_write_json(
        anchor_root / "competence_intersection_manifest.json",
        {
            "definition": "background_valid_and_foreground_raw_margin_gt_0_and_background_raw_margin_gt_0",
            "count": int(len(competence_frame)),
            "biased_val_count": int(len(joined["img_id"])),
            "retained_fraction": float(len(competence_frame) / len(joined["img_id"])),
            "per_class_count": class_counts,
            "per_class_retained_fraction": {
                str(label): float(
                    class_counts[str(label)]
                    / max(1, int(np.sum(joined["labels"] == label)))
                )
                for label in (0, 1)
            },
            "invalid_background_exclusions": int(
                np.sum(~joined["background_valid"].astype(bool))
            ),
            "source_hash": competence.source_hash,
        },
    )
    subset_frame = biased_frame.loc[biased_frame["img_id"].isin(subset_ids)].sort_values("img_id")
    criterion_subset_path = anchor_root / "criterion_subset.csv"
    atomic_write_text(
        criterion_subset_path,
        subset_frame.to_csv(index=False, lineterminator="\n"),
    )
    scales = {
        "foreground": competence.foreground_scale,
        "background": competence.background_scale,
        "full_intersection_count": int(len(competence.img_ids)),
        "full_intersection_hash": competence.source_hash,
        "criterion_subset_hash": hash_object(subset_ids.tolist()),
        "criterion_subset_seed": int(config["seeds"]["anchor_subset"]),
        "criterion_subset_per_class": {
            str(label): int((subset_frame["y"] == label).sum()) for label in (0, 1)
        },
        "criterion_subset_cap_per_class": int(
            config["anchorcal"]["anchor_subset_per_class"]
        ),
    }
    foreground_competence_margins = foreground_margins[competence_member]
    background_competence_margins = background_margins[competence_member]
    foreground_normalized = foreground_competence_margins / (
        competence.foreground_scale + MARGIN_SCALE_EPSILON
    )
    background_normalized = background_competence_margins / (
        competence.background_scale + MARGIN_SCALE_EPSILON
    )

    def margin_diagnostics(raw: np.ndarray, normalized: np.ndarray) -> dict[str, Any]:
        raw_counts, raw_edges = np.histogram(raw, bins=20)
        normalized_counts, normalized_edges = np.histogram(normalized, bins=20)
        return {
            "raw_median": float(np.median(raw)),
            "normalized_median": float(np.median(normalized)),
            "normalized_positive_fraction": float(np.mean(normalized > 0.0)),
            "raw_histogram": {
                "counts": raw_counts.astype(int).tolist(),
                "bin_edges": raw_edges.tolist(),
            },
            "normalized_histogram": {
                "counts": normalized_counts.astype(int).tolist(),
                "bin_edges": normalized_edges.tolist(),
            },
        }

    scales["normalization_epsilon"] = MARGIN_SCALE_EPSILON
    scales["diagnostics"] = {
        "foreground": margin_diagnostics(
            foreground_competence_margins, foreground_normalized
        ),
        "background": margin_diagnostics(
            background_competence_margins, background_normalized
        ),
    }
    margin_scales_path = anchor_root / "margin_scales.json"
    atomic_write_json(margin_scales_path, scales)
    dataset = EvaluationDataset(
        subset_frame,
        config["paths"]["waterbirds_root"],
        mask_bank,
        preprocessing=preprocessing,
    )
    views = _view_lookup(config)
    with (geometry_artifact_root(config) / "donor_assignments.json").open(
        "r", encoding="utf-8"
    ) as handle:
        donor_payload = {int(item["recipient_id"]): item for item in json.load(handle)}
    background_by_id = {
        int(img_id): logits
        for img_id, logits, valid in zip(
            joined["img_id"],
            joined["background_logits"],
            joined["background_valid"],
            strict=True,
        )
        if bool(valid) and np.isfinite(logits).all()
    }
    lambdas = np.asarray(config["anchorcal"]["lambdas"], dtype=np.float64)
    labels = subset_frame["y"].to_numpy(dtype=np.int64)
    image_count = len(dataset)
    saliency_values = np.empty((image_count, len(lambdas)), dtype=np.float64)
    saliency_fallback_absolute = np.zeros(
        (image_count, len(lambdas)), dtype=bool
    )
    saliency_zero_attribution = np.zeros(
        (image_count, len(lambdas)), dtype=bool
    )
    clean_correct = np.empty_like(saliency_values)
    swap_values = np.empty_like(saliency_values)
    swap_margin_drop = np.empty_like(saliency_values)
    swap_prediction_flip = np.empty_like(saliency_values)
    swap_donor_margin_variance = np.empty_like(saliency_values)
    blur_values = np.empty_like(saliency_values)
    foreground_only_values = np.empty_like(saliency_values)
    foreground_logits_all = np.empty((image_count, 2), dtype=np.float32)
    background_logits_all = np.empty((image_count, 2), dtype=np.float32)
    parity: dict[str, dict[str, float | bool]] = {}
    parity_sample_count = min(4, image_count)
    repeated_coordinate_parity_cases = 0
    foreground_stream_differences: dict[str, dict[str, float]] = {
        intervention.value: {}
        for intervention in (
            InterventionType.TOKEN_SWAP_BACKGROUND,
            InterventionType.BLUR_BACKGROUND,
            InterventionType.FOREGROUND_ONLY_GREENSCREEN,
        )
    }

    def record_foreground_stream_audit(
        intervention: InterventionType, diagnostics: dict[str, float]
    ) -> None:
        aggregate = foreground_stream_differences[intervention.value]
        for name, value in diagnostics.items():
            aggregate[name] = max(float(value), aggregate.get(name, 0.0))

    for index in range(image_count):
        sample = dataset[index]
        img_id = int(sample["img_id"])
        label = int(sample["y"])
        extreme = _branch_extremes_and_saliency(
            foreground,
            background,
            sample,
            views[img_id],
            label,
            competence.foreground_scale,
            competence.background_scale,
            device,
        )
        foreground_logits_all[index] = extreme["foreground_logits"]
        background_logits_all[index] = extreme["background_logits"]
        cache = ExtremeCache(
            extreme["foreground_logits"][None, :],
            extreme["background_logits"][None, :],
        )
        clean_foreground_stream = {
            "input_image": extreme["foreground_input_image"].copy(),
            "input_mask": extreme["foreground_input_mask"].copy(),
            "logits": extreme["foreground_logits"].copy(),
            "patch_activations": extreme["foreground_patch_activations"].copy(),
            "patch_valid": extreme["foreground_patch_valid"].copy(),
            "source_indices": extreme["foreground_source_indices"].copy(),
        }
        foreground_indices = sample["foreground_indices"].numpy()
        background_indices = sample["background_indices"].numpy()
        donor_ids = [
            int(item["donor_id"]) for item in donor_payload[img_id]["donors"]
        ]
        missing_donors = [value for value in donor_ids if value not in background_by_id]
        if missing_donors:
            raise AuditFailure(
                f"anchor recipient {img_id} has fixed-K-invalid background donors: "
                f"{missing_donors}"
            )
        donor_background = np.stack([background_by_id[value] for value in donor_ids])
        if not np.isfinite(donor_background).all():
            raise AuditFailure(f"anchor recipient {img_id} has non-finite donor logits")
        # Token swapping is background-only.  Re-run a separately materialized
        # foreground path and bind both its inputs and complete outputs to the
        # clean stream before consuming the swapped background evidence.
        token_swap_foreground = _evaluate_foreground_stream(
            foreground,
            sample["foreground_image"],
            sample["mask"],
            device,
        )
        record_foreground_stream_audit(
            InterventionType.TOKEN_SWAP_BACKGROUND,
            _assert_explicit_foreground_intervention(
                InterventionType.TOKEN_SWAP_BACKGROUND,
                clean_foreground_stream,
                token_swap_foreground,
            ),
        )
        # Background blur reuses the clean source-index views.
        blurred_background_logits = []
        for sigma in config["criteria"]["blur_sigmas"]:
            blurred = mask_normalized_background_blur(
                sample["unnormalized_image"], sample["mask"].numpy(), sigma=float(sigma)
            )
            blurred_tensor = torch.from_numpy(
                normalize_image(
                    blurred,
                    mean=preprocessing.mean,
                    std=preprocessing.std,
                )
            ).unsqueeze(0).to(device)
            view_tensor = torch.from_numpy(views[img_id].astype(np.int64)).to(device)
            with evaluation_inference(device):
                projected = background.project(blurred_tensor)
                logits = torch.stack(
                    [
                        background.forward_from_projected(
                            projected, view_tensor[view].unsqueeze(0)
                        ).logits
                        for view in range(len(view_tensor))
                    ]
                ).mean(dim=0)
            blurred_background_logits.append(logits.float().cpu().numpy()[0])
        blurred_background = np.stack(blurred_background_logits)
        blur_foreground = _evaluate_foreground_stream(
            foreground,
            sample["foreground_image"],
            sample["mask"],
            device,
        )
        record_foreground_stream_audit(
            InterventionType.BLUR_BACKGROUND,
            _assert_explicit_foreground_intervention(
                InterventionType.BLUR_BACKGROUND,
                clean_foreground_stream,
                blur_foreground,
            ),
        )
        # Foreground-only runs the background stream on green pixels at fixed locations.
        green_tensor = sample["foreground_image"].unsqueeze(0).to(device)
        view_tensor = torch.from_numpy(views[img_id].astype(np.int64)).to(device)
        with evaluation_inference(device):
            projected_green = background.project(green_tensor)
            green_background = torch.stack(
                [
                    background.forward_from_projected(
                        projected_green, view_tensor[view].unsqueeze(0)
                    ).logits
                    for view in range(len(view_tensor))
                ]
            ).mean(dim=0).float().cpu().numpy()[0]
        foreground_only_foreground = _evaluate_foreground_stream(
            foreground,
            sample["foreground_image"],
            sample["mask"],
            device,
        )
        record_foreground_stream_audit(
            InterventionType.FOREGROUND_ONLY_GREENSCREEN,
            _assert_explicit_foreground_intervention(
                InterventionType.FOREGROUND_ONLY_GREENSCREEN,
                clean_foreground_stream,
                foreground_only_foreground,
            ),
        )
        for lambda_index, reliance_lambda in enumerate(lambdas):
            mixed = cached_logits(
                cache,
                float(reliance_lambda),
                competence.foreground_scale,
                competence.background_scale,
            )[0]
            clean_correct[index, lambda_index] = float(np.argmax(mixed) == label)
            saliency_result = anchor_image_alignment(
                extreme["foreground_signed"],
                extreme["foreground_source"],
                extreme["background_signed"],
                extreme["background_source"],
                foreground_indices,
                background_indices,
                float(reliance_lambda),
            )
            saliency_values[index, lambda_index] = saliency_result.alignment
            saliency_fallback_absolute[index, lambda_index] = (
                saliency_result.fallback_absolute
            )
            saliency_zero_attribution[index, lambda_index] = (
                saliency_result.zero_scored_attribution
            )
            donor_mixed = (
                float(reliance_lambda)
                * (extreme["foreground_logits"] - extreme["foreground_logits"].mean())
                / (competence.foreground_scale + MARGIN_SCALE_EPSILON)
                + (1.0 - float(reliance_lambda))
                * (donor_background - donor_background.mean(axis=1, keepdims=True))
                / (competence.background_scale + MARGIN_SCALE_EPSILON)
            )
            swap_values[index, lambda_index] = np.mean(
                donor_mixed.argmax(axis=1) == label
            )
            clean_margin = float(mixed[label] - mixed[1 - label])
            donor_margins = donor_mixed[:, label] - donor_mixed[:, 1 - label]
            swap_margin_drop[index, lambda_index] = clean_margin - float(
                donor_margins.mean()
            )
            swap_prediction_flip[index, lambda_index] = float(
                np.mean(donor_mixed.argmax(axis=1) != int(np.argmax(mixed)))
            )
            swap_donor_margin_variance[index, lambda_index] = float(
                donor_margins.var()
            )
            blur_mixed = (
                float(reliance_lambda)
                * (extreme["foreground_logits"] - extreme["foreground_logits"].mean())
                / (competence.foreground_scale + MARGIN_SCALE_EPSILON)
                + (1.0 - float(reliance_lambda))
                * (blurred_background - blurred_background.mean(axis=1, keepdims=True))
                / (competence.background_scale + MARGIN_SCALE_EPSILON)
            )
            blur_values[index, lambda_index] = np.mean(blur_mixed.argmax(axis=1) == label)
            fg_only_mixed = (
                float(reliance_lambda)
                * (extreme["foreground_logits"] - extreme["foreground_logits"].mean())
                / (competence.foreground_scale + MARGIN_SCALE_EPSILON)
                + (1.0 - float(reliance_lambda))
                * (green_background - green_background.mean())
                / (competence.background_scale + MARGIN_SCALE_EPSILON)
            )
            foreground_only_values[index, lambda_index] = float(
                np.argmax(fg_only_mixed) == label
            )
            if index < parity_sample_count and float(reliance_lambda) in set(
                map(float, config["anchorcal"]["parity_lambdas"])
            ):
                direct_saliency = _direct_lambda_saliency(
                    foreground,
                    background,
                    sample,
                    views[img_id],
                    label,
                    competence.foreground_scale,
                    competence.background_scale,
                    float(reliance_lambda),
                    device,
                )
                direct_alignment = image_alignment(
                    np.concatenate(
                        [
                            direct_saliency["foreground_signed"],
                            direct_saliency["background_signed"],
                        ]
                    ),
                    np.concatenate(
                        [
                            direct_saliency["foreground_source"],
                            direct_saliency["background_source"],
                        ]
                    ),
                    foreground_indices,
                    background_indices,
                ).alignment
                cached_alignment = saliency_values[index, lambda_index]
                parity_key = f"img{img_id}_lambda{float(reliance_lambda):.2f}"
                has_repeated_coordinate = bool(
                    len(np.unique(extreme["background_source"]))
                    < len(extreme["background_source"])
                )
                if has_repeated_coordinate:
                    repeated_coordinate_parity_cases += 1
                parity[parity_key] = {
                    "repeated_background_coordinate": has_repeated_coordinate,
                    "logits_max_abs": require_cache_parity(
                        direct_saliency["logits"], mixed, tolerance=1e-6, quantity="logits"
                    ),
                    "foreground_saliency_max_abs": require_cache_parity(
                        direct_saliency["foreground_signed"],
                        float(reliance_lambda) * extreme["foreground_signed"],
                        tolerance=1e-5,
                        quantity="foreground saliency",
                    ),
                    "background_saliency_max_abs": require_cache_parity(
                        direct_saliency["background_signed"],
                        (1.0 - float(reliance_lambda)) * extreme["background_signed"],
                        tolerance=1e-5,
                        quantity="background saliency",
                    ),
                    "criterion_abs": require_cache_parity(
                        np.asarray([direct_alignment]),
                        np.asarray([cached_alignment]),
                        tolerance=1e-6,
                        quantity="criterion",
                    ),
                }
    expected_parity_cases = parity_sample_count * len(
        config["anchorcal"]["parity_lambdas"]
    )
    if len(parity) != expected_parity_cases:
        raise AuditFailure(
            f"cache parity covered {len(parity)}/{expected_parity_cases} required cases"
        )
    if repeated_coordinate_parity_cases == 0:
        raise AuditFailure(
            "cache parity did not include a repeated background source coordinate"
        )
    if not np.all(clean_correct == 1.0):
        failures = int(np.sum(clean_correct != 1.0))
        raise AuditFailure(f"anchor correctness invariant failed {failures} times")
    per_image = {
        "ordinary_accuracy": clean_correct,
        "saliency_harmonic": saliency_values,
        "token_swap_harmonic": swap_values,
        "background_blur_harmonic": blur_values,
        "foreground_only_harmonic": foreground_only_values,
        # Anchor clean accuracy is exactly one, so the product diagnostic is
        # the class-balanced intervention component itself.
        "saliency_product": saliency_values,
        "swap_product": swap_values,
        "blur_product": blur_values,
    }
    score_vectors: dict[str, np.ndarray] = {}
    for name, values in per_image.items():
        base = class_balanced_mean(values, labels)
        if name == "ordinary_accuracy" or name.endswith("_product"):
            score_vectors[name] = base
        else:
            score_vectors[name] = 2.0 * base / (
                1.0 + base + HARMONIC_EPSILON
            )
    point_metrics = {
        name: evaluate_anchor_scores(
            score_vectors[name],
            lambdas,
            tolerance=float(config["anchorcal"]["anchor_score_tolerance"]),
        )
        for name in score_vectors
    }
    transforms = {
        name: (
            lambda values: 2.0
            * values
            / (1.0 + values + HARMONIC_EPSILON)
        )
        for name in score_vectors
        if name != "ordinary_accuracy" and not name.endswith("_product")
    }
    bootstrap = paired_class_stratified_bootstrap(
        labels,
        per_image,
        lambdas,
        replicates=int(config["anchorcal"]["bootstrap_replicates"]),
        seed=int(config["seeds"]["anchor_bootstrap"]),
        tolerance=float(config["anchorcal"]["anchor_score_tolerance"]),
        score_transforms=transforms,
    )
    costs = {
        "ordinary_accuracy": 1.0,
        "saliency_harmonic": 4.0,
        "token_swap_harmonic": 3.0,
        "background_blur_harmonic": 2.0,
    }
    decision = choose_criterion(
        point_metrics,
        bootstrap.criteria,
        eligible_criteria=config["criteria"]["eligible"],
        computational_cost=costs,
        tolerance=float(config["anchorcal"]["anchor_score_tolerance"]),
    )
    per_image_path = anchor_root / "anchor_per_image_outputs.npz"
    np.savez_compressed(
        per_image_path,
        img_id=subset_frame["img_id"].to_numpy(),
        labels=labels,
        lambdas=lambdas,
        foreground_logits=foreground_logits_all,
        background_logits=background_logits_all,
        saliency_fallback_absolute=saliency_fallback_absolute,
        saliency_zero_attribution=saliency_zero_attribution,
        swap_margin_drop=swap_margin_drop,
        swap_prediction_flip=swap_prediction_flip,
        swap_donor_margin_variance=swap_donor_margin_variance,
        **per_image,
        bootstrap_indices=bootstrap.indices,
        intervention_types=np.asarray(
            [
                InterventionType.TOKEN_SWAP_BACKGROUND.value,
                InterventionType.BLUR_BACKGROUND.value,
                InterventionType.FOREGROUND_ONLY_GREENSCREEN.value,
            ]
        ),
    )
    bootstrap_vectors_path = anchor_root / "anchor_bootstrap_score_vectors.npz"
    np.savez_compressed(
        bootstrap_vectors_path,
        lambdas=lambdas,
        **{
            name: result.score_vectors
            for name, result in bootstrap.criteria.items()
        },
    )
    point_payload = {
        name: {
            "scores": score_vectors[name].tolist(),
            "metrics": point_metrics[name].to_dict(),
            "bootstrap": {
                key: summary.to_dict()
                for key, summary in bootstrap.criteria[name].summaries().items()
            },
        }
        for name in score_vectors
    }
    criterion_results_path = anchor_root / "criterion_results.json"
    atomic_write_json(criterion_results_path, point_payload)
    anchor_scores_path = anchor_root / "anchor_scores.csv"
    anchor_scores_frame = pd.DataFrame(
        [
            {
                "criterion": name,
                "lambda": float(reliance_lambda),
                "score": float(score_vectors[name][lambda_index]),
                "eligible_for_decision": name in config["criteria"]["eligible"],
            }
            for name in score_vectors
            for lambda_index, reliance_lambda in enumerate(lambdas)
        ]
    )
    atomic_write_text(
        anchor_scores_path,
        anchor_scores_frame.to_csv(index=False, lineterminator="\n"),
    )
    bootstrap_metrics_path = anchor_root / "anchor_bootstrap_metrics.csv"
    bootstrap_metrics_frame = pd.DataFrame(
        [
            {
                "criterion": name,
                "metric": metric,
                **summary.to_dict(),
                "ace_endpoint_clipping_floor_retained": metric == "ace",
            }
            for name, result in bootstrap.criteria.items()
            for metric, summary in result.summaries().items()
        ]
    )
    atomic_write_text(
        bootstrap_metrics_path,
        bootstrap_metrics_frame.to_csv(index=False, lineterminator="\n"),
    )
    intervention_diagnostics_path = anchor_root / "anchor_intervention_diagnostics.csv"
    intervention_diagnostics_frame = pd.DataFrame(
        [
            {
                "lambda": float(reliance_lambda),
                "swap_mean_true_class_margin_drop": float(
                    class_balanced_mean(swap_margin_drop[:, index], labels)
                ),
                "swap_prediction_flip_rate": float(
                    class_balanced_mean(swap_prediction_flip[:, index], labels)
                ),
                "swap_donor_margin_variance": float(
                    class_balanced_mean(
                        swap_donor_margin_variance[:, index], labels
                    )
                ),
            }
            for index, reliance_lambda in enumerate(lambdas)
        ]
    )
    atomic_write_text(
        intervention_diagnostics_path,
        intervention_diagnostics_frame.to_csv(index=False, lineterminator="\n"),
    )
    parity_payload = {
        "sample_count": parity_sample_count,
        "repeated_coordinate_cases": repeated_coordinate_parity_cases,
        "cases": parity,
    }
    cache_parity_path = anchor_root / "cache_parity.json"
    atomic_write_json(cache_parity_path, parity_payload)
    foreground_stream_audit_path = (
        anchor_root / "foreground_stream_intervention_audit.json"
    )
    atomic_write_json(
        foreground_stream_audit_path,
        {
            "schema_version": "anchorcal-foreground-stream-intervention-audit-v1",
            "sample_count": image_count,
            "contract": (
                "exact_input_image_mask_and_token_metadata;"
                "finite_logits_and_patch_activations_with_abs_tolerance_1e-6"
            ),
            "interventions": foreground_stream_differences,
            "failures": 0,
        },
    )
    audit_namespace = "debug/audits" if config["runtime"]["debug"] else "audits"
    branch_audits_path = output / audit_namespace / "branch_audits.json"
    competence_manifest_path = anchor_root / "competence_intersection_manifest.json"
    artifact_paths = {
        "branch_audits": branch_audits_path,
        "competence_intersection": competence_path,
        "competence_intersection_manifest": competence_manifest_path,
        "criterion_subset": criterion_subset_path,
        "margin_scales": margin_scales_path,
        "anchor_per_image_outputs": per_image_path,
        "anchor_bootstrap_score_vectors": bootstrap_vectors_path,
        "criterion_results": criterion_results_path,
        "anchor_scores": anchor_scores_path,
        "anchor_bootstrap_metrics": bootstrap_metrics_path,
        "anchor_intervention_diagnostics": intervention_diagnostics_path,
        "cache_parity": cache_parity_path,
        "foreground_stream_intervention_audit": foreground_stream_audit_path,
    }
    for name, path in artifact_paths.items():
        if not path.is_file():
            raise AuditFailure(f"required AnchorCal artifact was not written: {name}")
    artifact_manifest_path = anchor_root / "artifact_manifest.json"
    atomic_write_json(
        artifact_manifest_path,
        {
            "schema_version": "anchorcal-anchor-artifacts-v1",
            "resolved_config_sha256": config["resolved_config_sha256"],
            "criterion_result_keys": sorted(score_vectors),
            "definitions": {
                "margin_normalization": "centered_raw_logits_divided_by_(competence_median_true_margin_plus_epsilon)",
                "margin_normalization_epsilon": MARGIN_SCALE_EPSILON,
                "harmonic_mean": "2*a*b/(a+b+epsilon), zero if either input is zero",
                "harmonic_mean_epsilon": HARMONIC_EPSILON,
                "swap_margin_drop": "clean_normalized_true_class_margin_minus_mean_donor_normalized_true_class_margin",
                "swap_prediction_flip_rate": "mean_donor_indicator(swapped_argmax_differs_from_clean_argmax)",
                "swap_donor_margin_variance": "population_variance_ddof_0_of_donor_normalized_true_class_margins",
                "product_diagnostic": "clean_class_balanced_accuracy_times_intervention_class_balanced_score",
            },
            "files": {
                name: {
                    "path": str(path.resolve().relative_to(output.resolve())),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for name, path in sorted(artifact_paths.items())
            },
        },
    )
    selector_mask_receipt_path = (
        output / "preflight" / "selector_mask_receipt.json"
    )
    selector_mask_receipt_binding = preflight_report.get(
        "selector_mask_receipt"
    )
    if (
        not selector_mask_receipt_path.is_file()
        or not isinstance(selector_mask_receipt_binding, dict)
        or selector_mask_receipt_binding.get("sha256")
        != sha256_file(selector_mask_receipt_path)
    ):
        raise PreflightError(
            "anchor stage lacks a verified selector-safe mask receipt"
        )
    receipt_root = output / ("debug/receipt" if config["runtime"]["debug"] else "receipt")
    receipt = write_decision_receipt(
        receipt_root,
        decision,
        formulas={
            "ordinary_accuracy": "class_balanced_clean_accuracy",
            "saliency_harmonic": "H_eps1e-8(clean_accuracy,class_balanced_saliency_alignment)",
            "token_swap_harmonic": "H_eps1e-8(clean_accuracy,class_balanced_mean_donor_correctness)",
            "background_blur_harmonic": "H_eps1e-8(clean_accuracy,class_balanced_mean_sigma_correctness)",
        },
        anchor_subset_hash=hash_object(subset_ids.tolist()),
        anchor_family={
            "type": "normalized_centered_raw_logit_reliance",
            "lambdas": lambdas.tolist(),
            "foreground_scale": competence.foreground_scale,
            "background_scale": competence.background_scale,
            "normalization_epsilon": MARGIN_SCALE_EPSILON,
            "harmonic_mean_epsilon": HARMONIC_EPSILON,
        },
        branch_hashes={
            "foreground": sha256_file(foreground_checkpoint),
            "background": sha256_file(background_checkpoint),
        },
        config_hashes={"resolved": config["resolved_config_sha256"]},
        extra_provenance={
            "parity": parity_payload,
            "preflight_report": str(preflight_path.resolve()),
            "preflight_report_sha256": sha256_file(preflight_path),
            "metadata_sha256": preflight_report["metadata_sha256"],
            "mask_bank_sha256": preflight_report["mask_bank_sha256"],
            "mask_manifest_sha256": preflight_report["mask_manifest_sha256"],
            "mask_source": preflight_report["mask_source"],
            "mask_contract": dict(config["masks"]),
            "selector_mask_receipt": str(
                selector_mask_receipt_path.resolve()
            ),
            "selector_mask_receipt_sha256": sha256_file(
                selector_mask_receipt_path
            ),
            "preprocessing_manifest_sha256": preflight_report["preprocessing"][
                "manifest_sha256"
            ],
            "git_commit": preflight_report["git"]["commit"],
            "paths": dict(config["paths"]),
            "seeds": dict(config["seeds"]),
            "criterion_results": str(criterion_results_path.resolve()),
            "criterion_results_sha256": sha256_file(criterion_results_path),
            "anchor_artifact_manifest": str(artifact_manifest_path.resolve()),
            "anchor_artifact_manifest_sha256": sha256_file(
                artifact_manifest_path
            ),
            "runtime_job_receipt": os.environ.get("ANCHORCAL_JOB_RECEIPT"),
            "runtime_job_receipt_sha256": os.environ.get(
                "ANCHORCAL_JOB_RECEIPT_SHA256"
            ),
            "nondeterminism_warning_record": (
                "job-receipt stderr_log"
                if os.environ.get("ANCHORCAL_JOB_RECEIPT")
                else "process stderr"
            ),
        },
    )
    return {
        "winner": decision.winner,
        "credible_set": list(decision.credible_set),
        "receipt": str(receipt.receipt),
        "receipt_sha256": sha256_file(receipt.receipt),
        "subset_count": int(len(subset_frame)),
    }
