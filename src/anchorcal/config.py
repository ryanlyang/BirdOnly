"""Resolved configuration loading and locked-value validation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .errors import ConfigurationError
from .io import hash_object, read_yaml


REQUIRED_PATH_KEYS = (
    "repo_root",
    "waterbirds_root",
    "metadata_path",
    "vlm_mask_root",
    "fcv_split_manifest_root",
    "hf_home",
    "output_root",
)

LOCKED = {
    ("schema_version",): "anchorcal-config-v4",
    ("data", "release"): "waterbird_1.0_forest2water2",
    (
        "data",
        "waterbirds100_definition",
    ): "complete official split 0; hard-assert y == place",
    ("data", "split_manifest_schema"): "anchorcal-splits-v4",
    ("data", "development_membership_source"): "waterbirds100_fcv_seed0",
    ("data", "development_split_seed"): 0,
    ("data", "reuse_frozen_membership"): True,
    ("data", "selector_safe_split_root"): "splits",
    ("data", "protected_split_root"): "analysis_only/splits",
    ("data", "selector_safe_excludes"): [
        "metadata_index",
        "place",
        "group",
        "group_name",
    ],
    ("data", "image_size"): 224,
    ("data", "patch_size"): 16,
    ("data", "expert_calibration_seed"): 2718,
    ("data", "dilation_radius"): 8,
    ("data", "candidate_train_fraction"): 0.80,
    ("data", "expert_calibration_fraction"): 0.10,
    ("data", "fcv_reference", "study_id"): "fcv_vit_waterbirds100_first_study",
    ("data", "fcv_reference", "protocol_version"): "1",
    (
        "data",
        "fcv_reference",
        "source_metadata_sha256",
    ): "220b3b54cc65fd195a6d1f4499f970cd143eca1a9ffbd948e8e8c6d86d366694",
    ("data", "fcv_reference", "source_train_count"): 4795,
    ("data", "fcv_reference", "candidate_train_count"): 3836,
    ("data", "fcv_reference", "biased_val_count"): 959,
    (
        "data",
        "fcv_reference",
        "manifest_bundle_sha256",
    ): "10051eaa3f898abebeced3e4445b744f8e84e813d13888398564b01bf2d28cc5",
    (
        "data",
        "fcv_reference",
        "split_indices_sha256",
    ): "e26200da47d2810748fe386b3367752039f6e46d67028f3eabadff4dc8adc13f",
    (
        "data",
        "fcv_reference",
        "candidate_train_csv_sha256",
    ): "5429550ba0cff705ec78a7a98f18128c0c8232f42bbbaac7351d74300a7dc114",
    (
        "data",
        "fcv_reference",
        "biased_val_csv_sha256",
    ): "3856d6cd33455a00de56a5306433629ec00f9f72cd85434dc20e06d878e2a6ca",
    (
        "data",
        "fcv_reference",
        "candidate_train_metadata_indices_sha256",
    ): "b890276b5a5297c289a71b5081524882865827f01a93639196a05512534ba857",
    (
        "data",
        "fcv_reference",
        "biased_val_metadata_indices_sha256",
    ): "10bcd762bd4937aac0ea526915f09773c93f24105db024ab869a936a6a5c7376",
    ("data", "evaluation_crop_pct"): 0.9,
    ("data", "normalization_mean"): [0.5, 0.5, 0.5],
    ("data", "normalization_std"): [0.5, 0.5, 0.5],
    ("data", "random_resized_crop_scale"): [0.70, 1.00],
    ("data", "random_resized_crop_ratio"): [0.75, 1.3333333333],
    ("data", "crop_max_attempts"): 10,
    (
        "masks",
        "source",
    ): "waterbirds100_openclip_laion_dinovit_weclipplus_prediction_cmap",
    ("masks", "mapping_mode"): "weclip_producer_first_with_explicit_legacy_fallbacks",
    ("masks", "mapping_version"): "weclip-img-filename-v1",
    ("masks", "decoder_version"): "pascal-voc-rgb-class-id-v1",
    ("masks", "manifest_schema"): "anchorcal-vlm-mask-manifest-v3",
    ("masks", "format"): "voc_colormap_class_ids",
    ("masks", "foreground_class_ids"): [1],
    ("masks", "allowed_class_ids"): [0, 1],
    ("masks", "interpolation"): "nearest",
    ("masks", "minimum_foreground_fraction"): 0.0,
    ("masks", "maximum_foreground_fraction"): 1.0,
    ("masks", "selector_required_official_splits"): [0],
    ("masks", "analysis_only_audit_official_splits"): [1],
    ("masks", "optional_official_splits"): [2],
    ("masks", "runtime_resolve_from_manifest_only"): True,
    ("pretrained", "model"): "hf_hub:timm/vit_small_patch16_224.augreg_in21k_ft_in1k",
    ("pretrained", "repository"): "timm/vit_small_patch16_224.augreg_in21k_ft_in1k",
    ("pretrained", "revision"): "7e2c55630205e1266030f18370f4c6ed1a514b52",
    ("pretrained", "model_safetensors_sha256"): "79c03c635cdfd798a364a9d8c4e5c0b7255b975ea2c9616046d4f77ab01435aa",
    ("pretrained", "timm_version"): "1.0.28",
    ("branches", "frozen_epoch"): 30,
    ("branches", "epochs"): 30,
    ("branches", "copied_blocks"): [0, 1, 2, 3, 4, 5],
    ("branches", "embed_dim"): 384,
    ("branches", "heads"): 6,
    ("branches", "mlp_ratio"): 4.0,
    ("branches", "batch_size"): 64,
    ("branches", "learning_rate"): 3.0e-5,
    ("branches", "weight_decay"): 0.05,
    ("branches", "warmup_epochs"): 3,
    ("branches", "foreground_position_mode"): "object_relative",
    ("branches", "foreground_fill_rgb"): [0, 255, 0],
    ("branches", "background_min_coverage"): 0.95,
    ("branches", "background_max_biased_val_invalid_fraction"): 0.01,
    ("branches", "background_eval_views"): 8,
    ("candidate_grid", "epochs"): 40,
    ("candidate_grid", "architecture"): "vit_small_patch16_224",
    ("candidate_grid", "learning_rates"): [1.0e-5, 3.0e-5, 1.0e-4],
    ("candidate_grid", "weight_decays"): [0.01, 0.05],
    ("candidate_grid", "warmup_epochs"): 4,
    ("candidate_grid", "batch_size"): 64,
    ("candidate_grid", "seed"): 1234,
    ("anchorcal", "bootstrap_replicates"): 200,
    ("anchorcal", "minimum_intersection_per_class"): 50,
    ("anchorcal", "anchor_subset_per_class"): 512,
    ("anchorcal", "anchor_score_tolerance"): 1.0e-10,
    ("anchorcal", "candidate_score_tolerance"): 1.0e-8,
    ("anchorcal", "final_metric_bootstrap_replicates"): 2000,
    ("criteria", "selector_eval_per_class"): 256,
    ("criteria", "swap_donors"): 4,
    ("criteria", "blur_sigmas"): [2, 4, 8],
    ("optimization", "adamw_betas"): [0.9, 0.999],
    ("optimization", "adamw_epsilon"): 1.0e-8,
    ("optimization", "warmup_start_factor"): 0.01,
    ("optimization", "cosine_min_factor"): 0.01,
    ("optimization", "gradient_clip_norm"): 1.0,
    ("optimization", "autocast_dtype"): "bfloat16",
    ("optimization", "num_workers"): 8,
    ("optimization", "pin_memory"): True,
    ("optimization", "persistent_workers"): True,
    ("optimization", "prefetch_factor"): 2,
    ("storage", "hard_budget_gib"): 40.0,
    ("storage", "launch_guard_gib"): 35.0,
    ("storage", "minimum_filesystem_free_gib"): 16.0,
    ("storage", "worst_case_concurrent_growth_gib"): 6.0,
    (
        "storage",
        "projected_full_campaign_components_gib",
    ): {
        "candidate_checkpoints": 6.0,
        "restart_and_atomic_staging": 2.0,
        "candidate_hdf5_and_analysis": 1.0,
        "branches_and_anchors": 1.0,
        "manifests_galleries_and_reserve": 2.0,
    },
    ("runtime", "account"): "reu-aisocial",
    ("runtime", "partition"): "tigris",
    ("runtime", "gpu"): "gpu:gh200:1",
    ("runtime", "expected_architecture"): "aarch64",
    ("runtime", "python"): "/home/ryreu/miniforge3-aarch64/envs/fcv_gh200/bin/python",
    ("runtime", "require_clean_commit"): True,
}

LOCKED_SEEDS = {
    "foreground_branch_train": 6001,
    "background_branch_train": 6002,
    "background_sampling": 6003,
    "branch_audit_bootstrap": 7001,
    "anchor_bootstrap": 7002,
    "final_metric_bootstrap": 7003,
    "geometry_auditor_split": 8001,
    "geometry_auditor_model": 8002,
    "random_token_audit": 8003,
    "debug": 9001,
    "selector_eval": 16180,
    "donor_assignment": 31415,
    "anchor_subset": 424242,
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _get(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = config
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise ConfigurationError(f"missing required config key: {'.'.join(path)}")
        node = node[key]
    return node


def validate_locked_config(config: dict[str, Any], *, debug: bool = False) -> None:
    for key_path, expected in LOCKED.items():
        if debug and key_path in {
            ("branches", "frozen_epoch"),
            ("branches", "epochs"),
            ("branches", "background_eval_views"),
            ("candidate_grid", "epochs"),
            ("candidate_grid", "learning_rates"),
            ("candidate_grid", "weight_decays"),
            ("anchorcal", "bootstrap_replicates"),
            ("anchorcal", "final_metric_bootstrap_replicates"),
            ("criteria", "selector_eval_per_class"),
            ("criteria", "swap_donors"),
            ("criteria", "blur_sigmas"),
        }:
            continue
        actual = _get(config, key_path)
        if actual != expected:
            raise ConfigurationError(
                f"locked value {'.'.join(key_path)}={actual!r}; expected {expected!r}"
            )
    lambdas = config["anchorcal"]["lambdas"]
    expected_lambdas = [round(index * 0.05, 2) for index in range(21)]
    debug_lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
    if lambdas != (debug_lambdas if debug else expected_lambdas):
        raise ConfigurationError("anchor lambda ladder does not match the locked design")
    expected_parity_lambdas = (
        debug_lambdas if debug else [0.0, 0.35, 0.5, 0.8, 1.0]
    )
    if config["anchorcal"].get("parity_lambdas") != expected_parity_lambdas:
        raise ConfigurationError(
            "direct/cache parity lambdas do not match the locked design"
        )
    candidates = config["branches"]["background_token_budget_candidates"]
    if candidates != [64, 48, 32]:
        raise ConfigurationError("background token-budget order must be [64, 48, 32]")
    if config["seeds"] != LOCKED_SEEDS:
        raise ConfigurationError("named scientific seeds differ from the locked seed table")
    if config["branches"].get("no_background_sampling_replacement") is not True:
        raise ConfigurationError("primary background sampling must forbid replacement")
    if float(config["data"].get("crop_fallback_rate_gate", -1)) != 0.001:
        raise ConfigurationError("branch crop fallback-rate gate must be 0.001")
    criteria = config["criteria"]["eligible"]
    expected_criteria = [
        "ordinary_accuracy",
        "saliency_harmonic",
        "token_swap_harmonic",
        "background_blur_harmonic",
    ]
    if criteria != expected_criteria:
        raise ConfigurationError("eligible criteria differ from the locked four")
    if config["criteria"].get("diagnostic_only") != [
        "foreground_only_harmonic",
        "product_variants",
    ]:
        raise ConfigurationError(
            "diagnostic-only criteria differ from the locked controls"
        )
    if debug:
        debug_expected = {
            ("branches", "frozen_epoch"): 3,
            ("branches", "epochs"): 3,
            ("branches", "background_eval_views"): 2,
            ("candidate_grid", "learning_rates"): [3.0e-5],
            ("candidate_grid", "weight_decays"): [0.05],
            ("candidate_grid", "epochs"): 3,
            ("anchorcal", "bootstrap_replicates"): 20,
            ("anchorcal", "final_metric_bootstrap_replicates"): 20,
            ("criteria", "selector_eval_per_class"): 32,
            ("criteria", "swap_donors"): 2,
            ("criteria", "blur_sigmas"): [4],
        }
        for key_path, expected in debug_expected.items():
            if _get(config, key_path) != expected:
                raise ConfigurationError(
                    f"debug lock {'.'.join(key_path)} must equal {expected!r}"
                )
        if config["criteria"].get("selector_eval_total") != 64:
            raise ConfigurationError("debug selector_eval_total must equal 64")


def resolve_paths(config: dict[str, Any], *, require_complete: bool) -> dict[str, Any]:
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ConfigurationError("missing paths mapping; supply paths.local.yaml")
    for key in REQUIRED_PATH_KEYS:
        value = paths.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(f"paths.{key} must be a non-empty string")
        if value == "REQUIRED_ABSOLUTE_PATH":
            if require_complete:
                raise ConfigurationError(
                    f"paths.{key} is unresolved; edit configs/anchorcal/paths.local.yaml"
                )
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ConfigurationError(f"paths.{key} must be absolute: {value}")
        paths[key] = str(path.resolve(strict=False))
    metadata_expected = Path(paths["waterbirds_root"]) / "metadata.csv"
    if require_complete and Path(paths["metadata_path"]) != metadata_expected:
        raise ConfigurationError(
            "paths.metadata_path must be <waterbirds_root>/metadata.csv"
        )
    return config


def load_config(
    config_path: str | Path,
    paths_path: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
    require_paths: bool = True,
) -> dict[str, Any]:
    config = read_yaml(config_path)
    if paths_path is not None:
        path_values = read_yaml(paths_path)
        config = deep_merge(config, {"paths": path_values.get("paths", path_values)})
    if overrides:
        config = deep_merge(config, overrides)
    debug = bool(config.get("runtime", {}).get("debug", False))
    validate_locked_config(config, debug=debug)
    resolve_paths(config, require_complete=require_paths)
    config["resolved_config_sha256"] = hash_object(config)
    return config
