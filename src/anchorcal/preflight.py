"""Dataset, mask, split, environment, and provenance preflight."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from .data import (
    image_path,
    load_metadata,
    validated_waterbirds100_official_splits,
)
from .errors import PreflightError
from .io import atomic_write_json, atomic_write_yaml, hash_object, sha256_file
from .mask_identity import SELECTOR_MASK_RECEIPT_SCHEMA
from .mask_visual_audit import render_mask_visual_audit
from .pretrained import resolve_snapshot
from .preprocessing import (
    preprocessing_from_manifest,
    resolve_preprocessing_manifest,
    write_preprocessing_manifest,
)
from .prepare import prepare_geometry_artifacts
from .runtime import git_state, save_environment_manifest, write_package_lock
from .splits import construct_splits, persist_splits
from .storage_budget import assess_storage_budget
from .vlm_masks import (
    ANALYSIS_ONLY_MASK_AUDIT_RELATIVE_PATH,
    ANALYSIS_ONLY_MASK_AUDIT_SCHEMA,
    VLM_ALLOWED_CLASS_IDS,
    VLM_ANALYSIS_ONLY_AUDIT_OFFICIAL_SPLITS,
    VLM_DECODER_VERSION,
    VLM_FOREGROUND_CLASS_IDS,
    VLM_INTERPOLATION,
    VLM_MAPPING_MODE,
    VLM_MAPPING_VERSION,
    VLM_MASK_FORMAT,
    VLM_MASK_MANIFEST_SCHEMA,
    VLM_OPTIONAL_OFFICIAL_SPLITS,
    VLM_PREFLIGHT_AUDITED_OFFICIAL_SPLITS,
    VLM_PRODUCER,
    VLM_SELECTOR_REQUIRED_OFFICIAL_SPLITS,
    decode_vlm_mask,
    foreground_area_summary,
    producer_vlm_mask_name,
    resolve_teacher_map,
    teacher_map_candidates,
    vlm_mask_bank_hash,
    vlm_mask_contract_hash,
    vlm_mask_manifest_entry,
)


def validate_release(config: dict[str, Any]) -> tuple[Any, str]:
    root = Path(config["paths"]["waterbirds_root"])
    expected_name = config["data"]["release"]
    if root.name != expected_name:
        raise PreflightError(f"Waterbirds root basename must be {expected_name!r}: {root}")
    metadata_path = Path(config["paths"]["metadata_path"])
    if metadata_path != root / "metadata.csv" or not metadata_path.is_file():
        raise PreflightError("authoritative <waterbirds_root>/metadata.csv is missing")
    frame = load_metadata(metadata_path)
    validated_waterbirds100_official_splits(frame)
    return frame, sha256_file(metadata_path)


def validate_images_and_masks(
    config: dict[str, Any],
    frame: Any,
    *,
    metadata_sha256: str,
    failure_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit every image and freeze all required split-0/1 VLM mappings.

    Official test (split 2) maps are never required or decoded.  Existing
    split-2 candidates are inventory only and cannot enter any runtime lookup.
    """

    image_root = Path(config["paths"]["waterbirds_root"])
    mask_root = Path(config["paths"]["vlm_mask_root"])
    resolved_image_root = image_root.resolve()
    resolved_mask_root = mask_root.resolve()
    if not resolved_image_root.is_dir():
        raise PreflightError(f"Waterbirds image root is missing: {resolved_image_root}")
    if not resolved_mask_root.is_dir():
        raise PreflightError(f"VLM mask root is missing: {resolved_mask_root}")
    mask_config = config["masks"]
    minimum_fraction = float(mask_config["minimum_foreground_fraction"])
    maximum_fraction = float(mask_config["maximum_foreground_fraction"])
    audited_splits = set(VLM_PREFLIGHT_AUDITED_OFFICIAL_SPLITS)
    selector_splits = set(VLM_SELECTOR_REQUIRED_OFFICIAL_SPLITS)
    analysis_only_splits = set(VLM_ANALYSIS_ONLY_AUDIT_OFFICIAL_SPLITS)
    optional_splits = set(VLM_OPTIONAL_OFFICIAL_SPLITS)
    problems: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    seen_resolved: dict[Path, int] = {}
    image_sizes: dict[int, tuple[int, int]] = {}
    producer_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_candidate_files: set[Path] = set()
    optional_inventory = {"expected": 0, "missing": 0, "unique": 0, "ambiguous": 0}

    for metadata_index, row in enumerate(frame.itertuples(index=False)):
        try:
            producer_name = producer_vlm_mask_name(str(row.img_filename))
            # Split 2 is inventory-only and has no runtime mask contract.  It
            # therefore cannot create a fatal producer-name collision with a
            # required split-0/1 row (or with another split-2 row).
            if int(row.split) in audited_splits:
                producer_rows[producer_name].append(
                    {
                        "img_id": int(row.img_id),
                        "official_split": int(row.split),
                        "img_filename": str(row.img_filename),
                    }
                )
            for _, candidate in teacher_map_candidates(
                resolved_mask_root, str(row.img_filename)
            ):
                if candidate.is_file():
                    all_candidate_files.add(candidate.resolve())
        except PreflightError as error:
            problems.append(
                {
                    "img_id": int(row.img_id),
                    "official_split": int(row.split),
                    "reasons": [f"invalid_img_filename:{error}"],
                }
            )

    collisions = [
        {"producer_mask_name": name, "rows": rows}
        for name, rows in sorted(producer_rows.items())
        if len(rows) > 1
    ]
    for collision in collisions:
        for row in collision["rows"]:
            problems.append(
                {
                    "img_id": int(row["img_id"]),
                    "official_split": int(row["official_split"]),
                    "reasons": [
                        "producer_name_collision:"
                        f"{collision['producer_mask_name']}"
                    ],
                }
            )

    for metadata_index, row in enumerate(frame.itertuples(index=False)):
        reasons: list[str] = []
        try:
            image_file = image_path(image_root, str(row.img_filename))
        except PreflightError as error:
            image_file = None
            reasons.append(f"invalid_image_path:{error}")
        if image_file is not None:
            if not image_file.is_file():
                reasons.append("missing_image")
            else:
                try:
                    with Image.open(image_file) as image:
                        image.load()
                        image_sizes[int(row.img_id)] = image.size
                except (OSError, ValueError) as error:
                    reasons.append(f"invalid_image:{error}")

        split = int(row.split)
        if split in optional_splits:
            optional_inventory["expected"] += 1
            try:
                optional_candidates = teacher_map_candidates(
                    resolved_mask_root, str(row.img_filename)
                )
                matches = {
                    candidate.resolve()
                    for _, candidate in optional_candidates
                    if candidate.is_file()
                }
            except PreflightError as error:
                matches = set()
                reasons.append(f"invalid_optional_vlm_mapping:{error}")
            if not matches:
                optional_inventory["missing"] += 1
            elif len(matches) == 1:
                optional_inventory["unique"] += 1
            else:
                optional_inventory["ambiguous"] += 1
        elif split in audited_splits:
            try:
                mask_file, mapping_rule = resolve_teacher_map(
                    resolved_mask_root, str(row.img_filename)
                )
                if mask_file in seen_resolved:
                    reasons.append(
                        "duplicate_mask_with_img_id_"
                        f"{seen_resolved[mask_file]}"
                    )
                else:
                    seen_resolved[mask_file] = int(row.img_id)
                decoded = decode_vlm_mask(
                    mask_file,
                    minimum_foreground_fraction=minimum_fraction,
                    maximum_foreground_fraction=maximum_fraction,
                )
                image_size = image_sizes.get(int(row.img_id))
                if image_size is not None and (
                    decoded.binary.shape[1], decoded.binary.shape[0]
                ) != image_size:
                    reasons.append("mask_dimension_mismatch")
                entry = vlm_mask_manifest_entry(
                    img_id=int(row.img_id),
                    metadata_index=int(
                        getattr(row, "metadata_row_index", metadata_index)
                    ),
                    img_filename=str(row.img_filename),
                    split=split,
                    root=resolved_mask_root,
                    path=mask_file,
                    mapping_rule=mapping_rule,
                    decoded=decoded,
                )
                if image_size is not None:
                    entry["image_width"] = int(image_size[0])
                    entry["image_height"] = int(image_size[1])
                if image_file is not None and image_file.is_file():
                    entry["image_relative_path"] = (
                        image_file.resolve()
                        .relative_to(resolved_image_root)
                        .as_posix()
                    )
                    entry["image_sha256"] = sha256_file(image_file)
                    entry["image_size_bytes"] = image_file.stat().st_size
                if not reasons:
                    entries.append(entry)
            except (PreflightError, OSError, ValueError) as error:
                reasons.append(f"invalid_vlm_mask:{error}")

        if reasons:
            problems.append(
                {
                    "img_id": int(row.img_id),
                    "official_split": split,
                    "img_filename": str(row.img_filename),
                    "reasons": reasons,
                }
            )

    expected_audited = int(frame["split"].isin(audited_splits).sum())
    audited_coverage = {
        str(split): {
            "expected": int((frame["split"] == split).sum()),
            "present": sum(
                int(entry["official_split"]) == split for entry in entries
            ),
        }
        for split in sorted(audited_splits)
    }
    if problems or len(entries) != expected_audited:
        if len(entries) != expected_audited and not problems:
            problems.append(
                {
                    "img_id": None,
                    "official_split": None,
                    "reasons": [
                        f"audited_mapping_count_{len(entries)}_expected_"
                        f"{expected_audited}"
                    ],
                }
            )
        protected_problems = [
            problem
            for problem in problems
            if problem.get("official_split")
            in VLM_ANALYSIS_ONLY_AUDIT_OFFICIAL_SPLITS
        ]
        public_problems = [
            problem for problem in problems if problem not in protected_problems
        ]
        if protected_problems:
            atomic_write_json(
                Path(config["paths"]["output_root"])
                / ANALYSIS_ONLY_MASK_AUDIT_RELATIVE_PATH,
                {
                    "schema_version": ANALYSIS_ONLY_MASK_AUDIT_SCHEMA,
                    "status": "failed",
                    "namespace": "analysis_only",
                    "selector_visible": False,
                    "reporting_only": True,
                    "official_splits": list(
                        VLM_ANALYSIS_ONLY_AUDIT_OFFICIAL_SPLITS
                    ),
                    "metadata_sha256": metadata_sha256,
                    "resolved_config_sha256": config[
                        "resolved_config_sha256"
                    ],
                    "failure_count": len(protected_problems),
                    "failures": protected_problems,
                },
            )
        if failure_report_path is not None:
            public_collisions = []
            for collision in collisions:
                public_rows = [
                    row
                    for row in collision["rows"]
                    if int(row["official_split"])
                    not in VLM_ANALYSIS_ONLY_AUDIT_OFFICIAL_SPLITS
                ]
                if public_rows:
                    public_collisions.append(
                        {
                            "producer_mask_name": collision[
                                "producer_mask_name"
                            ],
                            "rows": public_rows,
                        }
                    )
            atomic_write_json(
                failure_report_path,
                {
                    "schema_version": "anchorcal-vlm-mask-validation-failure-v1",
                    "status": "failed",
                    "failure_count": len(public_problems),
                    "coverage": audited_coverage,
                    "analysis_only_mask_audit_status": (
                        "failed" if protected_problems else "passed"
                    ),
                    "producer_name_collisions": public_collisions,
                    "failures": public_problems,
                },
            )
        preview = (
            json.dumps(public_problems[:20], indent=2)
            if public_problems
            else "analysis-only mask audit failed; inspect protected audit"
        )
        raise PreflightError(
            f"image/VLM audit failed for {len(problems)} conditions; "
            f"first failures:\n{preview}"
        )

    entries.sort(key=lambda item: int(item["img_id"]))
    public_entries = [
        {
            key: value
            for key, value in entry.items()
            if key != "metadata_index"
        }
        for entry in entries
        if int(entry["official_split"]) in selector_splits
    ]
    protected_entries = [
        entry
        for entry in entries
        if int(entry["official_split"]) in analysis_only_splits
    ]
    expected_public = int(frame["split"].isin(selector_splits).sum())
    expected_protected = int(frame["split"].isin(analysis_only_splits).sum())
    if (
        len(public_entries) != expected_public
        or len(protected_entries) != expected_protected
    ):
        raise PreflightError(
            "validated VLM rows do not partition into selector-safe split 0 "
            "and analysis-only split 1"
        )
    all_pngs = {
        path.resolve()
        for path in resolved_mask_root.rglob("*.png")
        if path.is_file()
    }
    extras = sorted(
        path.relative_to(resolved_mask_root).as_posix()
        for path in all_pngs - all_candidate_files
        if path.is_relative_to(resolved_mask_root)
    )
    mapping_counts = Counter(
        str(entry["mapping_rule"]) for entry in public_entries
    )
    class_pixel_counts: Counter[str] = Counter()
    observed_colors: set[tuple[int, int, int]] = set()
    for entry in public_entries:
        class_pixel_counts.update(entry["class_pixel_counts"])
        observed_colors.update(
            tuple(int(channel) for channel in color)
            for color in entry["observed_rgb_colors"]
        )
    bank_hash = vlm_mask_bank_hash(
        public_entries,
        root=resolved_mask_root,
        minimum_foreground_fraction=minimum_fraction,
        maximum_foreground_fraction=maximum_fraction,
    )
    if failure_report_path is not None:
        Path(failure_report_path).unlink(missing_ok=True)
    protected_audit = {
        "schema_version": ANALYSIS_ONLY_MASK_AUDIT_SCHEMA,
        "status": "passed",
        "namespace": "analysis_only",
        "selector_visible": False,
        "reporting_only": True,
        "official_splits": list(VLM_ANALYSIS_ONLY_AUDIT_OFFICIAL_SPLITS),
        "dataset_root": str(resolved_image_root),
        "metadata_path": str(Path(config["paths"]["metadata_path"]).resolve()),
        "metadata_sha256": metadata_sha256,
        "resolved_config_sha256": config["resolved_config_sha256"],
        "mask_contract_sha256": vlm_mask_contract_hash(config),
        "root": str(resolved_mask_root),
        "producer": VLM_PRODUCER,
        "mapping_mode": VLM_MAPPING_MODE,
        "mapping_version": VLM_MAPPING_VERSION,
        "decoder_version": VLM_DECODER_VERSION,
        "format": VLM_MASK_FORMAT,
        "foreground_class_ids": list(VLM_FOREGROUND_CLASS_IDS),
        "allowed_class_ids": list(VLM_ALLOWED_CLASS_IDS),
        "interpolation": VLM_INTERPOLATION,
        "minimum_foreground_fraction": minimum_fraction,
        "maximum_foreground_fraction": maximum_fraction,
        "row_count": len(protected_entries),
        "coverage": {
            str(split): audited_coverage[str(split)]
            for split in VLM_ANALYSIS_ONLY_AUDIT_OFFICIAL_SPLITS
        },
        "mapping_rule_counts": dict(
            sorted(
                Counter(
                    str(entry["mapping_rule"])
                    for entry in protected_entries
                ).items()
            )
        ),
        "foreground_area_summary": foreground_area_summary(
            protected_entries,
            official_splits=VLM_ANALYSIS_ONLY_AUDIT_OFFICIAL_SPLITS,
        ),
        "entries_sha256": hash_object(protected_entries),
        "entries": protected_entries,
    }
    protected_audit_path = (
        Path(config["paths"]["output_root"])
        / ANALYSIS_ONLY_MASK_AUDIT_RELATIVE_PATH
    )
    atomic_write_json(protected_audit_path, protected_audit)
    return {
        "schema_version": VLM_MASK_MANIFEST_SCHEMA,
        "status": "passed",
        "dataset_root": str(resolved_image_root),
        "metadata_path": str(Path(config["paths"]["metadata_path"]).resolve()),
        "metadata_sha256": metadata_sha256,
        "resolved_config_sha256": config["resolved_config_sha256"],
        "mask_contract_sha256": vlm_mask_contract_hash(config),
        "root": str(resolved_mask_root),
        "producer": VLM_PRODUCER,
        "mapping_mode": VLM_MAPPING_MODE,
        "mapping_version": VLM_MAPPING_VERSION,
        "decoder_version": VLM_DECODER_VERSION,
        "format": VLM_MASK_FORMAT,
        "foreground_class_ids": list(VLM_FOREGROUND_CLASS_IDS),
        "allowed_class_ids": list(VLM_ALLOWED_CLASS_IDS),
        "interpolation": VLM_INTERPOLATION,
        "minimum_foreground_fraction": minimum_fraction,
        "maximum_foreground_fraction": maximum_fraction,
        "selector_required_official_splits": list(
            VLM_SELECTOR_REQUIRED_OFFICIAL_SPLITS
        ),
        "analysis_only_audit_official_splits": list(
            VLM_ANALYSIS_ONLY_AUDIT_OFFICIAL_SPLITS
        ),
        "optional_official_splits": list(VLM_OPTIONAL_OFFICIAL_SPLITS),
        "runtime_resolution": "frozen_manifest_only",
        "required_count": expected_public,
        "required_mapping_audit": {
            "expected": expected_public,
            "resolved_unique": len(public_entries),
            "missing": 0,
            "ambiguous": 0,
            "reused": 0,
            "producer_name_collisions": 0,
        },
        "coverage": {
            str(split): audited_coverage[str(split)]
            for split in VLM_SELECTOR_REQUIRED_OFFICIAL_SPLITS
        },
        "optional_split_inventory": optional_inventory,
        "producer_name_collisions": collisions,
        "mapping_rule_counts": dict(sorted(mapping_counts.items())),
        "aggregate_class_pixel_counts": dict(sorted(class_pixel_counts.items())),
        "observed_rgb_colors": [list(color) for color in sorted(observed_colors)],
        "foreground_area_summary": foreground_area_summary(public_entries),
        "extras_inventory": {
            "count": len(extras),
            "relative_paths": extras,
        },
        "mask_bank_sha256": bank_hash,
        "entries": public_entries,
    }


def run_preflight(
    config: dict[str, Any], *, allow_download: bool, require_gh200: bool
) -> dict[str, Any]:
    if require_gh200:
        fixed_paths = {
            "repo_root": "/home/ryreu/guided_cnn/BirdOnly",
            "waterbirds_root": (
                "/home/ryreu/guided_cnn/waterbirds/"
                "waterbird_1.0_forest2water2"
            ),
            "metadata_path": (
                "/home/ryreu/guided_cnn/waterbirds/"
                "waterbird_1.0_forest2water2/metadata.csv"
            ),
            "vlm_mask_root": (
                "/home/ryreu/guided_cnn/Food101/LearningToLook/code/"
                "WeCLIPPlus/results_waterbirds100_openclip_laion_dinovit/"
                "val/prediction_cmap"
            ),
            "fcv_split_manifest_root": (
                "/home/ryreu/guided_cnn/logsWaterbird/"
                "fcv_vit_waterbirds100_first_study/split_manifests"
            ),
            "hf_home": "/home/ryreu/.cache/huggingface",
            "output_root": "/home/ryreu/guided_cnn/BirdOnly/outputs/anchorcal/waterbirds100_pilot",
        }
        for name, expected in fixed_paths.items():
            if Path(config["paths"][name]).resolve() != Path(expected).resolve():
                raise PreflightError(
                    f"TIGRIS paths.{name} must resolve to {expected}, found "
                    f"{config['paths'][name]}"
                )
    output = Path(config["paths"]["output_root"])
    preflight_dir = output / "preflight"
    splits_dir = output / "splits"
    environment_dir = output / "environment"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    storage_budget = assess_storage_budget(
        config,
        output,
        stage="production_preflight" if require_gh200 else "preflight",
    )
    storage_budget_path = preflight_dir / "storage_budget.json"
    atomic_write_json(storage_budget_path, storage_budget)
    repo_state = git_state(config["paths"]["repo_root"])
    if (
        bool(config.get("runtime", {}).get("require_clean_commit", False))
        and not bool(config.get("runtime", {}).get("debug", False))
        and not repo_state["clean"]
    ):
        raise PreflightError(
            "production preflight requires a clean Git worktree; commit the "
            "AnchorCal implementation before launching the campaign"
        )
    if require_gh200 and Path(sys.executable).resolve() != Path(
        config["runtime"]["python"]
    ).resolve():
        raise PreflightError(
            "production is using the wrong Python interpreter: "
            f"{sys.executable}; expected {config['runtime']['python']}"
        )
    # Fail on missing analysis/runtime dependencies before model or dataset
    # work.  This gate includes matplotlib even though plotting happens only in
    # the final reporting job.
    env_manifest = save_environment_manifest(
        environment_dir / "environment.json", require_gh200=require_gh200
    )
    write_package_lock(environment_dir / "package-lock.txt")
    frame, metadata_hash = validate_release(config)
    mask_manifest = validate_images_and_masks(
        config,
        frame,
        metadata_sha256=metadata_hash,
        failure_report_path=preflight_dir / "mask_validation_failure_report.json",
    )
    mask_manifest["git_commit"] = repo_state["commit"]
    mask_visual_audit = render_mask_visual_audit(config, frame, mask_manifest)
    mask_manifest["visual_audit"] = mask_visual_audit
    mask_manifest_path = preflight_dir / "mask_manifest.json"
    atomic_write_json(mask_manifest_path, mask_manifest)
    selector_mask_receipt = {
        "schema_version": SELECTOR_MASK_RECEIPT_SCHEMA,
        "status": "passed",
        "namespace": "selector_visible",
        "contains_per_row_records": False,
        "selector_required_official_splits": list(
            VLM_SELECTOR_REQUIRED_OFFICIAL_SPLITS
        ),
        "resolved_config_sha256": config["resolved_config_sha256"],
        "metadata_sha256": metadata_hash,
        "git_commit": repo_state["commit"],
        "mask_source": VLM_PRODUCER,
        "mask_contract_sha256": mask_manifest["mask_contract_sha256"],
        "mask_bank_sha256": mask_manifest["mask_bank_sha256"],
        "mask_manifest_sha256": sha256_file(mask_manifest_path),
        "foreground_area_summary_sha256": hash_object(
            mask_manifest["foreground_area_summary"]
        ),
        "mask_visual_audit_manifest_sha256": mask_visual_audit[
            "manifest_sha256"
        ],
        "mask_visual_audit_selection_sha256": mask_visual_audit[
            "selection_sha256"
        ],
    }
    selector_mask_receipt_path = preflight_dir / "selector_mask_receipt.json"
    atomic_write_json(selector_mask_receipt_path, selector_mask_receipt)
    splits = construct_splits(
        frame,
        metadata_hash,
        fcv_split_manifest_root=config["paths"]["fcv_split_manifest_root"],
        fcv_reference=config["data"]["fcv_reference"],
        calibration_seed=int(config["data"]["expert_calibration_seed"]),
    )
    split_manifest = persist_splits(
        splits,
        splits_dir,
        analysis_only_output_dir=(
            output / str(config["data"]["protected_split_root"])
        ),
        source_metadata=frame,
        source_metadata_sha256=metadata_hash,
        source_release=str(config["data"]["release"]),
        fcv_split_manifest_root=config["paths"]["fcv_split_manifest_root"],
        fcv_reference=config["data"]["fcv_reference"],
    )
    model_manifest = resolve_snapshot(
        config["paths"]["hf_home"], allow_download=allow_download
    )
    pretrained_manifest_path = preflight_dir / "pretrained_manifest.json"
    atomic_write_json(pretrained_manifest_path, model_manifest)
    preprocessing_manifest = resolve_preprocessing_manifest(model_manifest)
    preprocessing_manifest_path = preflight_dir / "preprocessing_manifest.json"
    write_preprocessing_manifest(preprocessing_manifest_path, preprocessing_manifest)
    preprocessing = preprocessing_from_manifest(preprocessing_manifest)
    # This first production consumer receives the timm-derived values directly;
    # subsequent jobs reload and hash-check this same serialized manifest.
    geometry_manifest = prepare_geometry_artifacts(
        config, preprocessing=preprocessing
    )
    atomic_write_yaml(preflight_dir / "resolved_config.yaml", config)
    report = {
        "schema_version": "anchorcal-preflight-v1",
        "status": "passed",
        "resolved_paths": dict(config["paths"]),
        "metadata_path": config["paths"]["metadata_path"],
        "metadata_sha256": metadata_hash,
        "metadata_rows": int(len(frame)),
        "mask_source": VLM_PRODUCER,
        "mask_contract_sha256": mask_manifest["mask_contract_sha256"],
        "mask_bank_sha256": mask_manifest["mask_bank_sha256"],
        "foreground_area_summary": mask_manifest["foreground_area_summary"],
        "mask_manifest_path": str(mask_manifest_path.resolve()),
        "mask_manifest_sha256": sha256_file(mask_manifest_path),
        "mask_visual_audit": mask_visual_audit,
        "selector_mask_receipt": {
            "schema_version": SELECTOR_MASK_RECEIPT_SCHEMA,
            "sha256": sha256_file(selector_mask_receipt_path),
        },
        "split_manifest": split_manifest,
        "storage_budget": {
            **storage_budget,
            "manifest_path": str(storage_budget_path.resolve()),
            "manifest_sha256": sha256_file(storage_budget_path),
        },
        "geometry_manifest": geometry_manifest,
        "pretrained": model_manifest,
        "preprocessing": {
            "manifest_path": str(preprocessing_manifest_path.resolve()),
            "manifest_sha256": sha256_file(preprocessing_manifest_path),
            "effective_resize_shortest": preprocessing.effective_resize_shortest,
            "parity": preprocessing_manifest["parity"],
        },
        "environment": env_manifest,
        "git": repo_state,
        "resolved_config_sha256": config["resolved_config_sha256"],
    }
    atomic_write_json(preflight_dir / "report.json", report)
    return report
