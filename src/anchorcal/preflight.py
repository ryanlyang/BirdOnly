"""Dataset, mask, split, environment, and provenance preflight."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .data import image_path, load_metadata, mask_path, mask_relative_path
from .errors import PreflightError
from .io import atomic_write_json, atomic_write_yaml, sha256_file
from .masks import load_binary_mask, mask_bank_hash, mask_manifest_entry
from .pretrained import resolve_snapshot
from .preprocessing import (
    preprocessing_from_manifest,
    resolve_preprocessing_manifest,
    write_preprocessing_manifest,
)
from .prepare import prepare_geometry_artifacts
from .runtime import git_state, save_environment_manifest, write_package_lock
from .splits import construct_splits, persist_splits


def validate_release(config: dict[str, Any]) -> tuple[Any, str]:
    root = Path(config["paths"]["waterbirds_root"])
    expected_name = config["data"]["release"]
    if root.name != expected_name:
        raise PreflightError(f"Waterbirds root basename must be {expected_name!r}: {root}")
    metadata_path = Path(config["paths"]["metadata_path"])
    if metadata_path != root / "metadata.csv" or not metadata_path.is_file():
        raise PreflightError("authoritative <waterbirds_root>/metadata.csv is missing")
    frame = load_metadata(metadata_path)
    return frame, sha256_file(metadata_path)


def validate_images_and_masks(
    config: dict[str, Any],
    frame: Any,
    *,
    failure_report_path: str | Path | None = None,
) -> dict[str, Any]:
    image_root = Path(config["paths"]["waterbirds_root"])
    source_root = Path(config["paths"]["cub_source_segmentation_root"])
    final_root = Path(config["paths"]["cub_waterbirds_mask_root"])
    same_tree = source_root.resolve() == final_root.resolve()
    mapping_manifest_path = final_root / "anchorcal_mapping_manifest.json"
    mapping_manifest: dict[str, Any] | None = None
    mapping_by_id: dict[int, dict[str, Any]] = {}
    resolved_roots = {
        "image": image_root.resolve(),
        "source": source_root.resolve(),
        "final": final_root.resolve(),
    }
    if not same_tree:
        if not mapping_manifest_path.is_file():
            raise PreflightError(
                "separate CUB source/final mask roots require "
                f"{mapping_manifest_path} with immutable mapping provenance"
            )
        try:
            mapping_manifest = json.loads(
                mapping_manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise PreflightError("CUB mapping provenance manifest is invalid") from error
        if (
            mapping_manifest.get("schema_version")
            != "anchorcal-cub-waterbirds-mask-mapping-v1"
            or Path(mapping_manifest.get("source_segmentation_root", "")).resolve()
            != source_root.resolve()
            or not isinstance(mapping_manifest.get("generation_method"), str)
            or not mapping_manifest["generation_method"].strip()
        ):
            raise PreflightError("CUB mapping provenance manifest fields are invalid")
        try:
            mapping_by_id = {
                int(item["img_id"]): item for item in mapping_manifest["entries"]
            }
        except (KeyError, TypeError, ValueError) as error:
            raise PreflightError("CUB mapping manifest entries are malformed") from error
        if len(mapping_by_id) != len(mapping_manifest["entries"]):
            raise PreflightError("CUB mapping manifest contains duplicate img_id entries")
    problems: list[dict[str, Any]] = []
    entries: list[dict[str, object]] = []
    seen_resolved: dict[Path, int] = {}
    for row in frame.itertuples(index=False):
        image_file = image_path(image_root, str(row.img_filename))
        source_file = mask_path(source_root, str(row.img_filename))
        final_file = mask_path(final_root, str(row.img_filename))
        reasons: list[str] = []
        for name, candidate in (
            ("image", image_file),
            ("source", source_file),
            ("final", final_file),
        ):
            if not candidate.resolve(strict=False).is_relative_to(
                resolved_roots[name]
            ):
                reasons.append(f"{name}_path_escapes_root")
        if not image_file.is_file():
            reasons.append("missing_image")
        if not source_file.is_file():
            reasons.append("missing_source_cub_mask")
        if not final_file.is_file():
            reasons.append("missing_waterbirds_coordinate_mask")
        if reasons:
            problems.append({"img_id": int(row.img_id), "reasons": reasons})
            continue
        resolved = final_file.resolve()
        if resolved in seen_resolved:
            problems.append(
                {
                    "img_id": int(row.img_id),
                    "reasons": [f"duplicate_mask_with_img_id_{seen_resolved[resolved]}"],
                }
            )
            continue
        seen_resolved[resolved] = int(row.img_id)
        try:
            binary = load_binary_mask(final_file)
            source_binary = load_binary_mask(source_file)
            with Image.open(image_file) as image:
                image_size = image.size
            if (binary.shape[1], binary.shape[0]) != image_size:
                reasons.append("final_mask_dimension_mismatch")
            if (source_binary.shape[1], source_binary.shape[0]) != image_size:
                reasons.append("source_mask_dimension_mismatch")
            if source_root.resolve() == final_root.resolve() and not np.array_equal(
                source_binary, binary
            ):
                reasons.append("same_tree_source_final_mismatch")
            entry = mask_manifest_entry(
                int(row.img_id), str(mask_relative_path(str(row.img_filename))), final_file
            )
            entry["source_relative_path"] = str(
                mask_relative_path(str(row.img_filename))
            )
            entry["source_sha256"] = sha256_file(source_file)
            if not same_tree:
                mapping = mapping_by_id.get(int(row.img_id))
                relative = str(mask_relative_path(str(row.img_filename)))
                if mapping is None:
                    reasons.append("missing_mapping_provenance_entry")
                elif (
                    mapping.get("source_relative_path") != relative
                    or mapping.get("final_relative_path") != relative
                    or mapping.get("source_sha256") != sha256_file(source_file)
                    or mapping.get("final_sha256") != sha256_file(final_file)
                ):
                    reasons.append("mapping_provenance_hash_or_path_mismatch")
            entries.append(entry)
        except (PreflightError, OSError, ValueError) as error:
            reasons.append(f"invalid_mask:{error}")
        if reasons:
            problems.append({"img_id": int(row.img_id), "reasons": reasons})
    if problems:
        if failure_report_path is not None:
            atomic_write_json(
                failure_report_path,
                {
                    "schema_version": "anchorcal-mask-validation-failure-v1",
                    "status": "failed",
                    "failure_count": len(problems),
                    # This is intentionally the complete list.  The exception
                    # preview is kept short for usable scheduler logs, while
                    # the durable report is the authoritative audit record.
                    "failures": problems,
                },
            )
        preview = json.dumps(problems[:20], indent=2)
        raise PreflightError(
            f"image/mask audit failed for {len(problems)} img_ids; first failures:\n{preview}"
        )
    if len(entries) != len(frame):
        raise PreflightError("one-mask-per-img_id assertion failed")
    return {
        "schema_version": "anchorcal-mask-manifest-v1",
        "count": len(entries),
        "source_root": str(source_root.resolve()),
        "final_root": str(final_root.resolve()),
        "same_tree": same_tree,
        "mapping_manifest": (
            None
            if same_tree
            else {
                "path": str(mapping_manifest_path.resolve()),
                "sha256": sha256_file(mapping_manifest_path),
                "generation_method": mapping_manifest["generation_method"],
            }
        ),
        "mask_bank_sha256": mask_bank_hash(entries),
        "entries": entries,
    }


def run_preflight(
    config: dict[str, Any], *, allow_download: bool, require_gh200: bool
) -> dict[str, Any]:
    if require_gh200:
        fixed_paths = {
            "repo_root": "/home/ryreu/guided_cnn/BirdOnly",
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
        failure_report_path=preflight_dir / "mask_validation_failure_report.json",
    )
    splits = construct_splits(frame, metadata_hash)
    split_manifest = persist_splits(splits, splits_dir)
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
    atomic_write_json(preflight_dir / "mask_manifest.json", mask_manifest)
    report = {
        "schema_version": "anchorcal-preflight-v1",
        "status": "passed",
        "resolved_paths": dict(config["paths"]),
        "metadata_path": config["paths"]["metadata_path"],
        "metadata_sha256": metadata_hash,
        "metadata_rows": int(len(frame)),
        "mask_bank_sha256": mask_manifest["mask_bank_sha256"],
        "split_manifest": split_manifest,
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
