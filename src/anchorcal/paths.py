"""Machine-local path discovery that never silently resolves ambiguity."""

from __future__ import annotations

from pathlib import Path


def geometry_artifact_root(config: dict) -> Path:
    output = Path(config["paths"]["output_root"])
    if bool(config.get("runtime", {}).get("debug", False)):
        return output / "debug" / "preflight" / "geometry"
    return output / "preflight" / "geometry"


def discover_candidates(search_roots: list[str | Path]) -> dict[str, list[str]]:
    """Return possible paths; callers must explicitly confirm a unique choice."""

    releases: set[str] = set()
    metadata: set[str] = set()
    vlm_masks: set[str] = set()
    fcv_split_manifests: set[str] = set()
    for root_value in search_roots:
        root = Path(root_value).expanduser()
        if not root.exists():
            continue
        for candidate in root.rglob("waterbird_1.0_forest2water2"):
            if candidate.is_dir() and (candidate / "metadata.csv").is_file():
                releases.add(str(candidate.resolve()))
                metadata.add(str((candidate / "metadata.csv").resolve()))
        for results_root in root.rglob(
            "results_waterbirds100_openclip_laion_dinovit"
        ):
            candidate = results_root / "val" / "prediction_cmap"
            if candidate.is_dir():
                vlm_masks.add(str(candidate.resolve()))
        for candidate in root.rglob("fcv_vit_waterbirds100_first_study"):
            split_root = candidate / "split_manifests"
            required = (
                "manifest_bundle.json",
                "split_indices.json",
                "metadata_train.csv",
                "metadata_val.csv",
            )
            if split_root.is_dir() and all(
                (split_root / name).is_file() for name in required
            ):
                fcv_split_manifests.add(str(split_root.resolve()))
    return {
        "waterbirds_root": sorted(releases),
        "metadata_path": sorted(metadata),
        "vlm_mask_root": sorted(vlm_masks),
        "fcv_split_manifest_root": sorted(fcv_split_manifests),
    }
