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
    segmentations: set[str] = set()
    for root_value in search_roots:
        root = Path(root_value).expanduser()
        if not root.exists():
            continue
        for candidate in root.rglob("waterbird_complete95_forest2water2"):
            if candidate.is_dir() and (candidate / "metadata.csv").is_file():
                releases.add(str(candidate.resolve()))
                metadata.add(str((candidate / "metadata.csv").resolve()))
        for candidate in root.rglob("segmentations"):
            if candidate.is_dir() and candidate.parent.name == "CUB_200_2011":
                segmentations.add(str(candidate.resolve()))
    return {
        "waterbirds_root": sorted(releases),
        "metadata_path": sorted(metadata),
        "cub_source_segmentation_root": sorted(segmentations),
        "cub_waterbirds_mask_root": sorted(segmentations),
    }
