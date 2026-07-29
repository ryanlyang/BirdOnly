"""Strict discovery and validation of VLM-generated masks."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from setv.data.joint_transforms import binarize_mask
from setv.errors import DataValidationError
from setv.utils.hashing import sha256_file


@dataclass(frozen=True)
class ResolvedMask:
    path: Path
    relative_path: str
    mapping_rule: str


class MaskResolver:
    """Resolve masks by an explicit, deterministic, ambiguity-rejecting rule."""

    def __init__(
        self,
        root: str | Path,
        allowed_extensions: Iterable[str],
        mapping_mode: str,
    ):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise DataValidationError(f"VLM mask root is missing: {self.root}")
        extensions = {extension.lower() for extension in allowed_extensions}
        paths = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )
        if not paths:
            raise DataValidationError(f"No supported masks found under {self.root}")
        self.mapping_mode = mapping_mode
        self._relative: dict[str, list[Path]] = defaultdict(list)
        self._basename: dict[str, list[Path]] = defaultdict(list)
        for path in paths:
            relative = path.relative_to(self.root)
            relative_key = relative.with_suffix("").as_posix().casefold()
            basename_key = path.stem.casefold()
            self._relative[relative_key].append(path)
            self._basename[basename_key].append(path)
        self.total_mask_files = len(paths)

    @staticmethod
    def _unique(paths: list[Path], description: str) -> Path | None:
        if len(paths) > 1:
            formatted = ", ".join(str(path) for path in paths[:5])
            raise DataValidationError(
                f"Ambiguous VLM mask mapping for {description}: {formatted}"
            )
        return paths[0] if paths else None

    def resolve(self, image_filename: str, sample_id: str) -> ResolvedMask:
        image = Path(str(image_filename))
        relative_key = image.with_suffix("").as_posix().casefold()
        basename_key = image.stem.casefold()
        sample_key = str(sample_id).casefold()

        candidates: list[tuple[str, list[Path]]] = []
        if self.mapping_mode == "relative_stem":
            candidates.append(("relative_stem", self._relative.get(relative_key, [])))
        elif self.mapping_mode == "unique_basename":
            candidates.append(("unique_basename", self._basename.get(basename_key, [])))
        elif self.mapping_mode == "relative_stem_then_unique_basename":
            candidates.extend(
                [
                    ("relative_stem", self._relative.get(relative_key, [])),
                    ("unique_basename", self._basename.get(basename_key, [])),
                ]
            )
        elif self.mapping_mode == "sample_id":
            candidates.append(("sample_id", self._basename.get(sample_key, [])))
        else:
            raise DataValidationError(f"Unsupported mapping mode: {self.mapping_mode}")

        for rule, paths in candidates:
            resolved = self._unique(
                paths, f"sample_id={sample_id}, image={image_filename}, rule={rule}"
            )
            if resolved is not None:
                return ResolvedMask(
                    path=resolved,
                    relative_path=resolved.relative_to(self.root).as_posix(),
                    mapping_rule=rule,
                )
        raise DataValidationError(
            f"No VLM mask for sample_id={sample_id}, image={image_filename}, "
            f"mapping_mode={self.mapping_mode}"
        )


def inspect_mask(
    image_path: str | Path,
    mask_path: str | Path,
    *,
    threshold_normalized: float,
    foreground_is_high: bool,
    require_same_dimensions: bool,
    minimum_foreground_fraction: float,
    maximum_foreground_fraction: float,
) -> dict:
    image_path = Path(image_path)
    mask_path = Path(mask_path)
    try:
        with Image.open(image_path) as opened:
            image_size = opened.size
            opened.verify()
        with Image.open(mask_path) as opened:
            mask_size = opened.size
            raw = np.asarray(opened.convert("L"), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise DataValidationError(
            f"Could not read image/mask pair {image_path} / {mask_path}: {exc}"
        ) from exc

    if require_same_dimensions and image_size != mask_size:
        raise DataValidationError(
            f"Image-mask dimension mismatch: image={image_path} {image_size}, "
            f"mask={mask_path} {mask_size}"
        )

    binary = binarize_mask(
        Image.fromarray(raw, mode="L"),
        threshold_normalized=threshold_normalized,
        foreground_is_high=foreground_is_high,
    )
    binary_array = np.asarray(binary, dtype=np.uint8) > 0
    fraction = float(binary_array.mean())
    if not minimum_foreground_fraction <= fraction <= maximum_foreground_fraction:
        raise DataValidationError(
            f"Mask foreground fraction {fraction:.6f} outside "
            f"[{minimum_foreground_fraction}, {maximum_foreground_fraction}] "
            f"for {mask_path}"
        )

    raw_unique = np.unique(raw)
    is_binary_source = bool(set(raw_unique.tolist()).issubset({0, 1, 255}))
    return {
        "image_width": image_size[0],
        "image_height": image_size[1],
        "mask_width": mask_size[0],
        "mask_height": mask_size[1],
        "source_mask_binary": is_binary_source,
        "source_mask_min": int(raw.min()),
        "source_mask_max": int(raw.max()),
        "foreground_fraction": fraction,
        "mask_sha256": sha256_file(mask_path),
        "mask_size_bytes": mask_path.stat().st_size,
    }

