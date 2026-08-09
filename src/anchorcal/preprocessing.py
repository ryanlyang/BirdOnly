"""Resolve, audit, serialize, and consume the pinned timm preprocessing.

The effective evaluation resize is deliberately read from the transform built
by timm.  Production code consumes the resulting manifest; it never
reconstructs ``224 / crop_pct`` independently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .errors import PreflightError
from .io import atomic_write_json, sha256_file
from .pretrained import EXPECTED_SHA256, REPOSITORY, REVISION, create_pretrained_vit


SCHEMA_VERSION = "anchorcal-preprocessing-v1"
EXPECTED_TIMM_VERSION = "1.0.28"
EXPECTED_INPUT_SIZE = (3, 224, 224)
EXPECTED_INTERPOLATION = "bicubic"
EXPECTED_ANTIALIAS = True
EXPECTED_CROP_PCT = 0.9
EXPECTED_CROP_MODE = "center"
EXPECTED_MEAN = (0.5, 0.5, 0.5)
EXPECTED_STD = (0.5, 0.5, 0.5)
EXPECTED_EFFECTIVE_RESIZE = 248
PARITY_SHAPES = ((257, 311), (311, 257), (300, 300), (225, 500), (500, 225))
PARITY_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class EvaluationPreprocessing:
    """Validated evaluation and normalization values used by production."""

    input_size: tuple[int, int, int]
    interpolation: str
    antialias: bool
    crop_pct: float
    crop_mode: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    effective_resize_shortest: int

    @property
    def image_size(self) -> int:
        return int(self.input_size[-1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_size": list(self.input_size),
            "interpolation": self.interpolation,
            "antialias": self.antialias,
            "crop_pct": self.crop_pct,
            "crop_mode": self.crop_mode,
            "mean": list(self.mean),
            "std": list(self.std),
            "effective_resize_shortest": self.effective_resize_shortest,
        }


def _plain(value: Any) -> Any:
    """Convert timm/torchvision configuration values to deterministic JSON."""

    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _interpolation_name(value: Any) -> str:
    if hasattr(value, "value"):
        value = value.value
    elif hasattr(value, "name"):
        value = value.name
    text = str(value).lower()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text


def _tuple(value: Any, *, length: int, name: str, cast: Callable[[Any], Any]) -> tuple[Any, ...]:
    try:
        result = tuple(cast(item) for item in value)
    except (TypeError, ValueError) as error:
        raise PreflightError(f"resolved preprocessing {name} is malformed: {value!r}") from error
    if len(result) != length:
        raise PreflightError(
            f"resolved preprocessing {name} has length {len(result)}; expected {length}"
        )
    return result


def _component(transform: Any, class_name: str) -> Any:
    components = list(getattr(transform, "transforms", (transform,)))
    matches = [item for item in components if item.__class__.__name__ == class_name]
    if len(matches) != 1:
        names = [item.__class__.__name__ for item in components]
        raise PreflightError(
            f"timm evaluation transform requires exactly one {class_name}; found {names}"
        )
    return matches[0]


def _square_size(value: Any, *, name: str) -> int:
    if isinstance(value, (tuple, list)):
        if len(value) != 2 or int(value[0]) != int(value[1]):
            raise PreflightError(f"{name} must be a square scalar/2-tuple, found {value!r}")
        value = value[0]
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise PreflightError(f"{name} has invalid size {value!r}") from error
    if result <= 0:
        raise PreflightError(f"{name} must be positive, found {result}")
    return result


def extract_evaluation_preprocessing(
    resolved_config: dict[str, Any], evaluation_transform: Any
) -> EvaluationPreprocessing:
    """Extract and fail-closed validate timm's resolved data configuration.

    ``effective_resize_shortest`` comes from the concrete torchvision Resize
    component produced by :func:`timm.data.create_transform`.
    """

    input_size = _tuple(
        resolved_config.get("input_size"), length=3, name="input_size", cast=int
    )
    interpolation = _interpolation_name(resolved_config.get("interpolation"))
    crop_pct = float(resolved_config.get("crop_pct", float("nan")))
    crop_mode = str(resolved_config.get("crop_mode", "")).lower()
    mean = _tuple(resolved_config.get("mean"), length=3, name="mean", cast=float)
    std = _tuple(resolved_config.get("std"), length=3, name="std", cast=float)

    resize = _component(evaluation_transform, "Resize")
    center_crop = _component(evaluation_transform, "CenterCrop")
    normalize = _component(evaluation_transform, "Normalize")
    effective_resize = _square_size(resize.size, name="timm Resize")
    center_size = _square_size(center_crop.size, name="timm CenterCrop")
    resize_interpolation = _interpolation_name(resize.interpolation)
    antialias = getattr(resize, "antialias", None)
    normalize_mean = _tuple(normalize.mean, length=3, name="Normalize.mean", cast=float)
    normalize_std = _tuple(normalize.std, length=3, name="Normalize.std", cast=float)

    expected_pairs = (
        ("input_size", input_size, EXPECTED_INPUT_SIZE),
        ("interpolation", interpolation, EXPECTED_INTERPOLATION),
        ("crop_mode", crop_mode, EXPECTED_CROP_MODE),
        ("mean", mean, EXPECTED_MEAN),
        ("std", std, EXPECTED_STD),
        ("Resize.interpolation", resize_interpolation, EXPECTED_INTERPOLATION),
        ("Resize.antialias", antialias, EXPECTED_ANTIALIAS),
        ("effective_resize_shortest", effective_resize, EXPECTED_EFFECTIVE_RESIZE),
        ("CenterCrop.size", center_size, EXPECTED_INPUT_SIZE[-1]),
        ("Normalize.mean", normalize_mean, EXPECTED_MEAN),
        ("Normalize.std", normalize_std, EXPECTED_STD),
    )
    for name, actual, expected in expected_pairs:
        if actual != expected:
            raise PreflightError(
                f"resolved pretrained preprocessing {name}={actual!r}; expected {expected!r}"
            )
    if not np.isclose(crop_pct, EXPECTED_CROP_PCT, rtol=0.0, atol=1.0e-12):
        raise PreflightError(
            f"resolved pretrained preprocessing crop_pct={crop_pct!r}; "
            f"expected {EXPECTED_CROP_PCT!r}"
        )
    if input_size[1] != input_size[2]:
        raise PreflightError("resolved pretrained input must be square")
    return EvaluationPreprocessing(
        input_size=input_size,
        interpolation=interpolation,
        antialias=bool(antialias),
        crop_pct=crop_pct,
        crop_mode=crop_mode,
        mean=mean,
        std=std,
        effective_resize_shortest=effective_resize,
    )


def _synthetic_image(width: int, height: int) -> Image.Image:
    yy, xx = np.indices((height, width), dtype=np.int64)
    values = np.stack(
        [
            (17 * xx + 31 * yy + 7) % 256,
            (43 * xx + 11 * yy + 19) % 256,
            (5 * xx + 59 * yy + 101) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(values, mode="RGB")


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def run_candidate_preprocessing_parity(
    evaluation_transform: Callable[[Image.Image], Any],
    preprocessing: EvaluationPreprocessing,
    *,
    shapes: tuple[tuple[int, int], ...] = PARITY_SHAPES,
    tolerance: float = PARITY_TOLERANCE,
) -> dict[str, Any]:
    """Compare AnchorCal evaluation tensors against timm on varied aspect ratios."""

    # Local imports avoid a preprocessing -> transforms import cycle at module load.
    from .transforms import deterministic_image_eval_transform, normalize_image

    if len(shapes) < 3:
        raise ValueError("preprocessing parity requires at least three image shapes")
    cases: list[dict[str, Any]] = []
    maximum = 0.0
    for width, height in shapes:
        image = _synthetic_image(int(width), int(height))
        reference = _array(evaluation_transform(image.copy()))
        candidate_image = deterministic_image_eval_transform(
            image,
            image_size=preprocessing.image_size,
            resize_shortest=preprocessing.effective_resize_shortest,
        ).image
        candidate = normalize_image(
            candidate_image, mean=preprocessing.mean, std=preprocessing.std
        )
        if reference.shape != candidate.shape:
            raise PreflightError(
                f"timm/AnchorCal preprocessing shape mismatch for {(width, height)}: "
                f"{reference.shape} versus {candidate.shape}"
            )
        difference = float(np.max(np.abs(reference - candidate)))
        if not np.isfinite(difference) or difference > tolerance:
            raise PreflightError(
                f"timm/AnchorCal preprocessing parity failed for {(width, height)}: "
                f"max_abs={difference}, tolerance={tolerance}"
            )
        maximum = max(maximum, difference)
        cases.append(
            {
                "source_width": int(width),
                "source_height": int(height),
                "output_shape": list(candidate.shape),
                "max_abs_difference": difference,
            }
        )
    return {
        "status": "passed",
        "tolerance": tolerance,
        "maximum_absolute_difference": maximum,
        "case_count": len(cases),
        "cases": cases,
    }


def resolve_preprocessing_manifest(pretrained_manifest: dict[str, Any]) -> dict[str, Any]:
    """Resolve the pinned timm transform and return its audited manifest."""

    import timm
    from timm.data import create_transform, resolve_model_data_config

    if timm.__version__ != EXPECTED_TIMM_VERSION:
        raise PreflightError(
            f"timm {EXPECTED_TIMM_VERSION} is required, found {timm.__version__}"
        )
    if pretrained_manifest.get("revision") != REVISION:
        raise PreflightError("preprocessing cannot bind to an unexpected model revision")
    if pretrained_manifest.get("weights_sha256") != EXPECTED_SHA256:
        raise PreflightError("preprocessing cannot bind to unexpected model weights")
    model = create_pretrained_vit(pretrained_manifest["weights_path"])
    resolved = resolve_model_data_config(model)
    evaluation_transform = create_transform(**resolved, is_training=False)
    preprocessing = extract_evaluation_preprocessing(resolved, evaluation_transform)
    parity = run_candidate_preprocessing_parity(evaluation_transform, preprocessing)
    components = list(
        getattr(evaluation_transform, "transforms", (evaluation_transform,))
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "model_repository": pretrained_manifest["repository"],
            "model_revision": pretrained_manifest["revision"],
            "weights_sha256": pretrained_manifest["weights_sha256"],
            "timm_version": timm.__version__,
            "resolution_api": "timm.data.resolve_model_data_config",
            "transform_api": "timm.data.create_transform(is_training=False)",
        },
        "resolved_model_data_config": _plain(resolved),
        "evaluation": preprocessing.to_dict(),
        "transform_components": [
            {"class": item.__class__.__name__, "repr": repr(item)}
            for item in components
        ],
        "parity": parity,
    }


def _validate_serialized_manifest(value: dict[str, Any]) -> EvaluationPreprocessing:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise PreflightError("preprocessing manifest schema is missing or unsupported")
    source = value.get("source", {})
    if (
        source.get("model_repository") != REPOSITORY
        or source.get("model_revision") != REVISION
        or source.get("weights_sha256") != EXPECTED_SHA256
        or source.get("timm_version") != EXPECTED_TIMM_VERSION
    ):
        raise PreflightError("preprocessing manifest is not bound to the pinned model/runtime")
    evaluation = value.get("evaluation", {})
    preprocessing = EvaluationPreprocessing(
        input_size=_tuple(
            evaluation.get("input_size"), length=3, name="input_size", cast=int
        ),
        interpolation=str(evaluation.get("interpolation", "")).lower(),
        antialias=evaluation.get("antialias") is True,
        crop_pct=float(evaluation.get("crop_pct", float("nan"))),
        crop_mode=str(evaluation.get("crop_mode", "")).lower(),
        mean=_tuple(evaluation.get("mean"), length=3, name="mean", cast=float),
        std=_tuple(evaluation.get("std"), length=3, name="std", cast=float),
        effective_resize_shortest=int(
            evaluation.get("effective_resize_shortest", -1)
        ),
    )
    expected = EvaluationPreprocessing(
        input_size=EXPECTED_INPUT_SIZE,
        interpolation=EXPECTED_INTERPOLATION,
        antialias=EXPECTED_ANTIALIAS,
        crop_pct=EXPECTED_CROP_PCT,
        crop_mode=EXPECTED_CROP_MODE,
        mean=EXPECTED_MEAN,
        std=EXPECTED_STD,
        effective_resize_shortest=EXPECTED_EFFECTIVE_RESIZE,
    )
    if preprocessing != expected:
        raise PreflightError(
            f"serialized preprocessing values changed: {preprocessing!r}; expected {expected!r}"
        )
    parity = value.get("parity", {})
    if parity.get("status") != "passed" or int(parity.get("case_count", 0)) < 3:
        raise PreflightError("preprocessing manifest lacks a passed multi-shape parity audit")
    return preprocessing


def write_preprocessing_manifest(path: str | Path, value: dict[str, Any]) -> None:
    _validate_serialized_manifest(value)
    atomic_write_json(path, value)


def preprocessing_from_manifest(value: dict[str, Any]) -> EvaluationPreprocessing:
    """Return the validated runtime values from an in-memory manifest."""

    return _validate_serialized_manifest(value)


def load_preprocessing_manifest(
    output_root: str | Path, *, verify_preflight_report: bool = True
) -> EvaluationPreprocessing:
    """Load the immutable production preprocessing contract from preflight."""

    root = Path(output_root)
    path = root / "preflight" / "preprocessing_manifest.json"
    if not path.is_file():
        raise PreflightError(f"preprocessing manifest is missing: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError("preprocessing manifest is unreadable") from error
    preprocessing = _validate_serialized_manifest(value)
    if verify_preflight_report:
        report_path = root / "preflight" / "report.json"
        if not report_path.is_file():
            raise PreflightError("passed preflight report is required to consume preprocessing")
        try:
            with report_path.open("r", encoding="utf-8") as handle:
                report = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise PreflightError("preflight report is unreadable") from error
        binding = report.get("preprocessing", {})
        if (
            report.get("status") != "passed"
            or Path(binding.get("manifest_path", "")).resolve() != path.resolve()
            or binding.get("manifest_sha256") != sha256_file(path)
            or int(binding.get("effective_resize_shortest", -1))
            != preprocessing.effective_resize_shortest
        ):
            raise PreflightError("preprocessing manifest does not match the passed preflight")
    return preprocessing
