"""Locked image/mask transforms and background-only interventions.

All scientific randomness in this module is stateless with respect to worker
processes.  A training view is a pure function of ``run_seed``, ``epoch``,
``img_id``, and ``purpose``.  Foreground green-screening is deliberately a
source-resolution operation so interpolation can never mix the hidden source
background into a retained bird edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
from PIL import Image

from .errors import AuditFailure, PreflightError
from .masks import classify_patches, dilate_mask
from .seeds import stateless_rng


IMAGE_SIZE = 224
PATCH_SIZE = 16
EVALUATION_CROP_PCT = 0.9
TRAIN_SCALE = (0.70, 1.00)
TRAIN_RATIO = (3.0 / 4.0, 4.0 / 3.0)
HORIZONTAL_FLIP_PROBABILITY = 0.5
MAX_VISIBLE_FOREGROUND_ATTEMPTS = 10
MASK_DILATION_RADIUS = 8
NORMALIZATION_MEAN = (0.5, 0.5, 0.5)
NORMALIZATION_STD = (0.5, 0.5, 0.5)
MAX_FALLBACK_RATE = 0.001


@dataclass(frozen=True)
class Geometry:
    """A crop in source coordinates followed by resize and optional flip."""

    left: int
    top: int
    width: int
    height: int
    output_size: int = IMAGE_SIZE
    horizontal_flip: bool = False
    kind: str = "random_resized_crop"


@dataclass(frozen=True)
class JointTransformResult:
    """One transformed RGB image, exact binary mask, and replay metadata."""

    image: Image.Image
    mask: np.ndarray
    geometry: Geometry
    attempt_count: int
    fallback_used: bool


@dataclass(frozen=True)
class ImageTransformResult:
    """A mask-free transform result for ordinary candidate models."""

    image: Image.Image
    geometry: Geometry


@dataclass(frozen=True)
class CriterionPatchEligibility:
    """Model-independent pure-foreground and safe-background patch sets."""

    foreground_indices: tuple[int, ...]
    background_indices: tuple[int, ...]
    eligible: bool
    exclusion_reasons: tuple[str, ...]


def _binary_mask(mask: np.ndarray | Image.Image, expected_size: tuple[int, int]) -> np.ndarray:
    if isinstance(mask, Image.Image):
        array = np.asarray(mask)
    else:
        array = np.asarray(mask)
    if array.ndim == 3:
        if not np.all(array == array[..., :1]):
            raise ValueError("mask channels must be identical")
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError("mask must have shape HxW")
    if (array.shape[1], array.shape[0]) != expected_size:
        raise ValueError(
            f"image and mask sizes differ: image={expected_size}, "
            f"mask={(array.shape[1], array.shape[0])}"
        )
    return array > 0


def _mask_pil(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L")


def green_screen_source(
    image: Image.Image,
    mask: np.ndarray | Image.Image,
    *,
    green_rgb: tuple[int, int, int] = (0, 255, 0),
) -> Image.Image:
    """Replace background at source resolution, before any interpolation."""

    rgb = image.convert("RGB")
    foreground = _binary_mask(mask, rgb.size)
    if any(channel < 0 or channel > 255 for channel in green_rgb):
        raise ValueError("green_rgb channels must lie in [0, 255]")
    source = np.asarray(rgb, dtype=np.uint8)
    output = np.empty_like(source)
    output[...] = np.asarray(green_rgb, dtype=np.uint8)
    output[foreground] = source[foreground]
    return Image.fromarray(output, mode="RGB")


def evaluation_resize_shortest(
    image_size: int = IMAGE_SIZE, crop_pct: float = EVALUATION_CROP_PCT
) -> int:
    """Resolve timm's integer evaluation resize from image size/crop percent."""

    if image_size <= 0 or not 0.0 < crop_pct <= 1.0:
        raise ValueError("invalid evaluation image size or crop percentage")
    return int(image_size / crop_pct)


def _resize_shortest_pair(
    image: Image.Image, mask: np.ndarray, shortest: int
) -> tuple[Image.Image, np.ndarray]:
    width, height = image.size
    if width <= height:
        resized_width = shortest
        resized_height = int(shortest * height / width)
    else:
        resized_height = shortest
        resized_width = int(shortest * width / height)
    resized_size = (max(1, resized_width), max(1, resized_height))
    resized_image = image.resize(resized_size, resample=Image.Resampling.BICUBIC)
    resized_mask = _mask_pil(mask).resize(
        resized_size, resample=Image.Resampling.NEAREST
    )
    return resized_image, np.asarray(resized_mask, dtype=np.uint8) > 0


def _center_crop_pair(
    image: Image.Image, mask: np.ndarray, output_size: int
) -> tuple[Image.Image, np.ndarray, Geometry]:
    width, height = image.size
    if width < output_size or height < output_size:
        raise ValueError(
            f"cannot center crop {output_size} from transformed size {image.size}"
        )
    # Match torchvision CenterCrop's round-to-nearest placement.
    left = int(round((width - output_size) / 2.0))
    top = int(round((height - output_size) / 2.0))
    box = (left, top, left + output_size, top + output_size)
    return (
        image.crop(box),
        mask[top : top + output_size, left : left + output_size].copy(),
        Geometry(
            left=left,
            top=top,
            width=output_size,
            height=output_size,
            output_size=output_size,
            horizontal_flip=False,
            kind="evaluation_center_crop_after_resize",
        ),
    )


def deterministic_eval_transform(
    image: Image.Image,
    mask: np.ndarray | Image.Image,
    *,
    image_size: int = IMAGE_SIZE,
    crop_pct: float = EVALUATION_CROP_PCT,
    resize_shortest: int | None = None,
    require_visible_foreground: bool = True,
) -> JointTransformResult:
    """Apply locked evaluation geometry.

    Production callers pass ``resize_shortest`` from the serialized timm
    preprocessing manifest.  The formula fallback exists only so this pure
    transform remains convenient to test in isolation.
    """

    rgb = image.convert("RGB")
    binary = _binary_mask(mask, rgb.size)
    if resize_shortest is None:
        resize_shortest = evaluation_resize_shortest(image_size, crop_pct)
    if int(resize_shortest) < image_size:
        raise ValueError("evaluation shortest-side resize cannot be smaller than crop")
    resized_image, resized_mask = _resize_shortest_pair(
        rgb, binary, int(resize_shortest)
    )
    cropped_image, cropped_mask, geometry = _center_crop_pair(
        resized_image, resized_mask, image_size
    )
    if require_visible_foreground and not cropped_mask.any():
        raise PreflightError(
            "deterministic evaluation crop contains no foreground pixels"
        )
    return JointTransformResult(
        image=cropped_image,
        mask=cropped_mask,
        geometry=geometry,
        attempt_count=1,
        fallback_used=False,
    )


def deterministic_image_eval_transform(
    image: Image.Image,
    *,
    image_size: int = IMAGE_SIZE,
    crop_pct: float = EVALUATION_CROP_PCT,
    resize_shortest: int | None = None,
) -> ImageTransformResult:
    """Apply evaluation geometry without loading or consulting any mask."""

    blank = np.zeros((image.height, image.width), dtype=bool)
    transformed = deterministic_eval_transform(
        image,
        blank,
        image_size=image_size,
        crop_pct=crop_pct,
        resize_shortest=resize_shortest,
        require_visible_foreground=False,
    )
    return ImageTransformResult(transformed.image, transformed.geometry)


def foreground_eval_transform(
    image: Image.Image,
    mask: np.ndarray | Image.Image,
    *,
    green_rgb: tuple[int, int, int] = (0, 255, 0),
    image_size: int = IMAGE_SIZE,
    crop_pct: float = EVALUATION_CROP_PCT,
    resize_shortest: int | None = None,
) -> JointTransformResult:
    """Green-screen at source resolution, then apply evaluation geometry."""

    binary = _binary_mask(mask, image.size)
    composed = green_screen_source(image, binary, green_rgb=green_rgb)
    return deterministic_eval_transform(
        composed,
        binary,
        image_size=image_size,
        crop_pct=crop_pct,
        resize_shortest=resize_shortest,
        require_visible_foreground=True,
    )


def _random_resized_crop_geometry(
    width: int,
    height: int,
    rng: np.random.Generator,
    *,
    output_size: int,
    scale: tuple[float, float],
    ratio: tuple[float, float],
) -> Geometry:
    """Torchvision-compatible RandomResizedCrop parameter distribution."""

    area = width * height
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target_area = area * float(rng.uniform(scale[0], scale[1]))
        aspect_ratio = math.exp(float(rng.uniform(log_ratio[0], log_ratio[1])))
        crop_width = int(round(math.sqrt(target_area * aspect_ratio)))
        crop_height = int(round(math.sqrt(target_area / aspect_ratio)))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            left = int(rng.integers(0, width - crop_width + 1))
            top = int(rng.integers(0, height - crop_height + 1))
            return Geometry(left, top, crop_width, crop_height, output_size)

    source_ratio = width / height
    if source_ratio < ratio[0]:
        crop_width = width
        crop_height = int(round(crop_width / ratio[0]))
    elif source_ratio > ratio[1]:
        crop_height = height
        crop_width = int(round(crop_height * ratio[1]))
    else:
        crop_width, crop_height = width, height
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return Geometry(left, top, crop_width, crop_height, output_size)


def _apply_geometry(
    image: Image.Image, mask: np.ndarray, geometry: Geometry
) -> tuple[Image.Image, np.ndarray]:
    box = (
        geometry.left,
        geometry.top,
        geometry.left + geometry.width,
        geometry.top + geometry.height,
    )
    transformed_image = image.crop(box).resize(
        (geometry.output_size, geometry.output_size),
        resample=Image.Resampling.BICUBIC,
    )
    transformed_mask_image = _mask_pil(mask).crop(box).resize(
        (geometry.output_size, geometry.output_size),
        resample=Image.Resampling.NEAREST,
    )
    transformed_mask = np.asarray(transformed_mask_image, dtype=np.uint8) > 0
    if geometry.horizontal_flip:
        transformed_image = transformed_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        transformed_mask = np.fliplr(transformed_mask).copy()
    return transformed_image, transformed_mask


def stateless_train_transform(
    image: Image.Image,
    mask: np.ndarray | Image.Image,
    *,
    run_seed: int,
    epoch: int,
    img_id: int | str,
    purpose: str,
    require_visible_foreground: bool,
    image_size: int = IMAGE_SIZE,
    scale: tuple[float, float] = TRAIN_SCALE,
    ratio: tuple[float, float] = TRAIN_RATIO,
    horizontal_flip_probability: float = HORIZONTAL_FLIP_PROBABILITY,
    max_attempts: int = MAX_VISIBLE_FOREGROUND_ATTEMPTS,
    crop_pct: float = EVALUATION_CROP_PCT,
    resize_shortest: int | None = None,
) -> JointTransformResult:
    """Create a replayable crop/flip, rejecting empty branch crops if requested."""

    if not purpose:
        raise ValueError("purpose must be a nonempty named stochastic purpose")
    if max_attempts < 1 or max_attempts > MAX_VISIBLE_FOREGROUND_ATTEMPTS:
        raise ValueError("max_attempts must lie in [1, 10]")
    if not 0.0 <= horizontal_flip_probability <= 1.0:
        raise ValueError("horizontal_flip_probability must lie in [0, 1]")
    rgb = image.convert("RGB")
    binary = _binary_mask(mask, rgb.size)
    rng = stateless_rng(run_seed, epoch, img_id, purpose)

    for attempt in range(1, max_attempts + 1):
        geometry = _random_resized_crop_geometry(
            *rgb.size,
            rng,
            output_size=image_size,
            scale=scale,
            ratio=ratio,
        )
        geometry = replace(
            geometry,
            horizontal_flip=bool(rng.random() < horizontal_flip_probability),
        )
        transformed_image, transformed_mask = _apply_geometry(rgb, binary, geometry)
        if not require_visible_foreground or transformed_mask.any():
            return JointTransformResult(
                image=transformed_image,
                mask=transformed_mask,
                geometry=geometry,
                attempt_count=attempt,
                fallback_used=False,
            )

    fallback = deterministic_eval_transform(
        rgb,
        binary,
        image_size=image_size,
        crop_pct=crop_pct,
        resize_shortest=resize_shortest,
        require_visible_foreground=True,
    )
    return JointTransformResult(
        image=fallback.image,
        mask=fallback.mask,
        geometry=fallback.geometry,
        attempt_count=max_attempts,
        fallback_used=True,
    )


def stateless_image_train_transform(
    image: Image.Image,
    *,
    run_seed: int,
    epoch: int,
    img_id: int | str,
    purpose: str = "candidate_train",
    image_size: int = IMAGE_SIZE,
    scale: tuple[float, float] = TRAIN_SCALE,
    ratio: tuple[float, float] = TRAIN_RATIO,
    horizontal_flip_probability: float = HORIZONTAL_FLIP_PROBABILITY,
) -> ImageTransformResult:
    """Create the ordinary candidate view without accepting a mask argument."""

    if not purpose:
        raise ValueError("purpose must be a nonempty named stochastic purpose")
    if not 0.0 <= horizontal_flip_probability <= 1.0:
        raise ValueError("horizontal_flip_probability must lie in [0, 1]")
    rgb = image.convert("RGB")
    rng = stateless_rng(run_seed, epoch, img_id, purpose)
    geometry = _random_resized_crop_geometry(
        *rgb.size,
        rng,
        output_size=image_size,
        scale=scale,
        ratio=ratio,
    )
    geometry = replace(
        geometry,
        horizontal_flip=bool(rng.random() < horizontal_flip_probability),
    )
    blank = np.zeros((rgb.height, rgb.width), dtype=bool)
    transformed, _ = _apply_geometry(rgb, blank, geometry)
    return ImageTransformResult(transformed, geometry)


def foreground_train_transform(
    image: Image.Image,
    mask: np.ndarray | Image.Image,
    *,
    run_seed: int,
    epoch: int,
    img_id: int | str,
    purpose: str = "foreground_branch_train",
    green_rgb: tuple[int, int, int] = (0, 255, 0),
    **transform_options: object,
) -> JointTransformResult:
    """Green-screen at source resolution, then make a replayable branch view."""

    binary = _binary_mask(mask, image.size)
    composed = green_screen_source(image, binary, green_rgb=green_rgb)
    return stateless_train_transform(
        composed,
        binary,
        run_seed=run_seed,
        epoch=epoch,
        img_id=img_id,
        purpose=purpose,
        require_visible_foreground=True,
        **transform_options,
    )


def check_fallback_rate(
    fallback_count: int,
    sample_count: int,
    *,
    maximum_rate: float = MAX_FALLBACK_RATE,
) -> float:
    """Return the fallback rate or fail the locked 0.1-percent branch gate."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if fallback_count < 0 or fallback_count > sample_count:
        raise ValueError("fallback_count must lie in [0, sample_count]")
    if not 0.0 <= maximum_rate <= 1.0:
        raise ValueError("maximum_rate must lie in [0, 1]")
    rate = fallback_count / sample_count
    if rate > maximum_rate:
        raise AuditFailure(
            f"branch crop fallback rate {rate:.8f} exceeds {maximum_rate:.8f}"
        )
    return rate


def normalize_image(
    image: Image.Image | np.ndarray,
    *,
    mean: tuple[float, float, float] = NORMALIZATION_MEAN,
    std: tuple[float, float, float] = NORMALIZATION_STD,
) -> np.ndarray:
    """Convert unnormalized RGB to a float32 CHW array using locked statistics."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must have shape HxWx3")
    if np.issubdtype(array.dtype, np.integer):
        rgb = array.astype(np.float32) / 255.0
    else:
        rgb = array.astype(np.float32)
        if not np.isfinite(rgb).all() or rgb.min() < 0.0 or rgb.max() > 1.0:
            raise ValueError("floating RGB input must be finite and lie in [0, 1]")
    mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_array = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    if np.any(std_array <= 0):
        raise ValueError("normalization standard deviations must be positive")
    return ((rgb - mean_array) / std_array).transpose(2, 0, 1).copy()


def final_coordinate_dilated_mask(
    mask: np.ndarray | Image.Image,
    *,
    radius: int = MASK_DILATION_RADIUS,
    expected_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Apply the locked disk dilation to an already transformed square mask."""

    binary = _binary_mask(mask, (expected_size, expected_size))
    return dilate_mask(binary, radius=radius)


def criterion_patch_eligibility(
    mask: np.ndarray | Image.Image,
    *,
    radius: int = MASK_DILATION_RADIUS,
    patch_size: int = PATCH_SIZE,
    expected_size: int = IMAGE_SIZE,
) -> CriterionPatchEligibility:
    """Build the common, model-independent patch eligibility definition."""

    binary = _binary_mask(mask, (expected_size, expected_size))
    patch_sets = classify_patches(binary, radius=radius, patch_size=patch_size)
    foreground = tuple(int(value) for value in np.flatnonzero(patch_sets.foreground))
    background = tuple(int(value) for value in np.flatnonzero(patch_sets.background))
    reasons = []
    if not foreground:
        reasons.append("zero_pure_foreground_patches")
    if not background:
        reasons.append("zero_safe_background_patches")
    return CriterionPatchEligibility(
        foreground_indices=foreground,
        background_indices=background,
        eligible=not reasons,
        exclusion_reasons=tuple(reasons),
    )


def require_common_eligibility(
    records: Iterable[tuple[int | str, int, CriterionPatchEligibility]],
    *,
    minimum_per_class: int = 50,
) -> dict[int, int]:
    """Enforce the common saliency/swap/blur eligibility floor by class."""

    counts: dict[int, int] = {}
    for _img_id, label, eligibility in records:
        if eligibility.eligible:
            counts[int(label)] = counts.get(int(label), 0) + 1
    for label in (0, 1):
        if counts.get(label, 0) < minimum_per_class:
            raise AuditFailure(
                f"common criterion eligibility has {counts.get(label, 0)} examples "
                f"for class {label}; requires {minimum_per_class}"
            )
    return {label: counts.get(label, 0) for label in (0, 1)}


def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """Return the exact normalized radius-ceil(3*sigma) Gaussian kernel."""

    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    radius = int(math.ceil(3.0 * sigma))
    coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (coordinates / np.float32(sigma)) ** 2)
    kernel /= kernel.sum(dtype=np.float64)
    return kernel.astype(np.float32)


def _separable_reflect_blur(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import convolve1d
    except ImportError as error:
        raise RuntimeError("scipy is required for background-only Gaussian blur") from error
    result = convolve1d(array, kernel, axis=0, mode="reflect")
    return convolve1d(result, kernel, axis=1, mode="reflect").astype(np.float32)


def mask_normalized_background_blur(
    image: Image.Image | np.ndarray,
    mask: np.ndarray | Image.Image,
    *,
    sigma: float,
) -> np.ndarray:
    """Blur only background evidence without allowing foreground color bleed.

    The input and returned image are unnormalized RGB.  The returned float32
    array lies in ``[0, 1]`` and preserves every foreground pixel exactly.
    """

    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("image must have shape HxWx3")
    if np.issubdtype(rgb.dtype, np.integer):
        rgb = rgb.astype(np.float32) / 255.0
    else:
        rgb = rgb.astype(np.float32)
        if not np.isfinite(rgb).all() or rgb.min() < 0.0 or rgb.max() > 1.0:
            raise ValueError("floating RGB input must be finite and lie in [0, 1]")
    height, width = rgb.shape[:2]
    foreground = _binary_mask(mask, (width, height))
    background = (~foreground).astype(np.float32)
    kernel = gaussian_kernel_1d(sigma)
    denominator = _separable_reflect_blur(background, kernel)
    numerator = _separable_reflect_blur(rgb * background[..., None], kernel)
    blurred_background = numerator / np.maximum(denominator[..., None], 1e-6)
    output = np.where(foreground[..., None], rgb, blurred_background)
    return np.clip(output, 0.0, 1.0).astype(np.float32)
