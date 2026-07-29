"""Geometry-safe PIL transforms shared by images and segmentation masks."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np
from PIL import Image

from setv.errors import DataValidationError


class JointTransform(Protocol):
    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        ...


def _assert_aligned(image: Image.Image, mask: Image.Image) -> None:
    if image.size != mask.size:
        raise DataValidationError(
            f"Image and mask must have identical geometry before a joint "
            f"transform; got image={image.size}, mask={mask.size}"
        )


class JointCompose:
    def __init__(self, transforms: Iterable[JointTransform]):
        self.transforms = tuple(transforms)

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        _assert_aligned(image, mask)
        for transform in self.transforms:
            image, mask = transform(image, mask)
            _assert_aligned(image, mask)
        return image, mask


@dataclass(frozen=True)
class JointResizeShortest:
    size: int

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        _assert_aligned(image, mask)
        width, height = image.size
        if min(width, height) == self.size:
            return image, mask
        scale = self.size / min(width, height)
        resized = (int(round(width * scale)), int(round(height * scale)))
        return (
            image.resize(resized, Image.Resampling.BICUBIC),
            mask.resize(resized, Image.Resampling.NEAREST),
        )


@dataclass(frozen=True)
class JointCenterCrop:
    size: int

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        _assert_aligned(image, mask)
        width, height = image.size
        if width < self.size or height < self.size:
            raise DataValidationError(
                f"Cannot center-crop {self.size} from image of size {image.size}"
            )
        left = (width - self.size) // 2
        top = (height - self.size) // 2
        box = (left, top, left + self.size, top + self.size)
        return image.crop(box), mask.crop(box)


class JointRandomHorizontalFlip:
    def __init__(self, probability: float = 0.5, rng: random.Random | None = None):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must lie in [0, 1]")
        self.probability = probability
        self.rng = rng if rng is not None else random.Random()

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        _assert_aligned(image, mask)
        if self.rng.random() < self.probability:
            return (
                image.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
                mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
            )
        return image, mask


class JointRandomResizedCrop:
    """Torchvision-compatible random resized crop using one shared crop box."""

    def __init__(
        self,
        size: int,
        scale: tuple[float, float] = (0.7, 1.0),
        ratio: tuple[float, float] = (0.75, 4.0 / 3.0),
        rng: random.Random | None = None,
    ):
        if not 0.0 < scale[0] <= scale[1]:
            raise ValueError("invalid scale range")
        if not 0.0 < ratio[0] <= ratio[1]:
            raise ValueError("invalid ratio range")
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.rng = rng if rng is not None else random.Random()

    def _parameters(self, width: int, height: int) -> tuple[int, int, int, int]:
        area = width * height
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        for _ in range(10):
            target_area = area * self.rng.uniform(*self.scale)
            aspect = math.exp(self.rng.uniform(*log_ratio))
            crop_width = int(round(math.sqrt(target_area * aspect)))
            crop_height = int(round(math.sqrt(target_area / aspect)))
            if 0 < crop_width <= width and 0 < crop_height <= height:
                left = self.rng.randint(0, width - crop_width)
                top = self.rng.randint(0, height - crop_height)
                return left, top, crop_width, crop_height

        input_ratio = width / height
        if input_ratio < self.ratio[0]:
            crop_width = width
            crop_height = int(round(crop_width / self.ratio[0]))
        elif input_ratio > self.ratio[1]:
            crop_height = height
            crop_width = int(round(crop_height * self.ratio[1]))
        else:
            crop_width, crop_height = width, height
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        return left, top, crop_width, crop_height

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        _assert_aligned(image, mask)
        left, top, width, height = self._parameters(*image.size)
        box = (left, top, left + width, top + height)
        return (
            image.crop(box).resize((self.size, self.size), Image.Resampling.BICUBIC),
            mask.crop(box).resize((self.size, self.size), Image.Resampling.NEAREST),
        )


def binarize_mask(
    mask: Image.Image,
    threshold_normalized: float = 0.5,
    foreground_is_high: bool = True,
    *,
    map_format: str = "threshold",
    foreground_class_ids: Iterable[int] = (1,),
) -> Image.Image:
    """Decode a source mask to an exact 0/255 binary PIL image.

    ``voc_colormap_class_ids`` preserves the categorical RGB encoding and
    rejects every color that is not in the Pascal/VOC palette. ``threshold``
    remains available for legacy binary fixtures and non-categorical masks.
    """
    if map_format == "voc_colormap_class_ids":
        rgb = np.asarray(mask.convert("RGB"), dtype=np.uint8)
        packed = (
            (rgb[..., 0].astype(np.uint32) << 16)
            | (rgb[..., 1].astype(np.uint32) << 8)
            | rgb[..., 2].astype(np.uint32)
        )
        palette = _voc_palette()
        palette_packed = (
            (palette[:, 0].astype(np.uint32) << 16)
            | (palette[:, 1].astype(np.uint32) << 8)
            | palette[:, 2].astype(np.uint32)
        )
        lookup = {int(color): class_id for class_id, color in enumerate(palette_packed)}
        unique = np.unique(packed)
        unknown = [int(color) for color in unique if int(color) not in lookup]
        if unknown:
            colors = [
                [(value >> 16) & 255, (value >> 8) & 255, value & 255]
                for value in unknown[:10]
            ]
            raise DataValidationError(
                f"Unexpected VOC colors in categorical mask: {colors}"
            )
        class_ids = np.empty(packed.shape, dtype=np.uint8)
        for color in unique:
            class_ids[packed == color] = lookup[int(color)]
        selected = tuple(int(value) for value in foreground_class_ids)
        if not selected or any(value < 0 or value > 255 for value in selected):
            raise ValueError("foreground_class_ids must contain VOC IDs in [0, 255]")
        foreground = np.isin(class_ids, np.asarray(selected, dtype=np.uint8))
        return Image.fromarray(foreground.astype(np.uint8) * 255, mode="L")
    if map_format != "threshold":
        raise ValueError(f"Unsupported mask map_format: {map_format}")
    if not 0.0 <= threshold_normalized <= 1.0:
        raise ValueError("threshold_normalized must lie in [0, 1]")
    array = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    foreground = array >= threshold_normalized
    if not foreground_is_high:
        foreground = ~foreground
    return Image.fromarray((foreground.astype(np.uint8) * 255), mode="L")


def _voc_palette() -> np.ndarray:
    """Return the canonical 256-entry Pascal/VOC RGB palette."""
    palette = np.zeros((256, 3), dtype=np.uint8)
    for class_id in range(256):
        value = class_id
        for bit in range(8):
            palette[class_id, 0] |= ((value >> 0) & 1) << (7 - bit)
            palette[class_id, 1] |= ((value >> 1) & 1) << (7 - bit)
            palette[class_id, 2] |= ((value >> 2) & 1) << (7 - bit)
            value >>= 3
    return palette


def green_fill(
    image: Image.Image,
    mask: Image.Image,
    *,
    keep_foreground: bool,
    green_rgb: tuple[int, int, int] = (0, 255, 0),
) -> Image.Image:
    """Compose an RGB image with an exact raw-RGB green fill."""
    _assert_aligned(image, mask)
    image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    foreground = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    output = np.empty_like(image_array)
    output[...] = np.asarray(green_rgb, dtype=np.uint8)
    selected = foreground if keep_foreground else ~foreground
    output[selected] = image_array[selected]
    return Image.fromarray(output, mode="RGB")


def build_eval_transform(config: dict) -> JointCompose:
    transform = config["transforms"]
    return JointCompose(
        [
            JointResizeShortest(int(transform["evaluation_resize_shortest"])),
            JointCenterCrop(int(transform["image_size"])),
        ]
    )


def build_train_transform(config: dict, seed: int) -> JointCompose:
    transform = config["transforms"]
    rng = random.Random(seed)
    return JointCompose(
        [
            JointRandomResizedCrop(
                int(transform["image_size"]),
                tuple(float(value) for value in transform["train_random_resized_crop_scale"]),
                tuple(float(value) for value in transform["train_random_resized_crop_ratio"]),
                rng=rng,
            ),
            JointRandomHorizontalFlip(
                float(transform["train_horizontal_flip_probability"]), rng=rng
            ),
        ]
    )
