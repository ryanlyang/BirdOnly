from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from anchorcal.errors import PreflightError
from anchorcal.io import atomic_write_json, sha256_file
from anchorcal.preprocessing import (
    EXPECTED_EFFECTIVE_RESIZE,
    EXPECTED_SHA256,
    EXPECTED_TIMM_VERSION,
    REPOSITORY,
    REVISION,
    SCHEMA_VERSION,
    extract_evaluation_preprocessing,
    load_preprocessing_manifest,
    run_candidate_preprocessing_parity,
    write_preprocessing_manifest,
)
from anchorcal.runtime import REQUIRED_PACKAGES
from anchorcal.transforms import (
    deterministic_image_eval_transform,
    normalize_image,
)


class Resize:
    def __init__(self, *, antialias: bool = True, size: int = 248) -> None:
        self.size = size
        self.interpolation = "bicubic"
        self.antialias = antialias


class CenterCrop:
    def __init__(self) -> None:
        self.size = (224, 224)


class Normalize:
    def __init__(self) -> None:
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)


class FakeCompose:
    def __init__(self, resize: Resize | None = None) -> None:
        self.transforms = [resize or Resize(), CenterCrop(), Normalize()]


def _resolved_config() -> dict[str, object]:
    return {
        "input_size": (3, 224, 224),
        "interpolation": "bicubic",
        "crop_pct": 0.9,
        "crop_mode": "center",
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
    }


def _manifest() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "model_repository": REPOSITORY,
            "model_revision": REVISION,
            "weights_sha256": EXPECTED_SHA256,
            "timm_version": EXPECTED_TIMM_VERSION,
        },
        "resolved_model_data_config": _resolved_config(),
        "evaluation": {
            **_resolved_config(),
            "antialias": True,
            "effective_resize_shortest": EXPECTED_EFFECTIVE_RESIZE,
        },
        "transform_components": [],
        "parity": {"status": "passed", "case_count": 5},
    }


class AnchorCalPreprocessingTests(unittest.TestCase):
    def test_runtime_gate_includes_final_analysis_dependency(self) -> None:
        self.assertIn("matplotlib", REQUIRED_PACKAGES)

    def test_extracts_effective_resize_from_concrete_timm_transform(self) -> None:
        result = extract_evaluation_preprocessing(_resolved_config(), FakeCompose())
        self.assertEqual(result.effective_resize_shortest, 248)
        self.assertEqual(result.input_size, (3, 224, 224))
        self.assertEqual(result.interpolation, "bicubic")
        self.assertTrue(result.antialias)
        self.assertEqual(result.crop_mode, "center")

    def test_extraction_rejects_antialias_or_resize_drift(self) -> None:
        with self.assertRaises(PreflightError):
            extract_evaluation_preprocessing(
                _resolved_config(), FakeCompose(Resize(antialias=False))
            )
        with self.assertRaises(PreflightError):
            extract_evaluation_preprocessing(
                _resolved_config(), FakeCompose(Resize(size=249))
            )

    def test_candidate_parity_runs_multiple_aspect_ratios_and_fails_closed(self) -> None:
        preprocessing = extract_evaluation_preprocessing(
            _resolved_config(), FakeCompose()
        )

        def reference(image: Image.Image) -> np.ndarray:
            transformed = deterministic_image_eval_transform(
                image,
                image_size=224,
                resize_shortest=248,
            )
            return normalize_image(transformed.image)

        result = run_candidate_preprocessing_parity(
            reference,
            preprocessing,
            shapes=((257, 311), (311, 257), (300, 300)),
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["case_count"], 3)
        with self.assertRaises(PreflightError):
            run_candidate_preprocessing_parity(
                lambda image: np.zeros((3, 224, 224), dtype=np.float32),
                preprocessing,
                shapes=((257, 311), (311, 257), (300, 300)),
            )

    def test_loader_verifies_preflight_hash_and_serialized_effective_resize(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "preflight" / "preprocessing_manifest.json"
            write_preprocessing_manifest(path, _manifest())
            atomic_write_json(
                root / "preflight" / "report.json",
                {
                    "status": "passed",
                    "preprocessing": {
                        "manifest_path": str(path.resolve()),
                        "manifest_sha256": sha256_file(path),
                        "effective_resize_shortest": 248,
                    },
                },
            )
            result = load_preprocessing_manifest(root)
            self.assertEqual(result.effective_resize_shortest, 248)
            value = json.loads(path.read_text(encoding="utf-8"))
            value["parity"]["note"] = "tamper"
            atomic_write_json(path, value)
            with self.assertRaises(PreflightError):
                load_preprocessing_manifest(root)

    def test_explicit_effective_resize_controls_candidate_geometry(self) -> None:
        yy, xx = np.indices((291, 407))
        image = Image.fromarray(
            np.stack(
                [(xx + yy) % 256, (3 * xx) % 256, (5 * yy) % 256], axis=-1
            ).astype(np.uint8),
            mode="RGB",
        )
        locked = deterministic_image_eval_transform(image, resize_shortest=248)
        changed = deterministic_image_eval_transform(image, resize_shortest=260)
        self.assertFalse(
            np.array_equal(np.asarray(locked.image), np.asarray(changed.image))
        )


if __name__ == "__main__":
    unittest.main()
