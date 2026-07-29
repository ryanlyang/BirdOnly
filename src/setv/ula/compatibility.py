"""Read-only Tigris compatibility probes for official uLA and its adapter."""

from __future__ import annotations

import importlib
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from setv.ula.provenance import audit_official_source


OFFICIAL_DISTRIBUTIONS = {
    "torch": "1.13.1+cu117",
    "torchvision": "0.14.1+cu117",
    "tqdm": "4.65.0",
    "scipy": "1.10.1",
    "scikit-learn": "1.2.2",
    "einops": "0.6.1",
    "pytorch-lightning": "1.6.4",
    "torchmetrics": "0.11.4",
    "lightning-bolts": "0.5.0",
    "wandb": "0.15.2",
    "timm": "0.6.13",
    "orion": "0.2.6",
}

MODULES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "tqdm": "tqdm",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "einops": "einops",
    "pytorch-lightning": "pytorch_lightning",
    "torchmetrics": "torchmetrics",
    "lightning-bolts": "pl_bolts",
    "wandb": "wandb",
    "timm": "timm",
    "orion": "orion",
    "pandas": "pandas",
    "Pillow": "PIL",
}

OFFICIAL_FUNCTIONAL_IMPORTS = (
    "torch",
    "torchvision",
    "tqdm",
    "scipy",
    "scikit-learn",
    "einops",
    "pytorch-lightning",
    "torchmetrics",
    "lightning-bolts",
    "timm",
    "pandas",
    "Pillow",
)


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _version_matches(actual: str | None, required: str) -> bool:
    if actual is None:
        return False
    # Local CUDA suffixes are not always retained in package metadata.
    return actual == required or actual == required.split("+", 1)[0]


def probe_ula_environment(
    official_repo: str | Path,
    *,
    mode: str,
    require_gpu: bool,
    expected_architecture: str = "aarch64",
    expected_gpu_substring: str = "GH200",
) -> dict[str, Any]:
    if mode not in {"official_ssl", "external_checkpoint"}:
        raise ValueError(f"Unknown uLA compatibility mode: {mode}")
    official = audit_official_source(official_repo)
    architecture = platform.machine()
    reasons: list[str] = []
    warnings: list[str] = []
    required_distributions = (
        list(OFFICIAL_FUNCTIONAL_IMPORTS)
        if mode == "official_ssl"
        else ["torch", "torchvision"]
    )
    imports = {}
    for distribution in required_distributions:
        module_name = MODULES[distribution]
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            imports[distribution] = {
                "ok": False,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
            reasons.append(f"import_failed:{distribution}")
        else:
            imports[distribution] = {"ok": True}

    versions = {
        name: {
            "installed": _distribution_version(name),
            "official_requirement": required,
            "exact_match": _version_matches(
                _distribution_version(name), required
            ),
        }
        for name, required in OFFICIAL_DISTRIBUTIONS.items()
    }
    exact_requirements_match = all(
        value["exact_match"] for value in versions.values()
    )
    if mode == "official_ssl" and not exact_requirements_match:
        warnings.append(
            "official_dependency_pins_do_not_match; functional smoke is "
            "required and the deviation must remain recorded"
        )
    if architecture != expected_architecture:
        reasons.append(
            f"architecture_mismatch:expected={expected_architecture}:actual={architecture}"
        )

    cuda = {
        "required": bool(require_gpu),
        "available": False,
        "torch_version": None,
        "torchvision_version": None,
        "built_cuda": None,
        "device_name": None,
        "capability": None,
        "cudnn": None,
        "kernel_probe": False,
    }
    if imports.get("torch", {}).get("ok"):
        import torch

        cuda["torch_version"] = str(torch.__version__)
        cuda["built_cuda"] = torch.version.cuda
        cuda["available"] = bool(torch.cuda.is_available())
        if cuda["available"]:
            cuda["device_name"] = str(torch.cuda.get_device_name(0))
            cuda["capability"] = list(torch.cuda.get_device_capability(0))
            cuda["cudnn"] = torch.backends.cudnn.version()
            try:
                layer = torch.nn.Conv2d(3, 4, 3).cuda()
                inputs = torch.randn(2, 3, 16, 16, device="cuda")
                output = layer(inputs)
                output.square().mean().backward()
                torch.cuda.synchronize()
                cuda["kernel_probe"] = True
            except Exception as exc:
                cuda["kernel_exception"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                reasons.append("cuda_kernel_probe_failed")
            if expected_gpu_substring not in str(cuda["device_name"]):
                reasons.append(
                    "gpu_mismatch:"
                    f"expected_substring={expected_gpu_substring}:"
                    f"actual={cuda['device_name']}"
                )
        elif require_gpu:
            reasons.append("cuda_unavailable")
    if imports.get("torchvision", {}).get("ok"):
        import torchvision

        cuda["torchvision_version"] = str(torchvision.__version__)

    official_api = {
        "trainer_add_argparse_args": False,
        "trainer_from_argparse_args": False,
        "ddp_strategy_import": False,
    }
    if mode == "official_ssl" and imports.get(
        "pytorch-lightning", {}
    ).get("ok"):
        try:
            from pytorch_lightning import Trainer
            from pytorch_lightning.strategies.ddp import DDPStrategy  # noqa: F401

            official_api["trainer_add_argparse_args"] = hasattr(
                Trainer, "add_argparse_args"
            )
            official_api["trainer_from_argparse_args"] = hasattr(
                Trainer, "from_argparse_args"
            )
            official_api["ddp_strategy_import"] = True
        except Exception as exc:
            official_api["exception"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        for key, value in official_api.items():
            if key != "exception" and not value:
                reasons.append(f"official_api_missing:{key}")

    accepted = not reasons
    return {
        "schema_version": 1,
        "status": "accepted" if accepted else "incompatible",
        "accepted": accepted,
        "mode": mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "host": {
            "architecture": architecture,
            "expected_architecture": expected_architecture,
            "platform": platform.platform(),
        },
        "official_source": official,
        "imports": imports,
        "versions": versions,
        "official_requirements_exact_match": exact_requirements_match,
        "official_lightning_api": official_api,
        "cuda": cuda,
        "blocking_reasons": reasons,
        "warnings": warnings,
        "interpretation": (
            "Acceptance establishes execution compatibility only. A version "
            "mismatch remains a recorded implementation deviation, and the "
            "SETV integration remains labeled uLA-style."
        ),
    }
