"""Locked mixed-precision contexts for model evaluation.

AnchorCal evaluates ordinary CUDA/GH200 model forwards under bfloat16
autocast. Saliency is the prespecified FP32 fallback: it retains gradients but
disables autocast so the direct and algebraically cached anchor paths can meet
their locked parity tolerances.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


def evaluation_autocast(device):
    """Return the binding evaluation autocast context for ``device``."""

    import torch

    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    )


def floating_tensor_to_numpy(value):
    """Export an autocast floating tensor through NumPy-compatible fp32."""

    return value.detach().float().cpu().numpy()


@contextmanager
def evaluation_inference(device) -> Iterator[None]:
    """Run a non-saliency evaluation forward without autograd state."""

    import torch

    with torch.inference_mode(), evaluation_autocast(device):
        yield


@contextmanager
def saliency_evaluation(device) -> Iterator[None]:
    """Run gradient-based saliency in the locked FP32 fallback mode."""

    import torch

    with torch.enable_grad(), torch.autocast(
        device_type=device.type,
        enabled=False,
    ):
        yield
