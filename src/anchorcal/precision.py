"""Locked mixed-precision contexts for model evaluation.

AnchorCal evaluates every CUDA/GH200 model forward under bfloat16 autocast.
Keeping the policy here prevents individual evaluation paths from silently
drifting to fp32.  Saliency needs gradients, but uses the same autocast policy
as all other evaluations.
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


@contextmanager
def evaluation_inference(device) -> Iterator[None]:
    """Run a non-saliency evaluation forward without autograd state."""

    import torch

    with torch.inference_mode(), evaluation_autocast(device):
        yield


@contextmanager
def saliency_evaluation(device) -> Iterator[None]:
    """Run a gradient-based saliency forward under the same bf16 policy."""

    import torch

    with torch.enable_grad(), evaluation_autocast(device):
        yield
