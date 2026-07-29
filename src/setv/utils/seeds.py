"""Deterministic seed derivation and runtime seeding."""

from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np


def derive_seed(base_seed: int, namespace: str) -> int:
    """Derive a stable 32-bit seed without using process-randomized hash()."""
    payload = f"{int(base_seed)}:{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def seed_python_numpy(seed: int) -> dict[str, Any]:
    """Seed dependencies available in Phase 0 and return a receipt."""
    normalized = int(seed) % (2**32)
    random.seed(normalized)
    np.random.seed(normalized)
    return {"python_random_seed": normalized, "numpy_seed": normalized}


def seed_torch_if_available(seed: int) -> dict[str, Any]:
    """Seed torch lazily so Phase 0 remains usable without a local torch build."""
    try:
        import torch
    except (ImportError, ModuleNotFoundError):
        return {"torch_available": False}
    normalized = int(seed) % (2**32)
    torch.manual_seed(normalized)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(normalized)
    return {
        "torch_available": True,
        "torch_seed": normalized,
        "cuda_available": bool(torch.cuda.is_available()),
    }
