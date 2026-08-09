"""Stable, stateless scientific randomness.

Python's process-randomized ``hash`` is never used.  All child seeds are the
first eight bytes of a SHA-256 digest, reduced to the target RNG's range.
"""

from __future__ import annotations

import hashlib
import os
import random
from collections.abc import Iterable

import numpy as np


UINT32_MODULUS = 2**32
UINT64_MODULUS = 2**64


def stable_seed(*parts: object, modulus: int = UINT32_MODULUS) -> int:
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % modulus


def background_view_seed(
    global_seed: int, sample_id: int | str, view_index: int, purpose: str
) -> int:
    """Return the locked, unreduced first eight SHA-256 bytes.

    Background-view seeds are scientific provenance stored as unsigned 64-bit
    values.  They deliberately do *not* inherit :func:`stable_seed`'s uint32
    default, which remains appropriate for APIs such as sklearn and PyTorch
    that require or conventionally consume narrower seeds.
    """

    return stable_seed(
        global_seed,
        sample_id,
        view_index,
        purpose,
        modulus=UINT64_MODULUS,
    )


def stateless_rng(*parts: object) -> np.random.Generator:
    return np.random.default_rng(stable_seed(*parts, modulus=2**63 - 1))


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed host and torch RNGs without importing torch at package import time."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(int(seed)))
    random.seed(seed)
    np.random.seed(seed % UINT32_MODULUS)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def seeded_worker_init(run_seed: int):
    """Return a DataLoader worker initializer derived only from run seed/id."""

    def initialize(worker_id: int) -> None:
        child = stable_seed(run_seed, worker_id, "dataloader_worker")
        random.seed(child)
        np.random.seed(child)
        try:
            import torch

            torch.manual_seed(child)
        except ImportError:
            pass

    return initialize


def require_named_purpose(purpose: str, allowed: Iterable[str]) -> None:
    if purpose not in set(allowed):
        raise ValueError(f"unnamed stochastic purpose {purpose!r}")
