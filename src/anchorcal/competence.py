"""Both-correct competence intersection and robust branch normalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import AuditFailure
from .io import hash_object
from .seeds import stateless_rng


@dataclass(frozen=True)
class CompetenceResult:
    img_ids: np.ndarray
    labels: np.ndarray
    foreground_scale: float
    background_scale: float
    source_hash: str


def true_margin(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    scores = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    return scores[np.arange(len(scores)), truth] - scores[np.arange(len(scores)), 1 - truth]


def construct_competence_intersection(
    img_ids: np.ndarray,
    labels: np.ndarray,
    foreground_logits: np.ndarray,
    background_logits: np.ndarray,
    background_valid: np.ndarray,
    *,
    minimum_per_class: int = 50,
) -> CompetenceResult:
    ids = np.asarray(img_ids, dtype=np.int64)
    truth = np.asarray(labels, dtype=np.int64)
    valid = np.asarray(background_valid, dtype=bool)
    foreground_margin = true_margin(foreground_logits, truth)
    background_margin = true_margin(background_logits, truth)
    member = valid & (foreground_margin > 0) & (background_margin > 0)
    for label in (0, 1):
        count = int(np.sum(member & (truth == label)))
        if count < minimum_per_class:
            raise AuditFailure(
                f"competence intersection class {label} has {count}; requires {minimum_per_class}"
            )
    fg_scale = float(np.median(foreground_margin[member]))
    bg_scale = float(np.median(background_margin[member]))
    if not fg_scale > 0 or not bg_scale > 0:
        raise AuditFailure("competence margin scales must be positive")
    return CompetenceResult(
        ids[member],
        truth[member],
        fg_scale,
        bg_scale,
        hash_object({"img_ids": ids[member].tolist(), "labels": truth[member].tolist()}),
    )


def cap_anchor_subset(
    result: CompetenceResult,
    geometrically_eligible_ids: set[int],
    *,
    per_class: int = 512,
    seed: int = 424242,
    minimum_per_class: int = 50,
) -> np.ndarray:
    eligible = np.asarray(
        [int(value) in geometrically_eligible_ids for value in result.img_ids], dtype=bool
    )
    selected: list[np.ndarray] = []
    for label in (0, 1):
        candidates = result.img_ids[eligible & (result.labels == label)]
        if len(candidates) < minimum_per_class:
            raise AuditFailure(
                f"anchor common subset class {label} has {len(candidates)}; "
                f"requires {minimum_per_class}"
            )
        ordered = np.sort(candidates)
        if len(ordered) > per_class:
            rng = stateless_rng(seed, label, "anchor_subset")
            ordered = np.sort(rng.choice(ordered, size=per_class, replace=False))
        selected.append(ordered)
    return np.sort(np.concatenate(selected))

