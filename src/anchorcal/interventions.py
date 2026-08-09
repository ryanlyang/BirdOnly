"""Typed, deterministic, stream-restricted interventions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np

from .errors import AuditFailure
from .seeds import stateless_rng


class InterventionType(str, Enum):
    NONE = "none"
    TOKEN_SWAP_BACKGROUND = "token_swap_background"
    BLUR_BACKGROUND = "blur_background"
    FOREGROUND_ONLY_GREENSCREEN = "foreground_only_greenscreen"


@dataclass(frozen=True)
class DonorAssignment:
    recipient_id: int
    donor_ids: tuple[int, ...]


@dataclass(frozen=True)
class PatchAssignment:
    recipient_position: int
    donor_source_index: int
    fallback: str


def coarse_bin(index: int, grid_size: int = 14) -> tuple[int, int]:
    if not 0 <= index < grid_size * grid_size:
        raise ValueError("patch index outside grid")
    row, column = divmod(index, grid_size)

    def bin_one(value: int) -> int:
        return min(2, int(np.floor(3.0 * (value + 0.5) / grid_size)))

    return bin_one(row), bin_one(column)


def assign_donors(
    img_ids: np.ndarray,
    labels: np.ndarray,
    eligible: np.ndarray,
    *,
    donor_eligible: np.ndarray | None = None,
    donors_per_recipient: int = 4,
    seed: int = 31415,
) -> list[DonorAssignment]:
    ids = np.asarray(img_ids, dtype=np.int64)
    truth = np.asarray(labels, dtype=np.int64)
    valid = np.asarray(eligible, dtype=bool)
    donor_valid = (
        valid.copy()
        if donor_eligible is None
        else np.asarray(donor_eligible, dtype=bool)
    )
    if ids.shape != truth.shape or ids.shape != valid.shape or ids.shape != donor_valid.shape:
        raise ValueError("donor inputs must have matching one-dimensional shapes")
    assignments: list[DonorAssignment] = []
    for recipient, label in zip(ids[valid], truth[valid], strict=True):
        choices = ids[donor_valid & (truth != label) & (ids != recipient)]
        if len(choices) < donors_per_recipient:
            raise AuditFailure(
                f"recipient {recipient} has only {len(choices)} distinct opposite-class donors"
            )
        rng = stateless_rng(seed, int(recipient), "donor_assignment")
        donors = tuple(
            int(value)
            for value in rng.choice(np.sort(choices), donors_per_recipient, replace=False)
        )
        assignments.append(DonorAssignment(int(recipient), donors))
    return assignments


def assign_candidate_donor_patches(
    recipient_id: int,
    donor_id: int,
    recipient_background: np.ndarray,
    donor_background: np.ndarray,
    *,
    seed: int = 31415,
) -> list[PatchAssignment]:
    recipient = np.asarray(recipient_background, dtype=np.int64)
    donor = np.asarray(donor_background, dtype=np.int64)
    if len(donor) == 0:
        raise AuditFailure(f"donor {donor_id} has no pure background patches")
    bins: dict[tuple[int, int], np.ndarray] = {
        spatial: donor[np.asarray([coarse_bin(int(value)) == spatial for value in donor])]
        for spatial in {(row, column) for row in range(3) for column in range(3)}
    }
    rng = stateless_rng(seed, recipient_id, donor_id, "token_swap_donor_patch")
    used: dict[tuple[int, int], set[int]] = {key: set() for key in bins}
    assignments: list[PatchAssignment] = []
    for recipient_position in recipient.tolist():
        spatial = coarse_bin(int(recipient_position))
        matching = bins[spatial]
        unique = np.asarray([value for value in matching if int(value) not in used[spatial]])
        if len(unique):
            selected = int(rng.choice(unique))
            fallback = "none"
            used[spatial].add(selected)
        elif len(matching):
            selected = int(rng.choice(matching))
            fallback = "matching_bin_with_replacement"
        else:
            selected = int(rng.choice(donor))
            fallback = "all_background_bin_empty"
        assignments.append(PatchAssignment(int(recipient_position), selected, fallback))
    return assignments


def apply_candidate_token_swap(
    recipient_tokens,
    donor_tokens,
    assignments: list[PatchAssignment],
):
    """Swap content embeddings before recipient absolute positions are added."""

    output = recipient_tokens.clone()
    for assignment in assignments:
        output[:, assignment.recipient_position] = donor_tokens[
            :, assignment.donor_source_index
        ]
    return output


def _require_exact_foreground_component(
    name: str, clean: np.ndarray, intervened: np.ndarray
) -> float:
    """Require bitwise identity for discrete/input stream state."""

    clean_array = np.asarray(clean)
    intervened_array = np.asarray(intervened)
    if clean_array.shape != intervened_array.shape:
        raise AuditFailure(
            f"background-only intervention changed foreground {name} shape: "
            f"{clean_array.shape} != {intervened_array.shape}"
        )
    if not np.array_equal(clean_array, intervened_array):
        raise AuditFailure(
            f"background-only intervention changed foreground {name}"
        )
    return 0.0


def _require_close_foreground_component(
    name: str,
    clean: np.ndarray,
    intervened: np.ndarray,
    *,
    tolerance: float,
) -> float:
    """Require finite, shape-identical numerical output within tolerance."""

    clean_array = np.asarray(clean)
    intervened_array = np.asarray(intervened)
    if clean_array.shape != intervened_array.shape:
        raise AuditFailure(
            f"background-only intervention changed foreground {name} shape: "
            f"{clean_array.shape} != {intervened_array.shape}"
        )
    if not np.isfinite(clean_array).all() or not np.isfinite(intervened_array).all():
        raise AuditFailure(
            f"background-only intervention produced non-finite foreground {name}"
        )
    difference = float(np.max(np.abs(clean_array - intervened_array), initial=0.0))
    if difference > tolerance:
        raise AuditFailure(
            f"background-only intervention changed foreground {name}: "
            f"{difference} > {tolerance}"
        )
    return difference


def assert_foreground_stream_unchanged(
    clean_logits: np.ndarray,
    intervened_logits: np.ndarray,
    tolerance: float = 1e-6,
    *,
    clean_input_image: np.ndarray | None = None,
    intervened_input_image: np.ndarray | None = None,
    clean_input_mask: np.ndarray | None = None,
    intervened_input_mask: np.ndarray | None = None,
    clean_patch_activations: np.ndarray | None = None,
    intervened_patch_activations: np.ndarray | None = None,
    clean_patch_valid: np.ndarray | None = None,
    intervened_patch_valid: np.ndarray | None = None,
    clean_source_indices: np.ndarray | None = None,
    intervened_source_indices: np.ndarray | None = None,
) -> dict[str, float]:
    """Assert that a background intervention did not alter foreground state.

    The logits-only form remains useful for small unit checks.  Production
    AnchorCal intervention contracts pass every explicit input, activation,
    and token-metadata pair so an accidental foreground mutation cannot be
    hidden by comparing two aliases of the same logits array.
    """

    paired = {
        "input_image": (clean_input_image, intervened_input_image, True),
        "input_mask": (clean_input_mask, intervened_input_mask, True),
        "patch_activations": (
            clean_patch_activations,
            intervened_patch_activations,
            False,
        ),
        "patch_valid": (clean_patch_valid, intervened_patch_valid, True),
        "source_indices": (clean_source_indices, intervened_source_indices, True),
    }
    diagnostics: dict[str, float] = {}
    for name, (clean, intervened, exact) in paired.items():
        if (clean is None) != (intervened is None):
            raise AuditFailure(
                f"foreground {name} comparison is missing one side"
            )
        if clean is None:
            continue
        diagnostics[f"{name}_max_abs_difference"] = (
            _require_exact_foreground_component(name, clean, intervened)
            if exact
            else _require_close_foreground_component(
                name, clean, intervened, tolerance=tolerance
            )
        )
    diagnostics["logits_max_abs_difference"] = _require_close_foreground_component(
        "logits", clean_logits, intervened_logits, tolerance=tolerance
    )
    return diagnostics


def assert_anchor_intervention_contract(
    intervention: InterventionType,
    clean_foreground_logits: np.ndarray,
    intervened_foreground_logits: np.ndarray,
    *,
    clean_input_image: np.ndarray,
    intervened_input_image: np.ndarray,
    clean_input_mask: np.ndarray,
    intervened_input_mask: np.ndarray,
    clean_patch_activations: np.ndarray,
    intervened_patch_activations: np.ndarray,
    clean_patch_valid: np.ndarray,
    intervened_patch_valid: np.ndarray,
    clean_source_indices: np.ndarray,
    intervened_source_indices: np.ndarray,
) -> dict[str, float]:
    if intervention not in set(InterventionType):
        raise AuditFailure(f"unknown anchor intervention: {intervention!r}")
    if intervention in {
        InterventionType.TOKEN_SWAP_BACKGROUND,
        InterventionType.BLUR_BACKGROUND,
        InterventionType.FOREGROUND_ONLY_GREENSCREEN,
    }:
        return assert_foreground_stream_unchanged(
            clean_foreground_logits,
            intervened_foreground_logits,
            clean_input_image=clean_input_image,
            intervened_input_image=intervened_input_image,
            clean_input_mask=clean_input_mask,
            intervened_input_mask=intervened_input_mask,
            clean_patch_activations=clean_patch_activations,
            intervened_patch_activations=intervened_patch_activations,
            clean_patch_valid=clean_patch_valid,
            intervened_patch_valid=intervened_patch_valid,
            clean_source_indices=clean_source_indices,
            intervened_source_indices=intervened_source_indices,
        )
    return {}


def require_candidate_intervention(intervention: InterventionType) -> None:
    if intervention not in {
        InterventionType.TOKEN_SWAP_BACKGROUND,
        InterventionType.BLUR_BACKGROUND,
        InterventionType.FOREGROUND_ONLY_GREENSCREEN,
    }:
        raise AuditFailure(
            f"unsupported candidate intervention request: {intervention.value}"
        )
