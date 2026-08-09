"""Hard leakage, competence, random-token, and branch bootstrap gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .errors import AuditFailure
from .metrics import class_balanced_mean
from .seeds import stateless_rng


@dataclass(frozen=True)
class BootstrapInterval:
    point: float
    lower: float
    upper: float
    replicates: int


def stratified_bootstrap_interval(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    replicates: int,
    seed: int,
    statistic: Callable[[np.ndarray, np.ndarray], float] = class_balanced_mean,
) -> BootstrapInterval:
    value_array = np.asarray(values)
    label_array = np.asarray(labels, dtype=np.int64)
    point = float(statistic(value_array, label_array))
    rng = stateless_rng(seed, "class_stratified_bootstrap")
    estimates = np.empty(replicates, dtype=np.float64)
    class_indices = [np.flatnonzero(label_array == label) for label in (0, 1)]
    if any(len(indices) == 0 for indices in class_indices):
        raise ValueError("both classes are required for stratified bootstrap")
    for replicate in range(replicates):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        estimates[replicate] = statistic(value_array[sampled], label_array[sampled])
    return BootstrapInterval(
        point,
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
        replicates,
    )


def require_branch_above_chance(
    correct: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 7001,
    replicates: int = 2000,
) -> BootstrapInterval:
    interval = stratified_bootstrap_interval(
        np.asarray(correct, dtype=np.float64),
        labels,
        replicates=replicates,
        seed=seed,
    )
    if interval.point <= 0.5 or interval.lower <= 0.5:
        raise AuditFailure(
            f"branch competence gate failed: balanced accuracy={interval.point:.4f}, "
            f"95% CI=[{interval.lower:.4f},{interval.upper:.4f}]"
        )
    return interval


def require_random_token_collapse(
    correct: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 8003,
    replicates: int = 2000,
) -> BootstrapInterval:
    interval = stratified_bootstrap_interval(
        np.asarray(correct, dtype=np.float64),
        labels,
        replicates=replicates,
        seed=seed,
    )
    if interval.point > 0.53 or not interval.lower <= 0.5 <= interval.upper:
        raise AuditFailure(
            f"random-token leakage gate failed: balanced accuracy={interval.point:.4f}, "
            f"95% CI=[{interval.lower:.4f},{interval.upper:.4f}]"
        )
    return interval


def require_foreground_invariance(
    first_logits: np.ndarray,
    second_logits: np.ndarray,
    *,
    first_tokens: np.ndarray | None = None,
    second_tokens: np.ndarray | None = None,
    first_source: np.ndarray | None = None,
    second_source: np.ndarray | None = None,
    tolerance: float = 1e-6,
) -> float:
    if first_tokens is not None and not np.array_equal(first_tokens, second_tokens):
        raise AuditFailure("foreground replacement changed token pixels")
    if first_source is not None and not np.array_equal(first_source, second_source):
        raise AuditFailure("foreground replacement changed token metadata")
    difference = float(np.max(np.abs(np.asarray(first_logits) - np.asarray(second_logits))))
    if not np.isfinite(difference) or difference > tolerance:
        raise AuditFailure(
            f"foreground background-replacement invariance failed: {difference} > {tolerance}"
        )
    return difference


def assert_background_patch_purity(
    source_indices: np.ndarray, dilated_fractions: np.ndarray
) -> None:
    indices = np.asarray(source_indices, dtype=np.int64)
    fractions = np.asarray(dilated_fractions)
    if np.any(indices < 0) or np.any(indices >= len(fractions)):
        raise AuditFailure("background source index is invalid")
    impure = indices[fractions[indices] != 0.0]
    if len(impure):
        raise AuditFailure(f"background branch received {len(impure)} impure tokens")


def build_image_disjoint_random_patch_pool(
    patch_values: np.ndarray,
    source_img_ids: np.ndarray,
    source_labels: np.ndarray,
    recipient_img_ids: np.ndarray,
    *,
    per_class: int,
    seed: int = 8003,
) -> np.ndarray:
    """Select equal real-patch counts by source class, disjoint from recipients."""

    patches = np.asarray(patch_values)
    ids = np.asarray(source_img_ids)
    labels = np.asarray(source_labels)
    allowed = ~np.isin(ids, np.asarray(recipient_img_ids))
    selected: list[np.ndarray] = []
    for label in (0, 1):
        choices = np.flatnonzero(allowed & (labels == label))
        if len(choices) < per_class:
            raise AuditFailure(
                f"random-token source class {label} has {len(choices)} patches; needs {per_class}"
            )
        rng = stateless_rng(seed, label, "random_token_audit_pool")
        selected.append(rng.choice(choices, size=per_class, replace=False))
    return patches[np.concatenate(selected)]


def fit_geometry_auditors(
    features: np.ndarray,
    labels: np.ndarray,
    img_ids: np.ndarray,
    *,
    split_seed: int = 8001,
    model_seed: int = 8002,
) -> dict[str, np.ndarray]:
    """Prespecified image-disjoint logistic and 64/32 MLP auditors."""

    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    indices = np.arange(len(labels))
    train, test = train_test_split(
        indices,
        test_size=0.25,
        random_state=split_seed,
        stratify=np.asarray(labels),
    )
    if set(np.asarray(img_ids)[train]) & set(np.asarray(img_ids)[test]):
        raise AssertionError("geometry auditor split is not image-disjoint")
    logistic = make_pipeline(
        StandardScaler(), LogisticRegression(random_state=model_seed, max_iter=1000)
    )
    logistic.fit(features[train], np.asarray(labels)[train])
    # The locked MLP uses GELU, which sklearn's MLP does not expose. Train the
    # small prespecified network directly in torch on standardized features.
    scaler = StandardScaler().fit(features[train])
    train_features = scaler.transform(features[train]).astype(np.float32)
    test_features = scaler.transform(features[test]).astype(np.float32)
    import torch
    from torch import nn

    torch.manual_seed(model_seed)
    mlp = nn.Sequential(
        nn.Linear(features.shape[1], 64),
        nn.GELU(),
        nn.Linear(64, 32),
        nn.GELU(),
        nn.Linear(32, 2),
    )
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-2)
    x_train = torch.from_numpy(train_features)
    y_train = torch.from_numpy(np.asarray(labels)[train].astype(np.int64))
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(mlp(x_train), y_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        mlp_predictions = mlp(torch.from_numpy(test_features)).argmax(dim=1).numpy()
    return {
        "test_indices": test,
        "logistic_predictions": logistic.predict(features[test]),
        "mlp_predictions": mlp_predictions,
        "labels": np.asarray(labels)[test],
    }
