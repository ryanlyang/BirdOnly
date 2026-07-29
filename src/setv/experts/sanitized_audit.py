"""Held-out-image leakage auditors for sanitized binary mask banks."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from setv.errors import DataValidationError
from setv.experts.sanitized_masks import FAMILIES, unpack_masks
from setv.utils.seeds import derive_seed, seed_torch_if_available


GEOMETRY_FEATURE_NAMES = (
    "area_fraction",
    "centroid_x",
    "centroid_y",
    "bbox_width",
    "bbox_height",
    "aspect_ratio",
    "perimeter_normalized",
    "compactness",
    "second_moment_xx",
    "second_moment_yy",
    "second_moment_xy",
    "family_rectangle",
    "family_ellipse",
    "family_smooth_blob",
)


def geometry_features(mask: np.ndarray, family_id: int) -> np.ndarray:
    try:
        from scipy.ndimage import binary_erosion
    except ImportError as exc:
        raise DataValidationError("scipy is required for mask geometry auditing") from exc
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not binary.any():
        raise DataValidationError("Auditor masks must be nonempty binary arrays")
    if family_id not in range(len(FAMILIES)):
        raise DataValidationError(f"Invalid sanitized family ID: {family_id}")
    height, width = binary.shape
    yy, xx = np.nonzero(binary)
    center_x = float(xx.mean())
    center_y = float(yy.mean())
    bbox_width = int(xx.max() - xx.min() + 1)
    bbox_height = int(yy.max() - yy.min() + 1)
    boundary = binary & ~binary_erosion(binary)
    perimeter = float(boundary.sum())
    area = float(binary.sum())
    centered_x = (xx - center_x) / max(1.0, width)
    centered_y = (yy - center_y) / max(1.0, height)
    family = np.zeros(3, dtype=np.float64)
    family[family_id] = 1.0
    return np.asarray(
        [
            area / (height * width),
            center_x / max(1, width - 1),
            center_y / max(1, height - 1),
            bbox_width / width,
            bbox_height / height,
            bbox_width / max(1, bbox_height),
            perimeter / (2.0 * (height + width)),
            (4.0 * math.pi * area / (perimeter * perimeter))
            if perimeter > 0
            else 0.0,
            float(np.square(centered_x).mean()),
            float(np.square(centered_y).mean()),
            float((centered_x * centered_y).mean()),
            *family.tolist(),
        ],
        dtype=np.float64,
    )


def extract_geometry_features(
    packed_masks: np.ndarray, family_ids: np.ndarray, width: int
) -> np.ndarray:
    if packed_masks.shape[:2] != family_ids.shape:
        raise DataValidationError("Packed masks and family IDs are misaligned")
    result = np.empty(
        (*family_ids.shape, len(GEOMETRY_FEATURE_NAMES)), dtype=np.float64
    )
    for image_index in range(len(packed_masks)):
        masks = unpack_masks(packed_masks[image_index], width)
        for view_index in range(masks.shape[0]):
            result[image_index, view_index] = geometry_features(
                masks[view_index], int(family_ids[image_index, view_index])
            )
    return result


def heldout_image_split(
    labels: np.ndarray, *, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    train = []
    heldout = []
    for class_id in (0, 1):
        indices = np.flatnonzero(labels == class_id)
        if len(indices) < 2:
            raise DataValidationError(
                "Leakage audit requires at least two images from each class"
            )
        generator = np.random.default_rng(
            derive_seed(seed, f"auditor_image_split:class={class_id}")
        )
        indices = indices.copy()
        generator.shuffle(indices)
        heldout_count = min(
            len(indices) - 1, max(1, int(round(fraction * len(indices))))
        )
        heldout.extend(indices[:heldout_count].tolist())
        train.extend(indices[heldout_count:].tolist())
    return np.asarray(sorted(train), dtype=np.int64), np.asarray(
        sorted(heldout), dtype=np.int64
    )


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    recalls = []
    for class_id in (0, 1):
        selected = labels == class_id
        if not selected.any():
            raise DataValidationError("Balanced accuracy requires both classes")
        recalls.append(float((predictions[selected] == class_id).mean()))
    return float(np.mean(recalls))


def _bootstrap_interval(
    probabilities: np.ndarray,
    image_labels: np.ndarray,
    *,
    seed: int,
    repetitions: int,
    confidence: float,
) -> tuple[float, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(image_labels, dtype=np.int64)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(labels):
        raise DataValidationError("Bootstrap probabilities must have shape [images,views]")
    generator = np.random.default_rng(seed)
    by_class = {class_id: np.flatnonzero(labels == class_id) for class_id in (0, 1)}
    values = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sampled = np.concatenate(
            [
                generator.choice(indices, size=len(indices), replace=True)
                for indices in by_class.values()
            ]
        )
        repeated_labels = np.repeat(labels[sampled], probabilities.shape[1])
        repeated_predictions = (probabilities[sampled].reshape(-1) >= 0.5).astype(
            np.int64
        )
        values[repetition] = _balanced_accuracy(
            repeated_labels, repeated_predictions
        )
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(values, [tail, 1.0 - tail])
    return float(lower), float(upper)


def _auditor_metrics(
    probabilities: np.ndarray,
    image_labels: np.ndarray,
    *,
    seed: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    labels = np.asarray(image_labels, dtype=np.int64)
    per_mask_labels = np.repeat(labels, probabilities.shape[1])
    per_mask_predictions = (probabilities.reshape(-1) >= 0.5).astype(np.int64)
    point = _balanced_accuracy(per_mask_labels, per_mask_predictions)
    image_predictions = (probabilities.mean(axis=1) >= 0.5).astype(np.int64)
    image_point = _balanced_accuracy(labels, image_predictions)
    lower, upper = _bootstrap_interval(
        probabilities,
        labels,
        seed=seed,
        repetitions=int(config["bootstrap_repetitions"]),
        confidence=float(config["bootstrap_confidence"]),
    )
    ceiling = float(config["maximum_balanced_accuracy"])
    contains_chance = lower <= 0.5 <= upper
    return {
        "heldout_mask_balanced_accuracy": point,
        "heldout_image_aggregated_balanced_accuracy": image_point,
        "cluster_bootstrap_confidence_interval": [lower, upper],
        "bootstrap_resampling_unit": "heldout_image",
        "confidence_level": float(config["bootstrap_confidence"]),
        "point_below_or_equal_ceiling": point <= ceiling,
        "confidence_interval_contains_chance": contains_chance,
        "accepted": bool(point <= ceiling and contains_chance),
    }


def _fit_geometry_auditors(
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    heldout_indices: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> dict[str, np.ndarray]:
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.utils.class_weight import compute_sample_weight
    except ImportError as exc:
        raise DataValidationError("scikit-learn is required for mask auditors") from exc
    views = features.shape[1]
    train_x = features[train_indices].reshape(-1, features.shape[-1])
    train_y = np.repeat(labels[train_indices], views)
    heldout_x = features[heldout_indices].reshape(-1, features.shape[-1])
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_x)
    scaled_heldout = scaler.transform(heldout_x)
    logistic_config = config["logistic"]
    logistic = LogisticRegression(
        penalty="l2",
        C=float(logistic_config["C"]),
        class_weight=logistic_config["class_weight"],
        solver=logistic_config["solver"],
        max_iter=int(logistic_config["max_iter"]),
        random_state=derive_seed(seed, "mask_auditor:logistic"),
    )
    logistic.fit(scaled_train, train_y)
    tree_config = config["gradient_boosted_trees"]
    trees = GradientBoostingClassifier(
        n_estimators=int(tree_config["n_estimators"]),
        learning_rate=float(tree_config["learning_rate"]),
        max_depth=int(tree_config["max_depth"]),
        random_state=derive_seed(seed, "mask_auditor:gradient_boosted_trees"),
    )
    trees.fit(
        train_x,
        train_y,
        sample_weight=compute_sample_weight("balanced", train_y),
    )
    return {
        "logistic_geometry": logistic.predict_proba(scaled_heldout)[:, 1].reshape(
            len(heldout_indices), views
        ),
        "gradient_boosted_geometry": trees.predict_proba(heldout_x)[:, 1].reshape(
            len(heldout_indices), views
        ),
    }


def _fit_small_cnn(
    packed_masks: np.ndarray,
    width: int,
    labels: np.ndarray,
    train_indices: np.ndarray,
    heldout_indices: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:
        raise DataValidationError("torch is required for the small-CNN mask auditor") from exc
    requested = config["device"]
    if requested == "cuda" and not torch.cuda.is_available():
        raise DataValidationError("Small-CNN auditor requested CUDA but it is unavailable")
    device = torch.device(requested)
    seed_torch_if_available(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    views = packed_masks.shape[1]

    class PackedDataset(torch.utils.data.Dataset):
        def __init__(self, image_indices):
            self.items = [
                (int(image_index), view)
                for image_index in image_indices
                for view in range(views)
            ]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            image_index, view = self.items[index]
            mask = np.unpackbits(
                packed_masks[image_index, view],
                count=width,
                bitorder="little",
                axis=-1,
            ).astype(np.float32)
            return (
                torch.from_numpy(mask[None, ...]),
                int(labels[image_index]),
                image_index,
                view,
            )

    class MaskCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            pooled = int(config["pooled_resolution"])
            self.network = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d((pooled, pooled)),
                torch.nn.Conv2d(1, 8, 5, stride=2, padding=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 16, 3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(32, 2),
            )

        def forward(self, value):
            return self.network(value)

    generator = torch.Generator()
    generator.manual_seed(derive_seed(seed, "mask_auditor:cnn_loader"))
    train_loader = torch.utils.data.DataLoader(
        PackedDataset(train_indices),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    heldout_loader = torch.utils.data.DataLoader(
        PackedDataset(heldout_indices),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    model = MaskCNN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    counts = np.bincount(labels[train_indices], minlength=2).astype(np.float64)
    class_weights = torch.tensor(
        counts.sum() / np.maximum(counts, 1.0),
        dtype=torch.float32,
        device=device,
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    model.train()
    for _ in range(int(config["epochs"])):
        for masks, targets, _, _ in train_loader:
            masks = masks.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(masks), targets)
            loss.backward()
            optimizer.step()
    probabilities = np.empty((len(labels), views), dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for masks, _, image_indices, view_indices in heldout_loader:
            values = torch.softmax(model(masks.to(device)), dim=1)[:, 1].cpu().numpy()
            probabilities[
                image_indices.numpy().astype(int),
                view_indices.numpy().astype(int),
            ] = values
    return probabilities[heldout_indices]


def run_mask_auditors(
    packed_masks: np.ndarray,
    family_ids: np.ndarray,
    width: int,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    sample_ids = np.asarray(sample_ids).astype(str)
    labels = np.asarray(labels, dtype=np.int64)
    if packed_masks.shape[:2] != family_ids.shape or len(sample_ids) != len(labels):
        raise DataValidationError("Mask-auditor inputs are misaligned")
    if packed_masks.shape[0] != len(labels) or set(np.unique(labels)) != {0, 1}:
        raise DataValidationError("Mask audit requires aligned binary image labels")
    train_indices, heldout_indices = heldout_image_split(
        labels,
        fraction=float(config["heldout_image_fraction"]),
        seed=seed,
    )
    features = extract_geometry_features(packed_masks, family_ids, width)
    probabilities = _fit_geometry_auditors(
        features, labels, train_indices, heldout_indices, config, seed
    )
    probabilities["small_cnn_binary_mask"] = _fit_small_cnn(
        packed_masks,
        width,
        labels,
        train_indices,
        heldout_indices,
        config["small_cnn"],
        derive_seed(seed, "mask_auditor:small_cnn"),
    )
    metrics = {}
    for name, values in probabilities.items():
        metrics[name] = _auditor_metrics(
            values,
            labels[heldout_indices],
            seed=derive_seed(seed, f"mask_auditor_bootstrap:{name}"),
            config=config,
        )
    accepted = all(item["accepted"] for item in metrics.values())
    geometry_by_class = {}
    family_counts_by_class = {}
    for class_id in (0, 1):
        selected_features = features[labels == class_id].reshape(
            -1, features.shape[-1]
        )
        geometry_by_class[str(class_id)] = {
            feature_name: {
                "mean": float(selected_features[:, feature_index].mean()),
                "standard_deviation": float(
                    selected_features[:, feature_index].std()
                ),
            }
            for feature_index, feature_name in enumerate(GEOMETRY_FEATURE_NAMES)
        }
        selected_families = family_ids[labels == class_id]
        family_counts_by_class[str(class_id)] = {
            family: int((selected_families == family_id).sum())
            for family_id, family in enumerate(FAMILIES)
        }
    report = {
        "accepted": accepted,
        "acceptance_rule": (
            "Every auditor must have held-out-mask balanced accuracy <= 0.53 "
            "and an image-cluster-bootstrap confidence interval containing 0.50."
        ),
        "split_unit": "sample_id/image",
        "train_image_count": int(len(train_indices)),
        "heldout_image_count": int(len(heldout_indices)),
        "feature_names": list(GEOMETRY_FEATURE_NAMES),
        "geometry_feature_summary_by_class": geometry_by_class,
        "family_counts_by_class": family_counts_by_class,
        "auditors": metrics,
    }
    arrays = {
        "train_image_index": train_indices,
        "heldout_image_index": heldout_indices,
        "heldout_sample_id": sample_ids[heldout_indices].astype(np.str_),
        "heldout_true_label": labels[heldout_indices],
        "heldout_family_id": family_ids[heldout_indices],
        **{
            f"{name}_probability": values.astype(np.float32)
            for name, values in probabilities.items()
        },
    }
    return report, arrays
