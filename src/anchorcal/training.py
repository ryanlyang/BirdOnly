"""Locked AdamW grouping, update scheduler, AMP, and branch training helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .io import atomic_write_json


def parameter_groups(model, weight_decay: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    decay = []
    no_decay = []
    decay_names: list[str] = []
    no_decay_names: list[str] = []
    seen: set[int] = set()
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if not parameter.requires_grad:
            continue
        identifier = id(parameter)
        if identifier in seen:
            raise ValueError(f"trainable parameter appears more than once: {name}")
        seen.add(identifier)
        lower = name.lower()
        excluded = (
            name.endswith(".bias")
            or parameter.ndim == 1
            or "norm" in lower
            or "cls_token" in lower
            or "pos_embed" in lower
        )
        if excluded:
            no_decay.append(parameter)
            no_decay_names.append(name)
        else:
            decay.append(parameter)
            decay_names.append(name)
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if seen != expected:
        raise ValueError("not every trainable parameter was assigned exactly once")
    groups = [
        {"params": decay, "weight_decay": float(weight_decay), "group_name": "decay"},
        {"params": no_decay, "weight_decay": 0.0, "group_name": "no_decay"},
    ]
    manifest = {
        "decay_names": decay_names,
        "no_decay_names": no_decay_names,
        "decay_parameter_count": int(sum(parameter.numel() for parameter in decay)),
        "no_decay_parameter_count": int(sum(parameter.numel() for parameter in no_decay)),
    }
    return groups, manifest


def make_adamw(model, *, learning_rate: float, weight_decay: float):
    import torch

    groups, manifest = parameter_groups(model, weight_decay)
    optimizer = torch.optim.AdamW(
        groups,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
    )
    return optimizer, manifest


def update_factor(
    update_index: int,
    total_updates: int,
    warmup_updates: int,
    *,
    start_factor: float = 0.01,
    minimum_factor: float = 0.01,
) -> float:
    if total_updates < 1 or not 0 <= update_index < total_updates:
        raise ValueError("update index outside schedule")
    if not 0 <= warmup_updates <= total_updates:
        raise ValueError("invalid warmup update count")
    if warmup_updates > 0 and update_index < warmup_updates:
        if warmup_updates == 1:
            return start_factor
        progress = update_index / (warmup_updates - 1)
        return start_factor + progress * (1.0 - start_factor)
    decay_updates = total_updates - warmup_updates
    if decay_updates <= 1:
        return minimum_factor
    progress = (update_index - warmup_updates) / (decay_updates - 1)
    return minimum_factor + 0.5 * (1.0 - minimum_factor) * (
        1.0 + math.cos(math.pi * progress)
    )


class UpdateScheduler:
    """Scheduler whose current LR is the LR used by the next optimizer update."""

    def __init__(
        self,
        optimizer,
        *,
        base_lr: float,
        total_updates: int,
        warmup_updates: int,
    ) -> None:
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.total_updates = int(total_updates)
        self.warmup_updates = int(warmup_updates)
        self.completed_updates = 0
        self._set_for_index(0)

    def _set_for_index(self, index: int) -> None:
        factor = update_factor(index, self.total_updates, self.warmup_updates)
        for group in self.optimizer.param_groups:
            group["lr"] = self.base_lr * factor

    def step_after_optimizer(self) -> None:
        self.completed_updates += 1
        if self.completed_updates < self.total_updates:
            self._set_for_index(self.completed_updates)

    def state_dict(self) -> dict[str, int | float]:
        return {
            "base_lr": self.base_lr,
            "total_updates": self.total_updates,
            "warmup_updates": self.warmup_updates,
            "completed_updates": self.completed_updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key in ("base_lr", "total_updates", "warmup_updates"):
            if state[key] != getattr(self, key):
                raise ValueError(f"scheduler resume mismatch for {key}")
        self.completed_updates = int(state["completed_updates"])
        if self.completed_updates < self.total_updates:
            self._set_for_index(self.completed_updates)


@dataclass
class EpochResult:
    epoch: int
    loss: float
    accuracy: float
    fallback_count: int
    sample_count: int


def train_epoch(
    model,
    loader,
    optimizer,
    scheduler: UpdateScheduler,
    step_fn: Callable[[Any, Any], tuple[Any, Any, int]],
    *,
    device,
    gradient_clip_norm: float = 1.0,
    use_bfloat16: bool = True,
) -> EpochResult:
    import torch
    import torch.nn.functional as functional

    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    fallback_count = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bfloat16 and device.type == "cuda",
        ):
            logits, labels, fallbacks = step_fn(model, batch)
            loss = functional.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        scheduler.step_after_optimizer()
        count = int(labels.numel())
        total += count
        total_loss += float(loss.detach().item()) * count
        total_correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
        fallback_count += int(fallbacks)
    if total == 0:
        raise RuntimeError("empty training loader")
    return EpochResult(0, total_loss / total, total_correct / total, fallback_count, total)


def persist_optimizer_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    atomic_write_json(path, manifest)

