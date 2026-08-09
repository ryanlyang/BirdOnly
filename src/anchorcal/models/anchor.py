"""Differentiable raw-logit reliance anchor ladder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .branches import BackgroundBranch, BranchOutput, ForegroundBranch


MARGIN_SCALE_EPSILON = 1.0e-8


def centered_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits - logits.mean(dim=-1, keepdim=True)


def true_class_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] != 2:
        raise ValueError("the pilot requires binary logits")
    true = logits.gather(1, labels[:, None]).squeeze(1)
    other = logits.gather(1, (1 - labels)[:, None]).squeeze(1)
    return true - other


def robust_margin_scale(logits: torch.Tensor, labels: torch.Tensor) -> float:
    margins = true_class_margin(logits, labels)
    if torch.any(margins <= 0):
        raise ValueError("margin scale requires the both-correct competence intersection")
    scale = float(torch.median(margins).item())
    if not scale > 0:
        raise ValueError("branch margin scale is not positive")
    return scale


def mix_anchor_logits(
    foreground_logits: torch.Tensor,
    background_logits: torch.Tensor,
    reliance_lambda: float | torch.Tensor,
    foreground_scale: float,
    background_scale: float,
) -> torch.Tensor:
    lam = torch.as_tensor(
        reliance_lambda, dtype=foreground_logits.dtype, device=foreground_logits.device
    )
    if torch.any(lam < 0) or torch.any(lam > 1):
        raise ValueError("reliance lambda must be in [0,1]")
    foreground = centered_logits(foreground_logits) / (
        foreground_scale + MARGIN_SCALE_EPSILON
    )
    background = centered_logits(background_logits) / (
        background_scale + MARGIN_SCALE_EPSILON
    )
    return lam * foreground + (1.0 - lam) * background


@dataclass
class AnchorOutput:
    logits: torch.Tensor
    foreground_logits: torch.Tensor
    background_logits: torch.Tensor
    foreground: BranchOutput
    background_views: list[BranchOutput]


class RelianceAnchor(nn.Module):
    """Frozen two-stream model with activation gradients preserved."""

    def __init__(
        self,
        foreground: ForegroundBranch,
        background: BackgroundBranch,
        *,
        foreground_scale: float,
        background_scale: float,
        reliance_lambda: float,
    ) -> None:
        super().__init__()
        if foreground_scale <= 0 or background_scale <= 0:
            raise ValueError("anchor scales must be positive")
        if not 0 <= reliance_lambda <= 1:
            raise ValueError("reliance lambda must be in [0,1]")
        self.foreground = foreground
        self.background = background
        self.foreground_scale = float(foreground_scale)
        self.background_scale = float(background_scale)
        self.reliance_lambda = float(reliance_lambda)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def _background_views(
        self,
        background_images: torch.Tensor,
        view_indices: torch.Tensor,
        *,
        activation_leaf: bool,
    ) -> tuple[torch.Tensor, list[BranchOutput]]:
        if view_indices.ndim != 3:
            raise ValueError("view_indices must be [batch,views,K]")
        projected = self.background.project(background_images)
        outputs = [
            self.background.forward_from_projected(
                projected, view_indices[:, view], activation_leaf=activation_leaf
            )
            for view in range(view_indices.shape[1])
        ]
        averaged = torch.stack([output.logits for output in outputs], dim=0).mean(dim=0)
        return averaged, outputs

    def forward(
        self,
        foreground_images: torch.Tensor,
        masks: torch.Tensor,
        background_images: torch.Tensor,
        view_indices: torch.Tensor,
        *,
        activation_leaves: bool = False,
    ) -> AnchorOutput:
        foreground = self.foreground(
            foreground_images, masks, activation_leaf=activation_leaves
        )
        background_logits, background_outputs = self._background_views(
            background_images, view_indices, activation_leaf=activation_leaves
        )
        logits = mix_anchor_logits(
            foreground.logits,
            background_logits,
            self.reliance_lambda,
            self.foreground_scale,
            self.background_scale,
        )
        return AnchorOutput(
            logits,
            foreground.logits,
            background_logits,
            foreground,
            background_outputs,
        )


def assert_anchor_correct(logits: torch.Tensor, labels: torch.Tensor) -> None:
    failures = torch.nonzero(logits.argmax(dim=1) != labels, as_tuple=False).flatten()
    if failures.numel():
        raise AssertionError(
            f"anchor is not 100% correct on competence subset ({failures.numel()} failures)"
        )
