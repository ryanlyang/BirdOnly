"""Leakage-controlled foreground and background ViT-style branches."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum

import torch
from torch import nn

from .token_vit import VariableTokenEncoder, pad_token_sequences


def _flatten_projection(projected: torch.Tensor) -> torch.Tensor:
    return projected.flatten(2).transpose(1, 2)


class ObjectRelativePosition(nn.Module):
    def __init__(self, embed_dim: int = 384) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2, 128), nn.GELU(), nn.Linear(128, embed_dim))
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.network(coordinates.clamp(0.0, 1.0))


class ForegroundPositionMode(str, Enum):
    """Named foreground position families; only the locked primary is active."""

    OBJECT_RELATIVE = "object_relative"


def build_foreground_position_encoder(
    mode: str | ForegroundPositionMode, embed_dim: int
) -> nn.Module:
    """Construct a named encoder without silently enabling deferred variants."""

    try:
        resolved = ForegroundPositionMode(mode)
    except ValueError as error:
        raise ValueError(
            f"unsupported/deferred foreground position mode: {mode!r}"
        ) from error
    if resolved is ForegroundPositionMode.OBJECT_RELATIVE:
        return ObjectRelativePosition(embed_dim)
    raise AssertionError(f"unhandled foreground position mode: {resolved}")


def object_relative_coordinates(mask: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """Coordinates for every grid cell relative to the visible bird bbox."""

    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("mask must be a two-dimensional bool tensor")
    points = mask.nonzero(as_tuple=False)
    if points.numel() == 0:
        raise ValueError("foreground mask is empty")
    ymin, xmin = points.min(dim=0).values.float()
    ymax, xmax = points.max(dim=0).values.float()
    grid_h = mask.shape[0] // patch_size
    grid_w = mask.shape[1] // patch_size
    yy, xx = torch.meshgrid(
        torch.arange(grid_h, device=mask.device),
        torch.arange(grid_w, device=mask.device),
        indexing="ij",
    )
    centers_x = (xx.float() + 0.5) * patch_size
    centers_y = (yy.float() + 0.5) * patch_size
    u = (centers_x - xmin) / torch.clamp(xmax - xmin, min=1.0)
    v = (centers_y - ymin) / torch.clamp(ymax - ymin, min=1.0)
    return torch.stack([u.clamp(0, 1), v.clamp(0, 1)], dim=-1).reshape(-1, 2)


@dataclass
class BranchOutput:
    logits: torch.Tensor
    patch_activations: torch.Tensor
    patch_valid: torch.Tensor
    source_indices: torch.Tensor


class ForegroundBranch(nn.Module):
    """Variable-token branch over source-resolution-green-screened images."""

    def __init__(
        self,
        source_vit: nn.Module,
        depth: int = 6,
        *,
        position_mode: str | ForegroundPositionMode = ForegroundPositionMode.OBJECT_RELATIVE,
    ) -> None:
        super().__init__()
        self.patch_projection = copy.deepcopy(source_vit.patch_embed.proj)
        self.encoder = VariableTokenEncoder(source_vit, depth=depth, num_classes=2)
        try:
            self.position_mode = ForegroundPositionMode(position_mode)
        except ValueError as error:
            raise ValueError(
                f"unsupported/deferred foreground position mode: {position_mode!r}"
            ) from error
        self.position = build_foreground_position_encoder(
            self.position_mode, int(source_vit.embed_dim)
        )
        self.patch_size = 16

    def project(self, images: torch.Tensor) -> torch.Tensor:
        return _flatten_projection(self.patch_projection(images))

    def prepare_tokens(
        self, projected: torch.Tensor, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sequences: list[torch.Tensor] = []
        positions_list: list[torch.Tensor] = []
        source: list[torch.Tensor] = []
        for image_tokens, mask in zip(projected, masks, strict=True):
            fractions = mask.float().unfold(0, 16, 16).unfold(1, 16, 16).mean(dim=(-1, -2)).reshape(-1)
            indices = torch.nonzero(fractions > 0.0, as_tuple=False).flatten()
            if indices.numel() == 0:
                raise ValueError("foreground branch example has no retained token")
            positions = object_relative_coordinates(mask.bool(), self.patch_size)
            sequences.append(image_tokens[indices])
            positions_list.append(self.position(positions[indices]))
            source.append(indices)
        tokens, valid, source_pad = pad_token_sequences(sequences, source)
        position_pad, position_valid, _ = pad_token_sequences(positions_list)
        if not torch.equal(valid, position_valid):
            raise AssertionError("foreground token/position padding mismatch")
        assert source_pad is not None
        return tokens, position_pad, valid, source_pad

    def forward_from_projected(
        self,
        projected: torch.Tensor,
        masks: torch.Tensor,
        *,
        activation_leaf: bool = False,
    ) -> BranchOutput:
        tokens, positions, valid, source = self.prepare_tokens(projected, masks)
        if activation_leaf:
            tokens = tokens.detach().requires_grad_(True)
        logits, _ = self.encoder.forward_tokens(tokens + positions, valid)
        return BranchOutput(logits, tokens, valid, source)

    def forward(
        self, images: torch.Tensor, masks: torch.Tensor, *, activation_leaf: bool = False
    ) -> BranchOutput:
        return self.forward_from_projected(
            self.project(images), masks, activation_leaf=activation_leaf
        )


class BackgroundBranch(nn.Module):
    """Position-free fixed-cardinality set encoder over pure background patches."""

    def __init__(self, source_vit: nn.Module, token_budget: int, depth: int = 6) -> None:
        super().__init__()
        if token_budget <= 0:
            raise ValueError("token budget must be positive")
        self.patch_projection = copy.deepcopy(source_vit.patch_embed.proj)
        self.encoder = VariableTokenEncoder(source_vit, depth=depth, num_classes=2)
        self.token_budget = int(token_budget)

    def project(self, images: torch.Tensor) -> torch.Tensor:
        return _flatten_projection(self.patch_projection(images))

    def forward_from_projected(
        self,
        projected: torch.Tensor,
        source_indices: torch.Tensor,
        *,
        activation_leaf: bool = False,
    ) -> BranchOutput:
        if source_indices.ndim != 2 or source_indices.shape[1] != self.token_budget:
            raise ValueError("source_indices must be [batch, fixed token budget]")
        if torch.any(source_indices < 0) or torch.any(source_indices >= projected.shape[1]):
            raise ValueError("background source index is out of range")
        gather = source_indices.unsqueeze(-1).expand(-1, -1, projected.shape[-1])
        tokens = projected.gather(1, gather)
        if activation_leaf:
            tokens = tokens.detach().requires_grad_(True)
        valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        logits, _ = self.encoder.forward_tokens(tokens, valid)
        return BranchOutput(logits, tokens, valid, source_indices)

    def forward(
        self,
        images: torch.Tensor,
        source_indices: torch.Tensor,
        *,
        activation_leaf: bool = False,
    ) -> BranchOutput:
        return self.forward_from_projected(
            self.project(images), source_indices, activation_leaf=activation_leaf
        )


def assert_independent_branches(foreground: ForegroundBranch, background: BackgroundBranch) -> None:
    foreground_pointers = {parameter.data_ptr() for parameter in foreground.parameters()}
    shared = [
        name
        for name, parameter in background.named_parameters()
        if parameter.data_ptr() in foreground_pointers
    ]
    if shared:
        raise AssertionError(f"foreground/background branches share storage: {shared}")
