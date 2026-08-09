"""Ordinary full-image ViT candidate with explicit patch-token forward paths."""

from __future__ import annotations

import torch
from torch import nn

from ..pretrained import create_pretrained_vit


class CandidateViT(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        embed_dim = int(backbone.embed_dim)
        self.backbone.head = nn.Linear(embed_dim, 2)
        nn.init.trunc_normal_(self.backbone.head.weight, std=0.02)
        nn.init.zeros_(self.backbone.head.bias)

    @classmethod
    def from_verified_weights(cls, weights_path: str):
        return cls(create_pretrained_vit(weights_path))

    def project(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone.patch_embed.proj(images).flatten(2).transpose(1, 2)

    def forward_from_projected(self, projected: torch.Tensor) -> torch.Tensor:
        model = self.backbone
        batch = projected.shape[0]
        cls = model.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, projected], dim=1)
        tokens = model.pos_drop(tokens + model.pos_embed)
        tokens = model.patch_drop(tokens)
        tokens = model.norm_pre(tokens)
        for block in model.blocks:
            tokens = block(tokens)
        tokens = model.norm(tokens)
        pooled = tokens[:, 0]
        if hasattr(model, "fc_norm") and not isinstance(model.fc_norm, nn.Identity):
            pooled = model.fc_norm(pooled)
        pooled = model.head_drop(pooled)
        return model.head(pooled)

    def forward_with_patch_leaf(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.project(images).detach().requires_grad_(True)
        return self.forward_from_projected(projected), projected

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_from_projected(self.project(images))

