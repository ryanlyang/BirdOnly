"""Pretrained ViT blocks with explicit variable-token padding support."""

from __future__ import annotations

import copy
import math

import torch
from torch import nn


def _apply_layer_scale(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
    return module(value)


class MaskedClonedBlock(nn.Module):
    """Clone a timm block while making key/value padding explicit.

    The manual attention path uses the cloned qkv/projection/normalization/MLP
    weights. Invalid token outputs are zeroed after attention and after the MLP.
    """

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.norm1 = copy.deepcopy(source.norm1)
        self.qkv = copy.deepcopy(source.attn.qkv)
        self.q_norm = copy.deepcopy(getattr(source.attn, "q_norm", nn.Identity()))
        self.k_norm = copy.deepcopy(getattr(source.attn, "k_norm", nn.Identity()))
        self.attn_drop = copy.deepcopy(source.attn.attn_drop)
        self.proj = copy.deepcopy(source.attn.proj)
        self.proj_drop = copy.deepcopy(source.attn.proj_drop)
        self.norm2 = copy.deepcopy(source.norm2)
        self.mlp = copy.deepcopy(source.mlp)
        self.drop_path1 = copy.deepcopy(getattr(source, "drop_path1", nn.Identity()))
        self.drop_path2 = copy.deepcopy(getattr(source, "drop_path2", nn.Identity()))
        self.ls1 = copy.deepcopy(getattr(source, "ls1", nn.Identity()))
        self.ls2 = copy.deepcopy(getattr(source, "ls2", nn.Identity()))
        self.num_heads = int(source.attn.num_heads)
        self.scale = float(source.attn.scale)

    def _attention(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        head_dim = channels // self.num_heads
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, value = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        scores = (q * self.scale) @ k.transpose(-2, -1)
        scores = scores.masked_fill(~valid[:, None, None, :], torch.finfo(scores.dtype).min)
        attention = scores.softmax(dim=-1)
        attention = self.attn_drop(attention)
        output = (attention @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(output))

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if valid.dtype != torch.bool or valid.shape != x.shape[:2]:
            raise ValueError("valid must be a bool [batch,tokens] padding mask")
        if not valid[:, 0].all():
            raise ValueError("CLS must always be valid")
        mask = valid.unsqueeze(-1).to(x.dtype)
        attention = self._attention(self.norm1(x), valid)
        x = (x + self.drop_path1(_apply_layer_scale(self.ls1, attention))) * mask
        x = (x + self.drop_path2(_apply_layer_scale(self.ls2, self.mlp(self.norm2(x))))) * mask
        return x


class VariableTokenEncoder(nn.Module):
    def __init__(self, source_vit: nn.Module, depth: int = 6, num_classes: int = 2) -> None:
        super().__init__()
        self.embed_dim = int(source_vit.embed_dim)
        self.cls_token = nn.Parameter(source_vit.cls_token.detach().clone())
        self.blocks = nn.ModuleList(
            [MaskedClonedBlock(source_vit.blocks[index]) for index in range(depth)]
        )
        self.norm = copy.deepcopy(source_vit.norm)
        self.head = nn.Linear(self.embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward_tokens(
        self, patch_tokens: torch.Tensor, patch_valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = patch_tokens.shape[0]
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, patch_tokens], dim=1)
        valid = torch.cat(
            [torch.ones(batch, 1, dtype=torch.bool, device=patch_valid.device), patch_valid],
            dim=1,
        )
        tokens = tokens * valid.unsqueeze(-1).to(tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens, valid)
        tokens = self.norm(tokens)
        tokens = tokens * valid.unsqueeze(-1).to(tokens.dtype)
        return self.head(tokens[:, 0]), tokens


def pad_token_sequences(
    sequences: list[torch.Tensor], source_indices: list[torch.Tensor] | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not sequences or any(sequence.ndim != 2 for sequence in sequences):
        raise ValueError("sequences must be a non-empty list of [tokens,dim] tensors")
    maximum = max(sequence.shape[0] for sequence in sequences)
    if maximum == 0:
        raise ValueError("each example needs at least one valid token")
    batch = len(sequences)
    dim = sequences[0].shape[1]
    padded = sequences[0].new_zeros((batch, maximum, dim))
    valid = torch.zeros((batch, maximum), dtype=torch.bool, device=sequences[0].device)
    index_pad = None
    if source_indices is not None:
        index_pad = torch.full((batch, maximum), -1, dtype=torch.long, device=sequences[0].device)
    for row, sequence in enumerate(sequences):
        count = sequence.shape[0]
        padded[row, :count] = sequence
        valid[row, :count] = True
        if index_pad is not None:
            indices = source_indices[row]
            if indices.shape != (count,):
                raise ValueError("source index count does not match token count")
            index_pad[row, :count] = indices
    return padded, valid, index_pad

