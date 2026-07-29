"""Four-block pretrained ViT set transformer with masked background tokens."""

from __future__ import annotations

import copy
import hashlib
from typing import Any

from setv.errors import DataValidationError


def _module_digest(modules: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for prefix, module in sorted(modules.items()):
        if hasattr(module, "state_dict"):
            state = module.state_dict()
        else:
            state = {"value": module}
        for name, value in sorted(state.items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(f"{prefix}.{name}".encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class MaskedViTBlock:
    """A timm ViT block whose attention excludes padded/foreground tokens."""

    @staticmethod
    def build(source_block):
        import torch

        class _Block(torch.nn.Module):
            def __init__(self, source):
                super().__init__()
                self.norm1 = copy.deepcopy(source.norm1)
                self.attn = copy.deepcopy(source.attn)
                self.ls1 = copy.deepcopy(source.ls1)
                self.drop_path1 = copy.deepcopy(source.drop_path1)
                self.norm2 = copy.deepcopy(source.norm2)
                self.mlp = copy.deepcopy(source.mlp)
                self.ls2 = copy.deepcopy(source.ls2)
                self.drop_path2 = copy.deepcopy(source.drop_path2)

            def _attention(self, value, key_valid):
                batch, tokens, channels = value.shape
                heads = int(self.attn.num_heads)
                head_dim = channels // heads
                qkv = (
                    self.attn.qkv(value)
                    .reshape(batch, tokens, 3, heads, head_dim)
                    .permute(2, 0, 3, 1, 4)
                )
                query, key, val = qkv.unbind(0)
                query = self.attn.q_norm(query)
                key = self.attn.k_norm(key)
                query = query * self.attn.scale
                attention = query @ key.transpose(-2, -1)
                attention = attention.masked_fill(
                    ~key_valid[:, None, None, :],
                    torch.finfo(attention.dtype).min,
                )
                attention = self.attn.attn_drop(attention.softmax(dim=-1))
                output = (attention @ val).transpose(1, 2).reshape(
                    batch, tokens, channels
                )
                output = self.attn.proj(output)
                return self.attn.proj_drop(output)

            def forward(self, value, key_valid):
                value = value + self.drop_path1(
                    self.ls1(self._attention(self.norm1(value), key_valid))
                )
                value = value + self.drop_path2(self.ls2(self.mlp(self.norm2(value))))
                return value

        return _Block(source_block)


def coarse_spatial_bin_ids(
    grid_height: int, grid_width: int, bins_per_axis: int = 3
):
    import torch

    rows = torch.arange(grid_height)
    columns = torch.arange(grid_width)
    row_bins = torch.clamp(rows * bins_per_axis // grid_height, max=bins_per_axis - 1)
    column_bins = torch.clamp(
        columns * bins_per_axis // grid_width, max=bins_per_axis - 1
    )
    return (
        row_bins[:, None] * bins_per_axis + column_bins[None, :]
    ).reshape(-1)


class SetBackgroundTransformer:
    @staticmethod
    def build(backbone, config: dict[str, Any]):
        import torch

        class _Model(torch.nn.Module):
            def __init__(self, source, model_config):
                super().__init__()
                settings = model_config["model"]
                input_settings = model_config["input"]
                depth = int(settings["transformer_blocks"])
                if len(source.blocks) < depth:
                    raise DataValidationError("Pretrained ViT has too few blocks")
                projection = source.patch_embed.proj
                if tuple(projection.kernel_size) != (
                    int(input_settings["patch_size"]),
                    int(input_settings["patch_size"]),
                ):
                    raise DataValidationError("Pretrained patch projection size changed")
                embedding_dim = int(settings["embedding_dim"])
                if projection.out_channels != embedding_dim:
                    raise DataValidationError("Pretrained embedding dimension changed")
                if source.cls_token.shape[-1] != embedding_dim:
                    raise DataValidationError("Pretrained CLS dimension changed")
                expected_heads = int(settings["attention_heads"])
                expected_hidden = int(embedding_dim * float(settings["mlp_ratio"]))
                for index in range(depth):
                    block = source.blocks[index]
                    if int(block.attn.num_heads) != expected_heads:
                        raise DataValidationError(
                            f"Pretrained block {index} has the wrong attention-head count"
                        )
                    linear_layers = [
                        module
                        for module in block.mlp.modules()
                        if isinstance(module, torch.nn.Linear)
                    ]
                    if not linear_layers or linear_layers[0].out_features != expected_hidden:
                        raise DataValidationError(
                            f"Pretrained block {index} has the wrong MLP ratio"
                        )
                self.patch_projection = copy.deepcopy(projection)
                self.cls_token = torch.nn.Parameter(source.cls_token.detach().clone())
                self.blocks = torch.nn.ModuleList(
                    [
                        MaskedViTBlock.build(source.blocks[index])
                        for index in range(depth)
                    ]
                )
                self.norm = copy.deepcopy(source.norm)
                bins = int(input_settings["coarse_spatial_bins_per_axis"])
                self.coarse_position = torch.nn.Embedding(
                    bins * bins, embedding_dim
                )
                self.input_dropout = torch.nn.Dropout(float(settings["dropout"]))
                self.head = torch.nn.Linear(
                    embedding_dim, int(settings["num_classes"])
                )
                torch.nn.init.trunc_normal_(
                    self.coarse_position.weight, std=0.02
                )
                torch.nn.init.trunc_normal_(self.head.weight, std=0.02)
                torch.nn.init.zeros_(self.head.bias)
                grid_size = source.patch_embed.grid_size
                if isinstance(grid_size, int):
                    grid_size = (grid_size, grid_size)
                coarse = coarse_spatial_bin_ids(
                    int(grid_size[0]), int(grid_size[1]), bins
                )
                self.register_buffer("coarse_bin_ids", coarse, persistent=True)
                self.initialization_report = {
                    "source_architecture": settings["architecture"],
                    "pretrained": bool(settings["pretrained"]),
                    "copied": [
                        "patch_projection",
                        "cls_token",
                        f"transformer_blocks[0:{depth}]",
                        "block_layernorm_attention_mlp",
                        "final_layernorm",
                    ],
                    "scratch": ["coarse_3x3_position_embeddings", "classification_head"],
                    "dense_position_embeddings_discarded": True,
                    "second_attention_pooling_module": False,
                    "pooling": "pretrained_cls_through_four_pretrained_blocks",
                    "exact_patch_coordinates_supplied": False,
                    "coarse_spatial_bins": f"{bins}x{bins}",
                    "pretrained_component_sha256": _module_digest(
                        {
                            "patch_projection": self.patch_projection,
                            "cls_token": self.cls_token,
                            "blocks": self.blocks,
                            "norm": self.norm,
                        }
                    ),
                }

            def encode_patch_tokens(
                self, patch_tokens, token_mask, coarse_ids=None
            ):
                if patch_tokens.ndim != 3 or token_mask.ndim != 2:
                    raise DataValidationError("Set tokens/mask must have ranks 3 and 2")
                if patch_tokens.shape[:2] != token_mask.shape:
                    raise DataValidationError("Set tokens and validity mask are misaligned")
                if coarse_ids is None:
                    coarse_ids = self.coarse_bin_ids
                if coarse_ids.shape != (patch_tokens.shape[1],):
                    raise DataValidationError("Coarse-bin IDs do not match patch tokens")
                position = self.coarse_position(coarse_ids).unsqueeze(0)
                patch_tokens = (patch_tokens + position) * token_mask.unsqueeze(-1)
                cls = self.cls_token.expand(patch_tokens.shape[0], -1, -1)
                value = self.input_dropout(torch.cat([cls, patch_tokens], dim=1))
                key_valid = torch.cat(
                    [
                        torch.ones(
                            (token_mask.shape[0], 1),
                            dtype=torch.bool,
                            device=token_mask.device,
                        ),
                        token_mask,
                    ],
                    dim=1,
                )
                for block in self.blocks:
                    value = block(value, key_valid)
                return self.norm(value)[:, 0]

            def forward(self, images, token_mask):
                patches = self.patch_projection(images).flatten(2).transpose(1, 2)
                if token_mask.ndim == 2:
                    representation = self.encode_patch_tokens(patches, token_mask)
                    return self.head(representation)
                if token_mask.ndim != 3:
                    raise DataValidationError("Token mask must have shape [B,P] or [B,V,P]")
                batch, views, tokens = token_mask.shape
                if patches.shape[:2] != (batch, tokens):
                    raise DataValidationError("Image patches and token views are misaligned")
                expanded = (
                    patches[:, None, :, :]
                    .expand(batch, views, tokens, patches.shape[-1])
                    .reshape(batch * views, tokens, patches.shape[-1])
                )
                representation = self.encode_patch_tokens(
                    expanded, token_mask.reshape(batch * views, tokens)
                )
                return self.head(representation).reshape(batch, views, -1)

        return _Model(backbone, config)


def create_set_expert_model(config: dict[str, Any]):
    try:
        import timm
    except ImportError as exc:
        raise DataValidationError(
            "timm is required for set-expert training. On Tigris activate "
            "/home/ryreu/miniforge3-aarch64/envs/fcv_gh200."
        ) from exc
    settings = config["model"]
    backbone = timm.create_model(
        settings["architecture"],
        pretrained=bool(settings["pretrained"]),
        num_classes=int(settings["num_classes"]),
        drop_rate=float(settings["dropout"]),
        attn_drop_rate=float(settings["dropout"]),
        drop_path_rate=0.0,
    )
    model = SetBackgroundTransformer.build(backbone, config)
    model.pretrained_cfg = getattr(backbone, "pretrained_cfg", None)
    return model
