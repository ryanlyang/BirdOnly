from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

from setv.experts.set_config import load_set_expert_config
from setv.experts.set_model import SetBackgroundTransformer


if TORCH_AVAILABLE:

    class FakeAttention(torch.nn.Module):
        def __init__(self, dim: int = 24, heads: int = 6):
            super().__init__()
            self.num_heads = heads
            self.scale = (dim // heads) ** -0.5
            self.qkv = torch.nn.Linear(dim, dim * 3)
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
            self.attn_drop = torch.nn.Dropout(0.1)
            self.proj = torch.nn.Linear(dim, dim)
            self.proj_drop = torch.nn.Dropout(0.1)


    class FakeBlock(torch.nn.Module):
        def __init__(self, dim: int = 24):
            super().__init__()
            self.norm1 = torch.nn.LayerNorm(dim)
            self.attn = FakeAttention(dim)
            self.ls1 = torch.nn.Identity()
            self.drop_path1 = torch.nn.Identity()
            self.norm2 = torch.nn.LayerNorm(dim)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(dim, dim * 4),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(dim * 4, dim),
                torch.nn.Dropout(0.1),
            )
            self.ls2 = torch.nn.Identity()
            self.drop_path2 = torch.nn.Identity()


    class FakePatchEmbed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Conv2d(3, 24, kernel_size=16, stride=16)
            self.grid_size = (4, 4)


    class FakeBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = FakePatchEmbed()
            self.cls_token = torch.nn.Parameter(torch.randn(1, 1, 24))
            self.blocks = torch.nn.ModuleList([FakeBlock() for _ in range(4)])
            self.norm = torch.nn.LayerNorm(24)


@unittest.skipUnless(TORCH_AVAILABLE, "torch unavailable")
class SetModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_set_expert_config(
            ROOT / "configs" / "expert_background_set.yaml", seed=61
        )
        self.config["model"]["embedding_dim"] = 24

    def test_initialization_and_single_or_multiview_shapes(self) -> None:
        torch.manual_seed(7)
        source = FakeBackbone()
        projection = source.patch_embed.proj.weight.detach().clone()
        cls = source.cls_token.detach().clone()
        model = SetBackgroundTransformer.build(source, self.config).eval()
        self.assertTrue(
            torch.equal(model.patch_projection.weight.detach(), projection)
        )
        self.assertTrue(torch.equal(model.cls_token.detach(), cls))
        self.assertTrue(
            model.initialization_report["dense_position_embeddings_discarded"]
        )
        self.assertFalse(
            model.initialization_report["second_attention_pooling_module"]
        )
        images = torch.randn(2, 3, 64, 64)
        mask = torch.ones(2, 16, dtype=torch.bool)
        self.assertEqual(tuple(model(images, mask).shape), (2, 2))
        views = mask[:, None, :].expand(2, 8, 16).clone()
        self.assertEqual(tuple(model(images, views).shape), (2, 8, 2))

    def test_encoding_is_permutation_invariant_with_coarse_ids(self) -> None:
        torch.manual_seed(11)
        model = SetBackgroundTransformer.build(FakeBackbone(), self.config).eval()
        patches = torch.randn(2, 16, 24)
        mask = torch.ones(2, 16, dtype=torch.bool)
        mask[:, -2:] = False
        coarse = model.coarse_bin_ids.clone()
        permutation = torch.randperm(16)
        original = model.encode_patch_tokens(patches, mask, coarse)
        permuted = model.encode_patch_tokens(
            patches[:, permutation],
            mask[:, permutation],
            coarse[permutation],
        )
        self.assertTrue(torch.allclose(original, permuted, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
