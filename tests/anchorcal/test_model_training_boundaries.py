from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    from anchorcal.branch_pipeline import (
        _aggregate_crop_fallback_gate,
        _branch_restart_provenance,
        _persist_crop_fallback_events,
        _validate_branch_restart,
    )
    from anchorcal.errors import AuditFailure, PreflightError
    from anchorcal.io import atomic_write_json, sha256_file
    from anchorcal.anchor_cache import (
        ExtremeCache,
        cached_logits,
        cached_signed_contributions,
    )
    from anchorcal.datasets import collate_background_training
    from anchorcal.models.anchor import RelianceAnchor, mix_anchor_logits
    from anchorcal.models.branches import (
        BackgroundBranch,
        ForegroundBranch,
        ObjectRelativePosition,
        assert_independent_branches,
        object_relative_coordinates,
    )
    from anchorcal.models.candidate import CandidateViT
    from anchorcal.models.token_vit import VariableTokenEncoder
    from anchorcal.training import UpdateScheduler, parameter_groups, update_factor
    from anchorcal.transforms import check_fallback_rate


if TORCH_AVAILABLE:

    class TinyAttention(nn.Module):
        def __init__(self, dim: int = 8, heads: int = 2) -> None:
            super().__init__()
            self.num_heads = heads
            self.scale = (dim // heads) ** -0.5
            self.qkv = nn.Linear(dim, dim * 3)
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.attn_drop = nn.Dropout(0.0)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(0.0)


    class TinyBlock(nn.Module):
        def __init__(self, dim: int = 8) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = TinyAttention(dim)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim),
            )
            self.drop_path1 = nn.Identity()
            self.drop_path2 = nn.Identity()
            self.ls1 = nn.Identity()
            self.ls2 = nn.Identity()

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            # CandidateViT only needs a shape-preserving ordinary block.
            return value


    class TinyPatchEmbed(nn.Module):
        def __init__(self, dim: int = 8) -> None:
            super().__init__()
            self.proj = nn.Conv2d(3, dim, kernel_size=16, stride=16)


    class TinyViT(nn.Module):
        def __init__(self, dim: int = 8, depth: int = 2) -> None:
            super().__init__()
            self.embed_dim = dim
            self.patch_embed = TinyPatchEmbed(dim)
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
            self.pos_embed = nn.Parameter(torch.randn(1, 197, dim))
            self.pos_drop = nn.Identity()
            self.patch_drop = nn.Identity()
            self.norm_pre = nn.Identity()
            self.blocks = nn.ModuleList([TinyBlock(dim) for _ in range(depth)])
            self.norm = nn.LayerNorm(dim)
            self.fc_norm = nn.Identity()
            self.head_drop = nn.Identity()
            self.head = nn.Linear(dim, 1000)


@unittest.skipUnless(TORCH_AVAILABLE, "torch unavailable")
class OptimizerAndScheduleBoundaryTests(unittest.TestCase):
    def test_parameter_groups_cover_each_parameter_and_lock_exclusions(self) -> None:
        class GroupFixture(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.cls_token = nn.Parameter(torch.randn(1, 1, 8))
                self.pos_embed = nn.Parameter(torch.randn(1, 5, 8))
                self.matrix = nn.Parameter(torch.randn(8, 8))
                self.linear = nn.Linear(8, 4)
                self.norm = nn.LayerNorm(4)
                self.position_mlp = nn.Linear(2, 8)

        model = GroupFixture()
        groups, manifest = parameter_groups(model, 0.05)
        self.assertEqual(groups[0]["weight_decay"], 0.05)
        self.assertEqual(groups[1]["weight_decay"], 0.0)
        self.assertEqual(
            set(manifest["decay_names"]),
            {"linear.weight", "matrix", "position_mlp.weight"},
        )
        self.assertEqual(
            set(manifest["no_decay_names"]),
            {
                "cls_token",
                "linear.bias",
                "norm.bias",
                "norm.weight",
                "pos_embed",
                "position_mlp.bias",
            },
        )
        grouped = {id(parameter) for group in groups for parameter in group["params"]}
        expected = {id(parameter) for parameter in model.parameters()}
        self.assertEqual(grouped, expected)

    def test_schedule_uses_exact_first_warmup_and_final_endpoints(self) -> None:
        self.assertAlmostEqual(update_factor(0, 10, 3), 0.01)
        self.assertAlmostEqual(update_factor(2, 10, 3), 1.0)
        self.assertAlmostEqual(update_factor(3, 10, 3), 1.0)
        self.assertAlmostEqual(update_factor(9, 10, 3), 0.01)
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.2)
        scheduler = UpdateScheduler(
            optimizer, base_lr=0.2, total_updates=10, warmup_updates=3
        )
        observed = [optimizer.param_groups[0]["lr"]]
        for _ in range(9):
            scheduler.step_after_optimizer()
            observed.append(optimizer.param_groups[0]["lr"])
        self.assertAlmostEqual(observed[0], 0.002)
        self.assertAlmostEqual(observed[2], 0.2)
        self.assertAlmostEqual(observed[-1], 0.002)


@unittest.skipUnless(TORCH_AVAILABLE, "torch unavailable")
class CandidateBoundaryTests(unittest.TestCase):
    def test_candidate_head_is_two_class_zero_bias_and_trunc_normal_initialized(self) -> None:
        torch.manual_seed(17)
        backbone = TinyViT()
        old_head = backbone.head
        with patch(
            "torch.nn.init.trunc_normal_", wraps=torch.nn.init.trunc_normal_
        ) as initializer:
            candidate = CandidateViT(backbone)
        self.assertIsNot(candidate.backbone.head, old_head)
        self.assertEqual(tuple(candidate.backbone.head.weight.shape), (2, 8))
        torch.testing.assert_close(
            candidate.backbone.head.bias,
            torch.zeros_like(candidate.backbone.head.bias),
        )
        initializer.assert_called_once()
        self.assertEqual(initializer.call_args.kwargs["std"], 0.02)

    def test_candidate_training_dataset_api_is_mask_free(self) -> None:
        source_path = (
            Path(__file__).resolve().parents[2] / "src/anchorcal/datasets.py"
        )
        module = ast.parse(source_path.read_text(encoding="utf-8"))
        candidate_class = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef)
            and node.name == "CandidateTrainingDataset"
        )
        initializer = next(
            node
            for node in candidate_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        parameter_names = [argument.arg for argument in initializer.args.args]
        self.assertNotIn("mask_root", parameter_names)
        self.assertNotIn("mask_bank", parameter_names)
        class_names = {
            node.id for node in ast.walk(candidate_class) if isinstance(node, ast.Name)
        }
        self.assertNotIn("load_binary_mask", class_names)
        self.assertNotIn("load_vlm_mask_bank", class_names)
        self.assertNotIn("VlmMaskBank", class_names)
        self.assertIn("stateless_image_train_transform", class_names)


@unittest.skipUnless(TORCH_AVAILABLE, "torch unavailable")
class BranchAndAnchorBoundaryTests(unittest.TestCase):
    def test_restart_rejects_changed_data_and_branch_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            preflight = output / "preflight"
            splits = output / "splits"
            geometry = output / "debug" / "preflight" / "geometry"
            branch_root = output / "debug" / "branches" / "foreground"
            for directory in (preflight, splits, geometry, branch_root):
                directory.mkdir(parents=True, exist_ok=True)
            weights = output / "weights.safetensors"
            weights.write_bytes(b"synthetic pretrained weights")
            atomic_write_json(
                preflight / "pretrained_manifest.json",
                {
                    "repository": "synthetic/repository",
                    "revision": "frozen-revision",
                    "weights_path": str(weights.resolve()),
                    "weights_sha256": sha256_file(weights),
                },
            )
            for path, payload in (
                (preflight / "report.json", {"status": "passed"}),
                (
                    preflight / "mask_manifest.json",
                    {
                        "schema_version": "anchorcal-vlm-mask-manifest-v2",
                        "producer": (
                            "waterbirds100_openclip_laion_dinovit_weclipplus_prediction_cmap"
                        ),
                        "runtime_resolution": "frozen_manifest_only",
                    },
                ),
                (preflight / "preprocessing_manifest.json", {"resize": 248}),
                (splits / "manifest.json", {"split": "frozen"}),
                (geometry / "manifest.json", {"geometry": "frozen"}),
            ):
                atomic_write_json(path, payload)
            expert_split = splits / "waterbirds100_expert_train.csv"
            expert_split.write_text("img_id,y\n1,0\n", encoding="utf-8")
            optimizer_groups = branch_root / "optimizer_groups.json"
            atomic_write_json(optimizer_groups, {"decay_names": ["weight"]})
            config = {
                "paths": {
                    "output_root": str(output),
                    "metadata_path": str(output / "metadata.csv"),
                    "waterbirds_root": str(output / "waterbirds"),
                    "vlm_mask_root": str(output / "prediction_cmap"),
                },
                "masks": {
                    "source": (
                        "waterbirds100_openclip_laion_dinovit_weclipplus_prediction_cmap"
                    ),
                    "mapping_mode": (
                        "weclip_producer_first_with_explicit_legacy_fallbacks"
                    ),
                    "format": "voc_colormap_class_ids",
                    "foreground_class_ids": [1],
                    "allowed_class_ids": [0, 1],
                    "minimum_foreground_fraction": 0.0,
                    "maximum_foreground_fraction": 1.0,
                    "required_official_splits": [0, 1],
                    "optional_official_splits": [2],
                    "runtime_resolve_from_manifest_only": True,
                },
                "runtime": {"debug": True},
                "branches": {
                    "foreground_position_mode": "object_relative",
                    "copied_blocks": [0, 1, 2, 3, 4, 5],
                },
                "seeds": {"foreground_branch_train": 6001},
                "resolved_config_sha256": "a" * 64,
            }
            provenance = _branch_restart_provenance(
                config,
                "foreground",
                token_budget=None,
                optimizer_groups_path=optimizer_groups,
            )
            history = [
                {
                    "epoch": 1,
                    "sampled_examples": 1,
                    "fallback_count": 0,
                }
            ]
            history_path = branch_root / "history.json"
            atomic_write_json(history_path, history)
            state = {
                "schema_version": "anchorcal-branch-restart-v2",
                "branch": "foreground",
                "provenance": provenance,
                "epoch": 1,
                "history": history,
            }
            replay_history, next_epoch = _validate_branch_restart(
                state,
                provenance,
                branch="foreground",
                epochs=3,
                history_path=history_path,
            )
            self.assertEqual(replay_history, history)
            self.assertEqual(next_epoch, 2)

            expert_split.write_text("img_id,y\n1,1\n", encoding="utf-8")
            changed = _branch_restart_provenance(
                config,
                "foreground",
                token_budget=None,
                optimizer_groups_path=optimizer_groups,
            )
            with self.assertRaisesRegex(PreflightError, "provenance mismatch"):
                _validate_branch_restart(
                    state,
                    changed,
                    branch="foreground",
                    epochs=3,
                    history_path=history_path,
                )
            with self.assertRaisesRegex(PreflightError, "provenance mismatch"):
                _validate_branch_restart(
                    state,
                    provenance,
                    branch="background",
                    epochs=3,
                    history_path=history_path,
                )

    def test_crop_fallback_gate_uses_the_whole_training_run(self) -> None:
        # The first epoch alone is 1%, but the locked full-run rate is below
        # 0.1%; a per-epoch implementation would incorrectly abort here.
        result = _aggregate_crop_fallback_gate(
            [
                {"fallback_count": 1, "sampled_examples": 100},
                {"fallback_count": 0, "sampled_examples": 1000},
            ]
        )
        self.assertEqual(result["fallback_count"], 1)
        self.assertEqual(result["sampled_examples"], 1100)
        self.assertAlmostEqual(float(result["fallback_rate"]), 1.0 / 1100.0)

    def test_object_relative_coordinate_formula_and_mlp_initialization(self) -> None:
        mask = torch.zeros(32, 32, dtype=torch.bool)
        mask[4:21, 8:25] = True
        observed = object_relative_coordinates(mask, patch_size=16)
        expected = torch.tensor(
            [[0.0, 0.25], [1.0, 0.25], [0.0, 1.0], [1.0, 1.0]]
        )
        torch.testing.assert_close(observed, expected, atol=0, rtol=0)
        encoder = ObjectRelativePosition(embed_dim=8)
        linear = [
            layer for layer in encoder.network if isinstance(layer, nn.Linear)
        ]
        self.assertEqual(
            [(layer.in_features, layer.out_features) for layer in linear],
            [(2, 128), (128, 8)],
        )
        for layer in linear:
            torch.testing.assert_close(
                layer.bias, torch.zeros_like(layer.bias), atol=0, rtol=0
            )
            observed_std = float(layer.weight.detach().std())
            self.assertGreater(observed_std, 0.005)
            self.assertLess(observed_std, 0.04)

    def test_hard_fallback_failure_keeps_complete_event_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history = [
                {
                    "epoch": 1,
                    "sampled_examples": 100,
                    "fallback_count": 1,
                    "fallback_events": [
                        {
                            "img_id": 17,
                            "epoch": 1,
                            "attempt_count": 10,
                            "fallback": True,
                        }
                    ],
                }
            ]
            path, events = _persist_crop_fallback_events(
                root, "foreground", history
            )
            with self.assertRaises(AuditFailure):
                check_fallback_rate(1, 100)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["event_count"], 1)
            self.assertEqual(payload["events"], events)
            self.assertEqual(payload["events"][0]["img_id"], 17)

    def test_aggregate_gate_failure_persists_denominator_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "crop_fallback_gate.json"
            with self.assertRaises(AuditFailure):
                _aggregate_crop_fallback_gate(
                    [{"fallback_count": 2, "sampled_examples": 100}],
                    branch="background",
                    report_path=path,
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["fallback_count"], 2)
            self.assertEqual(payload["sampled_examples"], 100)

    def test_background_collation_preserves_each_fallback_event(self) -> None:
        samples = []
        for img_id, fallback, attempts in ((11, True, 10), (12, False, 2)):
            samples.append(
                {
                    "image": torch.zeros(3, 224, 224),
                    "mask": torch.zeros(224, 224, dtype=torch.bool),
                    "source_indices": torch.tensor([0, 1]),
                    "y": 0,
                    "img_id": img_id,
                    "fallback": fallback,
                    "attempt_count": attempts,
                    "epoch": 7,
                    "background_valid": True,
                }
            )
        batch = collate_background_training(samples)
        self.assertEqual(
            batch["fallback_events"],
            [{"img_id": 11, "epoch": 7, "attempt_count": 10, "fallback": True}],
        )

    def test_foreground_position_mode_is_explicit_and_deferred_modes_fail(self) -> None:
        source = TinyViT()
        branch = ForegroundBranch(source, depth=1, position_mode="object_relative")
        self.assertEqual(branch.position_mode.value, "object_relative")
        with self.assertRaisesRegex(ValueError, "unsupported/deferred"):
            ForegroundBranch(source, depth=1, position_mode="absolute")

    def setUp(self) -> None:
        torch.manual_seed(101)

    def test_foreground_padding_does_not_change_cls_logits(self) -> None:
        encoder = VariableTokenEncoder(TinyViT(), depth=2, num_classes=2).eval()
        valid_tokens = torch.randn(1, 2, 8)
        compact_valid = torch.ones(1, 2, dtype=torch.bool)
        padded_tokens = torch.cat(
            [valid_tokens, torch.full((1, 3, 8), 1000.0)], dim=1
        )
        padded_valid = torch.tensor([[True, True, False, False, False]])
        compact_logits, _ = encoder.forward_tokens(valid_tokens, compact_valid)
        padded_logits, padded_output = encoder.forward_tokens(
            padded_tokens, padded_valid
        )
        torch.testing.assert_close(compact_logits, padded_logits, atol=1e-6, rtol=0)
        torch.testing.assert_close(
            padded_output[:, 3:], torch.zeros_like(padded_output[:, 3:]), atol=0, rtol=0
        )

    def test_position_free_background_is_permutation_invariant(self) -> None:
        branch = BackgroundBranch(TinyViT(), token_budget=3, depth=2).eval()
        projected = torch.randn(2, 7, 8)
        indices = torch.tensor([[0, 3, 5], [1, 2, 6]])
        permutation = torch.tensor([2, 0, 1])
        original = branch.forward_from_projected(projected, indices).logits
        permuted = branch.forward_from_projected(
            projected, indices[:, permutation]
        ).logits
        torch.testing.assert_close(original, permuted, atol=1e-6, rtol=1e-6)

    def test_branches_clone_independent_parameter_storage(self) -> None:
        source = TinyViT()
        foreground = ForegroundBranch(source, depth=1)
        background = BackgroundBranch(source, token_budget=3, depth=1)
        assert_independent_branches(foreground, background)
        background_before = background.patch_projection.weight.detach().clone()
        with torch.no_grad():
            foreground.patch_projection.weight.add_(7.0)
        torch.testing.assert_close(
            background.patch_projection.weight, background_before, atol=0, rtol=0
        )

    def _endpoint_gradients(self, reliance_lambda: float):
        source = TinyViT(depth=1)
        foreground = ForegroundBranch(source, depth=1)
        background = BackgroundBranch(source, token_budget=3, depth=1)
        anchor = RelianceAnchor(
            foreground,
            background,
            foreground_scale=1.7,
            background_scale=2.3,
            reliance_lambda=reliance_lambda,
        ).eval()
        foreground_images = torch.randn(1, 3, 224, 224)
        background_images = torch.randn(1, 3, 224, 224)
        mask = torch.zeros(1, 224, 224, dtype=torch.bool)
        mask[:, 80:144, 80:144] = True
        views = torch.tensor([[[0, 1, 2], [10, 11, 12]]])
        output = anchor(
            foreground_images,
            mask,
            background_images,
            views,
            activation_leaves=True,
        )
        activations = [output.foreground.patch_activations] + [
            item.patch_activations for item in output.background_views
        ]
        gradients = torch.autograd.grad(output.logits[:, 0].sum(), activations)
        return gradients

    def test_anchor_endpoint_activation_gradients_are_finite_and_not_unused(self) -> None:
        zero_gradients = self._endpoint_gradients(0.0)
        one_gradients = self._endpoint_gradients(1.0)
        for gradients in (zero_gradients, one_gradients):
            for gradient in gradients:
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
        self.assertEqual(float(zero_gradients[0].abs().max()), 0.0)
        self.assertGreater(
            sum(float(value.abs().sum()) for value in zero_gradients[1:]), 0.0
        )
        self.assertGreater(float(one_gradients[0].abs().sum()), 0.0)
        self.assertEqual(
            sum(float(value.abs().max()) for value in one_gradients[1:]), 0.0
        )

    def test_cached_and_direct_logit_and_signed_component_formulas_match(self) -> None:
        foreground = np.asarray([[4.0, -1.0], [0.5, 2.5]], dtype=np.float32)
        background = np.asarray([[-2.0, 3.0], [5.0, 1.0]], dtype=np.float32)
        foreground_signed = np.asarray([[0.2, -0.1], [0.4, 0.3]])
        background_signed = np.asarray([[0.6, -0.2], [-0.5, 0.8]])
        cache = ExtremeCache(
            foreground,
            background,
            foreground_signed,
            background_signed,
        )
        for reliance_lambda in (0.0, 0.35, 0.5, 0.8, 1.0):
            direct = mix_anchor_logits(
                torch.from_numpy(foreground),
                torch.from_numpy(background),
                reliance_lambda,
                1.7,
                2.3,
            ).numpy()
            np.testing.assert_allclose(
                cached_logits(cache, reliance_lambda, 1.7, 2.3),
                direct,
                atol=1e-6,
                rtol=0,
            )
            cached_fg, cached_bg = cached_signed_contributions(
                cache, reliance_lambda
            )
            np.testing.assert_allclose(
                cached_fg, reliance_lambda * foreground_signed, atol=0, rtol=0
            )
            np.testing.assert_allclose(
                cached_bg,
                (1.0 - reliance_lambda) * background_signed,
                atol=0,
                rtol=0,
            )


if __name__ == "__main__":
    unittest.main()
