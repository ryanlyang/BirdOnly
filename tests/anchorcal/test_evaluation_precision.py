from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from anchorcal.precision import (
    evaluation_autocast,
    evaluation_inference,
    saliency_evaluation,
)


class EvaluationPrecisionPolicyTests(unittest.TestCase):
    def test_cuda_uses_binding_bfloat16_autocast(self) -> None:
        context = unittest.mock.MagicMock()
        with patch("torch.autocast", return_value=context) as autocast:
            returned = evaluation_autocast(SimpleNamespace(type="cuda"))
        self.assertIs(returned, context)
        autocast.assert_called_once_with(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        )

    def test_cpu_keeps_the_same_context_but_disables_autocast(self) -> None:
        context = unittest.mock.MagicMock()
        with patch("torch.autocast", return_value=context) as autocast:
            returned = evaluation_autocast(SimpleNamespace(type="cpu"))
        self.assertIs(returned, context)
        autocast.assert_called_once_with(
            device_type="cpu", dtype=torch.bfloat16, enabled=False
        )

    def test_non_saliency_context_enables_inference_mode(self) -> None:
        with evaluation_inference(torch.device("cpu")):
            self.assertTrue(torch.is_inference_mode_enabled())
            self.assertFalse(torch.is_grad_enabled())

    def test_saliency_context_keeps_gradients_enabled(self) -> None:
        value = torch.tensor(2.0, requires_grad=True)
        with torch.no_grad():
            with saliency_evaluation(torch.device("cpu")):
                self.assertTrue(torch.is_grad_enabled())
                result = value.square()
        self.assertEqual(float(torch.autograd.grad(result, value)[0]), 4.0)


def _callee_root_and_name(node: ast.AST) -> tuple[str | None, str | None]:
    if isinstance(node, ast.Name):
        return node.id, "__call__"
    if not isinstance(node, ast.Attribute):
        return None, None
    attribute = node.attr
    value = node.value
    while isinstance(value, ast.Attribute):
        value = value.value
    return (value.id if isinstance(value, ast.Name) else None), attribute


class _ForwardContextVisitor(ast.NodeVisitor):
    """Find model forwards not lexically guarded by a locked eval context."""

    _MODEL_NAMES = {"model", "foreground", "background"}
    _FORWARD_NAMES = {
        "__call__",
        "project",
        "forward_from_projected",
        "forward_with_patch_leaf",
        "forward_tokens",
    }
    _CONTEXT_NAMES = {"evaluation_inference", "saliency_evaluation"}

    def __init__(self) -> None:
        self.context_depth = 0
        self.offenders: list[tuple[int, str]] = []

    @staticmethod
    def _context_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Call):
            node = node.func
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    def visit_With(self, node: ast.With) -> None:
        guarded = any(
            self._context_name(item.context_expr) in self._CONTEXT_NAMES
            for item in node.items
        )
        if guarded:
            self.context_depth += 1
        for statement in node.body:
            self.visit(statement)
        if guarded:
            self.context_depth -= 1

    def visit_Call(self, node: ast.Call) -> None:
        root, name = _callee_root_and_name(node.func)
        if (
            root in self._MODEL_NAMES
            and name in self._FORWARD_NAMES
            and self.context_depth == 0
        ):
            self.offenders.append((node.lineno, f"{root}.{name}"))
        self.generic_visit(node)


class EvaluationForwardCoverageTests(unittest.TestCase):
    """Regression guard for every production model-evaluation entry point."""

    _ENTRY_POINTS = {
        "candidate_evaluation.py": {
            "evaluate_plain",
            "evaluate_practical_criteria",
        },
        "branch_pipeline.py": {"evaluate_branch"},
        "anchor_pipeline.py": {
            "_foreground_invariance_audit",
            "_random_token_audit",
            "_branch_extremes_and_saliency",
            "_direct_lambda_saliency",
            "evaluate_anchor_ladder",
        },
    }

    def test_all_evaluation_forwards_use_locked_precision_context(self) -> None:
        source_root = Path(__file__).resolve().parents[2] / "src" / "anchorcal"
        offenders: list[str] = []
        for filename, entry_points in self._ENTRY_POINTS.items():
            tree = ast.parse((source_root / filename).read_text(encoding="utf-8"))
            functions = {
                node.name: node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            self.assertEqual(entry_points - functions.keys(), set())
            for function_name in sorted(entry_points):
                visitor = _ForwardContextVisitor()
                for statement in functions[function_name].body:
                    visitor.visit(statement)
                offenders.extend(
                    f"{filename}:{line} {function_name}: {call}"
                    for line, call in visitor.offenders
                )
        self.assertEqual(offenders, [], "unguarded eval forwards:\n" + "\n".join(offenders))


if __name__ == "__main__":
    unittest.main()
