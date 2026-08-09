from __future__ import annotations

import ast
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PACKAGE = SRC / "anchorcal"

FORBIDDEN_SELECTOR_MODULES = {
    "analysis",
    "campaign_verification",
    "candidate_pipeline",
    "checkpoint_verification",
    "checkpoints",
    "hidden_analysis",
    "hidden_storage",
    "storage",
}
FORBIDDEN_SELECTOR_IDENTIFIERS = {
    "HIDDEN_CHECKPOINT_SCHEMA_VERSION",
    "HIDDEN_SELECTORS",
    "_verify_hidden_checkpoint_artifacts",
    "verify_candidate_checkpoint_artifacts",
}
FORBIDDEN_SELECTOR_LITERALS = {
    "anchorcal-hidden-checkpoints",
    "exploratory_hidden",
    "oracle_manifest.json",
}


def _module_path(module: str) -> Path:
    return PACKAGE.joinpath(*module.split(".")).with_suffix(".py")


def _relative_imports(module: str, tree: ast.AST) -> set[str]:
    """Resolve local AnchorCal imports used by one source module."""

    imported: set[str] = set()
    current_package = module.split(".")[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            keep = len(current_package) - (node.level - 1)
            if keep < 0:
                continue
            prefix = current_package[:keep]
            if node.module:
                candidates = [prefix + node.module.split(".")]
            else:
                candidates = [prefix + alias.name.split(".") for alias in node.names]
            for parts in candidates:
                candidate = ".".join(parts)
                if candidate and _module_path(candidate).is_file():
                    imported.add(candidate)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("anchorcal."):
                    candidate = alias.name.removeprefix("anchorcal.")
                    if _module_path(candidate).is_file():
                        imported.add(candidate)
    return imported


def _selector_import_closure() -> dict[str, ast.AST]:
    pending = ["selector_analysis"]
    closure: dict[str, ast.AST] = {}
    while pending:
        module = pending.pop()
        if module in closure:
            continue
        path = _module_path(module)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        closure[module] = tree
        pending.extend(sorted(_relative_imports(module, tree) - closure.keys()))
    return closure


class StorageNamespaceBoundaryTests(unittest.TestCase):
    def test_selector_analysis_imports_only_visible_storage_api(self) -> None:
        source = (SRC / "anchorcal" / "selector_analysis.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        relative_imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level == 1
        }
        self.assertIn("selector_storage", relative_imports)
        self.assertIn("visible_checkpoint_verification", relative_imports)
        self.assertTrue(
            {
                "checkpoint_verification",
                "storage",
                "hidden_storage",
                "hidden_analysis",
            }.isdisjoint(
                relative_imports
            )
        )

    def test_selector_local_import_closure_cannot_name_reporting_schema(self) -> None:
        closure = _selector_import_closure()
        self.assertIn("visible_checkpoint_verification", closure)
        self.assertTrue(
            FORBIDDEN_SELECTOR_MODULES.isdisjoint(closure),
            f"selector import closure includes reporting modules: "
            f"{sorted(FORBIDDEN_SELECTOR_MODULES.intersection(closure))}",
        )
        for module, tree in closure.items():
            identifiers: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    identifiers.add(node.id)
                elif isinstance(node, ast.Attribute):
                    identifiers.add(node.attr)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    identifiers.add(node.name)
                elif isinstance(node, ast.alias):
                    identifiers.add(node.name.rsplit(".", 1)[-1])
                    if node.asname:
                        identifiers.add(node.asname)
            forbidden_identifiers = FORBIDDEN_SELECTOR_IDENTIFIERS.intersection(
                identifiers
            )
            self.assertFalse(
                forbidden_identifiers,
                f"{module} names reporting checkpoint APIs: "
                f"{sorted(forbidden_identifiers)}",
            )
            literals = {
                node.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }
            found_literals = {
                marker
                for marker in FORBIDDEN_SELECTOR_LITERALS
                if any(marker in value for value in literals)
            }
            self.assertFalse(
                found_literals,
                f"{module} names reporting checkpoint paths/schema: "
                f"{sorted(found_literals)}",
            )

    def test_importing_selector_does_not_load_reporting_storage(self) -> None:
        code = """
import sys
import anchorcal.selector_analysis

forbidden = {
    'anchorcal.analysis',
    'anchorcal.campaign_verification',
    'anchorcal.candidate_pipeline',
    'anchorcal.checkpoint_verification',
    'anchorcal.checkpoints',
    'anchorcal.storage',
    'anchorcal.hidden_storage',
    'anchorcal.hidden_analysis',
}
loaded = forbidden.intersection(sys.modules)
if loaded:
    raise SystemExit('selector import loaded forbidden modules: ' + repr(sorted(loaded)))
if 'anchorcal.visible_checkpoint_verification' not in sys.modules:
    raise SystemExit('selector import did not load visible checkpoint verifier')
"""
        environment = dict(os.environ)
        prior = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(SRC) if not prior else os.pathsep.join((str(SRC), prior))
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def test_visible_api_cannot_name_reporting_artifacts(self) -> None:
        source = (SRC / "anchorcal" / "selector_storage.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        defined_names = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
        }
        self.assertFalse(any(name.startswith("HIDDEN") for name in defined_names))
        self.assertNotIn("exploratory_hidden_metrics.h5", source)


if __name__ == "__main__":
    unittest.main()
