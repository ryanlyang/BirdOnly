from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from anchorcal.errors import PreflightError
from anchorcal.io import sha256_file
from anchorcal.selector_anchor_verification import (
    ANCHOR_ARTIFACT_KEYS,
    ANCHOR_RESULT_KEYS,
    _expected_paths,
    verify_selector_anchor_artifacts,
)


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PACKAGE = SRC / "anchorcal"

FORBIDDEN_SELECTOR_MODULES = {
    "analysis",
    "analysis_only_splits",
    "anchor_artifacts",
    "branch_provenance",
    "campaign_verification",
    "candidate_pipeline",
    "checkpoint_verification",
    "checkpoints",
    "hidden_analysis",
    "hidden_storage",
    "mask_visual_audit",
    "preflight",
    "storage",
    "vlm_masks",
}
FORBIDDEN_SELECTOR_IDENTIFIERS = {
    "HIDDEN_CHECKPOINT_SCHEMA_VERSION",
    "HIDDEN_SELECTORS",
    "_verify_hidden_checkpoint_artifacts",
    "verify_candidate_checkpoint_artifacts",
    "load_vlm_mask_bank",
    "VlmMaskBank",
}
FORBIDDEN_SELECTOR_LITERALS = {
    "anchorcal-hidden-checkpoints",
    "anchorcal-analysis-only-splits",
    "analysis_only/splits",
    "waterbirds100_oracle_val.csv",
    "waterbirds100_test.csv",
    "exploratory_hidden",
    "oracle_manifest.json",
    "analysis_only/masks",
    "waterbirds100_oracle_val_mask_audit.json",
    "anchorcal-analysis-only-vlm-mask-audit",
    "preflight/mask_manifest.json",
    "preflight/report.json",
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
    def test_selector_anchor_manifest_rejects_any_unreviewed_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            config = {
                "paths": {"output_root": str(output)},
                "runtime": {"debug": False},
                "resolved_config_sha256": "a" * 64,
            }
            records: dict[str, dict[str, object]] = {}
            for name, path in _expected_paths(config).items():
                path.parent.mkdir(parents=True, exist_ok=True)
                if name == "criterion_results":
                    path.write_text(
                        json.dumps({key: {} for key in ANCHOR_RESULT_KEYS}),
                        encoding="utf-8",
                    )
                else:
                    path.write_bytes(b"selector-safe fixture\n")
                records[name] = {
                    "path": str(path.relative_to(output)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            self.assertEqual(set(records), ANCHOR_ARTIFACT_KEYS)
            unexpected = output / "anchors" / "unreviewed.json"
            unexpected.write_text("{}\n", encoding="utf-8")
            records["unreviewed"] = {
                "path": str(unexpected.relative_to(output)),
                "size_bytes": unexpected.stat().st_size,
                "sha256": sha256_file(unexpected),
            }
            manifest = {
                "schema_version": "anchorcal-anchor-artifacts-v1",
                "resolved_config_sha256": config["resolved_config_sha256"],
                "criterion_result_keys": sorted(ANCHOR_RESULT_KEYS),
                "files": records,
            }
            manifest_path = output / "anchors" / "artifact_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(PreflightError, "exactly allowlisted"):
                verify_selector_anchor_artifacts(
                    config, output / "missing_decision_receipt.json"
                )

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
    'anchorcal.analysis_only_splits',
    'anchorcal.anchor_artifacts',
    'anchorcal.branch_provenance',
    'anchorcal.campaign_verification',
    'anchorcal.candidate_pipeline',
    'anchorcal.checkpoint_verification',
    'anchorcal.checkpoints',
    'anchorcal.storage',
    'anchorcal.hidden_storage',
    'anchorcal.hidden_analysis',
    'anchorcal.mask_visual_audit',
    'anchorcal.preflight',
    'anchorcal.vlm_masks',
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

    def test_candidate_provenance_reads_only_compact_mask_receipt(self) -> None:
        source = (SRC / "anchorcal" / "candidate_provenance.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        literals = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertTrue(any("selector_mask_receipt.json" in value for value in literals))
        for forbidden in (
            "mask_manifest.json",
            "report.json",
            "analysis_only/masks",
            "oracle_val_mask_audit",
        ):
            self.assertFalse(
                any(forbidden in value for value in literals),
                f"candidate provenance names forbidden artifact {forbidden}",
            )
        identifiers = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        } | {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        }
        self.assertNotIn("load_vlm_mask_bank", identifiers)
        self.assertNotIn("VlmMaskBank", identifiers)

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
