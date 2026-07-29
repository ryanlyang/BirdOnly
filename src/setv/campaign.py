"""Frozen campaign manifest and read-only staged readiness checks."""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from setv.errors import ConfigurationError
from setv.utils.hashing import sha256_file, sha256_json


STAGES = ("phase0", "phase1", "phase2", "phase3", "phase4", "phase5", "phase6")
ENVIRONMENT_NAMES = {
    "SETV_CAMPAIGN_ROOT",
    "SETV_OBJECT_SEED",
    "SETV_EXACT_SEED",
    "SETV_MASK_SEED",
    "SETV_SANITIZED_SEED",
    "SETV_SET_SEED",
    "SETV_EXACT_FUSION_SEED",
    "SETV_SANITIZED_FUSION_SEED",
    "SETV_SET_FUSION_SEED",
    "SETV_CANDIDATE_SEEDS",
    "SETV_ULA_SEED",
    "SETV_ULA_REPO",
    "SETV_ULA_ENV",
    "SETV_ULA_SSL_CHECKPOINT",
}


def load_campaign_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigurationError(f"Campaign manifest not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict):
        raise ConfigurationError("Campaign manifest root must be a mapping")
    manifest = deepcopy(manifest)
    manifest["_manifest_path"] = str(source)
    validate_campaign_manifest(manifest)
    return manifest


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigurationError(f"{name} must be a nonnegative integer")
    return int(value)


def validate_campaign_manifest(manifest: dict[str, Any]) -> None:
    if int(manifest.get("schema_version", -1)) != 1:
        raise ConfigurationError("Campaign schema_version must be 1")
    if manifest.get("campaign_id") != "setv_waterbirds95_private_pilot_v1":
        raise ConfigurationError("Unexpected campaign_id")
    for key in (
        "repository",
        "campaign_root",
        "dataset_root",
        "mask_root",
        "ula_repository",
        "main_environment",
    ):
        if not manifest.get("paths", {}).get(key):
            raise ConfigurationError(f"Campaign path is not explicit: {key}")
    seeds = manifest.get("seeds", {})
    scalar_names = (
        "split",
        "visual_audit",
        "object_expert",
        "exact_background_expert",
        "sanitized_mask_bank",
        "sanitized_background_expert",
        "set_background_expert",
        "ula_proxy_and_ssl",
    )
    for name in scalar_names:
        _integer(seeds.get(name), f"seeds.{name}")
    if seeds["split"] != 1729 or seeds["visual_audit"] != 1729:
        raise ConfigurationError("Locked Phase 0 split/audit seeds must be 1729")
    fusion = seeds.get("fusion", {})
    if set(fusion) != {"exact", "sanitized", "set"}:
        raise ConfigurationError("Campaign requires three named fusion seeds")
    for name, value in fusion.items():
        _integer(value, f"seeds.fusion.{name}")
    candidates = seeds.get("candidate_erm")
    minimum = int(manifest.get("policy", {}).get("minimum_candidate_seeds", -1))
    if (
        not isinstance(candidates, list)
        or len(candidates) < minimum
        or len(set(candidates)) != len(candidates)
    ):
        raise ConfigurationError(
            "Campaign requires at least three unique candidate seeds"
        )
    for index, value in enumerate(candidates):
        _integer(value, f"seeds.candidate_erm[{index}]")
    locked_policy = {
        "minimum_candidate_seeds": 3,
        "clean_git_required": True,
        "overwrite_existing_outputs": False,
        "test_reporting_only": True,
        "hide_test_until_hashed_freeze": True,
        "require_gh200_smoke_per_training_phase": True,
        "require_ula_smoke_before_phase6": True,
    }
    for key, expected in locked_policy.items():
        if manifest.get("policy", {}).get(key) != expected:
            raise ConfigurationError(
                f"Locked campaign policy {key} must be {expected!r}"
            )
    if manifest.get("ula", {}).get("method_label") != "uLA-style":
        raise ConfigurationError("Campaign uLA method label must be uLA-style")


def campaign_environment(manifest: dict[str, Any]) -> dict[str, str]:
    seeds = manifest["seeds"]
    values = {
        "SETV_CAMPAIGN_ROOT": str(manifest["paths"]["campaign_root"]),
        "SETV_OBJECT_SEED": str(seeds["object_expert"]),
        "SETV_EXACT_SEED": str(seeds["exact_background_expert"]),
        "SETV_MASK_SEED": str(seeds["sanitized_mask_bank"]),
        "SETV_SANITIZED_SEED": str(seeds["sanitized_background_expert"]),
        "SETV_SET_SEED": str(seeds["set_background_expert"]),
        "SETV_EXACT_FUSION_SEED": str(seeds["fusion"]["exact"]),
        "SETV_SANITIZED_FUSION_SEED": str(seeds["fusion"]["sanitized"]),
        "SETV_SET_FUSION_SEED": str(seeds["fusion"]["set"]),
        "SETV_CANDIDATE_SEEDS": ",".join(
            str(value) for value in seeds["candidate_erm"]
        ),
        "SETV_ULA_SEED": str(seeds["ula_proxy_and_ssl"]),
        "SETV_ULA_REPO": str(manifest["paths"]["ula_repository"]),
    }
    legacy = manifest["ula"].get("legacy_environment")
    checkpoint = manifest["ula"].get("external_official_ssl_checkpoint")
    if legacy:
        values["SETV_ULA_ENV"] = str(legacy)
    if checkpoint:
        values["SETV_ULA_SSL_CHECKPOINT"] = str(checkpoint)
    if set(values) - ENVIRONMENT_NAMES:
        raise ConfigurationError("Campaign produced an unknown environment name")
    for name, value in values.items():
        if not re.fullmatch(r"SETV_[A-Z0-9_]+", name) or "\n" in value:
            raise ConfigurationError("Unsafe campaign environment value")
    return values


def expected_artifact_paths(manifest: dict[str, Any]) -> dict[str, list[Path]]:
    root = Path(manifest["paths"]["campaign_root"])
    seeds = manifest["seeds"]
    object_dir = root / "object_expert" / f"seed_{seeds['object_expert']}"
    exact_dir = (
        root / "background_exact" / f"seed_{seeds['exact_background_expert']}"
    )
    sanitized_dir = (
        root
        / "background_sanitized"
        / f"seed_{seeds['sanitized_background_expert']}"
    )
    set_dir = (
        root / "background_set" / f"seed_{seeds['set_background_expert']}"
    )
    return {
        "phase0": [root / "phase0"],
        "phase1": [object_dir],
        "phase2": [
            exact_dir,
            root
            / "fusion_exact"
            / (
                f"object_{seeds['object_expert']}_exact_"
                f"{seeds['exact_background_expert']}_fusion_"
                f"{seeds['fusion']['exact']}"
            ),
        ],
        "phase3": [
            root
            / "sanitized_mask_bank"
            / f"seed_{seeds['sanitized_mask_bank']}",
            sanitized_dir,
            root
            / "fusion_sanitized"
            / (
                f"object_{seeds['object_expert']}_sanitized_"
                f"{seeds['sanitized_background_expert']}_fusion_"
                f"{seeds['fusion']['sanitized']}"
            ),
        ],
        "phase4": [
            set_dir,
            root
            / "fusion_set"
            / (
                f"object_{seeds['object_expert']}_set_"
                f"{seeds['set_background_expert']}_fusion_"
                f"{seeds['fusion']['set']}"
            ),
        ],
        "phase5": [
            root / "candidate_erm" / f"seed_{seed}"
            for seed in seeds["candidate_erm"]
        ],
        "phase6": [
            root / "ula_proxy" / f"seed_{seeds['ula_proxy_and_ssl']}",
            root / "phase6",
        ],
    }


def _git(repo: Path, *arguments: str) -> tuple[int, str]:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode, result.stdout.strip()


def _check(
    name: str,
    ok: bool,
    *,
    blocking: bool,
    details: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "ok": bool(ok),
        "blocking": bool(blocking),
        "details": details,
    }


def _generic_artifact_hash_check(root: Path) -> dict[str, Any]:
    manifest_path = root / "artifact_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    files = manifest.get("files")
    if not isinstance(files, dict) or sha256_json(files) != manifest.get(
        "manifest_digest"
    ):
        raise ValueError("artifact manifest digest is invalid")
    for relative, expected in files.items():
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(expected["size_bytes"])
            or sha256_file(path) != expected["sha256"]
        ):
            raise ValueError(f"artifact changed: {relative}")
    return {"artifact_count": len(files)}


def _verify_artifact(stage: str, index: int, path: Path) -> dict[str, Any]:
    if stage == "phase0":
        from setv.phase0 import verify_phase0

        try:
            return verify_phase0(path, require_approval=True)
        except Exception as approval_error:
            approval_path = path / "mask_audit" / "visual_review_approval.json"
            # A present but invalid approval is corruption, not a pending
            # human gate. Preserve the original verification failure.
            if approval_path.exists():
                raise approval_error
            try:
                result = verify_phase0(path, require_approval=False)
            except Exception:
                raise approval_error
            return {
                **result,
                "status": "awaiting_visual_approval",
                "approval_error": str(approval_error),
            }
    if stage == "phase1":
        from setv.experts.train_object import verify_object_expert

        return verify_object_expert(path, load_checkpoint=False)
    if stage == "phase2":
        if index == 0:
            from setv.experts.train_exact import verify_exact_expert

            return verify_exact_expert(path, load_checkpoint=False)
        from setv.fusion.artifacts import verify_fusion_artifacts

        return verify_fusion_artifacts(path)
    if stage == "phase3":
        if index == 0:
            from setv.experts.sanitized_bank import verify_sanitized_mask_bank

            return verify_sanitized_mask_bank(
                path, require_accepted=True, verify_containment=False
            )
        if index == 1:
            from setv.experts.train_sanitized import verify_sanitized_expert

            return verify_sanitized_expert(path, load_checkpoint=False)
        from setv.fusion.sanitized_artifacts import (
            verify_sanitized_fusion_artifacts,
        )

        return verify_sanitized_fusion_artifacts(path)
    if stage == "phase4":
        if index == 0:
            from setv.experts.train_set import verify_set_expert

            return verify_set_expert(path, load_checkpoint=False)
        from setv.fusion.set_artifacts import verify_set_fusion_artifacts

        return verify_set_fusion_artifacts(path)
    if stage == "phase5":
        # Do not deserialize or parse reporting-only test values during
        # readiness. Hash every file and inspect only the frozen selection.
        result = _generic_artifact_hash_check(path)
        selection = json.loads(
            (path / "selection" / "selection_receipt.json").read_text()
        )
        if selection.get("test_metrics_seen") or selection.get(
            "test_metrics_used"
        ):
            raise ValueError("candidate selection receipt indicates test leakage")
        return {**result, "status": "hash_verified_selection_safe"}
    if stage == "phase6" and index == 0:
        from setv.ula.proxy import verify_ula_proxy

        return verify_ula_proxy(path, load_checkpoint=False)
    # A completed Phase 6 target is not read for a new submission decision;
    # its immutable hashes are enough to classify it as an existing output.
    return {
        **_generic_artifact_hash_check(path),
        "status": "hash_verified_existing_phase6",
    }


def _artifact_inventory(manifest: dict[str, Any]) -> dict[str, Any]:
    paths = expected_artifact_paths(manifest)
    inventory = {}
    receipt_names = {
        "phase0": ["artifact_manifest.json"],
        "phase1": ["phase1_receipt.json"],
        "phase2": ["phase2_exact_receipt.json", "fusion_receipt.json"],
        "phase3": [
            "sanitized_mask_bank_receipt.json",
            "phase3_sanitized_receipt.json",
            "fusion_receipt.json",
        ],
        "phase4": ["phase4_set_receipt.json", "fusion_receipt.json"],
        "phase5": ["phase5_receipt.json"] * len(paths["phase5"]),
        "phase6": ["phase6_ula_proxy_receipt.json", "phase6_receipt.json"],
    }
    for stage in STAGES:
        rows = []
        for index, (path, receipt) in enumerate(
            zip(paths[stage], receipt_names[stage])
        ):
            state = "absent"
            verification: dict[str, Any] | None = None
            if path.exists() and not (path / receipt).is_file():
                state = "partial_or_unverified"
            elif (path / receipt).is_file():
                try:
                    verification = _verify_artifact(stage, index, path)
                    state = (
                        "awaiting_human_gate"
                        if verification.get("status")
                        == "awaiting_visual_approval"
                        else "verified"
                    )
                except Exception as exc:
                    state = "invalid"
                    verification = {
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    }
            rows.append(
                {
                    "path": str(path),
                    "exists": path.exists(),
                    "receipt": receipt,
                    "receipt_exists": (path / receipt).is_file(),
                    "state": state,
                    "verification": verification,
                }
            )
        inventory[stage] = rows
    return inventory


def run_campaign_preflight(
    manifest: dict[str, Any],
    *,
    stage: str,
    repository: str | Path,
    check_tigris_filesystem: bool = True,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ConfigurationError(f"Unknown campaign stage: {stage}")
    repo = Path(repository).expanduser().resolve()
    checks = []
    git_code, commit = _git(repo, "rev-parse", "HEAD")
    _, dirty = _git(repo, "status", "--porcelain", "--untracked-files=normal")
    checks.append(
        _check(
            "clean_git_commit",
            git_code == 0 and not dirty,
            blocking=True,
            details={"commit": commit or None, "dirty_entries": dirty.splitlines()},
        )
    )
    required_source = [
        manifest["scientific_plan"],
        manifest["compute_handoff"],
        manifest["runtime_config"],
        "configs/data_waterbirds95.yaml",
        "configs/candidate_erm.yaml",
        "configs/phase6_analysis.yaml",
    ]
    missing_source = [name for name in required_source if not (repo / name).is_file()]
    checks.append(
        _check(
            "required_source_files",
            not missing_source,
            blocking=True,
            details={"missing": missing_source},
        )
    )
    slurm_files = sorted((repo / "slurm").glob("*.sbatch"))
    slurm_errors = []
    for path in slurm_files:
        text = path.read_text(encoding="utf-8")
        if "#SBATCH --account=reu-aisocial" not in text:
            slurm_errors.append(f"{path.name}:account")
        if "#SBATCH --partition=tigris" not in text:
            slurm_errors.append(f"{path.name}:partition")
        syntax = subprocess.run(
            ["bash", "-n", str(path)], capture_output=True, text=True, check=False
        )
        if syntax.returncode:
            slurm_errors.append(f"{path.name}:syntax:{syntax.stderr.strip()}")
    checks.append(
        _check(
            "slurm_contracts",
            bool(slurm_files) and not slurm_errors,
            blocking=True,
            details={"file_count": len(slurm_files), "errors": slurm_errors},
        )
    )
    config_mismatches = []
    runtime_path = repo / manifest["runtime_config"]
    if runtime_path.is_file():
        runtime = yaml.safe_load(runtime_path.read_text())
        for key, manifest_key in (
            ("repository", "repository"),
            ("logs_root", "campaign_root"),
            ("environment", "main_environment"),
        ):
            if str(runtime.get(key)) != str(manifest["paths"][manifest_key]):
                config_mismatches.append(f"runtime.{key}")
    data_path = repo / "configs" / "data_waterbirds95.yaml"
    if data_path.is_file():
        data = yaml.safe_load(data_path.read_text())
        if data["split"]["seed"] != manifest["seeds"]["split"]:
            config_mismatches.append("data.split.seed")
        if data["audit"]["visual_seed"] != manifest["seeds"]["visual_audit"]:
            config_mismatches.append("data.audit.visual_seed")
        if data["data"]["dataset_root"] != manifest["paths"]["dataset_root"]:
            config_mismatches.append("data.dataset_root")
        if data["masks"]["root"] != manifest["paths"]["mask_root"]:
            config_mismatches.append("data.mask_root")
    checks.append(
        _check(
            "manifest_config_alignment",
            not config_mismatches,
            blocking=True,
            details={"mismatches": config_mismatches},
        )
    )
    if check_tigris_filesystem:
        filesystem = {
            name: Path(manifest["paths"][name]).exists()
            for name in (
                "repository",
                "campaign_root",
                "dataset_root",
                "mask_root",
                "ula_repository",
                "main_environment",
            )
        }
        required_now = {"repository", "campaign_root", "main_environment"}
        if stage == "phase0":
            required_now.update({"dataset_root", "mask_root"})
        if stage == "phase6":
            required_now.add("ula_repository")
        missing_now = sorted(name for name in required_now if not filesystem[name])
        checks.append(
            _check(
                "tigris_filesystem",
                not missing_now,
                blocking=True,
                details={"paths_exist": filesystem, "required_missing": missing_now},
            )
        )
    inventory = _artifact_inventory(manifest)
    stage_index = STAGES.index(stage)
    prerequisite_stages = STAGES[:stage_index]
    missing_prerequisites = [
        prior
        for prior in prerequisite_stages
        if any(row["state"] != "verified" for row in inventory[prior])
    ]
    # Phases 2, 3, and 4 are independent siblings after Phase 1.
    if stage in {"phase3", "phase4"}:
        missing_prerequisites = [
            prior for prior in ("phase0", "phase1")
            if any(row["state"] != "verified" for row in inventory[prior])
        ]
    if stage == "phase5":
        missing_prerequisites = [
            prior
            for prior in ("phase0", "phase1", "phase2", "phase3", "phase4")
            if any(row["state"] != "verified" for row in inventory[prior])
        ]
    if stage == "phase6":
        missing_prerequisites = [
            prior
            for prior in ("phase0", "phase1", "phase2", "phase3", "phase4", "phase5")
            if any(row["state"] != "verified" for row in inventory[prior])
        ]
    target_states = [row["state"] for row in inventory[stage]]
    target_absent = all(value == "absent" for value in target_states)
    checks.append(
        _check(
            "artifact_stage_boundary",
            not missing_prerequisites and target_absent,
            blocking=True,
            details={
                "missing_prerequisite_stages": missing_prerequisites,
                "target_states": target_states,
                "target_output_exists_requires_operator": not target_absent,
                "partial_output_requires_recovery": any(
                    value not in {"absent", "verified"} for value in target_states
                ),
            },
        )
    )
    if stage == "phase6":
        ula = manifest["ula"]
        legacy_environment = (
            ula.get("legacy_environment") or os.environ.get("SETV_ULA_ENV")
        )
        external_checkpoint = (
            ula.get("external_official_ssl_checkpoint")
            or os.environ.get("SETV_ULA_SSL_CHECKPOINT")
        )
        legacy_python = (
            Path(legacy_environment) / "bin" / "python"
            if legacy_environment
            else None
        )
        legacy_usable = bool(
            legacy_python
            and legacy_python.is_file()
            and os.access(legacy_python, os.X_OK)
        )
        checkpoint_usable = bool(
            external_checkpoint and Path(external_checkpoint).is_file()
        )
        configured = legacy_usable or checkpoint_usable
        checks.append(
            _check(
                "ula_execution_source",
                configured,
                blocking=True,
                details={
                    "policy": ula["policy"],
                    "legacy_environment_configured": bool(legacy_environment),
                    "external_checkpoint_configured": bool(external_checkpoint),
                    "legacy_environment_usable": legacy_usable,
                    "external_checkpoint_usable": checkpoint_usable,
                    "runtime_override_used": bool(
                        (
                            not ula.get("legacy_environment")
                            and os.environ.get("SETV_ULA_ENV")
                        )
                        or (
                            not ula.get("external_official_ssl_checkpoint")
                            and os.environ.get("SETV_ULA_SSL_CHECKPOINT")
                        )
                    ),
                },
            )
        )
    ready = all(item["ok"] for item in checks if item["blocking"])
    if ready:
        next_action = f"bash scripts/submit_{stage}.sh"
        if stage == "phase1":
            next_action = "bash scripts/submit_phase1_object.sh"
        elif stage == "phase2":
            next_action = "bash scripts/submit_phase2_exact.sh"
        elif stage == "phase3":
            next_action = "bash scripts/submit_phase3_sanitized.sh"
        elif stage == "phase4":
            next_action = "bash scripts/submit_phase4_set.sh"
        elif stage == "phase5":
            next_action = "bash scripts/submit_phase5_candidate.sh"
    elif stage == "phase0" and target_states == ["awaiting_human_gate"]:
        next_action = (
            "Inspect every Phase 0 contact sheet, then run "
            "scripts/approve_mask_audit.py; do not rebuild Phase 0."
        )
    elif target_states and all(value == "verified" for value in target_states):
        next_action = (
            f"{stage} is already verified; do not resubmit or overwrite it."
        )
    elif any(value != "absent" for value in target_states):
        next_action = (
            "Existing target artifacts require operator recovery; do not "
            "delete, overwrite, or rerun the top-level launcher."
        )
    else:
        next_action = "Resolve blocking checks; do not submit this stage."
    source_hashes = {
        name: sha256_file(repo / name)
        for name in required_source
        if (repo / name).is_file()
    }
    return {
        "schema_version": 1,
        "status": "ready" if ready else "blocked",
        "ready": ready,
        "stage": stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": manifest["campaign_id"],
        "campaign_manifest": {
            "path": manifest["_manifest_path"],
            "sha256": sha256_file(manifest["_manifest_path"]),
            "resolved_digest": sha256_json(
                {
                    key: value
                    for key, value in manifest.items()
                    if key != "_manifest_path"
                }
            ),
        },
        "repository": str(repo),
        "commit": commit or None,
        "host": {
            "architecture": platform.machine(),
            "python": platform.python_version(),
        },
        "checks": checks,
        "artifact_inventory": inventory,
        "source_hashes": source_hashes,
        "reporting_only_metric_values_included": False,
        "preflight_used_for_method_selection": False,
        "next_action": next_action,
    }
