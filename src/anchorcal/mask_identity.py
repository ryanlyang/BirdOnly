"""Selector-safe aggregate identity for the frozen Waterbirds VLM masks.

This module deliberately contains no per-row manifest loader.  Selector-only
code may import it without gaining an import path to mask membership records.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .io import hash_object


VLM_MASK_MANIFEST_SCHEMA = "anchorcal-vlm-mask-manifest-v3"
VLM_PRODUCER = "waterbirds100_openclip_laion_dinovit_weclipplus_prediction_cmap"
SELECTOR_MASK_RECEIPT_SCHEMA = "anchorcal-selector-mask-receipt-v1"


def vlm_mask_contract_hash(config: Mapping[str, Any]) -> str:
    """Hash aggregate mask identity without opening any per-row artifact."""

    paths = config["paths"]
    return hash_object(
        {
            "schema_version": VLM_MASK_MANIFEST_SCHEMA,
            "waterbirds_root": str(Path(str(paths["waterbirds_root"])).resolve()),
            "metadata_path": str(Path(str(paths["metadata_path"])).resolve()),
            "vlm_mask_root": str(Path(str(paths["vlm_mask_root"])).resolve()),
            "masks": dict(config["masks"]),
        }
    )
