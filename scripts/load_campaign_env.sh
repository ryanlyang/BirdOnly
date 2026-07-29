#!/usr/bin/env bash
# Source this file; it validates the frozen campaign manifest and exports seeds.

SETV_LOADER_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SETV_LOADER_REPO=$(cd -- "${SETV_LOADER_DIR}/.." && pwd)
SETV_CAMPAIGN_CONFIG=${SETV_CAMPAIGN_CONFIG:-"${SETV_LOADER_REPO}/configs/campaign_waterbirds95.yaml"}
SETV_MANIFEST_PYTHON=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200/bin/python
if [[ ! -x "$SETV_MANIFEST_PYTHON" ]]; then
  SETV_MANIFEST_PYTHON=python3
fi
if ! SETV_CAMPAIGN_OUTPUT=$(
  "$SETV_MANIFEST_PYTHON" "${SETV_LOADER_REPO}/scripts/campaign_manifest.py" \
    --config "$SETV_CAMPAIGN_CONFIG" \
    --emit-env
); then
  echo "Failed to validate and load campaign manifest: $SETV_CAMPAIGN_CONFIG" >&2
  unset SETV_LOADER_DIR SETV_LOADER_REPO SETV_MANIFEST_PYTHON
  unset SETV_CAMPAIGN_OUTPUT
  return 2 2>/dev/null || exit 2
fi
mapfile -t SETV_CAMPAIGN_ASSIGNMENTS <<< "$SETV_CAMPAIGN_OUTPUT"
for SETV_CAMPAIGN_ASSIGNMENT in "${SETV_CAMPAIGN_ASSIGNMENTS[@]}"; do
  export "$SETV_CAMPAIGN_ASSIGNMENT"
done
export SETV_CAMPAIGN_CONFIG
unset SETV_CAMPAIGN_ASSIGNMENT SETV_CAMPAIGN_ASSIGNMENTS
unset SETV_CAMPAIGN_OUTPUT SETV_LOADER_DIR SETV_LOADER_REPO SETV_MANIFEST_PYTHON
