#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# < 2 || $# > 3 )); then
  echo "Usage: $0 PHASE REPORT.json [--resume-rejected-sanitized]" >&2
  exit 2
fi
stage=$1
report=$2
resume_flag=${3:-}
if [[ -n "$resume_flag" && "$resume_flag" != "--resume-rejected-sanitized" ]]; then
  echo "Unknown preflight option: $resume_flag" >&2
  exit 2
fi
if [[ -n "$resume_flag" ]]; then
  extra_args=("$resume_flag")
else
  extra_args=()
fi
if [[ ! "$stage" =~ ^phase[0-6]$ ]]; then
  echo "Invalid campaign stage: $stage" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SETV_REPO=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${SCRIPT_DIR}/load_campaign_env.sh"

SETV_PREFLIGHT_PYTHON=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200/bin/python
if [[ ! -x "$SETV_PREFLIGHT_PYTHON" ]]; then
  echo "Campaign preflight requires the Tigris main environment: $SETV_PREFLIGHT_PYTHON" >&2
  exit 2
fi

"$SETV_PREFLIGHT_PYTHON" "${SCRIPT_DIR}/preflight_campaign.py" \
  --config "$SETV_CAMPAIGN_CONFIG" \
  --stage "$stage" \
  --repository "$SETV_REPO" \
  --report "$report" \
  "${extra_args[@]}"

echo "Campaign preflight passed: $report"
