#!/usr/bin/env bash
# Start PI-DEX joint_29d WebSocket model server (robot keeps Zenoh/SDK locally).
#
# Example (loopback):
#   bash scripts/serve_joint29d.sh \
#     --checkpoint-dir /path/to/run/10000 \
#     --assets-dir /path/to/assets-Insert_Battery \
#     --asset-id sharpa_joint_29d_insert_battery \
#     --robot-id POC22005

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

: "${PI_DEX_ARTIFACTS:=/mnt/netdata/Team/Personal/congsheng/pi-dex-artifacts}"
: "${CONDA_ROOT:=/mnt/netdata/Team/Personal/congsheng/miniconda}"
: "${PI_DEX_CONDA_ENV:=pi-dex}"
: "${CONTRACT:=${ROOT}/configs/site/joint_29d_observation.reviewed.json}"
: "${HOST:=127.0.0.1}"
: "${PORT:=8000}"
: "${ASSET_ID:=sharpa_joint_29d_insert_battery}"
: "${ASSETS_DIR:=${PI_DEX_ARTIFACTS}/assets-Insert_Battery}"
: "${ROBOT_ID:=POC22005}"
: "${CHECKPOINT_DIR:=}"
: "${API_KEY:=}"
: "${ACTION_MODE:=absolute}"
: "${SKIP_CONDA:=0}"

if [[ "${SKIP_CONDA}" != "1" && -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${PI_DEX_CONDA_ENV}"
fi

EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --assets-dir) ASSETS_DIR="$2"; shift 2 ;;
    --asset-id) ASSET_ID="$2"; shift 2 ;;
    --robot-id) ROBOT_ID="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --contract) CONTRACT="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --action-mode) ACTION_MODE="$2"; shift 2 ;;
    --)
      shift
      EXTRA+=("$@")
      break
      ;;
    *)
      EXTRA+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${CHECKPOINT_DIR}" ]]; then
  echo "--checkpoint-dir is required (path to a published step dir with model.safetensors)" >&2
  exit 2
fi

ARGS=(
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --observation-contract "${CONTRACT}"
  --assets-dir "${ASSETS_DIR}"
  --asset-id "${ASSET_ID}"
  --robot-id "${ROBOT_ID}"
  --host "${HOST}"
  --port "${PORT}"
  --action-mode "${ACTION_MODE}"
)
if [[ -n "${API_KEY}" ]]; then
  ARGS+=(--api-key "${API_KEY}")
fi

echo "[$(date -Is)] pi-dex-serve ${HOST}:${PORT}"
echo "  checkpoint=${CHECKPOINT_DIR}"
echo "  assets=${ASSETS_DIR}/${ASSET_ID} action_mode=${ACTION_MODE}"
exec pi-dex-serve "${ARGS[@]}" "${EXTRA[@]}"
