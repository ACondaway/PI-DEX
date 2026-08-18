#!/usr/bin/env bash
# Robot-side Zenoh bridge: NorthObservation → pi-dex-serve → inference/action.
#
# Typical site layout:
#   1) Slave NUC: bash start.sh   # or start-nuc.sh + start-remote-orin.sh
#   2) GPU host:  bash scripts/serve_joint29d.sh --checkpoint-dir ...
#   3) Slave NUC: bash scripts/robot_client_joint29d.sh --serve-host <gpu-ip>
#   4) Pendant: F6 → inference mode, F2 → moving
#
# This script does not replace sharpa start.sh; it only fills the missing
# inference publisher that the robot stack expects on topic inference/action.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

: "${CONDA_ROOT:=/mnt/netdata/Team/Personal/congsheng/miniconda}"
: "${PI_DEX_CONDA_ENV:=pi-dex}"
: "${CONTRACT:=${ROOT}/configs/site/joint_29d_observation.reviewed.json}"
: "${SERVE_HOST:=127.0.0.1}"
: "${SERVE_PORT:=8000}"
: "${ROBOT_ID:=POC22005}"
: "${PROMPT:=}"
: "${API_KEY:=}"
: "${OBS_TOPIC:=north_observation}"
: "${ACTION_TOPIC:=inference/action}"
: "${SKIP_CONDA:=0}"

if [[ "${SKIP_CONDA}" != "1" && -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${PI_DEX_CONDA_ENV}"
fi

EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --serve-host) SERVE_HOST="$2"; shift 2 ;;
    --serve-port) SERVE_PORT="$2"; shift 2 ;;
    --robot-id) ROBOT_ID="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --contract) CONTRACT="$2"; shift 2 ;;
    --observation-topic) OBS_TOPIC="$2"; shift 2 ;;
    --action-topic) ACTION_TOPIC="$2"; shift 2 ;;
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

ARGS=(
  --mode bridge
  --observation-contract "${CONTRACT}"
  --serve-host "${SERVE_HOST}"
  --serve-port "${SERVE_PORT}"
  --robot-id "${ROBOT_ID}"
  --observation-topic "${OBS_TOPIC}"
  --action-topic "${ACTION_TOPIC}"
)
if [[ -n "${PROMPT}" ]]; then
  ARGS+=(--prompt "${PROMPT}")
fi
if [[ -n "${API_KEY}" ]]; then
  ARGS+=(--api-key "${API_KEY}")
fi

echo "[$(date -Is)] pi-dex-robot-client → ${SERVE_HOST}:${SERVE_PORT}"
echo "  topics ${OBS_TOPIC} → ${ACTION_TOPIC}"
echo "  Reminder: robot start.sh first; F6=inference, F2=moving"
exec pi-dex-robot-client "${ARGS[@]}" "${EXTRA[@]}"
