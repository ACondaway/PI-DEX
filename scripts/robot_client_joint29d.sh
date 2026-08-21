#!/usr/bin/env bash
# GPU host: OpenPI Runtime + ActionChunkBroker + North Zenoh (PI-DEX only on GPU).
#
#   1) Slave NUC: bash start.sh          # Sharpa only — no pi-dex on NUC
#   2) GPU: bash scripts/serve_joint29d.sh ...
#   3) GPU: bash scripts/robot_client_joint29d.sh --serve-host 127.0.0.1 ...
#   4) Pendant: F6 inference, F2 moving
#
# Bridge joins the robot Zenoh domain over the network (pass --zenoh-config if needed).

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
: "${ACTION_MODE:=absolute}"
: "${OUTPUT_CHUNK:=}"
: "${OFFSET:=0}"
: "${FIRST_CHUNK_SMOOTH:=0}"
: "${ZENOH_CONFIG:=}"
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
    --action-mode) ACTION_MODE="$2"; shift 2 ;;
    --output-chunk) OUTPUT_CHUNK="$2"; shift 2 ;;
    --offset) OFFSET="$2"; shift 2 ;;
    --first-chunk-smooth) FIRST_CHUNK_SMOOTH="$2"; shift 2 ;;
    --zenoh-config) ZENOH_CONFIG="$2"; shift 2 ;;
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
  --action-mode "${ACTION_MODE}"
  --offset "${OFFSET}"
  --first-chunk-smooth "${FIRST_CHUNK_SMOOTH}"
)
if [[ -n "${PROMPT}" ]]; then
  ARGS+=(--prompt "${PROMPT}")
fi
if [[ -n "${API_KEY}" ]]; then
  ARGS+=(--api-key "${API_KEY}")
fi
if [[ -n "${OUTPUT_CHUNK}" ]]; then
  ARGS+=(--output-chunk "${OUTPUT_CHUNK}")
fi
if [[ -n "${ZENOH_CONFIG}" ]]; then
  ARGS+=(--zenoh-config "${ZENOH_CONFIG}")
fi

echo "[$(date -Is)] GPU pi-dex-robot-client → serve ${SERVE_HOST}:${SERVE_PORT}"
echo "  topics ${OBS_TOPIC} → ${ACTION_TOPIC} (robot Zenoh domain)"
echo "  action_mode=${ACTION_MODE} offset=${OFFSET} output_chunk=${OUTPUT_CHUNK:-auto}"
echo "  Reminder: NUC start.sh only; F6=inference, F2=moving"
exec pi-dex-robot-client "${ARGS[@]}" "${EXTRA[@]}"
