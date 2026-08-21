#!/usr/bin/env bash
# GPU host one-shot: pi-dex-serve + pi-dex-robot-client (all PI-DEX on GPU).
#
#   1) NUC: bash start.sh                 # Sharpa only
#   2) GPU: bash scripts/run_robot_joint29d.sh --checkpoint-dir ... [flags]
#   3) Pendant: F6 inference, F2 moving
#
# Starts the WebSocket server in the background, waits until it answers, then
# runs the OpenPI Runtime Zenoh bridge in the foreground. Ctrl+C stops both.

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
: "${PROMPT:=}"
: "${ACTION_MODE:=absolute}"
: "${OUTPUT_CHUNK:=}"
: "${OFFSET:=0}"
: "${FIRST_CHUNK_SMOOTH:=0}"
: "${ZENOH_CONFIG:=}"
: "${OBS_TOPIC:=north_observation}"
: "${ACTION_TOPIC:=inference/action}"
: "${SERVE_READY_TIMEOUT_S:=180}"
: "${SKIP_CONDA:=0}"

if [[ "${SKIP_CONDA}" != "1" && -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${PI_DEX_CONDA_ENV}"
fi

SERVE_EXTRA=()
CLIENT_EXTRA=()
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
    --prompt) PROMPT="$2"; shift 2 ;;
    --action-mode) ACTION_MODE="$2"; shift 2 ;;
    --output-chunk) OUTPUT_CHUNK="$2"; shift 2 ;;
    --offset) OFFSET="$2"; shift 2 ;;
    --first-chunk-smooth) FIRST_CHUNK_SMOOTH="$2"; shift 2 ;;
    --zenoh-config) ZENOH_CONFIG="$2"; shift 2 ;;
    --observation-topic) OBS_TOPIC="$2"; shift 2 ;;
    --action-topic) ACTION_TOPIC="$2"; shift 2 ;;
    --serve-ready-timeout) SERVE_READY_TIMEOUT_S="$2"; shift 2 ;;
    --serve-extra)
      shift
      while [[ $# -gt 0 && "$1" != "--" && "$1" != "--client-extra" ]]; do
        SERVE_EXTRA+=("$1")
        shift
      done
      ;;
    --client-extra)
      shift
      while [[ $# -gt 0 && "$1" != "--" ]]; do
        CLIENT_EXTRA+=("$1")
        shift
      done
      ;;
    --)
      shift
      CLIENT_EXTRA+=("$@")
      break
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${CHECKPOINT_DIR}" ]]; then
  echo "--checkpoint-dir is required" >&2
  exit 2
fi

SERVE_PID=""
cleanup() {
  if [[ -n "${SERVE_PID}" ]] && kill -0 "${SERVE_PID}" 2>/dev/null; then
    echo "[$(date -Is)] stopping pi-dex-serve pid=${SERVE_PID}"
    kill "${SERVE_PID}" 2>/dev/null || true
    wait "${SERVE_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

SERVE_ARGS=(
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
  SERVE_ARGS+=(--api-key "${API_KEY}")
fi

echo "[$(date -Is)] starting pi-dex-serve ${HOST}:${PORT}"
echo "  checkpoint=${CHECKPOINT_DIR}"
echo "  assets=${ASSETS_DIR}/${ASSET_ID} action_mode=${ACTION_MODE}"
pi-dex-serve "${SERVE_ARGS[@]}" "${SERVE_EXTRA[@]}" &
SERVE_PID=$!

echo "[$(date -Is)] waiting for serve (timeout ${SERVE_READY_TIMEOUT_S}s)..."
deadline=$((SECONDS + SERVE_READY_TIMEOUT_S))
ready=0
while (( SECONDS < deadline )); do
  if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
    echo "pi-dex-serve exited before becoming ready" >&2
    wait "${SERVE_PID}" || true
    exit 1
  fi
  if pi-dex-serve-probe --host "${HOST}" --port "${PORT}" ${API_KEY:+--api-key "${API_KEY}"} >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "${ready}" != "1" ]]; then
  echo "timed out waiting for pi-dex-serve on ${HOST}:${PORT}" >&2
  exit 1
fi
echo "[$(date -Is)] serve ready"

CLIENT_ARGS=(
  --mode bridge
  --observation-contract "${CONTRACT}"
  --serve-host "${HOST}"
  --serve-port "${PORT}"
  --robot-id "${ROBOT_ID}"
  --observation-topic "${OBS_TOPIC}"
  --action-topic "${ACTION_TOPIC}"
  --action-mode "${ACTION_MODE}"
  --offset "${OFFSET}"
  --first-chunk-smooth "${FIRST_CHUNK_SMOOTH}"
)
if [[ -n "${PROMPT}" ]]; then
  CLIENT_ARGS+=(--prompt "${PROMPT}")
fi
if [[ -n "${API_KEY}" ]]; then
  CLIENT_ARGS+=(--api-key "${API_KEY}")
fi
if [[ -n "${OUTPUT_CHUNK}" ]]; then
  CLIENT_ARGS+=(--output-chunk "${OUTPUT_CHUNK}")
fi
if [[ -n "${ZENOH_CONFIG}" ]]; then
  CLIENT_ARGS+=(--zenoh-config "${ZENOH_CONFIG}")
fi

echo "[$(date -Is)] starting pi-dex-robot-client → ${HOST}:${PORT}"
echo "  topics ${OBS_TOPIC} → ${ACTION_TOPIC}"
echo "  offset=${OFFSET} output_chunk=${OUTPUT_CHUNK:-auto}"
echo "  Reminder: NUC start.sh only; F6=inference, F2=moving"
pi-dex-robot-client "${CLIENT_ARGS[@]}" "${CLIENT_EXTRA[@]}"
