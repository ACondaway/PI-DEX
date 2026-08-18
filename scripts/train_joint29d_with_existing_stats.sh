#!/usr/bin/env bash
# Train joint_29d with already-computed Sharpa norm stats.
#
# IMPORTANT: full OpenData stats (assets-opendata/sharpa_joint_29d_opendata_v0)
# may still be computing. This script defaults to the existing ClearPlate
# formal-norm assets (16-sample smoke stats). Do NOT mix those stats with the
# full OpenData root.
#
# Usage:
#   bash scripts/train_joint29d_with_existing_stats.sh
#   bash scripts/train_joint29d_with_existing_stats.sh --max-steps 100
#   STATS_PRESET=opendata MAX_STEPS=1000 bash scripts/train_joint29d_with_existing_stats.sh
#
# Background:
#   nohup bash scripts/train_joint29d_with_existing_stats.sh \
#     > /mnt/netdata/Team/Personal/congsheng/pi-dex-artifacts/logs/train_joint29d.nohup.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

: "${PI_DEX_ARTIFACTS:=/mnt/netdata/Team/Personal/congsheng/pi-dex-artifacts}"
: "${OPENPI_DATA_HOME:=${PI_DEX_ARTIFACTS}/openpi-data}"
: "${CONTRACT:=${ROOT}/configs/site/joint_29d_observation.reviewed.json}"
: "${CONVERTED_BASE:=${PI_DEX_ARTIFACTS}/converted/pi05_base-pytorch-bfloat16-K8}"
: "${EXPECTED_BASE_SHA256:=2f8539e2308611ea6fff84a5d7774f80d7c177c624769ff842008cf85dea9eeb}"

# clearplate = existing formal-norm (ready now)
# opendata   = full-corpus stats (only after prepare_opendata_full finishes)
: "${STATS_PRESET:=clearplate}"
: "${MAX_STEPS:=100}"
: "${BATCH_SIZE:=1}"
: "${LEARNING_RATE:=1e-5}"
: "${SEED:=0}"
: "${DEVICE:=cuda}"
: "${DTYPE:=bfloat16}"
: "${RUN_ID:=}"

# optional CLI overrides: --max-steps N --batch-size N --run-id NAME
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --stats-preset) STATS_PRESET="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

export OPENPI_DATA_HOME
export PATH="${PATH:-}"

case "${STATS_PRESET}" in
  clearplate)
    DATASET_ROOT="${DATASET_ROOT:-/mnt/netdata/Team/Academic/Data/North/SharpaOpenData/ClearPlate}"
    ASSETS_DIR="${ASSETS_DIR:-${PI_DEX_ARTIFACTS}/runs/formal-norm/assets}"
    ASSET_ID="${ASSET_ID:-sharpa_joint_29d}"
    # Keep data scale roughly aligned with the 16-sample smoke stats used in formal-norm.
    MAX_EPISODES="${MAX_EPISODES:-2}"
    DEFAULT_RUN_PREFIX="clearplate-joint29d"
    ;;
  opendata)
    DATASET_ROOT="${DATASET_ROOT:-/mnt/netdata/Team/Academic/Data/North/SharpaOpenData}"
    ASSETS_DIR="${ASSETS_DIR:-${PI_DEX_ARTIFACTS}/assets-opendata}"
    ASSET_ID="${ASSET_ID:-sharpa_joint_29d_opendata_v0}"
    MAX_EPISODES="${MAX_EPISODES:-}"
    DEFAULT_RUN_PREFIX="opendata-joint29d"
    ;;
  *)
    echo "STATS_PRESET must be clearplate or opendata, got: ${STATS_PRESET}" >&2
    exit 2
    ;;
esac

NORM_PATH="${ASSETS_DIR}/${ASSET_ID}/norm_stats.json"
if [[ ! -f "${NORM_PATH}" ]]; then
  echo "missing norm stats: ${NORM_PATH}" >&2
  if [[ "${STATS_PRESET}" == "opendata" ]]; then
    echo "full OpenData stats are not ready yet; wait for scripts/prepare_opendata_full.sh" >&2
    echo "or use: STATS_PRESET=clearplate bash $0" >&2
  fi
  exit 1
fi

if [[ ! -f "${CONVERTED_BASE}/model.safetensors" ]]; then
  echo "missing converted base: ${CONVERTED_BASE}/model.safetensors" >&2
  exit 1
fi

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="${DEFAULT_RUN_PREFIX}-$(date +%Y%m%d-%H%M%S)"
fi
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PI_DEX_ARTIFACTS}/runs/${RUN_ID}}"
OUTPUT_JSON="${OUTPUT_JSON:-${PI_DEX_ARTIFACTS}/runs/${RUN_ID}.json}"
mkdir -p "${PI_DEX_ARTIFACTS}/runs" "${PI_DEX_ARTIFACTS}/logs"

EXTRA_ARGS=()
if [[ -n "${MAX_EPISODES}" ]]; then
  EXTRA_ARGS+=(--max-episodes "${MAX_EPISODES}")
fi

echo "[$(date -Is)] train start"
echo "  STATS_PRESET=${STATS_PRESET}"
echo "  DATASET_ROOT=${DATASET_ROOT}"
echo "  ASSETS_DIR=${ASSETS_DIR}"
echo "  ASSET_ID=${ASSET_ID}"
echo "  CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "  MAX_STEPS=${MAX_STEPS} BATCH_SIZE=${BATCH_SIZE}"

pi-dex-train-pytorch \
  --action-representation joint_29d \
  --runner pi_dex.training_runner:run -- \
  --mode train \
  --observation-contract "${CONTRACT}" \
  --dataset-root "${DATASET_ROOT}" \
  --split train \
  --assets-dir "${ASSETS_DIR}" \
  --asset-id "${ASSET_ID}" \
  --pytorch-weight-path "${CONVERTED_BASE}" \
  --expected-base-sha256 "${EXPECTED_BASE_SHA256}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --run-id "${RUN_ID}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --batch-size "${BATCH_SIZE}" \
  --max-steps "${MAX_STEPS}" \
  --learning-rate "${LEARNING_RATE}" \
  --seed "${SEED}" \
  --save-interval "${SAVE_INTERVAL:-0}" \
  --log-interval "${LOG_INTERVAL:-1}" \
  --no-wandb \
  --output-json "${OUTPUT_JSON}" \
  "${EXTRA_ARGS[@]}"

echo "[$(date -Is)] train done -> ${CHECKPOINT_DIR}"
echo "  summary -> ${OUTPUT_JSON}"
