#!/usr/bin/env bash
# Generic joint_29d training launcher for 1..N nodes × 1..M GPUs.
#
# Topology (pick one):
#   1) Manual torchrun:
#        NNODES=2 NPROC_PER_NODE=8 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
#          bash scripts/train_ddp.sh --dataset-root ... --assets-dir ... --asset-id ...
#   2) Single node (default NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1):
#        NPROC_PER_NODE=8 bash scripts/train_ddp.sh ...
#   3) Single process (no torchrun) when world_size==1:
#        NPROC_PER_NODE=1 bash scripts/train_ddp.sh ...
#   4) Volcano MLP (reads MLP_*):
#        USE_VOLC=1 bash scripts/train_ddp.sh ...
#
# Dataset / assets / checkpoint can be set via env or long options below.
# Extra runner flags after a bare `--` are appended as-is.
#
# Wandb (default on):
#   export WANDB_API_KEY=...
#   WANDB_PROJECT=pi-dex bash scripts/train_ddp.sh ...
# Offline smoke: --no-wandb

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

: "${PI_DEX_ARTIFACTS:=/mnt/netdata/Team/Personal/congsheng/pi-dex-artifacts}"
: "${OPENPI_DATA_HOME:=${PI_DEX_ARTIFACTS}/openpi-data}"
: "${CONTRACT:=${ROOT}/configs/site/joint_29d_observation.reviewed.json}"
: "${CONVERTED_BASE:=${PI_DEX_ARTIFACTS}/converted/pi05_base-pytorch-bfloat16-K8}"
: "${EXPECTED_BASE_SHA256:=2f8539e2308611ea6fff84a5d7774f80d7c177c624769ff842008cf85dea9eeb}"

: "${NNODES:=1}"
: "${NPROC_PER_NODE:=}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"
: "${USE_VOLC:=0}"
: "${RDZV_ID:=pi_dex_ddp}"

: "${DATASET_ROOT:=}"
: "${ASSETS_DIR:=}"
: "${ASSET_ID:=}"
: "${ROBOT_ID:=POC22027}"
: "${CHECKPOINT_DIR:=}"
: "${RUN_ID:=}"
: "${MAX_STEPS:=1000}"
: "${BATCH_SIZE:=1}"
: "${LEARNING_RATE:=1e-5}"
: "${SEED:=0}"
: "${DEVICE:=cuda}"
: "${DTYPE:=bfloat16}"
: "${SPLIT:=train}"
: "${MAX_EPISODES:=}"
: "${RESUME_FROM:=}"
: "${OUTPUT_JSON:=}"
: "${DRY_RUN:=0}"
: "${SAVE_INTERVAL:=500}"
: "${LOG_INTERVAL:=10}"
: "${WANDB:=1}"
: "${WANDB_PROJECT:=pi-dex}"
: "${WANDB_ENTITY:=}"
: "${WANDB_RUN_NAME:=}"
# Auth: export WANDB_API_KEY=... (required when WANDB=1; no interactive login)
: "${WANDB_API_KEY:=}"

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nnodes) NNODES="$2"; shift 2 ;;
    --nproc-per-node) NPROC_PER_NODE="$2"; shift 2 ;;
    --node-rank) NODE_RANK="$2"; shift 2 ;;
    --master-addr) MASTER_ADDR="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    --use-volc) USE_VOLC=1; shift ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --assets-dir) ASSETS_DIR="$2"; shift 2 ;;
    --asset-id) ASSET_ID="$2"; shift 2 ;;
    --robot-id) ROBOT_ID="$2"; shift 2 ;;
    --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --max-episodes) MAX_EPISODES="$2"; shift 2 ;;
    --resume-from) RESUME_FROM="$2"; shift 2 ;;
    --output-json) OUTPUT_JSON="$2"; shift 2 ;;
    --contract) CONTRACT="$2"; shift 2 ;;
    --pytorch-weight-path) CONVERTED_BASE="$2"; shift 2 ;;
    --expected-base-sha256) EXPECTED_BASE_SHA256="$2"; shift 2 ;;
    --save-interval) SAVE_INTERVAL="$2"; shift 2 ;;
    --log-interval) LOG_INTERVAL="$2"; shift 2 ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
    --wandb-run-name) WANDB_RUN_NAME="$2"; shift 2 ;;
    --wandb) WANDB=1; shift ;;
    --no-wandb) WANDB=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    -h|--help)
      sed -n '1,25p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $1 (pass extras after --)" >&2
      exit 2
      ;;
  esac
done

export OPENPI_DATA_HOME

if [[ -z "${DATASET_ROOT}" || -z "${ASSETS_DIR}" || -z "${ASSET_ID}" ]]; then
  echo "require --dataset-root, --assets-dir, --asset-id (or matching env vars)" >&2
  exit 2
fi

if [[ "${WANDB}" == "1" && -z "${WANDB_API_KEY}" ]]; then
  echo "WANDB=1 requires WANDB_API_KEY in the environment (export WANDB_API_KEY=...)" >&2
  echo "or pass --no-wandb for offline smoke" >&2
  exit 2
fi
if [[ "${WANDB}" == "1" ]]; then
  export WANDB_API_KEY
  export WANDB_SILENT="${WANDB_SILENT:-true}"
fi

NORM_PATH="${ASSETS_DIR}/${ASSET_ID}/norm_stats.json"
if [[ ! -f "${NORM_PATH}" ]]; then
  echo "missing norm stats: ${NORM_PATH}" >&2
  echo "run scripts/prepare_task_dataset.sh first" >&2
  exit 1
fi
if [[ ! -d "${CONVERTED_BASE}" ]]; then
  echo "missing converted base: ${CONVERTED_BASE}" >&2
  exit 1
fi

if [[ -z "${CHECKPOINT_DIR}" ]]; then
  CHECKPOINT_DIR="${PI_DEX_ARTIFACTS}/runs/${ASSET_ID}-$(date +%Y%m%d-%H%M%S)"
fi
if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="$(basename "${CHECKPOINT_DIR}")"
fi
if [[ -z "${OUTPUT_JSON}" ]]; then
  OUTPUT_JSON="${CHECKPOINT_DIR%/}.json"
fi

mkdir -p "$(dirname "${CHECKPOINT_DIR}")" "$(dirname "${OUTPUT_JSON}")" "${PI_DEX_ARTIFACTS}/logs"

TRAIN_ARGS=(
  --action-representation joint_29d
  --runner pi_dex.training_runner:run
  --
  --mode train
  --observation-contract "${CONTRACT}"
  --dataset-root "${DATASET_ROOT}"
  --split "${SPLIT}"
  --assets-dir "${ASSETS_DIR}"
  --asset-id "${ASSET_ID}"
  --robot-id "${ROBOT_ID}"
  --pytorch-weight-path "${CONVERTED_BASE}"
  --expected-base-sha256 "${EXPECTED_BASE_SHA256}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --run-id "${RUN_ID}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --batch-size "${BATCH_SIZE}"
  --max-steps "${MAX_STEPS}"
  --learning-rate "${LEARNING_RATE}"
  --seed "${SEED}"
  --save-interval "${SAVE_INTERVAL}"
  --log-interval "${LOG_INTERVAL}"
  --output-json "${OUTPUT_JSON}"
)

if [[ "${WANDB}" == "1" ]]; then
  TRAIN_ARGS+=(--wandb --wandb-project "${WANDB_PROJECT}")
  if [[ -n "${WANDB_ENTITY}" ]]; then
    TRAIN_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
  fi
  if [[ -n "${WANDB_RUN_NAME}" ]]; then
    TRAIN_ARGS+=(--wandb-run-name "${WANDB_RUN_NAME}")
  fi
else
  TRAIN_ARGS+=(--no-wandb)
fi

if [[ -n "${MAX_EPISODES}" ]]; then
  TRAIN_ARGS+=(--max-episodes "${MAX_EPISODES}")
fi
if [[ -n "${RESUME_FROM}" ]]; then
  TRAIN_ARGS+=(--resume-from "${RESUME_FROM}")
fi
TRAIN_ARGS+=("${EXTRA_ARGS[@]}")

echo "[$(date -Is)] train_ddp"
echo "  dataset=${DATASET_ROOT}"
echo "  assets=${ASSETS_DIR}/${ASSET_ID}"
echo "  ckpt=${CHECKPOINT_DIR}"
echo "  batch_size(local)=${BATCH_SIZE} max_steps=${MAX_STEPS}"

if [[ "${USE_VOLC}" == "1" ]]; then
  echo "  launcher=volc (MLP_*)"
  if [[ "${DRY_RUN}" == "1" ]]; then
    pi-dex-volc-train --dry-run -- "${TRAIN_ARGS[@]}"
    exit 0
  fi
  exec bash "${ROOT}/scripts/volc_ddp_train.sh" -- "${TRAIN_ARGS[@]}"
fi

# Auto GPU count on single-node when unset
if [[ -z "${NPROC_PER_NODE}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  else
    NPROC_PER_NODE=1
  fi
  if [[ -z "${NPROC_PER_NODE}" || "${NPROC_PER_NODE}" -lt 1 ]]; then
    NPROC_PER_NODE=1
  fi
fi

WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
echo "  topology nnodes=${NNODES} nproc_per_node=${NPROC_PER_NODE} node_rank=${NODE_RANK} world_size=${WORLD_SIZE}"
echo "  master=${MASTER_ADDR}:${MASTER_PORT}"

LAUNCHER=(pi-dex-train-pytorch)
if [[ "${WORLD_SIZE}" -gt 1 ]]; then
  TRAIN_ARGS+=(--distributed)
  if [[ "${NNODES}" -eq 1 ]]; then
    LAUNCHER=(
      torchrun
      --standalone
      --nproc-per-node="${NPROC_PER_NODE}"
      "$(command -v pi-dex-train-pytorch)"
    )
  else
    LAUNCHER=(
      torchrun
      --nnodes="${NNODES}"
      --nproc-per-node="${NPROC_PER_NODE}"
      --node-rank="${NODE_RANK}"
      --master-addr="${MASTER_ADDR}"
      --master-port="${MASTER_PORT}"
      --rdzv-id="${RDZV_ID}"
      --rdzv-backend=c10d
      --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}"
      "$(command -v pi-dex-train-pytorch)"
    )
  fi
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'DRY_RUN:'
  printf ' %q' "${LAUNCHER[@]}" "${TRAIN_ARGS[@]}"
  printf '\n'
  exit 0
fi

exec "${LAUNCHER[@]}" "${TRAIN_ARGS[@]}"
