#!/usr/bin/env bash
# Volcano Engine MLP entry for PI-DEX joint_29d DDP training.
#
# Platform injects on each worker:
#   MLP_WORKER_NUM MLP_WORKER_GPU MLP_ROLE_INDEX
#   MLP_WORKER_0_HOST MLP_WORKER_0_PORT
#
# This script:
#   1) activates the pi-dex conda env
#   2) exports OPENPI_DATA_HOME / artifact paths
#   3) validates WANDB_API_KEY when wandb is on
#   4) launches torchrun via pi_dex.volc_launch
#
# Recommended MLP "自定义启动命令" (1 node × 8 GPU):
#   bash /mnt/netdata/Team/Personal/congsheng/PI-DEX/scripts/volc_ddp_train.sh
#
# Or with explicit overrides:
#   bash scripts/volc_ddp_train.sh -- \
#     --action-representation joint_29d \
#     --runner pi_dex.training_runner:run -- \
#     --mode train ...
#
# Dry-run (print torchrun only):
#   VOLC_DRY_RUN=1 bash scripts/volc_ddp_train.sh
#
# Local smoke without the platform (export MLP_* first):
#   export MLP_WORKER_NUM=1 MLP_WORKER_GPU=2 MLP_ROLE_INDEX=0
#   export MLP_WORKER_0_HOST=127.0.0.1 MLP_WORKER_0_PORT=29500
#   export WANDB_API_KEY=...   # or VOLC_WANDB=0
#   bash scripts/volc_ddp_train.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# --- site defaults (override via env before launching the job) ---
: "${PI_DEX_REPO:=${ROOT}}"
: "${PI_DEX_ARTIFACTS:=/mnt/netdata/Team/Personal/congsheng/pi-dex-artifacts}"
: "${CONDA_ROOT:=/mnt/netdata/Team/Personal/congsheng/miniconda}"
: "${PI_DEX_CONDA_ENV:=pi-dex}"
: "${OPENPI_DATA_HOME:=${PI_DEX_ARTIFACTS}/openpi-data}"
: "${CONTRACT:=${PI_DEX_REPO}/configs/site/joint_29d_observation.reviewed.json}"
: "${CONVERTED_BASE:=${PI_DEX_ARTIFACTS}/converted/pi05_base-pytorch-bfloat16-K8}"
: "${EXPECTED_BASE_SHA256:=2f8539e2308611ea6fff84a5d7774f80d7c177c624769ff842008cf85dea9eeb}"

# Insert_Battery defaults (override DATASET_ROOT / ASSETS_* for other tasks)
: "${DATASET_ROOT:=${PI_DEX_ARTIFACTS}/prepared/Insert_Battery}"
: "${ASSETS_DIR:=${PI_DEX_ARTIFACTS}/assets-Insert_Battery}"
: "${ASSET_ID:=sharpa_joint_29d_insert_battery}"
: "${ROBOT_ID:=POC22005}"

: "${CHECKPOINT_DIR:=${PI_DEX_ARTIFACTS}/runs/volc-${ASSET_ID}-$(date +%Y%m%d-%H%M%S)}"
: "${RUN_ID:=$(basename "${CHECKPOINT_DIR}")}"
: "${OUTPUT_JSON:=${CHECKPOINT_DIR%/}.json}"

: "${MAX_STEPS:=1000}"
: "${BATCH_SIZE:=1}"
: "${LEARNING_RATE:=1e-5}"
: "${LR_WARMUP_STEPS:=}"
: "${LR_DECAY_STEPS:=}"
: "${LR_END:=}"
: "${SEED:=0}"
: "${DEVICE:=cuda}"
: "${DTYPE:=bfloat16}"
: "${SPLIT:=train}"
: "${SAVE_INTERVAL:=500}"
: "${LOG_INTERVAL:=10}"
: "${MAX_EPISODES:=}"
: "${RESUME_FROM:=}"

: "${ACTION_MODE:=absolute}"
: "${COMMAND_SEMANTICS_VERSION:=sharpa_sdk_commanded_joint_position_absolute_v1}"

: "${VOLC_WANDB:=1}"
: "${WANDB_PROJECT:=pi-dex}"
: "${WANDB_ENTITY:=}"
: "${WANDB_RUN_NAME:=}"
: "${WANDB_API_KEY:=}"
: "${VOLC_DRY_RUN:=0}"
: "${VOLC_SKIP_CONDA:=0}"

# --- activate conda so torchrun / pi-dex-train-pytorch resolve correctly ---
if [[ "${VOLC_SKIP_CONDA}" != "1" ]]; then
  if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${PI_DEX_CONDA_ENV}"
  elif [[ -x "${CONDA_ROOT}/envs/${PI_DEX_CONDA_ENV}/bin/python" ]]; then
    export PATH="${CONDA_ROOT}/envs/${PI_DEX_CONDA_ENV}/bin:${PATH}"
  else
    echo "volc_ddp_train: cannot activate conda env ${PI_DEX_CONDA_ENV} under ${CONDA_ROOT}" >&2
    echo "set CONDA_ROOT / PI_DEX_CONDA_ENV, or VOLC_SKIP_CONDA=1 if already activated" >&2
    exit 2
  fi
fi

export PI_DEX_ARTIFACTS OPENPI_DATA_HOME
export PYTHONUNBUFFERED=1
export PATH="${PATH:-}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
fi

PYTHON_BIN="$(command -v python)"
TORCHRUN_BIN="$(command -v torchrun || true)"
TRAIN_BIN="$(command -v pi-dex-train-pytorch || true)"
echo "[$(date -Is)] volc_ddp_train env"
echo "  python=${PYTHON_BIN}"
echo "  torchrun=${TORCHRUN_BIN:-<missing>}"
echo "  pi-dex-train-pytorch=${TRAIN_BIN:-<missing>}"
echo "  CONDA_PREFIX=${CONDA_PREFIX:-}"
echo "  OPENPI_DATA_HOME=${OPENPI_DATA_HOME}"

# --- validate MLP_* (platform injects these) ---
for key in MLP_WORKER_NUM MLP_WORKER_GPU MLP_ROLE_INDEX MLP_WORKER_0_HOST MLP_WORKER_0_PORT; do
  if [[ -z "${!key:-}" ]]; then
    echo "volc_ddp_train: missing ${key} (Volcano MLP injects these on workers)" >&2
    exit 2
  fi
done
echo "  MLP: nnodes=${MLP_WORKER_NUM} nproc_per_node=${MLP_WORKER_GPU} node_rank=${MLP_ROLE_INDEX}"
echo "  master=${MLP_WORKER_0_HOST}:${MLP_WORKER_0_PORT}"

# --- wandb ---
if [[ "${VOLC_WANDB}" == "1" ]]; then
  if [[ -z "${WANDB_API_KEY}" ]]; then
    echo "volc_ddp_train: VOLC_WANDB=1 requires WANDB_API_KEY (or set VOLC_WANDB=0)" >&2
    exit 2
  fi
  export WANDB_API_KEY
  export WANDB_SILENT="${WANDB_SILENT:-true}"
fi

# --- optional: user passed full training argv after -- ---
USER_ARGS=()
if [[ "${1:-}" == "--" ]]; then
  shift
  USER_ARGS=("$@")
elif [[ $# -gt 0 ]]; then
  USER_ARGS=("$@")
fi

if [[ ${#USER_ARGS[@]} -eq 0 ]]; then
  if [[ ! -f "${ASSETS_DIR}/${ASSET_ID}/norm_stats.json" ]]; then
    echo "volc_ddp_train: missing norm stats: ${ASSETS_DIR}/${ASSET_ID}/norm_stats.json" >&2
    exit 1
  fi
  if [[ ! -d "${CONVERTED_BASE}" ]]; then
    echo "volc_ddp_train: missing converted base: ${CONVERTED_BASE}" >&2
    exit 1
  fi
  if [[ ! -d "${DATASET_ROOT}" ]]; then
    echo "volc_ddp_train: missing dataset root: ${DATASET_ROOT}" >&2
    exit 1
  fi

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
    --action-mode "${ACTION_MODE}"
    --command-semantics-version "${COMMAND_SEMANTICS_VERSION}"
  )
  if [[ -n "${MAX_EPISODES}" ]]; then
    TRAIN_ARGS+=(--max-episodes "${MAX_EPISODES}")
  fi
  if [[ -n "${RESUME_FROM}" ]]; then
    TRAIN_ARGS+=(--resume-from "${RESUME_FROM}")
  fi
  if [[ -n "${LR_WARMUP_STEPS}" ]]; then
    TRAIN_ARGS+=(--lr-warmup-steps "${LR_WARMUP_STEPS}")
  fi
  if [[ -n "${LR_DECAY_STEPS}" ]]; then
    TRAIN_ARGS+=(--lr-decay-steps "${LR_DECAY_STEPS}")
  fi
  if [[ -n "${LR_END}" ]]; then
    TRAIN_ARGS+=(--lr-end "${LR_END}")
  fi
  if [[ "${VOLC_WANDB}" == "1" ]]; then
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
else
  TRAIN_ARGS=("${USER_ARGS[@]}")
fi

echo "  dataset=${DATASET_ROOT}"
echo "  assets=${ASSETS_DIR}/${ASSET_ID}"
echo "  ckpt=${CHECKPOINT_DIR}"
echo "  batch_size(local)=${BATCH_SIZE} max_steps=${MAX_STEPS} save_interval=${SAVE_INTERVAL}"
echo "  action_mode=${ACTION_MODE} asset=${ASSET_ID}"
echo "  lr=${LEARNING_RATE} warmup=${LR_WARMUP_STEPS:-default} decay=${LR_DECAY_STEPS:-max_steps} end=${LR_END:-0.1x}"
echo "  wandb=${VOLC_WANDB} project=${WANDB_PROJECT}"

DRY_FLAG=()
if [[ "${VOLC_DRY_RUN}" == "1" ]]; then
  DRY_FLAG=(--dry-run)
fi

# Prefer env python module entry so we never pick a system python.
exec "${PYTHON_BIN}" -m pi_dex.volc_launch "${DRY_FLAG[@]}" -- "${TRAIN_ARGS[@]}"
