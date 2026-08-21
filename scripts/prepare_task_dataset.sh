#!/usr/bin/env bash
# Prepare any Sharpa-style task root for PI-DEX joint_29d training:
#   1) optional overlay with filled empty task_instruction
#   2) inventory
#   3) compute-norm-stats on train split
#
# Example (Insert_Battery):
#   nohup bash scripts/prepare_task_dataset.sh \
#     --source-root /mnt/netdata/Team/Academic/Data/Foundation_Model/Insert_Battery \
#     --task-name Insert_Battery \
#     --default-prompt "Pick up the large battery with the right hand and insert it into the large battery compartment. Then pick up the small battery with the right hand and insert it into the small battery compartment." \
#     > /mnt/netdata/Team/Personal/congsheng/pi-dex-artifacts/logs/prepare_insert_battery.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

: "${PI_DEX_ARTIFACTS:=/mnt/netdata/Team/Personal/congsheng/pi-dex-artifacts}"
: "${CONTRACT:=${ROOT}/configs/site/joint_29d_observation.reviewed.json}"
: "${OPENPI_DATA_HOME:=${PI_DEX_ARTIFACTS}/openpi-data}"
: "${SOURCE_ROOT:=}"
: "${TASK_NAME:=}"
: "${DEFAULT_PROMPT:=}"
: "${PREPARED_ROOT:=}"
: "${ASSETS_DIR:=}"
: "${ASSET_ID:=}"
: "${OVERWRITE_PREPARED:=0}"
: "${SKIP_NORM:=0}"
: "${ROBOT_ID:=}"
: "${NORM_WORKERS:=}"
: "${OMP_NUM_THREADS:=1}"
: "${MKL_NUM_THREADS:=1}"
: "${OPENBLAS_NUM_THREADS:=1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-root) SOURCE_ROOT="$2"; shift 2 ;;
    --task-name) TASK_NAME="$2"; shift 2 ;;
    --default-prompt) DEFAULT_PROMPT="$2"; shift 2 ;;
    --prepared-root) PREPARED_ROOT="$2"; shift 2 ;;
    --assets-dir) ASSETS_DIR="$2"; shift 2 ;;
    --asset-id) ASSET_ID="$2"; shift 2 ;;
    --robot-id) ROBOT_ID="$2"; shift 2 ;;
    --contract) CONTRACT="$2"; shift 2 ;;
    --overwrite-prepared) OVERWRITE_PREPARED=1; shift ;;
    --skip-norm) SKIP_NORM=1; shift ;;
    -h|--help)
      sed -n '1,20p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${SOURCE_ROOT}" ]]; then
  echo "--source-root is required" >&2
  exit 2
fi
if [[ -z "${TASK_NAME}" ]]; then
  TASK_NAME="$(basename "${SOURCE_ROOT}")"
fi
if [[ -z "${DEFAULT_PROMPT}" ]]; then
  echo "--default-prompt is required (Foundation_Model dumps often have empty task_instruction)" >&2
  exit 2
fi
if [[ -z "${PREPARED_ROOT}" ]]; then
  PREPARED_ROOT="${PI_DEX_ARTIFACTS}/prepared/${TASK_NAME}"
fi
if [[ -z "${ASSETS_DIR}" ]]; then
  ASSETS_DIR="${PI_DEX_ARTIFACTS}/assets-${TASK_NAME}"
fi
if [[ -z "${ASSET_ID}" ]]; then
  # asset ids are usually lowercase snake; keep task folder readable but stable
  ASSET_ID="sharpa_joint_29d_$(echo "${TASK_NAME}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
fi
if [[ -z "${ROBOT_ID}" ]]; then
  ROBOT_ID="$(find "${SOURCE_ROOT}" -maxdepth 1 -type d -name 'season_POC*' 2>/dev/null \
    | head -1 | sed -n 's/.*season_\(POC[0-9]*\)_.*/\1/p')"
fi
if [[ -z "${ROBOT_ID}" ]]; then
  ROBOT_ID="POC22027"
fi

export OPENPI_DATA_HOME
export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
export PATH="${PATH:-}"

mkdir -p "${PI_DEX_ARTIFACTS}/dataset" "${PI_DEX_ARTIFACTS}/logs" "${ASSETS_DIR}"

INVENTORY_JSON="${PI_DEX_ARTIFACTS}/dataset/${TASK_NAME}_inventory.json"
NORM_JSON="${PI_DEX_ARTIFACTS}/dataset/${TASK_NAME}_norm.json"
NORM_LOG="${PI_DEX_ARTIFACTS}/logs/prepare_${TASK_NAME}_norm.log"
PREPARE_META="${PI_DEX_ARTIFACTS}/dataset/${TASK_NAME}_prepared.json"

echo "[$(date -Is)] prepare_task_dataset start"
echo "  SOURCE_ROOT=${SOURCE_ROOT}"
echo "  PREPARED_ROOT=${PREPARED_ROOT}"
echo "  TASK_NAME=${TASK_NAME}"
echo "  ASSETS_DIR=${ASSETS_DIR}"
echo "  ASSET_ID=${ASSET_ID}"
echo "  ROBOT_ID=${ROBOT_ID}"

OVERWRITE_FLAG=()
if [[ "${OVERWRITE_PREPARED}" == "1" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

python -m pi_dex.data.dataset_prepare \
  --source-root "${SOURCE_ROOT}" \
  --prepared-root "${PREPARED_ROOT}" \
  --default-prompt "${DEFAULT_PROMPT}" \
  --output-json "${PREPARE_META}" \
  "${OVERWRITE_FLAG[@]}"

pi-dex-dataset-inventory \
  --dataset-root "${PREPARED_ROOT}" \
  --observation-contract "${CONTRACT}" \
  --output-json "${INVENTORY_JSON}"

if [[ "${SKIP_NORM}" == "1" ]]; then
  echo "[$(date -Is)] skip-norm requested; prepared root + inventory done"
  exit 0
fi

if [[ -f "${ASSETS_DIR}/${ASSET_ID}/norm_stats.json" ]]; then
  echo "[$(date -Is)] refuse overwrite: ${ASSETS_DIR}/${ASSET_ID}/norm_stats.json already exists"
  exit 1
fi

echo "[$(date -Is)] compute-norm-stats ..."
NORM_ARGS=()
if [[ -n "${NORM_WORKERS}" ]]; then
  NORM_ARGS+=(--norm-workers "${NORM_WORKERS}")
fi
pi-dex-train-pytorch \
  --action-representation joint_29d \
  --runner pi_dex.training.training_runner:run -- \
  --mode compute-norm-stats \
  --observation-contract "${CONTRACT}" \
  --dataset-root "${PREPARED_ROOT}" \
  --split train \
  --assets-dir "${ASSETS_DIR}" \
  --asset-id "${ASSET_ID}" \
  --robot-id "${ROBOT_ID}" \
  --output-json "${NORM_JSON}" \
  "${NORM_ARGS[@]}" \
  2>&1 | tee -a "${NORM_LOG}"

echo "[$(date -Is)] prepare_task_dataset done"
echo "  prepared:   ${PREPARED_ROOT}"
echo "  inventory:  ${INVENTORY_JSON}"
echo "  norm_stats: ${ASSETS_DIR}/${ASSET_ID}/norm_stats.json"
echo "  summary:    ${NORM_JSON}"
echo "  train with: bash scripts/train_ddp.sh --dataset-root ${PREPARED_ROOT} --assets-dir ${ASSETS_DIR} --asset-id ${ASSET_ID} --robot-id ${ROBOT_ID}"
