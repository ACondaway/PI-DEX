#!/usr/bin/env bash
# Full OpenData joint_29d dataset prep: inventory (if missing) + compute-norm-stats.
# Designed to be started under nohup / tmux.
#
# Usage:
#   nohup bash scripts/prepare_opendata_full.sh \
#     > /mnt/netdata/Team/Personal/congsheng/pi-dex-artifacts/logs/prepare_opendata_full.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

: "${PI_DEX_ARTIFACTS:=/mnt/netdata/Team/Personal/congsheng/pi-dex-artifacts}"
: "${OPENDATA_ROOT:=/mnt/netdata/Team/Academic/Data/North/SharpaOpenData}"
: "${CONTRACT:=${ROOT}/configs/site/joint_29d_observation.reviewed.json}"
: "${ASSETS_DIR:=${PI_DEX_ARTIFACTS}/assets-opendata}"
: "${ASSET_ID:=sharpa_joint_29d_opendata_v0}"
: "${OPENPI_DATA_HOME:=${PI_DEX_ARTIFACTS}/openpi-data}"
: "${NORM_WORKERS:=}"
: "${OMP_NUM_THREADS:=1}"
: "${MKL_NUM_THREADS:=1}"
: "${OPENBLAS_NUM_THREADS:=1}"

export OPENPI_DATA_HOME
export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS

mkdir -p "${PI_DEX_ARTIFACTS}/dataset" "${PI_DEX_ARTIFACTS}/logs" "${ASSETS_DIR}"

INVENTORY_JSON="${PI_DEX_ARTIFACTS}/dataset/opendata_inventory.json"
NORM_JSON="${PI_DEX_ARTIFACTS}/dataset/norm_opendata_full.json"
NORM_LOG="${PI_DEX_ARTIFACTS}/logs/norm_opendata_full.log"

echo "[$(date -Is)] prepare_opendata_full start"
echo "  OPENDATA_ROOT=${OPENDATA_ROOT}"
echo "  CONTRACT=${CONTRACT}"
echo "  ASSETS_DIR=${ASSETS_DIR}"
echo "  ASSET_ID=${ASSET_ID}"

if [[ ! -f "${INVENTORY_JSON}" ]]; then
  echo "[$(date -Is)] inventory missing; running pi-dex-dataset-inventory"
  pi-dex-dataset-inventory \
    --dataset-root "${OPENDATA_ROOT}" \
    --observation-contract "${CONTRACT}" \
    --output-json "${INVENTORY_JSON}"
else
  echo "[$(date -Is)] inventory present: ${INVENTORY_JSON}"
fi

if [[ -f "${ASSETS_DIR}/${ASSET_ID}/norm_stats.json" ]]; then
  echo "[$(date -Is)] refuse overwrite: ${ASSETS_DIR}/${ASSET_ID}/norm_stats.json already exists"
  exit 1
fi

echo "[$(date -Is)] compute-norm-stats (train split, full OpenData) ..."
echo "[$(date -Is)] detailed log: ${NORM_LOG}"
echo "  NORM_WORKERS=${NORM_WORKERS:-<auto>} OMP_NUM_THREADS=${OMP_NUM_THREADS}"

NORM_ARGS=()
if [[ -n "${NORM_WORKERS}" ]]; then
  NORM_ARGS+=(--norm-workers "${NORM_WORKERS}")
fi

pi-dex-train-pytorch \
  --action-representation joint_29d \
  --runner pi_dex.training.training_runner:run -- \
  --mode compute-norm-stats \
  --observation-contract "${CONTRACT}" \
  --dataset-root "${OPENDATA_ROOT}" \
  --split train \
  --assets-dir "${ASSETS_DIR}" \
  --asset-id "${ASSET_ID}" \
  --output-json "${NORM_JSON}" \
  "${NORM_ARGS[@]}" \
  2>&1 | tee -a "${NORM_LOG}"

echo "[$(date -Is)] prepare_opendata_full done"
echo "  norm_stats: ${ASSETS_DIR}/${ASSET_ID}/norm_stats.json"
echo "  summary:    ${NORM_JSON}"
