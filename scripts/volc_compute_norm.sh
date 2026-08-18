#!/usr/bin/env bash
# Volcano Engine MLP entry for PI-DEX joint_29d compute-norm-stats.
#
# This is a **single-process Python job** (no torchrun) that fans HDF5 episode
# scans across CPU workers inside that process. Do not wrap it in torchrun:
# every rank would rewrite the same assets dir. On a multi-GPU / multi-node MLP
# job only ``MLP_ROLE_INDEX=0`` runs; other workers exit 0.
#
# Prefer a high-core CPU node. GPU is unused. Set ``NORM_WORKERS`` (default in
# Python: min(cpu_count, 64)) and keep BLAS threads at 1 to avoid
# oversubscription.
#
# Recommended MLP "自定义启动命令" (1 node is enough; GPU unused):
#   bash /mnt/netdata/Team/Personal/congsheng/PI-DEX/scripts/volc_compute_norm.sh
#
# Defaults: full SharpaOpenData train split → assets-opendata/sharpa_joint_29d_opendata_v0
#
# Dry-run:
#   VOLC_DRY_RUN=1 bash scripts/volc_compute_norm.sh
#
# Local without platform MLP:
#   VOLC_SKIP_MLP=1 bash scripts/volc_compute_norm.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

: "${PI_DEX_REPO:=${ROOT}}"
: "${PI_DEX_ARTIFACTS:=/mnt/netdata/Team/Personal/congsheng/pi-dex-artifacts}"
: "${CONDA_ROOT:=/mnt/netdata/Team/Personal/congsheng/miniconda}"
: "${PI_DEX_CONDA_ENV:=pi-dex}"
: "${OPENPI_DATA_HOME:=${PI_DEX_ARTIFACTS}/openpi-data}"
: "${CONTRACT:=${PI_DEX_REPO}/configs/site/joint_29d_observation.reviewed.json}"

: "${DATASET_ROOT:=/mnt/netdata/Team/Academic/Data/North/SharpaOpenData}"
: "${ASSETS_DIR:=${PI_DEX_ARTIFACTS}/assets-opendata}"
: "${ASSET_ID:=sharpa_joint_29d_opendata_v0}"
: "${ROBOT_ID:=POC22027}"
: "${SPLIT:=train}"
: "${MAX_EPISODES:=}"
: "${MAX_SAMPLES:=}"
: "${NORM_WORKERS:=}"
: "${NORM_STRIDE:=1}"
: "${OMP_NUM_THREADS:=1}"
: "${MKL_NUM_THREADS:=1}"
: "${OPENBLAS_NUM_THREADS:=1}"

: "${OUTPUT_JSON:=${PI_DEX_ARTIFACTS}/dataset/norm_opendata_full.json}"
: "${NORM_FORCE:=0}"
: "${VOLC_DRY_RUN:=0}"
: "${VOLC_SKIP_CONDA:=0}"
: "${VOLC_SKIP_MLP:=0}"

if [[ "${VOLC_SKIP_CONDA}" != "1" ]]; then
  if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${PI_DEX_CONDA_ENV}"
  elif [[ -x "${CONDA_ROOT}/envs/${PI_DEX_CONDA_ENV}/bin/python" ]]; then
    export PATH="${CONDA_ROOT}/envs/${PI_DEX_CONDA_ENV}/bin:${PATH}"
  else
    echo "volc_compute_norm: cannot activate conda env ${PI_DEX_CONDA_ENV} under ${CONDA_ROOT}" >&2
    exit 2
  fi
fi

export PI_DEX_ARTIFACTS OPENPI_DATA_HOME
export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
fi

PYTHON_BIN="$(command -v python)"
TRAIN_BIN="$(command -v pi-dex-train-pytorch || true)"
echo "[$(date -Is)] volc_compute_norm env"
echo "  python=${PYTHON_BIN}"
echo "  pi-dex-train-pytorch=${TRAIN_BIN:-<missing>}"
echo "  CONDA_PREFIX=${CONDA_PREFIX:-}"
echo "  OPENPI_DATA_HOME=${OPENPI_DATA_HOME}"

if [[ "${VOLC_SKIP_MLP}" != "1" ]]; then
  for key in MLP_WORKER_NUM MLP_WORKER_GPU MLP_ROLE_INDEX MLP_WORKER_0_HOST MLP_WORKER_0_PORT; do
    if [[ -z "${!key:-}" ]]; then
      echo "volc_compute_norm: missing ${key} (Volcano MLP injects these; or set VOLC_SKIP_MLP=1)" >&2
      exit 2
    fi
  done
  echo "  MLP: nnodes=${MLP_WORKER_NUM} nproc_per_node=${MLP_WORKER_GPU} node_rank=${MLP_ROLE_INDEX}"
  if [[ "${MLP_ROLE_INDEX}" != "0" ]]; then
    echo "volc_compute_norm: worker ${MLP_ROLE_INDEX} idle (norm stats is rank-0 only)"
    exit 0
  fi
  if [[ "${MLP_WORKER_GPU}" != "1" || "${MLP_WORKER_NUM}" != "1" ]]; then
    echo "volc_compute_norm: NOTE single-process job; extra GPUs/nodes are idle. Prefer 1×1."
  fi
fi

USER_ARGS=()
if [[ "${1:-}" == "--" ]]; then
  shift
  USER_ARGS=("$@")
elif [[ $# -gt 0 ]]; then
  USER_ARGS=("$@")
fi

if [[ ${#USER_ARGS[@]} -eq 0 ]]; then
  if [[ ! -d "${DATASET_ROOT}" ]]; then
    echo "volc_compute_norm: missing dataset root: ${DATASET_ROOT}" >&2
    exit 1
  fi
  if [[ ! -f "${CONTRACT}" ]]; then
    echo "volc_compute_norm: missing contract: ${CONTRACT}" >&2
    exit 1
  fi
  NORM_PATH="${ASSETS_DIR}/${ASSET_ID}/norm_stats.json"
  if [[ -f "${NORM_PATH}" && "${NORM_FORCE}" != "1" ]]; then
    echo "volc_compute_norm: refuse overwrite: ${NORM_PATH} (set NORM_FORCE=1)" >&2
    exit 1
  fi
  mkdir -p "${ASSETS_DIR}" "$(dirname "${OUTPUT_JSON}")" "${PI_DEX_ARTIFACTS}/logs"

  TRAIN_ARGS=(
    --action-representation joint_29d
    --runner pi_dex.training_runner:run
    --
    --mode compute-norm-stats
    --observation-contract "${CONTRACT}"
    --dataset-root "${DATASET_ROOT}"
    --split "${SPLIT}"
    --assets-dir "${ASSETS_DIR}"
    --asset-id "${ASSET_ID}"
    --robot-id "${ROBOT_ID}"
    --output-json "${OUTPUT_JSON}"
  )
  if [[ -n "${MAX_EPISODES}" ]]; then
    TRAIN_ARGS+=(--max-episodes "${MAX_EPISODES}")
  fi
  if [[ -n "${MAX_SAMPLES}" ]]; then
    TRAIN_ARGS+=(--max-samples "${MAX_SAMPLES}")
  fi
  if [[ -n "${NORM_WORKERS}" ]]; then
    TRAIN_ARGS+=(--norm-workers "${NORM_WORKERS}")
  fi
  if [[ -n "${NORM_STRIDE}" && "${NORM_STRIDE}" != "1" ]]; then
    TRAIN_ARGS+=(--norm-stride "${NORM_STRIDE}")
  fi
else
  TRAIN_ARGS=("${USER_ARGS[@]}")
fi

echo "  dataset=${DATASET_ROOT}"
echo "  assets=${ASSETS_DIR}/${ASSET_ID}"
echo "  split=${SPLIT} output_json=${OUTPUT_JSON}"
echo "  norm_workers=${NORM_WORKERS:-<auto>} stride=${NORM_STRIDE} OMP_NUM_THREADS=${OMP_NUM_THREADS}"

if [[ -z "${TRAIN_BIN}" ]]; then
  LAUNCH=("${PYTHON_BIN}" -m pi_dex.training_launcher)
else
  LAUNCH=("${TRAIN_BIN}")
fi

if [[ "${VOLC_DRY_RUN}" == "1" ]]; then
  printf 'DRY: '
  printf '%q ' "${LAUNCH[@]}" "${TRAIN_ARGS[@]}"
  printf '\n'
  exit 0
fi

echo "[$(date -Is)] compute-norm-stats start"
exec "${LAUNCH[@]}" "${TRAIN_ARGS[@]}"
