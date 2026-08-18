#!/usr/bin/env bash
# Create or refresh the PI-DEX miniconda env.
#
# This host's ~/.condarc remaps conda-forge to flaky mirrors, so we prefer:
#   1) clone local env `kshift` (same Python 3.11, no network)
#   2) otherwise create from an absolute conda-forge URL
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${PI_DEX_CONDA_ROOT:-/mnt/netdata/Team/Personal/congsheng/miniconda}"
ENV_NAME="pi-dex"
ENV_PREFIX="${CONDA_ROOT}/envs/${ENV_NAME}"
CLONE_FROM="${PI_DEX_CONDA_CLONE_FROM:-kshift}"

if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then
  echo "conda not found at ${CONDA_ROOT}/bin/conda" >&2
  exit 1
fi

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${CONDA_ROOT}/pkgs}"
mkdir -p "${CONDA_PKGS_DIRS}"

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

if [[ -d "${ENV_PREFIX}" ]]; then
  echo "env exists: ${ENV_PREFIX}"
elif [[ -d "${CONDA_ROOT}/envs/${CLONE_FROM}" ]]; then
  echo "cloning ${CLONE_FROM} -> ${ENV_NAME} (offline-friendly)"
  conda create -y -n "${ENV_NAME}" --clone "${CLONE_FROM}"
else
  echo "creating ${ENV_NAME} from absolute conda-forge URL"
  conda create -y -p "${ENV_PREFIX}" python=3.11 pip \
    --override-channels \
    -c https://conda.anaconda.org/conda-forge
fi

conda activate "${ENV_NAME}"

# Keep pip downloads on netdata, not root overlay.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CONDA_ROOT}/pip-cache}"
mkdir -p "${PIP_CACHE_DIR}"

python -m pip install -U pip
python -m pip install \
  "numpy>=1.22.4,<2.0.0" \
  "pytest>=8.3.4" \
  "ruff>=0.8.6"

# Torch is large; allow long timeout. Prefer Tsinghua PyPI if reachable.
python -m pip install "torch==2.7.1" \
  -i "${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  --default-timeout 300

cd "${ROOT}"
python -m pip install -e .

python -c 'import sys, numpy; print(sys.executable); print("python", sys.version.split()[0], "numpy", numpy.__version__)'
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'

echo
echo "activate with:"
echo "  source ${CONDA_ROOT}/etc/profile.d/conda.sh"
echo "  conda activate ${ENV_NAME}"
