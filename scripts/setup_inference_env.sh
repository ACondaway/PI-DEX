#!/usr/bin/env bash
# Provision an inference host with the same ``pi-dex`` stack as the developer machine.
#
# Full guide: docs/inference-env.md
#
# Matches:
#   - conda env name: pi-dex
#   - Python 3.11
#   - torch==2.7.1 (CUDA 12.6 wheels by default)
#   - configs/inference/pip-lock.txt (exported from the developer env)
#   - editable installs of this repo: openpi-client, openpi, pi-dex
#   - OpenPI transformers_replace overlay (required for pi0.5 PyTorch)
#
# Typical GPU inference box (serve):
#   git clone <PI-DEX> && cd PI-DEX
#   bash scripts/setup_inference_env.sh
#   source "$(bash scripts/setup_inference_env.sh --print-activate)"
#   bash scripts/serve_joint29d.sh --checkpoint-dir /path/to/10000
#
# Robot NUC that only runs the Zenoh bridge (CPU ok):
#   bash scripts/setup_inference_env.sh --profile robot-client
#
# Regenerate the lock on the developer machine first if deps changed:
#   bash scripts/export_inference_lock.sh
#
# Environment knobs:
#   PI_DEX_CONDA_ROOT   Miniconda/Miniforge root (default: ~/miniconda3, else shared netdata path)
#   PI_DEX_CONDA_ENV    Env name (default: pi-dex)
#   PI_DEX_ARTIFACTS    Artifact root for OPENPI_DATA_HOME (optional)
#   PIP_INDEX_URL       Primary PyPI mirror (default: Tsinghua)
#   TORCH_CUDA          cu126|cu124|cpu (default: cu126)
#   INSTALL_MINICONDA=1 Download Miniconda if missing under PI_DEX_CONDA_ROOT
#   SKIP_LOCK=1         Skip pip-lock; only torch + editable packages (faster, less identical)
#   WITH_ZENOH=1        Force eclipse-zenoh (also on for robot-client profile)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_FILE="${ROOT}/configs/inference/pip-lock.txt"
PROFILE="gpu-serve"
VERIFY_ONLY=0
PRINT_ACTIVATE=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_inference_env.sh [options]

Options:
  --profile gpu-serve|robot-client|full   Default: gpu-serve
  --conda-root PATH                       Miniconda root
  --env-name NAME                         Default: pi-dex
  --torch-cuda cu126|cu124|cpu            Default: cu126
  --lock-file PATH                        Default: configs/inference/pip-lock.txt
  --skip-lock                             Do not install the frozen pip-lock
  --with-zenoh                            Install eclipse-zenoh
  --install-miniconda                     Bootstrap Miniconda under --conda-root if missing
  --verify-only                           Only run import/CUDA checks in existing env
  --print-activate                        Print `source ... && conda activate` lines and exit
  --dry-run                               Print planned actions
  -h, --help                              Show help
EOF
}

DEFAULT_SHARED_CONDA="/mnt/netdata/Team/Personal/congsheng/miniconda"
if [[ -n "${PI_DEX_CONDA_ROOT:-}" ]]; then
  CONDA_ROOT="${PI_DEX_CONDA_ROOT}"
elif [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
  CONDA_ROOT="${HOME}/miniconda3"
elif [[ -x "${HOME}/miniforge3/bin/conda" ]]; then
  CONDA_ROOT="${HOME}/miniforge3"
elif [[ -x "${DEFAULT_SHARED_CONDA}/bin/conda" ]]; then
  CONDA_ROOT="${DEFAULT_SHARED_CONDA}"
else
  CONDA_ROOT="${HOME}/miniconda3"
fi

ENV_NAME="${PI_DEX_CONDA_ENV:-pi-dex}"
TORCH_CUDA="${TORCH_CUDA:-cu126}"
SKIP_LOCK="${SKIP_LOCK:-0}"
WITH_ZENOH="${WITH_ZENOH:-0}"
INSTALL_MINICONDA="${INSTALL_MINICONDA:-0}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --conda-root) CONDA_ROOT="$2"; shift 2 ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --torch-cuda) TORCH_CUDA="$2"; shift 2 ;;
    --lock-file) LOCK_FILE="$2"; shift 2 ;;
    --skip-lock) SKIP_LOCK=1; shift ;;
    --with-zenoh) WITH_ZENOH=1; shift ;;
    --install-miniconda) INSTALL_MINICONDA=1; shift ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    --print-activate) PRINT_ACTIVATE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${PROFILE}" in
  gpu-serve|full) ;;
  robot-client)
    WITH_ZENOH=1
    ;;
  *)
    echo "invalid --profile ${PROFILE}" >&2
    exit 2
    ;;
esac

activate_snippet() {
  cat <<EOF
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate ${ENV_NAME}
export OPENPI_DATA_HOME=\${OPENPI_DATA_HOME:-\${PI_DEX_ARTIFACTS:-${ROOT}/../pi-dex-artifacts}/openpi-data}
EOF
}

if [[ "${PRINT_ACTIVATE}" == "1" ]]; then
  activate_snippet
  exit 0
fi

log() { printf '[setup_inference_env] %s\n' "$*"; }
run() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY: $*"
    return 0
  fi
  "$@"
}

ensure_miniconda() {
  if [[ -x "${CONDA_ROOT}/bin/conda" ]]; then
    return 0
  fi
  if [[ "${INSTALL_MINICONDA}" != "1" ]]; then
    echo "conda not found at ${CONDA_ROOT}/bin/conda" >&2
    echo "Re-run with --install-miniconda, or set PI_DEX_CONDA_ROOT to an existing install." >&2
    exit 1
  fi
  local installer="/tmp/Miniconda3-latest-Linux-x86_64.sh"
  log "installing Miniconda into ${CONDA_ROOT}"
  run curl -fsSL -o "${installer}" "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  run bash "${installer}" -b -p "${CONDA_ROOT}"
}

verify_env() {
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
  python - <<'PY'
from __future__ import annotations

import importlib
import sys

errors: list[str] = []

def need(mod: str, attr: str | None = None) -> None:
    try:
        m = importlib.import_module(mod)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{mod}: import failed ({exc})")
        return
    if attr:
        ver = getattr(m, attr, "?")
        print(f"OK {mod}={ver}")
    else:
        print(f"OK {mod}")

print("python", sys.version.split()[0], "exe", sys.executable)
need("numpy", "__version__")
need("torch", "__version__")
need("PIL", "__version__")
need("google.protobuf", "__version__")
need("websockets", "__version__")
need("openpi_client")
need("openpi")
need("pi_dex")
need("transformers", "__version__")

try:
    import torch
    print("torch.cuda", torch.version.cuda, "available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu0", torch.cuda.get_device_name(0))
except Exception as exc:  # noqa: BLE001
    errors.append(f"torch cuda probe failed: {exc}")

try:
    import zenoh  # noqa: F401
    print("OK zenoh")
except Exception:
    print("NOTE zenoh not installed (ok for gpu-serve-only)")

# transformers_replace marker: openpi ships custom modeling files under site-packages.
try:
    import transformers
    from pathlib import Path
    gemma = Path(transformers.__file__).resolve().parent / "models" / "gemma" / "modeling_gemma.py"
    text = gemma.read_text(encoding="utf-8", errors="ignore") if gemma.is_file() else ""
    if "openpi" in text.lower() or "paligemma" in text.lower() or gemma.is_file():
        print("OK transformers tree present at", gemma.parent)
except Exception as exc:  # noqa: BLE001
    errors.append(f"transformers overlay check: {exc}")

if errors:
    print("VERIFY FAILED:", file=sys.stderr)
    for e in errors:
        print(" -", e, file=sys.stderr)
    raise SystemExit(1)
print("VERIFY OK")
PY
}

apply_transformers_replace() {
  local replace_src="${ROOT}/openpi/src/openpi/models_pytorch/transformers_replace"
  if [[ ! -d "${replace_src}" ]]; then
    log "WARN: transformers_replace missing at ${replace_src}"
    return 0
  fi
  local site
  site="$(python -c 'import transformers, pathlib; print(pathlib.Path(transformers.__file__).resolve().parent)')"
  log "applying transformers_replace -> ${site}"
  while IFS= read -r -d '' src; do
    local rel="${src#"${replace_src}"/}"
    local dest="${site}/${rel}"
    run mkdir -p "$(dirname "${dest}")"
    run cp -f "${src}" "${dest}"
  done < <(find "${replace_src}" -type f -print0)
}

pip_install() {
  run python -m pip install \
    -i "${PIP_INDEX_URL}" \
    --trusted-host "${PIP_TRUSTED_HOST}" \
    --default-timeout 300 \
    "$@"
}

install_torch() {
  case "${TORCH_CUDA}" in
    cu126)
      log "installing torch==2.7.1 (cu126)"
      pip_install "torch==2.7.1" "torchvision==0.22.1" \
        --extra-index-url "https://download.pytorch.org/whl/cu126"
      ;;
    cu124)
      log "installing torch==2.7.1 (cu124)"
      pip_install "torch==2.7.1" "torchvision==0.22.1" \
        --extra-index-url "https://download.pytorch.org/whl/cu124"
      ;;
    cpu)
      log "installing torch==2.7.1 (cpu)"
      pip_install "torch==2.7.1" "torchvision==0.22.1" \
        --extra-index-url "https://download.pytorch.org/whl/cpu"
      ;;
    *)
      echo "invalid TORCH_CUDA=${TORCH_CUDA}" >&2
      exit 2
      ;;
  esac
}

install_lock() {
  if [[ "${SKIP_LOCK}" == "1" ]]; then
    log "SKIP_LOCK=1: installing minimal runtime pins only"
    pip_install "numpy>=1.22.4,<2.0.0" "protobuf==7.35.1" "websockets==17.0.1" \
      "pillow>=11.0.0" "safetensors==0.8.0" "einops==0.8.2" "transformers==4.53.2" \
      "msgpack==1.2.1" "tyro>=0.9.5" "tqdm-loggable>=0.2" "filelock>=3.16.1"
    return 0
  fi
  if [[ ! -f "${LOCK_FILE}" ]]; then
    echo "lock file missing: ${LOCK_FILE}" >&2
    echo "On the developer machine run: bash scripts/export_inference_lock.sh" >&2
    exit 1
  fi
  log "installing frozen lock ${LOCK_FILE}"
  # Torch is installed separately with the correct CUDA index; drop it from the lock pass.
  local filtered
  filtered="$(mktemp)"
  # Also drop openpi-client so the workspace editable wins.
  # Portable filter (do not require ripgrep on the inference host).
  grep -E -v '^(torch==|torchvision==|openpi-client==|#|$)' "${LOCK_FILE}" > "${filtered}" || true
  pip_install -r "${filtered}"
  rm -f "${filtered}"
}

install_editables() {
  log "editable installs from ${ROOT}"
  pip_install -e "${ROOT}/openpi/packages/openpi-client"
  # OpenPI pulls JAX etc.; lock usually already satisfied them.
  pip_install -e "${ROOT}/openpi"
  pip_install -e "${ROOT}"
}

install_zenoh_if_needed() {
  if [[ "${WITH_ZENOH}" != "1" ]]; then
    return 0
  fi
  log "installing eclipse-zenoh"
  pip_install "eclipse-zenoh>=1.3.4"
}

write_env_hint() {
  local hint="${ROOT}/configs/inference/activate.snippet.sh"
  activate_snippet > "${hint}"
  log "wrote ${hint}"
}

# --- main ---
log "ROOT=${ROOT}"
log "PROFILE=${PROFILE} CONDA_ROOT=${CONDA_ROOT} ENV=${ENV_NAME} TORCH_CUDA=${TORCH_CUDA}"

if [[ "${VERIFY_ONLY}" == "1" ]]; then
  verify_env
  exit 0
fi

ensure_miniconda
if [[ "${DRY_RUN}" == "1" ]]; then
  log "dry-run complete (skipped conda activate / installs)"
  exit 0
fi

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${CONDA_ROOT}/pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CONDA_ROOT}/pip-cache}"
run mkdir -p "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}"

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

ENV_PREFIX="${CONDA_ROOT}/envs/${ENV_NAME}"
if [[ ! -d "${ENV_PREFIX}" ]]; then
  log "creating conda env ${ENV_NAME} (python 3.11)"
  run conda create -y -p "${ENV_PREFIX}" python=3.11 pip \
    --override-channels \
    -c https://conda.anaconda.org/conda-forge
else
  log "env exists: ${ENV_PREFIX}"
fi

conda activate "${ENV_NAME}"
run python -m pip install -U pip setuptools wheel

if [[ "${PROFILE}" == "robot-client" ]]; then
  # Bridge box: still install torch CPU by default unless user overrides.
  if [[ "${TORCH_CUDA}" == "cu126" ]]; then
    TORCH_CUDA="cpu"
    log "robot-client profile: defaulting TORCH_CUDA=cpu (override with --torch-cuda)"
  fi
fi

install_torch
install_lock
install_editables
apply_transformers_replace
install_zenoh_if_needed
write_env_hint
verify_env

cat <<EOF

Inference env ready (aligned with developer lock + this checkout).

Activate:
  source ${CONDA_ROOT}/etc/profile.d/conda.sh
  conda activate ${ENV_NAME}
  # optional:
  # export PI_DEX_ARTIFACTS=...
  # export OPENPI_DATA_HOME=\${PI_DEX_ARTIFACTS}/openpi-data

GPU one-shot (serve + bridge on same machine; use --with-zenoh):
  bash scripts/run_robot_joint29d.sh --checkpoint-dir /path/to/step ...
  # NUC only runs Sharpa start.sh — do not install pi-dex on the slave.

Re-verify later:
  bash scripts/setup_inference_env.sh --verify-only --conda-root ${CONDA_ROOT}
EOF
