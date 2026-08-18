#!/usr/bin/env bash
# Set up the vendored OpenPI uv environment (separate from root miniconda).
#
# Root PI-DEX development stays on conda env `pi-dex`.
# OpenPI sync / convert / full train must use this env.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_DIR="$ROOT/openpi"
SHM_ROOT="${PI_DEX_OPENPI_SHM:-/dev/shm/pi-dex-openpi}"
ARTIFACTS="${PI_DEX_ARTIFACTS:-/mnt/netdata/Team/Personal/congsheng/pi-dex-artifacts}"

mkdir -p "$SHM_ROOT"/{cache,venv,python,tools} "$ARTIFACTS"/{openpi-data,converted,runs,logs}

export UV_CACHE_DIR="$SHM_ROOT/cache"
export UV_PYTHON_INSTALL_DIR="$SHM_ROOT/python"
export UV_TOOL_DIR="$SHM_ROOT/tools"
export UV_PROJECT_ENVIRONMENT="$SHM_ROOT/venv"
export UV_LINK_MODE=copy
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$ARTIFACTS/openpi-data}"

UV_BIN="${UV_BIN:-$(command -v uv)}"
if [[ -z "$UV_BIN" ]]; then
  echo "uv not found; install uv or set UV_BIN" >&2
  exit 1
fi

cd "$OPENPI_DIR"
"$UV_BIN" sync --frozen

# Editable install of the root PI-DEX package into the OpenPI env.
"$UV_BIN" pip install -e "$ROOT"

# Apply transformers_replace into this env only (copy mode; never shared hardlink cache).
REPLACE_SRC="$OPENPI_DIR/src/openpi/models_pytorch/transformers_replace"
SITE="$("$UV_BIN" run python -c 'import transformers, pathlib; print(pathlib.Path(transformers.__file__).resolve().parent)')"
echo "transformers site: $SITE"
if [[ -d "$REPLACE_SRC" ]]; then
  while IFS= read -r -d '' src; do
    rel="${src#"$REPLACE_SRC"/}"
    dest="$SITE/$rel"
    mkdir -p "$(dirname "$dest")"
    cp -f "$src" "$dest"
  done < <(find "$REPLACE_SRC" -type f -print0)
fi

cat <<EOF
OpenPI env ready.

Activate for one shell:
  source $SHM_ROOT/venv/bin/activate
  export OPENPI_DATA_HOME=$OPENPI_DATA_HOME

Or:
  cd $OPENPI_DIR && UV_PROJECT_ENVIRONMENT=$SHM_ROOT/venv uv run python -c 'import openpi; import pi_dex'

Note: /dev/shm is cleared on reboot; re-run this script after reboot.
EOF
