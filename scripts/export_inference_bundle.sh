#!/usr/bin/env bash
# Pack a PI-DEX joint_29d **inference** bundle from a published training step.
#
# Copies only what ``pi-dex-serve`` needs (not optimizer / dataset / converted base):
#   ckpt/model.safetensors
#   ckpt/pi_dex.json
#   ckpt/assets/<asset_id>/norm_stats.json
#   openpi-data/big_vision/paligemma_tokenizer.model
#   configs/joint_29d_observation.reviewed.json
#
# Example (Insert_Battery final step):
#   bash scripts/export_inference_bundle.sh \
#     --checkpoint-dir /mnt/netdata/Team/Personal/congsheng/pi-dex-artifacts/runs/sharpa_joint_29d_insert_battery-20260817-083004/20000
#
# From a run root (picks the highest numeric step dir):
#   bash scripts/export_inference_bundle.sh \
#     --run-dir /mnt/netdata/Team/Personal/congsheng/pi-dex-artifacts/runs/sharpa_joint_29d_insert_battery-20260817-083004
#
# Output defaults to:
#   ${PI_DEX_ARTIFACTS}/exports/<run-name>-step<N>/
#
# Same-filesystem copies use hardlinks so the ~7G weights are instant; use
# ``--copy`` if the bundle must be a real duplicate. Then rsync/scp the export
# directory to the inference machine.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

: "${PI_DEX_ARTIFACTS:=/mnt/netdata/Team/Personal/congsheng/pi-dex-artifacts}"
: "${OPENPI_DATA_HOME:=${PI_DEX_ARTIFACTS}/openpi-data}"
: "${CONTRACT:=${ROOT}/configs/site/joint_29d_observation.reviewed.json}"

CHECKPOINT_DIR=""
RUN_DIR=""
OUTPUT_DIR=""
MODE="hardlink"
FORCE=0
DRY_RUN=0
PACK=0
INCLUDE_OPTIMIZER=0

usage() {
  cat <<'EOF'
Usage: bash scripts/export_inference_bundle.sh --checkpoint-dir <step-dir> [options]
       bash scripts/export_inference_bundle.sh --run-dir <run-root> [options]

Required (one of):
  --checkpoint-dir DIR   Published step directory (contains model.safetensors)
  --run-dir DIR          Training run root; uses the highest numeric step/

Options:
  --output-dir DIR       Export root (default: $PI_DEX_ARTIFACTS/exports/<name>-step<N>)
  --openpi-data-home DIR Tokenizer cache (default: $OPENPI_DATA_HOME)
  --contract FILE        Observation contract JSON (default: reviewed site contract)
  --copy                 Always byte-copy (default: hardlink when same filesystem)
  --force                Overwrite an existing output directory
  --pack                 Also write a .tar next to the export (no gzip; ~7G)
  --include-optimizer    Also copy optimizer.pt (not needed for serve)
  --dry-run              Print plan and exit
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --openpi-data-home) OPENPI_DATA_HOME="$2"; shift 2 ;;
    --contract) CONTRACT="$2"; shift 2 ;;
    --copy) MODE="copy"; shift ;;
    --force) FORCE=1; shift ;;
    --pack) PACK=1; shift ;;
    --include-optimizer) INCLUDE_OPTIMIZER=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${CHECKPOINT_DIR}" && -z "${RUN_DIR}" ]]; then
  echo "export_inference_bundle: pass --checkpoint-dir or --run-dir" >&2
  usage >&2
  exit 2
fi
if [[ -n "${CHECKPOINT_DIR}" && -n "${RUN_DIR}" ]]; then
  echo "export_inference_bundle: use only one of --checkpoint-dir / --run-dir" >&2
  exit 2
fi

resolve_latest_step() {
  local root="$1"
  local best=""
  local best_n=-1
  local name n
  [[ -d "${root}" ]] || {
    echo "export_inference_bundle: run-dir is not a directory: ${root}" >&2
    exit 1
  }
  shopt -s nullglob
  for path in "${root}"/*/; do
    name="$(basename "${path}")"
    if [[ "${name}" =~ ^[0-9]+$ ]]; then
      n=$((10#${name}))
      if (( n > best_n )) && [[ -f "${path}/model.safetensors" ]]; then
        best_n="${n}"
        best="$(cd "${path}" && pwd)"
      fi
    fi
  done
  shopt -u nullglob
  if [[ -z "${best}" ]]; then
    echo "export_inference_bundle: no numeric step dir with model.safetensors under ${root}" >&2
    exit 1
  fi
  printf '%s\n' "${best}"
}

if [[ -n "${RUN_DIR}" ]]; then
  CHECKPOINT_DIR="$(resolve_latest_step "${RUN_DIR}")"
fi
CHECKPOINT_DIR="$(cd "${CHECKPOINT_DIR}" && pwd)"

if [[ ! -f "${CHECKPOINT_DIR}/model.safetensors" ]]; then
  echo "export_inference_bundle: missing ${CHECKPOINT_DIR}/model.safetensors" >&2
  echo "  (pass a step directory like .../runs/<run>/20000, not the run root)" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT_DIR}/pi_dex.json" ]]; then
  echo "export_inference_bundle: missing ${CHECKPOINT_DIR}/pi_dex.json" >&2
  exit 1
fi
if [[ ! -f "${CONTRACT}" ]]; then
  echo "export_inference_bundle: missing contract: ${CONTRACT}" >&2
  exit 1
fi

TOKENIZER="${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model"
if [[ ! -f "${TOKENIZER}" ]]; then
  echo "export_inference_bundle: missing tokenizer: ${TOKENIZER}" >&2
  echo "  set --openpi-data-home or OPENPI_DATA_HOME" >&2
  exit 1
fi

META="$(python - "${CHECKPOINT_DIR}/pi_dex.json" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
norm = payload.get("normalization") or {}
pi = payload.get("pi_dex") or {}
pt = payload.get("pytorch_training") or {}
asset_id = norm.get("asset_id") or ""
asset_file = norm.get("asset_file") or (f"assets/{asset_id}/norm_stats.json" if asset_id else "")
print(asset_id)
print(asset_file)
print(pi.get("robot_id") or "")
print(pi.get("action_representation") or "")
print(pt.get("weights_file") or "model.safetensors")
PY
)"
ASSET_ID="$(printf '%s\n' "${META}" | sed -n '1p')"
ASSET_FILE="$(printf '%s\n' "${META}" | sed -n '2p')"
ROBOT_ID="$(printf '%s\n' "${META}" | sed -n '3p')"
ACTION_REP="$(printf '%s\n' "${META}" | sed -n '4p')"
WEIGHTS_FILE="$(printf '%s\n' "${META}" | sed -n '5p')"

if [[ -z "${ASSET_ID}" || -z "${ASSET_FILE}" ]]; then
  echo "export_inference_bundle: pi_dex.json missing normalization.asset_id / asset_file" >&2
  exit 1
fi
if [[ "${ASSET_FILE}" = /* || "${ASSET_FILE}" == *..* ]]; then
  echo "export_inference_bundle: unexpected asset_file ${ASSET_FILE!r}" >&2
  exit 1
fi

NORM_SRC="${CHECKPOINT_DIR}/${ASSET_FILE}"
if [[ ! -f "${NORM_SRC}" ]]; then
  echo "export_inference_bundle: missing ${NORM_SRC}" >&2
  exit 1
fi

STEP_NAME="$(basename "${CHECKPOINT_DIR}")"
RUN_NAME="$(basename "$(dirname "${CHECKPOINT_DIR}")")"
if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${PI_DEX_ARTIFACTS}/exports/${RUN_NAME}-step${STEP_NAME}"
fi
mkdir -p "$(dirname "${OUTPUT_DIR}")"
OUTPUT_DIR="$(python - "${OUTPUT_DIR}" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"

CKPT_DST="${OUTPUT_DIR}/ckpt"
TOKENIZER_DST="${OUTPUT_DIR}/openpi-data/big_vision/paligemma_tokenizer.model"
CONTRACT_DST="${OUTPUT_DIR}/configs/joint_29d_observation.reviewed.json"
NORM_DST="${CKPT_DST}/${ASSET_FILE}"

place_file() {
  local src="$1" dst="$2"
  mkdir -p "$(dirname "${dst}")"
  if [[ "${MODE}" == "hardlink" ]]; then
    if ln "${src}" "${dst}" 2>/dev/null; then
      return 0
    fi
  fi
  cp -a "${src}" "${dst}"
}

echo "export_inference_bundle"
echo "  checkpoint=${CHECKPOINT_DIR}"
echo "  output=${OUTPUT_DIR}"
echo "  asset_id=${ASSET_ID}"
echo "  robot_id=${ROBOT_ID:-<unknown>}"
echo "  action=${ACTION_REP:-<unknown>}"
echo "  mode=${MODE} pack=${PACK} include_optimizer=${INCLUDE_OPTIMIZER}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY files:"
  echo "  ${CHECKPOINT_DIR}/${WEIGHTS_FILE} -> ${CKPT_DST}/${WEIGHTS_FILE}"
  echo "  ${CHECKPOINT_DIR}/pi_dex.json -> ${CKPT_DST}/pi_dex.json"
  echo "  ${NORM_SRC} -> ${NORM_DST}"
  echo "  ${TOKENIZER} -> ${TOKENIZER_DST}"
  echo "  ${CONTRACT} -> ${CONTRACT_DST}"
  if [[ "${INCLUDE_OPTIMIZER}" == "1" && -f "${CHECKPOINT_DIR}/optimizer.pt" ]]; then
    echo "  ${CHECKPOINT_DIR}/optimizer.pt -> ${CKPT_DST}/optimizer.pt"
  fi
  exit 0
fi

if [[ -e "${OUTPUT_DIR}" ]]; then
  if [[ "${FORCE}" != "1" ]]; then
    echo "export_inference_bundle: refuse overwrite: ${OUTPUT_DIR} (pass --force)" >&2
    exit 1
  fi
  rm -rf "${OUTPUT_DIR}"
fi

mkdir -p "${CKPT_DST}" "$(dirname "${TOKENIZER_DST}")" "$(dirname "${CONTRACT_DST}")"
place_file "${CHECKPOINT_DIR}/${WEIGHTS_FILE}" "${CKPT_DST}/${WEIGHTS_FILE}"
place_file "${CHECKPOINT_DIR}/pi_dex.json" "${CKPT_DST}/pi_dex.json"
place_file "${NORM_SRC}" "${NORM_DST}"
place_file "${TOKENIZER}" "${TOKENIZER_DST}"
place_file "${CONTRACT}" "${CONTRACT_DST}"
if [[ "${INCLUDE_OPTIMIZER}" == "1" && -f "${CHECKPOINT_DIR}/optimizer.pt" ]]; then
  place_file "${CHECKPOINT_DIR}/optimizer.pt" "${CKPT_DST}/optimizer.pt"
fi

python - "${OUTPUT_DIR}" "${CHECKPOINT_DIR}" "${ASSET_ID}" "${ASSET_FILE}" "${ROBOT_ID}" "${ACTION_REP}" "${CONTRACT}" "${TOKENIZER}" "${MODE}" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

out, src, asset_id, asset_file, robot_id, action_rep, contract, tokenizer, mode = sys.argv[1:]
out_p = Path(out)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


files = []
for path in sorted(p for p in out_p.rglob("*") if p.is_file() and p.name not in {"MANIFEST.json", "README.md", "serve.env"}):
    rel = path.relative_to(out_p).as_posix()
    files.append(
        {
            "path": rel,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    )

manifest = {
    "kind": "pi-dex-inference-bundle",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "source_checkpoint_dir": src,
    "asset_id": asset_id,
    "asset_file": asset_file,
    "robot_id": robot_id,
    "action_representation": action_rep,
    "source_contract": contract,
    "source_tokenizer": tokenizer,
    "copy_mode": mode,
    "hostname": os.uname().nodename,
    "files": files,
}
(out_p / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

serve_env = f"""# Source on the inference GPU after unpacking this directory.
# Example: set -a; source serve.env; set +a
BUNDLE_ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
export OPENPI_DATA_HOME="${{BUNDLE_ROOT}}/openpi-data"
CHECKPOINT_DIR="${{BUNDLE_ROOT}}/ckpt"
ASSETS_DIR="${{BUNDLE_ROOT}}/ckpt/assets"
ASSET_ID={json.dumps(asset_id)}
ROBOT_ID={json.dumps(robot_id)}
CONTRACT="${{BUNDLE_ROOT}}/configs/joint_29d_observation.reviewed.json"
HOST=0.0.0.0
PORT=8000
"""
(out_p / "serve.env").write_text(serve_env, encoding="utf-8")

readme = f"""# PI-DEX inference bundle

Source checkpoint: `{src}`
asset_id: `{asset_id}`
robot_id: `{robot_id or "<set --robot-id>"}`
action_representation: `{action_rep}`

This directory is **serve-only**. It does not include `optimizer.pt`, HDF5 data,
or the converted `pi05_base` init weights.

## Layout

```text
ckpt/model.safetensors
ckpt/pi_dex.json
ckpt/{asset_file}
openpi-data/big_vision/paligemma_tokenizer.model
configs/joint_29d_observation.reviewed.json
serve.env
MANIFEST.json
```

## On the inference GPU

1. Sync this whole directory (rsync/scp). Keep relative paths.
2. Clone PI-DEX at the same commit as training and install the env
   (`docs/inference-env.md`).
3. Serve:

```bash
cd /path/to/this/bundle
set -a; source serve.env; set +a
conda activate pi-dex

bash /path/to/PI-DEX/scripts/serve_joint29d.sh \\
  --checkpoint-dir "${{CHECKPOINT_DIR}}" \\
  --assets-dir "${{ASSETS_DIR}}" \\
  --asset-id "${{ASSET_ID}}" \\
  --robot-id "${{ROBOT_ID}}" \\
  --contract "${{CONTRACT}}" \\
  --host 0.0.0.0 \\
  --port 8000
```

`--checkpoint-dir` must be `ckpt/` (the step contents), not a training run root.

## Checksums

See `MANIFEST.json` (`sha256` per file).
"""
(out_p / "README.md").write_text(readme, encoding="utf-8")
print(f"wrote manifest ({len(files)} files, {sum(item['bytes'] for item in files)} bytes)")
PY

if [[ "${PACK}" == "1" ]]; then
  TAR="${OUTPUT_DIR}.tar"
  echo "packing ${TAR} (this may take a few minutes for ~7G)"
  tar -C "$(dirname "${OUTPUT_DIR}")" -cf "${TAR}" "$(basename "${OUTPUT_DIR}")"
  echo "  tar=${TAR}"
fi

echo "export_inference_bundle done"
echo "  bundle=${OUTPUT_DIR}"
echo "  rsync -aH --progress ${OUTPUT_DIR}/ <gpu>:/path/to/pi-dex-infer/"
