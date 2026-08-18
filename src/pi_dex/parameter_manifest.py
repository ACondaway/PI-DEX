"""Parameter and trainable-parameter manifests for full fine-tuning audits."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from torch import nn


def build_parameter_manifests(model: nn.Module) -> dict[str, Any]:
    """Return sorted all-parameter and trainable-parameter manifests with hashes.

    Full fine-tuning requires the trainable set to equal the full parameter set.
    """
    all_rows = _rows(model, trainable_only=False)
    trainable_rows = _rows(model, trainable_only=True)
    all_names = {row["name"] for row in all_rows}
    trainable_names = {row["name"] for row in trainable_rows}
    return {
        "all_parameters": all_rows,
        "trainable_parameters": trainable_rows,
        "all_parameters_sha256": _rows_sha256(all_rows),
        "trainable_parameters_sha256": _rows_sha256(trainable_rows),
        "sets_equal": all_names == trainable_names,
        "frozen_parameters": sorted(all_names - trainable_names),
        "total_numel": int(sum(row["numel"] for row in all_rows)),
        "trainable_numel": int(sum(row["numel"] for row in trainable_rows)),
    }


def require_full_finetune_manifest(manifest: dict[str, Any]) -> None:
    """Reject experiments that freeze parameters while claiming full fine-tuning."""
    if not manifest.get("sets_equal", False):
        frozen = manifest.get("frozen_parameters", [])
        raise ValueError(
            "full fine-tuning requires trainable parameters == all parameters; "
            f"frozen={frozen[:16]}{'...' if len(frozen) > 16 else ''}"
        )


def _rows(model: nn.Module, *, trainable_only: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if trainable_only and not parameter.requires_grad:
            continue
        rows.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "numel": int(parameter.numel()),
                "dtype": str(parameter.dtype).replace("torch.", ""),
                "requires_grad": bool(parameter.requires_grad),
            }
        )
    return rows


def _rows_sha256(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
