"""Minimal training checkpoint manager for PI-DEX joint_29d loops.

Atomically publishes:

* ``model.safetensors``
* ``optimizer.pt`` / ``rng.pt`` / ``train_state.json``
* ``assets/<asset_id>/norm_stats.json`` when provided
* ``pi_dex.json`` via :func:`pi_dex.training.checkpoints.save_training_contract`

``train_state.json`` must include a ``sampler_state`` with a deterministic sample
order seed and the next sample cursor for resume.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
import pathlib
import shutil
import tempfile
from typing import Any

import torch

from pi_dex.training.checkpoints import MODEL_WEIGHTS_FILENAME
from pi_dex.training.checkpoints import NORMALIZATION_ASSET_FILENAME
from pi_dex.training.checkpoints import save_training_contract
from pi_dex.weights.pi05_weights import file_sha256
from pi_dex.core.spec import BimanualActionSpec

TRAIN_STATE_FILENAME = "train_state.json"
OPTIMIZER_FILENAME = "optimizer.pt"
RNG_FILENAME = "rng.pt"
PARAMETER_MANIFEST_FILENAME = "parameter_manifests.json"


def publish_training_checkpoint(
    *,
    publish_dir: pathlib.Path | str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    spec: BimanualActionSpec,
    model_config: object,
    norm_stats: Mapping[str, Any],
    asset_id: str,
    global_step: int,
    run_id: str,
    action_representation: str,
    parent_base_provenance: Mapping[str, Any],
    sampler_state: Mapping[str, Any],
    parameter_manifests: Mapping[str, Any],
    extra_train_state: Mapping[str, Any] | None = None,
) -> pathlib.Path:
    """Write a complete training step snapshot to a new directory."""
    destination = pathlib.Path(publish_dir)
    if destination.exists():
        raise FileExistsError(f"publish_dir: refuse to overwrite {destination}")
    if type(global_step) is not int or global_step < 0:
        raise ValueError(f"global_step: expected non-negative int, got {global_step!r}")
    _validate_sampler_state(sampler_state)

    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(tempfile.mkdtemp(prefix=".pi_dex_ckpt_staging_", dir=str(parent)))
    try:
        weights_path = staging / MODEL_WEIGHTS_FILENAME
        from safetensors.torch import save_model

        from pi_dex.training.distributed import unwrap_model

        save_model(unwrap_model(model), str(weights_path))
        torch.save(optimizer.state_dict(), staging / OPTIMIZER_FILENAME)

        rng_payload = {
            "torch_cpu": torch.random.get_rng_state(),
            "numpy": None,
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        try:
            import numpy as np

            rng_payload["numpy"] = np.random.get_state()
        except Exception:
            pass
        torch.save(rng_payload, staging / RNG_FILENAME)

        assets_dir = staging / "assets" / asset_id
        assets_dir.mkdir(parents=True, exist_ok=True)
        _save_openpi_norm_stats(assets_dir, norm_stats)

        (staging / PARAMETER_MANIFEST_FILENAME).write_text(
            json.dumps(parameter_manifests, indent=2) + "\n",
            encoding="utf-8",
        )

        train_state = {
            "schema_version": 2,
            "global_step": global_step,
            "run_id": run_id,
            "action_representation": action_representation,
            "parent_base_provenance": dict(parent_base_provenance),
            "sampler_state": dict(sampler_state),
            "parameter_manifests_file": PARAMETER_MANIFEST_FILENAME,
            "parameter_manifests_sha256": file_sha256(staging / PARAMETER_MANIFEST_FILENAME),
            "model_weights_sha256": file_sha256(weights_path),
        }
        if extra_train_state:
            overlap = set(extra_train_state) & set(train_state)
            if overlap:
                raise ValueError(f"extra_train_state overlaps reserved keys: {sorted(overlap)}")
            train_state.update(dict(extra_train_state))
        (staging / TRAIN_STATE_FILENAME).write_text(json.dumps(train_state, indent=2) + "\n", encoding="utf-8")

        save_training_contract(
            staging,
            spec,
            model_config=model_config,
            norm_stats=norm_stats,
            asset_id=asset_id,
        )
        os.rename(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def resume_training_checkpoint(
    *,
    checkpoint_dir: pathlib.Path | str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """Restore model/optimizer/RNG from a published training checkpoint."""
    directory = pathlib.Path(checkpoint_dir)
    weights = directory / MODEL_WEIGHTS_FILENAME
    if not weights.is_file():
        raise FileNotFoundError(f"checkpoint_dir: missing {MODEL_WEIGHTS_FILENAME}")
    from pi_dex.weights.pi05_weights import load_pi05_pytorch_weights

    load_pi05_pytorch_weights(model, weights)

    optimizer.load_state_dict(torch.load(directory / OPTIMIZER_FILENAME, map_location="cpu", weights_only=False))
    rng_path = directory / RNG_FILENAME
    if rng_path.is_file():
        rng_payload = torch.load(rng_path, map_location="cpu", weights_only=False)
        if rng_payload.get("torch_cpu") is not None:
            torch.random.set_rng_state(rng_payload["torch_cpu"])
        if rng_payload.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_payload["cuda"])
        if rng_payload.get("numpy") is not None:
            import numpy as np

            np.random.set_state(rng_payload["numpy"])

    train_state = json.loads((directory / TRAIN_STATE_FILENAME).read_text(encoding="utf-8"))
    if "sampler_state" in train_state:
        _validate_sampler_state(train_state["sampler_state"])
    return train_state


def build_sample_order(length: int, *, seed: int) -> tuple[int, ...]:
    """Deterministic permutation used as the training data cursor order."""
    if type(length) is not int or length <= 0:
        raise ValueError(f"length: expected positive int, got {length!r}")
    if type(seed) is not int:
        raise TypeError(f"seed: expected int, got {type(seed).__name__}")
    generator = torch.Generator()
    generator.manual_seed(seed)
    order = torch.randperm(length, generator=generator).tolist()
    return tuple(int(index) for index in order)


def _validate_sampler_state(sampler_state: Mapping[str, Any]) -> None:
    required = {"seed", "dataset_length", "order_sha256", "next_sample_index", "batch_size"}
    if set(sampler_state) < required:
        raise ValueError(f"sampler_state: missing keys {sorted(required - set(sampler_state))}")
    if type(sampler_state["seed"]) is not int:
        raise TypeError("sampler_state.seed: expected int")
    if type(sampler_state["dataset_length"]) is not int or sampler_state["dataset_length"] <= 0:
        raise ValueError("sampler_state.dataset_length: expected positive int")
    if type(sampler_state["next_sample_index"]) is not int or sampler_state["next_sample_index"] < 0:
        raise ValueError("sampler_state.next_sample_index: expected non-negative int")
    if type(sampler_state["batch_size"]) is not int or sampler_state["batch_size"] <= 0:
        raise ValueError("sampler_state.batch_size: expected positive int")
    if type(sampler_state["order_sha256"]) is not str or len(sampler_state["order_sha256"]) != 64:
        raise ValueError("sampler_state.order_sha256: expected 64-char hex digest")
    if "world_size" in sampler_state:
        if type(sampler_state["world_size"]) is not int or sampler_state["world_size"] <= 0:
            raise ValueError("sampler_state.world_size: expected positive int")
    if "global_batch_size" in sampler_state:
        if type(sampler_state["global_batch_size"]) is not int or sampler_state["global_batch_size"] <= 0:
            raise ValueError("sampler_state.global_batch_size: expected positive int")


def order_sha256(order: Sequence[int]) -> str:
    import hashlib

    payload = json.dumps(list(order), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _save_openpi_norm_stats(asset_dir: pathlib.Path, norm_stats: Mapping[str, Any]) -> None:
    try:
        from openpi.shared import normalize as openpi_normalize
    except ImportError:
        serialized: dict[str, dict[str, list[float]]] = {}
        for key, stats in norm_stats.items():
            serialized[key] = {
                field: _as_float_list(getattr(stats, field) if hasattr(stats, field) else stats[field])
                for field in ("mean", "std", "q01", "q99")
            }
        (asset_dir / NORMALIZATION_ASSET_FILENAME).write_text(
            json.dumps(serialized, indent=2) + "\n",
            encoding="utf-8",
        )
        return
    openpi_normalize.save(asset_dir, dict(norm_stats))


def _as_float_list(values: Any) -> list[float]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return [float(value) for value in array.tolist()]
