"""Strict loading of converted ``pi05_base`` PyTorch initialization weights.

Converted base directories are initialization artifacts only: they must contain
``model.safetensors`` and must not be treated as deployable PI-DEX checkpoints.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import pathlib
from typing import Any

import torch

from pi_dex.core.actions import MODEL_ACTION_DIM
from pi_dex.core.actions import PRETRAINED_MODEL_ACTION_DIM
from pi_dex.training.checkpoints import MODEL_WEIGHTS_FILENAME
from pi_dex.training.pytorch_training import expand_action_projections_from_pretrained

_EXPECTED_PALIGEMMA = "gemma_2b"
_EXPECTED_EXPERT = "gemma_300m"
_ACTION_IN_WEIGHT_KEY = "action_in_proj.weight"
_ACTION_OUT_WEIGHT_KEY = "action_out_proj.weight"


def file_sha256(path: pathlib.Path | str) -> str:
    """Return lowercase hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def require_converted_base_dir(converted_base_dir: pathlib.Path | str) -> pathlib.Path:
    """Validate a converted base directory layout and return its path."""
    directory = pathlib.Path(converted_base_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"converted_base_dir: not a directory: {directory}")
    weights = directory / MODEL_WEIGHTS_FILENAME
    if not weights.is_file():
        raise FileNotFoundError(f"converted_base_dir: missing {MODEL_WEIGHTS_FILENAME} under {directory}")
    return directory


def checkpoint_action_projection_dim(weights_path: pathlib.Path | str) -> int | None:
    """Return the action input width stored in a PyTorch safetensors checkpoint."""
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ImportError("safetensors is required to inspect checkpoint action width") from error

    path = pathlib.Path(weights_path)
    with safe_open(str(path), framework="pt") as handle:
        if _ACTION_IN_WEIGHT_KEY not in handle.keys():
            return None
        tensor = handle.get_tensor(_ACTION_IN_WEIGHT_KEY)
    return int(tensor.shape[1])


def load_pi05_pytorch_weights(
    model: Any,
    weights_path: pathlib.Path | str,
    *,
    source_dim: int = PRETRAINED_MODEL_ACTION_DIM,
    target_dim: int = MODEL_ACTION_DIM,
) -> Mapping[str, Any]:
    """Load ``model.safetensors`` and expand 32D action projections when needed.

    When the checkpoint stores ``source_dim`` action projections and the live
    model expects ``target_dim``, pretrained rows/columns are copied and the
    motor block is zero-initialized via
    :func:`pi_dex.training.pytorch_training.expand_action_projections_from_pretrained`.

    Args:
        model: Instantiated OpenPI ``PI0Pytorch`` (or compatible) module.
        weights_path: Path to ``model.safetensors``.
        source_dim: Pretrained action projection width.
        target_dim: PI-DEX model action projection width.

    Returns:
        Metadata including whether action projections were expanded.

    Raises:
        ImportError: If ``safetensors`` is unavailable.
        ValueError: If ``load_model`` reports missing or unexpected keys.
    """
    path = pathlib.Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"weights_path: expected file, got {path}")

    try:
        from safetensors.torch import load_model
    except ImportError as error:
        raise ImportError("safetensors is required to load converted pi05_base weights") from error

    checkpoint_dim = checkpoint_action_projection_dim(path)
    model_input_dim = getattr(getattr(model, "action_in_proj", None), "in_features", None)
    needs_expand = (
        checkpoint_dim == source_dim
        and model_input_dim == target_dim
        and source_dim < target_dim
    )
    if needs_expand:
        _swap_action_projections_for_pretrained_load(model, source_dim=source_dim)

    incompatible = load_model(model, str(path), strict=True)
    missing = tuple(getattr(incompatible, "missing_keys", ()) or ())
    unexpected = tuple(getattr(incompatible, "unexpected_keys", ()) or ())
    if missing or unexpected:
        raise ValueError(
            "pi05 weight load_model reported unresolved keys: "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )

    expanded = False
    if needs_expand or (
        checkpoint_dim == source_dim
        and getattr(model.action_in_proj, "in_features", None) == source_dim
        and target_dim > source_dim
    ):
        expanded = expand_action_projections_from_pretrained(
            model,
            source_dim=source_dim,
            target_dim=target_dim,
        )
    return {
        "weights_file": str(path),
        "weights_sha256": file_sha256(path),
        "expanded_action_projections": expanded,
        "checkpoint_action_dim": checkpoint_dim,
        "model_action_dim": getattr(getattr(model, "action_in_proj", None), "in_features", None),
    }


def materialize_expanded_checkpoint_weights(
    weights_path: pathlib.Path | str,
    *,
    source_dim: int = PRETRAINED_MODEL_ACTION_DIM,
    target_dim: int = MODEL_ACTION_DIM,
) -> bool:
    """Rewrite ``model.safetensors`` in place when action projections are still 32D.

    OpenPI's stock ``load_pytorch`` path cannot load mismatched action projection
    shapes. Deployment copies checkpoints into a private snapshot; expanding the
    weights file there keeps upstream policy loading unchanged.

    Returns:
        ``True`` when the file was expanded and rewritten, ``False`` when already
        at ``target_dim`` or no action projection keys were found.
    """
    path = pathlib.Path(weights_path)
    checkpoint_dim = checkpoint_action_projection_dim(path)
    if checkpoint_dim is None or checkpoint_dim >= target_dim:
        return False
    if checkpoint_dim != source_dim:
        raise ValueError(
            f"checkpoint action projection width: expected {source_dim} or {target_dim}, "
            f"got {checkpoint_dim}"
        )

    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise ImportError("torch and safetensors are required to rewrite checkpoint weights") from error

    expanded_state = expand_action_projection_state_dict(
        _load_safetensors_state(path),
        source_dim=source_dim,
        target_dim=target_dim,
    )
    save_file(expanded_state, str(path))
    return True


def expand_action_projection_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    source_dim: int = PRETRAINED_MODEL_ACTION_DIM,
    target_dim: int = MODEL_ACTION_DIM,
) -> dict[str, torch.Tensor]:
    """Return a copy of ``state_dict`` with 32D action projections expanded to 36D."""
    if source_dim >= target_dim:
        raise ValueError(
            f"expand_action_projection_state_dict: source_dim ({source_dim}) "
            f"must be less than target_dim ({target_dim})"
        )
    required = (_ACTION_IN_WEIGHT_KEY, _ACTION_OUT_WEIGHT_KEY)
    for key in required:
        if key not in state_dict:
            raise ValueError(f"state_dict missing required key {key!r}")

    input_weight = state_dict[_ACTION_IN_WEIGHT_KEY]
    output_weight = state_dict[_ACTION_OUT_WEIGHT_KEY]
    if input_weight.shape[1] == target_dim:
        return dict(state_dict)
    if input_weight.shape[1] != source_dim or output_weight.shape[0] != source_dim:
        raise ValueError(
            "state_dict action projection shapes incompatible with expansion: "
            f"in={tuple(input_weight.shape)}, out={tuple(output_weight.shape)}"
        )

    hidden_dim = int(input_weight.shape[0])
    expanded: dict[str, torch.Tensor] = dict(state_dict)
    new_input_weight = torch.zeros(
        (hidden_dim, target_dim),
        dtype=input_weight.dtype,
        device=input_weight.device,
    )
    new_input_weight[:, :source_dim] = input_weight
    expanded[_ACTION_IN_WEIGHT_KEY] = new_input_weight
    if "action_in_proj.bias" in state_dict:
        expanded["action_in_proj.bias"] = state_dict["action_in_proj.bias"].clone()

    new_output_weight = torch.zeros(
        (target_dim, hidden_dim),
        dtype=output_weight.dtype,
        device=output_weight.device,
    )
    new_output_weight[:source_dim, :] = output_weight
    expanded[_ACTION_OUT_WEIGHT_KEY] = new_output_weight
    if "action_out_proj.bias" in state_dict:
        output_bias = state_dict["action_out_proj.bias"]
        new_output_bias = torch.zeros(
            (target_dim,),
            dtype=output_bias.dtype,
            device=output_bias.device,
        )
        new_output_bias[:source_dim] = output_bias
        expanded["action_out_proj.bias"] = new_output_bias
    return expanded


def load_verified_pi05_base(
    model: Any,
    converted_base_dir: pathlib.Path | str,
    *,
    expected_weights_sha256: str | None = None,
) -> Mapping[str, str]:
    """Strict-load converted ``model.safetensors`` into an OpenPI ``PI0Pytorch``.

    Args:
        model: Instantiated OpenPI ``PI0Pytorch`` (or compatible) module.
        converted_base_dir: Directory containing ``model.safetensors``.
        expected_weights_sha256: Optional lowercase hex digest that must match the
            weights file before any tensors are loaded.

    Returns:
        Provenance mapping with weights path and SHA-256.

    Raises:
        FileNotFoundError: If the weights file is missing.
        ValueError: If the digest mismatches, model config is not pi05 with the
            hard-coded Gemma variants, or ``load_state_dict`` reports missing or
            unexpected keys.
        ImportError: If ``safetensors`` is unavailable.
    """
    directory = require_converted_base_dir(converted_base_dir)
    weights_path = directory / MODEL_WEIGHTS_FILENAME
    actual_sha256 = file_sha256(weights_path)
    if expected_weights_sha256 is not None:
        expected = expected_weights_sha256.strip().lower()
        if actual_sha256 != expected:
            raise ValueError(f"converted base weights SHA-256 mismatch: expected {expected}, got {actual_sha256}")

    config = getattr(model, "config", None)
    if config is None:
        raise TypeError("model: expected an OpenPI module exposing .config")
    if getattr(config, "pi05", None) is not True:
        raise ValueError("model.config.pi05: expected True for pi05_base initialization")
    if getattr(config, "paligemma_variant", None) != _EXPECTED_PALIGEMMA:
        raise ValueError(
            f"model.config.paligemma_variant: expected {_EXPECTED_PALIGEMMA!r}, "
            f"got {getattr(config, 'paligemma_variant', None)!r}"
        )
    if getattr(config, "action_expert_variant", None) != _EXPECTED_EXPERT:
        raise ValueError(
            f"model.config.action_expert_variant: expected {_EXPECTED_EXPERT!r}, "
            f"got {getattr(config, 'action_expert_variant', None)!r}"
        )

    load_meta = load_pi05_pytorch_weights(model, weights_path)
    return {
        "weights_file": str(weights_path),
        "weights_sha256": actual_sha256,
        "converted_base_dir": str(directory),
        "expanded_action_projections": str(load_meta.get("expanded_action_projections", False)),
    }


def _swap_action_projections_for_pretrained_load(model: Any, *, source_dim: int) -> None:
    from torch import nn

    action_input_projection = model.action_in_proj
    action_output_projection = model.action_out_proj
    if not isinstance(action_input_projection, nn.Linear) or not isinstance(action_output_projection, nn.Linear):
        raise TypeError("model action projections: expected torch.nn.Linear")
    hidden_dim = action_input_projection.out_features
    device = action_input_projection.weight.device
    dtype = action_input_projection.weight.dtype
    model.action_in_proj = nn.Linear(
        source_dim,
        hidden_dim,
        bias=action_input_projection.bias is not None,
        device=device,
        dtype=dtype,
    )
    model.action_out_proj = nn.Linear(
        hidden_dim,
        source_dim,
        bias=action_output_projection.bias is not None,
        device=device,
        dtype=dtype,
    )


def _load_safetensors_state(path: pathlib.Path) -> dict[str, Any]:
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise ImportError("safetensors is required to load checkpoint tensors") from error
    return load_file(str(path))
