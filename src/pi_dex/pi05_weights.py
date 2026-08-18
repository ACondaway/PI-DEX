"""Strict loading of converted ``pi05_base`` PyTorch initialization weights.

Converted base directories are initialization artifacts only: they must contain
``model.safetensors`` and must not be treated as deployable PI-DEX checkpoints.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import pathlib
from typing import Any

from pi_dex.checkpoints import MODEL_WEIGHTS_FILENAME

_EXPECTED_PALIGEMMA = "gemma_2b"
_EXPECTED_EXPERT = "gemma_300m"


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

    try:
        from safetensors.torch import load_model
    except ImportError as error:
        raise ImportError("safetensors is required to load converted pi05_base weights") from error

    # Prefer load_model so tied embedding weights (e.g. embed_tokens) are restored.
    incompatible = load_model(model, str(weights_path), strict=True)
    missing = tuple(getattr(incompatible, "missing_keys", ()) or ())
    unexpected = tuple(getattr(incompatible, "unexpected_keys", ()) or ())
    if missing or unexpected:
        raise ValueError(
            "converted base load_model reported unresolved keys: "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )
    return {
        "weights_file": str(weights_path),
        "weights_sha256": actual_sha256,
        "converted_base_dir": str(directory),
    }
