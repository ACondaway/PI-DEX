"""Lightweight metadata contract shared by PyTorch training and deployment."""

from __future__ import annotations

import dataclasses
from typing import Any

from pi_dex.actions import MODEL_ACTION_DIM
from pi_dex.spec import BimanualActionSpec

PADDING_LOSS_POLICY = "exclude_invalid_dimensions_v1"
PADDING_NOISE_POLICY = "zero_invalid_dimensions_v1"
PADDING_INFERENCE_POLICY = "zero_dense_action_io_parameters_v1"
OPENPI_MODEL_CONTRACT_KEY = "openpi_model"

_SUPPORTED_MODEL_DTYPES = frozenset({"bfloat16", "float32"})
_SUPPORTED_GEMMA_VARIANTS = frozenset(
    {"dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"}
)
_MODEL_STRING_FIELDS = ("dtype", "paligemma_variant", "action_expert_variant")


def openpi_model_contract_metadata(
    model_config: object,
    spec: BimanualActionSpec,
) -> dict[str, Any]:
    """Return the inference-relevant OpenPI model configuration contract.

    Args:
        model_config: OpenPI pi05 model config used to construct the checkpoint.
        spec: PI-DEX action contract that fixes action width and horizon.

    Returns:
        A fresh JSON-compatible mapping that binds model structure, tokenizer
        length/state encoding, and training precision. Compile mode is omitted
        because it is a runtime optimization and does not change these semantics.

    Raises:
        TypeError: If a required field is absent or has an invalid exact type.
        ValueError: If a field is empty, unsupported, or conflicts with ``spec``.
    """
    validated_spec = _validated_spec_copy(spec)
    validated_spec.validate_openpi_model_config(model_config)
    for field_name in (*_MODEL_STRING_FIELDS, "max_token_len", "discrete_state_input"):
        if not hasattr(model_config, field_name):
            raise TypeError(f"model_config: missing required attribute {field_name!r}")

    string_values: dict[str, str] = {}
    for field_name in _MODEL_STRING_FIELDS:
        value = getattr(model_config, field_name)
        if type(value) is not str:
            raise TypeError(
                f"model_config.{field_name}: expected str, got {type(value).__name__}"
            )
        if not value.strip():
            raise ValueError(f"model_config.{field_name}: expected a non-empty value")
        string_values[field_name] = value
    if string_values["dtype"] not in _SUPPORTED_MODEL_DTYPES:
        raise ValueError(
            "model_config.dtype: expected one of "
            f"{sorted(_SUPPORTED_MODEL_DTYPES)!r}, got {string_values['dtype']!r}"
        )
    for field_name in ("paligemma_variant", "action_expert_variant"):
        if string_values[field_name] not in _SUPPORTED_GEMMA_VARIANTS:
            raise ValueError(
                f"model_config.{field_name}: expected one of "
                f"{sorted(_SUPPORTED_GEMMA_VARIANTS)!r}, got {string_values[field_name]!r}"
            )

    max_token_len = getattr(model_config, "max_token_len")
    if type(max_token_len) is not int:
        raise TypeError(
            "model_config.max_token_len: expected int, "
            f"got {type(max_token_len).__name__}"
        )
    if max_token_len <= 0:
        raise ValueError(
            f"model_config.max_token_len: expected a positive integer, got {max_token_len}"
        )
    discrete_state_input = getattr(model_config, "discrete_state_input")
    if type(discrete_state_input) is not bool:
        raise TypeError(
            "model_config.discrete_state_input: expected bool, "
            f"got {type(discrete_state_input).__name__}"
        )
    if discrete_state_input is not True:
        raise ValueError("model_config.discrete_state_input: pi05 requires discrete state input")

    return {
        "pi05": True,
        "action_dim": MODEL_ACTION_DIM,
        "action_horizon": validated_spec.model_action_horizon,
        "dtype": string_values["dtype"],
        "paligemma_variant": string_values["paligemma_variant"],
        "action_expert_variant": string_values["action_expert_variant"],
        "max_token_len": max_token_len,
        "discrete_state_input": discrete_state_input,
    }


def training_contract_metadata(
    spec: BimanualActionSpec,
    model_config: object,
) -> dict[str, Any]:
    """Build the action, model, and padding portion of checkpoint metadata.

    Args:
        spec: PI-DEX action contract. A validated copy is recorded so an object
            whose frozen fields were bypassed cannot publish invalid metadata.
        model_config: Exact OpenPI pi05 model configuration used for training.

    Returns:
        A fresh JSON-compatible mapping for checkpoint and deployment validation.

    Raises:
        TypeError: If the spec or model config has invalid runtime types.
        ValueError: If either contract is internally inconsistent.
    """
    validated_spec = _validated_spec_copy(spec)
    return {
        "pi_dex": validated_spec.to_metadata(),
        OPENPI_MODEL_CONTRACT_KEY: openpi_model_contract_metadata(model_config, validated_spec),
        "pytorch_training": {
            "padding_loss_policy": PADDING_LOSS_POLICY,
            "padding_noise_policy": PADDING_NOISE_POLICY,
            "padding_inference_policy": PADDING_INFERENCE_POLICY,
            "checkpoint_model": "unwrapped_openpi_pi0_pytorch",
        },
    }


def _validated_spec_copy(spec: object) -> BimanualActionSpec:
    if not isinstance(spec, BimanualActionSpec):
        raise TypeError(f"spec: expected BimanualActionSpec, got {type(spec).__name__}")
    return dataclasses.replace(spec)
