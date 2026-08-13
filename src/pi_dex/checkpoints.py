"""PI-DEX metadata stored alongside OpenPI-compatible PyTorch checkpoints."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import re
import uuid
from collections.abc import Mapping
from typing import Any

import numpy as np

from pi_dex.normalization import NORMALIZATION_FINGERPRINT_ALGORITHM
from pi_dex.normalization import normalization_state_dim
from pi_dex.normalization import normalization_stats_fingerprint
from pi_dex.spec import BimanualActionSpec
from pi_dex.training_contract import OPENPI_MODEL_CONTRACT_KEY
from pi_dex.training_contract import PADDING_INFERENCE_POLICY
from pi_dex.training_contract import PADDING_LOSS_POLICY
from pi_dex.training_contract import PADDING_NOISE_POLICY
from pi_dex.training_contract import openpi_model_contract_metadata
from pi_dex.training_contract import training_contract_metadata

CHECKPOINT_METADATA_FILENAME = "pi_dex.json"
MODEL_WEIGHTS_FILENAME = "model.safetensors"
NORMALIZATION_METADATA_KEY = "normalization"
NORMALIZATION_ASSET_FILENAME = "norm_stats.json"
NORMALIZATION_ASSET_FILE_FINGERPRINT_ALGORITHM = "sha256-bytes-v1"
MODEL_WEIGHTS_FINGERPRINT_ALGORITHM = "sha256-bytes-v1"
_ASSET_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def save_training_contract(
    checkpoint_dir: pathlib.Path | str,
    spec: BimanualActionSpec,
    *,
    model_config: object,
    norm_stats: Mapping[str, Any],
    asset_id: str,
) -> pathlib.Path:
    """Atomically save PI-DEX metadata beside an OpenPI PyTorch checkpoint.

    Args:
        checkpoint_dir: Directory containing ``model.safetensors`` and assets.
        spec: Training action contract.
        model_config: Exact OpenPI pi05 config used to construct the saved model.
        norm_stats: Complete ``state`` and per-hand pi0.5 normalization stats.
        asset_id: OpenPI asset directory identifier containing ``norm_stats``.

    Returns:
        Path to the written ``pi_dex.json``.

    Raises:
        TypeError: If ``asset_id`` or normalization values have invalid types.
        ValueError: If ``asset_id`` is empty or normalization values violate the
            PI-DEX contract.
        FileNotFoundError: If model weights or the serialized normalization asset
            have not been staged in ``checkpoint_dir``.
        OSError: If artifacts cannot be read or metadata cannot be written.

    Notes:
        This function does not save model parameters. Save the original unwrapped
        OpenPI ``PI0Pytorch`` instance so state-dict keys remain compatible. In
        distributed training, only the main rank may call this function, inside
        the same temporary directory that will atomically publish the checkpoint.
    """
    validated_spec = _validated_spec_copy(spec)
    normalized_asset_id = validate_normalization_asset_id(asset_id, field_name="asset_id")
    metadata = training_contract_metadata(validated_spec, model_config)
    fingerprint = normalization_stats_fingerprint(norm_stats, validated_spec, require_state=True)
    state_dim = normalization_state_dim(norm_stats, validated_spec)
    directory = pathlib.Path(checkpoint_dir)
    weights_path, normalization_asset_path, relative_asset_path = _require_checkpoint_artifacts(
        directory,
        asset_id=normalized_asset_id,
    )
    serialized_stats = _load_serialized_normalization_stats(normalization_asset_path)
    serialized_fingerprint = normalization_stats_fingerprint(
        serialized_stats,
        validated_spec,
        require_state=True,
    )
    if serialized_fingerprint != fingerprint:
        raise ValueError("checkpoint normalization asset content does not match supplied norm_stats")
    asset_file_fingerprint = _file_sha256(normalization_asset_path)
    weights_fingerprint = _file_sha256(weights_path)
    metadata["pytorch_training"].update(
        {
            "weights_file": MODEL_WEIGHTS_FILENAME,
            "weights_fingerprint_algorithm": MODEL_WEIGHTS_FINGERPRINT_ALGORITHM,
            "weights_fingerprint": weights_fingerprint,
        }
    )
    metadata[NORMALIZATION_METADATA_KEY] = {
        "asset_id": normalized_asset_id,
        "fingerprint_algorithm": NORMALIZATION_FINGERPRINT_ALGORITHM,
        "fingerprint": fingerprint,
        "state_dim": state_dim,
        "asset_file": relative_asset_path,
        "asset_file_fingerprint_algorithm": NORMALIZATION_ASSET_FILE_FINGERPRINT_ALGORITHM,
        "asset_file_fingerprint": asset_file_fingerprint,
    }
    serialized = json.dumps(metadata, indent=2, sort_keys=True) + "\n"

    metadata_path = directory / CHECKPOINT_METADATA_FILENAME
    temporary_path = directory / f".{CHECKPOINT_METADATA_FILENAME}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        temporary_path.write_text(serialized, encoding="utf-8")
        temporary_path.replace(metadata_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return metadata_path


def load_and_validate_training_contract(
    checkpoint_dir: pathlib.Path | str,
    spec: BimanualActionSpec,
    *,
    model_config: object,
    norm_stats: Mapping[str, Any],
    asset_id: str,
) -> dict[str, Any]:
    """Load checkpoint metadata and reject incompatible resume/deployment use.

    Args:
        checkpoint_dir: Directory containing ``pi_dex.json``.
        spec: Expected current action contract.
        model_config: Exact OpenPI pi05 model config used for resume or serving.
        norm_stats: Stats loaded from the bound checkpoint asset, validated
            against the sidecar semantic fingerprint.
        asset_id: Current OpenPI normalization asset identifier.

    Returns:
        Parsed metadata mapping after validation.

    Raises:
        FileNotFoundError: If metadata, model weights, or the bound normalization
            asset is absent.
        ValueError: If JSON/root/training/normalization fields are invalid or
            incompatible.
        TypeError: If metadata, ``asset_id``, or normalization values have invalid
            types.
    """
    validated_spec = _validated_spec_copy(spec)
    metadata_path = pathlib.Path(checkpoint_dir) / CHECKPOINT_METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"PI-DEX checkpoint metadata not found: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid PI-DEX checkpoint metadata JSON: {metadata_path}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"PI-DEX checkpoint metadata root must be an object: {metadata_path}")
    expected_root_fields = {
        "pi_dex",
        OPENPI_MODEL_CONTRACT_KEY,
        "pytorch_training",
        NORMALIZATION_METADATA_KEY,
    }
    if set(metadata) != expected_root_fields:
        raise ValueError(
            "checkpoint metadata fields: "
            f"expected exactly {sorted(expected_root_fields)!r}, "
            f"got {sorted(metadata, key=str)!r}"
        )

    validated_spec.validate_metadata(metadata)
    expected_model_contract = openpi_model_contract_metadata(model_config, validated_spec)
    _validate_exact_metadata_mapping(
        metadata.get(OPENPI_MODEL_CONTRACT_KEY),
        expected=expected_model_contract,
        field_name=f"checkpoint metadata {OPENPI_MODEL_CONTRACT_KEY!r}",
    )
    training_metadata = metadata.get("pytorch_training")
    if not isinstance(training_metadata, Mapping):
        raise ValueError("checkpoint metadata 'pytorch_training' must be an object")
    expected_fields = {
        "padding_loss_policy": PADDING_LOSS_POLICY,
        "padding_noise_policy": PADDING_NOISE_POLICY,
        "padding_inference_policy": PADDING_INFERENCE_POLICY,
        "checkpoint_model": "unwrapped_openpi_pi0_pytorch",
        "weights_file": MODEL_WEIGHTS_FILENAME,
        "weights_fingerprint_algorithm": MODEL_WEIGHTS_FINGERPRINT_ALGORITHM,
    }
    expected_training_fields = {*expected_fields, "weights_fingerprint"}
    if set(training_metadata) != expected_training_fields:
        raise ValueError(
            "checkpoint metadata 'pytorch_training' fields: "
            f"expected exactly {sorted(expected_training_fields)!r}, "
            f"got {sorted(training_metadata, key=str)!r}"
        )
    for field_name, expected_value in expected_fields.items():
        actual_value = training_metadata.get(field_name)
        if actual_value != expected_value:
            raise ValueError(
                f"checkpoint metadata 'pytorch_training.{field_name}': "
                f"expected {expected_value!r}, got {actual_value!r}"
            )
    recorded_weights_fingerprint = training_metadata["weights_fingerprint"]
    _validate_sha256(
        recorded_weights_fingerprint,
        field_name="pytorch_training.weights_fingerprint",
    )

    normalization_metadata = _validate_normalization_metadata(metadata)
    supplied_asset_id = validate_normalization_asset_id(asset_id, field_name="asset_id")
    recorded_asset_id = normalization_metadata["asset_id"]
    if supplied_asset_id != recorded_asset_id:
        raise ValueError(
            "checkpoint metadata 'normalization.asset_id': "
            f"expected recorded value {recorded_asset_id!r}, got supplied value {supplied_asset_id!r}"
        )
    actual_fingerprint = normalization_stats_fingerprint(
        norm_stats,
        validated_spec,
        require_state=True,
    )
    actual_state_dim = normalization_state_dim(norm_stats, validated_spec)
    if actual_state_dim != normalization_metadata["state_dim"]:
        raise ValueError(
            "checkpoint metadata 'normalization.state_dim': "
            f"expected {normalization_metadata['state_dim']}, got {actual_state_dim}"
        )
    recorded_fingerprint = normalization_metadata["fingerprint"]
    if actual_fingerprint != recorded_fingerprint:
        raise ValueError(
            "checkpoint metadata 'normalization.fingerprint': "
            f"expected recorded value {recorded_fingerprint!r}, "
            f"got {actual_fingerprint!r} for supplied stats"
        )

    weights_path, normalization_asset_path, relative_asset_path = _require_checkpoint_artifacts(
        pathlib.Path(checkpoint_dir),
        asset_id=recorded_asset_id,
    )
    if normalization_metadata["asset_file"] != relative_asset_path:
        raise ValueError(
            "checkpoint metadata 'normalization.asset_file': "
            f"expected {relative_asset_path!r}, got {normalization_metadata['asset_file']!r}"
        )
    actual_asset_file_fingerprint = _file_sha256(normalization_asset_path)
    if actual_asset_file_fingerprint != normalization_metadata["asset_file_fingerprint"]:
        raise ValueError("checkpoint metadata 'normalization.asset_file_fingerprint': serialized asset changed")
    serialized_stats = _load_serialized_normalization_stats(normalization_asset_path)
    serialized_fingerprint = normalization_stats_fingerprint(
        serialized_stats,
        validated_spec,
        require_state=True,
    )
    if serialized_fingerprint != recorded_fingerprint:
        raise ValueError("checkpoint metadata 'normalization.fingerprint': serialized asset content changed")
    if _file_sha256(weights_path) != recorded_weights_fingerprint:
        raise ValueError("checkpoint metadata 'pytorch_training.weights_fingerprint': model weights changed")
    return metadata


def _validate_normalization_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    normalization_metadata = metadata.get(NORMALIZATION_METADATA_KEY)
    if not isinstance(normalization_metadata, Mapping):
        raise ValueError("checkpoint metadata 'normalization' must be an object")

    expected_fields = {
        "asset_id",
        "fingerprint_algorithm",
        "fingerprint",
        "state_dim",
        "asset_file",
        "asset_file_fingerprint_algorithm",
        "asset_file_fingerprint",
    }
    actual_fields = set(normalization_metadata)
    if actual_fields != expected_fields:
        raise ValueError(
            "checkpoint metadata 'normalization' fields: "
            f"expected exactly {sorted(expected_fields)!r}, got {sorted(actual_fields, key=str)!r}"
        )
    recorded_asset_id = validate_normalization_asset_id(
        normalization_metadata["asset_id"],
        field_name="normalization.asset_id",
    )
    algorithm = normalization_metadata["fingerprint_algorithm"]
    if algorithm != NORMALIZATION_FINGERPRINT_ALGORITHM:
        raise ValueError(
            "checkpoint metadata 'normalization.fingerprint_algorithm': "
            f"expected {NORMALIZATION_FINGERPRINT_ALGORITHM!r}, got {algorithm!r}"
        )
    fingerprint = normalization_metadata["fingerprint"]
    _validate_sha256(fingerprint, field_name="normalization.fingerprint")
    state_dim = normalization_metadata["state_dim"]
    if type(state_dim) is not int:
        raise TypeError(
            "checkpoint metadata 'normalization.state_dim': "
            f"expected int, got {type(state_dim).__name__}"
        )
    if state_dim <= 0:
        raise ValueError("checkpoint metadata 'normalization.state_dim': expected a positive integer")
    asset_file = normalization_metadata["asset_file"]
    if type(asset_file) is not str:
        raise TypeError(
            "checkpoint metadata 'normalization.asset_file': "
            f"expected str, got {type(asset_file).__name__}"
        )
    asset_file_algorithm = normalization_metadata["asset_file_fingerprint_algorithm"]
    if asset_file_algorithm != NORMALIZATION_ASSET_FILE_FINGERPRINT_ALGORITHM:
        raise ValueError(
            "checkpoint metadata 'normalization.asset_file_fingerprint_algorithm': "
            f"expected {NORMALIZATION_ASSET_FILE_FINGERPRINT_ALGORITHM!r}, "
            f"got {asset_file_algorithm!r}"
        )
    asset_file_fingerprint = normalization_metadata["asset_file_fingerprint"]
    _validate_sha256(
        asset_file_fingerprint,
        field_name="normalization.asset_file_fingerprint",
    )
    return {
        "asset_id": recorded_asset_id,
        "fingerprint_algorithm": algorithm,
        "fingerprint": fingerprint,
        "state_dim": state_dim,
        "asset_file": asset_file,
        "asset_file_fingerprint_algorithm": asset_file_algorithm,
        "asset_file_fingerprint": asset_file_fingerprint,
    }


def validate_normalization_asset_id(asset_id: object, *, field_name: str = "asset_id") -> str:
    """Validate and return one safe OpenPI normalization asset directory name.

    Args:
        asset_id: Candidate identifier. It must be a single non-empty directory
            name using only ASCII letters, digits, dot, underscore, or hyphen.
        field_name: Logical field name included in validation errors.

    Returns:
        The validated identifier unchanged.

    Raises:
        TypeError: If ``asset_id`` or ``field_name`` is not a string.
        ValueError: If the identifier is unsafe, empty, or names ``.``/``..``.
    """
    if type(field_name) is not str:
        raise TypeError(f"field_name: expected str, got {type(field_name).__name__}")
    if type(asset_id) is not str:
        raise TypeError(f"{field_name}: expected str, got {type(asset_id).__name__}")
    if not _ASSET_ID_PATTERN.fullmatch(asset_id) or asset_id in {".", ".."}:
        raise ValueError(
            f"{field_name}: expected a non-empty directory name containing only letters, "
            "digits, dot, underscore, or hyphen"
        )
    return asset_id


def _validated_spec_copy(spec: object) -> BimanualActionSpec:
    if not isinstance(spec, BimanualActionSpec):
        raise TypeError(f"spec: expected BimanualActionSpec, got {type(spec).__name__}")
    return dataclasses.replace(spec)


def _require_checkpoint_artifacts(
    checkpoint_dir: pathlib.Path,
    *,
    asset_id: str,
) -> tuple[pathlib.Path, pathlib.Path, str]:
    weights_path = checkpoint_dir / MODEL_WEIGHTS_FILENAME
    if not weights_path.is_file():
        raise FileNotFoundError(f"PyTorch checkpoint weights not found: {weights_path}")
    relative_asset_path = f"assets/{asset_id}/{NORMALIZATION_ASSET_FILENAME}"
    normalization_asset_path = checkpoint_dir / pathlib.PurePosixPath(relative_asset_path)
    if not normalization_asset_path.is_file():
        raise FileNotFoundError(f"checkpoint normalization asset not found: {normalization_asset_path}")
    return weights_path, normalization_asset_path, relative_asset_path


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_serialized_normalization_stats(path: pathlib.Path) -> dict[str, dict[str, np.ndarray]]:
    """Parse the exact OpenPI ``norm_stats.json`` payload without importing OpenPI."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid checkpoint normalization JSON: {path}") from error
    if not isinstance(payload, dict) or set(payload) != {"norm_stats"}:
        raise ValueError("checkpoint normalization JSON root: expected exactly a 'norm_stats' object")
    raw_stats = payload["norm_stats"]
    if not isinstance(raw_stats, Mapping):
        raise TypeError("checkpoint normalization JSON 'norm_stats': expected an object")

    parsed: dict[str, dict[str, np.ndarray]] = {}
    for key, entry in raw_stats.items():
        if type(key) is not str:
            raise TypeError("checkpoint normalization JSON keys: expected strings")
        if not isinstance(entry, Mapping):
            raise TypeError(f"checkpoint normalization JSON norm_stats[{key!r}]: expected an object")
        parsed[key] = {}
        for field_name, values in entry.items():
            if type(field_name) is not str:
                raise TypeError(f"checkpoint normalization JSON norm_stats[{key!r}] fields: expected strings")
            parsed[key][field_name] = np.asarray(values)
    return parsed


def _validate_sha256(value: object, *, field_name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"checkpoint metadata {field_name!r}: expected str, got {type(value).__name__}")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"checkpoint metadata {field_name!r}: expected 64 lowercase hexadecimal digits")


def _validate_exact_metadata_mapping(
    value: object,
    *,
    expected: Mapping[str, Any],
    field_name: str,
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name}: expected a mapping, got {type(value).__name__}")
    if set(value) != set(expected):
        raise ValueError(
            f"{field_name} fields: expected exactly {sorted(expected)!r}, "
            f"got {sorted(value, key=str)!r}"
        )
    for key, expected_value in expected.items():
        actual_value = value[key]
        if type(actual_value) is not type(expected_value):
            raise TypeError(
                f"{field_name}.{key}: expected {type(expected_value).__name__}, "
                f"got {type(actual_value).__name__}"
            )
        if actual_value != expected_value:
            raise ValueError(
                f"{field_name}.{key}: expected {expected_value!r}, got {actual_value!r}"
            )
