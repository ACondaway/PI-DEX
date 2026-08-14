"""Validation and stable fingerprints for PI-DEX normalization assets.

This module deliberately does not import OpenPI.  It accepts either OpenPI-like
objects exposing ``mean``, ``std``, ``q01``, and ``q99`` attributes or mappings
with those fields, then validates the PI-DEX pi0.5 normalization contract.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np

from pi_dex.spec import BimanualActionSpec
from pi_dex.spec import HandNormalization

NORMALIZATION_KEYS = ("state", "left_actions", "right_actions")
NORMALIZATION_STAT_FIELDS = ("mean", "std", "q01", "q99")
NORMALIZATION_FINGERPRINT_ALGORITHM = "sha256-float64-le-v1"


def validate_normalization_stats(
    norm_stats: Mapping[str, Any] | None,
    spec: BimanualActionSpec,
    *,
    require_state: bool = True,
) -> None:
    """Validate normalization statistics without importing OpenPI.

    Args:
        norm_stats: Mapping whose values expose ``mean``, ``std``, ``q01``, and
            ``q99`` either as attributes or mapping entries. Every statistic must
            be a finite, one-dimensional floating NumPy array. Action statistics
            have shape ``[spec.logical_action_dim]``; state statistics share one
            non-empty shape.
        spec: PI-DEX semantic contract. Shared-hand normalization additionally
            requires all left/right action statistics to be exactly equal after
            canonical float64 conversion.
        require_state: If true, require exactly ``state``, ``left_actions``, and
            ``right_actions``. If false, require exactly the two action keys.

    Raises:
        TypeError: If mappings, arrays, dtypes, or ``spec`` have invalid types.
        ValueError: If keys, shapes, values, quantile order, standard deviations,
            or shared-hand statistics violate the pi0.5 contract.

    Notes:
        PI-DEX targets pi0.5, whose quantile normalization requires both ``q01``
        and ``q99``. They are therefore mandatory even if a caller also retains
        mean/std statistics for diagnostics.
    """
    _canonicalize_normalization_stats(norm_stats, spec, require_state=require_state)


def normalization_stats_fingerprint(
    norm_stats: Mapping[str, Any] | None,
    spec: BimanualActionSpec,
    *,
    require_state: bool = True,
) -> str:
    """Return a deterministic SHA-256 fingerprint for validated statistics.

    Arrays are hashed in canonical key/field order after conversion to contiguous
    little-endian float64. Shape and component names are length-prefixed in the
    digest, so dtype byte order, source precision, mapping insertion order, and
    host endianness do not alter the fingerprint for equal numeric values.

    Args:
        norm_stats: Normalization mapping accepted by
            :func:`validate_normalization_stats`.
        spec: PI-DEX semantic contract controlling shared-hand validation.
        require_state: Whether ``state`` is part of the required asset.

    Returns:
        Lowercase 64-character SHA-256 hexadecimal digest.

    Raises:
        TypeError: If the normalization structure or values have invalid types.
        ValueError: If the normalization contract is invalid.
    """
    canonical = _canonicalize_normalization_stats(norm_stats, spec, require_state=require_state)
    digest = hashlib.sha256()
    _update_digest(digest, NORMALIZATION_FINGERPRINT_ALGORITHM.encode("ascii"))
    for key in _expected_keys(require_state=require_state):
        for field_name in NORMALIZATION_STAT_FIELDS:
            values = canonical[key][field_name]
            _update_digest(digest, key.encode("utf-8"))
            _update_digest(digest, field_name.encode("ascii"))
            _update_digest(digest, len(values.shape).to_bytes(8, byteorder="little", signed=False))
            for dimension in values.shape:
                _update_digest(digest, dimension.to_bytes(8, byteorder="little", signed=False))
            _update_digest(digest, b"<f8")
            _update_digest(digest, values.tobytes(order="C"))
    return digest.hexdigest()


def normalization_state_dim(
    norm_stats: Mapping[str, Any] | None,
    spec: BimanualActionSpec,
) -> int:
    """Return the exact validated one-dimensional state width.

    The width is part of the pi0.5 prompt-conditioning contract: OpenPI's
    quantile transform otherwise accepts a shorter state by truncating the
    checkpoint statistics.
    """
    canonical = _canonicalize_normalization_stats(norm_stats, spec, require_state=True)
    return int(canonical["state"]["mean"].shape[0])


def _canonicalize_normalization_stats(
    norm_stats: Mapping[str, Any] | None,
    spec: BimanualActionSpec,
    *,
    require_state: bool,
) -> dict[str, dict[str, np.ndarray]]:
    if not isinstance(spec, BimanualActionSpec):
        raise TypeError(f"spec: expected BimanualActionSpec, got {type(spec).__name__}")
    validated_spec = dataclasses.replace(spec)
    if not isinstance(require_state, bool):
        raise TypeError(f"require_state: expected bool, got {type(require_state).__name__}")
    if not isinstance(norm_stats, Mapping):
        raise TypeError(f"norm_stats: expected a mapping, got {type(norm_stats).__name__}")

    expected_keys = _expected_keys(require_state=require_state)
    actual_keys = tuple(norm_stats.keys())
    missing_keys = [key for key in expected_keys if key not in norm_stats]
    unexpected_keys = [key for key in actual_keys if key not in expected_keys]
    if missing_keys or unexpected_keys:
        raise ValueError(
            "norm_stats keys: "
            f"expected exactly {list(expected_keys)!r}, got {list(actual_keys)!r}; "
            f"missing={missing_keys!r}, unexpected={unexpected_keys!r}"
        )

    canonical: dict[str, dict[str, np.ndarray]] = {}
    for key in expected_keys:
        expected_shape = None if key == "state" else (validated_spec.logical_action_dim,)
        canonical[key] = _canonicalize_stats_entry(norm_stats[key], key=key, expected_shape=expected_shape)

    if validated_spec.hand_normalization is HandNormalization.SHARED:
        for field_name in NORMALIZATION_STAT_FIELDS:
            left_values = canonical["left_actions"][field_name]
            right_values = canonical["right_actions"][field_name]
            if not np.array_equal(left_values, right_values):
                raise ValueError(
                    "norm_stats shared hand normalization: "
                    f"left_actions.{field_name} and right_actions.{field_name} must be equal"
                )
    return canonical


def _canonicalize_stats_entry(
    stats: object,
    *,
    key: str,
    expected_shape: tuple[int, ...] | None,
) -> dict[str, np.ndarray]:
    if isinstance(stats, Mapping):
        actual_fields = tuple(stats.keys())
        missing_fields = [field_name for field_name in NORMALIZATION_STAT_FIELDS if field_name not in stats]
        unexpected_fields = [field_name for field_name in actual_fields if field_name not in NORMALIZATION_STAT_FIELDS]
        if missing_fields or unexpected_fields:
            raise ValueError(
                f"norm_stats[{key!r}] fields: expected exactly {list(NORMALIZATION_STAT_FIELDS)!r}, "
                f"got {list(actual_fields)!r}; missing={missing_fields!r}, unexpected={unexpected_fields!r}"
            )
        raw_fields = {field_name: stats[field_name] for field_name in NORMALIZATION_STAT_FIELDS}
    else:
        raw_fields = {}
        for field_name in NORMALIZATION_STAT_FIELDS:
            if not hasattr(stats, field_name):
                raise TypeError(f"norm_stats[{key!r}]: missing required attribute {field_name!r}")
            raw_fields[field_name] = getattr(stats, field_name)

    canonical: dict[str, np.ndarray] = {}
    entry_shape = expected_shape
    for field_name in NORMALIZATION_STAT_FIELDS:
        values = _canonicalize_stats_array(raw_fields[field_name], field_name=f"norm_stats[{key!r}].{field_name}")
        if entry_shape is None:
            entry_shape = values.shape
            if values.size == 0:
                raise ValueError(f"norm_stats[{key!r}].{field_name}.shape: expected a non-empty one-dimensional array")
        if values.shape != entry_shape:
            raise ValueError(
                f"norm_stats[{key!r}].{field_name}.shape: expected {entry_shape}, got {values.shape}"
            )
        canonical[field_name] = values

    if np.any(canonical["std"] < 0.0):
        invalid_dimensions = np.flatnonzero(canonical["std"] < 0.0).tolist()
        raise ValueError(f"norm_stats[{key!r}].std: negative values at dimensions {invalid_dimensions}")
    invalid_quantiles = np.flatnonzero(canonical["q01"] > canonical["q99"])
    if invalid_quantiles.size:
        raise ValueError(
            f"norm_stats[{key!r}]: q01 exceeds q99 at dimensions {invalid_quantiles.tolist()}"
        )
    return canonical


def _canonicalize_stats_array(values: object, *, field_name: str) -> np.ndarray:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{field_name}: expected numpy.ndarray, got {type(values).__name__}")
    if values.ndim != 1:
        raise ValueError(f"{field_name}.shape: expected a one-dimensional array, got {values.shape}")
    if values.dtype.kind != "f":
        raise TypeError(f"{field_name}.dtype: expected a floating dtype, got {values.dtype}")
    if not np.all(np.isfinite(values)):
        invalid_dimensions = np.flatnonzero(~np.isfinite(values)).tolist()
        raise ValueError(f"{field_name}: non-finite values at dimensions {invalid_dimensions}")

    with np.errstate(over="ignore", invalid="ignore"):
        canonical = np.array(values, dtype=np.dtype("<f8"), order="C", copy=True)
    if not np.all(np.isfinite(canonical)):
        raise ValueError(f"{field_name}: values must be representable as finite float64")
    canonical[canonical == 0.0] = 0.0
    return canonical


def _expected_keys(*, require_state: bool) -> tuple[str, ...]:
    return NORMALIZATION_KEYS if require_state else NORMALIZATION_KEYS[1:]


def _update_digest(digest: Any, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, byteorder="little", signed=False))
    digest.update(payload)
