"""Canonical PI-DEX action layout and bimanual sequence conversions.

A logical single-hand action has 31 values: wrist position in metres, a
dimensionless continuous 6D rotation representation, and 22 hand joint angles
in radians. The coordinate frame and absolute, relative, or residual semantics
must be established by the caller; these conversions preserve them unchanged.

OpenPI's pi05 projection expects 32 values, so the model representation appends
one nonsemantic zero. A bimanual chunk interleaves those model actions along the
sequence axis in left-then-right order for each physical control step.
"""

from typing import Any
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

WRIST_POSITION_DIM = 3
WRIST_ROTATION_6D_DIM = 6
HAND_JOINT_DIM = 22
LOGICAL_ACTION_DIM = WRIST_POSITION_DIM + WRIST_ROTATION_6D_DIM + HAND_JOINT_DIM
MODEL_ACTION_DIM = 32
VALID_ACTION_MASK = (True,) * LOGICAL_ACTION_DIM + (False,)

FloatingArray: TypeAlias = npt.NDArray[np.floating[Any]]


def pad_action(actions: FloatingArray) -> FloatingArray:
    """Append the nonsemantic model padding dimension to logical actions.

    Args:
        actions: CPU NumPy floating-point array with shape ``[..., 31]``. The
            last axis contains wrist position (metres), 6D rotation, and hand
            joint angles (radians), in that order.

    Returns:
        A new array with the same dtype and shape ``[..., 32]`` whose last value
        is zero.

    Raises:
        TypeError: If ``actions`` is not a floating-point NumPy array.
        ValueError: If the last dimension is not 31 or a semantic value is not
            finite.
    """
    _validate_actions(actions, field_name="actions", expected_width=LOGICAL_ACTION_DIM)
    _require_finite_semantic_values(actions, field_name="actions")
    padding = np.zeros((*actions.shape[:-1], MODEL_ACTION_DIM - LOGICAL_ACTION_DIM), dtype=actions.dtype)
    return np.concatenate((actions, padding), axis=-1)


def unpad_action(actions: FloatingArray) -> FloatingArray:
    """Discard the nonsemantic model padding dimension.

    Args:
        actions: CPU NumPy floating-point array with shape ``[..., 32]`` in the
            PI-DEX model action layout. The final value is discarded even when a
            model predicts a nonzero value for it.

    Returns:
        A view with the same dtype and shape ``[..., 31]`` containing wrist
        position (metres), 6D rotation, and hand joint angles (radians).

    Raises:
        TypeError: If ``actions`` is not a floating-point NumPy array.
        ValueError: If the last dimension is not 32 or a retained semantic value
            is not finite. The discarded padding value may be arbitrary.
    """
    _validate_actions(actions, field_name="actions", expected_width=MODEL_ACTION_DIM)
    _require_finite_semantic_values(actions, field_name="actions")
    return actions[..., :LOGICAL_ACTION_DIM]


def interleave(left_actions: FloatingArray, right_actions: FloatingArray) -> FloatingArray:
    """Interleave same-timestep left and right model actions as ``L, R``.

    Args:
        left_actions: CPU NumPy floating-point array with shape ``[..., K, 32]``.
        right_actions: Array with the same shape and dtype as ``left_actions``.
            Both arrays must use identical units, coordinate frames, timing, and
            absolute, relative, or residual semantics.

    Returns:
        A new array with shape ``[..., 2 * K, 32]`` ordered as
        ``[left_0, right_0, ..., left_K-1, right_K-1]``.

    Raises:
        TypeError: If either input is not a floating-point NumPy array or their
            dtypes differ.
        ValueError: If either input lacks a sequence axis, has a last dimension
            other than 32, contains a non-finite value or nonzero padding, or
            their shapes differ.
    """
    _validate_model_action_sequence(left_actions, field_name="left_actions")
    _validate_model_action_sequence(right_actions, field_name="right_actions")
    _require_finite_model_values(left_actions, field_name="left_actions")
    _require_finite_model_values(right_actions, field_name="right_actions")
    _require_zero_padding(left_actions, field_name="left_actions")
    _require_zero_padding(right_actions, field_name="right_actions")
    if left_actions.shape != right_actions.shape:
        raise ValueError(
            "left_actions and right_actions must have matching shapes; "
            f"got left_actions.shape={left_actions.shape} and right_actions.shape={right_actions.shape}"
        )
    if left_actions.dtype != right_actions.dtype:
        raise TypeError(
            "left_actions and right_actions must have matching dtypes; "
            f"got {left_actions.dtype} and {right_actions.dtype}"
        )

    stacked_actions = np.stack((left_actions, right_actions), axis=-2)
    return stacked_actions.reshape(*left_actions.shape[:-2], left_actions.shape[-2] * 2, MODEL_ACTION_DIM)


def deinterleave(actions: FloatingArray) -> tuple[FloatingArray, FloatingArray]:
    """Split an even model horizon into left and right physical-step chunks.

    Args:
        actions: CPU NumPy floating-point array with shape ``[..., 2 * K, 32]``
            ordered left then right at every physical control step.

    Returns:
        A ``(left_actions, right_actions)`` pair. Each view has shape
        ``[..., K, 32]`` and preserves the input dtype, units, coordinate frame,
        timing, and action semantics.

    Raises:
        TypeError: If ``actions`` is not a floating-point NumPy array.
        ValueError: If the input lacks a sequence axis, its last dimension is
            not 32, its semantic values are not finite, or its model action
            horizon is odd. The discarded padding values may be arbitrary.
    """
    _validate_model_action_sequence(actions, field_name="actions")
    _require_finite_semantic_values(actions, field_name="actions")
    model_action_horizon = actions.shape[-2]
    if model_action_horizon % 2 != 0:
        raise ValueError(f"actions action horizon must be even; got {model_action_horizon}")
    return actions[..., 0::2, :], actions[..., 1::2, :]


def _validate_model_action_sequence(actions: np.ndarray, *, field_name: str) -> None:
    if isinstance(actions, np.ndarray) and actions.ndim < 2:
        raise ValueError(
            f"{field_name}.ndim: expected at least 2 axes (..., horizon, action), got shape {actions.shape}"
        )
    _validate_actions(actions, field_name=field_name, expected_width=MODEL_ACTION_DIM)
    if actions.shape[-2] == 0:
        raise ValueError(f"{field_name}.shape[-2]: expected a non-empty action horizon")


def _validate_actions(actions: np.ndarray, *, field_name: str, expected_width: int) -> None:
    if not isinstance(actions, np.ndarray):
        raise TypeError(f"{field_name}: expected numpy.ndarray on CPU, got {type(actions).__name__}")
    if actions.ndim == 0:
        raise ValueError(
            f"{field_name}.shape[-1]: expected {expected_width}, got no last dimension for shape {actions.shape}"
        )
    if actions.shape[-1] != expected_width:
        raise ValueError(f"{field_name}.shape[-1]: expected {expected_width}, got {actions.shape[-1]}")
    if not np.issubdtype(actions.dtype, np.floating):
        raise TypeError(f"{field_name}.dtype: expected a floating dtype, got {actions.dtype}")


def _require_finite_semantic_values(actions: np.ndarray, *, field_name: str) -> None:
    semantic_values = actions[..., :LOGICAL_ACTION_DIM]
    if not np.all(np.isfinite(semantic_values)):
        raise ValueError(f"{field_name}: expected all semantic action values to be finite")


def _require_finite_model_values(actions: np.ndarray, *, field_name: str) -> None:
    if not np.all(np.isfinite(actions)):
        raise ValueError(f"{field_name}: expected all model action values to be finite")


def _require_zero_padding(actions: np.ndarray, *, field_name: str) -> None:
    if np.any(actions[..., LOGICAL_ACTION_DIM:] != 0):
        raise ValueError(f"{field_name}: expected nonsemantic padding dimensions to be exactly zero")
