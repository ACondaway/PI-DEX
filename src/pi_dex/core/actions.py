"""Canonical PI-DEX action representations and bimanual conversions.

PI-DEX supports two single-hand logical action representations:

* ``cartesian_31d``: wrist position, continuous rotation 6D, and hand joints;
* ``joint_29d``: arm joints, hand joints, and a duplicated waist/motor block.

For ``joint_29d`` the on-wire logical width is ``36 = 29 + 7``: the trailing
seven dimensions are ``action/motor/joint_angle``, copied onto **both** left and
right targets. At deployment only one side's motor slice is published.

OpenPI's action projection width is :data:`MODEL_ACTION_DIM` (36). Cartesian
actions keep a 31D semantic prefix and zero-pad the remainder. A bimanual chunk
interleaves model actions along the sequence axis in left-then-right order at
each physical step (``H_model = 2K``).
"""

import enum
from typing import Any
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

WRIST_POSITION_DIM = 3
WRIST_ROTATION_6D_DIM = 6
ARM_JOINT_DIM = 7
HAND_JOINT_DIM = 22
MOTOR_JOINT_DIM = 7
# Historical name: arm+hand only (before motor was appended).
JOINT_ARM_HAND_DIM = ARM_JOINT_DIM + HAND_JOINT_DIM
CARTESIAN_LOGICAL_ACTION_DIM = WRIST_POSITION_DIM + WRIST_ROTATION_6D_DIM + HAND_JOINT_DIM
# joint_29d logical vector: [arm(7) | hand(22) | motor(7)].
JOINT_LOGICAL_ACTION_DIM = JOINT_ARM_HAND_DIM + MOTOR_JOINT_DIM
MODEL_ACTION_DIM = 36
# Pretrained OpenPI pi05 checkpoints ship with 32D action projections.
PRETRAINED_MODEL_ACTION_DIM = 32

JOINT_MOTOR_SLICE = slice(JOINT_ARM_HAND_DIM, JOINT_LOGICAL_ACTION_DIM)
JOINT_ARM_HAND_SLICE = slice(0, JOINT_ARM_HAND_DIM)


class ActionRepresentation(enum.StrEnum):
    """Semantic layout of one unpadded single-hand action vector."""

    CARTESIAN_31D = "cartesian_31d"
    JOINT_29D = "joint_29d"

    @property
    def logical_action_dim(self) -> int:
        """Return the number of semantic values before model padding."""
        if self is ActionRepresentation.CARTESIAN_31D:
            return CARTESIAN_LOGICAL_ACTION_DIM
        if self is ActionRepresentation.JOINT_29D:
            return JOINT_LOGICAL_ACTION_DIM
        raise AssertionError(f"unsupported action representation: {self!r}")


FloatingArray: TypeAlias = npt.NDArray[np.floating[Any]]


def _validate_representation(representation: object) -> ActionRepresentation:
    if not isinstance(representation, ActionRepresentation):
        raise TypeError(
            "representation: expected ActionRepresentation, "
            f"got {type(representation).__name__}"
        )
    return representation


def valid_action_mask(representation: ActionRepresentation) -> tuple[bool, ...]:
    """Return the immutable model-width semantic mask for ``representation``.

    Args:
        representation: Exact logical layout used by the sample and model run.

    Returns:
        A ``MODEL_ACTION_DIM``-element tuple whose semantic prefix is true and
        padding suffix false.

    Raises:
        TypeError: If ``representation`` is not :class:`ActionRepresentation`.
    """
    validated_representation = _validate_representation(representation)
    logical_dim = validated_representation.logical_action_dim
    return (True,) * logical_dim + (False,) * (MODEL_ACTION_DIM - logical_dim)


# Deprecated module-local compatibility aliases for the original Cartesian
# layout. They are intentionally not re-exported from ``pi_dex``; new code must
# use the explicit Cartesian name or a representation-derived width/mask.
LOGICAL_ACTION_DIM = CARTESIAN_LOGICAL_ACTION_DIM
VALID_ACTION_MASK = valid_action_mask(ActionRepresentation.CARTESIAN_31D)


def split_joint_logical_action(
    actions: FloatingArray,
) -> tuple[FloatingArray, FloatingArray, FloatingArray]:
    """Split ``[..., 36]`` joint actions into arm, hand, and motor blocks."""
    _validate_actions(actions, field_name="actions", expected_width=JOINT_LOGICAL_ACTION_DIM)
    arm = actions[..., :ARM_JOINT_DIM]
    hand = actions[..., ARM_JOINT_DIM:JOINT_ARM_HAND_DIM]
    motor = actions[..., JOINT_MOTOR_SLICE]
    return arm, hand, motor


def append_motor_to_arm_hand(
    arm_hand_actions: FloatingArray,
    motor_actions: FloatingArray,
) -> FloatingArray:
    """Concatenate ``[..., 29]`` arm+hand with ``[..., 7]`` motor → ``[..., 36]``."""
    _validate_actions(arm_hand_actions, field_name="arm_hand_actions", expected_width=JOINT_ARM_HAND_DIM)
    _validate_actions(motor_actions, field_name="motor_actions", expected_width=MOTOR_JOINT_DIM)
    if arm_hand_actions.shape[:-1] != motor_actions.shape[:-1]:
        raise ValueError(
            "arm_hand_actions/motor_actions leading shapes must match; "
            f"got {arm_hand_actions.shape[:-1]} vs {motor_actions.shape[:-1]}"
        )
    if arm_hand_actions.dtype != motor_actions.dtype:
        raise TypeError(
            "arm_hand_actions/motor_actions dtypes must match; "
            f"got {arm_hand_actions.dtype} vs {motor_actions.dtype}"
        )
    if not np.all(np.isfinite(arm_hand_actions)) or not np.all(np.isfinite(motor_actions)):
        raise ValueError("arm_hand_actions/motor_actions: expected finite values")
    return np.concatenate((arm_hand_actions, motor_actions), axis=-1)


def pad_action(
    actions: FloatingArray,
    *,
    representation: ActionRepresentation,
) -> FloatingArray:
    """Append the nonsemantic suffix required by the model action projection.

    Args:
        actions: CPU NumPy floating-point array with shape ``[..., D]``, where
            ``D`` is 31 for ``cartesian_31d`` and 36 for ``joint_29d``.
        representation: Semantic layout of the input vectors.

    Returns:
        A new array with the same dtype and shape ``[..., MODEL_ACTION_DIM]``.
        Every padding value in the suffix is exactly zero.

    Raises:
        TypeError: If the representation or array dtype/type is invalid.
        ValueError: If the logical width is wrong or a semantic value is not
            finite.
    """
    validated_representation = _validate_representation(representation)
    logical_dim = validated_representation.logical_action_dim
    _validate_actions(actions, field_name="actions", expected_width=logical_dim)
    _require_finite_semantic_values(
        actions,
        field_name="actions",
        representation=validated_representation,
    )
    padding_width = MODEL_ACTION_DIM - logical_dim
    if padding_width == 0:
        return np.asarray(actions, dtype=actions.dtype)
    padding = np.zeros((*actions.shape[:-1], padding_width), dtype=actions.dtype)
    return np.concatenate((actions, padding), axis=-1)


def unpad_action(
    actions: FloatingArray,
    *,
    representation: ActionRepresentation,
) -> FloatingArray:
    """Discard the model padding suffix for one explicit action representation.

    Args:
        actions: CPU NumPy floating-point array with shape
            ``[..., MODEL_ACTION_DIM]``.
        representation: Semantic layout that determines how many leading values
            are retained.

    Returns:
        A view with the same dtype and shape ``[..., D]``. Padding values are
        discarded even if the model predicted nonzero or non-finite values.

    Raises:
        TypeError: If the representation or array dtype/type is invalid.
        ValueError: If the model width is wrong or a retained value is not finite.
    """
    validated_representation = _validate_representation(representation)
    _validate_actions(actions, field_name="actions", expected_width=MODEL_ACTION_DIM)
    _require_finite_semantic_values(
        actions,
        field_name="actions",
        representation=validated_representation,
    )
    return actions[..., : validated_representation.logical_action_dim]


def interleave(
    left_actions: FloatingArray,
    right_actions: FloatingArray,
    *,
    representation: ActionRepresentation,
) -> FloatingArray:
    """Interleave same-timestep left and right model actions as ``L, R``.

    Args:
        left_actions: CPU NumPy floating-point array shaped
            ``[..., K, MODEL_ACTION_DIM]``.
        right_actions: Array with the same shape and dtype as ``left_actions``.
        representation: Semantic layout used to validate the zero padding suffix.

    Returns:
        A new array shaped ``[..., 2 * K, MODEL_ACTION_DIM]`` and ordered as
        ``[left_0, right_0, ..., left_K-1, right_K-1]``.

    Raises:
        TypeError: If the representation, array type, or dtypes are invalid.
        ValueError: If widths, shapes, finite values, or zero padding conflict.
    """
    validated_representation = _validate_representation(representation)
    _validate_model_action_sequence(left_actions, field_name="left_actions")
    _validate_model_action_sequence(right_actions, field_name="right_actions")
    _require_finite_model_values(left_actions, field_name="left_actions")
    _require_finite_model_values(right_actions, field_name="right_actions")
    _require_zero_padding(
        left_actions,
        field_name="left_actions",
        representation=validated_representation,
    )
    _require_zero_padding(
        right_actions,
        field_name="right_actions",
        representation=validated_representation,
    )
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
    return stacked_actions.reshape(
        *left_actions.shape[:-2],
        left_actions.shape[-2] * 2,
        MODEL_ACTION_DIM,
    )


def deinterleave(
    actions: FloatingArray,
    *,
    representation: ActionRepresentation,
) -> tuple[FloatingArray, FloatingArray]:
    """Split an even model horizon into left/right physical-step chunks.

    Args:
        actions: CPU NumPy floating-point array shaped
            ``[..., 2 * K, MODEL_ACTION_DIM]`` and ordered left then right at
            every physical control step.
        representation: Semantic layout used to validate retained values.

    Returns:
        A ``(left_actions, right_actions)`` pair of
        ``[..., K, MODEL_ACTION_DIM]`` views.

    Raises:
        TypeError: If the representation or array type/dtype is invalid.
        ValueError: If the shape, semantic values, or even-horizon invariant fails.
            Padding values may be arbitrary because callers unpad model output next.
    """
    validated_representation = _validate_representation(representation)
    _validate_model_action_sequence(actions, field_name="actions")
    _require_finite_semantic_values(
        actions,
        field_name="actions",
        representation=validated_representation,
    )
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


def _require_finite_semantic_values(
    actions: np.ndarray,
    *,
    field_name: str,
    representation: ActionRepresentation,
) -> None:
    semantic_values = actions[..., : representation.logical_action_dim]
    if not np.all(np.isfinite(semantic_values)):
        raise ValueError(f"{field_name}: expected all semantic action values to be finite")


def _require_finite_model_values(actions: np.ndarray, *, field_name: str) -> None:
    if not np.all(np.isfinite(actions)):
        raise ValueError(f"{field_name}: expected all model action values to be finite")


def _require_zero_padding(
    actions: np.ndarray,
    *,
    field_name: str,
    representation: ActionRepresentation,
) -> None:
    logical_dim = representation.logical_action_dim
    if logical_dim >= MODEL_ACTION_DIM:
        return
    if np.any(actions[..., logical_dim:] != 0):
        raise ValueError(f"{field_name}: expected nonsemantic padding dimensions to be exactly zero")
