"""Joint_29d absolute/delta conversion (scheme A).

Training target when ``ActionMode.DELTA``::

    delta[k] = absolute_cmd[k] - measured_joint_state(t0)

Deployment composes absolute commanded joints before publishing to the robot::

    absolute_cmd[k] = measured_joint_state(t0) + delta[k]

HDF5 ``action/*`` streams remain absolute commanded joints; delta is derived at
sample load / norm / inference boundaries.
"""

from __future__ import annotations

import numpy as np

from pi_dex.core.actions import ActionRepresentation
from pi_dex.core.spec import ActionMode
from pi_dex.core.spec import BimanualActionSpec


def bimanual_joint_reference_from_state(
    state: np.ndarray,
    *,
    spec: BimanualActionSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Return left/right ``[36]`` (or ``[N,36]``) reference joints from 65D observation state.

    State layout: ``[left arm+hand(29) | right arm+hand(29) | motor(7)]``. The shared
    motor block at ``[..., 58:65]`` is concatenated onto both hand references.
    """
    if spec.action_representation is not ActionRepresentation.JOINT_29D:
        raise ValueError("bimanual_joint_reference_from_state: only joint_29d is supported")
    array = np.asarray(state, dtype=np.float32)
    dim = spec.logical_action_dim
    required = 65
    if array.shape[-1] < required:
        raise ValueError(
            f"state: expected at least {required} bimanual joint dims for joint_29d, "
            f"got width {array.shape[-1]}"
        )
    motor = np.asarray(array[..., 58:65], dtype=np.float32)
    left = np.concatenate((array[..., :29], motor), axis=-1)
    right = np.concatenate((array[..., 29:58], motor), axis=-1)
    if left.shape[-1] != dim or right.shape[-1] != dim:
        raise AssertionError(
            f"internal joint reference layout error: expected width {dim}, "
            f"got left={left.shape[-1]} right={right.shape[-1]}"
        )
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("state: expected finite left/right joint reference values")
    return left.astype(np.float32, copy=False), right.astype(np.float32, copy=False)


def _broadcast_add_reference(
    actions: np.ndarray,
    reference: np.ndarray,
    *,
    subtract: bool,
) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    if ref.ndim == 1:
        expanded = ref.reshape((1,) * (values.ndim - 1) + (ref.shape[0],))
    elif ref.ndim == 2 and values.ndim == 3:
        expanded = ref[:, np.newaxis, :]
    else:
        raise ValueError(
            f"reference/actions rank mismatch: reference.ndim={ref.ndim}, actions.ndim={values.ndim}"
        )
    return values - expanded if subtract else values + expanded


def absolute_joint_actions_to_delta(
    left_actions: np.ndarray,
    right_actions: np.ndarray,
    reference_state: np.ndarray,
    *,
    spec: BimanualActionSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert absolute joint chunks to scheme-A delta targets."""
    if spec.action_mode is not ActionMode.DELTA:
        raise ValueError(
            f"absolute_joint_actions_to_delta: expected ActionMode.DELTA, got {spec.action_mode.value!r}"
        )
    left_ref, right_ref = bimanual_joint_reference_from_state(reference_state, spec=spec)
    left_delta = _broadcast_add_reference(left_actions, left_ref, subtract=True)
    right_delta = _broadcast_add_reference(right_actions, right_ref, subtract=True)
    return left_delta, right_delta


def delta_joint_actions_to_absolute(
    left_actions: np.ndarray,
    right_actions: np.ndarray,
    reference_state: np.ndarray,
    *,
    spec: BimanualActionSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose absolute joint commands from delta predictions and state(t0)."""
    if spec.action_mode is not ActionMode.DELTA:
        raise ValueError(
            f"delta_joint_actions_to_absolute: expected ActionMode.DELTA, got {spec.action_mode.value!r}"
        )
    left_ref, right_ref = bimanual_joint_reference_from_state(reference_state, spec=spec)
    left_abs = _broadcast_add_reference(left_actions, left_ref, subtract=False)
    right_abs = _broadcast_add_reference(right_actions, right_ref, subtract=False)
    return left_abs, right_abs


def maybe_convert_training_joint_actions(
    left_actions: np.ndarray,
    right_actions: np.ndarray,
    reference_state: np.ndarray,
    *,
    spec: BimanualActionSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Return training labels in the spec's action mode."""
    if spec.action_mode is ActionMode.ABSOLUTE:
        return left_actions, right_actions
    if spec.action_mode is ActionMode.DELTA:
        return absolute_joint_actions_to_delta(
            left_actions,
            right_actions,
            reference_state,
            spec=spec,
        )
    raise ValueError(
        f"maybe_convert_training_joint_actions: unsupported action_mode {spec.action_mode.value!r}"
    )


def maybe_compose_deployment_joint_actions(
    left_actions: np.ndarray,
    right_actions: np.ndarray,
    reference_state: np.ndarray,
    *,
    spec: BimanualActionSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Return absolute joint commands for robot dispatch."""
    if spec.action_mode is ActionMode.ABSOLUTE:
        return left_actions, right_actions
    if spec.action_mode is ActionMode.DELTA:
        return delta_joint_actions_to_absolute(
            left_actions,
            right_actions,
            reference_state,
            spec=spec,
        )
    raise ValueError(
        f"maybe_compose_deployment_joint_actions: unsupported action_mode {spec.action_mode.value!r}"
    )
