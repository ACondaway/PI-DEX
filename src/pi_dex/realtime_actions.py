"""Convert PI-DEX paired action chunks into Sharpa North SDK action dicts.

Logical joint_29d layout per hand is ``[7 arm | 22 hand]`` matching the reviewed
observation contract and ``BimanualActionSpec`` joint orders. Motor commands are
not part of the joint_29d action chunk; callers that must stream motor may pass
``motor_positions`` separately.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from pi_dex.spec import BimanualActionSpec

# Live SDK action keys consumed by NorthDirect._send_actions / _prepare_action_dict.
SDK_LEFT_ARM_ACTION = "/action/left_arm/joint_angle"
SDK_LEFT_HAND_ACTION = "/action/left_hand/joint_angle"
SDK_RIGHT_ARM_ACTION = "/action/right_arm/joint_angle"
SDK_RIGHT_HAND_ACTION = "/action/right_hand/joint_angle"
SDK_MOTOR_ACTION = "/action/motor/joint_angle"

ARM_DIM = 7
HAND_DIM = 22
JOINT_29D_DIM = ARM_DIM + HAND_DIM


def split_joint_29d_hand_chunk(hand_actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split ``[E, 29]`` into arm ``[E, 7]`` and hand ``[E, 22]`` float32 views."""
    array = np.asarray(hand_actions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != JOINT_29D_DIM:
        raise ValueError(f"hand_actions: expected [E, {JOINT_29D_DIM}], got {array.shape}")
    if array.shape[0] <= 0:
        raise ValueError("hand_actions: expected a positive execution horizon")
    if not np.isfinite(array).all():
        raise ValueError("hand_actions: expected finite values")
    return array[:, :ARM_DIM].copy(), array[:, ARM_DIM:].copy()


def policy_result_to_sdk_action_dict(
    policy_result: Mapping[str, Any],
    spec: BimanualActionSpec,
    *,
    motor_positions: np.ndarray | None = None,
) -> dict[str, list[list[float]]]:
    """Convert ``BimanualPolicyAdapter.infer`` output into NorthDirect action lists.

    Returns a dict whose values are length-``E`` lists of per-step joint vectors,
    suitable for ``NorthDirect._send_actions``.
    """
    if spec.logical_action_dim != JOINT_29D_DIM:
        raise ValueError(
            f"spec.logical_action_dim: expected {JOINT_29D_DIM} for joint_29d, "
            f"got {spec.logical_action_dim}"
        )
    try:
        actions = policy_result["actions"]
        left = np.asarray(actions["left"], dtype=np.float32)
        right = np.asarray(actions["right"], dtype=np.float32)
    except (KeyError, TypeError) as error:
        raise KeyError("policy_result: expected actions.left/right") from error
    if left.shape != right.shape:
        raise ValueError(f"actions.left/right shape mismatch: {left.shape} vs {right.shape}")
    if left.shape[1] != spec.logical_action_dim:
        raise ValueError(
            f"actions.left: expected width {spec.logical_action_dim}, got {left.shape[1]}"
        )
    left_arm, left_hand = split_joint_29d_hand_chunk(left)
    right_arm, right_hand = split_joint_29d_hand_chunk(right)
    payload: dict[str, list[list[float]]] = {
        SDK_LEFT_ARM_ACTION: left_arm.tolist(),
        SDK_LEFT_HAND_ACTION: left_hand.tolist(),
        SDK_RIGHT_ARM_ACTION: right_arm.tolist(),
        SDK_RIGHT_HAND_ACTION: right_hand.tolist(),
    }
    if motor_positions is not None:
        motor = np.asarray(motor_positions, dtype=np.float32)
        if motor.ndim == 1:
            motor = np.repeat(motor[None, :], left.shape[0], axis=0)
        if motor.shape != (left.shape[0], ARM_DIM):
            raise ValueError(
                f"motor_positions: expected [{left.shape[0]}, {ARM_DIM}] or [{ARM_DIM}], "
                f"got {motor.shape}"
            )
        payload[SDK_MOTOR_ACTION] = motor.tolist()
    return payload


def sdk_action_pub_keys() -> tuple[str, ...]:
    """Default ``action_output`` keys for the reference NorthDirect env."""
    return (
        SDK_LEFT_ARM_ACTION,
        SDK_LEFT_HAND_ACTION,
        SDK_RIGHT_ARM_ACTION,
        SDK_RIGHT_HAND_ACTION,
    )
