"""Tests for joint_29d scheme-A absolute/delta conversion."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from pi_dex.core.actions import JOINT_ARM_HAND_DIM
from pi_dex.core.actions import JOINT_LOGICAL_ACTION_DIM
from pi_dex.core.actions import MOTOR_JOINT_DIM
from pi_dex.core.actions import ActionRepresentation
from pi_dex.core.actions import append_motor_to_arm_hand
from pi_dex.core.joint_action_mode import absolute_joint_actions_to_delta
from pi_dex.core.joint_action_mode import bimanual_joint_reference_from_state
from pi_dex.core.joint_action_mode import delta_joint_actions_to_absolute
from pi_dex.core.joint_action_mode import maybe_compose_deployment_joint_actions
from pi_dex.core.joint_action_mode import maybe_convert_training_joint_actions
from pi_dex.core.spec import ActionMode
from tests.helpers import spec_for_representation

_STATE_DIM = JOINT_ARM_HAND_DIM * 2 + MOTOR_JOINT_DIM  # 65


@pytest.fixture
def joint_action_spec(action_spec):
    return spec_for_representation(action_spec, ActionRepresentation.JOINT_29D)


def _full_side(arm_hand: np.ndarray, motor: np.ndarray) -> np.ndarray:
    return append_motor_to_arm_hand(arm_hand, motor)


def test_bimanual_joint_reference_from_state_splits_left_right(joint_action_spec) -> None:
    state = np.arange(_STATE_DIM, dtype=np.float32)
    left, right = bimanual_joint_reference_from_state(state, spec=joint_action_spec)
    expected_left = np.concatenate([state[:29], state[58:65]])
    expected_right = np.concatenate([state[29:58], state[58:65]])
    np.testing.assert_array_equal(left, expected_left)
    np.testing.assert_array_equal(right, expected_right)
    assert left.shape == (JOINT_LOGICAL_ACTION_DIM,)


def test_scheme_a_delta_roundtrip(joint_action_spec) -> None:
    delta_spec = dataclasses.replace(joint_action_spec, action_mode=ActionMode.DELTA)
    state = np.linspace(0.1, 0.5, _STATE_DIM, dtype=np.float32)
    left_ref = np.concatenate([state[:29], state[58:65]])
    right_ref = np.concatenate([state[29:58], state[58:65]])
    left_abs = np.stack([left_ref + 0.01, left_ref + 0.02], axis=0)
    right_abs = np.stack([right_ref + 0.03, right_ref + 0.04], axis=0)

    left_delta, right_delta = absolute_joint_actions_to_delta(
        left_abs,
        right_abs,
        state,
        spec=delta_spec,
    )
    np.testing.assert_allclose(left_delta[0], 0.01, rtol=0, atol=1e-6)
    np.testing.assert_allclose(left_delta[1], 0.02, rtol=0, atol=1e-6)
    np.testing.assert_allclose(right_delta[0], 0.03, rtol=0, atol=1e-6)
    np.testing.assert_allclose(right_delta[1], 0.04, rtol=0, atol=1e-6)

    left_out, right_out = delta_joint_actions_to_absolute(
        left_delta,
        right_delta,
        state,
        spec=delta_spec,
    )
    np.testing.assert_allclose(left_out, left_abs)
    np.testing.assert_allclose(right_out, right_abs)


def test_maybe_convert_training_joint_actions_absolute_passthrough(joint_action_spec) -> None:
    k = joint_action_spec.physical_horizon
    left = np.ones((k, JOINT_LOGICAL_ACTION_DIM), dtype=np.float32)
    right = np.ones((k, JOINT_LOGICAL_ACTION_DIM), dtype=np.float32) * 2
    state = np.zeros(_STATE_DIM, dtype=np.float32)
    out_left, out_right = maybe_convert_training_joint_actions(left, right, state, spec=joint_action_spec)
    np.testing.assert_array_equal(out_left, left)
    np.testing.assert_array_equal(out_right, right)


def test_maybe_compose_deployment_joint_actions_delta(joint_action_spec) -> None:
    delta_spec = dataclasses.replace(joint_action_spec, action_mode=ActionMode.DELTA)
    state = np.linspace(0.0, 1.0, _STATE_DIM, dtype=np.float32)
    k = joint_action_spec.physical_horizon
    left_ref = np.concatenate([state[:29], state[58:65]])
    right_ref = np.concatenate([state[29:58], state[58:65]])
    left_delta = np.zeros((k, JOINT_LOGICAL_ACTION_DIM), dtype=np.float32)
    right_delta = np.zeros((k, JOINT_LOGICAL_ACTION_DIM), dtype=np.float32)
    left_delta[:, 0] = 0.05
    left_out, right_out = maybe_compose_deployment_joint_actions(
        left_delta,
        right_delta,
        state,
        spec=delta_spec,
    )
    np.testing.assert_allclose(left_out[:, 0], left_ref[0] + 0.05)
    for joint_idx in range(1, JOINT_LOGICAL_ACTION_DIM):
        np.testing.assert_allclose(left_out[:, joint_idx], left_ref[joint_idx])
    for row in right_out:
        np.testing.assert_allclose(row, right_ref)


def test_delta_training_rejects_residual(joint_action_spec) -> None:
    residual_spec = dataclasses.replace(joint_action_spec, action_mode=ActionMode.RESIDUAL)
    with pytest.raises(ValueError, match="unsupported action_mode"):
        maybe_convert_training_joint_actions(
            np.zeros((joint_action_spec.physical_horizon, JOINT_LOGICAL_ACTION_DIM), dtype=np.float32),
            np.zeros((joint_action_spec.physical_horizon, JOINT_LOGICAL_ACTION_DIM), dtype=np.float32),
            np.zeros(_STATE_DIM, dtype=np.float32),
            spec=residual_spec,
        )
