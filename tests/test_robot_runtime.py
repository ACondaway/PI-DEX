"""Tests for OpenPI+harobotsDL robot package (no Zenoh / no GPU)."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from openpi_client import action_chunk_broker

from pi_dex.robot.north_codec import sdk_action_chunk_to_step_dicts
from pi_dex.data.observation_contract import load_observation_contract
from pi_dex.robot.realtime_actions import JOINT_29D_DIM
from pi_dex.robot.realtime_actions import SDK_LEFT_ARM_ACTION
from pi_dex.robot.realtime_actions import policy_result_to_sdk_action_dict
from pi_dex.robot.environment import hold_action
from pi_dex.robot.environment import is_action_on_work
from pi_dex.robot.environment import should_publish_actions
from pi_dex.robot.main import main as robot_main
from pi_dex.robot.north_env import NorthZmqEnv
from pi_dex.robot.north_env import apply_first_chunk_smooth
from pi_dex.robot.north_env import smooth_action_accel
from pi_dex.robot.remote_policy import WebsocketJointPolicy
from pi_dex.training.training_runner import build_joint_spec_from_contract


ROOT = pathlib.Path(__file__).resolve().parents[1]
REVIEWED_CONTRACT = ROOT / "configs/site/joint_29d_observation.reviewed.json"


class _FakeClient:
    def __init__(self, left: np.ndarray, right: np.ndarray) -> None:
        self._left = left
        self._right = right

    def infer(self, obs: dict) -> dict:
        assert "_north_sdk" not in obs
        return {"actions": {"left": self._left.copy(), "right": self._right.copy()}}

    def get_server_metadata(self) -> dict:
        return {"pi_dex": {"clock_domain": "unix_realtime", "action_mode": "absolute"}}

    def reset(self) -> None:
        return None


def test_mode_gates() -> None:
    assert should_publish_actions({"mode": {"operation_mode": 2, "state": 2, "sub_state": 1}})
    assert not is_action_on_work({"mode": {"operation_mode": 2, "state": 2, "sub_state": 0}})


def test_north_env_replace_and_clear() -> None:
    env = NorthZmqEnv(decode_images=False)
    chunk = {
        SDK_LEFT_ARM_ACTION: np.zeros((4, 7), dtype=np.float32).tolist(),
        "/action/left_hand/joint_angle": np.zeros((4, 22), dtype=np.float32).tolist(),
        "/action/right_arm/joint_angle": np.zeros((4, 7), dtype=np.float32).tolist(),
        "/action/right_hand/joint_angle": np.zeros((4, 22), dtype=np.float32).tolist(),
    }
    assert env._send_actions(chunk) == 4
    assert env.action_buffer_len() == 4
    env.clear_action_and_history()
    assert env.action_buffer_len() == 0
    env.enqueue_single_step({k: v[0] for k, v in chunk.items()})
    assert env.action_buffer_len() == 1


def test_smooth_and_first_chunk() -> None:
    chunk = np.asarray([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]], dtype=np.float32)
    smoothed = smooth_action_accel(chunk, np.zeros(2, dtype=np.float32), 2)
    assert smoothed.shape == chunk.shape
    actions = {
        SDK_LEFT_ARM_ACTION: np.ones((3, 7), dtype=np.float32).tolist(),
        "/action/left_hand/joint_angle": np.ones((3, 22), dtype=np.float32).tolist(),
        "/action/right_arm/joint_angle": np.ones((3, 7), dtype=np.float32).tolist(),
        "/action/right_hand/joint_angle": np.ones((3, 22), dtype=np.float32).tolist(),
    }
    state = {
        "/state/left_arm/joint_angle": [0.0] * 7,
        "/state/left_hand/joint_angle": [0.0] * 22,
        "/state/right_arm/joint_angle": [0.0] * 7,
        "/state/right_hand/joint_angle": [0.0] * 22,
    }
    out = apply_first_chunk_smooth(actions, state, smooth_chunk_size=2)
    assert len(out[SDK_LEFT_ARM_ACTION]) == 3


def test_websocket_joint_policy_offset_and_broker() -> None:
    contract = load_observation_contract(REVIEWED_CONTRACT)
    spec = build_joint_spec_from_contract(
        contract,
        robot_id="POC22005",
        embodiment_version="sharpa_north_v1",
        command_semantics_version="sharpa_sdk_commanded_joint_position_absolute_v1",
        hand_mapping_version="sharpa_north_hand_mapping_v1",
        clock_domain="unix_realtime",
    )
    horizon = int(contract.physical_horizon)
    left = np.arange(horizon * JOINT_29D_DIM, dtype=np.float32).reshape(horizon, JOINT_29D_DIM)
    right = left + 1
    policy = WebsocketJointPolicy(
        _FakeClient(left, right),
        spec=spec,
        contract=contract,
        offset=1,
        output_chunk=horizon - 1,
    )
    broker = action_chunk_broker.ActionChunkBroker(policy=policy, action_horizon=policy.output_chunk)
    obs = {
        "state": np.zeros(contract.state_dim, dtype=np.float32),
        "image": {},
        "image_mask": {},
        "prompt": "x",
        "observation_timestamp_ns": 1,
        "clock_domain": "unix_realtime",
        "_north_sdk": {"mode": {"operation_mode": 2, "state": 2, "sub_state": 1}},
    }
    step0 = broker.infer(obs)
    assert step0["actions"]["left"].shape == (JOINT_29D_DIM,)
    np.testing.assert_allclose(step0["actions"]["left"], left[1])
    # Mid-chunk: no second remote call (fake has no counter; ensure shapes stable).
    step1 = broker.infer(obs)
    np.testing.assert_allclose(step1["actions"]["left"], left[2])


def test_hold_action_shape() -> None:
    contract = load_observation_contract(REVIEWED_CONTRACT)
    spec = build_joint_spec_from_contract(
        contract,
        robot_id="POC22005",
        embodiment_version="sharpa_north_v1",
        command_semantics_version="sharpa_sdk_commanded_joint_position_absolute_v1",
        hand_mapping_version="sharpa_north_hand_mapping_v1",
        clock_domain="unix_realtime",
    )
    held = hold_action(spec)
    assert held["skip"] is True
    assert held["actions"]["left"].shape == (JOINT_29D_DIM,)


def test_robot_client_codec_smoke_cli() -> None:
    code = robot_main(
        [
            "--mode",
            "codec-smoke",
            "--observation-contract",
            str(REVIEWED_CONTRACT),
            "--prompt",
            "smoke",
        ]
    )
    assert code == 0


def test_sdk_chunk_steps_roundtrip() -> None:
    contract = load_observation_contract(REVIEWED_CONTRACT)
    spec = build_joint_spec_from_contract(
        contract,
        robot_id="POC22005",
        embodiment_version="sharpa_north_v1",
        command_semantics_version="sharpa_sdk_commanded_joint_position_absolute_v1",
        hand_mapping_version="sharpa_north_hand_mapping_v1",
        clock_domain="unix_realtime",
    )
    left = np.zeros((2, JOINT_29D_DIM), dtype=np.float32)
    sdk = policy_result_to_sdk_action_dict({"actions": {"left": left, "right": left}}, spec)
    steps = sdk_action_chunk_to_step_dicts(sdk)
    assert len(steps) == 2
