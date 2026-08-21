"""OpenPI ``Environment`` adapter over harobotsDL ``NorthZmqEnv``."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from pi_dex.data.observation_contract import SharpaObservationContract
from pi_dex.robot.realtime_actions import ARM_DIM
from pi_dex.robot.realtime_actions import SDK_LEFT_ARM_ACTION
from pi_dex.robot.realtime_actions import SDK_LEFT_HAND_ACTION
from pi_dex.robot.realtime_actions import SDK_MOTOR_ACTION
from pi_dex.robot.realtime_actions import SDK_RIGHT_ARM_ACTION
from pi_dex.robot.realtime_actions import SDK_RIGHT_HAND_ACTION
from pi_dex.robot.realtime_actions import split_joint_29d_hand_chunk
from pi_dex.robot.realtime_observation import build_policy_observation_from_sdk
from pi_dex.robot.realtime_observation import resolve_live_prompt
from pi_dex.robot.realtime_observation import resolve_observation_timestamp_ns
from pi_dex.robot.north_env import NorthZmqEnv
from pi_dex.core.spec import BimanualActionSpec

logger = logging.getLogger(__name__)


def is_inference_operation_mode(obs: Mapping[str, Any]) -> bool:
    mode = obs.get("mode")
    if not isinstance(mode, Mapping):
        return False
    return int(mode.get("operation_mode", 0)) == 2


def is_action_on_work(obs: Mapping[str, Any]) -> bool:
    """harobotsDL work gate: moving under inference."""
    mode = obs.get("mode")
    if not isinstance(mode, Mapping):
        return False
    return int(mode.get("state", 0)) == 2 and int(mode.get("sub_state", 0)) == 1


def should_publish_actions(obs: Mapping[str, Any]) -> bool:
    return is_inference_operation_mode(obs) and is_action_on_work(obs)


class NorthRealEnvironment(_environment.Environment):
    """OpenPI Environment: policy observation in, one broker step out to Zenoh."""

    def __init__(
        self,
        hardware: NorthZmqEnv,
        *,
        contract: SharpaObservationContract,
        spec: BimanualActionSpec,
        prompt: str | None = None,
        include_motor_hold: bool = False,
        obs_timeout_s: float = 30.0,
        poll_interval_s: float = 0.005,
    ) -> None:
        self._hw = hardware
        self._contract = contract
        self._spec = spec
        self._prompt = prompt
        self._include_motor_hold = include_motor_hold
        self._obs_timeout_s = float(obs_timeout_s)
        self._poll_interval_s = float(poll_interval_s)
        self._last_sdk_obs: dict[str, Any] | None = None
        self._skip_publish = False

    @property
    def hardware(self) -> NorthZmqEnv:
        return self._hw

    @property
    def last_sdk_observation(self) -> dict[str, Any] | None:
        return None if self._last_sdk_obs is None else dict(self._last_sdk_obs)

    @override
    def reset(self) -> None:
        self._hw.reset()
        self._last_sdk_obs = None
        self._skip_publish = False
        deadline = time.monotonic() + self._obs_timeout_s
        while time.monotonic() < deadline:
            obs = self._hw.get_observation()
            if obs is not None:
                self._last_sdk_obs = obs
                return
            time.sleep(self._poll_interval_s)
        raise TimeoutError(
            f"NorthRealEnvironment.reset: no observation within {self._obs_timeout_s}s"
        )

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        deadline = time.monotonic() + self._obs_timeout_s
        while True:
            sdk = self._hw.get_observation()
            if sdk is not None:
                self._last_sdk_obs = sdk
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("NorthRealEnvironment.get_observation: timed out")
            time.sleep(self._poll_interval_s)

        assert self._last_sdk_obs is not None
        self._skip_publish = not should_publish_actions(self._last_sdk_obs)
        if self._skip_publish:
            self._hw.clear_action_and_history()

        prompt = resolve_live_prompt(self._last_sdk_obs, fallback=self._prompt)
        observation = build_policy_observation_from_sdk(
            self._last_sdk_obs,
            self._contract,
            prompt=prompt,
            observation_timestamp_ns=resolve_observation_timestamp_ns(self._last_sdk_obs),
            clock_domain=self._spec.clock_domain,
        )
        # Local-only: mode gate for GatedPolicyAgent (stripped before Websocket infer).
        observation["_north_sdk"] = self._last_sdk_obs
        return observation

    @override
    def apply_action(self, action: dict) -> None:
        if self._skip_publish or action.get("skip"):
            self._hw.clear_action_and_history()
            return
        left = np.asarray(action["actions"]["left"], dtype=np.float32)
        right = np.asarray(action["actions"]["right"], dtype=np.float32)
        if left.ndim != 1 or left.shape[0] != self._spec.logical_action_dim:
            raise ValueError(
                f"actions.left: expected shape ({self._spec.logical_action_dim},), got {left.shape}"
            )
        if right.shape != left.shape:
            raise ValueError(f"actions.right shape {right.shape} != left {left.shape}")

        left_arm, left_hand, left_motor = split_joint_29d_hand_chunk(left[None, :])
        right_arm, right_hand, _right_motor = split_joint_29d_hand_chunk(right[None, :])
        step_sdk: dict[str, list[float]] = {
            SDK_LEFT_ARM_ACTION: left_arm[0].tolist(),
            SDK_LEFT_HAND_ACTION: left_hand[0].tolist(),
            SDK_RIGHT_ARM_ACTION: right_arm[0].tolist(),
            SDK_RIGHT_HAND_ACTION: right_hand[0].tolist(),
            SDK_MOTOR_ACTION: left_motor[0].tolist(),
        }
        if self._include_motor_hold and self._last_sdk_obs is not None:
            motor = self._last_sdk_obs.get("/state/motor/joint_angle")
            if motor is not None:
                motor_arr = np.asarray(motor, dtype=np.float32).reshape(-1)
                if motor_arr.shape[0] == ARM_DIM:
                    step_sdk[SDK_MOTOR_ACTION] = motor_arr.tolist()

        self._hw.enqueue_single_step(step_sdk)


def hold_action(spec: BimanualActionSpec) -> dict[str, Any]:
    """Zero joint step used when the pendant is not in work mode."""
    zeros = np.zeros((spec.logical_action_dim,), dtype=np.float32)
    return {"actions": {"left": zeros, "right": zeros}, "skip": True}
