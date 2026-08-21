"""Remote joint_29d policy for OpenPI ``ActionChunkBroker``.

Calls ``pi-dex-serve`` over Websocket (OpenPI client), applies delta→absolute
compose, latency ``offset``, and optional first-chunk smooth on the full chunk
before the broker consumes open-loop steps.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from pi_dex.core.joint_action_mode import maybe_compose_deployment_joint_actions
from pi_dex.data.observation_contract import SharpaObservationContract
from pi_dex.robot.realtime_actions import policy_result_to_sdk_action_dict
from pi_dex.robot.north_env import apply_first_chunk_smooth
from pi_dex.core.spec import BimanualActionSpec


class WebsocketJointPolicy(_base_policy.BasePolicy):
    """OpenPI ``BasePolicy`` over ``WebsocketClientPolicy`` + PI-DEX compose."""

    def __init__(
        self,
        client: Any,
        *,
        spec: BimanualActionSpec,
        contract: SharpaObservationContract,
        offset: int = 0,
        output_chunk: int | None = None,
        first_chunk_smooth_size: int = 0,
        include_motor_hold: bool = False,
        get_sdk_observation: Any | None = None,
    ) -> None:
        if offset < 0:
            raise ValueError(f"offset: expected >= 0, got {offset}")
        self._client = client
        self._spec = spec
        self._contract = contract
        self._offset = int(offset)
        physical = int(contract.physical_horizon)
        max_out = physical - self._offset
        if max_out <= 0:
            raise ValueError(
                f"offset {self._offset} leaves no steps in physical_horizon {physical}"
            )
        self._output_chunk = max_out if output_chunk is None else int(output_chunk)
        if self._output_chunk <= 0 or self._output_chunk > max_out:
            raise ValueError(
                f"output_chunk: expected in [1, {max_out}], got {self._output_chunk}"
            )
        self._first_chunk_smooth_size = int(first_chunk_smooth_size)
        self._include_motor_hold = include_motor_hold
        self._get_sdk_observation = get_sdk_observation
        self._metadata = dict(getattr(client, "get_server_metadata", lambda: {})() or {})

    @property
    def output_chunk(self) -> int:
        return self._output_chunk

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def get_server_metadata(self) -> dict[str, Any]:
        return self.metadata

    @override
    def infer(self, obs: dict) -> dict:  # noqa: UP006
        wire_obs = {key: value for key, value in obs.items() if not str(key).startswith("_")}
        result = self._client.infer(wire_obs)
        actions = result["actions"]
        left = np.asarray(actions["left"], dtype=np.float32)
        right = np.asarray(actions["right"], dtype=np.float32)
        state = np.asarray(wire_obs["state"], dtype=np.float32)
        left, right = maybe_compose_deployment_joint_actions(
            left, right, state, spec=self._spec
        )

        if self._offset > 0 or self._output_chunk < left.shape[0]:
            stop = min(self._offset + self._output_chunk, left.shape[0])
            if self._offset >= stop:
                raise ValueError(
                    f"empty publish window: offset={self._offset}, stop={stop}, "
                    f"horizon={left.shape[0]}"
                )
            left = left[self._offset : stop]
            right = right[self._offset : stop]

        if self._first_chunk_smooth_size > 0 and self._get_sdk_observation is not None:
            sdk = self._get_sdk_observation()
            if isinstance(sdk, Mapping):
                motor = None
                if self._include_motor_hold and "/state/motor/joint_angle" in sdk:
                    motor = np.asarray(sdk["/state/motor/joint_angle"], dtype=np.float32)
                sdk_chunk = policy_result_to_sdk_action_dict(
                    {"actions": {"left": left, "right": right}},
                    self._spec,
                    motor_positions=motor,
                )
                smoothed = apply_first_chunk_smooth(
                    sdk_chunk,
                    sdk,
                    smooth_chunk_size=self._first_chunk_smooth_size,
                )
                # Rebuild left/right [T,29] from smoothed arm/hand lists.
                left_arm = np.asarray(smoothed["/action/left_arm/joint_angle"], dtype=np.float32)
                left_hand = np.asarray(smoothed["/action/left_hand/joint_angle"], dtype=np.float32)
                right_arm = np.asarray(smoothed["/action/right_arm/joint_angle"], dtype=np.float32)
                right_hand = np.asarray(smoothed["/action/right_hand/joint_angle"], dtype=np.float32)
                motor = np.asarray(
                    smoothed.get("/action/motor/joint_angle", left[:, self._spec.logical_action_dim - 7 :]),
                    dtype=np.float32,
                )
                left = np.concatenate([left_arm, left_hand, motor], axis=1)
                right = np.concatenate([right_arm, right_hand, motor], axis=1)

        out = dict(result)
        out["actions"] = {"left": left, "right": right}
        return out

    @override
    def reset(self) -> None:
        reset = getattr(self._client, "reset", None)
        if callable(reset):
            reset()
