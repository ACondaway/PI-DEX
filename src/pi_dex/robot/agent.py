"""OpenPI PolicyAgent with harobotsDL pendant mode gating."""

from __future__ import annotations

from typing import Any

from openpi_client import base_policy as _base_policy
from openpi_client.runtime import agent as _agent
from openpi_client.runtime.agents import policy_agent as _policy_agent
from typing_extensions import override

from pi_dex.robot.environment import hold_action
from pi_dex.robot.environment import should_publish_actions
from pi_dex.core.spec import BimanualActionSpec


class GatedPolicyAgent(_agent.Agent):
    """Reset the inner OpenPI broker when leaving inference work mode."""

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        *,
        spec: BimanualActionSpec,
        on_standby: Any | None = None,
    ) -> None:
        self._inner = _policy_agent.PolicyAgent(policy)
        self._spec = spec
        self._on_standby = on_standby
        self._was_active = False

    @override
    def get_action(self, observation: dict) -> dict:
        # Mode lives on the raw North SDK snapshot attached by Environment when skip.
        # Policy observations do not carry mode; Environment sets skip via apply path.
        # We gate using optional ``_north_mode`` injected by Environment into observation.
        mode_obs = observation.get("_north_sdk")
        if isinstance(mode_obs, dict) and not should_publish_actions(mode_obs):
            if self._was_active:
                self._inner.reset()
                if callable(self._on_standby):
                    self._on_standby()
            self._was_active = False
            return hold_action(self._spec)
        self._was_active = True
        return self._inner.get_action(observation)

    @override
    def reset(self) -> None:
        self._was_active = False
        self._inner.reset()
