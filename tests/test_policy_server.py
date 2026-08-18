"""Loopback tests for PI-DEX WebSocket policy server (no GPU / OpenPI weights)."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pytest

from pi_dex.actions import JOINT_LOGICAL_ACTION_DIM
from pi_dex.actions import ActionRepresentation
from pi_dex.deployment import BimanualPolicyAdapter
from pi_dex.deployment import SESSION_ID_FIELD
from pi_dex.deployment import validate_deployment_metadata
from pi_dex.policy_server import PiDexWebsocketPolicyServer
from pi_dex.spec import BimanualActionSpec
from tests.helpers import spec_for_representation


class _FakeJointPolicy:
    def __init__(self, action_spec: BimanualActionSpec) -> None:
        self.spec = spec_for_representation(action_spec, ActionRepresentation.JOINT_29D)
        self.calls = 0
        self.metadata: dict[str, Any] = {"pi_dex": self.spec.to_metadata()}
        dim = self.spec.logical_action_dim
        self.left = np.zeros((self.spec.physical_horizon, dim), dtype=np.float32)
        self.right = np.zeros((self.spec.physical_horizon, dim), dtype=np.float32)
        self.left[:, 0] = np.arange(self.spec.physical_horizon, dtype=np.float32) + 1.0
        self.right[:, 0] = np.arange(self.spec.physical_horizon, dtype=np.float32) + 3.0

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        del observation
        return {"left_actions": self.left.copy(), "right_actions": self.right.copy()}

    def reset(self) -> None:
        return None


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_observation(*, clock_domain: str) -> dict[str, Any]:
    return {
        "state": np.zeros((4,), dtype=np.float32),
        "observation_timestamp_ns": 1_000_000_000,
        "clock_domain": clock_domain,
    }


def test_policy_server_loopback_infer(action_spec: BimanualActionSpec) -> None:
    """Requires websockets + openpi_client in the active interpreter."""
    pytest.importorskip("websockets")
    pytest.importorskip("openpi_client")
    import websockets.sync.client as ws_client
    from openpi_client import msgpack_numpy

    fake = _FakeJointPolicy(action_spec)
    adapter = BimanualPolicyAdapter(fake, fake.spec, execution_horizon=1)
    port = _free_port()
    server = PiDexWebsocketPolicyServer(
        adapter,
        host="127.0.0.1",
        port=port,
        metadata=adapter.metadata,
        max_message_bytes=4 * 1024 * 1024,
        api_key="test-key",
    )

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5.0
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            packer = msgpack_numpy.Packer()
            uri = f"ws://127.0.0.1:{port}"
            with ws_client.connect(
                uri,
                compression=None,
                max_size=4 * 1024 * 1024,
                additional_headers={"Authorization": "Api-Key test-key"},
                open_timeout=1,
            ) as conn:
                metadata = msgpack_numpy.unpackb(conn.recv())
                assert isinstance(metadata, dict)
                validate_deployment_metadata(metadata, fake.spec, expected_execution_horizon=1)
                assert SESSION_ID_FIELD in metadata["pi_dex"]

                conn.send(packer.pack(_make_observation(clock_domain=fake.spec.clock_domain)))
                response = conn.recv()
                assert isinstance(response, (bytes, bytearray))
                payload = msgpack_numpy.unpackb(response)
                assert payload["actions"]["left"].shape == (1, JOINT_LOGICAL_ACTION_DIM)
                assert payload["actions"]["right"].shape == (1, JOINT_LOGICAL_ACTION_DIM)
                assert payload[SESSION_ID_FIELD] == metadata["pi_dex"][SESSION_ID_FIELD]
                assert "server_timing" in payload
                assert fake.calls == 1
                return
        except Exception as error:  # noqa: BLE001 - retry until server is listening
            last_error = error
            time.sleep(0.05)
    raise AssertionError(f"server loopback failed: {last_error}")


def test_policy_server_rejects_bad_timeout(action_spec: BimanualActionSpec) -> None:
    fake = _FakeJointPolicy(action_spec)
    adapter = BimanualPolicyAdapter(fake, fake.spec, execution_horizon=1)
    with pytest.raises(ValueError, match="infer_timeout_s"):
        PiDexWebsocketPolicyServer(adapter, infer_timeout_s=0)


def test_policy_server_rejects_unauthorized(action_spec: BimanualActionSpec) -> None:
    pytest.importorskip("websockets")
    pytest.importorskip("openpi_client")
    import websockets.sync.client as ws_client

    fake = _FakeJointPolicy(action_spec)
    adapter = BimanualPolicyAdapter(fake, fake.spec, execution_horizon=1)
    port = _free_port()
    server = PiDexWebsocketPolicyServer(
        adapter,
        host="127.0.0.1",
        port=port,
        api_key="secret",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.3)

    with pytest.raises(Exception):
        ws_client.connect(
            f"ws://127.0.0.1:{port}",
            compression=None,
            additional_headers={"Authorization": "Api-Key wrong"},
            open_timeout=2,
        )
