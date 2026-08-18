"""PI-DEX joint_29d model server (WebSocket).

Robot-side clients keep Zenoh/SDK locally. This process only:

1. loads a training checkpoint as :class:`~pi_dex.deployment.BimanualPolicyAdapter`
2. handshakes with PI-DEX deployment metadata
3. accepts OpenPI-style observations (+ ``observation_timestamp_ns`` /
   ``clock_domain``) and returns wire v3 action chunks

Wire format matches OpenPI's msgpack-numpy WebSocket policy protocol so existing
``WebsocketClientPolicy`` clients can talk to it, with safer defaults than the
vendored stock server (bound host, message size cap, no traceback leak).
"""

from __future__ import annotations

import argparse
import asyncio
import http
import logging
import pathlib
import socket
import time
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MAX_MESSAGE_BYTES = 32 * 1024 * 1024


class PiDexWebsocketPolicyServer:
    """Serve a :class:`~pi_dex.deployment.BimanualPolicyAdapter` over WebSocket."""

    def __init__(
        self,
        policy: Any,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        metadata: Mapping[str, Any] | None = None,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
        api_key: str | None = None,
        infer_timeout_s: float | None = None,
    ) -> None:
        if type(port) is not int or port <= 0:
            raise ValueError(f"port: expected positive int, got {port!r}")
        if type(max_message_bytes) is not int or max_message_bytes <= 0:
            raise ValueError(f"max_message_bytes: expected positive int, got {max_message_bytes!r}")
        if infer_timeout_s is not None:
            if type(infer_timeout_s) not in (float, int) or float(infer_timeout_s) <= 0:
                raise ValueError(f"infer_timeout_s: expected positive number or None, got {infer_timeout_s!r}")
        self._policy = policy
        self._host = host
        self._port = port
        if metadata is None:
            metadata = getattr(policy, "metadata", None)
            if callable(metadata):
                metadata = metadata()
        if not isinstance(metadata, Mapping):
            raise TypeError("policy metadata: expected a mapping")
        self._metadata = dict(metadata)
        self._max_message_bytes = max_message_bytes
        self._api_key = api_key.strip() if type(api_key) is str and api_key.strip() else None
        self._infer_timeout_s = float(infer_timeout_s) if infer_timeout_s is not None else None

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def serve_forever(self) -> None:
        """Block and serve until interrupted."""
        asyncio.run(self.run())

    async def run(self) -> None:
        try:
            import websockets.asyncio.server as ws_server
            from openpi_client import msgpack_numpy
        except ImportError as error:
            raise ImportError(
                "pi-dex-serve requires openpi_client + websockets "
                "(install the OpenPI env and editable pi-dex)"
            ) from error

        packer = msgpack_numpy.Packer()

        async def handler(websocket: Any) -> None:
            await self._handler(websocket, packer=packer, unpackb=msgpack_numpy.unpackb)

        def process_request(connection: Any, request: Any) -> Any:
            if request.path == "/healthz":
                return connection.respond(http.HTTPStatus.OK, "OK\n")
            if self._api_key is not None:
                auth = request.headers.get("Authorization", "")
                expected = f"Api-Key {self._api_key}"
                if auth != expected:
                    return connection.respond(http.HTTPStatus.UNAUTHORIZED, "unauthorized\n")
            return None

        logging.getLogger("websockets.server").setLevel(logging.INFO)
        async with ws_server.serve(
            handler,
            self._host,
            self._port,
            compression=None,
            max_size=self._max_message_bytes,
            process_request=process_request,
        ):
            logger.info(
                "PI-DEX policy server listening on ws://%s:%s (max_message_bytes=%s)",
                self._host,
                self._port,
                self._max_message_bytes,
            )
            await asyncio.Future()

    async def _handler(self, websocket: Any, *, packer: Any, unpackb: Any) -> None:
        import websockets

        peer = getattr(websocket, "remote_address", None)
        logger.info("connection opened from %s", peer)
        await websocket.send(packer.pack(self._metadata))

        prev_total_time: float | None = None
        while True:
            try:
                start_time = time.monotonic()
                raw = await websocket.recv()
                if isinstance(raw, str):
                    await websocket.send("invalid payload: expected msgpack bytes")
                    await websocket.close(code=1003, reason="expected binary msgpack")
                    break
                if len(raw) > self._max_message_bytes:
                    await websocket.send("payload too large")
                    await websocket.close(code=1009, reason="message too large")
                    break

                observation = unpackb(raw)
                if not isinstance(observation, dict):
                    await websocket.send("invalid observation: expected mapping")
                    continue

                infer_started = time.monotonic()
                if self._infer_timeout_s is None:
                    action = await asyncio.to_thread(self._policy.infer, observation)
                else:
                    action = await asyncio.wait_for(
                        asyncio.to_thread(self._policy.infer, observation),
                        timeout=self._infer_timeout_s,
                    )
                infer_ms = (time.monotonic() - infer_started) * 1000.0
                if not isinstance(action, dict):
                    raise TypeError(f"policy.infer: expected dict, got {type(action).__name__}")
                action = dict(action)
                action["server_timing"] = {"infer_ms": infer_ms}
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000.0
                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time
            except websockets.ConnectionClosed:
                logger.info("connection closed from %s", peer)
                break
            except TimeoutError:
                logger.exception("inference timeout")
                await websocket.send("inference timeout")
                await websocket.close(code=1011, reason="inference timeout")
                break
            except Exception as error:  # noqa: BLE001 - boundary: never leak traceback to client
                logger.exception("inference failed")
                await websocket.send(f"inference failed: {type(error).__name__}")
                await websocket.close(code=1011, reason="inference failed")
                break


def build_joint29d_server_policy(
    *,
    checkpoint_dir: pathlib.Path | str,
    observation_contract: pathlib.Path | str,
    assets_dir: pathlib.Path | str | None = None,
    asset_id: str = "sharpa_joint_29d",
    robot_id: str = "POC22027",
    embodiment_version: str = "sharpa_north_v1",
    command_semantics_version: str = "sharpa_sdk_commanded_joint_position_absolute_v1",
    hand_mapping_version: str = "sharpa_north_hand_mapping_v1",
    clock_domain: str = "unix_realtime",
    execution_horizon: int | None = None,
    default_prompt: str | None = None,
    pytorch_device: str | None = None,
    allow_unreviewed_contract: bool = False,
) -> Any:
    """Load checkpoint + reviewed contract into a deployment policy adapter."""
    from pi_dex.observation_contract import load_observation_contract
    from pi_dex.realtime_inference import load_joint29d_policy
    from pi_dex.training_runner import build_joint_spec_from_contract

    contract = load_observation_contract(observation_contract)
    if not allow_unreviewed_contract:
        contract.require_reviewed_for_training()
    spec = build_joint_spec_from_contract(
        contract,
        robot_id=robot_id,
        embodiment_version=embodiment_version,
        command_semantics_version=command_semantics_version,
        hand_mapping_version=hand_mapping_version,
        clock_domain=clock_domain,
    )
    return load_joint29d_policy(
        checkpoint_dir=checkpoint_dir,
        spec=spec,
        asset_id=asset_id,
        assets_dirs=assets_dir,
        pytorch_device=pytorch_device,
        execution_horizon=execution_horizon,
        default_prompt=default_prompt,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pi-dex-serve",
        description="Serve a joint_29d PI-DEX checkpoint as a WebSocket model server",
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--observation-contract", required=True)
    parser.add_argument("--assets-dir", default="")
    parser.add_argument("--asset-id", default="sharpa_joint_29d")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--max-message-bytes", type=int, default=DEFAULT_MAX_MESSAGE_BYTES)
    parser.add_argument("--api-key", default="", help="Optional Api-Key; clients send Authorization: Api-Key …")
    parser.add_argument("--infer-timeout-s", type=float, default=None)
    parser.add_argument("--execution-horizon", type=int, default=None)
    parser.add_argument("--default-prompt", default="")
    parser.add_argument("--pytorch-device", default=None)
    parser.add_argument("--robot-id", default="POC22027")
    parser.add_argument("--embodiment-version", default="sharpa_north_v1")
    parser.add_argument(
        "--command-semantics-version",
        default="sharpa_sdk_commanded_joint_position_absolute_v1",
    )
    parser.add_argument("--hand-mapping-version", default="sharpa_north_hand_mapping_v1")
    parser.add_argument("--clock-domain", default="unix_realtime")
    parser.add_argument("--allow-unreviewed-contract", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    policy = build_joint29d_server_policy(
        checkpoint_dir=args.checkpoint_dir,
        observation_contract=args.observation_contract,
        assets_dir=args.assets_dir or None,
        asset_id=args.asset_id,
        robot_id=args.robot_id,
        embodiment_version=args.embodiment_version,
        command_semantics_version=args.command_semantics_version,
        hand_mapping_version=args.hand_mapping_version,
        clock_domain=args.clock_domain,
        execution_horizon=args.execution_horizon,
        default_prompt=args.default_prompt or None,
        pytorch_device=args.pytorch_device,
        allow_unreviewed_contract=args.allow_unreviewed_contract,
    )

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError:
        local_ip = "unknown"
    logger.info("starting PI-DEX server host=%s ip=%s bind=%s:%s", hostname, local_ip, args.host, args.port)
    logger.info("metadata keys=%s", sorted(policy.metadata.keys()))

    server = PiDexWebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
        max_message_bytes=args.max_message_bytes,
        api_key=args.api_key or None,
        infer_timeout_s=args.infer_timeout_s,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
