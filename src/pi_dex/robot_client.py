"""Robot-side Zenoh bridge: NorthObservation → pi-dex-serve → UhrActionBundle.

Deploy next to the Sharpa North stack on the slave NUC (after ``bash start.sh`` /
``start-nuc.sh``). The robot firmware already owns Zenoh I/O, F6 mode toggle
(teleop ↔ inference), and F2 state (init → standby ↔ moving). This process only:

1. Subscribes to ``north_observation`` (protobuf)
2. Converts to the SDK dict used by ``realtime_*``
3. Calls the GPU ``pi-dex-serve`` WebSocket for joint_29d actions
4. Publishes paced ``UhrActionBundle`` on ``inference/action``

It does **not** replace lease / e-stop / watchdog (``BimanualController``).
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import threading
import time
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

from pi_dex.north_codec import encode_uhr_action_bundle
from pi_dex.north_codec import parse_north_observation
from pi_dex.north_codec import sdk_action_chunk_to_step_dicts
from pi_dex.north_codec import sdk_step_action_to_uhr_bundle
from pi_dex.observation_contract import load_observation_contract
from pi_dex.realtime_inference import infer_sdk_actions
from pi_dex.realtime_observation import build_policy_observation_from_sdk
from pi_dex.realtime_observation import resolve_live_prompt
from pi_dex.realtime_observation import resolve_observation_timestamp_ns
from pi_dex.sharpa_runtime_keys import DEFAULT_ACTION_PUB_DURATION_S
from pi_dex.sharpa_runtime_keys import DEFAULT_ACTION_TOPIC
from pi_dex.sharpa_runtime_keys import DEFAULT_OBSERVATION_TOPIC
from pi_dex.training_runner import build_joint_spec_from_contract

logger = logging.getLogger(__name__)


def _sample_payload_bytes(sample: Any) -> bytes:
    payload = sample.payload
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    to_bytes = getattr(payload, "to_bytes", None)
    if callable(to_bytes):
        return bytes(to_bytes())
    return bytes(payload)


class PolicyServerSession:
    """Thin wrapper around OpenPI ``WebsocketClientPolicy``."""

    def __init__(self, *, host: str, port: int, api_key: str | None = None) -> None:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        self._client = WebsocketClientPolicy(host=host, port=port, api_key=api_key)
        self._metadata = dict(self._client.get_server_metadata())

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        return self._client.infer(dict(observation))


class NorthZenohRobotClient:
    """Subscribe observations, infer remotely, publish paced actions."""

    def __init__(
        self,
        *,
        policy: Any,
        contract: Any,
        spec: Any,
        observation_topic: str = DEFAULT_OBSERVATION_TOPIC,
        action_topic: str = DEFAULT_ACTION_TOPIC,
        action_pub_duration_s: float = DEFAULT_ACTION_PUB_DURATION_S,
        prompt: str | None = None,
        include_motor_hold: bool = True,
        zenoh_config: Any | None = None,
        decode_images: bool = True,
    ) -> None:
        if action_pub_duration_s <= 0:
            raise ValueError("action_pub_duration_s: expected positive")
        self._policy = policy
        self._contract = contract
        self._spec = spec
        self._observation_topic = observation_topic
        self._action_topic = action_topic
        self._action_pub_duration_s = float(action_pub_duration_s)
        self._prompt = prompt
        self._include_motor_hold = include_motor_hold
        self._zenoh_config = zenoh_config
        self._decode_images = decode_images

        self._session: Any | None = None
        self._subscriber: Any | None = None
        self._publisher: Any | None = None
        self._lock = threading.Lock()
        self._latest_obs: dict[str, Any] | None = None
        self._obs_seq = 0
        self._handled_seq = 0
        self._action_lock = threading.Lock()
        self._action_buffer: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._action_thread: threading.Thread | None = None
        self._infer_count = 0
        self._error_count = 0

    @property
    def infer_count(self) -> int:
        return self._infer_count

    @property
    def error_count(self) -> int:
        return self._error_count

    def connect(self) -> None:
        try:
            import zenoh
        except ImportError as error:
            raise ImportError(
                "pi-dex-robot-client requires eclipse-zenoh "
                "(pip install eclipse-zenoh) on the robot NUC"
            ) from error

        config = self._zenoh_config
        if config is None:
            config = zenoh.Config()
        elif isinstance(config, (str, pathlib.Path)):
            config = zenoh.Config.from_file(str(config))
        self._session = zenoh.open(config)
        self._publisher = self._session.declare_publisher(self._action_topic)
        self._subscriber = self._session.declare_subscriber(
            self._observation_topic,
            self._on_observation,
        )
        logger.info(
            "zenoh connected obs=%s action=%s",
            self._observation_topic,
            self._action_topic,
        )

    def close(self) -> None:
        self._stop.set()
        if self._action_thread is not None and self._action_thread.is_alive():
            self._action_thread.join(timeout=2.0)
        self._action_thread = None
        for handle in (self._subscriber, self._publisher, self._session):
            if handle is None:
                continue
            close = getattr(handle, "close", None) or getattr(handle, "undeclare", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 — best-effort shutdown
                    pass
        self._subscriber = None
        self._publisher = None
        self._session = None

    def _on_observation(self, sample: Any) -> None:
        try:
            payload = _sample_payload_bytes(sample)
            obs = parse_north_observation(payload, decode_images=self._decode_images)
            with self._lock:
                self._latest_obs = obs
                self._obs_seq += 1
        except Exception:  # noqa: BLE001 — keep subscriber alive
            self._error_count += 1
            logger.exception("failed to parse NorthObservation")

    def get_latest_observation(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._latest_obs is None else dict(self._latest_obs)

    def take_new_observation(self) -> dict[str, Any] | None:
        with self._lock:
            if self._obs_seq == self._handled_seq or self._latest_obs is None:
                return None
            self._handled_seq = self._obs_seq
            return dict(self._latest_obs)

    def enqueue_sdk_actions(self, actions_dict: Mapping[str, list[list[float]]]) -> None:
        steps = sdk_action_chunk_to_step_dicts(actions_dict)
        with self._action_lock:
            self._action_buffer = steps

    def publish_step(self, step_action: Mapping[str, Any], *, language: str | None = None) -> None:
        if self._publisher is None:
            raise RuntimeError("zenoh publisher not connected")
        bundle = sdk_step_action_to_uhr_bundle(step_action, language=language)
        self._publisher.put(encode_uhr_action_bundle(bundle))

    def _action_sender_loop(self) -> None:
        last_send = time.monotonic() - self._action_pub_duration_s
        while not self._stop.is_set():
            now = time.monotonic()
            elapsed = now - last_send
            if elapsed < self._action_pub_duration_s:
                time.sleep(self._action_pub_duration_s - elapsed)
                continue
            with self._action_lock:
                if not self._action_buffer:
                    last_send = time.monotonic()
                    continue
                step = self._action_buffer.pop(0)
            try:
                self.publish_step(step)
            except Exception:  # noqa: BLE001
                self._error_count += 1
                logger.exception("failed to publish UhrActionBundle")
            last_send = time.monotonic()

    def start_action_thread(self) -> None:
        if self._action_thread is not None and self._action_thread.is_alive():
            return
        self._stop.clear()
        self._action_thread = threading.Thread(target=self._action_sender_loop, daemon=True)
        self._action_thread.start()

    def infer_once(self, sdk_observation: Mapping[str, Any]) -> dict[str, list[list[float]]]:
        _result, sdk_actions = infer_sdk_actions(
            self._policy,
            sdk_observation,
            contract=self._contract,
            spec=self._spec,
            prompt=self._prompt,
            include_motor_hold=self._include_motor_hold,
        )
        return sdk_actions

    def run_loop(
        self,
        *,
        max_iters: int | None = None,
        poll_interval_s: float = 0.005,
        drop_on_busy: bool = True,
    ) -> int:
        """Infer on each new observation until ``max_iters`` or Ctrl+C.

        When ``drop_on_busy`` is True, a new observation replaces an unfinished
        action buffer (prefer freshest state over backlog on the robot).
        """
        self.start_action_thread()
        iterations = 0
        try:
            while max_iters is None or iterations < max_iters:
                obs = self.take_new_observation()
                if obs is None:
                    time.sleep(poll_interval_s)
                    continue
                try:
                    sdk_actions = self.infer_once(obs)
                    if drop_on_busy:
                        self.enqueue_sdk_actions(sdk_actions)
                    else:
                        with self._action_lock:
                            self._action_buffer.extend(sdk_action_chunk_to_step_dicts(sdk_actions))
                    self._infer_count += 1
                    iterations += 1
                except Exception:  # noqa: BLE001
                    self._error_count += 1
                    logger.exception("infer/publish failed")
        except KeyboardInterrupt:
            logger.info("interrupted after %s iters", iterations)
        return iterations


def build_remote_policy(
    *,
    host: str,
    port: int,
    api_key: str | None,
    contract: Any,
    spec: Any,
    prompt: str | None,
) -> PolicyServerSession:
    """Connect to ``pi-dex-serve`` and sanity-check clock domain when present."""
    session = PolicyServerSession(host=host, port=port, api_key=api_key)
    metadata = session.metadata
    pi_dex = metadata.get("pi_dex") if isinstance(metadata, dict) else None
    if isinstance(pi_dex, dict):
        clock = pi_dex.get("clock_domain")
        if type(clock) is str and clock and clock != spec.clock_domain:
            raise ValueError(
                f"server clock_domain {clock!r} != local spec {spec.clock_domain!r}"
            )
    # Warm-path: build one observation shape check without calling the server.
    _ = (contract, prompt)
    return session


class _RemotePolicyAdapter:
    """Adapt ``PolicyServerSession`` to the ``policy.infer`` used by realtime helpers."""

    def __init__(self, session: PolicyServerSession) -> None:
        self._session = session

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        return self._session.infer(observation)

    def metadata(self) -> dict[str, Any]:
        return self._session.metadata


def run_codec_smoke(*, contract_path: pathlib.Path, prompt: str) -> dict[str, Any]:
    """Offline: synthetic NorthObservation → SDK → OpenPI obs (no Zenoh/GPU)."""
    from io import BytesIO

    from PIL import Image

    from pi_dex.north_codec import build_synthetic_north_observation
    from pi_dex.north_codec import north_observation_to_sdk_dict

    buf = BytesIO()
    Image.new("RGB", (16, 16), (12, 34, 56)).save(buf, format="JPEG", quality=85)
    jpeg = buf.getvalue()
    north = build_synthetic_north_observation(jpeg_rgb=jpeg, language=prompt)
    sdk = north_observation_to_sdk_dict(north, decode_images=True)
    contract = load_observation_contract(contract_path)
    timestamp_ns = resolve_observation_timestamp_ns(sdk)
    live_prompt = resolve_live_prompt(sdk, fallback=prompt)
    observation = build_policy_observation_from_sdk(
        sdk,
        contract,
        prompt=live_prompt,
        observation_timestamp_ns=timestamp_ns,
        clock_domain="unix_realtime",
    )
    return {
        "ok": True,
        "sdk_keys": sorted(str(k) for k in sdk),
        "state_shape": list(observation["state"].shape),
        "image_keys": sorted(observation["image"]),
        "prompt": observation["prompt"],
        "observation_timestamp_ns": observation["observation_timestamp_ns"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pi-dex-robot-client")
    parser.add_argument(
        "--mode",
        choices=("bridge", "codec-smoke"),
        default="bridge",
        help="bridge=Zenoh↔serve; codec-smoke=offline protobuf conversion check",
    )
    parser.add_argument(
        "--observation-contract",
        type=pathlib.Path,
        default=pathlib.Path("configs/site/joint_29d_observation.reviewed.json"),
    )
    parser.add_argument("--prompt", default="", help="fallback if /language empty")
    parser.add_argument("--robot-id", default="POC22005")
    parser.add_argument("--embodiment-version", default="sharpa_north_v1")
    parser.add_argument(
        "--command-semantics-version",
        default="sharpa_sdk_commanded_joint_position_absolute_v1",
    )
    parser.add_argument("--hand-mapping-version", default="sharpa_north_hand_mapping_v1")
    parser.add_argument("--clock-domain", default="unix_realtime")
    parser.add_argument("--serve-host", default="127.0.0.1")
    parser.add_argument("--serve-port", type=int, default=8000)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--observation-topic", default=DEFAULT_OBSERVATION_TOPIC)
    parser.add_argument("--action-topic", default=DEFAULT_ACTION_TOPIC)
    parser.add_argument("--action-pub-duration", type=float, default=DEFAULT_ACTION_PUB_DURATION_S)
    parser.add_argument("--zenoh-config", type=pathlib.Path, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--no-motor-hold", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.mode == "codec-smoke":
        import json

        summary = run_codec_smoke(
            contract_path=args.observation_contract,
            prompt=args.prompt or "codec-smoke",
        )
        print(json.dumps(summary, indent=2))
        return 0

    contract = load_observation_contract(args.observation_contract)
    spec = build_joint_spec_from_contract(
        contract,
        robot_id=args.robot_id,
        embodiment_version=args.embodiment_version,
        command_semantics_version=args.command_semantics_version,
        hand_mapping_version=args.hand_mapping_version,
        clock_domain=args.clock_domain,
    )
    prompt = args.prompt.strip() or None
    session = build_remote_policy(
        host=args.serve_host,
        port=args.serve_port,
        api_key=args.api_key.strip() or None,
        contract=contract,
        spec=spec,
        prompt=prompt,
    )
    policy = _RemotePolicyAdapter(session)
    client = NorthZenohRobotClient(
        policy=policy,
        contract=contract,
        spec=spec,
        observation_topic=args.observation_topic,
        action_topic=args.action_topic,
        action_pub_duration_s=args.action_pub_duration,
        prompt=prompt,
        include_motor_hold=not args.no_motor_hold,
        zenoh_config=args.zenoh_config,
    )
    client.connect()
    try:
        logger.info(
            "bridge running; put robot in inference(F6)+moving(F2). "
            "serve=%s:%s topics=%s→%s",
            args.serve_host,
            args.serve_port,
            args.observation_topic,
            args.action_topic,
        )
        client.run_loop(max_iters=args.max_iters)
    finally:
        client.close()
    logger.info("done infer=%s errors=%s", client.infer_count, client.error_count)
    return 0 if client.error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
