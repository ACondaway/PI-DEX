"""CLI: OpenPI Runtime + ActionChunkBroker + harobotsDL NorthZmqEnv."""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
from collections.abc import Sequence
from typing import Any

from openpi_client import action_chunk_broker
from openpi_client.runtime import runtime as _runtime
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from pi_dex.data.observation_contract import load_observation_contract
from pi_dex.robot.agent import GatedPolicyAgent
from pi_dex.robot.environment import NorthRealEnvironment
from pi_dex.robot.north_env import NorthZmqEnv
from pi_dex.robot.remote_policy import WebsocketJointPolicy
from pi_dex.robot.sharpa_runtime_keys import DEFAULT_ACTION_PUB_DURATION_S
from pi_dex.robot.sharpa_runtime_keys import DEFAULT_ACTION_TOPIC
from pi_dex.robot.sharpa_runtime_keys import DEFAULT_OBSERVATION_TOPIC
from pi_dex.core.spec import ActionMode
from pi_dex.training.training_runner import build_joint_spec_from_contract

logger = logging.getLogger(__name__)


def run_codec_smoke(*, contract_path: pathlib.Path, prompt: str) -> dict[str, Any]:
    from io import BytesIO

    from PIL import Image

    from pi_dex.robot.north_codec import build_synthetic_north_observation
    from pi_dex.robot.north_codec import north_observation_to_sdk_dict
    from pi_dex.robot.realtime_observation import build_policy_observation_from_sdk
    from pi_dex.robot.realtime_observation import resolve_live_prompt
    from pi_dex.robot.realtime_observation import resolve_observation_timestamp_ns

    buf = BytesIO()
    Image.new("RGB", (16, 16), (12, 34, 56)).save(buf, format="JPEG", quality=85)
    north = build_synthetic_north_observation(jpeg_rgb=buf.getvalue(), language=prompt)
    sdk = north_observation_to_sdk_dict(north, decode_images=True)
    contract = load_observation_contract(contract_path)
    observation = build_policy_observation_from_sdk(
        sdk,
        contract,
        prompt=resolve_live_prompt(sdk, fallback=prompt),
        observation_timestamp_ns=resolve_observation_timestamp_ns(sdk),
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


def _validate_server_metadata(metadata: dict[str, Any], *, spec: Any) -> None:
    pi_dex = metadata.get("pi_dex") if isinstance(metadata, dict) else None
    if not isinstance(pi_dex, dict):
        return
    clock = pi_dex.get("clock_domain")
    if type(clock) is str and clock and clock != spec.clock_domain:
        raise ValueError(f"server clock_domain {clock!r} != local {spec.clock_domain!r}")
    action_mode = pi_dex.get("action_mode")
    if type(action_mode) is str and action_mode and action_mode != spec.action_mode.value:
        raise ValueError(
            f"server action_mode {action_mode!r} != local {spec.action_mode.value!r}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pi-dex-robot-client",
        description="OpenPI Runtime + ActionChunkBroker over harobotsDL NorthZmqEnv",
    )
    parser.add_argument("--mode", choices=("bridge", "codec-smoke"), default="bridge")
    parser.add_argument(
        "--observation-contract",
        type=pathlib.Path,
        default=pathlib.Path("configs/site/joint_29d_observation.reviewed.json"),
    )
    parser.add_argument("--prompt", default="")
    parser.add_argument("--robot-id", default="POC22005")
    parser.add_argument("--embodiment-version", default="sharpa_north_v1")
    parser.add_argument(
        "--command-semantics-version",
        default="sharpa_sdk_commanded_joint_position_absolute_v1",
    )
    parser.add_argument("--hand-mapping-version", default="sharpa_north_hand_mapping_v1")
    parser.add_argument("--clock-domain", default="unix_realtime")
    parser.add_argument(
        "--action-mode",
        choices=(ActionMode.ABSOLUTE.value, ActionMode.DELTA.value),
        default=ActionMode.ABSOLUTE.value,
    )
    parser.add_argument("--serve-host", default="127.0.0.1")
    parser.add_argument("--serve-port", type=int, default=8000)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--observation-topic", default=DEFAULT_OBSERVATION_TOPIC)
    parser.add_argument("--action-topic", default=DEFAULT_ACTION_TOPIC)
    parser.add_argument("--action-pub-duration", type=float, default=DEFAULT_ACTION_PUB_DURATION_S)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--output-chunk",
        type=int,
        default=None,
        help="OpenPI action_horizon / harobots output_chunk (default=physical_horizon-offset)",
    )
    parser.add_argument(
        "--first-chunk-smooth",
        type=int,
        default=0,
        help="harobotsDL first-chunk smooth size on the inferred chunk (0=off)",
    )
    parser.add_argument("--zenoh-config", type=pathlib.Path, default=None)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=0,
        help="OpenPI Runtime max steps per episode (0=unlimited)",
    )
    parser.add_argument(
        "--motor-hold",
        action="store_true",
        help="Hold motor at last observed state instead of publishing predicted motor",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.mode == "codec-smoke":
        print(
            json.dumps(
                run_codec_smoke(
                    contract_path=args.observation_contract,
                    prompt=args.prompt or "codec-smoke",
                ),
                indent=2,
            )
        )
        return 0

    contract = load_observation_contract(args.observation_contract)
    spec = build_joint_spec_from_contract(
        contract,
        robot_id=args.robot_id,
        embodiment_version=args.embodiment_version,
        command_semantics_version=args.command_semantics_version,
        hand_mapping_version=args.hand_mapping_version,
        clock_domain=args.clock_domain,
        action_mode=ActionMode(args.action_mode),
    )
    prompt = args.prompt.strip() or None

    ws = WebsocketClientPolicy(
        host=args.serve_host,
        port=args.serve_port,
        api_key=args.api_key.strip() or None,
    )
    _validate_server_metadata(ws.get_server_metadata(), spec=spec)

    hardware = NorthZmqEnv(
        observation_topic=args.observation_topic,
        action_topic=args.action_topic,
        action_pub_duration=args.action_pub_duration,
        zenoh_config=args.zenoh_config,
        first_chunk_smooth_size=0,  # smooth applied on full chunk in WebsocketJointPolicy
        language=prompt,
    )
    hardware.connect()
    env = NorthRealEnvironment(
        hardware,
        contract=contract,
        spec=spec,
        prompt=prompt,
        include_motor_hold=args.motor_hold,
    )
    remote = WebsocketJointPolicy(
        ws,
        spec=spec,
        contract=contract,
        offset=args.offset,
        output_chunk=args.output_chunk,
        first_chunk_smooth_size=args.first_chunk_smooth,
        include_motor_hold=args.motor_hold,
        get_sdk_observation=lambda: env.last_sdk_observation,
    )
    broker = action_chunk_broker.ActionChunkBroker(
        policy=remote,
        action_horizon=remote.output_chunk,
    )
    agent = GatedPolicyAgent(
        broker,
        spec=spec,
        on_standby=hardware.clear_action_and_history,
    )
    max_hz = 1.0 / args.action_pub_duration
    runtime = _runtime.Runtime(
        environment=env,
        agent=agent,
        subscribers=[],
        max_hz=max_hz,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    logger.info(
        "OpenPI Runtime + harobotsDL NorthZmqEnv: serve=%s:%s max_hz=%.2f "
        "offset=%s output_chunk=%s action_mode=%s topics=%s→%s",
        args.serve_host,
        args.serve_port,
        max_hz,
        args.offset,
        remote.output_chunk,
        args.action_mode,
        args.observation_topic,
        args.action_topic,
    )
    try:
        runtime.run()
    except KeyboardInterrupt:
        logger.info("interrupted")
    finally:
        hardware.disconnect()
    return 0 if hardware.error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
