"""Real-robot joint_29d inference entry for Sharpa North + PI-DEX checkpoints.

This module converts live SDK observation dicts into OpenPI observations, runs
:class:`~pi_dex.deployment.BimanualPolicyAdapter`, and emits NorthDirect-compatible
action dicts. It does **not** implement ``BimanualController`` lease / atomic
apply / e-stop semantics; wire those through ``pi_dex.deployment`` before
unattended hardware runs.

Reference SDK: ``examples/sharpa_north_sdk.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

import numpy as np

from pi_dex.observation_contract import SharpaObservationContract
from pi_dex.observation_contract import load_observation_contract
from pi_dex.joint_action_mode import maybe_compose_deployment_joint_actions
from pi_dex.realtime_actions import policy_result_to_sdk_action_dict
from pi_dex.realtime_actions import sdk_action_pub_keys
from pi_dex.realtime_observation import build_policy_observation_from_sdk
from pi_dex.realtime_observation import resolve_live_prompt
from pi_dex.realtime_observation import resolve_observation_timestamp_ns
from pi_dex.spec import ActionMode
from pi_dex.spec import BimanualActionSpec
from pi_dex.training_runner import build_joint_spec_from_contract


def load_joint29d_policy(
    *,
    checkpoint_dir: pathlib.Path | str,
    spec: BimanualActionSpec,
    asset_id: str = "sharpa_joint_29d",
    assets_dirs: pathlib.Path | str | None = None,
    pytorch_device: str | None = None,
    execution_horizon: int | None = None,
    default_prompt: str | None = None,
    sample_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Load a training checkpoint as a :class:`~pi_dex.deployment.BimanualPolicyAdapter`."""
    from openpi.training import checkpoints as openpi_checkpoints
    from openpi.training import config as openpi_config
    from openpi import transforms as openpi_transforms
    from pi_dex.openpi_integration import BimanualDataConfigFactory
    from pi_dex.openpi_integration import create_bimanual_trained_policy
    from pi_dex.openpi_integration import create_pi05_model_config

    checkpoint = pathlib.Path(checkpoint_dir)
    if assets_dirs is not None:
        assets_root = pathlib.Path(assets_dirs)
    elif (checkpoint / "assets" / asset_id / "norm_stats.json").is_file():
        assets_root = checkpoint / "assets"
    else:
        raise FileNotFoundError(
            f"assets for {asset_id!r} not found under checkpoint/assets; pass --assets-dir"
        )

    model_config = create_pi05_model_config(spec, dtype="bfloat16", pytorch_compile_mode=None)
    empty = openpi_transforms.Group()
    norm_stats = openpi_checkpoints.load_norm_stats(assets_root, asset_id)

    @dataclasses.dataclass(frozen=True)
    class _PinnedAssetsFactory:
        assets: openpi_config.AssetsConfig

        def create(self, _assets_dirs: pathlib.Path, model_cfg: Any) -> Any:
            return openpi_config.DataConfig(
                repo_id=f"pi-dex/{asset_id}",
                asset_id=asset_id,
                norm_stats=norm_stats,
                use_quantile_norm=True,
                repack_transforms=empty,
                data_transforms=empty,
                model_transforms=openpi_config.ModelTransformFactory()(model_cfg),
            )

    train_config = openpi_config.TrainConfig(
        name=assets_root.name,
        exp_name="pi_dex_realtime",
        model=model_config,
        data=BimanualDataConfigFactory(
            _PinnedAssetsFactory(assets=openpi_config.AssetsConfig(asset_id=asset_id)),
            spec,
        ),
        assets_base_dir=str(assets_root.parent),
        wandb_enabled=False,
    )
    return create_bimanual_trained_policy(
        train_config,
        checkpoint,
        spec,
        execution_horizon=execution_horizon,
        default_prompt=default_prompt,
        pytorch_device=pytorch_device,
        sample_kwargs=sample_kwargs,
    )


def infer_sdk_actions(
    policy: Any,
    sdk_observation: Mapping[str, Any],
    *,
    contract: SharpaObservationContract,
    spec: BimanualActionSpec,
    prompt: str | None = None,
    clock_domain: str | None = None,
    include_motor_hold: bool = False,
) -> tuple[dict[str, Any], dict[str, list[list[float]]]]:
    """Run one policy step and return ``(policy_result, sdk_action_dict)``."""
    resolved_prompt = resolve_live_prompt(sdk_observation, fallback=prompt)
    timestamp_ns = resolve_observation_timestamp_ns(sdk_observation)
    domain = clock_domain or spec.clock_domain
    observation = build_policy_observation_from_sdk(
        sdk_observation,
        contract,
        prompt=resolved_prompt,
        observation_timestamp_ns=timestamp_ns,
        clock_domain=domain,
    )
    result = policy.infer(observation)
    actions = result["actions"]
    left = np.asarray(actions["left"], dtype=np.float32)
    right = np.asarray(actions["right"], dtype=np.float32)
    left, right = maybe_compose_deployment_joint_actions(
        left,
        right,
        observation["state"],
        spec=spec,
    )
    composed_result = dict(result)
    composed_result["actions"] = {"left": left, "right": right}
    motor = None
    if include_motor_hold and "/state/motor/joint_angle" in sdk_observation:
        motor = np.asarray(sdk_observation["/state/motor/joint_angle"], dtype=np.float32)
    sdk_actions = policy_result_to_sdk_action_dict(composed_result, spec, motor_positions=motor)
    return composed_result, sdk_actions


def run_control_loop(
    *,
    policy: Any,
    contract: SharpaObservationContract,
    spec: BimanualActionSpec,
    get_observation: Callable[[], Mapping[str, Any] | None],
    publish_actions: Callable[[dict[str, list[list[float]]]], None],
    prompt: str | None = None,
    max_iters: int | None = None,
    poll_interval_s: float = 0.01,
    include_motor_hold: bool = False,
) -> int:
    """Poll observations, infer, and publish until ``max_iters`` or Ctrl+C.

    ``get_observation`` should return the latest SDK dict or ``None`` when no new
    frame is available. This loop is a research entrypoint; production hardware
    must add lease / hold / e-stop via ``deployment.BimanualController``.
    """
    iterations = 0
    try:
        while max_iters is None or iterations < max_iters:
            sdk_observation = get_observation()
            if sdk_observation is None:
                time.sleep(poll_interval_s)
                continue
            _result, sdk_actions = infer_sdk_actions(
                policy,
                sdk_observation,
                contract=contract,
                spec=spec,
                prompt=prompt,
                include_motor_hold=include_motor_hold,
            )
            publish_actions(sdk_actions)
            iterations += 1
    except KeyboardInterrupt:
        return iterations
    return iterations


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pi-dex-realtime-infer")
    parser.add_argument(
        "--mode",
        choices=("convert-smoke", "infer-once", "print-action-keys"),
        required=True,
    )
    parser.add_argument("--observation-contract", required=True)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--assets-dir", default="")
    parser.add_argument("--asset-id", default="sharpa_joint_29d")
    parser.add_argument("--sdk-observation-json", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--pytorch-device", default=None)
    parser.add_argument("--execution-horizon", type=int, default=None)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--robot-id", default="POC22027")
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
    parser.add_argument("--allow-unreviewed-contract", action="store_true")
    parser.add_argument("--include-motor-hold", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    contract = load_observation_contract(args.observation_contract)
    if not args.allow_unreviewed_contract:
        contract.require_reviewed_for_training()
    spec = build_joint_spec_from_contract(
        contract,
        robot_id=args.robot_id,
        embodiment_version=args.embodiment_version,
        command_semantics_version=args.command_semantics_version,
        hand_mapping_version=args.hand_mapping_version,
        clock_domain=args.clock_domain,
        action_mode=ActionMode(args.action_mode),
    )

    if args.mode == "print-action-keys":
        payload = {"action_pub_keys": list(sdk_action_pub_keys())}
        _write_json(args.output_json, payload)
        print(json.dumps(payload, indent=2))
        return 0

    if args.mode == "convert-smoke":
        sdk_observation = _load_or_synthesize_sdk_observation(args, contract)
        prompt = args.prompt or "realtime convert smoke"
        observation = build_policy_observation_from_sdk(
            sdk_observation,
            contract,
            prompt=prompt,
            observation_timestamp_ns=resolve_observation_timestamp_ns(sdk_observation),
            clock_domain=args.clock_domain,
        )
        execution_horizon = args.execution_horizon or contract.physical_horizon
        fake_result = {
            "actions": {
                "left": np.zeros((execution_horizon, 29), dtype=np.float32),
                "right": np.zeros((execution_horizon, 29), dtype=np.float32),
            }
        }
        sdk_actions = policy_result_to_sdk_action_dict(fake_result, spec)
        payload = {
            "mode": "convert-smoke",
            "state_shape": list(observation["state"].shape),
            "image_keys": sorted(observation["image"]),
            "sdk_action_keys": sorted(sdk_actions),
            "execution_horizon": execution_horizon,
        }
        _write_json(args.output_json, payload)
        print(json.dumps(payload, indent=2))
        return 0

    if args.mode == "infer-once":
        if not args.checkpoint_dir:
            raise ValueError("infer-once requires --checkpoint-dir")
        sdk_observation = _load_or_synthesize_sdk_observation(args, contract)
        policy = load_joint29d_policy(
            checkpoint_dir=args.checkpoint_dir,
            spec=spec,
            asset_id=args.asset_id,
            assets_dirs=args.assets_dir or None,
            pytorch_device=args.pytorch_device,
            execution_horizon=args.execution_horizon,
            default_prompt=args.prompt or None,
        )
        result, sdk_actions = infer_sdk_actions(
            policy,
            sdk_observation,
            contract=contract,
            spec=spec,
            prompt=args.prompt or None,
            clock_domain=args.clock_domain,
            include_motor_hold=args.include_motor_hold,
        )
        payload = {
            "mode": "infer-once",
            "chunk_sequence_id": result.get("chunk_sequence_id"),
            "source_timestamp_ns": result.get("source_timestamp_ns"),
            "left_shape": list(np.asarray(result["actions"]["left"]).shape),
            "right_shape": list(np.asarray(result["actions"]["right"]).shape),
            "sdk_action_keys": sorted(sdk_actions),
            "sdk_action_horizon": len(next(iter(sdk_actions.values()))),
        }
        _write_json(args.output_json, payload)
        print(json.dumps(payload, indent=2))
        return 0

    raise ValueError(f"unsupported mode {args.mode!r}")


def _load_or_synthesize_sdk_observation(
    args: argparse.Namespace,
    contract: SharpaObservationContract,
) -> dict[str, Any]:
    if args.sdk_observation_json:
        payload = json.loads(pathlib.Path(args.sdk_observation_json).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("sdk-observation-json: expected a JSON object")
        return _decode_sdk_json(payload)
    return synthesize_sdk_observation(contract)


def synthesize_sdk_observation(
    contract: SharpaObservationContract,
    *,
    height: int = 64,
    width: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Build a synthetic NorthDirect-shaped observation for dry runs."""
    rng = np.random.default_rng(seed)
    obs: dict[str, Any] = {
        "timestamp": int(time.time() * 1_000_000_000),
        "/language": "synthetic realtime prompt",
    }
    for column in contract.state_columns:
        values = rng.normal(size=(max(column.slice_stop, column.slice_stop - column.slice_start),)).astype(
            np.float32
        )
        obs[f"/{column.source_path}"] = values
    for slot in contract.image_slots:
        if slot.sharpa_group is None:
            continue
        key = f"/{slot.sharpa_group}/rgb"
        obs[key] = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return obs


def _decode_sdk_json(payload: dict[str, Any]) -> dict[str, Any]:
    """Decode JSON where image arrays may be nested lists."""
    decoded: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(key, str) and "/observe/vision/" in key and isinstance(value, list):
            decoded[key] = np.asarray(value, dtype=np.uint8)
        elif isinstance(key, str) and key.startswith("/state/") and isinstance(value, list):
            decoded[key] = np.asarray(value, dtype=np.float32)
        else:
            decoded[key] = value
    return decoded


def _write_json(path: str, payload: Mapping[str, Any]) -> None:
    if not path:
        return
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
