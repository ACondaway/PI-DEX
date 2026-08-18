"""Tests for DDP helpers, Volcano launch, and realtime SDK adapters."""

from __future__ import annotations

import json
import pathlib
import types

import numpy as np
import pytest

from pi_dex.distributed import is_main_process
from pi_dex.distributed import launched_under_torch_distributed
from pi_dex.distributed import resolve_rank_world_local
from pi_dex.distributed import unwrap_model
from pi_dex.observation_contract import load_observation_contract
from pi_dex.realtime_actions import JOINT_29D_DIM
from pi_dex.realtime_actions import policy_result_to_sdk_action_dict
from pi_dex.realtime_actions import split_joint_29d_hand_chunk
from pi_dex.realtime_inference import synthesize_sdk_observation
from pi_dex.realtime_observation import build_policy_observation_from_sdk
from pi_dex.realtime_observation import hdf5_path_to_sdk_key
from pi_dex.training_runner import _distributed_sampler_num_samples
from pi_dex.training_runner import build_joint_spec_from_contract
from pi_dex.volc_launch import build_torchrun_command
from pi_dex.volc_launch import read_mlp_launch_env


ROOT = pathlib.Path(__file__).resolve().parents[1]
REVIEWED_CONTRACT = ROOT / "configs/site/joint_29d_observation.reviewed.json"


def test_resolve_rank_world_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert resolve_rank_world_local() == (1, 4, 1)
    assert launched_under_torch_distributed()
    assert not is_main_process()


def test_unwrap_model_returns_inner_module() -> None:
    inner = object()
    wrapped = types.SimpleNamespace(module=inner)
    assert unwrap_model(wrapped) is inner
    assert unwrap_model(inner) is inner


def test_distributed_sampler_num_samples_matches_drop_last() -> None:
    assert _distributed_sampler_num_samples(10, world_size=2) == 5
    assert _distributed_sampler_num_samples(11, world_size=2) == 5
    assert _distributed_sampler_num_samples(7, world_size=2) == 3


def test_read_mlp_launch_env_and_torchrun_command() -> None:
    env = {
        "MLP_WORKER_NUM": "2",
        "MLP_WORKER_GPU": "8",
        "MLP_ROLE_INDEX": "1",
        "MLP_WORKER_0_HOST": "10.0.0.1",
        "MLP_WORKER_0_PORT": "29500",
    }
    mlp = read_mlp_launch_env(env)
    assert mlp["nnodes"] == "2"
    assert mlp["node_rank"] == "1"
    command = build_torchrun_command(
        training_argv=["pi-dex-train-pytorch", "--help"],
        mlp=mlp,
        python_executable="python",
    )
    joined = " ".join(command)
    assert "--nnodes=2" in joined
    assert "--nproc_per_node=8" in joined
    assert "--node_rank=1" in joined
    assert "--master_addr=10.0.0.1" in joined


def test_build_torchrun_prefers_conda_prefix_torchrun(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    fake_torchrun = tmp_path / "bin" / "torchrun"
    fake_torchrun.parent.mkdir(parents=True)
    fake_torchrun.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_torchrun.chmod(0o755)
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))
    mlp = {
        "nnodes": "1",
        "nproc_per_node": "8",
        "node_rank": "0",
        "master_addr": "127.0.0.1",
        "master_port": "29500",
    }
    command = build_torchrun_command(
        training_argv=["pi-dex-train-pytorch", "--help"],
        mlp=mlp,
        python_executable="python",
    )
    assert command[0] == str(fake_torchrun)
    assert "--nproc_per_node=8" in command


def test_read_mlp_launch_env_rejects_missing() -> None:
    with pytest.raises(ValueError, match="missing"):
        read_mlp_launch_env({})


def test_sdk_observation_roundtrip_shapes() -> None:
    contract = load_observation_contract(REVIEWED_CONTRACT)
    sdk = synthesize_sdk_observation(contract, height=32, width=32, seed=1)
    observation = build_policy_observation_from_sdk(
        sdk,
        contract,
        prompt="clear the plate",
        observation_timestamp_ns=1,
        clock_domain="unix_realtime",
    )
    assert observation["state"].shape == (contract.state_dim,)
    assert set(observation["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert observation["image"]["base_0_rgb"].shape == (32, 32, 3)
    assert observation["observation_timestamp_ns"] == 1


def test_hdf5_to_sdk_key() -> None:
    assert hdf5_path_to_sdk_key("state/left_arm/joint_angle") == "/state/left_arm/joint_angle"
    with pytest.raises(ValueError):
        hdf5_path_to_sdk_key("/state/left_arm/joint_angle")


def test_policy_result_to_sdk_action_dict() -> None:
    contract = load_observation_contract(REVIEWED_CONTRACT)
    spec = build_joint_spec_from_contract(
        contract,
        robot_id="POC22027",
        embodiment_version="sharpa_north_v1",
        command_semantics_version="sharpa_sdk_commanded_joint_position_absolute_v1",
        hand_mapping_version="sharpa_north_hand_mapping_v1",
        clock_domain="unix_realtime",
    )
    left = np.arange(2 * JOINT_29D_DIM, dtype=np.float32).reshape(2, JOINT_29D_DIM)
    right = left + 100
    result = {"actions": {"left": left, "right": right}}
    sdk = policy_result_to_sdk_action_dict(result, spec)
    assert len(sdk["/action/left_arm/joint_angle"]) == 2
    assert len(sdk["/action/left_arm/joint_angle"][0]) == 7
    assert len(sdk["/action/left_hand/joint_angle"][0]) == 22
    arm, hand = split_joint_29d_hand_chunk(left)
    assert arm.shape == (2, 7)
    assert hand.shape == (2, 22)


def test_realtime_convert_smoke_cli(tmp_path: pathlib.Path) -> None:
    from pi_dex.realtime_inference import main

    out = tmp_path / "out.json"
    code = main(
        [
            "--mode",
            "convert-smoke",
            "--observation-contract",
            str(REVIEWED_CONTRACT),
            "--output-json",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["mode"] == "convert-smoke"
    assert payload["state_shape"] == [65]


def test_volc_ensure_distributed_flag() -> None:
    from pi_dex.volc_launch import _ensure_runner_distributed_flag

    argv = [
        "pi-dex-train-pytorch",
        "--action-representation",
        "joint_29d",
        "--runner",
        "pi_dex.training_runner:run",
        "--",
        "--mode",
        "train",
    ]
    updated = _ensure_runner_distributed_flag(argv)
    assert updated[-1] == "--distributed"
    assert "--mode" in updated
