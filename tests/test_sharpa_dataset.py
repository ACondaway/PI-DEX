"""Tests for Sharpa joint_29d dataset and training runner smoke paths."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pi_dex.actions import ActionRepresentation
from pi_dex.observation_contract import load_observation_contract
from pi_dex.sharpa_dataset import SyntheticJoint29dDataset
from pi_dex.training_launcher import PytorchTrainingLaunchContext
from pi_dex.training_runner import build_joint_spec_from_contract
from pi_dex.training_runner import run

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = REPO_ROOT / "configs/site/joint_29d_observation.unreviewed.json"
REAL_EPISODE_ROOT = Path(
    "/mnt/netdata/Team/Academic/Data/North/SharpaOpenData/ClearPlate/season_POC22027_2026_04_18_10_54_31_train"
)


def test_synthetic_joint_dataset_shapes() -> None:
    contract = load_observation_contract(CONTRACT)
    dataset = SyntheticJoint29dDataset(
        physical_horizon=contract.physical_horizon,
        state_dim=contract.state_dim,
        length=3,
    )
    sample = dataset[1]
    assert sample["state"].shape == (contract.state_dim,)
    assert sample["left_actions"].shape == (contract.physical_horizon, 29)
    assert sample["right_actions"].shape == (contract.physical_horizon, 29)
    assert sample["image"]["base_0_rgb"].dtype == np.uint8


def test_runner_synthetic_smoke(tmp_path: Path) -> None:
    output = tmp_path / "smoke.json"
    context = PytorchTrainingLaunchContext(
        action_representation=ActionRepresentation.JOINT_29D,
        runner_args=(
            "--mode",
            "synthetic-smoke",
            "--observation-contract",
            str(CONTRACT),
            "--output-json",
            str(output),
            "--max-samples",
            "4",
        ),
    )
    assert run(context) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["mode"] == "synthetic-smoke"
    assert payload["logical_action_dim"] == 29


@pytest.mark.skipif(not REAL_EPISODE_ROOT.is_dir(), reason="SharpaOpenData not mounted")
def test_runner_validate_one_real_season(tmp_path: Path) -> None:
    output = tmp_path / "validate.json"
    context = PytorchTrainingLaunchContext(
        action_representation=ActionRepresentation.JOINT_29D,
        runner_args=(
            "--mode",
            "validate-data",
            "--observation-contract",
            str(CONTRACT),
            "--dataset-root",
            str(REAL_EPISODE_ROOT),
            "--max-episodes",
            "1",
            "--max-samples",
            "2",
            "--allow-unreviewed-contract",
            "--output-json",
            str(output),
        ),
    )
    assert run(context) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["sample0"]["left_actions_shape"] == [8, 29]
    assert payload["sample0"]["state_shape"] == [65]


def test_build_joint_spec_matches_contract() -> None:
    contract = load_observation_contract(CONTRACT)
    spec = build_joint_spec_from_contract(
        contract,
        robot_id="POC22027",
        embodiment_version="sharpa_north_v1",
        command_semantics_version="sharpa_sdk_commanded_joint_position_absolute_v1",
        hand_mapping_version="sharpa_north_hand_mapping_v1",
        clock_domain="unix_realtime",
    )
    contract.validate_against_action_spec(spec)
    assert spec.requires_forward_kinematics is False
