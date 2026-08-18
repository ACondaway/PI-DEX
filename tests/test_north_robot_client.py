"""Tests for North protobuf ↔ SDK codec used by the robot Zenoh bridge."""

from __future__ import annotations

import io
import pathlib

import numpy as np
import pytest
from PIL import Image

from pi_dex.north_codec import build_synthetic_north_observation
from pi_dex.north_codec import encode_uhr_action_bundle
from pi_dex.north_codec import north_observation_to_sdk_dict
from pi_dex.north_codec import parse_north_observation
from pi_dex.north_codec import sdk_action_chunk_to_step_dicts
from pi_dex.north_codec import sdk_step_action_to_uhr_bundle
from pi_dex.observation_contract import load_observation_contract
from pi_dex.realtime_actions import JOINT_29D_DIM
from pi_dex.realtime_actions import policy_result_to_sdk_action_dict
from pi_dex.realtime_observation import build_policy_observation_from_sdk
from pi_dex.realtime_observation import resolve_live_prompt
from pi_dex.realtime_observation import resolve_observation_timestamp_ns
from pi_dex.robot_client import main as robot_client_main
from pi_dex.training_runner import build_joint_spec_from_contract


ROOT = pathlib.Path(__file__).resolve().parents[1]
REVIEWED_CONTRACT = ROOT / "configs/site/joint_29d_observation.reviewed.json"


def _tiny_jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (24, 16), (1, 2, 3)).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def test_north_observation_roundtrip_to_policy_obs() -> None:
    jpeg = _tiny_jpeg()
    north = build_synthetic_north_observation(jpeg_rgb=jpeg, language="insert battery", timestamp_s=2.5)
    payload = north.SerializeToString()
    sdk = parse_north_observation(payload, decode_images=True)
    assert "/state/left_arm/joint_angle" in sdk
    assert len(sdk["/state/left_arm/joint_angle"]) == 7
    assert len(sdk["/state/left_hand/joint_angle"]) == 22
    assert sdk["/language"] == "insert battery"
    assert sdk["/observe/vision/head/stereo/lefteye/rgb"].dtype == np.uint8
    assert sdk["/observe/vision/head/stereo/lefteye/rgb"].shape[2] == 3

    contract = load_observation_contract(REVIEWED_CONTRACT)
    observation = build_policy_observation_from_sdk(
        sdk,
        contract,
        prompt=resolve_live_prompt(sdk, fallback="unused"),
        observation_timestamp_ns=resolve_observation_timestamp_ns(sdk),
        clock_domain="unix_realtime",
    )
    assert observation["state"].shape == (contract.state_dim,)
    assert observation["prompt"] == "insert battery"


def test_sdk_action_chunk_to_uhr_bundle() -> None:
    contract = load_observation_contract(REVIEWED_CONTRACT)
    spec = build_joint_spec_from_contract(
        contract,
        robot_id="POC22005",
        embodiment_version="sharpa_north_v1",
        command_semantics_version="sharpa_sdk_commanded_joint_position_absolute_v1",
        hand_mapping_version="sharpa_north_hand_mapping_v1",
        clock_domain="unix_realtime",
    )
    left = np.arange(2 * JOINT_29D_DIM, dtype=np.float32).reshape(2, JOINT_29D_DIM)
    right = left + 10
    sdk = policy_result_to_sdk_action_dict({"actions": {"left": left, "right": right}}, spec)
    steps = sdk_action_chunk_to_step_dicts(sdk)
    assert len(steps) == 2
    assert "left_arm" in steps[0] and "left_hand" in steps[0]
    bundle = sdk_step_action_to_uhr_bundle(steps[0], language="hi")
    raw = encode_uhr_action_bundle(bundle)
    assert isinstance(raw, (bytes, bytearray)) and len(raw) > 0
    assert list(bundle.left_arm.joint.position) == pytest.approx(list(left[0, :7]))
    assert list(bundle.left_glove.joint.position) == pytest.approx(list(left[0, 7:]))
    assert bundle.language == "hi"


def test_language_empty_omitted() -> None:
    north = build_synthetic_north_observation(language="")
    sdk = north_observation_to_sdk_dict(north, decode_images=False)
    assert "/language" not in sdk


def test_robot_client_codec_smoke_cli() -> None:
    code = robot_client_main(
        [
            "--mode",
            "codec-smoke",
            "--observation-contract",
            str(REVIEWED_CONTRACT),
            "--prompt",
            "smoke",
        ]
    )
    assert code == 0
