"""Tests for site observation contracts and JPEG EOI trimming."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from pi_dex.actions import ActionRepresentation
from pi_dex.jpeg_eoi import trim_padded_jpeg
from pi_dex.observation_contract import ReviewStatus
from pi_dex.observation_contract import load_observation_contract
from pi_dex.observation_contract import observation_contract_to_mapping
from pi_dex.spec import ActionTimebase
from tests.helpers import spec_for_representation

REPO_ROOT = Path(__file__).resolve().parents[1]
UNREVIEWED_CONTRACT = REPO_ROOT / "configs/site/joint_29d_observation.unreviewed.json"


def test_trim_padded_jpeg_cuts_at_final_eoi_not_trailing_zeros() -> None:
    # Embedded NUL before EOI must be preserved; only post-EOI padding is dropped.
    payload = b"\xff\xd8\x00fake\xff\xd9\x00\x00\x00"
    assert trim_padded_jpeg(payload) == b"\xff\xd8\x00fake\xff\xd9"
    assert trim_padded_jpeg(np.frombuffer(payload, dtype=np.uint8)) == b"\xff\xd8\x00fake\xff\xd9"


def test_trim_padded_jpeg_rejects_missing_eoi() -> None:
    with pytest.raises(ValueError, match="EOI"):
        trim_padded_jpeg(b"\xff\xd8not-a-finished-jpeg\x00\x00")


def test_unreviewed_site_contract_loads_but_blocks_training(action_spec) -> None:
    contract = load_observation_contract(UNREVIEWED_CONTRACT)
    assert contract.review_status is ReviewStatus.UNREVIEWED
    assert contract.action_representation is ActionRepresentation.JOINT_29D
    assert contract.state_dim == 65
    assert contract.unused_sharpa_vision_groups == ("observe/vision/head/stereo/righteye",)
    with pytest.raises(ValueError, match="review_status"):
        contract.require_reviewed_for_training()

    joint_spec = dataclasses.replace(
        spec_for_representation(action_spec, ActionRepresentation.JOINT_29D),
        physical_horizon=contract.physical_horizon,
        timebase=ActionTimebase.RAW_CONTROL_60_HZ,
        control_frequency_hz=contract.control_frequency_hz,
        max_group_timestamp_skew_ms=contract.max_group_timestamp_skew_ms,
        max_alignment_timestamp_error_ms=contract.max_alignment_timestamp_error_ms,
        max_control_period_error_ms=contract.max_control_period_error_ms,
    )
    contract.validate_against_action_spec(joint_spec)


def test_observation_contract_roundtrip(tmp_path: Path) -> None:
    contract = load_observation_contract(UNREVIEWED_CONTRACT)
    mapping = observation_contract_to_mapping(contract)
    out = tmp_path / "contract.json"
    out.write_text(json.dumps(mapping), encoding="utf-8")
    reloaded = load_observation_contract(out)
    assert observation_contract_to_mapping(reloaded) == mapping


def test_reviewed_contract_requires_reviewer(tmp_path: Path) -> None:
    payload = json.loads(UNREVIEWED_CONTRACT.read_text(encoding="utf-8"))
    payload["review_status"] = "reviewed"
    payload["reviewed_by"] = None
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="reviewed_by"):
        load_observation_contract(bad)
