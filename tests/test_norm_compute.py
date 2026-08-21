"""Tests for vectorized joint_29d norm-stat extraction."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from pi_dex.core.actions import ActionRepresentation
from pi_dex.data.norm_compute import compute_joint29d_normalization_stats
from pi_dex.data.norm_compute import extract_episode_norm_batch
from pi_dex.data.norm_compute import resolve_norm_workers
from pi_dex.data.norm_compute import valid_aligned_starts
from pi_dex.data.observation_contract import load_observation_contract
from pi_dex.data.sharpa_data import derive_bimanual_logical_action_chunk
from pi_dex.data.sharpa_dataset import _load_action_chunk
from pi_dex.data.sharpa_dataset import _load_state
from pi_dex.data.sharpa_dataset import build_sample_index
from pi_dex.data.sharpa_dataset import discover_episodes
from pi_dex.training.training_runner import build_joint_spec_from_contract
from tests.helpers import spec_for_representation
from tests.test_sharpa_data import BASE_TIME_S
from tests.test_sharpa_data import RAW_PERIOD_S
from tests.test_sharpa_data import make_group
from tests.test_sharpa_data import make_groups
from tests.test_sharpa_data import make_provenance
from tests.test_sharpa_data import make_timeline

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = REPO_ROOT / "configs/site/joint_29d_observation.unreviewed.json"
REAL_EPISODE_ROOT = Path(
    "/mnt/netdata/Team/Academic/Data/North/SharpaOpenData/ClearPlate/season_POC22027_2026_04_18_10_54_31_train"
)


def _joint_spec(action_spec, *, horizon: int | None = None, period_error_ms: float | None = None):
    spec = spec_for_representation(action_spec, ActionRepresentation.JOINT_29D)
    replacements: dict[str, object] = {}
    if horizon is not None:
        replacements["physical_horizon"] = horizon
    if period_error_ms is not None:
        replacements["max_control_period_error_ms"] = period_error_ms
        replacements["max_alignment_timestamp_error_ms"] = period_error_ms
        replacements["max_group_timestamp_skew_ms"] = period_error_ms
    return dataclasses.replace(spec, **replacements) if replacements else spec


def test_resolve_norm_workers_cli_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NORM_WORKERS", raising=False)
    assert resolve_norm_workers(4) == 4
    monkeypatch.setenv("NORM_WORKERS", "12")
    assert resolve_norm_workers(None) == 12
    assert resolve_norm_workers(3) == 3
    with pytest.raises(ValueError, match=">= 1"):
        resolve_norm_workers(0)


def test_valid_starts_match_per_window_probe(action_spec) -> None:
    spec = _joint_spec(action_spec, horizon=2)
    timeline = make_timeline(length=3)
    groups = make_groups()
    starts = valid_aligned_starts(timeline=timeline, groups=groups, spec=spec)
    expected: list[int] = []
    provenance = make_provenance()
    for start in range(timeline.time.shape[0]):
        try:
            derive_bimanual_logical_action_chunk(
                aligned_timeline=timeline,
                provenance=provenance,
                left_arm=groups[0],
                left_hand=groups[1],
                right_arm=groups[2],
                right_hand=groups[3],
                motor=groups[4],
                start_aligned_frame=start,
                spec=spec,
                kinematics=None,
            )
        except ValueError:
            continue
        expected.append(start)
    np.testing.assert_array_equal(starts, np.asarray(expected, dtype=np.int64))


def test_valid_starts_accept_one_skipped_raw_tick(action_spec) -> None:
    spec_20 = _joint_spec(action_spec, horizon=2, period_error_ms=20.0)
    spec_8 = _joint_spec(action_spec, horizon=2, period_error_ms=8.0)
    raw_length = 6
    aligned = np.array([0, 2, 4], dtype=np.int32)
    timestamps = BASE_TIME_S + np.arange(raw_length, dtype=np.float64) * RAW_PERIOD_S
    timestamps[3:] += RAW_PERIOD_S
    groups = tuple(
        make_group(field, raw_length=raw_length, aligned_index=aligned)
        for field in (
            "action/left_arm/joint_angle",
            "action/left_hand/joint_angle",
            "action/right_arm/joint_angle",
            "action/right_hand/joint_angle",
            "action/motor/joint_angle",
        )
    )
    patched = []
    for group in groups:
        time = np.array(group.time, copy=True)
        time[:, 0] = timestamps
        patched.append(
            group.__class__(
                group.field_name,
                group.joint_order,
                np.array(group.joint_angles, copy=True),
                time,
                np.array(group.aligned_index, copy=True),
            )
        )
    groups = tuple(patched)
    timeline = make_timeline(length=3)
    starts_20 = valid_aligned_starts(timeline=timeline, groups=groups, spec=spec_20)
    starts_8 = valid_aligned_starts(timeline=timeline, groups=groups, spec=spec_8)
    assert 1 in set(starts_20.tolist())
    assert 1 not in set(starts_8.tolist())


def _write_synthetic_episode(episode_dir: Path, *, aligned_length: int = 16, seed: int = 0) -> None:
    episode_dir.mkdir(parents=True, exist_ok=True)
    raw_length = aligned_length * 2
    aligned_index = (np.arange(aligned_length, dtype=np.int32) * 2)
    raw_time = BASE_TIME_S + np.arange(raw_length, dtype=np.float64) * RAW_PERIOD_S
    aligned_time = BASE_TIME_S + np.arange(aligned_length, dtype=np.float64) * (2.0 * RAW_PERIOD_S)
    time_raw = np.stack((raw_time, np.ones(raw_length, dtype=np.float64)), axis=1)
    time_aligned = np.stack((aligned_time, np.ones(aligned_length, dtype=np.float64)), axis=1)
    rng = np.random.default_rng(seed)
    action_fields = {
        "action/left_arm/joint_angle": 7,
        "action/left_hand/joint_angle": 22,
        "action/right_arm/joint_angle": 7,
        "action/right_hand/joint_angle": 22,
        "action/motor/joint_angle": 7,
    }
    state_fields = {
        "state/left_arm/joint_angle": 7,
        "state/left_hand/joint_angle": 22,
        "state/right_arm/joint_angle": 7,
        "state/right_hand/joint_angle": 22,
        "state/motor/joint_angle": 7,
    }
    with h5py.File(episode_dir / "train_synthetic.hdf5", "w") as handle:
        handle.create_dataset("observe/vision/head/stereo/lefteye/time", data=time_aligned)
        handle.create_dataset("mode/sub_state", data=np.ones(aligned_length, dtype=np.int16))
        for field_name, width in action_fields.items():
            group_root = field_name.rsplit("/", 1)[0]
            handle.create_dataset(field_name, data=rng.normal(size=(raw_length, width)).astype(np.float32))
            handle.create_dataset(f"{group_root}/time", data=time_raw)
            handle.create_dataset(f"{group_root}/aligned_index", data=aligned_index)
        for field_name, width in state_fields.items():
            group_root = field_name.rsplit("/", 1)[0]
            handle.create_dataset(field_name, data=rng.normal(size=(raw_length, width)).astype(np.float32))
            handle.create_dataset(f"{group_root}/aligned_index", data=aligned_index)
    (episode_dir / "anno.json").write_text(
        json.dumps({"tags": {"task_instruction": "synthetic joint task"}}),
        encoding="utf-8",
    )


def _contract_and_spec():
    contract = load_observation_contract(CONTRACT)
    spec = build_joint_spec_from_contract(
        contract,
        robot_id="POC22027",
        embodiment_version="sharpa_north_v1",
        command_semantics_version="sharpa_sdk_commanded_joint_position_absolute_v1",
        hand_mapping_version="sharpa_north_hand_mapping_v1",
        clock_domain="unix_realtime",
    )
    return contract, spec, make_provenance()


def test_extract_matches_dataset_on_synthetic_hdf5(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    _write_synthetic_episode(root / "ep0", seed=1)
    contract, spec, provenance = _contract_and_spec()
    episodes = discover_episodes(root)
    batch = extract_episode_norm_batch(episodes[0], spec, contract)
    assert batch is not None
    sample_index = build_sample_index(episodes, spec=spec, contract=contract, provenance=provenance)
    assert batch.state.shape[0] == len(sample_index)
    with h5py.File(episodes[0].hdf5_path, "r") as handle:
        for index, sample_ref in enumerate(sample_index):
            if index not in {0, len(sample_index) // 2, len(sample_index) - 1}:
                continue
            state = _load_state(handle, sample_ref.start_aligned_frame, contract)
            chunk = _load_action_chunk(
                handle,
                start_aligned_frame=sample_ref.start_aligned_frame,
                spec=spec,
                provenance=provenance,
            )
            np.testing.assert_allclose(batch.state[index], state)
            np.testing.assert_allclose(batch.left_actions[index], chunk.left_actions)
            np.testing.assert_allclose(batch.right_actions[index], chunk.right_actions)


def test_multiprocess_matches_serial_window_count(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    _write_synthetic_episode(root / "ep0", seed=1)
    _write_synthetic_episode(root / "ep1", seed=2)
    contract, spec, provenance = _contract_and_spec()
    episodes = discover_episodes(root)
    stats_one, meta_one = compute_joint29d_normalization_stats(
        episodes, spec=spec, contract=contract, provenance=provenance, workers=1
    )
    stats_two, meta_two = compute_joint29d_normalization_stats(
        episodes, spec=spec, contract=contract, provenance=provenance, workers=2
    )
    assert meta_one["samples"] == meta_two["samples"]
    assert meta_two["workers"] == 2
    np.testing.assert_allclose(stats_one["state"].mean, stats_two["state"].mean)
    np.testing.assert_allclose(stats_one["left_actions"].mean, stats_two["left_actions"].mean)
    np.testing.assert_allclose(stats_one["right_actions"].mean, stats_two["right_actions"].mean)


@pytest.mark.skipif(not REAL_EPISODE_ROOT.is_dir(), reason="SharpaOpenData not mounted")
def test_extract_matches_real_episode_index() -> None:
    contract, spec, provenance = _contract_and_spec()
    episodes = discover_episodes(REAL_EPISODE_ROOT)[:1]
    batch = extract_episode_norm_batch(episodes[0], spec, contract)
    assert batch is not None
    sample_index = build_sample_index(episodes, spec=spec, contract=contract, provenance=provenance)
    assert batch.state.shape[0] == len(sample_index)
    assert batch.left_actions.shape == (len(sample_index), spec.physical_horizon, spec.logical_action_dim)
    with h5py.File(episodes[0].hdf5_path, "r") as handle:
        for index in (0, min(7, len(sample_index) - 1)):
            sample_ref = sample_index[index]
            state = _load_state(handle, sample_ref.start_aligned_frame, contract)
            chunk = _load_action_chunk(
                handle,
                start_aligned_frame=sample_ref.start_aligned_frame,
                spec=spec,
                provenance=provenance,
            )
            np.testing.assert_allclose(batch.state[index], state)
            np.testing.assert_allclose(batch.left_actions[index], chunk.left_actions)
            np.testing.assert_allclose(batch.right_actions[index], chunk.right_actions)
