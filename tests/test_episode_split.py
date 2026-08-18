"""Tests for episode split assignment."""

from __future__ import annotations

import json
from pathlib import Path

from pi_dex.episode_split import SplitName
from pi_dex.episode_split import assign_episode_splits
from pi_dex.episode_split import filter_episodes_for_split
from pi_dex.observation_contract import load_observation_contract
from pi_dex.sharpa_dataset import EpisodeRef

REPO_ROOT = Path(__file__).resolve().parents[1]
REVIEWED = REPO_ROOT / "configs/site/joint_29d_observation.reviewed.json"


def _episode(tmp_path: Path, episode_id: str, task: str) -> EpisodeRef:
    episode_dir = tmp_path / episode_id
    episode_dir.mkdir(parents=True)
    anno = episode_dir / "anno.json"
    anno.write_text(json.dumps({"tags": {"task_instruction": task}}), encoding="utf-8")
    hdf5 = episode_dir / "train_dummy.hdf5"
    hdf5.write_bytes(b"")
    return EpisodeRef(episode_id=episode_id, episode_dir=episode_dir, hdf5_path=hdf5, anno_path=anno)


def test_reviewed_contract_loads_and_allows_training() -> None:
    contract = load_observation_contract(REVIEWED)
    contract.require_reviewed_for_training()
    assert contract.reviewed_by == "congsheng"


def test_hash_split_is_deterministic_and_covers_all_splits(tmp_path: Path) -> None:
    contract = load_observation_contract(REVIEWED)
    episodes = tuple(_episode(tmp_path, f"ep{i:03d}", "Pick plate") for i in range(40))
    first = assign_episode_splits(episodes, contract.split_policy)
    second = assign_episode_splits(episodes, contract.split_policy)
    assert first == second
    assert set(first.values()) >= {SplitName.TRAIN, SplitName.VALIDATION, SplitName.TEST}
    train = filter_episodes_for_split(episodes, contract=contract, split=SplitName.TRAIN)
    assert len(train) >= 20


def test_empty_task_instruction_is_rejected_from_splits(tmp_path: Path) -> None:
    from pi_dex.episode_split import list_split_rejects

    contract = load_observation_contract(REVIEWED)
    good = _episode(tmp_path, "good", "Wipe the plate")
    bad = _episode(tmp_path, "bad", "")
    episodes = (good, bad)
    assignments = assign_episode_splits(episodes, contract.split_policy)
    assert "good" in assignments
    assert "bad" not in assignments
    rejects = list_split_rejects(episodes)
    assert len(rejects) == 1
    assert rejects[0]["reason"] == "empty task_instruction"
