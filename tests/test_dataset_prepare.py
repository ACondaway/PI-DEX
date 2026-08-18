"""Unit tests for dataset prepare overlay."""

from __future__ import annotations

import json
import pathlib

from pi_dex.dataset_prepare import materialize_prepared_dataset
from pi_dex.sharpa_dataset import discover_episodes


def _write_fake_episode(root: pathlib.Path, season: str, episode: str, prompt: str) -> None:
    episode_dir = root / season / episode
    episode_dir.mkdir(parents=True)
    (episode_dir / "train_fake.hdf5").write_bytes(b"not-a-real-hdf5")
    (episode_dir / "anno.json").write_text(
        json.dumps({"tags": {"task_instruction": prompt, "task_name": ""}}, indent=2) + "\n",
        encoding="utf-8",
    )


def test_materialize_fills_empty_prompt(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "source"
    prepared = tmp_path / "prepared"
    _write_fake_episode(source, "season_POC22005_a", "ep1", "")
    _write_fake_episode(source, "season_POC22005_a", "ep2", "keep me")

    meta = materialize_prepared_dataset(
        source_root=source,
        prepared_root=prepared,
        default_prompt="  Fill battery  ",
        fill_empty_prompt=True,
        overwrite=False,
    )
    assert meta["episode_count"] == 2
    assert meta["filled_empty_prompt"] == 1
    assert meta["kept_existing_prompt"] == 1

    episodes = discover_episodes(prepared)
    assert len(episodes) == 2
    prompts = {
        ep.episode_id: json.loads(ep.anno_path.read_text(encoding="utf-8"))["tags"]["task_instruction"]
        for ep in episodes
    }
    assert prompts["season_POC22005_a/ep1"] == "Fill battery"
    assert prompts["season_POC22005_a/ep2"] == "keep me"
    # HDF5 is a symlink into source (immutable Academic trees stay untouched).
    hdf5 = next(prepared.rglob("train_fake.hdf5"))
    assert hdf5.is_symlink()
    assert not (source / "season_POC22005_a" / "ep1" / "anno.json").read_text().count("Fill battery")
