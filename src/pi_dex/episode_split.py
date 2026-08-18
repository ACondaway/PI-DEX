"""Episode-level train/validation/test splits for Sharpa observation contracts."""

from __future__ import annotations

from collections.abc import Sequence
import enum
import hashlib
import json
from typing import Any

from pi_dex.observation_contract import SharpaObservationContract
from pi_dex.observation_contract import SplitPolicy
from pi_dex.sharpa_dataset import EpisodeRef


class SplitName(enum.StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


_SUPPORTED_STRATEGY = "episode_hash_stratified_by_task"


def assign_episode_splits(
    episodes: Sequence[EpisodeRef],
    split_policy: SplitPolicy,
) -> dict[str, SplitName]:
    """Assign each usable episode id to train/validation/test.

    Strategy ``episode_hash_stratified_by_task``:
    * optional dedupe by ``episode_id``
    * stratify by ``anno.json`` task instruction text
    * within each stratum, assign by stable hash of ``f"{seed}:{episode_id}"``
      into contiguous probability buckets matching the declared fractions

    Episodes with missing/empty ``tags.task_instruction`` are omitted from the
    returned mapping (they belong to no split). See :func:`list_split_rejects`.
    """
    if split_policy.strategy != _SUPPORTED_STRATEGY:
        raise ValueError(
            f"split_policy.strategy: unsupported {split_policy.strategy!r}; expected {_SUPPORTED_STRATEGY!r}"
        )

    selected: list[EpisodeRef] = []
    seen: set[str] = set()
    for episode in episodes:
        if split_policy.dedupe_by_episode_id and episode.episode_id in seen:
            continue
        seen.add(episode.episode_id)
        selected.append(episode)

    by_task: dict[str, list[EpisodeRef]] = {}
    for episode in selected:
        task = try_task_key(episode)
        if task is None:
            continue
        by_task.setdefault(task, []).append(episode)

    assignments: dict[str, SplitName] = {}
    train_end = float(split_policy.train_fraction)
    val_end = train_end + float(split_policy.validation_fraction)
    for task in sorted(by_task):
        members = sorted(by_task[task], key=lambda item: item.episode_id)
        for episode in members:
            bucket = _unit_interval_hash(f"{split_policy.seed}:{episode.episode_id}")
            if bucket < train_end:
                split = SplitName.TRAIN
            elif bucket < val_end:
                split = SplitName.VALIDATION
            else:
                split = SplitName.TEST
            assignments[episode.episode_id] = split
    return assignments


def list_split_rejects(episodes: Sequence[EpisodeRef]) -> list[dict[str, str]]:
    """Return episodes that cannot be stratified (missing/empty task_instruction)."""
    rejects: list[dict[str, str]] = []
    seen: set[str] = set()
    for episode in episodes:
        if episode.episode_id in seen:
            continue
        seen.add(episode.episode_id)
        reason = task_key_reject_reason(episode)
        if reason is not None:
            rejects.append(
                {
                    "episode_id": episode.episode_id,
                    "anno_path": str(episode.anno_path),
                    "reason": reason,
                }
            )
    return rejects


def filter_episodes_for_split(
    episodes: Sequence[EpisodeRef],
    *,
    contract: SharpaObservationContract,
    split: SplitName,
) -> tuple[EpisodeRef, ...]:
    """Return episodes belonging to ``split`` under the contract split policy."""
    assignments = assign_episode_splits(episodes, contract.split_policy)
    filtered = tuple(episode for episode in episodes if assignments.get(episode.episode_id) is split)
    if not filtered:
        raise ValueError(f"split {split.value}: no episodes assigned under {contract.contract_id!r}")
    return filtered


def split_manifest(
    episodes: Sequence[EpisodeRef],
    *,
    contract: SharpaObservationContract,
) -> dict[str, Any]:
    """JSON-serializable episode split inventory."""
    assignments = assign_episode_splits(episodes, contract.split_policy)
    rejects = list_split_rejects(episodes)
    counts = {name.value: 0 for name in SplitName}
    rows = []
    for episode in episodes:
        split = assignments.get(episode.episode_id)
        if split is None:
            continue
        counts[split.value] += 1
        task = try_task_key(episode)
        rows.append(
            {
                "episode_id": episode.episode_id,
                "split": split.value,
                "task": task if task is not None else "",
            }
        )
    return {
        "contract_id": contract.contract_id,
        "strategy": contract.split_policy.strategy,
        "seed": contract.split_policy.seed,
        "fractions": {
            "train": float(contract.split_policy.train_fraction),
            "validation": float(contract.split_policy.validation_fraction),
            "test": float(contract.split_policy.test_fraction),
        },
        "counts": counts,
        "rejected_count": len(rejects),
        "rejected_episodes": rejects,
        "episodes": rows,
    }


def try_task_key(episode: EpisodeRef) -> str | None:
    """Return normalized task_instruction, or ``None`` if unusable for stratification."""
    return None if task_key_reject_reason(episode) is not None else _task_key_unchecked(episode)


def task_key_reject_reason(episode: EpisodeRef) -> str | None:
    """Return a short reject reason, or ``None`` when the episode can be stratified."""
    try:
        payload = json.loads(episode.anno_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return f"unreadable anno.json: {error}"
    try:
        prompt = payload["tags"]["task_instruction"]
    except KeyError:
        return "missing tags.task_instruction"
    if type(prompt) is not str:
        return f"task_instruction type {type(prompt).__name__}"
    if not prompt.strip():
        return "empty task_instruction"
    return None


def _task_key_unchecked(episode: EpisodeRef) -> str:
    payload = json.loads(episode.anno_path.read_text(encoding="utf-8"))
    prompt = payload["tags"]["task_instruction"]
    return " ".join(str(prompt).split())


def _unit_interval_hash(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Use 53 bits so the float is exactly representable.
    value = int.from_bytes(digest[:7], "big") >> 3
    return value / float(1 << 53)
