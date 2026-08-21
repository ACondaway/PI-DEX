"""Inventory SharpaOpenData roots without building the full sample index.

Use this before ``compute-norm-stats`` / ``train`` on the multi-task OpenData root
so you can confirm episode coverage and split counts without opening every HDF5
for horizon probing.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter
from collections.abc import Sequence
from typing import Any

from pi_dex.data.episode_split import SplitName
from pi_dex.data.episode_split import assign_episode_splits
from pi_dex.data.episode_split import list_split_rejects
from pi_dex.data.episode_split import split_manifest
from pi_dex.data.observation_contract import load_observation_contract
from pi_dex.data.sharpa_dataset import discover_episodes


def inventory_dataset(
    *,
    dataset_root: pathlib.Path | str,
    observation_contract: pathlib.Path | str,
) -> dict[str, Any]:
    """Discover episodes and summarize per-task + split coverage."""
    contract = load_observation_contract(observation_contract)
    episodes = discover_episodes(dataset_root)
    assignments = assign_episode_splits(episodes, contract.split_policy)
    rejects = list_split_rejects(episodes)
    # Path prefix under dataset-root (task collection when root is OpenData).
    path_counts = Counter(_task_bucket(episode.episode_id) for episode in episodes)
    split_counts = Counter(split.value for split in assignments.values())
    usable = len(assignments)
    return {
        "dataset_root": str(pathlib.Path(dataset_root).resolve()),
        "contract_id": contract.contract_id,
        "review_status": contract.review_status.value,
        "episode_count": len(episodes),
        "usable_episode_count": usable,
        "rejected_episode_count": len(rejects),
        "rejected_episodes_sample": rejects[:50],
        "task_count": len(path_counts),
        "tasks": [
            {"task": name, "episode_count": count}
            for name, count in sorted(path_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "split_counts": {
            "train": int(split_counts.get(SplitName.TRAIN.value, 0)),
            "validation": int(split_counts.get(SplitName.VALIDATION.value, 0)),
            "test": int(split_counts.get(SplitName.TEST.value, 0)),
        },
        "split_manifest": split_manifest(episodes, contract=contract),
    }


def _task_bucket(episode_id: str) -> str:
    return episode_id.split("/", 1)[0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pi-dex-dataset-inventory")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--observation-contract", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--allow-unreviewed-contract",
        action="store_true",
        help="Allow inventory against an unreviewed contract (not for training)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    contract = load_observation_contract(args.observation_contract)
    if not args.allow_unreviewed_contract:
        contract.require_reviewed_for_training()
    payload = inventory_dataset(
        dataset_root=args.dataset_root,
        observation_contract=args.observation_contract,
    )
    text = json.dumps(payload, indent=2) + "\n"
    if args.output_json:
        out = pathlib.Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Full split_manifest with every episode can be huge; write summary + optional full file.
        summary = {
            key: payload[key]
            for key in (
                "dataset_root",
                "contract_id",
                "review_status",
                "episode_count",
                "usable_episode_count",
                "rejected_episode_count",
                "rejected_episodes_sample",
                "task_count",
                "tasks",
                "split_counts",
            )
        }
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        full_path = out.with_suffix(".full.json")
        full_path.write_text(text, encoding="utf-8")
        print(json.dumps({**summary, "full_manifest": str(full_path)}, indent=2))
    else:
        # stdout: summary only
        summary = {
            key: payload[key]
            for key in (
                "dataset_root",
                "contract_id",
                "review_status",
                "episode_count",
                "usable_episode_count",
                "rejected_episode_count",
                "rejected_episodes_sample",
                "task_count",
                "tasks",
                "split_counts",
            )
        }
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
