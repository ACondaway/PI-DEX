"""Materialize a training-ready Sharpa dataset overlay without mutating the source tree.

Creates a prepared root where each episode directory contains:

* a rewritten ``anno.json`` (empty ``task_instruction`` filled when requested)
* symlinks to the original HDF5 and other sidecars

This keeps Academic/source data immutable while allowing PI-DEX split/prompt policies
to accept the episodes.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
from collections.abc import Sequence
from typing import Any

from pi_dex.data.sharpa_dataset import discover_episodes


def materialize_prepared_dataset(
    *,
    source_root: pathlib.Path | str,
    prepared_root: pathlib.Path | str,
    default_prompt: str | None = None,
    fill_empty_prompt: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build ``prepared_root`` from ``source_root`` episode discovery."""
    source = pathlib.Path(source_root).resolve()
    prepared = pathlib.Path(prepared_root).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source_root: not a directory: {source}")
    if fill_empty_prompt:
        if type(default_prompt) is not str or not default_prompt.strip():
            raise ValueError("default_prompt: required non-empty string when fill_empty_prompt=True")
        default_prompt = " ".join(default_prompt.split())

    if prepared.exists():
        if not overwrite:
            raise FileExistsError(f"prepared_root already exists (pass overwrite=True): {prepared}")
        shutil.rmtree(prepared)
    prepared.mkdir(parents=True, exist_ok=False)

    episodes = discover_episodes(source)
    filled = 0
    kept = 0
    for episode in episodes:
        rel = pathlib.Path(episode.episode_id)
        dest_dir = prepared / rel
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Symlink every file except anno.json (which we may rewrite).
        for path in sorted(episode.episode_dir.iterdir()):
            if not path.is_file():
                continue
            if path.name == "anno.json":
                continue
            target = dest_dir / path.name
            if target.exists() or target.is_symlink():
                target.unlink()
            os.symlink(path.resolve(), target)

        anno_payload = json.loads(episode.anno_path.read_text(encoding="utf-8"))
        if not isinstance(anno_payload, dict):
            raise TypeError(f"{episode.anno_path}: expected JSON object")
        tags = anno_payload.get("tags")
        if not isinstance(tags, dict):
            tags = {}
            anno_payload["tags"] = tags
        prompt = tags.get("task_instruction")
        if fill_empty_prompt and (type(prompt) is not str or not prompt.strip()):
            tags["task_instruction"] = default_prompt
            if type(tags.get("task_name")) is not str or not str(tags.get("task_name")).strip():
                tags["task_name"] = pathlib.Path(source.name).as_posix()
            filled += 1
        else:
            kept += 1
        (dest_dir / "anno.json").write_text(
            json.dumps(anno_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    # Marker for lineage.
    meta = {
        "schema_version": 1,
        "source_root": str(source),
        "prepared_root": str(prepared),
        "episode_count": len(episodes),
        "filled_empty_prompt": filled,
        "kept_existing_prompt": kept,
        "default_prompt": default_prompt if fill_empty_prompt else None,
    }
    (prepared / "pi_dex_prepared.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pi-dex-prepare-dataset")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument(
        "--default-prompt",
        default="",
        help="Prompt written into empty tags.task_instruction fields",
    )
    parser.add_argument(
        "--no-fill-empty-prompt",
        action="store_true",
        help="Only symlink/copy structure; do not rewrite empty prompts",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)
    meta = materialize_prepared_dataset(
        source_root=args.source_root,
        prepared_root=args.prepared_root,
        default_prompt=args.default_prompt or None,
        fill_empty_prompt=not args.no_fill_empty_prompt,
        overwrite=args.overwrite,
    )
    text = json.dumps(meta, indent=2)
    print(text)
    if args.output_json:
        out = pathlib.Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
