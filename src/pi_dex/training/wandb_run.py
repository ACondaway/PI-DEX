"""Weights & Biases helpers for PI-DEX joint_29d training (rank-0 only).

Authentication is via the ``WANDB_API_KEY`` environment variable only (no
interactive ``wandb login``). Example:

```bash
export WANDB_API_KEY=...
pi-dex-train-pytorch ... --wandb --wandb-project pi-dex
```
"""

from __future__ import annotations

from collections.abc import Mapping
import os
import pathlib
from typing import Any

WANDB_ID_FILENAME = "wandb_id.txt"
WANDB_API_KEY_ENV = "WANDB_API_KEY"


def require_wandb_api_key() -> str:
    """Return a non-empty ``WANDB_API_KEY`` from the environment."""
    api_key = os.environ.get(WANDB_API_KEY_ENV, "").strip()
    if not api_key:
        raise EnvironmentError(
            f"{WANDB_API_KEY_ENV} is required when wandb logging is enabled "
            "(export WANDB_API_KEY=...; do not use interactive wandb login). "
            "Pass --no-wandb for offline smoke runs."
        )
    return api_key


def init_train_wandb(
    *,
    enabled: bool,
    project: str,
    run_name: str,
    config: Mapping[str, Any],
    run_dir: pathlib.Path | str,
    entity: str | None = None,
    resume: bool = False,
) -> Any | None:
    """Initialize wandb for a training run.

    When ``enabled`` is False, returns ``None`` and does not touch the wandb
    global state (so unit tests / offline smoke do not need a login).

    When enabled, requires ``WANDB_API_KEY`` in the environment and disables
    interactive login prompts.
    """
    if not enabled:
        return None
    api_key = require_wandb_api_key()
    # Ensure the process env is the source of truth for the SDK.
    os.environ[WANDB_API_KEY_ENV] = api_key
    # Avoid interactive browser/login prompts on headless nodes.
    os.environ.setdefault("WANDB_SILENT", "true")

    try:
        import wandb
    except ImportError as error:
        raise ImportError(
            "wandb logging requested (--wandb) but the wandb package is not installed"
        ) from error

    directory = pathlib.Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    id_path = directory / WANDB_ID_FILENAME
    init_kwargs: dict[str, Any] = {
        "project": project,
        "name": run_name,
        "config": dict(config),
        "settings": wandb.Settings(api_key=api_key),
    }
    if entity:
        init_kwargs["entity"] = entity
    if resume:
        if not id_path.is_file():
            raise FileNotFoundError(f"wandb resume requires {id_path}")
        init_kwargs["id"] = id_path.read_text(encoding="utf-8").strip()
        init_kwargs["resume"] = "must"
        init_kwargs.pop("name", None)
    run = wandb.init(**init_kwargs)
    if not resume:
        id_path.write_text(f"{run.id}\n", encoding="utf-8")
    return run


def log_train_metrics(run: Any | None, metrics: Mapping[str, Any], *, step: int) -> None:
    """Log scalar training metrics when a wandb run is active."""
    if run is None:
        return
    run.log(dict(metrics), step=int(step))


def finish_train_wandb(run: Any | None) -> None:
    """Finish an active wandb run (no-op when logging was disabled)."""
    if run is None:
        return
    run.finish()


def should_save_checkpoint(*, global_step: int, save_interval: int, is_final: bool) -> bool:
    """Decide whether to publish a checkpoint for the current step."""
    if type(global_step) is not int or global_step <= 0:
        return False
    if is_final:
        return True
    if type(save_interval) is not int or save_interval <= 0:
        return False
    return global_step % save_interval == 0
