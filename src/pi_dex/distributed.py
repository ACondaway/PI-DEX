"""Multi-process DDP helpers for PI-DEX PyTorch training.

Supports torchrun and Volcano Engine MLP env vars (``MLP_WORKER_*`` /
``MLP_ROLE_INDEX``) once the launcher has mapped them onto the standard
``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` / ``MASTER_*`` contract.
"""

from __future__ import annotations

import os
from typing import Any


def launched_under_torch_distributed() -> bool:
    """Return True when torchrun/elastic (or an equivalent launcher) set RANK."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def resolve_rank_world_local() -> tuple[int, int, int]:
    """Read ``RANK``, ``WORLD_SIZE``, and ``LOCAL_RANK`` from the environment."""
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except KeyError as error:
        raise RuntimeError(
            "distributed: RANK and WORLD_SIZE must be set (use torchrun or pi-dex-volc-train)"
        ) from error
    except ValueError as error:
        raise ValueError("distributed: RANK/WORLD_SIZE must be integers") from error
    if world_size <= 0:
        raise ValueError(f"distributed: WORLD_SIZE must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"distributed: RANK {rank} is outside [0, {world_size})")
    local_rank_raw = os.environ.get("LOCAL_RANK")
    if local_rank_raw is None:
        local_rank = 0 if world_size == 1 else rank
    else:
        local_rank = int(local_rank_raw)
        if local_rank < 0:
            raise ValueError(f"distributed: LOCAL_RANK must be non-negative, got {local_rank}")
    return rank, world_size, local_rank


def is_main_process(*, rank: int | None = None) -> bool:
    """Return True for global rank 0 (or when distributed is inactive)."""
    if rank is not None:
        return rank == 0
    if not launched_under_torch_distributed():
        return True
    return int(os.environ["RANK"]) == 0


def init_process_group(*, backend: str | None = None) -> tuple[int, int, int]:
    """Initialize the default process group and bind the local CUDA device.

    Returns:
        ``(rank, world_size, local_rank)``.
    """
    import torch
    import torch.distributed as dist

    rank, world_size, local_rank = resolve_rank_world_local()
    if dist.is_available() and dist.is_initialized():
        return rank, world_size, local_rank

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device_id = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            device_id=device_id,
        )
    else:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def cleanup_process_group() -> None:
    """Destroy the default process group when it is initialized."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def unwrap_model(model: Any) -> Any:
    """Return the underlying ``nn.Module`` when ``model`` is DDP-wrapped."""
    return getattr(model, "module", model)


def device_for_rank(*, local_rank: int, requested: str = "cuda") -> Any:
    """Map a CLI device request onto the process-local device."""
    import torch

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("distributed: CUDA requested but torch.cuda.is_available() is False")
        return torch.device(f"cuda:{local_rank}")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"distributed: unsupported device {requested!r}")


def barrier() -> None:
    """Synchronize all ranks when a process group is active."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
