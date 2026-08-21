"""Volcano Engine (火山引擎 MLP) multi-node launch helpers.

The platform injects ``MLP_WORKER_NUM``, ``MLP_WORKER_GPU``, ``MLP_ROLE_INDEX``,
``MLP_WORKER_0_HOST``, and ``MLP_WORKER_0_PORT``. This module maps those onto
``torchrun`` / ``torch.distributed.run`` so ``pi_dex.training.training_runner`` can use
standard ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import Mapping
from typing import Sequence


MLP_ENV_KEYS = (
    "MLP_WORKER_NUM",
    "MLP_WORKER_GPU",
    "MLP_ROLE_INDEX",
    "MLP_WORKER_0_HOST",
    "MLP_WORKER_0_PORT",
)


def read_mlp_launch_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Read and validate Volcano MLP distributed launch variables."""
    env = os.environ if environ is None else environ
    missing = [key for key in MLP_ENV_KEYS if not str(env.get(key, "")).strip()]
    if missing:
        raise ValueError(
            "Volcano MLP launch requires env vars: "
            + ", ".join(MLP_ENV_KEYS)
            + f"; missing: {', '.join(missing)}"
        )
    nnodes = int(env["MLP_WORKER_NUM"])
    nproc = int(env["MLP_WORKER_GPU"])
    node_rank = int(env["MLP_ROLE_INDEX"])
    if nnodes <= 0 or nproc <= 0:
        raise ValueError("MLP_WORKER_NUM and MLP_WORKER_GPU must be positive integers")
    if node_rank < 0 or node_rank >= nnodes:
        raise ValueError(f"MLP_ROLE_INDEX {node_rank} outside [0, {nnodes})")
    return {
        "nnodes": str(nnodes),
        "nproc_per_node": str(nproc),
        "node_rank": str(node_rank),
        "master_addr": str(env["MLP_WORKER_0_HOST"]).strip(),
        "master_port": str(env["MLP_WORKER_0_PORT"]).strip(),
    }


def build_torchrun_command(
    *,
    training_argv: Sequence[str],
    mlp: Mapping[str, str] | None = None,
    python_executable: str | None = None,
) -> list[str]:
    """Build a ``torchrun`` command for the current MLP worker."""
    resolved = dict(mlp) if mlp is not None else read_mlp_launch_env()
    python = python_executable or sys.executable
    # Prefer the active conda env's torchrun (Volcano images often have several).
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    torchrun = None
    if conda_prefix:
        candidate = os.path.join(conda_prefix, "bin", "torchrun")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            torchrun = candidate
    if torchrun is None:
        torchrun = shutil.which("torchrun")
    if torchrun is None:
        # Fallback when the torch scripts path is not on PATH.
        return [
            python,
            "-m",
            "torch.distributed.run",
            f"--nnodes={resolved['nnodes']}",
            f"--nproc_per_node={resolved['nproc_per_node']}",
            f"--node_rank={resolved['node_rank']}",
            f"--master_addr={resolved['master_addr']}",
            f"--master_port={resolved['master_port']}",
            *list(training_argv),
        ]
    return [
        torchrun,
        f"--nnodes={resolved['nnodes']}",
        f"--nproc_per_node={resolved['nproc_per_node']}",
        f"--node_rank={resolved['node_rank']}",
        f"--master_addr={resolved['master_addr']}",
        f"--master_port={resolved['master_port']}",
        *list(training_argv),
    ]


def default_training_argv(extra: Sequence[str] | None = None) -> list[str]:
    """Default worker entry: installed ``pi-dex-train-pytorch`` plus user args."""
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    launcher = None
    if conda_prefix:
        candidate = os.path.join(conda_prefix, "bin", "pi-dex-train-pytorch")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            launcher = candidate
    if launcher is None:
        launcher = shutil.which("pi-dex-train-pytorch")
    if launcher is None:
        prefix = [sys.executable, "-m", "pi_dex.training.training_launcher"]
    else:
        prefix = [launcher]
    return [*prefix, *list(extra or ())]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry for Volcano custom-training start commands."""
    parser = argparse.ArgumentParser(
        prog="pi-dex-volc-train",
        description=(
            "Launch PI-DEX multi-node DDP via torchrun using Volcano MLP env vars. "
            "Pass training arguments after '--'."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the torchrun command without executing it",
    )
    parser.add_argument(
        "training_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to pi-dex-train-pytorch (include a leading --)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    forwarded = list(args.training_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if not forwarded:
        raise SystemExit(
            "pi-dex-volc-train: pass training args after '--', e.g. "
            "pi-dex-volc-train -- --action-representation joint_29d "
            "--runner pi_dex.training.training_runner:run -- --mode train ..."
        )

    # If the user already passed a full module path as first token, use it as-is.
    if forwarded[0].endswith(".py") or forwarded[0] in {"-m", "python", "python3"}:
        training_argv = forwarded
    else:
        training_argv = default_training_argv(forwarded)

    # Prefer injecting --distributed into the runner tail (after the second '--').
    training_argv = _ensure_runner_distributed_flag(training_argv)

    command = build_torchrun_command(training_argv=training_argv)
    if args.dry_run:
        print(" ".join(command))
        return 0
    print("volc_launch:", " ".join(command), flush=True)
    return subprocess.call(command)


def _ensure_runner_distributed_flag(argv: list[str]) -> list[str]:
    """Append ``--distributed`` to the training_runner argument section when needed."""
    if "--distributed" in argv:
        return argv
    # pi-dex-train-pytorch ... --runner ... -- --mode train ...
    if "--" in argv:
        head, tail = argv[: argv.index("--") + 1], argv[argv.index("--") + 1 :]
        return [*head, *tail, "--distributed"]
    return [*argv, "--distributed"]


if __name__ == "__main__":
    raise SystemExit(main())
