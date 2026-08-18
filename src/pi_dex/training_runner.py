"""First-party PI-DEX PyTorch training runner for joint_29d.

Wire through the launcher:

```bash
pi-dex-train-pytorch \\
  --action-representation joint_29d \\
  --runner pi_dex.training_runner:run -- \\
  --mode validate-data \\
  --observation-contract configs/site/....json \\
  --dataset-root /path/to/SharpaOpenData/ClearPlate \\
  --allow-unreviewed-contract
```

``train`` mode requires a converted ``pi05_base`` PyTorch weight directory, OpenPI
on ``PYTHONPATH``/uv env, precomputed assets, and a reviewed observation contract.
Random initialization is refused.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import dataclasses
import json
import pathlib
from typing import Any
import uuid

from pi_dex.actions import ActionRepresentation
from pi_dex.checkpoint_manager import build_sample_order
from pi_dex.checkpoint_manager import order_sha256
from pi_dex.checkpoint_manager import publish_training_checkpoint
from pi_dex.checkpoint_manager import resume_training_checkpoint
from pi_dex.episode_split import SplitName
from pi_dex.episode_split import filter_episodes_for_split
from pi_dex.episode_split import split_manifest
from pi_dex.observation_contract import OPENPI_IMAGE_KEYS
from pi_dex.observation_contract import SharpaObservationContract
from pi_dex.observation_contract import load_observation_contract
from pi_dex.wandb_run import finish_train_wandb
from pi_dex.wandb_run import init_train_wandb
from pi_dex.wandb_run import log_train_metrics
from pi_dex.wandb_run import should_save_checkpoint
from pi_dex.parameter_manifest import build_parameter_manifests
from pi_dex.parameter_manifest import require_full_finetune_manifest
from pi_dex.pi05_weights import load_verified_pi05_base
from pi_dex.pi05_weights import require_converted_base_dir
from pi_dex.sharpa_data import EpisodeActionProvenance
from pi_dex.sharpa_dataset import SharpaJoint29dDataset
from pi_dex.sharpa_dataset import SyntheticJoint29dDataset
from pi_dex.sharpa_dataset import build_sample_index
from pi_dex.sharpa_dataset import dataset_manifest
from pi_dex.sharpa_dataset import discover_episodes
from pi_dex.spec import ActionMode
from pi_dex.spec import BimanualActionSpec
from pi_dex.spec import HandNormalization
from pi_dex.training_launcher import PytorchTrainingLaunchContext


def run(context: PytorchTrainingLaunchContext) -> int:
    """Launcher entrypoint: ``pi_dex.training_runner:run``."""
    if context.action_representation is not ActionRepresentation.JOINT_29D:
        raise ValueError("training_runner: only joint_29d is implemented in this milestone")
    args = _parse_runner_args(context.runner_args)
    contract = load_observation_contract(args.observation_contract)
    if contract.action_representation is not ActionRepresentation.JOINT_29D:
        raise ValueError("observation contract must declare joint_29d")

    spec = build_joint_spec_from_contract(
        contract,
        robot_id=args.robot_id,
        embodiment_version=args.embodiment_version,
        command_semantics_version=args.command_semantics_version,
        hand_mapping_version=args.hand_mapping_version,
        clock_domain=args.clock_domain,
    )
    context.bind_action_spec(spec)

    if args.mode == "synthetic-smoke":
        return _run_synthetic_smoke(args=args, spec=spec, contract=contract)

    if not args.allow_unreviewed_contract:
        contract.require_reviewed_for_training()
    elif args.mode == "train" and not args.allow_unreviewed_train_smoke:
        raise ValueError(
            "train mode forbids --allow-unreviewed-contract unless "
            "--allow-unreviewed-train-smoke is also set (pipeline wiring only; not an A/B PASS)"
        )

    provenance = EpisodeActionProvenance(
        robot_id=args.robot_id,
        embodiment_version=args.embodiment_version,
        command_semantics_version=args.command_semantics_version,
        hand_mapping_version=args.hand_mapping_version,
    )
    episodes = discover_episodes(args.dataset_root)
    if args.mode in {"validate-data", "compute-norm-stats", "train"}:
        split = SplitName(args.split)
        episodes = filter_episodes_for_split(episodes, contract=contract, split=split)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    if args.mode == "compute-norm-stats" and not args.norm_legacy:
        return _run_compute_norm_stats_fast(
            episodes=episodes,
            spec=spec,
            contract=contract,
            provenance=provenance,
            args=args,
        )
    sample_index = build_sample_index(
        episodes,
        spec=spec,
        contract=contract,
        provenance=provenance,
        max_episodes=None,
    )
    if args.max_samples is not None:
        sample_index = sample_index[: args.max_samples]

    dataset = SharpaJoint29dDataset(
        episodes=episodes,
        sample_index=sample_index,
        spec=spec,
        contract=contract,
        provenance=provenance,
        require_reviewed=not args.allow_unreviewed_contract,
    )
    try:
        if args.mode == "validate-data":
            return _run_validate_data(dataset=dataset, contract=contract, args=args)
        if args.mode == "compute-norm-stats":
            return _run_compute_norm_stats(dataset=dataset, spec=spec, args=args)
        if args.mode == "train":
            return _run_train(dataset=dataset, spec=spec, contract=contract, args=args)
        raise ValueError(f"unsupported mode {args.mode!r}")
    finally:
        dataset.close()


def build_joint_spec_from_contract(
    contract: SharpaObservationContract,
    *,
    robot_id: str,
    embodiment_version: str,
    command_semantics_version: str,
    hand_mapping_version: str,
    clock_domain: str,
) -> BimanualActionSpec:
    """Build a joint_29d ``BimanualActionSpec`` aligned with the observation contract."""
    arm = tuple(f"left_arm_j{i}" for i in range(7))
    right_arm = tuple(f"right_arm_j{i}" for i in range(7))
    left_hand = tuple(f"left_hand_j{i}" for i in range(22))
    right_hand = tuple(f"right_hand_j{i}" for i in range(22))
    return BimanualActionSpec(
        physical_horizon=contract.physical_horizon,
        timebase=contract.timebase,
        control_frequency_hz=float(contract.control_frequency_hz),
        robot_id=robot_id,
        embodiment_version=embodiment_version,
        coordinate_frame=None,
        action_mode=ActionMode.ABSOLUTE,
        action_representation=ActionRepresentation.JOINT_29D,
        hand_normalization=HandNormalization.PER_HAND,
        rotation_6d_convention=None,
        kinematics_calibration_version=None,
        command_semantics_version=command_semantics_version,
        left_arm_joint_order=arm,
        right_arm_joint_order=right_arm,
        left_hand_joint_order=left_hand,
        right_hand_joint_order=right_hand,
        hand_mapping_version=hand_mapping_version,
        left_wrist_link=None,
        right_wrist_link=None,
        clock_domain=clock_domain,
        max_group_timestamp_skew_ms=float(contract.max_group_timestamp_skew_ms),
        max_alignment_timestamp_error_ms=float(contract.max_alignment_timestamp_error_ms),
        max_control_period_error_ms=float(contract.max_control_period_error_ms),
        max_observation_age_ms=50.0,
        max_command_lead_ms=25.0,
    )


def _run_validate_data(
    *, dataset: SharpaJoint29dDataset, contract: SharpaObservationContract, args: argparse.Namespace
) -> int:
    sample = dataset[0]
    _validate_sample_shapes(sample, contract=contract, physical_horizon=dataset.spec.physical_horizon)
    manifest = dataset_manifest(
        episodes=dataset.episodes,
        sample_index=dataset.sample_index,
        contract=contract,
    )
    output = {
        "mode": "validate-data",
        "manifest": manifest,
        "sample0": {
            "episode_id": sample["episode_id"],
            "start_aligned_frame": sample["start_aligned_frame"],
            "state_shape": list(sample["state"].shape),
            "left_actions_shape": list(sample["left_actions"].shape),
            "right_actions_shape": list(sample["right_actions"].shape),
            "image_shapes": {key: list(value.shape) for key, value in sample["image"].items()},
            "prompt": sample["prompt"][:120],
        },
    }
    _write_json(args.output_json, output)
    print(json.dumps(output, indent=2))
    return 0


def _run_compute_norm_stats_fast(
    *,
    episodes: Sequence,
    spec: BimanualActionSpec,
    contract: SharpaObservationContract,
    provenance: EpisodeActionProvenance,
    args: argparse.Namespace,
) -> int:
    asset_id, assets_root = _norm_assets_target(args)
    try:
        from openpi.shared import normalize as openpi_normalize

        from pi_dex.norm_compute import compute_joint29d_normalization_stats
        from pi_dex.norm_compute import resolve_norm_workers
    except ImportError as error:
        raise ImportError(
            "compute-norm-stats with OpenPI quantile assets requires the OpenPI stack "
            "(use the openpi uv env with editable pi-dex). Root conda alone is insufficient."
        ) from error

    workers = resolve_norm_workers(args.norm_workers)
    norm_stats, meta = compute_joint29d_normalization_stats(
        episodes,
        spec=spec,
        contract=contract,
        provenance=provenance,
        max_samples=args.max_samples,
        workers=workers,
        stride=args.norm_stride,
    )
    return _write_norm_stats_payload(
        args=args,
        spec=spec,
        asset_id=asset_id,
        assets_root=assets_root,
        norm_stats=norm_stats,
        extra=meta,
        save=openpi_normalize.save,
    )


def _run_compute_norm_stats(
    *, dataset: SharpaJoint29dDataset, spec: BimanualActionSpec, args: argparse.Namespace
) -> int:
    asset_id, assets_root = _norm_assets_target(args)
    try:
        from openpi.shared import normalize as openpi_normalize

        from openpi import transforms as openpi_transforms
        from pi_dex.openpi_integration import compute_bimanual_normalization_stats
    except ImportError as error:
        raise ImportError(
            "compute-norm-stats with OpenPI quantile assets requires the OpenPI stack "
            "(use the openpi uv env with editable pi-dex). Root conda alone is insufficient."
        ) from error

    empty_group = openpi_transforms.Group()
    data_config = _make_data_config(
        repo_id=f"pi-dex/{asset_id}",
        asset_id=asset_id,
        norm_stats=None,
        use_quantile_norm=True,
        repack_transforms=empty_group,
        data_transforms=empty_group,
        model_transforms=empty_group,
    )
    norm_stats = compute_bimanual_normalization_stats(
        dataset,
        data_config,
        spec,
        max_samples=args.max_samples,
    )
    sample_count = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    return _write_norm_stats_payload(
        args=args,
        spec=spec,
        asset_id=asset_id,
        assets_root=assets_root,
        norm_stats=norm_stats,
        extra={"samples": sample_count, "path": "legacy_dataset"},
        save=openpi_normalize.save,
    )


def _norm_assets_target(args: argparse.Namespace) -> tuple[str, pathlib.Path]:
    asset_id = args.asset_id or "sharpa_joint_29d"
    assets_root = pathlib.Path(args.assets_dir) if args.assets_dir else None
    if assets_root is None and args.output_json:
        assets_root = pathlib.Path(args.output_json).resolve().parent / "assets"
    if assets_root is None:
        raise ValueError("compute-norm-stats requires --assets-dir or --output-json to place assets/")
    return asset_id, assets_root


def _write_norm_stats_payload(
    *,
    args: argparse.Namespace,
    spec: BimanualActionSpec,
    asset_id: str,
    assets_root: pathlib.Path,
    norm_stats: Any,
    extra: Mapping[str, Any],
    save: Any,
) -> int:
    target = assets_root / asset_id
    target.mkdir(parents=True, exist_ok=True)
    save(target, norm_stats)
    payload = {
        "mode": "compute-norm-stats",
        "asset_id": asset_id,
        "assets_dir": str(assets_root),
        "norm_stats_path": str(target / "norm_stats.json"),
        "logical_action_dim": spec.logical_action_dim,
        **extra,
    }
    _write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2))
    return 0


def _run_train(
    *,
    dataset: SharpaJoint29dDataset,
    spec: BimanualActionSpec,
    contract: SharpaObservationContract,
    args: argparse.Namespace,
) -> int:
    if not args.pytorch_weight_path:
        raise ValueError(
            "train mode requires --pytorch-weight-path pointing at a converted pi05_base "
            "model.safetensors directory (handoff phase 2). Random init is forbidden."
        )
    converted_base = require_converted_base_dir(args.pytorch_weight_path)
    if not args.checkpoint_dir:
        raise ValueError("train mode requires --checkpoint-dir for atomic publish")
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)

    try:
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from openpi.shared import normalize as openpi_normalize
        from openpi.training.config import ModelTransformFactory
        import torch
        from torch.nn.parallel import DistributedDataParallel as DDP

        from openpi import transforms as openpi_transforms
        from pi_dex.distributed import barrier
        from pi_dex.distributed import cleanup_process_group
        from pi_dex.distributed import device_for_rank
        from pi_dex.distributed import init_process_group
        from pi_dex.distributed import is_main_process
        from pi_dex.distributed import launched_under_torch_distributed
        from pi_dex.distributed import unwrap_model
        from pi_dex.openpi_integration import BimanualDataConfigFactory
        from pi_dex.openpi_integration import create_pi05_model_config
        from pi_dex.openpi_integration import create_pytorch_data_loader_from_dataset
        from pi_dex.pytorch_training import PiDexPytorchTrainer
    except ImportError as error:
        raise ImportError(
            "train mode requires OpenPI + torch in the active interpreter "
            "(openpi uv env with editable pi-dex). "
            f"Import failed: {error}"
        ) from error

    use_distributed = bool(args.distributed) or launched_under_torch_distributed()
    rank = 0
    world_size = 1
    local_rank = 0
    if use_distributed:
        rank, world_size, local_rank = init_process_group()
        device = device_for_rank(local_rank=local_rank, requested=args.device)
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("train mode: --device cuda requested but torch.cuda.is_available() is False")

    if is_main_process(rank=rank):
        if not args.resume_from and checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            raise FileExistsError(
                f"checkpoint_dir already exists and is non-empty (refuse overwrite): {checkpoint_dir}"
            )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if use_distributed:
        barrier()

    try:
        return _run_train_body(
            dataset=dataset,
            spec=spec,
            contract=contract,
            args=args,
            converted_base=converted_base,
            checkpoint_dir=checkpoint_dir,
            device=device,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            use_distributed=use_distributed,
            PI0Pytorch=PI0Pytorch,
            openpi_normalize=openpi_normalize,
            ModelTransformFactory=ModelTransformFactory,
            torch=torch,
            DDP=DDP,
            openpi_transforms=openpi_transforms,
            BimanualDataConfigFactory=BimanualDataConfigFactory,
            create_pi05_model_config=create_pi05_model_config,
            create_pytorch_data_loader_from_dataset=create_pytorch_data_loader_from_dataset,
            PiDexPytorchTrainer=PiDexPytorchTrainer,
            unwrap_model=unwrap_model,
            is_main_process=is_main_process,
            barrier=barrier,
        )
    finally:
        if use_distributed:
            cleanup_process_group()


def _run_train_body(
    *,
    dataset: SharpaJoint29dDataset,
    spec: BimanualActionSpec,
    contract: SharpaObservationContract,
    args: argparse.Namespace,
    converted_base: pathlib.Path,
    checkpoint_dir: pathlib.Path,
    device: Any,
    rank: int,
    world_size: int,
    local_rank: int,
    use_distributed: bool,
    PI0Pytorch: Any,
    openpi_normalize: Any,
    ModelTransformFactory: Any,
    torch: Any,
    DDP: Any,
    openpi_transforms: Any,
    BimanualDataConfigFactory: Any,
    create_pi05_model_config: Any,
    create_pytorch_data_loader_from_dataset: Any,
    PiDexPytorchTrainer: Any,
    unwrap_model: Any,
    is_main_process: Any,
    barrier: Any,
) -> int:
    del local_rank  # used only for device selection upstream
    asset_id = args.asset_id or "sharpa_joint_29d"
    assets_root = pathlib.Path(args.assets_dir)
    if not assets_root.is_dir():
        raise FileNotFoundError(f"assets_dir: expected directory containing {asset_id}/norm_stats.json")
    norm_stats = openpi_normalize.load(assets_root / asset_id)

    if args.dtype == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("train mode: bfloat16 requested but unsupported on this GPU")

    model_config = create_pi05_model_config(spec, dtype=args.dtype, pytorch_compile_mode=None)
    if getattr(model_config, "dtype", None) != args.dtype:
        raise ValueError("model dtype must match --dtype (no silent rewrite)")

    model = PI0Pytorch(model_config).to(device)
    parent_provenance = load_verified_pi05_base(
        model,
        converted_base,
        expected_weights_sha256=args.expected_base_sha256 or None,
    )
    parameter_manifests = build_parameter_manifests(model)
    require_full_finetune_manifest(parameter_manifests)

    learning_rate = float(args.learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    global_step = 0
    run_id = args.run_id or f"joint29d-{uuid.uuid4().hex[:10]}"
    next_sample_index = 0
    sample_order = build_sample_order(len(dataset), seed=args.seed)
    if args.resume_from:
        train_state = resume_training_checkpoint(
            checkpoint_dir=args.resume_from,
            model=model,
            optimizer=optimizer,
        )
        global_step = int(train_state["global_step"])
        run_id = str(train_state.get("run_id", run_id))
        parent_provenance = dict(train_state.get("parent_base_provenance", parent_provenance))
        resumed_sampler = train_state["sampler_state"]
        if int(resumed_sampler["seed"]) != int(args.seed):
            raise ValueError(f"resume sampler seed {resumed_sampler['seed']} conflicts with --seed {args.seed}")
        if int(resumed_sampler["dataset_length"]) != len(dataset):
            raise ValueError(
                "resume sampler dataset_length conflicts with current dataset: "
                f"{resumed_sampler['dataset_length']} vs {len(dataset)}"
            )
        if order_sha256(sample_order) != resumed_sampler["order_sha256"]:
            raise ValueError("resume sampler order_sha256 mismatch for current seed/dataset length")
        next_sample_index = int(resumed_sampler["next_sample_index"])
        if int(resumed_sampler["batch_size"]) != int(args.batch_size):
            raise ValueError("resume sampler batch_size conflicts with --batch-size")
        resumed_world = int(resumed_sampler.get("world_size", 1))
        if resumed_world != world_size:
            raise ValueError(
                f"resume sampler world_size {resumed_world} conflicts with current world_size {world_size}"
            )

    if use_distributed:
        # pi0.5 training leaves a few expert/padding params unused in the FM loss;
        # match OpenPI's train_pytorch.py and enable unused-param detection.
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    trainer = PiDexPytorchTrainer(model, optimizer, spec, gradient_clip_norm=args.grad_clip_norm)

    empty_group = openpi_transforms.Group()
    pinned_data = _make_data_config(
        repo_id=f"pi-dex/{asset_id}",
        asset_id=asset_id,
        norm_stats=norm_stats,
        use_quantile_norm=True,
        repack_transforms=empty_group,
        data_transforms=empty_group,
        model_transforms=ModelTransformFactory()(model_config),
    )

    @dataclasses.dataclass(frozen=True)
    class _PinnedFactory:
        data_config: Any

        def create(self, assets_dirs: pathlib.Path, model_config_arg: Any) -> Any:
            del assets_dirs, model_config_arg
            return self.data_config

    data_factory = BimanualDataConfigFactory(_PinnedFactory(pinned_data), spec)
    configured = data_factory.create(assets_root, model_config)
    data_factory = BimanualDataConfigFactory(_PinnedFactory(configured), spec)

    train_dataset = _OrderedModelInputDataset(
        dataset,
        order=sample_order,
        start_index=next_sample_index,
    )
    remaining = len(train_dataset)
    if remaining == 0:
        raise ValueError("train cursor exhausted: next_sample_index is at end of ordered dataset")

    local_batch_size = int(args.batch_size)
    global_batch_size = local_batch_size * world_size
    if use_distributed:
        samples_per_rank = _distributed_sampler_num_samples(remaining, world_size=world_size)
        planned_batches = samples_per_rank // local_batch_size
    else:
        planned_batches = remaining // local_batch_size
    if args.max_steps is not None:
        planned_batches = min(planned_batches, args.max_steps)
    if planned_batches <= 0:
        raise ValueError("no full batches remain after resume cursor / batch_size / world_size")

    loader = create_pytorch_data_loader_from_dataset(
        train_dataset,
        data_factory,
        assets_root,
        model_config,
        spec,
        batch_size=local_batch_size,
        shuffle=False,
        num_workers=0,
        seed=args.seed,
        num_batches=planned_batches,
        skip_norm_stats=False,
        distributed=use_distributed,
    )

    wandb_run = None
    if is_main_process(rank=rank):
        from pi_dex.wandb_run import WANDB_ID_FILENAME

        resume_wandb = bool(args.resume_from) and (checkpoint_dir / WANDB_ID_FILENAME).is_file()
        wandb_run = init_train_wandb(
            enabled=bool(args.wandb),
            project=str(args.wandb_project),
            entity=str(args.wandb_entity) or None,
            run_name=str(args.wandb_run_name or run_id),
            config={
                "run_id": run_id,
                "asset_id": asset_id,
                "dataset_root": args.dataset_root,
                "split": args.split,
                "batch_size": local_batch_size,
                "global_batch_size": global_batch_size,
                "world_size": world_size,
                "learning_rate": learning_rate,
                "max_steps": args.max_steps,
                "save_interval": args.save_interval,
                "log_interval": args.log_interval,
                "dtype": args.dtype,
                "seed": args.seed,
                "robot_id": args.robot_id,
            },
            run_dir=checkpoint_dir,
            resume=resume_wandb,
        )

    losses: list[float] = []
    log_buffer: list[dict[str, float]] = []
    samples_consumed_local = 0
    published: pathlib.Path | None = None
    last_saved_step = -1
    log_interval = max(1, int(args.log_interval))
    save_interval = int(args.save_interval)

    try:
        for batch_observation, batch_actions in loader:
            observation = _tree_to_device(batch_observation, device)
            actions = batch_actions.to(device=device, dtype=torch.float32)
            result = trainer.train_step(observation, actions)
            global_step += 1
            samples_consumed_local += local_batch_size
            loss_value = float(result.loss.detach().cpu())
            losses.append(loss_value)

            grad_norm_value: float | None = None
            if result.gradient_norm is not None:
                grad_tensor = result.gradient_norm
                if hasattr(grad_tensor, "detach"):
                    grad_norm_value = float(grad_tensor.detach().cpu())
                else:
                    grad_norm_value = float(grad_tensor)

            if is_main_process(rank=rank):
                entry: dict[str, float] = {"loss": loss_value}
                if grad_norm_value is not None:
                    entry["grad_norm"] = grad_norm_value
                log_buffer.append(entry)
                if global_step % log_interval == 0:
                    avg_loss = sum(item["loss"] for item in log_buffer) / len(log_buffer)
                    metrics: dict[str, Any] = {
                        "loss": avg_loss,
                        "learning_rate": learning_rate,
                        "samples": samples_consumed_local * world_size,
                    }
                    grad_vals = [item["grad_norm"] for item in log_buffer if "grad_norm" in item]
                    if grad_vals:
                        metrics["grad_norm"] = sum(grad_vals) / len(grad_vals)
                    log_train_metrics(wandb_run, metrics, step=global_step)
                    print(
                        f"step={global_step} loss={avg_loss:.6f} "
                        f"lr={learning_rate:.2e} world_size={world_size}",
                        flush=True,
                    )
                    log_buffer.clear()

            if should_save_checkpoint(
                global_step=global_step,
                save_interval=save_interval,
                is_final=False,
            ):
                if use_distributed:
                    barrier()
                published = _publish_train_step(
                    checkpoint_dir=checkpoint_dir,
                    model=model,
                    optimizer=optimizer,
                    unwrap_model=unwrap_model,
                    build_parameter_manifests=build_parameter_manifests,
                    spec=spec,
                    model_config=model_config,
                    norm_stats=norm_stats,
                    asset_id=asset_id,
                    global_step=global_step,
                    run_id=run_id,
                    parent_base_provenance=parent_provenance,
                    next_sample_index=next_sample_index + samples_consumed_local * world_size,
                    sample_order=sample_order,
                    dataset=dataset,
                    contract=contract,
                    args=args,
                    device=device,
                    learning_rate=learning_rate,
                    local_batch_size=local_batch_size,
                    global_batch_size=global_batch_size,
                    world_size=world_size,
                    rank=rank,
                    losses=losses,
                    is_main=is_main_process(rank=rank),
                )
                if published is not None:
                    last_saved_step = global_step
                    log_train_metrics(wandb_run, {"checkpoint_step": global_step}, step=global_step)
                if use_distributed:
                    barrier()

        samples_consumed_global = samples_consumed_local * world_size
        next_sample_index += samples_consumed_global

        if use_distributed:
            barrier()

        if should_save_checkpoint(
            global_step=global_step,
            save_interval=save_interval,
            is_final=True,
        ) and global_step != last_saved_step:
            published = _publish_train_step(
                checkpoint_dir=checkpoint_dir,
                model=model,
                optimizer=optimizer,
                unwrap_model=unwrap_model,
                build_parameter_manifests=build_parameter_manifests,
                spec=spec,
                model_config=model_config,
                norm_stats=norm_stats,
                asset_id=asset_id,
                global_step=global_step,
                run_id=run_id,
                parent_base_provenance=parent_provenance,
                next_sample_index=next_sample_index,
                sample_order=sample_order,
                dataset=dataset,
                contract=contract,
                args=args,
                device=device,
                learning_rate=learning_rate,
                local_batch_size=local_batch_size,
                global_batch_size=global_batch_size,
                world_size=world_size,
                rank=rank,
                losses=losses,
                is_main=is_main_process(rank=rank),
            )
            if published is not None:
                last_saved_step = global_step
                log_train_metrics(wandb_run, {"checkpoint_step": global_step}, step=global_step)

        if is_main_process(rank=rank):
            if log_buffer:
                avg_loss = sum(item["loss"] for item in log_buffer) / len(log_buffer)
                log_train_metrics(
                    wandb_run,
                    {
                        "loss": avg_loss,
                        "learning_rate": learning_rate,
                        "samples": samples_consumed_local * world_size,
                    },
                    step=global_step,
                )
            payload = {
                "mode": "train",
                "run_dir": str(checkpoint_dir),
                "checkpoint_dir": str(published) if published is not None else None,
                "global_step": global_step,
                "run_id": run_id,
                "loss_last": losses[-1] if losses else None,
                "world_size": world_size,
                "global_batch_size": global_batch_size,
                "save_interval": save_interval,
                "wandb": bool(args.wandb),
                "parent_base": parent_provenance,
            }
            _write_json(args.output_json, payload)
            print(json.dumps(payload, indent=2))
    finally:
        finish_train_wandb(wandb_run)

    if use_distributed:
        barrier()
    return 0


def _publish_train_step(
    *,
    checkpoint_dir: pathlib.Path,
    model: Any,
    optimizer: Any,
    unwrap_model: Any,
    build_parameter_manifests: Any,
    spec: BimanualActionSpec,
    model_config: Any,
    norm_stats: Any,
    asset_id: str,
    global_step: int,
    run_id: str,
    parent_base_provenance: Mapping[str, Any],
    next_sample_index: int,
    sample_order: Sequence[int],
    dataset: SharpaJoint29dDataset,
    contract: SharpaObservationContract,
    args: argparse.Namespace,
    device: Any,
    learning_rate: float,
    local_batch_size: int,
    global_batch_size: int,
    world_size: int,
    rank: int,
    losses: Sequence[float],
    is_main: bool,
) -> pathlib.Path | None:
    """Publish ``checkpoint_dir/<global_step>/`` on the main process only."""
    if not is_main:
        return None
    unwrapped = unwrap_model(model)
    parameter_manifests = build_parameter_manifests(unwrapped)
    sampler_state = {
        "seed": int(args.seed),
        "dataset_length": len(dataset),
        "order_sha256": order_sha256(sample_order),
        "next_sample_index": int(next_sample_index),
        "batch_size": int(local_batch_size),
        "world_size": int(world_size),
        "global_batch_size": int(global_batch_size),
    }
    step_dir = checkpoint_dir / str(global_step)
    return publish_training_checkpoint(
        publish_dir=step_dir,
        model=unwrapped,
        optimizer=optimizer,
        spec=spec,
        model_config=model_config,
        norm_stats=norm_stats,
        asset_id=asset_id,
        global_step=global_step,
        run_id=run_id,
        action_representation=spec.action_representation.value,
        parent_base_provenance=parent_base_provenance,
        sampler_state=sampler_state,
        parameter_manifests=parameter_manifests,
        extra_train_state={
            "contract_id": contract.contract_id,
            "split": args.split,
            "split_manifest": split_manifest(dataset.episodes, contract=contract),
            "dataset_manifest": dataset_manifest(
                episodes=dataset.episodes,
                sample_index=dataset.sample_index,
                contract=contract,
            ),
            "dtype": args.dtype,
            "device": str(device),
            "learning_rate": learning_rate,
            "batch_size": local_batch_size,
            "global_batch_size": global_batch_size,
            "world_size": world_size,
            "rank": rank,
            "seed": args.seed,
            "final_losses": list(losses[-16:]),
            "all_parameters_sha256": parameter_manifests["all_parameters_sha256"],
            "trainable_parameters_sha256": parameter_manifests["trainable_parameters_sha256"],
        },
    )


def _distributed_sampler_num_samples(dataset_length: int, *, world_size: int) -> int:
    """Mirror ``torch.utils.data.distributed.DistributedSampler`` with ``drop_last=True``."""
    import math

    if world_size <= 0:
        raise ValueError(f"world_size: expected positive int, got {world_size}")
    if dataset_length % world_size != 0:
        return math.ceil((dataset_length - world_size) / world_size)
    return math.ceil(dataset_length / world_size)


def _run_synthetic_smoke(
    *, args: argparse.Namespace, spec: BimanualActionSpec, contract: SharpaObservationContract
) -> int:
    dataset = SyntheticJoint29dDataset(
        physical_horizon=spec.physical_horizon,
        state_dim=contract.state_dim,
        length=max(4, args.max_samples or 4),
    )
    sample = dataset[0]
    _validate_sample_shapes(sample, contract=contract, physical_horizon=spec.physical_horizon)
    payload = {
        "mode": "synthetic-smoke",
        "samples": len(dataset),
        "state_dim": contract.state_dim,
        "physical_horizon": spec.physical_horizon,
        "logical_action_dim": spec.logical_action_dim,
    }
    _write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2))
    return 0


def _make_data_config(**kwargs: Any) -> Any:
    @dataclasses.dataclass(frozen=True)
    class _DataConfig:
        repo_id: str | None
        asset_id: str | None
        norm_stats: Any
        use_quantile_norm: bool
        repack_transforms: Any
        data_transforms: Any
        model_transforms: Any

    return _DataConfig(**kwargs)


_MODEL_INPUT_KEYS = (
    "image",
    "image_mask",
    "state",
    "prompt",
    "left_actions",
    "right_actions",
)


@dataclasses.dataclass(frozen=True)
class _ModelInputDataset:
    """Drop dataset metadata fields that OpenPI's TorchDataLoader cannot tensorize."""

    dataset: Any

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        return {key: sample[key] for key in _MODEL_INPUT_KEYS}


@dataclasses.dataclass(frozen=True)
class _OrderedModelInputDataset:
    """Deterministic sample order with an absolute resume cursor."""

    dataset: Any
    order: tuple[int, ...]
    start_index: int = 0

    def __post_init__(self) -> None:
        if self.start_index < 0 or self.start_index > len(self.order):
            raise ValueError(f"start_index out of range: {self.start_index} for order length {len(self.order)}")

    def __len__(self) -> int:
        return len(self.order) - self.start_index

    def __getitem__(self, index: int) -> dict[str, Any]:
        if type(index) is not int:
            raise TypeError(f"index: expected int, got {type(index).__name__}")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        absolute = self.order[self.start_index + index]
        sample = self.dataset[absolute]
        return {key: sample[key] for key in _MODEL_INPUT_KEYS}


def _tree_to_device(tree: Any, device: Any) -> Any:
    import torch

    try:
        import jax

        return jax.tree.map(lambda value: value.to(device) if isinstance(value, torch.Tensor) else value, tree)
    except ImportError:
        pass
    if isinstance(tree, torch.Tensor):
        return tree.to(device)
    if dataclasses.is_dataclass(tree) and not isinstance(tree, type):
        return dataclasses.replace(
            tree,
            **{field.name: _tree_to_device(getattr(tree, field.name), device) for field in dataclasses.fields(tree)},
        )
    if isinstance(tree, Mapping):
        return type(tree)({key: _tree_to_device(value, device) for key, value in tree.items()})
    if isinstance(tree, tuple):
        return tuple(_tree_to_device(value, device) for value in tree)
    if isinstance(tree, list):
        return [_tree_to_device(value, device) for value in tree]
    return tree


def _validate_sample_shapes(
    sample: Mapping[str, Any], *, contract: SharpaObservationContract, physical_horizon: int
) -> None:
    import numpy as np

    if sample["state"].shape != (contract.state_dim,):
        raise ValueError(f"state shape: expected {(contract.state_dim,)}, got {sample['state'].shape}")
    expected = (physical_horizon, 29)
    if sample["left_actions"].shape != expected or sample["right_actions"].shape != expected:
        raise ValueError("hand action shapes mismatch")
    for key in OPENPI_IMAGE_KEYS:
        image = sample["image"][key]
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image[{key}]: expected HWC uint8 RGB, got {image.shape}/{image.dtype}")


def _write_json(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    output = pathlib.Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _parse_runner_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pi_dex.training_runner")
    parser.add_argument(
        "--mode",
        required=True,
        choices=("validate-data", "compute-norm-stats", "train", "synthetic-smoke"),
    )
    parser.add_argument("--observation-contract", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--pytorch-weight-path", default="")
    parser.add_argument("--expected-base-sha256", default="")
    parser.add_argument("--assets-dir", default="")
    parser.add_argument("--asset-id", default="sharpa_joint_29d")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1, help="Local per-rank batch size")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable DDP (also auto-enabled when RANK/WORLD_SIZE are set by torchrun)",
    )
    parser.add_argument(
        "--split",
        choices=tuple(name.value for name in SplitName),
        default=SplitName.TRAIN.value,
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--norm-workers",
        type=int,
        default=None,
        help="CPU processes for compute-norm-stats (default: NORM_WORKERS or min(cpu, 64))",
    )
    parser.add_argument(
        "--norm-stride",
        type=int,
        default=1,
        help="Keep every Nth valid start window when computing norm stats (default 1)",
    )
    parser.add_argument(
        "--norm-legacy",
        action="store_true",
        help="Use the slow Dataset+image path for compute-norm-stats (debug only)",
    )
    parser.add_argument("--allow-unreviewed-contract", action="store_true")
    parser.add_argument(
        "--allow-unreviewed-train-smoke",
        action="store_true",
        help="With --allow-unreviewed-contract, permit train for pipeline smoke only",
    )
    parser.add_argument("--robot-id", default="POC22027")
    parser.add_argument("--embodiment-version", default="sharpa_north_v1")
    parser.add_argument(
        "--command-semantics-version",
        default="sharpa_sdk_commanded_joint_position_absolute_v1",
    )
    parser.add_argument("--hand-mapping-version", default="sharpa_north_hand_mapping_v1")
    parser.add_argument("--clock-domain", default="unix_realtime")
    parser.add_argument(
        "--save-interval",
        type=int,
        default=500,
        help="Save checkpoint_dir/<step>/ every N steps (also always saves the final step). 0=final only",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Average and log loss to stdout/wandb every N steps",
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log train loss to Weights & Biases (rank 0). Requires WANDB_API_KEY. Use --no-wandb to disable",
    )
    parser.add_argument("--wandb-project", default="pi-dex")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-run-name", default="", help="Defaults to --run-id")
    args = parser.parse_args(list(argv))
    if args.mode in {"validate-data", "compute-norm-stats", "train"} and not args.dataset_root:
        raise ValueError(f"--dataset-root is required for mode {args.mode}")
    if args.mode == "train" and not args.assets_dir:
        raise ValueError("train mode requires --assets-dir with precomputed norm stats")
    if args.save_interval < 0:
        raise ValueError("--save-interval must be >= 0")
    if args.log_interval <= 0:
        raise ValueError("--log-interval must be >= 1")
    if args.norm_stride < 1:
        raise ValueError("--norm-stride must be >= 1")
    if args.norm_workers is not None and args.norm_workers < 1:
        raise ValueError("--norm-workers must be >= 1")
    return args
