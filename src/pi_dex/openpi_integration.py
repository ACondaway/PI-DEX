"""Narrow integration points with the vendored OpenPI snapshot.

Imports stay inside functions so the core PI-DEX action/deployment package does
not require the heavyweight OpenPI training stack at import time.
"""

from __future__ import annotations

import dataclasses
import pathlib
import shutil
import tempfile
from typing import Any

import numpy as np

from pi_dex.actions import LOGICAL_ACTION_DIM
from pi_dex.actions import MODEL_ACTION_DIM
from pi_dex.checkpoints import CHECKPOINT_METADATA_FILENAME
from pi_dex.checkpoints import MODEL_WEIGHTS_FILENAME
from pi_dex.checkpoints import NORMALIZATION_ASSET_FILENAME
from pi_dex.checkpoints import load_and_validate_training_contract
from pi_dex.checkpoints import validate_normalization_asset_id
from pi_dex.deployment import BimanualPolicyAdapter
from pi_dex.deployment import validate_execution_horizon
from pi_dex.openpi_transforms import PackBimanualActions
from pi_dex.openpi_transforms import UnpackBimanualActions
from pi_dex.openpi_transforms import ValidateBimanualSample
from pi_dex.spec import BimanualActionSpec
from pi_dex.spec import HandNormalization
from pi_dex.training_contract import OPENPI_MODEL_CONTRACT_KEY
from pi_dex.training_contract import openpi_model_contract_metadata

_SUPPORTED_MODEL_DTYPES = frozenset({"bfloat16", "float32"})
_SUPPORTED_GEMMA_VARIANTS = frozenset(
    {
        "dummy",
        "gemma_300m",
        "gemma_300m_lora",
        "gemma_2b",
        "gemma_2b_lora",
    }
)
_SUPPORTED_PYTORCH_COMPILE_MODES = frozenset(
    {
        "default",
        "max-autotune",
        "max-autotune-no-cudagraphs",
        "reduce-overhead",
    }
)


@dataclasses.dataclass(frozen=True)
class BimanualDataConfigFactory:
    """Decorate an OpenPI data factory with PI-DEX model-boundary transforms.

    Attributes:
        base_factory: Trusted OpenPI data-config factory exposing ``create``. It
            must honor the exact ``model_config`` passed to ``create``; tokenizer
            internals are not safely introspectable at this boundary.
        spec: Immutable PI-DEX action contract injected into every config.
    """

    base_factory: Any
    spec: BimanualActionSpec

    def __post_init__(self) -> None:
        validated_spec = _validated_spec_copy(self.spec)
        if not callable(getattr(self.base_factory, "create", None)):
            raise TypeError("base_factory.create: expected a callable")
        object.__setattr__(self, "spec", validated_spec)

    def create(self, assets_dirs: pathlib.Path, model_config: Any) -> Any:
        """Create and augment the underlying OpenPI ``DataConfig``.

        Args:
            assets_dirs: OpenPI assets root forwarded to the base factory.
            model_config: Model config validated against ``spec``.

        Returns:
            An OpenPI data config with the paired PI-DEX transforms installed.

        Raises:
            TypeError: If the returned config lacks required transform groups.
            ValueError: If model or transform semantics conflict with ``spec``.
        """
        openpi_model_contract_metadata(model_config, self.spec)
        _prevalidate_declared_factory_asset_id(self.base_factory)
        data_config = self.base_factory.create(assets_dirs, model_config)
        returned_asset_id = getattr(data_config, "asset_id", None)
        if returned_asset_id is not None:
            validate_normalization_asset_id(
                returned_asset_id,
                field_name="data_config.asset_id",
            )
        return configure_bimanual_data(data_config, model_config, self.spec)


@dataclasses.dataclass(frozen=True)
class _PinnedDataConfigFactory:
    """Return one validated data-config instance for one exact model/assets pair."""

    data_config: Any
    assets_dirs: pathlib.Path
    model_config: Any

    def create(self, assets_dirs: pathlib.Path, model_config: Any) -> Any:
        """Return the pinned config after rejecting a changed construction context.

        Args:
            assets_dirs: Assets root that must equal the root used to materialize
                ``data_config``.
            model_config: Exact model-config object used for initial validation.

        Returns:
            The same validated OpenPI ``DataConfig`` instance on every call.

        Raises:
            TypeError: If ``assets_dirs`` is not path-like.
            ValueError: If the assets root or model-config identity changed.
        """
        try:
            actual_assets_dirs = pathlib.Path(assets_dirs)
        except TypeError:
            raise TypeError(
                f"assets_dirs: expected a path-like value, got {type(assets_dirs).__name__}"
            ) from None
        if actual_assets_dirs != self.assets_dirs:
            raise ValueError(
                f"assets_dirs: expected validated path {self.assets_dirs}, got {actual_assets_dirs}"
            )
        if model_config is not self.model_config:
            raise ValueError("model_config: expected the exact instance used to validate the data config")
        return self.data_config


def create_pi05_model_config(
    spec: BimanualActionSpec,
    *,
    dtype: str = "bfloat16",
    paligemma_variant: str = "gemma_2b",
    action_expert_variant: str = "gemma_300m",
    max_token_len: int = 200,
    pytorch_compile_mode: str | None = "max-autotune",
) -> Any:
    """Create an OpenPI pi05 config with the exact PI-DEX action shape.

    The action projection remains 32D for pretrained checkpoint compatibility;
    only the runtime horizon changes to ``2 * physical_horizon``.

    Args:
        spec: PI-DEX contract defining the even model horizon ``2*K``.
        dtype: PyTorch model precision, either ``"bfloat16"`` or ``"float32"``.
        paligemma_variant: One supported vendored OpenPI Gemma backbone name.
        action_expert_variant: One supported vendored OpenPI Gemma expert name.
        max_token_len: Positive prompt token limit passed to OpenPI.
        pytorch_compile_mode: ``None`` or one of PyTorch's four modes supported by
            the vendored ``Pi0Config``.

    Returns:
        A validated OpenPI ``Pi0Config`` with ``pi05=True`` and action width 32.

    Raises:
        ImportError: If the vendored OpenPI model package is unavailable.
        TypeError: If ``spec`` or a config field has an invalid type.
        ValueError: If a value or the resulting OpenPI config violates ``spec``.
    """
    validated_spec = _validated_spec_copy(spec)
    _validate_choice(dtype, field_name="dtype", supported=_SUPPORTED_MODEL_DTYPES)
    _validate_choice(
        paligemma_variant,
        field_name="paligemma_variant",
        supported=_SUPPORTED_GEMMA_VARIANTS,
    )
    _validate_choice(
        action_expert_variant,
        field_name="action_expert_variant",
        supported=_SUPPORTED_GEMMA_VARIANTS,
    )
    if isinstance(max_token_len, bool) or not isinstance(max_token_len, int):
        raise TypeError(f"max_token_len: expected int, got {type(max_token_len).__name__}")
    if max_token_len <= 0:
        raise ValueError(f"max_token_len: expected a positive integer, got {max_token_len}")
    if pytorch_compile_mode is not None:
        _validate_choice(
            pytorch_compile_mode,
            field_name="pytorch_compile_mode",
            supported=_SUPPORTED_PYTORCH_COMPILE_MODES,
        )

    from openpi.models.pi0_config import Pi0Config

    model_config = Pi0Config(
        dtype=dtype,
        paligemma_variant=paligemma_variant,
        action_expert_variant=action_expert_variant,
        action_dim=MODEL_ACTION_DIM,
        action_horizon=validated_spec.model_action_horizon,
        max_token_len=max_token_len,
        pi05=True,
        discrete_state_input=True,
        pytorch_compile_mode=pytorch_compile_mode,
    )
    validated_spec.validate_openpi_model_config(model_config)
    return model_config


def configure_bimanual_train_config(train_config: Any, spec: BimanualActionSpec) -> Any:
    """Return an OpenPI ``TrainConfig`` carrying PI-DEX data and wire metadata.

    This does not make the stock OpenPI LeRobot loader bimanual-aware. Training
    must still use ``create_pytorch_data_loader_from_dataset`` with a custom
    K-physical-step dataset.

    Args:
        train_config: Dataclass-like OpenPI ``TrainConfig`` to copy and decorate.
        spec: PI-DEX action contract required by the model and policy metadata.

    Returns:
        A dataclass-replaced train config carrying the bimanual data factory and
        exact ``pi_dex`` metadata.

    Raises:
        TypeError: If the train config, policy metadata, or referenced config
            objects have invalid types.
        ValueError: If model, metadata, or an existing factory conflicts with
            ``spec``.
    """
    validated_spec = _validated_spec_copy(spec)
    if not dataclasses.is_dataclass(train_config) or isinstance(train_config, type):
        raise TypeError(
            "train_config: expected a dataclass instance with model, data, and policy_metadata fields"
        )
    for attribute in ("model", "data", "policy_metadata"):
        if not hasattr(train_config, attribute):
            raise TypeError(f"train_config: missing required attribute {attribute!r}")
    model_contract = openpi_model_contract_metadata(train_config.model, validated_spec)
    base_metadata = train_config.policy_metadata
    if base_metadata is None:
        base_metadata = {}
    if not isinstance(base_metadata, dict):
        raise TypeError(f"train_config.policy_metadata: expected dict or None, got {type(base_metadata).__name__}")
    metadata = dict(base_metadata)
    if "pi_dex" in metadata:
        validated_spec.validate_metadata(metadata)
    if (
        OPENPI_MODEL_CONTRACT_KEY in metadata
        and metadata[OPENPI_MODEL_CONTRACT_KEY] != model_contract
    ):
        raise ValueError(
            "train_config.policy_metadata['openpi_model'] conflicts with the model config"
        )
    metadata["pi_dex"] = validated_spec.to_metadata()
    metadata[OPENPI_MODEL_CONTRACT_KEY] = model_contract

    data_factory = train_config.data
    if isinstance(data_factory, BimanualDataConfigFactory):
        if data_factory.spec != validated_spec:
            raise ValueError("train_config.data already carries a different PI-DEX action spec")
    else:
        data_factory = BimanualDataConfigFactory(data_factory, validated_spec)
    return dataclasses.replace(train_config, data=data_factory, policy_metadata=metadata)


def create_bimanual_trained_policy(
    train_config: Any,
    checkpoint_dir: pathlib.Path | str,
    spec: BimanualActionSpec,
    *,
    execution_horizon: int | None = None,
    default_prompt: str | None = None,
    pytorch_device: str | None = None,
    sample_kwargs: dict[str, Any] | None = None,
) -> BimanualPolicyAdapter:
    """Load an OpenPI PyTorch checkpoint and expose paired PI-DEX chunks.

    The decorated data config is materialized and validated exactly once, then a
    pinned factory returns that same instance when upstream constructs the policy.
    This prevents a stateful factory from changing transforms or normalization
    mode between PI-DEX validation and actual policy creation.

    Args:
        train_config: OpenPI training config. It is decorated with PI-DEX data
            transforms and metadata before policy creation.
        checkpoint_dir: OpenPI-compatible directory containing
            ``model.safetensors`` and per-hand normalization assets.
        spec: Contract that must match the model, stats, and checkpoint metadata.
        execution_horizon: Leading physical steps to expose, in ``[1, K]``.
        default_prompt: Optional prompt injected by OpenPI.
        pytorch_device: Explicit PyTorch device, or OpenPI's automatic choice.
        sample_kwargs: Optional arguments such as the denoising step count.

    Returns:
        A stateful wire adapter whose results contain paired float32 chunks under
        ``actions.left/right`` and a monotonic per-instance chunk identifier.

    Raises:
        ImportError: If the vendored OpenPI policy stack is unavailable.
        FileNotFoundError: If the resolved checkpoint has no PyTorch
            ``model.safetensors`` file.
        TypeError: If configs, paths, metadata, asset identifiers, or
            normalization values have invalid types.
        ValueError: If model, data, normalization, checkpoint, or execution
            contracts do not match ``spec``.
        OSError: If checkpoint download or asset access fails.
    """
    validated_spec = _validated_spec_copy(spec)
    if not isinstance(checkpoint_dir, str | pathlib.Path):
        raise TypeError(
            "checkpoint_dir: expected str or pathlib.Path, "
            f"got {type(checkpoint_dir).__name__}"
        )
    checkpoint_reference = str(checkpoint_dir)
    validated_execution_horizon = validate_execution_horizon(execution_horizon, validated_spec)
    validated_sample_kwargs = _validate_pytorch_sample_kwargs(sample_kwargs)
    _validate_optional_nonempty_string(default_prompt, field_name="default_prompt")
    _validate_optional_nonempty_string(pytorch_device, field_name="pytorch_device")
    configured_train = configure_bimanual_train_config(train_config, validated_spec)
    model_contract = openpi_model_contract_metadata(configured_train.model, validated_spec)
    if model_contract["dtype"] != "bfloat16":
        raise ValueError(
            "train_config.model.dtype: vendored OpenPI policy loading currently supports "
            "PI-DEX deployment only with 'bfloat16' because it converts selected model parameters"
        )
    validated_assets_dirs = pathlib.Path(configured_train.assets_dirs)
    validated_model_config = configured_train.model
    declared_asset_id = _declared_factory_asset_id(configured_train.data.base_factory)
    if declared_asset_id is None:
        raise ValueError(
            "train_config.data: a declared assets.asset_id or repo_id is required "
            "before checkpoint-backed data configuration"
        )
    asset_id = validate_normalization_asset_id(
        declared_asset_id,
        field_name="train_config.data asset_id",
    )

    from openpi.policies import policy_config
    from openpi.shared import download
    from openpi.training import checkpoints as openpi_checkpoints
    from pi_dex.normalization import normalization_state_dim
    from pi_dex.normalization import validate_normalization_stats
    from pi_dex.pytorch_training import neutralize_openpi_dense_action_io

    local_checkpoint_dir = pathlib.Path(download.maybe_download(checkpoint_reference))
    pytorch_weight_path = local_checkpoint_dir / MODEL_WEIGHTS_FILENAME
    if not pytorch_weight_path.is_file():
        raise FileNotFoundError(f"PI-DEX requires a PyTorch checkpoint at {pytorch_weight_path}")
    with tempfile.TemporaryDirectory(prefix="pi-dex-checkpoint-") as snapshot_directory_name:
        snapshot_directory = pathlib.Path(snapshot_directory_name)
        _copy_pytorch_checkpoint_snapshot(
            local_checkpoint_dir,
            snapshot_directory,
            asset_id=asset_id,
        )
        norm_stats = openpi_checkpoints.load_norm_stats(snapshot_directory / "assets", asset_id)
        validate_normalization_stats(norm_stats, validated_spec, require_state=True)
        load_and_validate_training_contract(
            snapshot_directory,
            validated_spec,
            model_config=validated_model_config,
            norm_stats=norm_stats,
            asset_id=asset_id,
        )
        data_config = configured_train.data.create(validated_assets_dirs, validated_model_config)
        if getattr(data_config, "use_quantile_norm", None) is not True:
            raise ValueError("data_config.use_quantile_norm: PI-DEX pi0.5 deployment requires True")
        if data_config.asset_id is None:
            raise ValueError("data_config.asset_id: required to load checkpoint normalization stats")
        materialized_asset_id = validate_normalization_asset_id(
            data_config.asset_id,
            field_name="data_config.asset_id",
        )
        if materialized_asset_id != asset_id:
            raise ValueError(
                "data_config.asset_id conflicts with the identifier declared before checkpoint validation"
            )
        data_config = configure_bimanual_data(
            data_config,
            validated_model_config,
            validated_spec,
            state_dim=normalization_state_dim(norm_stats, validated_spec),
        )
        configured_train = dataclasses.replace(
            configured_train,
            data=_PinnedDataConfigFactory(
                data_config=data_config,
                assets_dirs=validated_assets_dirs,
                model_config=validated_model_config,
            ),
        )
        openpi_policy = policy_config.create_trained_policy(
            configured_train,
            snapshot_directory,
            sample_kwargs=validated_sample_kwargs,
            default_prompt=default_prompt,
            norm_stats=norm_stats,
            pytorch_device=pytorch_device,
        )
        loaded_model = getattr(openpi_policy, "_model", None)
        neutralize_openpi_dense_action_io(loaded_model, validated_spec)
        load_and_validate_training_contract(
            snapshot_directory,
            validated_spec,
            model_config=validated_model_config,
            norm_stats=norm_stats,
            asset_id=asset_id,
        )
        return BimanualPolicyAdapter(
            openpi_policy,
            validated_spec,
            execution_horizon=validated_execution_horizon,
        )


def configure_bimanual_data(
    data_config: Any,
    model_config: Any,
    spec: BimanualActionSpec,
    *,
    state_dim: int | None = None,
) -> Any:
    """Append PI-DEX packing transforms to an existing OpenPI data config.

    Args:
        data_config: OpenPI ``DataConfig`` whose model transforms already resize
            images, tokenize prompts, and pad state as appropriate for pi05.
        model_config: OpenPI ``Pi0Config`` with ``pi05=True``, ``action_dim=32``,
            and ``action_horizon=2*K``.
        spec: PI-DEX semantic action contract defining ``K``.
        state_dim: Optional exact state width. If omitted, it is derived from
            ``data_config.norm_stats`` when available; it may remain unbound only
            during first-pass normalization-statistics computation.

    Returns:
        A dataclass-replaced OpenPI ``DataConfig``. A final data transform checks
        unbatched ``state[D]`` and optional per-hand ``[K,31]`` targets before
        normalization. On training input, normalized targets are packed after
        existing model transforms. On inference output, model actions are
        unpacked before unnormalization.

    Raises:
        TypeError: If ``data_config`` does not expose an OpenPI-like transform
            group.
        ValueError: If the model config conflicts with ``spec``.

    Notes:
        The normalization stats must be defined over the 31D fields
        ``left_actions`` and ``right_actions``. Do not compute statistics over the
        packed 32D representation.
    """
    validated_spec = _validated_spec_copy(spec)
    openpi_model_contract_metadata(model_config, validated_spec)
    if state_dim is None and getattr(data_config, "norm_stats", None) is not None:
        from pi_dex.normalization import normalization_state_dim

        state_dim = normalization_state_dim(data_config.norm_stats, validated_spec)
    if state_dim is not None:
        if type(state_dim) is not int:
            raise TypeError(f"state_dim: expected int or None, got {type(state_dim).__name__}")
        if state_dim <= 0:
            raise ValueError(f"state_dim: expected a positive integer, got {state_dim}")
    _require_empty_output_transforms(data_config, group_name="data_transforms")
    _require_empty_output_transforms(data_config, group_name="repack_transforms")
    data_transforms = getattr(data_config, "data_transforms", None)
    if data_transforms is None or not callable(getattr(data_transforms, "push", None)):
        raise TypeError("data_config.data_transforms: expected an OpenPI transforms.Group")
    existing_data_inputs = tuple(getattr(data_transforms, "inputs", ()))
    validators = tuple(
        transform
        for transform in existing_data_inputs
        if isinstance(transform, ValidateBimanualSample)
    )
    if len(validators) > 1:
        raise ValueError("data_config.data_transforms: duplicated PI-DEX sample validator")
    if validators:
        validator = validators[0]
        if existing_data_inputs[-1] is not validator:
            raise ValueError(
                "data_config.data_transforms: PI-DEX sample validator must be the final input transform"
            )
        if validator.physical_horizon != validated_spec.physical_horizon:
            raise ValueError(
                "data_config.data_transforms: PI-DEX sample validator physical horizon conflicts with spec"
            )
        if state_dim is not None and validator.state_dim not in (None, state_dim):
            raise ValueError(
                "data_config.data_transforms: PI-DEX sample validator state width conflicts with stats"
            )
        if state_dim is not None and validator.state_dim is None:
            configured_inputs = (
                *existing_data_inputs[:-1],
                ValidateBimanualSample(validated_spec.physical_horizon, state_dim=state_dim),
            )
            configured_data_transforms = dataclasses.replace(
                data_transforms,
                inputs=configured_inputs,
            )
        else:
            configured_data_transforms = data_transforms
    else:
        configured_data_transforms = data_transforms.push(
            inputs=[
                ValidateBimanualSample(
                    validated_spec.physical_horizon,
                    state_dim=state_dim,
                )
            ]
        )

    model_transforms = getattr(data_config, "model_transforms", None)
    if model_transforms is None or not callable(getattr(model_transforms, "push", None)):
        raise TypeError("data_config.model_transforms: expected an OpenPI transforms.Group")

    existing_inputs = tuple(getattr(model_transforms, "inputs", ()))
    existing_outputs = tuple(getattr(model_transforms, "outputs", ()))
    pack_count = sum(isinstance(transform, PackBimanualActions) for transform in existing_inputs)
    unpack_count = sum(isinstance(transform, UnpackBimanualActions) for transform in existing_outputs)
    if pack_count == unpack_count == 1:
        if not isinstance(existing_inputs[-1], PackBimanualActions) or not (
            len(existing_outputs) == 1 and isinstance(existing_outputs[0], UnpackBimanualActions)
        ):
            raise ValueError(
                "data_config.model_transforms: PI-DEX pack/unpack must be the final input "
                "and sole output transform"
            )
        configured_model_transforms = model_transforms
    elif pack_count or unpack_count:
        raise ValueError(
            "data_config.model_transforms: incomplete or duplicated PI-DEX pack/unpack transform pair"
        )
    else:
        if existing_outputs:
            raise ValueError(
                "data_config.model_transforms.outputs: expected no output transforms before PI-DEX unpacking, "
                f"got {len(existing_outputs)}"
            )

        configured_model_transforms = model_transforms.push(
            inputs=[PackBimanualActions()],
            outputs=[UnpackBimanualActions()],
        )
    if configured_data_transforms is data_transforms and configured_model_transforms is model_transforms:
        return data_config
    return dataclasses.replace(
        data_config,
        data_transforms=configured_data_transforms,
        model_transforms=configured_model_transforms,
    )


def create_pytorch_data_loader_from_dataset(
    dataset: Any,
    data_factory: BimanualDataConfigFactory,
    assets_dirs: pathlib.Path | str,
    model_config: Any,
    spec: BimanualActionSpec,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
) -> Any:
    """Build OpenPI's PyTorch loader around a PI-DEX K-step dataset.

    Args:
        dataset: Random-access dataset yielding unbatched dictionaries. After its
            repack/data transforms, each sample must expose ``left_actions`` and
            ``right_actions`` as floating NumPy arrays with shape ``[K, 31]`` in
            the units and frame declared by ``spec``. The dataset—not OpenPI's
            standard LeRobot loader—owns physical-time horizon selection.
        data_factory: PI-DEX-decorated OpenPI data factory. Requiring the factory
            ensures transforms are materialized from this exact model config.
        assets_dirs: OpenPI configuration-assets root forwarded to the factory.
        model_config: Matching OpenPI pi05 model config.
        spec: PI-DEX action contract.
        batch_size: Positive local batch size passed to the OpenPI loader.
        shuffle: Whether to shuffle samples.
        num_workers: Non-negative number of spawned data-loader workers.
        seed: Integer PyTorch data-loader shuffle seed.
        num_batches: Optional positive number of batches before iteration stops.
        skip_norm_stats: Skip normalization only for explicit diagnostic/statistics
            workflows.

    Returns:
        OpenPI ``DataLoaderImpl`` yielding CPU PyTorch observations and actions
        with shape ``[B, 2 * K, 32]``. Action dtype is inherited from the
        transformed NumPy samples; the trainer boundary requires float32.

    Raises:
        ImportError: If PyTorch or the vendored OpenPI loader is unavailable.
        TypeError: If control arguments have ambiguous or invalid types.
        ValueError: If batch/worker limits, normalization mode, stats, or model
            config are invalid.
        NotImplementedError: If ``torch.distributed`` is already initialized or
            OpenPI detects more than one JAX process.

    Notes:
        This uses a small set of OpenPI internal loader classes because the stock
        LeRobot path interprets ``action_horizon`` as consecutive physical time
        steps and would incorrectly read ``2*K`` steps before interleaving.
    """
    validated_spec = _validated_spec_copy(spec)
    if not isinstance(data_factory, BimanualDataConfigFactory):
        raise TypeError(
            "data_factory: expected BimanualDataConfigFactory created by "
            "configure_bimanual_train_config"
        )
    if data_factory.spec != validated_spec:
        raise ValueError("data_factory.spec conflicts with the requested PI-DEX action spec")
    openpi_model_contract_metadata(model_config, validated_spec)
    try:
        validated_assets_dirs = pathlib.Path(assets_dirs)
    except TypeError:
        raise TypeError(
            f"assets_dirs: expected a path-like value, got {type(assets_dirs).__name__}"
        ) from None
    if not isinstance(shuffle, bool):
        raise TypeError(f"shuffle: expected bool, got {type(shuffle).__name__}")
    if not isinstance(skip_norm_stats, bool):
        raise TypeError(f"skip_norm_stats: expected bool, got {type(skip_norm_stats).__name__}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed: expected int, got {type(seed).__name__}")
    if isinstance(num_batches, bool) or (num_batches is not None and not isinstance(num_batches, int)):
        raise TypeError(f"num_batches: expected int or None, got {type(num_batches).__name__}")
    if num_batches is not None and num_batches <= 0:
        raise ValueError(f"num_batches: expected a positive integer or None, got {num_batches}")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError(f"batch_size: expected int, got {type(batch_size).__name__}")
    if batch_size <= 0:
        raise ValueError(f"batch_size: expected a positive integer, got {batch_size!r}")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int):
        raise TypeError(f"num_workers: expected int, got {type(num_workers).__name__}")
    if num_workers < 0:
        raise ValueError(f"num_workers: expected a non-negative integer, got {num_workers!r}")

    import torch

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise NotImplementedError(
            "initialized torch.distributed/DDP is not supported by the PI-DEX custom dataset loader"
        )

    configured_data = data_factory.create(validated_assets_dirs, model_config)

    if not skip_norm_stats:
        from pi_dex.normalization import validate_normalization_stats

        if configured_data.norm_stats is None:
            raise ValueError("data_config.norm_stats: required for PI-DEX PyTorch training")
        validate_normalization_stats(configured_data.norm_stats, validated_spec, require_state=True)
        if getattr(configured_data, "repo_id", None) == "fake":
            raise ValueError(
                "data_config.repo_id: 'fake' disables normalization inside OpenPI; "
                "use a non-sentinel id for PI-DEX training"
            )
        if getattr(configured_data, "use_quantile_norm", None) is not True:
            raise ValueError("data_config.use_quantile_norm: PI-DEX pi0.5 training requires True")

    from openpi.training import data_loader as openpi_data_loader

    transformed_dataset = openpi_data_loader.transform_dataset(
        dataset,
        configured_data,
        skip_norm_stats=skip_norm_stats,
    )

    loader = openpi_data_loader.TorchDataLoader(
        transformed_dataset,
        local_batch_size=batch_size,
        shuffle=shuffle,
        sampler=None,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework="pytorch",
    )
    return openpi_data_loader.DataLoaderImpl(configured_data, loader)


def compute_bimanual_normalization_stats(
    dataset: Any,
    data_config: Any,
    spec: BimanualActionSpec,
    *,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Compute OpenPI-compatible stats before padding and interleaving.

    Args:
        dataset: Random-access dataset yielding raw, unbatched dictionaries.
            After repack/data transforms, ``state`` must be a finite floating
            vector shaped ``[D]`` and each hand target is ``[K,31]``.
        data_config: OpenPI ``DataConfig``. Only repack and data input transforms
            are applied; normalization and model transforms are intentionally not.
        spec: Contract selecting per-hand or shared statistics.
        max_samples: Optional positive cap on dataset samples used.

    Returns:
        A mapping containing ``state``, ``left_actions``, and ``right_actions``
        OpenPI ``NormStats``. Every action statistic has exactly 31 values. In
        shared mode both hand keys reference equal pooled statistics.

    Raises:
        TypeError: If transformed samples or arrays have invalid types/dtypes.
        ValueError: If the dataset is empty, fields are absent, shapes are invalid,
            or too few vectors exist to compute statistics.

    Notes:
        This offline pass iterates samples in Python for a custom K-step dataset.
        The training hot path remains batched. Use OpenPI's ``normalize.save`` to
        write the returned mapping under the configured asset id.
    """
    validated_spec = _validated_spec_copy(spec)
    if max_samples is not None:
        if isinstance(max_samples, bool) or not isinstance(max_samples, int):
            raise TypeError(f"max_samples: expected int or None, got {type(max_samples).__name__}")
        if max_samples <= 0:
            raise ValueError(f"max_samples: expected a positive integer, got {max_samples}")
    try:
        dataset_length = len(dataset)
    except TypeError:
        raise TypeError("dataset: expected a sized random-access dataset") from None
    sample_count = dataset_length if max_samples is None else min(dataset_length, max_samples)
    if sample_count <= 0:
        raise ValueError("dataset: expected at least one sample")

    from openpi import transforms as openpi_transforms
    from openpi.shared import normalize

    transform = openpi_transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
        ]
    )
    state_stats = normalize.RunningStats()
    if validated_spec.hand_normalization is HandNormalization.PER_HAND:
        left_stats = normalize.RunningStats()
        right_stats = normalize.RunningStats()
    else:
        left_stats = right_stats = normalize.RunningStats()

    for sample_index in range(sample_count):
        transformed = transform(dataset[sample_index])
        if not isinstance(transformed, dict):
            raise TypeError(
                f"transformed sample {sample_index}: expected dict, got {type(transformed).__name__}"
            )
        state = _require_stats_array(
            transformed,
            "state",
            sample_index=sample_index,
            expected_width=None,
            expected_ndim=1,
        )
        left_actions = _require_stats_array(
            transformed,
            "left_actions",
            sample_index=sample_index,
            expected_shape=(validated_spec.physical_horizon, LOGICAL_ACTION_DIM),
        )
        right_actions = _require_stats_array(
            transformed,
            "right_actions",
            sample_index=sample_index,
            expected_shape=(validated_spec.physical_horizon, LOGICAL_ACTION_DIM),
        )
        if left_actions.dtype != right_actions.dtype:
            raise TypeError(
                f"transformed sample {sample_index} action dtypes: expected left/right to match, "
                f"got {left_actions.dtype} and {right_actions.dtype}"
            )
        state_stats.update(state)
        left_stats.update(left_actions)
        right_stats.update(right_actions)

    stats = {
        "state": state_stats.get_statistics(),
        "left_actions": left_stats.get_statistics(),
        "right_actions": right_stats.get_statistics(),
    }
    from pi_dex.normalization import validate_normalization_stats

    validate_normalization_stats(stats, validated_spec, require_state=True)
    return stats


def _require_stats_array(
    sample: dict[str, Any],
    field_name: str,
    *,
    sample_index: int,
    expected_width: int | None = None,
    expected_shape: tuple[int, ...] | None = None,
    expected_ndim: int | None = None,
) -> np.ndarray:
    if field_name not in sample:
        raise ValueError(f"transformed sample {sample_index}: missing required field {field_name!r}")
    value = sample[field_name]
    if not isinstance(value, np.ndarray):
        raise TypeError(
            f"transformed sample {sample_index} field {field_name!r}: "
            f"expected numpy.ndarray, got {type(value).__name__}"
        )
    if value.ndim == 0:
        raise ValueError(f"transformed sample {sample_index} field {field_name!r}: expected at least one axis")
    if expected_ndim is not None and value.ndim != expected_ndim:
        raise ValueError(
            f"transformed sample {sample_index} field {field_name!r} ndim: "
            f"expected {expected_ndim}, got {value.ndim} for shape {value.shape}"
        )
    if expected_shape is not None and value.shape != expected_shape:
        raise ValueError(
            f"transformed sample {sample_index} field {field_name!r} shape: "
            f"expected {expected_shape}, got {value.shape}"
        )
    if expected_width is not None and value.shape[-1] != expected_width:
        raise ValueError(
            f"transformed sample {sample_index} field {field_name!r} shape[-1]: "
            f"expected {expected_width}, got {value.shape[-1]}"
        )
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(
            f"transformed sample {sample_index} field {field_name!r} dtype: "
            f"expected floating, got {value.dtype}"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError(f"transformed sample {sample_index} field {field_name!r}: expected finite values")
    return value


def _require_empty_output_transforms(data_config: Any, *, group_name: str) -> None:
    transform_group = getattr(data_config, group_name, None)
    if transform_group is None or not hasattr(transform_group, "outputs"):
        raise TypeError(f"data_config.{group_name}: expected an OpenPI transforms.Group")
    outputs = tuple(transform_group.outputs)
    if outputs:
        raise ValueError(
            f"data_config.{group_name}.outputs: expected no output transforms before PI-DEX unpacking, "
            f"got {len(outputs)}"
        )


def _validated_spec_copy(spec: object) -> BimanualActionSpec:
    if not isinstance(spec, BimanualActionSpec):
        raise TypeError(f"spec: expected BimanualActionSpec, got {type(spec).__name__}")
    return dataclasses.replace(spec)


def _validate_choice(value: object, *, field_name: str, supported: frozenset[str]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name}: expected str, got {type(value).__name__}")
    if value not in supported:
        raise ValueError(f"{field_name}: expected one of {sorted(supported)!r}, got {value!r}")
    return value


def _validate_pytorch_sample_kwargs(
    sample_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    if sample_kwargs is None:
        return {}
    if type(sample_kwargs) is not dict:
        raise TypeError(
            f"sample_kwargs: expected dict or None, got {type(sample_kwargs).__name__}"
        )
    if not sample_kwargs:
        return {}
    if set(sample_kwargs) != {"num_steps"}:
        raise ValueError(
            "sample_kwargs: expected exactly the optional 'num_steps' field when non-empty"
        )
    num_steps = sample_kwargs["num_steps"]
    if type(num_steps) is not int:
        raise TypeError(
            f"sample_kwargs['num_steps']: expected int, got {type(num_steps).__name__}"
        )
    if not 1 <= num_steps <= 1_000:
        raise ValueError("sample_kwargs['num_steps']: expected an integer in [1, 1000]")
    return {"num_steps": num_steps}


def _validate_optional_nonempty_string(value: object, *, field_name: str) -> None:
    if value is None:
        return
    if type(value) is not str:
        raise TypeError(f"{field_name}: expected str or None, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{field_name}: expected a non-empty value or None")


def _copy_pytorch_checkpoint_snapshot(
    source_directory: pathlib.Path,
    destination_directory: pathlib.Path,
    *,
    asset_id: str,
) -> None:
    """Copy the exact files used by loading into one private validated snapshot."""
    destination_directory.chmod(0o700)
    source_files = {
        pathlib.Path(MODEL_WEIGHTS_FILENAME): source_directory / MODEL_WEIGHTS_FILENAME,
        pathlib.Path(CHECKPOINT_METADATA_FILENAME): source_directory / CHECKPOINT_METADATA_FILENAME,
        pathlib.Path("assets") / asset_id / NORMALIZATION_ASSET_FILENAME: (
            source_directory / "assets" / asset_id / NORMALIZATION_ASSET_FILENAME
        ),
    }
    for relative_path, source_path in source_files.items():
        if not source_path.is_file():
            raise FileNotFoundError(f"PI-DEX checkpoint artifact not found: {source_path}")
        destination_path = destination_directory / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination_path)


def _declared_factory_asset_id(base_factory: object) -> str | None:
    """Return the standard OpenPI asset identifier declared without factory I/O."""
    assets = getattr(base_factory, "assets", None)
    explicit_asset_id = getattr(assets, "asset_id", None)
    candidate = explicit_asset_id
    if candidate is None:
        repo_id = getattr(base_factory, "repo_id", None)
        candidate = repo_id if type(repo_id) is str else None
    return candidate


def _prevalidate_declared_factory_asset_id(base_factory: object) -> None:
    """Reject unsafe standard OpenPI asset identifiers before factory I/O."""
    candidate = _declared_factory_asset_id(base_factory)
    if candidate is not None:
        validate_normalization_asset_id(
            candidate,
            field_name="data factory asset_id",
        )
