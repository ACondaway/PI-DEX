"""PyTorch training core for PI-DEX on top of OpenPI's pi05 model.

The upstream PyTorch model returns an elementwise flow-matching loss with shape
``[B, 2 * K, 32]``. This module keeps the pretrained 32D projections and state
dict unchanged while making the nonsemantic padding channel neutral:

* the target action and sampled noise are forced to zero in invalid dimensions;
* the loss is averaged over the 31 semantic dimensions only.

The caller remains responsible for DDP setup, device transfer, scheduling,
checkpointing, and constructing an OpenPI-compatible observation object.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import torch
from torch import Tensor
from torch import nn

from pi_dex.actions import LOGICAL_ACTION_DIM
from pi_dex.actions import MODEL_ACTION_DIM
from pi_dex.spec import BimanualActionSpec
from pi_dex.training_contract import openpi_model_contract_metadata


@dataclasses.dataclass(frozen=True)
class TrainingStepResult:
    """Detached tensor-valued results from one optimizer update.

    Attributes:
        loss: Detached scalar mean over batch, horizon, and the 31 semantic
            action dimensions. It remains on the training device.
        gradient_norm: Scalar norm returned by PyTorch clipping, or ``None`` when
            clipping was disabled. A tensor result is detached and remains on the
            training device.
    """

    loss: Tensor
    gradient_norm: Tensor | float | None


def neutralize_openpi_dense_action_io(
    model: nn.Module,
    spec: BimanualActionSpec,
) -> None:
    """Project the loaded pi05 action I/O layers onto the 31D subspace.

    This serving-time projection changes only parameters that multiply or emit
    the nonsemantic 32nd action value. Training always presents zero in that
    input column and excludes the corresponding output row from the loss, so
    zeroing them makes OpenPI's stock denoising sampler padding-neutral without
    changing any semantic checkpoint parameter.

    Args:
        model: Loaded OpenPI ``PI0Pytorch`` model, optionally DDP-wrapped.
        spec: Action contract matching the model configuration.

    Raises:
        TypeError: If the model does not expose compatible ``nn.Linear`` action
            projections.
        ValueError: If projection shapes or the model contract are incompatible.
    """
    validated_spec = _validated_spec_copy(spec)
    unwrapped_model = _unwrap_model(model)
    openpi_model_contract_metadata(unwrapped_model.config, validated_spec)
    action_input_projection = getattr(unwrapped_model, "action_in_proj", None)
    action_output_projection = getattr(unwrapped_model, "action_out_proj", None)
    if not isinstance(action_input_projection, nn.Linear):
        raise TypeError("model.action_in_proj: expected torch.nn.Linear")
    if not isinstance(action_output_projection, nn.Linear):
        raise TypeError("model.action_out_proj: expected torch.nn.Linear")
    if action_input_projection.in_features != MODEL_ACTION_DIM:
        raise ValueError(
            "model.action_in_proj.in_features: "
            f"expected {MODEL_ACTION_DIM}, got {action_input_projection.in_features}"
        )
    if action_output_projection.out_features != MODEL_ACTION_DIM:
        raise ValueError(
            "model.action_out_proj.out_features: "
            f"expected {MODEL_ACTION_DIM}, got {action_output_projection.out_features}"
        )
    if action_output_projection.bias is None:
        raise ValueError("model.action_out_proj.bias: expected a bias tensor")

    with torch.no_grad():
        action_input_projection.weight[:, LOGICAL_ACTION_DIM:] = 0
        action_output_projection.weight[LOGICAL_ACTION_DIM:, :] = 0
        action_output_projection.bias[LOGICAL_ACTION_DIM:] = 0


def validate_model_action_batch(actions: Tensor, spec: BimanualActionSpec) -> None:
    """Validate a collated PI-DEX target batch at the model boundary.

    Args:
        actions: PyTorch float32 tensor with shape ``[B, 2 * K, 32]`` on any
            device. The sequence order is left then right for each physical step.
        spec: Semantic action contract defining ``K``.

    Raises:
        TypeError: If ``actions`` is not a tensor or is not float32.
        ValueError: If its rank, batch size, horizon, or action width is invalid.

    Notes:
        This hot-path validation intentionally does not inspect tensor values,
        which would synchronize accelerator execution. ``PackBimanualActions``
        establishes the zero-padding invariant before collation, and
        ``neutralize_model_padding`` enforces it without a host readback.
    """
    validated_spec = _validated_spec_copy(spec)
    if not isinstance(actions, Tensor):
        raise TypeError(f"actions: expected torch.Tensor, got {type(actions).__name__}")
    if actions.ndim != 3:
        raise ValueError(f"actions.ndim: expected 3 for [B, 2K, 32], got {actions.ndim} with shape {actions.shape}")
    if actions.shape[0] <= 0:
        raise ValueError(f"actions.shape[0]: expected a positive batch size, got {actions.shape[0]}")
    if actions.shape[1] != validated_spec.model_action_horizon:
        raise ValueError(
            "actions.shape[1]: expected "
            f"{validated_spec.model_action_horizon} for 2K model horizon, got {actions.shape[1]}"
        )
    if actions.shape[2] != MODEL_ACTION_DIM:
        raise ValueError(f"actions.shape[2]: expected {MODEL_ACTION_DIM}, got {actions.shape[2]}")
    if actions.dtype != torch.float32:
        raise TypeError(f"actions.dtype: expected torch.float32, got {actions.dtype}")


def neutralize_model_padding(values: Tensor) -> Tensor:
    """Return a tensor with every nonsemantic action dimension set to zero.

    Args:
        values: Floating PyTorch tensor with final dimension 32, on any device.

    Returns:
        A new tensor with the same shape, dtype, and device. The 31 semantic
        values are unchanged and the final padding value is zero.

    Raises:
        TypeError: If ``values`` is not a floating PyTorch tensor.
        ValueError: If the final dimension is not 32.
    """
    _validate_elementwise_tensor(values, field_name="values")
    neutralized = values.clone()
    neutralized[..., LOGICAL_ACTION_DIM:] = 0
    return neutralized


def reduce_semantic_action_loss(elementwise_loss: Tensor) -> Tensor:
    """Average an elementwise loss over the 31 semantic dimensions only.

    Args:
        elementwise_loss: Floating tensor with shape ``[..., 32]``. For OpenPI
            pi05 training the full shape is ``[B, 2 * K, 32]``.

    Returns:
        A scalar tensor. Invalid dimensions are excluded before the mean, so the
        loss scale is not reduced by a factor of 31/32.

    Raises:
        TypeError: If the input is not a floating PyTorch tensor.
        ValueError: If it has no elements or its final dimension is not 32.
        FloatingPointError: If the reduced semantic loss is NaN or infinite.
    """
    _validate_elementwise_tensor(elementwise_loss, field_name="elementwise_loss")
    if elementwise_loss.numel() == 0:
        raise ValueError("elementwise_loss: expected at least one value")
    semantic_loss = elementwise_loss[..., :LOGICAL_ACTION_DIM].mean()
    if not torch.isfinite(semantic_loss).item():
        raise FloatingPointError("semantic action loss: expected a finite scalar before backward")
    return semantic_loss


def compute_semantic_flow_matching_loss(
    model: nn.Module,
    observation: Any,
    actions: Tensor,
    spec: BimanualActionSpec,
    *,
    noise: Tensor | None = None,
    time: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Compute PI-DEX's padding-neutral pi05 flow-matching loss.

    Args:
        model: OpenPI ``PI0Pytorch`` model or a DDP-wrapped instance. Its forward
            method must return elementwise loss with the same shape as ``actions``.
        observation: OpenPI PyTorch observation already moved to the model device.
        actions: Float32 tensor with shape ``[B, 2 * K, 32]`` on the model device.
        spec: Bimanual action contract matching the model config.
        noise: Optional explicit float32 noise tensor with the same shape and
            device as ``actions``. Its padding channel is neutralized in a copy.
        time: Optional explicit float32 diffusion times with shape ``[B]`` on
            the same device as ``actions``.
        generator: Optional PyTorch generator used only when ``noise`` is omitted.

    Returns:
        Scalar loss averaged across all semantic action values.

    Raises:
        TypeError: If tensor types or dtypes are invalid.
        ValueError: If model config, shapes, devices, or model output differ from
            the PI-DEX contract.
        FloatingPointError: If the semantic loss is NaN or infinite.
    """
    validated_spec = _validated_spec_copy(spec)
    validate_model_action_batch(actions, validated_spec)
    openpi_model_contract_metadata(_unwrap_model(model).config, validated_spec)
    if time is not None:
        _validate_matching_time(time, actions=actions)

    semantic_actions = neutralize_model_padding(actions)
    if noise is None:
        if generator is not None:
            _validate_generator_device(generator, actions=actions)
        noise = torch.randn(
            semantic_actions.shape,
            dtype=semantic_actions.dtype,
            device=semantic_actions.device,
            generator=generator,
        )
    else:
        _validate_matching_noise(noise, actions=semantic_actions)
    semantic_noise = neutralize_model_padding(noise)

    elementwise_loss = model(observation, semantic_actions, noise=semantic_noise, time=time)
    if not isinstance(elementwise_loss, Tensor):
        raise TypeError(f"model output: expected torch.Tensor, got {type(elementwise_loss).__name__}")
    if elementwise_loss.shape != semantic_actions.shape:
        raise ValueError(
            f"model output shape: expected {tuple(semantic_actions.shape)}, got {tuple(elementwise_loss.shape)}"
        )
    if elementwise_loss.dtype != semantic_actions.dtype:
        raise TypeError(f"model output dtype: expected {semantic_actions.dtype}, got {elementwise_loss.dtype}")
    if elementwise_loss.device != semantic_actions.device:
        raise ValueError(f"model output device: expected {semantic_actions.device}, got {elementwise_loss.device}")
    return reduce_semantic_action_loss(elementwise_loss)


class PiDexPytorchTrainer:
    """Perform optimizer steps without altering the underlying OpenPI model.

    The trainer is intentionally not an ``nn.Module``. Checkpoints must continue
    to save the original model (or unwrapped DDP module), so state-dict keys remain
    compatible with OpenPI's ``model.safetensors`` loader.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        spec: BimanualActionSpec,
        *,
        gradient_clip_norm: float | None,
    ) -> None:
        """Initialize the training-step adapter and validate model shape config.

        Args:
            model: OpenPI ``PI0Pytorch`` module or DDP-wrapped equivalent.
            optimizer: Optimizer bound to ``model`` parameters.
            spec: Action contract required to match ``model.config``.
            gradient_clip_norm: Positive finite global norm, or ``None`` to
                disable gradient clipping.

        Raises:
            TypeError: If model, optimizer, or clipping arguments have invalid
                types, or the model exposes no OpenPI config.
            ValueError: If the clipping norm or model config violates the
                PI-DEX contract.
        """
        if not isinstance(model, nn.Module):
            raise TypeError(f"model: expected torch.nn.Module, got {type(model).__name__}")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f"optimizer: expected torch.optim.Optimizer, got {type(optimizer).__name__}")
        validated_gradient_clip_norm: float | None = None
        if gradient_clip_norm is not None:
            if type(gradient_clip_norm) not in (int, float):
                raise TypeError(
                    "gradient_clip_norm: expected a real number or None, "
                    f"got {type(gradient_clip_norm).__name__}"
                )
            try:
                validated_gradient_clip_norm = float(gradient_clip_norm)
            except OverflowError:
                raise ValueError("gradient_clip_norm: expected a positive finite value or None") from None
            if not math.isfinite(validated_gradient_clip_norm) or validated_gradient_clip_norm <= 0:
                raise ValueError(
                    "gradient_clip_norm: expected a positive finite value or None, "
                    f"got {gradient_clip_norm}"
                )
        validated_spec = _validated_spec_copy(spec)
        openpi_model_contract_metadata(_unwrap_model(model).config, validated_spec)
        _validate_optimizer_parameters(model, optimizer)

        self._model = model
        self._optimizer = optimizer
        self._spec = validated_spec
        self._gradient_clip_norm = validated_gradient_clip_norm

    def train_step(
        self,
        observation: Any,
        actions: Tensor,
        *,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> TrainingStepResult:
        """Run forward, backward, optional clipping, and one optimizer update.

        Args:
            observation: OpenPI observation already resident on the model device.
            actions: Float32 tensor shaped ``[B, 2 * K, 32]`` on that device.
            noise: Optional deterministic flow-matching noise.
            time: Optional deterministic diffusion times.
            generator: Optional generator used to sample noise when omitted.

        Returns:
            Detached tensor-valued loss and optional gradient norm.

        Raises:
            TypeError: If model inputs have invalid types or dtypes.
            ValueError: If action/model shapes violate the declared contract.
            FloatingPointError: If the semantic loss is NaN or infinite.
            RuntimeError: If PyTorch detects a non-finite total gradient norm.
        """
        self._model.train()
        self._optimizer.zero_grad(set_to_none=True)
        loss = compute_semantic_flow_matching_loss(
            self._model,
            observation,
            actions,
            self._spec,
            noise=noise,
            time=time,
            generator=generator,
        )
        loss.backward()

        checked_gradient_norm = torch.nn.utils.clip_grad_norm_(
            self._model.parameters(),
            max_norm=(self._gradient_clip_norm if self._gradient_clip_norm is not None else math.inf),
            error_if_nonfinite=True,
        )
        gradient_norm: Tensor | float | None = (
            checked_gradient_norm if self._gradient_clip_norm is not None else None
        )
        self._optimizer.step()
        detached_gradient_norm = gradient_norm.detach() if isinstance(gradient_norm, Tensor) else gradient_norm
        return TrainingStepResult(loss=loss.detach(), gradient_norm=detached_gradient_norm)


def _validate_elementwise_tensor(values: Tensor, *, field_name: str) -> None:
    if not isinstance(values, Tensor):
        raise TypeError(f"{field_name}: expected torch.Tensor, got {type(values).__name__}")
    if values.ndim == 0:
        raise ValueError(f"{field_name}.shape[-1]: expected {MODEL_ACTION_DIM}, got no final dimension")
    if values.shape[-1] != MODEL_ACTION_DIM:
        raise ValueError(f"{field_name}.shape[-1]: expected {MODEL_ACTION_DIM}, got {values.shape[-1]}")
    if not values.is_floating_point():
        raise TypeError(f"{field_name}.dtype: expected a floating dtype, got {values.dtype}")


def _validate_matching_noise(noise: Tensor, *, actions: Tensor) -> None:
    if not isinstance(noise, Tensor):
        raise TypeError(f"noise: expected torch.Tensor, got {type(noise).__name__}")
    if noise.shape != actions.shape:
        raise ValueError(f"noise.shape: expected {tuple(actions.shape)}, got {tuple(noise.shape)}")
    if noise.dtype != actions.dtype:
        raise TypeError(f"noise.dtype: expected {actions.dtype}, got {noise.dtype}")
    if noise.device != actions.device:
        raise ValueError(f"noise.device: expected {actions.device}, got {noise.device}")


def _validate_matching_time(time: Tensor, *, actions: Tensor) -> None:
    if not isinstance(time, Tensor):
        raise TypeError(f"time: expected torch.Tensor, got {type(time).__name__}")
    expected_shape = (actions.shape[0],)
    if time.shape != expected_shape:
        raise ValueError(f"time.shape: expected {expected_shape}, got {tuple(time.shape)}")
    if time.dtype != torch.float32:
        raise TypeError(f"time.dtype: expected torch.float32, got {time.dtype}")
    if time.device != actions.device:
        raise ValueError(f"time.device: expected {actions.device}, got {time.device}")
    if not torch.all(torch.isfinite(time)).item():
        raise ValueError("time: expected finite diffusion values")
    if not torch.all((time >= 0.0) & (time <= 1.0)).item():
        raise ValueError("time: expected diffusion values in the closed interval [0, 1]")


def _validate_generator_device(generator: torch.Generator, *, actions: Tensor) -> None:
    if not isinstance(generator, torch.Generator):
        raise TypeError(f"generator: expected torch.Generator, got {type(generator).__name__}")
    generator_device = torch.device(generator.device)
    action_device = actions.device
    same_backend = generator_device.type == action_device.type
    same_index = generator_device.index in (None, action_device.index) or action_device.index is None
    if not same_backend or not same_index:
        raise ValueError(f"generator.device: expected {actions.device}, got {generator_device}")


def _unwrap_model(model: nn.Module) -> nn.Module:
    unwrapped_model = getattr(model, "module", model)
    if not hasattr(unwrapped_model, "config"):
        raise TypeError("model: expected an OpenPI model exposing a 'config' attribute")
    return unwrapped_model


def _validate_optimizer_parameters(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Require one optimizer entry for every trainable model parameter.

    Frozen parameters from the same model may remain in optimizer groups, which
    matches common OpenPI fine-tuning setup. Foreign parameters, duplicates, and
    omitted trainable parameters are rejected by object identity.
    """
    model_parameters = {id(parameter): parameter for parameter in model.parameters()}
    trainable_parameter_ids = {
        parameter_id
        for parameter_id, parameter in model_parameters.items()
        if parameter.requires_grad
    }
    optimizer_parameter_ids: set[int] = set()
    for group_index, parameter_group in enumerate(optimizer.param_groups):
        for parameter_index, parameter in enumerate(parameter_group.get("params", ())):
            parameter_id = id(parameter)
            if parameter_id not in model_parameters:
                raise ValueError(
                    "optimizer parameter groups: foreign parameter at "
                    f"group {group_index}, index {parameter_index}"
                )
            if parameter_id in optimizer_parameter_ids:
                raise ValueError(
                    "optimizer parameter groups: duplicate model parameter at "
                    f"group {group_index}, index {parameter_index}"
                )
            optimizer_parameter_ids.add(parameter_id)
    missing_parameter_ids = trainable_parameter_ids - optimizer_parameter_ids
    if missing_parameter_ids:
        raise ValueError(
            "optimizer parameter groups: omitted "
            f"{len(missing_parameter_ids)} trainable model parameter(s)"
        )


def _validated_spec_copy(spec: object) -> BimanualActionSpec:
    if not isinstance(spec, BimanualActionSpec):
        raise TypeError(f"spec: expected BimanualActionSpec, got {type(spec).__name__}")
    return dataclasses.replace(spec)
