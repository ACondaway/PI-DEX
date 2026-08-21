"""NumPy transforms at the PI-DEX/OpenPI action boundary.

These transforms deliberately use duck typing instead of importing OpenPI. They
can be inserted into an OpenPI ``transforms.Group`` while keeping the selected
logical-action/32D model layout contract owned by PI-DEX.
"""

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np

from pi_dex.core.actions import ActionRepresentation
from pi_dex.core.actions import deinterleave
from pi_dex.core.actions import interleave
from pi_dex.core.actions import pad_action
from pi_dex.core.actions import unpad_action

MODEL_ACTIONS_KEY = "actions"
LEFT_ACTIONS_KEY = "left_actions"
RIGHT_ACTIONS_KEY = "right_actions"


@dataclasses.dataclass(frozen=True)
class ValidateBimanualSample:
    """Validate one unbatched sample before normalization and tokenization.

    Args:
        physical_horizon: Exact number ``K`` of per-hand target steps.
        action_representation: Logical action layout used by both hand targets.
        state_dim: Exact state width bound by normalization stats, or ``None``
            only while computing those statistics for the first time.

    Training samples require floating ``state[D]`` and matching finite
    ``left_actions/right_actions[K, logical_action_dim]``. Inference observations
    may omit both target fields but still require a finite one-dimensional state
    vector.
    """

    physical_horizon: int
    action_representation: ActionRepresentation
    state_dim: int | None = None

    def __post_init__(self) -> None:
        if type(self.physical_horizon) is not int:
            raise TypeError(
                "physical_horizon: expected int, "
                f"got {type(self.physical_horizon).__name__}"
            )
        if self.physical_horizon <= 0:
            raise ValueError(
                f"physical_horizon: expected a positive integer, got {self.physical_horizon}"
            )
        if not isinstance(self.action_representation, ActionRepresentation):
            raise TypeError(
                "action_representation: expected ActionRepresentation, "
                f"got {type(self.action_representation).__name__}"
            )
        if self.state_dim is not None:
            if type(self.state_dim) is not int:
                raise TypeError(
                    f"state_dim: expected int or None, got {type(self.state_dim).__name__}"
                )
            if self.state_dim <= 0:
                raise ValueError(f"state_dim: expected a positive integer, got {self.state_dim}")

    def __call__(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Return a shallow sample copy after validating state and targets.

        Raises:
            TypeError: If the sample, state, or targets use invalid containers
                or non-floating dtypes.
            KeyError: If state or exactly one per-hand target is missing.
            ValueError: If shape, finite values, or target dtypes conflict.
        """
        mutable_data = _copy_mapping(data)
        state = _pop_required(mutable_data, "state")
        _validate_floating_array(
            state,
            field_name="state",
            expected_shape=None if self.state_dim is None else (self.state_dim,),
            expected_ndim=1,
        )
        mutable_data["state"] = state

        has_left_actions = LEFT_ACTIONS_KEY in mutable_data
        has_right_actions = RIGHT_ACTIONS_KEY in mutable_data
        if not has_left_actions and not has_right_actions:
            return mutable_data
        if has_left_actions != has_right_actions:
            missing_field = RIGHT_ACTIONS_KEY if has_left_actions else LEFT_ACTIONS_KEY
            raise KeyError(f"data: missing required field {missing_field!r}")

        expected_action_shape = (
            self.physical_horizon,
            self.action_representation.logical_action_dim,
        )
        left_actions = mutable_data[LEFT_ACTIONS_KEY]
        right_actions = mutable_data[RIGHT_ACTIONS_KEY]
        _validate_floating_array(
            left_actions,
            field_name=LEFT_ACTIONS_KEY,
            expected_shape=expected_action_shape,
            expected_ndim=2,
        )
        _validate_floating_array(
            right_actions,
            field_name=RIGHT_ACTIONS_KEY,
            expected_shape=expected_action_shape,
            expected_ndim=2,
        )
        if left_actions.dtype != right_actions.dtype:
            raise TypeError(
                "data left/right action dtypes must match; "
                f"got {left_actions.dtype} and {right_actions.dtype}"
            )
        return mutable_data


@dataclasses.dataclass(frozen=True)
class PackBimanualActions:
    """Pack normalized per-hand logical actions into OpenPI model actions.

    Input data must contain ``left_actions`` and ``right_actions`` as NumPy
    floating arrays with matching shape
    ``[..., K, action_representation.logical_action_dim]``. The transform
    removes those fields and adds ``actions`` with shape ``[..., 2 * K, 32]``
    in left-then-right order. Units and action semantics are preserved.
    """

    action_representation: ActionRepresentation

    def __post_init__(self) -> None:
        if not isinstance(self.action_representation, ActionRepresentation):
            raise TypeError(
                "action_representation: expected ActionRepresentation, "
                f"got {type(self.action_representation).__name__}"
            )

    def __call__(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Apply the training-side packing transform.

        When both target fields are absent, the transform is a no-op so the same
        OpenPI input pipeline can process inference observations. A pre-packed
        ``actions`` field is never accepted at this boundary.

        Raises:
            TypeError: If ``data`` is not a mapping or action values are not
                floating NumPy arrays with equal dtypes.
            KeyError: If either per-hand action field is missing.
            ValueError: If shapes differ or violate the logical action layout.
        """
        mutable_data = _copy_mapping(data)
        has_left_actions = LEFT_ACTIONS_KEY in mutable_data
        has_right_actions = RIGHT_ACTIONS_KEY in mutable_data
        if not has_left_actions and not has_right_actions:
            if MODEL_ACTIONS_KEY in mutable_data:
                raise ValueError(
                    f"data contains reserved field {MODEL_ACTIONS_KEY!r} without per-hand PI-DEX targets"
                )
            return mutable_data
        if has_left_actions != has_right_actions:
            missing_field = RIGHT_ACTIONS_KEY if has_left_actions else LEFT_ACTIONS_KEY
            raise KeyError(f"data: missing required field {missing_field!r}")
        left_actions = _pop_required(mutable_data, LEFT_ACTIONS_KEY)
        right_actions = _pop_required(mutable_data, RIGHT_ACTIONS_KEY)
        if MODEL_ACTIONS_KEY in mutable_data:
            raise ValueError(f"data already contains reserved field {MODEL_ACTIONS_KEY!r}")

        _require_finite_logical_actions(left_actions, field_name=LEFT_ACTIONS_KEY)
        _require_finite_logical_actions(right_actions, field_name=RIGHT_ACTIONS_KEY)
        padded_left = pad_action(left_actions, representation=self.action_representation)
        padded_right = pad_action(right_actions, representation=self.action_representation)
        mutable_data[MODEL_ACTIONS_KEY] = interleave(
            padded_left,
            padded_right,
            representation=self.action_representation,
        )
        return mutable_data


@dataclasses.dataclass(frozen=True)
class UnpackBimanualActions:
    """Unpack OpenPI model actions for per-hand inverse normalization.

    Input data must contain ``actions`` with shape ``[..., 2 * K, 32]``. The
    transform removes it and adds ``left_actions`` and ``right_actions`` with
    shape ``[..., K, action_representation.logical_action_dim]``. All model
    padding values are discarded.
    """

    action_representation: ActionRepresentation

    def __post_init__(self) -> None:
        if not isinstance(self.action_representation, ActionRepresentation):
            raise TypeError(
                "action_representation: expected ActionRepresentation, "
                f"got {type(self.action_representation).__name__}"
            )

    def __call__(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Apply the inference-side unpacking transform.

        Raises:
            TypeError: If ``data`` is not a mapping or actions are not a
                floating NumPy array.
            KeyError: If ``actions`` is missing.
            ValueError: If output fields collide or the action shape is invalid.
        """
        mutable_data = _copy_mapping(data)
        model_actions = _pop_required(mutable_data, MODEL_ACTIONS_KEY)
        for output_key in (LEFT_ACTIONS_KEY, RIGHT_ACTIONS_KEY):
            if output_key in mutable_data:
                raise ValueError(f"data already contains reserved field {output_key!r}")

        left_actions, right_actions = deinterleave(
            model_actions,
            representation=self.action_representation,
        )
        mutable_data[LEFT_ACTIONS_KEY] = unpad_action(
            left_actions,
            representation=self.action_representation,
        )
        mutable_data[RIGHT_ACTIONS_KEY] = unpad_action(
            right_actions,
            representation=self.action_representation,
        )
        return mutable_data


def _copy_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise TypeError(f"data: expected a mapping, got {type(data).__name__}")
    return dict(data)


def _pop_required(data: dict[str, Any], field_name: str) -> np.ndarray:
    try:
        value = data.pop(field_name)
    except KeyError:
        raise KeyError(f"data: missing required field {field_name!r}") from None
    if not isinstance(value, np.ndarray):
        raise TypeError(f"data[{field_name!r}]: expected numpy.ndarray, got {type(value).__name__}")
    return value


def _require_finite_logical_actions(actions: np.ndarray, *, field_name: str) -> None:
    if not np.all(np.isfinite(actions)):
        raise ValueError(f"data[{field_name!r}]: expected all semantic action values to be finite")


def _validate_floating_array(
    value: object,
    *,
    field_name: str,
    expected_shape: tuple[int, ...] | None,
    expected_ndim: int,
) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"data[{field_name!r}]: expected numpy.ndarray, got {type(value).__name__}")
    if value.ndim != expected_ndim:
        raise ValueError(
            f"data[{field_name!r}].ndim: expected {expected_ndim}, got {value.ndim} "
            f"for shape {value.shape}"
        )
    if expected_shape is None:
        if value.size == 0:
            raise ValueError(f"data[{field_name!r}]: expected a non-empty vector")
    elif value.shape != expected_shape:
        raise ValueError(
            f"data[{field_name!r}].shape: expected {expected_shape}, got {value.shape}"
        )
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"data[{field_name!r}].dtype: expected floating, got {value.dtype}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"data[{field_name!r}]: expected finite values")
