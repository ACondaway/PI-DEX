"""Validated Sharpa commanded-action alignment and representation boundary.

This module does not provide a Sharpa North kinematics model. The repository has
no versioned robot description or calibration from which a trustworthy forward
kinematics implementation could be built. Cartesian 31D callers must inject a
provider whose structured metadata exactly matches ``BimanualActionSpec``. Joint
29D callers instead preserve the recorded commanded arm and hand joint targets
without invoking kinematics. Neither path falls back to optional measured
``state/<side>_arm/tcp_pose`` values.
"""

from __future__ import annotations

import dataclasses
import enum
import numbers
from typing import Protocol

import numpy as np

from pi_dex.actions import ARM_JOINT_DIM
from pi_dex.actions import HAND_JOINT_DIM
from pi_dex.actions import WRIST_POSITION_DIM
from pi_dex.actions import WRIST_ROTATION_6D_DIM
from pi_dex.actions import ActionRepresentation
from pi_dex.spec import ActionMode
from pi_dex.spec import ActionTimebase
from pi_dex.spec import BimanualActionSpec

TIMESTAMP_COLUMNS = 2
CANONICAL_ALIGNED_TIME_FIELD = "observe/vision/head/stereo/lefteye/time"
ROTATION_6D_ORTHOGONALITY_TOLERANCE = 1e-4

_COMMANDED_JOINT_DIMS = {
    "action/left_arm/joint_angle": ARM_JOINT_DIM,
    "action/left_hand/joint_angle": HAND_JOINT_DIM,
    "action/right_arm/joint_angle": ARM_JOINT_DIM,
    "action/right_hand/joint_angle": HAND_JOINT_DIM,
}


class HandSide(enum.StrEnum):
    """Physical side used by the versioned kinematics provider."""

    LEFT = "left"
    RIGHT = "right"


class ForwardKinematicsProvider(Protocol):
    """External, calibrated mapping from 7D arm commands to wrist pose.

    Implementations operate on NumPy float32 arrays. They own the exact joint
    axes/signs/zero offsets, link geometry, station calibration, left/right
    extrinsics, and wrist link. Provider conformance must additionally be tested
    against trusted joint-to-pose golden pairs outside this repository.
    """

    robot_id: str
    embodiment_version: str
    calibration_version: str
    coordinate_frame: str
    rotation_6d_convention: str
    left_arm_joint_order: tuple[str, ...]
    right_arm_joint_order: tuple[str, ...]
    input_joint_unit: str
    output_position_unit: str
    left_wrist_link: str
    right_wrist_link: str

    def wrist_pose(
        self,
        side: HandSide,
        arm_joint_angles: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map ``[K,7]`` commanded radians to ``([K,3] m, [K,6])``."""


@dataclasses.dataclass(frozen=True)
class EpisodeActionProvenance:
    """Externally verified action identity and mapping for one episode.

    Args:
        robot_id: Physical station that recorded the commanded actions.
        embodiment_version: Robot-description and kinematic-chain identity.
        command_semantics_version: Evidence that joint positions are absolute
            commanded targets rather than measured state or deltas.
        hand_mapping_version: Version of the left/right 22-column hand mapping,
            including any mirroring, axis, and sign conventions.

    Raises:
        TypeError: If any provenance value is not a string.
        ValueError: If any provenance value is empty.
    """

    robot_id: str
    embodiment_version: str
    command_semantics_version: str
    hand_mapping_version: str

    def __post_init__(self) -> None:
        for field_name in (
            "robot_id",
            "embodiment_version",
            "command_semantics_version",
            "hand_mapping_version",
        ):
            value = getattr(self, field_name)
            if type(value) is not str:
                raise TypeError(f"{field_name}: expected str, got {type(value).__name__}")
            if not value.strip():
                raise ValueError(f"{field_name}: expected a non-empty value")


@dataclasses.dataclass(frozen=True)
class AlignedTimeline:
    """Canonical 30 Hz head-camera timeline for one episode.

    ``time`` is copied onto immutable byte backing after one O(N) validation at
    episode load, avoiding repeated full-array scans for every training chunk.

    Args:
        field_name: Exact canonical head-camera time field.
        time: Float64 ``[N,2]`` array containing increasing UNIX timestamps and
            validity flags equal to one.

    Raises:
        TypeError: If ``time`` is not a float64 NumPy array.
        ValueError: If the field identity, shape, timestamps, or validity flags
            violate the canonical timeline contract.
    """

    field_name: str
    time: np.ndarray

    def __post_init__(self) -> None:
        if self.field_name != CANONICAL_ALIGNED_TIME_FIELD:
            raise ValueError(
                f"field_name: expected canonical timeline {CANONICAL_ALIGNED_TIME_FIELD!r}, got {self.field_name!r}"
            )
        _validate_time_array(self.time, field_name=self.field_name, expected_length=None)
        object.__setattr__(self, "time", _immutable_array_copy(self.time))


@dataclasses.dataclass(frozen=True)
class CommandedJointGroup:
    """One validated raw 60 Hz HDF5 commanded joint group.

    Only the four canonical ``action/{left,right}_{arm,hand}/joint_angle`` paths
    are accepted. Arrays are copied onto immutable byte backing after one
    O(M+N) episode validation, so chunk selection is O(K) and cannot be
    invalidated by mutation or by re-enabling NumPy's ``writeable`` flag.

    Args:
        field_name: Exact canonical commanded arm or hand joint field.
        joint_order: Names of the physical columns in ``joint_angles``. Arm
            groups require seven names and hand groups require twenty-two.
        joint_angles: Float32 ``[M,J]`` commanded joint angles in radians.
        time: Float64 ``[M,2]`` increasing UNIX timestamps and validity flags.
        aligned_index: Int32 ``[N]`` mapping from canonical camera frames to this
            group's raw rows.

    Raises:
        TypeError: If a joint order or array has the wrong container or dtype.
        ValueError: If the field, joint order, shapes, values, timestamps, or
            aligned indices violate the recorded-group contract.
    """

    field_name: str
    joint_order: tuple[str, ...]
    joint_angles: np.ndarray
    time: np.ndarray
    aligned_index: np.ndarray

    def __post_init__(self) -> None:
        expected_joint_dim = _COMMANDED_JOINT_DIMS.get(self.field_name)
        if expected_joint_dim is None:
            raise ValueError(
                f"field_name: expected one of {sorted(_COMMANDED_JOINT_DIMS)}, got {self.field_name!r}"
            )
        _validate_joint_order(
            self.joint_order,
            field_name=f"{self.field_name}.joint_order",
            expected_length=expected_joint_dim,
        )
        _validate_joint_matrix(
            self.joint_angles,
            field_name=self.field_name,
            expected_width=expected_joint_dim,
        )
        raw_length = self.joint_angles.shape[0]
        group_path = self.field_name.rsplit("/", 1)[0]
        _validate_time_array(self.time, field_name=f"{group_path}/time", expected_length=raw_length)
        _validate_aligned_index(
            self.aligned_index,
            field_name=f"{group_path}/aligned_index",
            raw_length=raw_length,
        )

        for field_name in ("joint_angles", "time", "aligned_index"):
            object.__setattr__(self, field_name, _immutable_array_copy(getattr(self, field_name)))


@dataclasses.dataclass(frozen=True)
class SelectedJointHorizon:
    """Selected K-step values and their source rows/timestamps.

    Args:
        joint_angles: Copied float32 ``[K,J]`` commanded angles in radians.
        timestamps_s: Copied float64 ``[K]`` source timestamps in seconds.
        raw_indices: Int64 ``[K]`` rows selected from the raw command group.
    """

    joint_angles: np.ndarray
    timestamps_s: np.ndarray
    raw_indices: np.ndarray


@dataclasses.dataclass(frozen=True)
class BimanualLogicalActionChunk:
    """Derived per-hand logical actions for one synchronized training chunk.

    Args:
        left_actions: Float32 ``[K,D]`` left-hand logical actions, where ``D``
            is 31 for Cartesian wrist targets or 29 for joint targets.
        right_actions: Float32 ``[K,D]`` right-hand logical actions using the
            same representation as ``left_actions``.
        timestamps_s: Float64 ``[K]`` physical-step timestamps in seconds.
        source_aligned_frame: Canonical camera-frame index that starts the chunk.
    """

    left_actions: np.ndarray
    right_actions: np.ndarray
    timestamps_s: np.ndarray
    source_aligned_frame: int


def select_commanded_joint_horizon(
    group: CommandedJointGroup,
    *,
    start_aligned_frame: int,
    spec: BimanualActionSpec,
) -> SelectedJointHorizon:
    """Select K commands using this group's own ``aligned_index``.

    Args:
        group: Prevalidated canonical ``action/<side>_<part>`` group.
        start_aligned_frame: Canonical camera-frame index starting the chunk.
            Python and NumPy integer scalars are accepted; booleans are rejected.
        spec: Action timebase, frequency, and horizon contract.

    Returns:
        K copied joint vectors in radians, float64 UNIX timestamps, and int64 raw
        indices. A raw-timebase horizon uses the group's aligned row as its first
        index and then advances through consecutive 60 Hz rows. An aligned
        horizon indexes every step through the recorded mapping.

    Raises:
        TypeError: If input types are invalid.
        ValueError: If the horizon exceeds this group or timestamps violate the
            configured raw control cadence.
    """
    if not isinstance(group, CommandedJointGroup):
        raise TypeError(f"group: expected CommandedJointGroup, got {type(group).__name__}")
    if not isinstance(spec, BimanualActionSpec):
        raise TypeError(f"spec: expected BimanualActionSpec, got {type(spec).__name__}")
    validated_spec = dataclasses.replace(spec)
    _validate_group_joint_order(group, spec=validated_spec)
    start_frame = _validate_start_frame(start_aligned_frame, aligned_length=group.aligned_index.shape[0])

    if validated_spec.timebase is ActionTimebase.ALIGNED_30_HZ:
        stop_aligned_frame = start_frame + validated_spec.physical_horizon
        if stop_aligned_frame > group.aligned_index.shape[0]:
            raise ValueError(
                f"{group.field_name}: aligned horizon [{start_frame}, {stop_aligned_frame}) "
                f"exceeds N={group.aligned_index.shape[0]}"
            )
        raw_indices = group.aligned_index[start_frame:stop_aligned_frame].astype(np.int64, copy=True)
    elif validated_spec.timebase is ActionTimebase.RAW_CONTROL_60_HZ:
        first_raw_index = int(group.aligned_index[start_frame])
        stop_raw_index = first_raw_index + validated_spec.physical_horizon
        if stop_raw_index > group.joint_angles.shape[0]:
            raise ValueError(
                f"{group.field_name}: raw horizon [{first_raw_index}, {stop_raw_index}) "
                f"exceeds M={group.joint_angles.shape[0]}"
            )
        raw_indices = np.arange(first_raw_index, stop_raw_index, dtype=np.int64)
    else:
        raise ValueError(f"spec.timebase: unsupported value {validated_spec.timebase!r}")

    timestamps_s = group.time[raw_indices, 0].copy()
    if validated_spec.timebase is ActionTimebase.RAW_CONTROL_60_HZ:
        _validate_control_cadence(timestamps_s, field_name=group.field_name, spec=validated_spec)
    return SelectedJointHorizon(
        joint_angles=group.joint_angles[raw_indices].copy(),
        timestamps_s=timestamps_s,
        raw_indices=raw_indices,
    )


def derive_bimanual_logical_action_chunk(
    *,
    aligned_timeline: AlignedTimeline,
    provenance: EpisodeActionProvenance,
    left_arm: CommandedJointGroup,
    left_hand: CommandedJointGroup,
    right_arm: CommandedJointGroup,
    right_hand: CommandedJointGroup,
    start_aligned_frame: int,
    spec: BimanualActionSpec,
    kinematics: ForwardKinematicsProvider | None,
) -> BimanualLogicalActionChunk:
    """Align commanded joints and derive synchronized ``[K,D]`` chunks.

    The caller must supply episode provenance that externally verifies the robot
    identity and that recorded commanded joint positions are absolute targets.
    This function never infers those semantics from numeric values.

    Args:
        aligned_timeline: Canonical head-camera timeline defining N and frame time.
        provenance: Verified robot, command semantics, and hand-mapping identity.
        left_arm: Exact ``action/left_arm/joint_angle`` group.
        left_hand: Exact ``action/left_hand/joint_angle`` group.
        right_arm: Exact ``action/right_arm/joint_angle`` group.
        right_hand: Exact ``action/right_hand/joint_angle`` group.
        start_aligned_frame: Canonical aligned frame at which the chunk starts.
        spec: Full training/deployment semantic contract.
        kinematics: Injected and version-matched Sharpa North FK provider for
            ``CARTESIAN_31D``. It must be ``None`` for ``JOINT_29D``.

    Returns:
        Left/right float32 logical actions, their physical-step timestamps, and
        the source aligned frame. For aligned timebase, timestamps are canonical
        camera times; for raw timebase, they are left-arm command times.

    Raises:
        TypeError: If typed inputs or Cartesian provider outputs violate the
            interface, including a missing Cartesian FK provider.
        ValueError: If source roles, N, provenance, semantics, alignment,
            timestamps, representation-specific kinematics use, or outputs
            violate the contract.
    """
    if not isinstance(aligned_timeline, AlignedTimeline):
        raise TypeError(
            "aligned_timeline: expected AlignedTimeline, "
            f"got {type(aligned_timeline).__name__}"
        )
    if not isinstance(spec, BimanualActionSpec):
        raise TypeError(f"spec: expected BimanualActionSpec, got {type(spec).__name__}")
    validated_spec = dataclasses.replace(spec)
    _validate_absolute_action_contract(validated_spec)
    _validate_provenance(provenance, spec=validated_spec)
    if validated_spec.action_representation is ActionRepresentation.CARTESIAN_31D:
        _validate_kinematics_provider(kinematics, spec=validated_spec)
    elif validated_spec.action_representation is ActionRepresentation.JOINT_29D:
        if kinematics is not None:
            raise ValueError(
                "kinematics: expected None for JOINT_29D actions; joint targets must not use FK"
            )
    else:
        raise ValueError(
            "spec.action_representation: unsupported value "
            f"{validated_spec.action_representation!r}"
        )
    _require_group_role(
        left_arm,
        expected_field="action/left_arm/joint_angle",
        expected_joint_order=validated_spec.left_arm_joint_order,
    )
    _require_group_role(
        left_hand,
        expected_field="action/left_hand/joint_angle",
        expected_joint_order=validated_spec.left_hand_joint_order,
    )
    _require_group_role(
        right_arm,
        expected_field="action/right_arm/joint_angle",
        expected_joint_order=validated_spec.right_arm_joint_order,
    )
    _require_group_role(
        right_hand,
        expected_field="action/right_hand/joint_angle",
        expected_joint_order=validated_spec.right_hand_joint_order,
    )

    groups = (left_arm, left_hand, right_arm, right_hand)
    canonical_length = aligned_timeline.time.shape[0]
    for group in groups:
        group_aligned_length = group.aligned_index.shape[0]
        if group_aligned_length != canonical_length:
            raise ValueError(
                f"{group.field_name.rsplit('/', 1)[0]}/aligned_index.shape[0]: "
                f"expected canonical N={canonical_length}, got {group_aligned_length}"
            )
    start_frame = _validate_start_frame(start_aligned_frame, aligned_length=canonical_length)

    selected = tuple(
        select_commanded_joint_horizon(group, start_aligned_frame=start_frame, spec=validated_spec)
        for group in groups
    )
    selected_left_arm, selected_left_hand, selected_right_arm, selected_right_hand = selected
    _validate_cross_group_timestamps(selected, spec=validated_spec)
    _validate_canonical_alignment(
        selected,
        aligned_timeline=aligned_timeline,
        start_aligned_frame=start_frame,
        spec=validated_spec,
    )

    if validated_spec.action_representation is ActionRepresentation.CARTESIAN_31D:
        if kinematics is None:
            raise AssertionError("validated Cartesian action path lost its kinematics provider")
        left_actions = derive_logical_actions(
            HandSide.LEFT,
            selected_left_arm.joint_angles,
            selected_left_hand.joint_angles,
            provenance=provenance,
            arm_joint_order=left_arm.joint_order,
            hand_joint_order=left_hand.joint_order,
            spec=validated_spec,
            kinematics=kinematics,
        )
        right_actions = derive_logical_actions(
            HandSide.RIGHT,
            selected_right_arm.joint_angles,
            selected_right_hand.joint_angles,
            provenance=provenance,
            arm_joint_order=right_arm.joint_order,
            hand_joint_order=right_hand.joint_order,
            spec=validated_spec,
            kinematics=kinematics,
        )
    else:
        left_actions = derive_joint_actions(
            HandSide.LEFT,
            selected_left_arm.joint_angles,
            selected_left_hand.joint_angles,
            provenance=provenance,
            arm_joint_order=left_arm.joint_order,
            hand_joint_order=left_hand.joint_order,
            spec=validated_spec,
        )
        right_actions = derive_joint_actions(
            HandSide.RIGHT,
            selected_right_arm.joint_angles,
            selected_right_hand.joint_angles,
            provenance=provenance,
            arm_joint_order=right_arm.joint_order,
            hand_joint_order=right_hand.joint_order,
            spec=validated_spec,
        )
    if validated_spec.timebase is ActionTimebase.ALIGNED_30_HZ:
        stop_frame = start_frame + validated_spec.physical_horizon
        timestamps_s = aligned_timeline.time[start_frame:stop_frame, 0].copy()
        _validate_control_cadence(
            timestamps_s,
            field_name=aligned_timeline.field_name,
            spec=validated_spec,
        )
    else:
        timestamps_s = selected_left_arm.timestamps_s.copy()
    return BimanualLogicalActionChunk(
        left_actions=left_actions,
        right_actions=right_actions,
        timestamps_s=timestamps_s,
        source_aligned_frame=start_frame,
    )


def derive_logical_actions(
    side: HandSide,
    arm_joint_angles: np.ndarray,
    hand_joint_angles: np.ndarray,
    *,
    provenance: EpisodeActionProvenance,
    arm_joint_order: tuple[str, ...],
    hand_joint_order: tuple[str, ...],
    spec: BimanualActionSpec,
    kinematics: ForwardKinematicsProvider,
) -> np.ndarray:
    """Derive one side's absolute Cartesian ``[K,31]`` action sequence.

    Args:
        side: Physical left or right side.
        arm_joint_angles: Float32 commanded angles shaped ``[K,7]`` in radians.
        hand_joint_angles: Float32 commanded angles shaped ``[K,22]`` in radians,
            ordered according to ``hand_joint_order``.
        provenance: Verified episode identity, absolute command semantics, and
            hand-column mapping version.
        arm_joint_order: Seven declared input columns for ``arm_joint_angles``.
        hand_joint_order: Twenty-two declared input columns for
            ``hand_joint_angles``.
        spec: Absolute ``CARTESIAN_31D`` semantic contract matching the provider.
        kinematics: Calibrated provider returning wrist position and rotation 6D.

    Returns:
        Float32 actions ordered as 3D wrist position in metres, rotation 6D, and
        22 hand joint angles in radians.

    Raises:
        TypeError: If arrays or provider outputs have invalid dtypes/types.
        ValueError: If the representation, provenance, joint order, action mode,
            shapes, metadata, or pose values are invalid.
    """
    if not isinstance(spec, BimanualActionSpec):
        raise TypeError(f"spec: expected BimanualActionSpec, got {type(spec).__name__}")
    validated_spec = dataclasses.replace(spec)
    _require_action_representation(
        validated_spec,
        expected=ActionRepresentation.CARTESIAN_31D,
        function_name="derive_logical_actions",
    )
    _validate_absolute_action_contract(validated_spec)
    if not isinstance(side, HandSide):
        raise TypeError(f"side: expected HandSide, got {type(side).__name__}")
    _validate_provenance(provenance, spec=validated_spec)
    expected_arm_order = (
        validated_spec.left_arm_joint_order
        if side is HandSide.LEFT
        else validated_spec.right_arm_joint_order
    )
    expected_hand_order = (
        validated_spec.left_hand_joint_order
        if side is HandSide.LEFT
        else validated_spec.right_hand_joint_order
    )
    _require_declared_joint_order(
        arm_joint_order,
        expected=expected_arm_order,
        field_name="arm_joint_order",
    )
    _require_declared_joint_order(
        hand_joint_order,
        expected=expected_hand_order,
        field_name="hand_joint_order",
    )
    _validate_joint_chunk(
        arm_joint_angles,
        field_name=f"{side.value}_arm_joint_angles",
        expected_shape=(validated_spec.physical_horizon, ARM_JOINT_DIM),
    )
    _validate_joint_chunk(
        hand_joint_angles,
        field_name=f"{side.value}_hand_joint_angles",
        expected_shape=(validated_spec.physical_horizon, HAND_JOINT_DIM),
    )
    _validate_kinematics_provider(kinematics, spec=validated_spec)

    wrist_position, wrist_rotation_6d = kinematics.wrist_pose(side, arm_joint_angles.copy())
    _validate_kinematics_output(
        wrist_position,
        field_name=f"kinematics.{side.value}.wrist_position",
        expected_shape=(validated_spec.physical_horizon, WRIST_POSITION_DIM),
    )
    _validate_kinematics_output(
        wrist_rotation_6d,
        field_name=f"kinematics.{side.value}.wrist_rotation_6d",
        expected_shape=(validated_spec.physical_horizon, WRIST_ROTATION_6D_DIM),
    )
    _validate_rotation_6d(wrist_rotation_6d, field_name=f"kinematics.{side.value}.wrist_rotation_6d")

    logical_actions = np.concatenate((wrist_position, wrist_rotation_6d, hand_joint_angles), axis=-1)
    if logical_actions.shape != (
        validated_spec.physical_horizon,
        validated_spec.logical_action_dim,
    ):
        raise AssertionError(f"internal action layout error: got shape {logical_actions.shape}")
    return logical_actions


def derive_joint_actions(
    side: HandSide,
    arm_joint_angles: np.ndarray,
    hand_joint_angles: np.ndarray,
    *,
    provenance: EpisodeActionProvenance,
    arm_joint_order: tuple[str, ...],
    hand_joint_order: tuple[str, ...],
    spec: BimanualActionSpec,
) -> np.ndarray:
    """Preserve one side's absolute commanded joints as ``[K,29]`` actions.

    Args:
        side: Physical left or right side.
        arm_joint_angles: Float32 commanded angles shaped ``[K,7]`` in radians.
        hand_joint_angles: Float32 commanded angles shaped ``[K,22]`` in radians.
        provenance: Verified episode identity, absolute command semantics, and
            hand-column mapping version.
        arm_joint_order: Seven declared arm columns in dataset order.
        hand_joint_order: Twenty-two declared hand columns in dataset order.
        spec: Absolute ``JOINT_29D`` semantic contract.

    Returns:
        A new float32 ``[K,29]`` array ordered as the seven arm joint targets
        followed by the twenty-two hand joint targets, all in radians.

    Raises:
        TypeError: If typed inputs or arrays violate the interface.
        ValueError: If representation, provenance, joint order, action mode, or
            shapes violate the joint-action contract.
    """
    if not isinstance(spec, BimanualActionSpec):
        raise TypeError(f"spec: expected BimanualActionSpec, got {type(spec).__name__}")
    validated_spec = dataclasses.replace(spec)
    _require_action_representation(
        validated_spec,
        expected=ActionRepresentation.JOINT_29D,
        function_name="derive_joint_actions",
    )
    _validate_absolute_action_contract(validated_spec)
    if not isinstance(side, HandSide):
        raise TypeError(f"side: expected HandSide, got {type(side).__name__}")
    _validate_provenance(provenance, spec=validated_spec)
    expected_arm_order = (
        validated_spec.left_arm_joint_order
        if side is HandSide.LEFT
        else validated_spec.right_arm_joint_order
    )
    expected_hand_order = (
        validated_spec.left_hand_joint_order
        if side is HandSide.LEFT
        else validated_spec.right_hand_joint_order
    )
    _require_declared_joint_order(
        arm_joint_order,
        expected=expected_arm_order,
        field_name="arm_joint_order",
    )
    _require_declared_joint_order(
        hand_joint_order,
        expected=expected_hand_order,
        field_name="hand_joint_order",
    )
    _validate_joint_chunk(
        arm_joint_angles,
        field_name=f"{side.value}_arm_joint_angles",
        expected_shape=(validated_spec.physical_horizon, ARM_JOINT_DIM),
    )
    _validate_joint_chunk(
        hand_joint_angles,
        field_name=f"{side.value}_hand_joint_angles",
        expected_shape=(validated_spec.physical_horizon, HAND_JOINT_DIM),
    )

    logical_actions = np.concatenate((arm_joint_angles, hand_joint_angles), axis=-1)
    if logical_actions.shape != (
        validated_spec.physical_horizon,
        validated_spec.logical_action_dim,
    ):
        raise AssertionError(f"internal joint action layout error: got shape {logical_actions.shape}")
    return logical_actions


def _require_action_representation(
    spec: BimanualActionSpec,
    *,
    expected: ActionRepresentation,
    function_name: str,
) -> None:
    if spec.action_representation is not expected:
        raise ValueError(
            f"spec.action_representation: {function_name} requires {expected.value!r}, "
            f"got {spec.action_representation.value!r}"
        )


def _validate_absolute_action_contract(spec: BimanualActionSpec) -> None:
    if spec.action_mode is not ActionMode.ABSOLUTE:
        raise ValueError(
            "spec.action_mode: commanded joint positions can only derive absolute actions; "
            f"got {spec.action_mode.value!r}"
        )


def _validate_provenance(provenance: EpisodeActionProvenance, *, spec: BimanualActionSpec) -> None:
    if not isinstance(provenance, EpisodeActionProvenance):
        raise TypeError(f"provenance: expected EpisodeActionProvenance, got {type(provenance).__name__}")
    expected_fields = {
        "robot_id": spec.robot_id,
        "embodiment_version": spec.embodiment_version,
        "command_semantics_version": spec.command_semantics_version,
        "hand_mapping_version": spec.hand_mapping_version,
    }
    for field_name, expected_value in expected_fields.items():
        actual_value = getattr(provenance, field_name)
        if actual_value != expected_value:
            raise ValueError(f"provenance.{field_name}: expected {expected_value!r}, got {actual_value!r}")


def _require_group_role(
    group: object,
    *,
    expected_field: str,
    expected_joint_order: tuple[str, ...],
) -> None:
    if not isinstance(group, CommandedJointGroup):
        raise TypeError(f"{expected_field}: expected CommandedJointGroup, got {type(group).__name__}")
    if group.field_name != expected_field:
        raise ValueError(f"group role: expected {expected_field!r}, got {group.field_name!r}")
    if group.joint_order != expected_joint_order:
        raise ValueError(
            f"{expected_field}.joint_order: expected {expected_joint_order!r}, "
            f"got {group.joint_order!r}"
        )


def _validate_group_joint_order(
    group: CommandedJointGroup,
    *,
    spec: BimanualActionSpec,
) -> None:
    expected_orders = {
        "action/left_arm/joint_angle": spec.left_arm_joint_order,
        "action/left_hand/joint_angle": spec.left_hand_joint_order,
        "action/right_arm/joint_angle": spec.right_arm_joint_order,
        "action/right_hand/joint_angle": spec.right_hand_joint_order,
    }
    expected_order = expected_orders[group.field_name]
    if group.joint_order != expected_order:
        raise ValueError(
            f"{group.field_name}.joint_order: expected {expected_order!r}, "
            f"got {group.joint_order!r}"
        )


def _require_declared_joint_order(
    joint_order: object,
    *,
    expected: tuple[str, ...],
    field_name: str,
) -> None:
    _validate_joint_order(
        joint_order,
        field_name=field_name,
        expected_length=len(expected),
    )
    if joint_order != expected:
        raise ValueError(f"{field_name}: expected {expected!r}, got {joint_order!r}")


def _validate_cross_group_timestamps(
    selected: tuple[SelectedJointHorizon, ...],
    *,
    spec: BimanualActionSpec,
) -> None:
    all_timestamps = np.stack(tuple(horizon.timestamps_s for horizon in selected), axis=0)
    timestamp_spread_ms = np.ptp(all_timestamps, axis=0) * 1_000.0
    invalid_steps = np.flatnonzero(timestamp_spread_ms > spec.max_group_timestamp_skew_ms)
    if invalid_steps.size:
        raise ValueError(
            "command group timestamp skew exceeds "
            f"{spec.max_group_timestamp_skew_ms} ms at physical steps {invalid_steps.tolist()}; "
            f"actual spread ms={timestamp_spread_ms[invalid_steps].tolist()}"
        )


def _validate_canonical_alignment(
    selected: tuple[SelectedJointHorizon, ...],
    *,
    aligned_timeline: AlignedTimeline,
    start_aligned_frame: int,
    spec: BimanualActionSpec,
) -> None:
    if spec.timebase is ActionTimebase.ALIGNED_30_HZ:
        stop_frame = start_aligned_frame + spec.physical_horizon
        if stop_frame > aligned_timeline.time.shape[0]:
            raise ValueError(
                f"canonical aligned horizon [{start_aligned_frame}, {stop_frame}) "
                f"exceeds N={aligned_timeline.time.shape[0]}"
            )
        canonical_timestamps = aligned_timeline.time[start_aligned_frame:stop_frame, 0]
        command_timestamps = np.stack(tuple(horizon.timestamps_s for horizon in selected), axis=0)
    else:
        canonical_timestamps = aligned_timeline.time[start_aligned_frame : start_aligned_frame + 1, 0]
        command_timestamps = np.stack(tuple(horizon.timestamps_s[:1] for horizon in selected), axis=0)

    alignment_error_ms = np.abs(command_timestamps - canonical_timestamps[np.newaxis, :]) * 1_000.0
    invalid = np.argwhere(alignment_error_ms > spec.max_alignment_timestamp_error_ms)
    if invalid.size:
        group_index, step_index = invalid[0]
        raise ValueError(
            "command/canonical timestamp alignment exceeds "
            f"{spec.max_alignment_timestamp_error_ms} ms at group {int(group_index)}, "
            f"physical step {int(step_index)}; actual error ms={alignment_error_ms[group_index, step_index]}"
        )


def _validate_control_cadence(
    timestamps_s: np.ndarray,
    *,
    field_name: str,
    spec: BimanualActionSpec,
) -> None:
    if timestamps_s.size < 2:
        return
    expected_period_s = 1.0 / spec.control_frequency_hz
    period_error_ms = np.abs(np.diff(timestamps_s) - expected_period_s) * 1_000.0
    invalid_steps = np.flatnonzero(period_error_ms > spec.max_control_period_error_ms)
    if invalid_steps.size:
        raise ValueError(
            f"{field_name}: control-period error exceeds {spec.max_control_period_error_ms} ms "
            f"after steps {invalid_steps.tolist()}; actual error ms={period_error_ms[invalid_steps].tolist()}"
        )


def _validate_joint_matrix(values: object, *, field_name: str, expected_width: int) -> None:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{field_name}: expected numpy.ndarray, got {type(values).__name__}")
    if values.ndim != 2 or values.shape[1] != expected_width:
        raise ValueError(f"{field_name}.shape: expected [M, {expected_width}], got {values.shape}")
    if values.shape[0] == 0:
        raise ValueError(f"{field_name}.shape[0]: expected at least one raw row")
    if values.dtype != np.float32:
        raise TypeError(f"{field_name}.dtype: expected float32, got {values.dtype}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{field_name}: expected all commanded joint angles to be finite")


def _validate_joint_order(value: object, *, field_name: str, expected_length: int) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{field_name}: expected tuple[str, ...], got {type(value).__name__}")
    if len(value) != expected_length:
        raise ValueError(f"{field_name}: expected exactly {expected_length} joint names, got {len(value)}")
    for index, joint_name in enumerate(value):
        if type(joint_name) is not str:
            raise TypeError(
                f"{field_name}[{index}]: expected str, got {type(joint_name).__name__}"
            )
        if not joint_name.strip():
            raise ValueError(f"{field_name}[{index}]: expected a non-empty value")
    if len(set(value)) != len(value):
        raise ValueError(f"{field_name}: expected unique joint names")


def _immutable_array_copy(values: np.ndarray) -> np.ndarray:
    """Copy an array onto a read-only buffer whose flag cannot be reopened."""
    immutable = np.frombuffer(values.tobytes(order="C"), dtype=values.dtype).reshape(values.shape)
    immutable.flags.writeable = False
    return immutable


def _validate_time_array(values: object, *, field_name: str, expected_length: int | None) -> None:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{field_name}: expected numpy.ndarray, got {type(values).__name__}")
    if values.ndim != 2 or values.shape[1] != TIMESTAMP_COLUMNS:
        expected = "[N, 2]" if expected_length is None else str((expected_length, TIMESTAMP_COLUMNS))
        raise ValueError(f"{field_name}.shape: expected {expected}, got {values.shape}")
    if expected_length is not None and values.shape[0] != expected_length:
        raise ValueError(f"{field_name}.shape[0]: expected {expected_length}, got {values.shape[0]}")
    if values.shape[0] == 0:
        raise ValueError(f"{field_name}.shape[0]: expected at least one timestamp")
    if values.dtype != np.float64:
        raise TypeError(f"{field_name}.dtype: expected float64, got {values.dtype}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{field_name}: expected finite timestamps and validity flags")
    invalid_rows = np.flatnonzero(values[:, 1] != 1.0)
    if invalid_rows.size:
        raise ValueError(f"{field_name}[:,1]: invalid rows {invalid_rows.tolist()}; expected validity flag 1.0")
    # Sharpa OpenData camera timelines can contain duplicate timestamps (diff == 0).
    # Reject only true regressions so real episodes stay loadable.
    if np.any(np.diff(values[:, 0]) < 0):
        raise ValueError(f"{field_name}[:,0]: expected monotonically non-decreasing timestamps")


def _validate_aligned_index(values: object, *, field_name: str, raw_length: int) -> None:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{field_name}: expected numpy.ndarray, got {type(values).__name__}")
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{field_name}.shape: expected non-empty [N], got {values.shape}")
    if values.dtype != np.int32:
        raise TypeError(f"{field_name}.dtype: expected int32, got {values.dtype}")
    if np.any(np.diff(values) < 0):
        raise ValueError(f"{field_name}: expected monotonically non-decreasing indices")
    invalid_indices = np.flatnonzero((values < 0) | (values >= raw_length))
    if invalid_indices.size:
        raise ValueError(f"{field_name}: rows {invalid_indices.tolist()} contain indices outside [0, {raw_length})")


def _validate_joint_chunk(values: object, *, field_name: str, expected_shape: tuple[int, int]) -> None:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{field_name}: expected numpy.ndarray, got {type(values).__name__}")
    if values.shape != expected_shape:
        raise ValueError(f"{field_name}.shape: expected {expected_shape}, got {values.shape}")
    if values.dtype != np.float32:
        raise TypeError(f"{field_name}.dtype: expected float32, got {values.dtype}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{field_name}: expected all joint angles to be finite")


def _validate_start_frame(start_aligned_frame: object, *, aligned_length: int) -> int:
    if isinstance(start_aligned_frame, bool) or not isinstance(start_aligned_frame, numbers.Integral):
        raise TypeError(f"start_aligned_frame: expected integer, got {type(start_aligned_frame).__name__}")
    start_frame = int(start_aligned_frame)
    if not 0 <= start_frame < aligned_length:
        raise ValueError(f"start_aligned_frame: expected 0 <= frame < {aligned_length}, got {start_frame}")
    return start_frame


def _validate_kinematics_provider(
    kinematics: ForwardKinematicsProvider,
    *,
    spec: BimanualActionSpec,
) -> None:
    if kinematics is None or not callable(getattr(kinematics, "wrist_pose", None)):
        raise TypeError("kinematics: an explicit calibrated ForwardKinematicsProvider is required")
    if (
        spec.coordinate_frame is None
        or spec.rotation_6d_convention is None
        or spec.kinematics_calibration_version is None
        or spec.left_wrist_link is None
        or spec.right_wrist_link is None
    ):
        raise AssertionError("validated Cartesian spec lost required FK metadata")
    expected_metadata = {
        "robot_id": spec.robot_id,
        "embodiment_version": spec.embodiment_version,
        "calibration_version": spec.kinematics_calibration_version,
        "coordinate_frame": spec.coordinate_frame,
        "rotation_6d_convention": spec.rotation_6d_convention.value,
        "left_arm_joint_order": spec.left_arm_joint_order,
        "right_arm_joint_order": spec.right_arm_joint_order,
        "input_joint_unit": "rad",
        "output_position_unit": "m",
        "left_wrist_link": spec.left_wrist_link,
        "right_wrist_link": spec.right_wrist_link,
    }
    for attribute, expected_value in expected_metadata.items():
        actual_value = getattr(kinematics, attribute, None)
        if type(actual_value) is not type(expected_value):
            raise TypeError(
                f"kinematics.{attribute}: expected {type(expected_value).__name__}, "
                f"got {type(actual_value).__name__}"
            )
        if isinstance(expected_value, tuple):
            for index, (actual_item, expected_item) in enumerate(
                zip(actual_value, expected_value, strict=False)
            ):
                if type(actual_item) is not type(expected_item):
                    raise TypeError(
                        f"kinematics.{attribute}[{index}]: expected "
                        f"{type(expected_item).__name__}, got {type(actual_item).__name__}"
                    )
        if actual_value != expected_value:
            raise ValueError(f"kinematics.{attribute}: expected {expected_value!r}, got {actual_value!r}")


def _validate_kinematics_output(values: object, *, field_name: str, expected_shape: tuple[int, int]) -> None:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{field_name}: expected numpy.ndarray, got {type(values).__name__}")
    if values.shape != expected_shape:
        raise ValueError(f"{field_name}.shape: expected {expected_shape}, got {values.shape}")
    if values.dtype != np.float32:
        raise TypeError(f"{field_name}.dtype: expected float32, got {values.dtype}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{field_name}: expected all pose values to be finite")


def _validate_rotation_6d(values: np.ndarray, *, field_name: str) -> None:
    rotation_columns = values.reshape(values.shape[0], 2, 3)
    column_norms = np.linalg.norm(rotation_columns, axis=-1)
    column_dots = np.sum(rotation_columns[:, 0, :] * rotation_columns[:, 1, :], axis=-1)
    valid_norms = np.all(
        np.abs(column_norms - 1.0) <= ROTATION_6D_ORTHOGONALITY_TOLERANCE,
        axis=-1,
    )
    valid_orthogonality = np.abs(column_dots) <= ROTATION_6D_ORTHOGONALITY_TOLERANCE
    invalid_rows = np.flatnonzero(~(valid_norms & valid_orthogonality))
    if invalid_rows.size:
        raise ValueError(
            f"{field_name}: rows {invalid_rows.tolist()} are not two orthonormal rotation-matrix columns"
        )
