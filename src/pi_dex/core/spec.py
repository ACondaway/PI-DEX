"""Versioned semantic contract for PI-DEX bimanual actions."""

from __future__ import annotations

import dataclasses
import enum
import math
from collections.abc import Mapping
from typing import Any

from pi_dex.core.actions import MODEL_ACTION_DIM
from pi_dex.core.actions import ActionRepresentation
from pi_dex.core.actions import valid_action_mask

ACTION_LAYOUT_VERSION = 4
ACTION_METADATA_SCHEMA_VERSION = 3
HAND_ORDER = ("left", "right")


class ActionMode(enum.StrEnum):
    """Meaning of each predicted action relative to the current robot state."""

    ABSOLUTE = "absolute"
    DELTA = "delta"
    RESIDUAL = "residual"


class ActionTimebase(enum.StrEnum):
    """Dataset clock used for consecutive physical actions in a chunk."""

    ALIGNED_30_HZ = "aligned_30_hz"
    RAW_CONTROL_60_HZ = "raw_control_60_hz"


class HandNormalization(enum.StrEnum):
    """Whether left/right hands use separate or pooled action statistics."""

    PER_HAND = "per_hand"
    SHARED = "shared"


class Rotation6DConvention(enum.StrEnum):
    """Supported continuous rotation layouts at the PI-DEX boundary."""

    MATRIX_FIRST_TWO_COLUMNS_COLUMN_MAJOR = "rotation_matrix_first_two_columns_column_major_v1"


@dataclasses.dataclass(frozen=True)
class BimanualActionSpec:
    """Declare the semantics shared by training, serving, and robot control.

    Args:
        physical_horizon: Number ``K`` of simultaneous bimanual control steps.
            The OpenPI model horizon is derived as ``2 * K``.
        timebase: Whether consecutive steps follow the aligned 30 Hz timeline or
            the raw 60 Hz control timeline.
        control_frequency_hz: Actual command frequency in Hz. This is explicit
            because observed dataset rates are only approximately 30 or 60 Hz.
        robot_id: Physical Sharpa station identifier whose calibration applies.
        embodiment_version: Versioned robot-description/kinematic-chain identity.
        coordinate_frame: Reference frame for wrist positions and rotations.
            Required for Cartesian actions and required to be ``None`` for joint
            actions.
        action_mode: Whether actions are absolute, delta, or residual commands.
        action_representation: Whether a logical action contains Cartesian wrist
            pose and hand joints (31D, padded to 36) or arm, hand, and duplicated
            motor joints (36D).
        hand_normalization: Whether each side has independent action normalization
            statistics or both sides share statistics pooled before interleaving.
        rotation_6d_convention: Definition and component order of continuous 6D
            rotations. Required for Cartesian actions and ``None`` for joint
            actions.
        kinematics_calibration_version: Version of the robot model and
            calibration used to derive wrist poses from arm joint commands.
            Required for Cartesian actions and ``None`` for joint actions.
        command_semantics_version: Provenance/version that verifies recorded
            commanded joint positions are absolute targets.
        left_arm_joint_order: Seven left-arm input joint names in dataset order.
        right_arm_joint_order: Seven right-arm input joint names in dataset order.
        left_hand_joint_order: Twenty-two left-hand input joint names in dataset
            column order.
        right_hand_joint_order: Twenty-two right-hand input joint names in dataset
            column order.
        hand_mapping_version: Versioned contract for left/right hand column
            mappings, including any mirroring, axis, or sign convention.
        left_wrist_link: Left FK output link identity for Cartesian actions;
            ``None`` for joint actions.
        right_wrist_link: Right FK output link identity for Cartesian actions;
            ``None`` for joint actions.
        clock_domain: Clock identifier shared by observation, runtime, target, and
            controller timestamps.
        max_group_timestamp_skew_ms: Maximum timestamp spread allowed among the
            four left/right arm/hand command groups for one physical step.
        max_alignment_timestamp_error_ms: Maximum difference between the
            canonical camera time and each aligned command at the starting frame.
        max_control_period_error_ms: Maximum error from the configured control
            period within a selected K-step action horizon.
        max_observation_age_ms: Maximum accepted observation age at dispatch in
            milliseconds. Enforcement belongs to the runtime receiving timestamped
            observations.
        max_command_lead_ms: Maximum future target-command lead relative to the
            runtime clock.

    Raises:
        TypeError: If a field does not have its declared runtime type.
        ValueError: If a numeric field is non-positive/non-finite, a required
            semantic identifier is empty, or joint/link identities are invalid.
    """

    physical_horizon: int
    timebase: ActionTimebase
    control_frequency_hz: float
    robot_id: str
    embodiment_version: str
    coordinate_frame: str | None
    action_mode: ActionMode
    action_representation: ActionRepresentation
    hand_normalization: HandNormalization
    rotation_6d_convention: Rotation6DConvention | None
    kinematics_calibration_version: str | None
    command_semantics_version: str
    left_arm_joint_order: tuple[str, ...]
    right_arm_joint_order: tuple[str, ...]
    left_hand_joint_order: tuple[str, ...]
    right_hand_joint_order: tuple[str, ...]
    hand_mapping_version: str
    left_wrist_link: str | None
    right_wrist_link: str | None
    clock_domain: str
    max_group_timestamp_skew_ms: float
    max_alignment_timestamp_error_ms: float
    max_control_period_error_ms: float
    max_observation_age_ms: float
    max_command_lead_ms: float

    def __post_init__(self) -> None:
        _require_positive_int(self.physical_horizon, field_name="physical_horizon")
        _require_positive_finite(self.control_frequency_hz, field_name="control_frequency_hz")
        _require_nonempty(self.robot_id, field_name="robot_id")
        _require_nonempty(self.embodiment_version, field_name="embodiment_version")
        _require_nonempty(self.command_semantics_version, field_name="command_semantics_version")
        _require_joint_order(
            self.left_arm_joint_order,
            field_name="left_arm_joint_order",
            expected_length=7,
        )
        _require_joint_order(
            self.right_arm_joint_order,
            field_name="right_arm_joint_order",
            expected_length=7,
        )
        _require_joint_order(
            self.left_hand_joint_order,
            field_name="left_hand_joint_order",
            expected_length=22,
        )
        _require_joint_order(
            self.right_hand_joint_order,
            field_name="right_hand_joint_order",
            expected_length=22,
        )
        _require_nonempty(self.hand_mapping_version, field_name="hand_mapping_version")
        _require_nonempty(self.clock_domain, field_name="clock_domain")
        _require_positive_finite(
            self.max_group_timestamp_skew_ms,
            field_name="max_group_timestamp_skew_ms",
        )
        _require_positive_finite(
            self.max_alignment_timestamp_error_ms,
            field_name="max_alignment_timestamp_error_ms",
        )
        _require_positive_finite(
            self.max_control_period_error_ms,
            field_name="max_control_period_error_ms",
        )
        _require_positive_finite(self.max_observation_age_ms, field_name="max_observation_age_ms")
        _require_positive_finite(self.max_command_lead_ms, field_name="max_command_lead_ms")
        if not isinstance(self.timebase, ActionTimebase):
            raise TypeError(f"timebase: expected ActionTimebase, got {type(self.timebase).__name__}")
        if not isinstance(self.action_mode, ActionMode):
            raise TypeError(f"action_mode: expected ActionMode, got {type(self.action_mode).__name__}")
        if not isinstance(self.action_representation, ActionRepresentation):
            raise TypeError(
                "action_representation: expected ActionRepresentation, "
                f"got {type(self.action_representation).__name__}"
            )
        if not isinstance(self.hand_normalization, HandNormalization):
            raise TypeError(
                f"hand_normalization: expected HandNormalization, got {type(self.hand_normalization).__name__}"
            )
        if self.action_representation is ActionRepresentation.CARTESIAN_31D:
            _require_nonempty(self.coordinate_frame, field_name="coordinate_frame")
            _require_nonempty(
                self.kinematics_calibration_version,
                field_name="kinematics_calibration_version",
            )
            if not isinstance(self.rotation_6d_convention, Rotation6DConvention):
                raise TypeError(
                    "rotation_6d_convention: expected Rotation6DConvention, "
                    f"got {type(self.rotation_6d_convention).__name__}"
                )
            _require_nonempty(self.left_wrist_link, field_name="left_wrist_link")
            _require_nonempty(self.right_wrist_link, field_name="right_wrist_link")
            if self.left_wrist_link == self.right_wrist_link:
                raise ValueError(
                    "left_wrist_link and right_wrist_link must identify different physical links"
                )
        elif self.action_representation is ActionRepresentation.JOINT_29D:
            for field_name in (
                "coordinate_frame",
                "rotation_6d_convention",
                "kinematics_calibration_version",
                "left_wrist_link",
                "right_wrist_link",
            ):
                if getattr(self, field_name) is not None:
                    raise ValueError(
                        f"{field_name}: expected None for joint_29d actions"
                    )
        all_joint_orders = {
            "left_arm_joint_order": self.left_arm_joint_order,
            "right_arm_joint_order": self.right_arm_joint_order,
            "left_hand_joint_order": self.left_hand_joint_order,
            "right_hand_joint_order": self.right_hand_joint_order,
        }
        joint_owners: dict[str, str] = {}
        for order_name, joint_order in all_joint_orders.items():
            for joint_name in joint_order:
                previous_owner = joint_owners.setdefault(joint_name, order_name)
                if previous_owner != order_name:
                    raise ValueError(
                        f"{order_name} must be disjoint from every other joint order; "
                        f"joint {joint_name!r} also appears in {previous_owner}"
                    )

    @property
    def model_action_horizon(self) -> int:
        """Return the even OpenPI sequence length ``2 * K``."""
        return 2 * self.physical_horizon

    @property
    def logical_action_dim(self) -> int:
        """Return the representation-specific semantic width, 31 or 36."""
        if not isinstance(self.action_representation, ActionRepresentation):
            raise TypeError(
                "action_representation: expected ActionRepresentation, "
                f"got {type(self.action_representation).__name__}"
            )
        return self.action_representation.logical_action_dim

    @property
    def valid_action_mask(self) -> tuple[bool, ...]:
        """Return the ``MODEL_ACTION_DIM`` semantic-prefix mask for this representation."""
        return valid_action_mask(self.action_representation)

    @property
    def requires_forward_kinematics(self) -> bool:
        """Whether commanded arm joints must be converted to a Cartesian wrist pose."""
        if not isinstance(self.action_representation, ActionRepresentation):
            raise TypeError(
                "action_representation: expected ActionRepresentation, "
                f"got {type(self.action_representation).__name__}"
            )
        return self.action_representation is ActionRepresentation.CARTESIAN_31D

    def validate_openpi_model_config(self, model_config: object) -> None:
        """Validate a pi05 model config against this bimanual contract.

        Args:
            model_config: Object exposing ``pi05``, ``action_dim``, and
                ``action_horizon``, such as OpenPI's ``Pi0Config``.

        Raises:
            TypeError: If a required config attribute is absent or has the
                wrong exact wire type.
            ValueError: If this is not pi05 or its action shape differs from the
                PI-DEX contract.
        """
        validated_spec = dataclasses.replace(self)
        for attribute in ("pi05", "action_dim", "action_horizon"):
            if not hasattr(model_config, attribute):
                raise TypeError(f"model_config: missing required attribute {attribute!r}")

        pi05 = getattr(model_config, "pi05")
        if type(pi05) is not bool:
            raise TypeError(f"model_config.pi05: expected bool, got {type(pi05).__name__}")
        if pi05 is not True:
            raise ValueError(f"model_config.pi05: expected True, got {pi05!r}")

        actual_action_dim = getattr(model_config, "action_dim")
        _require_plain_int(actual_action_dim, field_name="model_config.action_dim")
        if actual_action_dim != MODEL_ACTION_DIM:
            raise ValueError(f"model_config.action_dim: expected {MODEL_ACTION_DIM}, got {actual_action_dim}")

        actual_horizon = getattr(model_config, "action_horizon")
        _require_plain_int(actual_horizon, field_name="model_config.action_horizon")
        if actual_horizon != validated_spec.model_action_horizon:
            raise ValueError(
                "model_config.action_horizon: "
                f"expected {validated_spec.model_action_horizon} for "
                f"physical_horizon={validated_spec.physical_horizon}, "
                f"got {actual_horizon}"
            )

    def to_metadata(self) -> dict[str, Any]:
        """Return serializable, independently versioned semantic metadata.

        ``layout_version`` versions the numerical layout and semantic ownership of
        every dimension. Version 4 widens joint_29d logical actions to 36D by
        appending a duplicated ``action/motor/joint_angle`` block to both hands;
        Cartesian remains 31D semantic padded to 36. Version 3 introduced explicit
        Cartesian versus joint representations with a 32D pretrained projection.
        ``metadata_schema_version`` versions the required semantic fields.
        """
        validated_spec = dataclasses.replace(self)
        joint_includes_duplicated_motor = (
            validated_spec.action_representation is ActionRepresentation.JOINT_29D
        )
        return {
            "metadata_schema_version": ACTION_METADATA_SCHEMA_VERSION,
            "layout_version": ACTION_LAYOUT_VERSION,
            "joint_includes_duplicated_motor": joint_includes_duplicated_motor,
            "hand_order": list(HAND_ORDER),
            "action_representation": validated_spec.action_representation.value,
            "logical_action_dim": validated_spec.logical_action_dim,
            "model_action_dim": MODEL_ACTION_DIM,
            "physical_horizon": validated_spec.physical_horizon,
            "model_action_horizon": validated_spec.model_action_horizon,
            "timebase": validated_spec.timebase.value,
            "control_frequency_hz": validated_spec.control_frequency_hz,
            "robot_id": validated_spec.robot_id,
            "embodiment_version": validated_spec.embodiment_version,
            "coordinate_frame": validated_spec.coordinate_frame,
            "action_mode": validated_spec.action_mode.value,
            "hand_normalization": validated_spec.hand_normalization.value,
            "rotation_6d_convention": (
                validated_spec.rotation_6d_convention.value
                if validated_spec.rotation_6d_convention is not None
                else None
            ),
            "kinematics_calibration_version": validated_spec.kinematics_calibration_version,
            "command_semantics_version": validated_spec.command_semantics_version,
            "left_arm_joint_order": list(validated_spec.left_arm_joint_order),
            "right_arm_joint_order": list(validated_spec.right_arm_joint_order),
            "left_hand_joint_order": list(validated_spec.left_hand_joint_order),
            "right_hand_joint_order": list(validated_spec.right_hand_joint_order),
            "hand_mapping_version": validated_spec.hand_mapping_version,
            "left_wrist_link": validated_spec.left_wrist_link,
            "right_wrist_link": validated_spec.right_wrist_link,
            "clock_domain": validated_spec.clock_domain,
            "max_group_timestamp_skew_ms": validated_spec.max_group_timestamp_skew_ms,
            "max_alignment_timestamp_error_ms": validated_spec.max_alignment_timestamp_error_ms,
            "max_control_period_error_ms": validated_spec.max_control_period_error_ms,
            "max_observation_age_ms": validated_spec.max_observation_age_ms,
            "max_command_lead_ms": validated_spec.max_command_lead_ms,
            "arm_joint_unit": "rad",
            "hand_joint_unit": "rad",
            "position_unit": (
                "m"
                if validated_spec.action_representation is ActionRepresentation.CARTESIAN_31D
                else "not_applicable"
            ),
            "rotation_6d_unit": (
                "dimensionless"
                if validated_spec.action_representation is ActionRepresentation.CARTESIAN_31D
                else "not_applicable"
            ),
        }

    def validate_metadata(
        self,
        metadata: Mapping[str, Any],
        *,
        allowed_extra_fields: frozenset[str] = frozenset(),
    ) -> None:
        """Require wire/checkpoint metadata to exactly match this contract.

        Args:
            metadata: Full policy metadata containing a ``pi_dex`` mapping.
            allowed_extra_fields: Explicit extension names accepted inside the
                ``pi_dex`` mapping. The empty default keeps checkpoint metadata
                exact; deployment allowlists only its versioned wire fields.

        Raises:
            TypeError: If either metadata level, the extension allowlist, or a
                field has a different exact wire type from the contract.
            ValueError: If a field is missing, differs, or is not allowlisted.
        """
        validated_spec = dataclasses.replace(self)
        if not isinstance(metadata, Mapping):
            raise TypeError(f"metadata: expected a mapping, got {type(metadata).__name__}")
        if not isinstance(allowed_extra_fields, frozenset):
            raise TypeError(
                "allowed_extra_fields: expected frozenset[str], "
                f"got {type(allowed_extra_fields).__name__}"
            )
        for field_name in allowed_extra_fields:
            _require_nonempty(field_name, field_name="allowed_extra_fields item")
        pi_dex_metadata = metadata.get("pi_dex")
        if not isinstance(pi_dex_metadata, Mapping):
            raise TypeError("metadata['pi_dex']: expected a mapping")

        expected_metadata = validated_spec.to_metadata()
        overlapping_extensions = allowed_extra_fields & expected_metadata.keys()
        if overlapping_extensions:
            raise ValueError(
                "allowed_extra_fields: fields already belong to the action contract: "
                f"{sorted(overlapping_extensions)!r}"
            )
        unexpected_fields = set(pi_dex_metadata) - expected_metadata.keys() - allowed_extra_fields
        if unexpected_fields:
            formatted_fields = sorted(repr(field) for field in unexpected_fields)
            raise ValueError(f"metadata['pi_dex']: unexpected fields {formatted_fields}")

        for key, expected_value in expected_metadata.items():
            if key not in pi_dex_metadata:
                raise ValueError(f"metadata['pi_dex']: missing required field {key!r}")
            actual_value = pi_dex_metadata[key]
            _require_matching_metadata_type(
                actual_value,
                expected_value=expected_value,
                field_name=f"metadata['pi_dex'][{key!r}]",
            )
            if actual_value != expected_value:
                raise ValueError(
                    f"metadata['pi_dex'][{key!r}]: expected {expected_value!r}, got {actual_value!r}"
                )


def _require_positive_int(value: object, *, field_name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{field_name}: expected int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{field_name}: expected a positive integer, got {value}")


def _require_plain_int(value: object, *, field_name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{field_name}: expected int, got {type(value).__name__}")


def _require_positive_finite(value: object, *, field_name: str) -> None:
    if type(value) is not float:
        raise TypeError(f"{field_name}: expected float, got {type(value).__name__}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field_name}: expected a positive finite value, got {value}")


def _require_nonempty(value: object, *, field_name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{field_name}: expected str, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{field_name}: expected a non-empty value")


def _require_joint_order(value: object, *, field_name: str, expected_length: int) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{field_name}: expected tuple[str, ...], got {type(value).__name__}")
    if len(value) != expected_length:
        raise ValueError(f"{field_name}: expected exactly {expected_length} joint names, got {len(value)}")
    for index, joint_name in enumerate(value):
        _require_nonempty(joint_name, field_name=f"{field_name}[{index}]")
    if len(set(value)) != len(value):
        raise ValueError(f"{field_name}: expected unique joint names")


def _require_matching_metadata_type(
    value: object,
    *,
    expected_value: object,
    field_name: str,
) -> None:
    if type(value) is not type(expected_value):
        raise TypeError(
            f"{field_name}: expected {type(expected_value).__name__}, "
            f"got {type(value).__name__}"
        )
    if isinstance(expected_value, list):
        for index, (item, expected_item) in enumerate(zip(value, expected_value, strict=False)):
            if type(item) is not type(expected_item):
                raise TypeError(
                    f"{field_name}[{index}]: expected {type(expected_item).__name__}, "
                    f"got {type(item).__name__}"
                )
