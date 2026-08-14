import dataclasses
import enum
import types

import pytest

from pi_dex.actions import MODEL_ACTION_DIM
from pi_dex.actions import ActionRepresentation
from pi_dex.actions import valid_action_mask
from pi_dex.spec import ACTION_LAYOUT_VERSION
from pi_dex.spec import ACTION_METADATA_SCHEMA_VERSION
from pi_dex.spec import BimanualActionSpec
from tests.helpers import spec_for_representation


def test_spec_derives_even_model_horizon(action_spec: BimanualActionSpec) -> None:
    assert action_spec.physical_horizon == 2
    assert action_spec.model_action_horizon == 4


@pytest.mark.parametrize("representation", list(ActionRepresentation))
def test_spec_derives_representation_contract(
    action_spec: BimanualActionSpec,
    representation: ActionRepresentation,
) -> None:
    spec = spec_for_representation(action_spec, representation)

    assert spec.logical_action_dim == representation.logical_action_dim
    assert spec.valid_action_mask == valid_action_mask(representation)
    assert len(spec.valid_action_mask) == MODEL_ACTION_DIM
    assert spec.requires_forward_kinematics is (
        representation is ActionRepresentation.CARTESIAN_31D
    )


def test_spec_validates_matching_pi05_config(action_spec: BimanualActionSpec) -> None:
    model_config = types.SimpleNamespace(pi05=True, action_dim=32, action_horizon=4)

    action_spec.validate_openpi_model_config(model_config)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("pi05", 1),
        ("action_dim", 32.0),
        ("action_dim", True),
        ("action_horizon", 4.0),
        ("action_horizon", True),
    ],
)
def test_spec_rejects_model_config_with_equal_but_wrong_type(
    action_spec: BimanualActionSpec,
    field_name: str,
    value: object,
) -> None:
    config_values = {"pi05": True, "action_dim": 32, "action_horizon": 4}
    config_values[field_name] = value

    with pytest.raises(TypeError, match=field_name):
        action_spec.validate_openpi_model_config(types.SimpleNamespace(**config_values))


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("pi05", False),
        ("action_dim", 31),
        ("action_horizon", 3),
    ],
)
def test_spec_rejects_incompatible_model_config(
    action_spec: BimanualActionSpec,
    field_name: str,
    value: object,
) -> None:
    config_values = {"pi05": True, "action_dim": 32, "action_horizon": 4}
    config_values[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        action_spec.validate_openpi_model_config(types.SimpleNamespace(**config_values))


def test_spec_metadata_round_trip_contract(action_spec: BimanualActionSpec) -> None:
    metadata = {"pi_dex": action_spec.to_metadata()}

    action_spec.validate_metadata(metadata)
    assert metadata["pi_dex"]["layout_version"] == ACTION_LAYOUT_VERSION
    assert metadata["pi_dex"]["metadata_schema_version"] == ACTION_METADATA_SCHEMA_VERSION
    assert metadata["pi_dex"]["action_representation"] == action_spec.action_representation.value
    assert metadata["pi_dex"]["logical_action_dim"] == action_spec.logical_action_dim
    assert metadata["pi_dex"]["left_hand_joint_order"] == list(action_spec.left_hand_joint_order)
    assert metadata["pi_dex"]["right_hand_joint_order"] == list(action_spec.right_hand_joint_order)
    assert metadata["pi_dex"]["hand_mapping_version"] == action_spec.hand_mapping_version


def test_joint_spec_metadata_records_29d_joint_layout(action_spec: BimanualActionSpec) -> None:
    joint_spec = spec_for_representation(action_spec, ActionRepresentation.JOINT_29D)

    metadata = {"pi_dex": joint_spec.to_metadata()}

    joint_spec.validate_metadata(metadata)
    assert metadata["pi_dex"]["layout_version"] == 3
    assert metadata["pi_dex"]["metadata_schema_version"] == 3
    assert metadata["pi_dex"]["action_representation"] == "joint_29d"
    assert metadata["pi_dex"]["logical_action_dim"] == 29
    assert metadata["pi_dex"]["model_action_dim"] == 32
    assert metadata["pi_dex"]["arm_joint_unit"] == "rad"
    assert metadata["pi_dex"]["coordinate_frame"] is None
    assert metadata["pi_dex"]["rotation_6d_convention"] is None
    assert metadata["pi_dex"]["kinematics_calibration_version"] is None
    assert metadata["pi_dex"]["left_wrist_link"] is None
    assert metadata["pi_dex"]["right_wrist_link"] is None
    assert metadata["pi_dex"]["position_unit"] == "not_applicable"
    assert metadata["pi_dex"]["rotation_6d_unit"] == "not_applicable"


@pytest.mark.parametrize(
    "field_name",
    [
        "coordinate_frame",
        "rotation_6d_convention",
        "kinematics_calibration_version",
        "left_wrist_link",
        "right_wrist_link",
    ],
)
def test_joint_spec_rejects_cartesian_only_fields(
    action_spec: BimanualActionSpec,
    field_name: str,
) -> None:
    replacements = {
        "action_representation": ActionRepresentation.JOINT_29D,
        "coordinate_frame": None,
        "rotation_6d_convention": None,
        "kinematics_calibration_version": None,
        "left_wrist_link": None,
        "right_wrist_link": None,
    }
    replacements[field_name] = getattr(action_spec, field_name)

    with pytest.raises(ValueError, match=rf"{field_name}.*None.*joint_29d"):
        dataclasses.replace(action_spec, **replacements)


def test_spec_rejects_wrong_action_representation_type(action_spec: BimanualActionSpec) -> None:
    with pytest.raises(TypeError, match="action_representation"):
        dataclasses.replace(action_spec, action_representation="cartesian_31d")


def test_metadata_rejects_other_action_representation(action_spec: BimanualActionSpec) -> None:
    metadata = action_spec.to_metadata()
    metadata["action_representation"] = "joint_29d"

    with pytest.raises(ValueError, match="action_representation"):
        action_spec.validate_metadata({"pi_dex": metadata})


@pytest.mark.parametrize("version_field", ["layout_version", "metadata_schema_version"])
def test_v3_spec_rejects_legacy_v2_metadata(
    action_spec: BimanualActionSpec,
    version_field: str,
) -> None:
    metadata = action_spec.to_metadata()
    metadata[version_field] = 2

    with pytest.raises(ValueError, match=version_field):
        action_spec.validate_metadata({"pi_dex": metadata})


def test_spec_rejects_string_subclasses_that_do_not_round_trip_exactly(
    action_spec: BimanualActionSpec,
) -> None:
    class RobotId(enum.StrEnum):
        SHARPA = "POC22027"

    with pytest.raises(TypeError, match=r"robot_id.*expected str"):
        dataclasses.replace(action_spec, robot_id=RobotId.SHARPA)


def test_spec_public_boundaries_revalidate_bypassed_frozen_fields(
    action_spec: BimanualActionSpec,
) -> None:
    invalid_spec = dataclasses.replace(action_spec)
    object.__setattr__(invalid_spec, "physical_horizon", 0)
    model_config = types.SimpleNamespace(pi05=True, action_dim=32, action_horizon=0)

    with pytest.raises(ValueError, match="physical_horizon"):
        invalid_spec.validate_openpi_model_config(model_config)
    with pytest.raises(ValueError, match="physical_horizon"):
        invalid_spec.to_metadata()
    with pytest.raises(ValueError, match="physical_horizon"):
        invalid_spec.validate_metadata({"pi_dex": action_spec.to_metadata()})


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("layout_version", True),
        ("physical_horizon", 2.0),
        ("max_command_lead_ms", 25),
        ("hand_order", ("left", "right")),
    ],
)
def test_spec_metadata_rejects_equal_or_similar_value_with_wrong_type(
    action_spec: BimanualActionSpec,
    field_name: str,
    value: object,
) -> None:
    pi_dex_metadata = action_spec.to_metadata()
    pi_dex_metadata[field_name] = value

    with pytest.raises(TypeError, match=field_name):
        action_spec.validate_metadata({"pi_dex": pi_dex_metadata})


def test_spec_metadata_rejects_different_hand_order(action_spec: BimanualActionSpec) -> None:
    pi_dex_metadata = action_spec.to_metadata()
    pi_dex_metadata["hand_order"] = ["right", "left"]

    with pytest.raises(ValueError, match="hand_order"):
        action_spec.validate_metadata({"pi_dex": pi_dex_metadata})


def test_spec_metadata_extensions_require_an_explicit_allowlist(action_spec: BimanualActionSpec) -> None:
    metadata = {"pi_dex": {**action_spec.to_metadata(), "wire_format": "test"}}

    with pytest.raises(ValueError, match=r"unexpected fields.*wire_format"):
        action_spec.validate_metadata(metadata)

    action_spec.validate_metadata(
        metadata,
        allowed_extra_fields=frozenset({"wire_format"}),
    )


def test_spec_rejects_same_wrist_link_for_both_sides(action_spec: BimanualActionSpec) -> None:
    with pytest.raises(ValueError, match="different physical links"):
        dataclasses.replace(action_spec, right_wrist_link=action_spec.left_wrist_link)


def test_spec_rejects_overlapping_left_right_joint_names(action_spec: BimanualActionSpec) -> None:
    with pytest.raises(ValueError, match="must be disjoint"):
        dataclasses.replace(action_spec, right_arm_joint_order=action_spec.left_arm_joint_order)

    with pytest.raises(ValueError, match=r"right_hand_joint_order.*must be disjoint"):
        dataclasses.replace(action_spec, right_hand_joint_order=action_spec.left_hand_joint_order)

    with pytest.raises(ValueError, match=r"left_hand_joint_order.*must be disjoint"):
        dataclasses.replace(
            action_spec,
            left_hand_joint_order=(
                action_spec.left_arm_joint_order[0],
                *action_spec.left_hand_joint_order[1:],
            ),
        )


@pytest.mark.parametrize("field_name", ["left_hand_joint_order", "right_hand_joint_order"])
def test_spec_requires_exactly_22_unique_nonempty_hand_joint_names(
    action_spec: BimanualActionSpec,
    field_name: str,
) -> None:
    valid_order = getattr(action_spec, field_name)

    with pytest.raises(ValueError, match="exactly 22"):
        dataclasses.replace(action_spec, **{field_name: valid_order[:-1]})

    duplicate_order = valid_order[:-1] + (valid_order[0],)
    with pytest.raises(ValueError, match="unique joint names"):
        dataclasses.replace(action_spec, **{field_name: duplicate_order})

    empty_order = valid_order[:3] + (" ",) + valid_order[4:]
    with pytest.raises(ValueError, match=r"\[3\].*non-empty"):
        dataclasses.replace(action_spec, **{field_name: empty_order})


def test_spec_requires_nonempty_hand_mapping_version(action_spec: BimanualActionSpec) -> None:
    with pytest.raises(ValueError, match="hand_mapping_version"):
        dataclasses.replace(action_spec, hand_mapping_version=" ")


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("physical_horizon", True),
        ("physical_horizon", 2.0),
        ("control_frequency_hz", 59),
        ("control_frequency_hz", True),
        ("max_command_lead_ms", 25),
    ],
)
def test_spec_rejects_bool_int_float_type_confusion(
    action_spec: BimanualActionSpec,
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(TypeError, match=field_name):
        dataclasses.replace(action_spec, **{field_name: value})
