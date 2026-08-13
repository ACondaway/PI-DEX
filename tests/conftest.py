import pytest

from pi_dex.spec import ActionMode
from pi_dex.spec import ActionTimebase
from pi_dex.spec import BimanualActionSpec
from pi_dex.spec import HandNormalization
from pi_dex.spec import Rotation6DConvention


@pytest.fixture
def action_spec() -> BimanualActionSpec:
    return BimanualActionSpec(
        physical_horizon=2,
        timebase=ActionTimebase.RAW_CONTROL_60_HZ,
        control_frequency_hz=59.4,
        robot_id="POC22027",
        embodiment_version="sharpa_north_v1",
        coordinate_frame="north_base_v1",
        action_mode=ActionMode.ABSOLUTE,
        hand_normalization=HandNormalization.PER_HAND,
        rotation_6d_convention=Rotation6DConvention.MATRIX_FIRST_TWO_COLUMNS_COLUMN_MAJOR,
        kinematics_calibration_version="north_calibration_2026_08",
        command_semantics_version="sharpa_sdk_commanded_joint_position_absolute_v1",
        left_arm_joint_order=tuple(f"left_arm_j{index}" for index in range(7)),
        right_arm_joint_order=tuple(f"right_arm_j{index}" for index in range(7)),
        left_hand_joint_order=tuple(f"left_hand_j{index}" for index in range(22)),
        right_hand_joint_order=tuple(f"right_hand_j{index}" for index in range(22)),
        hand_mapping_version="sharpa_north_hand_mapping_v1",
        left_wrist_link="left_wrist",
        right_wrist_link="right_wrist",
        clock_domain="unix_realtime",
        max_group_timestamp_skew_ms=2.0,
        max_alignment_timestamp_error_ms=2.0,
        max_control_period_error_ms=8.0,
        max_observation_age_ms=50.0,
        max_command_lead_ms=25.0,
    )
