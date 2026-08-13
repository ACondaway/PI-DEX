import dataclasses

import numpy as np
import pytest

from pi_dex.actions import HAND_JOINT_DIM
from pi_dex.actions import LOGICAL_ACTION_DIM
from pi_dex.sharpa_data import ARM_JOINT_DIM
from pi_dex.sharpa_data import AlignedTimeline
from pi_dex.sharpa_data import CommandedJointGroup
from pi_dex.sharpa_data import EpisodeActionProvenance
from pi_dex.sharpa_data import HandSide
from pi_dex.sharpa_data import derive_bimanual_logical_action_chunk
from pi_dex.sharpa_data import derive_logical_actions
from pi_dex.sharpa_data import select_commanded_joint_horizon
from pi_dex.spec import ActionMode
from pi_dex.spec import ActionTimebase
from pi_dex.spec import BimanualActionSpec

BASE_TIME_S = 1_800_000_000.0
RAW_PERIOD_S = 1.0 / 59.4
ALIGNED_PERIOD_S = 2.0 * RAW_PERIOD_S


class FakeKinematics:
    robot_id = "POC22027"
    embodiment_version = "sharpa_north_v1"
    calibration_version = "north_calibration_2026_08"
    coordinate_frame = "north_base_v1"
    rotation_6d_convention = "rotation_matrix_first_two_columns_column_major_v1"
    left_arm_joint_order = tuple(f"left_arm_j{index}" for index in range(7))
    right_arm_joint_order = tuple(f"right_arm_j{index}" for index in range(7))
    input_joint_unit = "rad"
    output_position_unit = "m"
    left_wrist_link = "left_wrist"
    right_wrist_link = "right_wrist"

    def wrist_pose(self, side: HandSide, arm_joint_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        side_offset = np.float32(0.1 if side is HandSide.RIGHT else 0.0)
        position = arm_joint_angles[:, :3] * np.float32(0.01) + side_offset
        identity_first_two_columns = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        rotation = np.tile(identity_first_two_columns, (arm_joint_angles.shape[0], 1))
        return position, rotation


def make_provenance() -> EpisodeActionProvenance:
    return EpisodeActionProvenance(
        robot_id="POC22027",
        embodiment_version="sharpa_north_v1",
        command_semantics_version="sharpa_sdk_commanded_joint_position_absolute_v1",
        hand_mapping_version="sharpa_north_hand_mapping_v1",
    )


def make_timeline(*, length: int = 3) -> AlignedTimeline:
    timestamps = BASE_TIME_S + np.arange(length, dtype=np.float64) * ALIGNED_PERIOD_S
    time = np.stack((timestamps, np.ones(length, dtype=np.float64)), axis=1)
    return AlignedTimeline("observe/vision/head/stereo/lefteye/time", time)


def make_group(
    field_name: str,
    *,
    raw_length: int = 6,
    aligned_index: np.ndarray | None = None,
    joint_order: tuple[str, ...] | None = None,
    time_origin_s: float = BASE_TIME_S,
    timestamp_offset_s: float = 0.0,
) -> CommandedJointGroup:
    width = ARM_JOINT_DIM if "_arm/" in field_name else HAND_JOINT_DIM
    side = "left" if "left_" in field_name else "right"
    part = "arm" if "_arm/" in field_name else "hand"
    if joint_order is None:
        joint_order = tuple(f"{side}_{part}_j{index}" for index in range(width))
    values = np.linspace(-0.5, 0.5, raw_length * width, dtype=np.float32).reshape(raw_length, width)
    timestamps = time_origin_s + np.arange(raw_length, dtype=np.float64) * RAW_PERIOD_S
    timestamps += timestamp_offset_s
    time = np.stack((timestamps, np.ones(raw_length, dtype=np.float64)), axis=1)
    if aligned_index is None:
        aligned_index = np.array([0, 2, raw_length - 1], dtype=np.int32)
    return CommandedJointGroup(field_name, joint_order, values, time, aligned_index)


def make_groups() -> tuple[CommandedJointGroup, ...]:
    default_index = np.array([0, 2, 4], dtype=np.int32)
    shifted_index = np.array([1, 3, 5], dtype=np.int32)
    shifted_origin = BASE_TIME_S - RAW_PERIOD_S
    return (
        make_group("action/left_arm/joint_angle", aligned_index=default_index),
        make_group(
            "action/left_hand/joint_angle",
            aligned_index=shifted_index,
            time_origin_s=shifted_origin,
            timestamp_offset_s=0.0002,
        ),
        make_group(
            "action/right_arm/joint_angle",
            aligned_index=default_index,
            timestamp_offset_s=0.0004,
        ),
        make_group(
            "action/right_hand/joint_angle",
            aligned_index=shifted_index,
            time_origin_s=shifted_origin,
            timestamp_offset_s=0.0006,
        ),
    )


def derive_chunk(
    action_spec: BimanualActionSpec,
    groups: tuple[CommandedJointGroup, ...],
    *,
    timeline: AlignedTimeline | None = None,
    provenance: EpisodeActionProvenance | None = None,
):
    left_arm, left_hand, right_arm, right_hand = groups
    return derive_bimanual_logical_action_chunk(
        aligned_timeline=timeline or make_timeline(),
        provenance=provenance or make_provenance(),
        left_arm=left_arm,
        left_hand=left_hand,
        right_arm=right_arm,
        right_hand=right_hand,
        start_aligned_frame=np.int32(1),
        spec=action_spec,
        kinematics=FakeKinematics(),
    )


def derive_left_actions(
    arm_joint_angles: np.ndarray,
    hand_joint_angles: np.ndarray,
    action_spec: BimanualActionSpec,
    kinematics: object,
) -> np.ndarray:
    return derive_logical_actions(
        HandSide.LEFT,
        arm_joint_angles,
        hand_joint_angles,
        provenance=make_provenance(),
        arm_joint_order=action_spec.left_arm_joint_order,
        hand_joint_order=action_spec.left_hand_joint_order,
        spec=action_spec,
        kinematics=kinematics,
    )


def test_raw_horizon_uses_group_aligned_index_then_consecutive_rows(action_spec: BimanualActionSpec) -> None:
    group = make_group(
        "action/left_arm/joint_angle",
        aligned_index=np.array([0, 2, 4], dtype=np.int32),
    )

    selected = select_commanded_joint_horizon(group, start_aligned_frame=np.int64(1), spec=action_spec)

    np.testing.assert_array_equal(selected.raw_indices, [2, 3])
    np.testing.assert_array_equal(selected.joint_angles, group.joint_angles[[2, 3]])
    assert selected.raw_indices.dtype == np.int64
    assert selected.timestamps_s.dtype == np.float64


def test_aligned_horizon_uses_recorded_indices_and_supports_repeats(action_spec: BimanualActionSpec) -> None:
    aligned_spec = dataclasses.replace(
        action_spec,
        timebase=ActionTimebase.ALIGNED_30_HZ,
        control_frequency_hz=29.7,
    )
    group = make_group(
        "action/left_arm/joint_angle",
        raw_length=5,
        aligned_index=np.array([0, 1, 1], dtype=np.int32),
    )

    selected = select_commanded_joint_horizon(group, start_aligned_frame=1, spec=aligned_spec)

    np.testing.assert_array_equal(selected.raw_indices, [1, 1])


def test_non_strict_two_to_one_timeline_uses_recorded_last_index(action_spec: BimanualActionSpec) -> None:
    group = make_group(
        "action/left_arm/joint_angle",
        raw_length=5,
        aligned_index=np.array([0, 2, 4], dtype=np.int32),
    )
    final_step_spec = dataclasses.replace(action_spec, physical_horizon=1)

    selected = select_commanded_joint_horizon(group, start_aligned_frame=2, spec=final_step_spec)

    np.testing.assert_array_equal(selected.raw_indices, [4])


def test_group_constructor_rejects_nonmonotonic_aligned_index() -> None:
    with pytest.raises(ValueError, match=r"aligned_index.*monotonically"):
        make_group(
            "action/left_arm/joint_angle",
            aligned_index=np.array([0, 4, 3], dtype=np.int32),
        )


def test_group_constructor_rejects_duplicate_raw_timestamps() -> None:
    group = make_group("action/left_arm/joint_angle")
    time = group.time.copy()
    time[2, 0] = time[1, 0]

    with pytest.raises(ValueError, match="strictly increasing timestamps"):
        CommandedJointGroup(
            group.field_name,
            group.joint_order,
            group.joint_angles,
            time,
            group.aligned_index,
        )


def test_group_constructor_requires_unique_nonempty_joint_order() -> None:
    valid_order = tuple(f"left_hand_j{index}" for index in range(HAND_JOINT_DIM))

    with pytest.raises(ValueError, match="exactly 22"):
        make_group("action/left_hand/joint_angle", joint_order=valid_order[:-1])

    with pytest.raises(ValueError, match="unique joint names"):
        make_group(
            "action/left_hand/joint_angle",
            joint_order=valid_order[:-1] + (valid_order[0],),
        )

    with pytest.raises(ValueError, match=r"\[4\].*non-empty"):
        make_group(
            "action/left_hand/joint_angle",
            joint_order=valid_order[:4] + (" ",) + valid_order[5:],
        )


def test_episode_arrays_cannot_reenable_writeable_flag() -> None:
    timeline = make_timeline()
    group = make_group("action/left_hand/joint_angle")

    for values in (timeline.time, group.joint_angles, group.time, group.aligned_index):
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values.flags.writeable = True


def test_group_constructor_rejects_measured_state_path() -> None:
    with pytest.raises(ValueError, match="expected one of"):
        make_group("state/left_arm/joint_angle")


def test_derive_bimanual_chunk_uses_each_groups_own_alignment(action_spec: BimanualActionSpec) -> None:
    groups = make_groups()

    chunk = derive_chunk(action_spec, groups)

    left_arm, left_hand, right_arm, right_hand = groups
    assert chunk.left_actions.shape == (2, LOGICAL_ACTION_DIM)
    assert chunk.right_actions.shape == (2, LOGICAL_ACTION_DIM)
    assert chunk.left_actions.dtype == np.float32
    assert chunk.right_actions.dtype == np.float32
    assert chunk.timestamps_s.shape == (action_spec.physical_horizon,)
    assert chunk.timestamps_s.dtype == np.float64
    np.testing.assert_array_equal(chunk.left_actions[:, -HAND_JOINT_DIM:], left_hand.joint_angles[[3, 4]])
    np.testing.assert_array_equal(chunk.right_actions[:, -HAND_JOINT_DIM:], right_hand.joint_angles[[3, 4]])
    np.testing.assert_allclose(chunk.left_actions[:, :3], left_arm.joint_angles[[2, 3], :3] * 0.01)
    np.testing.assert_allclose(chunk.right_actions[:, :3], right_arm.joint_angles[[2, 3], :3] * 0.01 + 0.1)
    expected_rotation = np.tile(
        np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        (action_spec.physical_horizon, 1),
    )
    np.testing.assert_array_equal(chunk.left_actions[:, 3:9], expected_rotation)
    np.testing.assert_allclose(chunk.timestamps_s, left_arm.time[[2, 3], 0])
    assert chunk.source_aligned_frame == 1


def test_derive_aligned_chunk_uses_canonical_timestamps(
    action_spec: BimanualActionSpec,
) -> None:
    aligned_spec = dataclasses.replace(
        action_spec,
        timebase=ActionTimebase.ALIGNED_30_HZ,
        control_frequency_hz=29.7,
    )
    timeline = make_timeline()

    chunk = derive_chunk(aligned_spec, make_groups(), timeline=timeline)

    np.testing.assert_array_equal(chunk.timestamps_s, timeline.time[1:3, 0])
    assert chunk.left_actions.shape == (2, LOGICAL_ACTION_DIM)
    assert chunk.right_actions.shape == (2, LOGICAL_ACTION_DIM)


def test_derive_chunk_rejects_cross_group_timestamp_skew(action_spec: BimanualActionSpec) -> None:
    left_arm, left_hand, right_arm, _ = make_groups()
    right_hand = make_group(
        "action/right_hand/joint_angle",
        aligned_index=np.array([1, 3, 5], dtype=np.int32),
        time_origin_s=BASE_TIME_S - RAW_PERIOD_S,
        timestamp_offset_s=0.01,
    )

    with pytest.raises(ValueError, match="timestamp skew"):
        derive_chunk(action_spec, (left_arm, left_hand, right_arm, right_hand))


def test_derive_chunk_rejects_synchronized_commands_offset_from_camera(
    action_spec: BimanualActionSpec,
) -> None:
    shifted_groups = []
    for group in make_groups():
        shifted_time = group.time.copy()
        shifted_time[:, 0] += 0.01
        shifted_groups.append(dataclasses.replace(group, time=shifted_time))

    with pytest.raises(ValueError, match="command/canonical timestamp alignment"):
        derive_chunk(action_spec, tuple(shifted_groups))


def test_derive_chunk_rejects_irregular_raw_control_cadence(
    action_spec: BimanualActionSpec,
) -> None:
    irregular_groups = []
    for group in make_groups():
        first_raw_index = int(group.aligned_index[1])
        irregular_time = group.time.copy()
        irregular_time[first_raw_index + 1 :, 0] += 0.02
        irregular_groups.append(dataclasses.replace(group, time=irregular_time))

    with pytest.raises(ValueError, match="control-period error"):
        derive_chunk(action_spec, tuple(irregular_groups))


def test_derive_chunk_rejects_group_with_wrong_role(action_spec: BimanualActionSpec) -> None:
    left_arm, left_hand, right_arm, right_hand = make_groups()

    with pytest.raises(ValueError, match=r"group role.*left_arm"):
        derive_chunk(action_spec, (right_arm, left_hand, left_arm, right_hand))


def test_derive_chunk_rejects_hand_columns_out_of_spec_order(
    action_spec: BimanualActionSpec,
) -> None:
    left_arm, left_hand, right_arm, right_hand = make_groups()
    left_hand = dataclasses.replace(left_hand, joint_order=tuple(reversed(left_hand.joint_order)))

    with pytest.raises(ValueError, match=r"left_hand.*joint_order"):
        derive_chunk(action_spec, (left_arm, left_hand, right_arm, right_hand))


def test_derive_chunk_rejects_wrong_hand_mapping_provenance(
    action_spec: BimanualActionSpec,
) -> None:
    provenance = dataclasses.replace(
        make_provenance(),
        hand_mapping_version="sharpa_north_hand_mapping_v2",
    )

    with pytest.raises(ValueError, match=r"provenance\.hand_mapping_version"):
        derive_chunk(action_spec, make_groups(), provenance=provenance)


def test_derive_chunk_rejects_group_n_different_from_canonical(action_spec: BimanualActionSpec) -> None:
    left_arm, left_hand, right_arm, right_hand = make_groups()
    left_arm = make_group(
        "action/left_arm/joint_angle",
        aligned_index=np.array([0, 2], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="canonical N=3"):
        derive_chunk(action_spec, (left_arm, left_hand, right_arm, right_hand))


@pytest.mark.parametrize("start_aligned_frame", [-1, 3])
def test_select_rejects_start_frame_outside_aligned_timeline(
    action_spec: BimanualActionSpec,
    start_aligned_frame: int,
) -> None:
    group = make_group("action/left_arm/joint_angle")

    with pytest.raises(ValueError, match="start_aligned_frame"):
        select_commanded_joint_horizon(
            group,
            start_aligned_frame=start_aligned_frame,
            spec=action_spec,
        )


def test_select_rejects_boolean_start_frame(action_spec: BimanualActionSpec) -> None:
    group = make_group("action/left_arm/joint_angle")

    with pytest.raises(TypeError, match="start_aligned_frame"):
        select_commanded_joint_horizon(group, start_aligned_frame=True, spec=action_spec)


def test_raw_horizon_rejects_last_frame_when_k_rows_do_not_remain(action_spec: BimanualActionSpec) -> None:
    group = make_group(
        "action/left_arm/joint_angle",
        raw_length=5,
        aligned_index=np.array([0, 2, 4], dtype=np.int32),
    )

    with pytest.raises(ValueError, match=r"raw horizon \[4, 6\).*exceeds M=5"):
        select_commanded_joint_horizon(group, start_aligned_frame=2, spec=action_spec)


def test_derive_actions_requires_explicit_matching_kinematics(action_spec: BimanualActionSpec) -> None:
    arm = np.zeros((2, ARM_JOINT_DIM), dtype=np.float32)
    hand = np.zeros((2, HAND_JOINT_DIM), dtype=np.float32)

    with pytest.raises(TypeError, match="explicit calibrated"):
        derive_left_actions(arm, hand, action_spec, None)


def test_derive_actions_rejects_non_absolute_semantics(action_spec: BimanualActionSpec) -> None:
    arm = np.zeros((2, ARM_JOINT_DIM), dtype=np.float32)
    hand = np.zeros((2, HAND_JOINT_DIM), dtype=np.float32)
    delta_spec = dataclasses.replace(action_spec, action_mode=ActionMode.DELTA)

    with pytest.raises(ValueError, match="only derive absolute"):
        derive_left_actions(arm, hand, delta_spec, FakeKinematics())


def test_derive_actions_rejects_wrong_robot_calibration(action_spec: BimanualActionSpec) -> None:
    arm = np.zeros((2, ARM_JOINT_DIM), dtype=np.float32)
    hand = np.zeros((2, HAND_JOINT_DIM), dtype=np.float32)
    kinematics = FakeKinematics()
    kinematics.robot_id = "POC99999"

    with pytest.raises(ValueError, match="robot_id"):
        derive_left_actions(arm, hand, action_spec, kinematics)


def test_derive_actions_rejects_nonorthogonal_rotation_6d(action_spec: BimanualActionSpec) -> None:
    class InvalidRotationKinematics(FakeKinematics):
        def wrist_pose(self, side: HandSide, arm_joint_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            position, rotation = super().wrist_pose(side, arm_joint_angles)
            rotation[:, 3:] = rotation[:, :3]
            return position, rotation

    arm = np.zeros((2, ARM_JOINT_DIM), dtype=np.float32)
    hand = np.zeros((2, HAND_JOINT_DIM), dtype=np.float32)

    with pytest.raises(ValueError, match="orthonormal"):
        derive_left_actions(arm, hand, action_spec, InvalidRotationKinematics())


def test_rotation_6d_uses_first_two_columns_in_column_major_order(
    action_spec: BimanualActionSpec,
) -> None:
    class QuarterTurnKinematics(FakeKinematics):
        def wrist_pose(self, side: HandSide, arm_joint_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            position, _ = super().wrist_pose(side, arm_joint_angles)
            first_two_columns = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32)
            return position, np.tile(first_two_columns, (arm_joint_angles.shape[0], 1))

    arm = np.zeros((action_spec.physical_horizon, ARM_JOINT_DIM), dtype=np.float32)
    hand = np.zeros((action_spec.physical_horizon, HAND_JOINT_DIM), dtype=np.float32)

    actions = derive_left_actions(arm, hand, action_spec, QuarterTurnKinematics())

    expected = np.tile(
        np.array([0.0, 1.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32),
        (action_spec.physical_horizon, 1),
    )
    np.testing.assert_array_equal(actions[:, 3:9], expected)


def test_direct_derive_requires_joint_order_and_hand_mapping_provenance(
    action_spec: BimanualActionSpec,
) -> None:
    arm = np.zeros((action_spec.physical_horizon, ARM_JOINT_DIM), dtype=np.float32)
    hand = np.zeros((action_spec.physical_horizon, HAND_JOINT_DIM), dtype=np.float32)

    with pytest.raises(ValueError, match="hand_joint_order"):
        derive_logical_actions(
            HandSide.LEFT,
            arm,
            hand,
            provenance=make_provenance(),
            arm_joint_order=action_spec.left_arm_joint_order,
            hand_joint_order=tuple(reversed(action_spec.left_hand_joint_order)),
            spec=action_spec,
            kinematics=FakeKinematics(),
        )

    wrong_mapping = dataclasses.replace(make_provenance(), hand_mapping_version="other_mapping")
    with pytest.raises(ValueError, match=r"provenance\.hand_mapping_version"):
        derive_logical_actions(
            HandSide.LEFT,
            arm,
            hand,
            provenance=wrong_mapping,
            arm_joint_order=action_spec.left_arm_joint_order,
            hand_joint_order=action_spec.left_hand_joint_order,
            spec=action_spec,
            kinematics=FakeKinematics(),
        )
