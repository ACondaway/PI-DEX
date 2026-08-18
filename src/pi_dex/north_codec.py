"""Convert Sharpa North protobuf messages ↔ PI-DEX / NorthDirect SDK dicts.

Source of truth for the schema is ``examples/north.proto``; the first-party
module is regenerated ``pi_dex.north_pb2``. Conversion matches
``examples/sharpa_north_sdk.py`` (NorthDirect) so the robot-side Zenoh bridge
and offline ``realtime_*`` paths share the same keys.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from pi_dex import north_pb2 as pb
_VISION_SLOT_TO_SDK = (
    ("image_left", "/observe/vision/head/stereo/lefteye/rgb"),
    ("image_right", "/observe/vision/head/stereo/righteye/rgb"),
    ("fish_left", "/observe/vision/left_wrist/fisheye/rgb"),
    ("fish_right", "/observe/vision/right_wrist/fisheye/rgb"),
)


def decode_jpeg_bgr_to_rgb(payload: bytes) -> np.ndarray:
    """Decode a JPEG payload to contiguous HWC RGB uint8."""
    if not payload:
        raise ValueError("jpeg payload: empty")
    try:
        from turbojpeg import TurboJPEG

        bgr = TurboJPEG().decode(np.frombuffer(payload, dtype=np.uint8))
        return np.asarray(bgr[:, :, ::-1], dtype=np.uint8, order="C")
    except Exception:
        from PIL import Image
        import io

        rgb = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"), dtype=np.uint8)
        return np.asarray(rgb, order="C")


def north_observation_to_sdk_dict(
    north_obs: pb.NorthObservation,
    *,
    decode_images: bool = True,
) -> dict[str, Any]:
    """Mirror ``NorthDirect._convert_north_observation_to_dict``."""
    obs_dict: dict[str, Any] = {
        "timestamp": float(north_obs.timestamp),
        "reward": float(north_obs.reward),
        "on_sleep": bool(north_obs.on_sleep),
        "/task_code": 0,
    }

    if north_obs.HasField("robot_state"):
        robot_state = north_obs.robot_state
        if robot_state.HasField("left_arm"):
            left_arm = robot_state.left_arm
            obs_dict["/state/left_arm/joint_angle"] = list(left_arm.joint.position)
            force = [left_arm.wrench.force.x, left_arm.wrench.force.y, left_arm.wrench.force.z]
            torque = [
                left_arm.wrench.torque.x,
                left_arm.wrench.torque.y,
                left_arm.wrench.torque.z,
            ]
            obs_dict["/state/left_arm/tcp_forces"] = force + torque
        if robot_state.HasField("right_arm"):
            right_arm = robot_state.right_arm
            obs_dict["/state/right_arm/joint_angle"] = list(right_arm.joint.position)
            force = [right_arm.wrench.force.x, right_arm.wrench.force.y, right_arm.wrench.force.z]
            torque = [
                right_arm.wrench.torque.x,
                right_arm.wrench.torque.y,
                right_arm.wrench.torque.z,
            ]
            obs_dict["/state/right_arm/tcp_forces"] = force + torque
        if robot_state.HasField("left_hand"):
            left_hand = robot_state.left_hand
            obs_dict["/state/left_hand/joint_angle"] = list(left_hand.joint.position)
            obs_dict["/state/left_hand/effort"] = list(left_hand.joint.effort)
        if robot_state.HasField("right_hand"):
            right_hand = robot_state.right_hand
            obs_dict["/state/right_hand/joint_angle"] = list(right_hand.joint.position)
            obs_dict["/state/right_hand/effort"] = list(right_hand.joint.effort)
        if robot_state.HasField("motor"):
            motors = robot_state.motor.motors
            obs_dict["/state/motor/joint_angle"] = [motor.position for motor in motors]
            obs_dict["/state/motor/joint_velocity"] = [motor.velocity for motor in motors]
            obs_dict["/state/motor/joint_effort"] = [motor.torque for motor in motors]

    if north_obs.HasField("vision") and decode_images:
        vision = north_obs.vision
        for field_name, sdk_key in _VISION_SLOT_TO_SDK:
            if vision.HasField(field_name):
                compressed = getattr(vision, field_name)
                obs_dict[sdk_key] = decode_jpeg_bgr_to_rgb(bytes(compressed.data))

    if north_obs.HasField("mode"):
        mode = north_obs.mode
        obs_dict["mode"] = {
            "operation_mode": mode.operation_mode,
            "state": mode.state,
            "sub_state": mode.sub_state,
        }
        obs_dict["/mode/act"] = mode.state
        obs_dict["/mode/sub_act"] = mode.sub_state

    # proto3 scalar: no field presence; treat non-empty as set.
    if north_obs.language:
        obs_dict["/language"] = north_obs.language

    return obs_dict


def parse_north_observation(payload: bytes, *, decode_images: bool = True) -> dict[str, Any]:
    """Parse a Zenoh payload into an SDK observation dict."""
    north_obs = pb.NorthObservation()
    north_obs.ParseFromString(payload)
    return north_observation_to_sdk_dict(north_obs, decode_images=decode_images)


def sdk_step_action_to_uhr_bundle(
    step_action: Mapping[str, Any],
    *,
    timestamp_s: float | None = None,
    language: str | None = None,
) -> pb.UhrActionBundle:
    """Encode one paced step (actuator → position dict) as ``UhrActionBundle``."""
    bundle = pb.UhrActionBundle()
    now = time.time() if timestamp_s is None else float(timestamp_s)

    if "left_hand" in step_action:
        _fill_joint(bundle.left_glove.joint, step_action["left_hand"])
        _set_header_timestamp(bundle.left_glove.header, now)
    if "right_hand" in step_action:
        _fill_joint(bundle.right_glove.joint, step_action["right_hand"])
        _set_header_timestamp(bundle.right_glove.header, now)
    if "left_arm" in step_action:
        _fill_joint(bundle.left_arm.joint, step_action["left_arm"])
        _set_header_timestamp(bundle.left_arm.header, now)
    if "right_arm" in step_action:
        _fill_joint(bundle.right_arm.joint, step_action["right_arm"])
        _set_header_timestamp(bundle.right_arm.header, now)
    if "motor" in step_action:
        motor = bundle.motor
        motor.timestamp = int(now * 1000)
        _fill_motor_commands(motor, step_action["motor"])
    if language:
        bundle.language = language
    return bundle


def sdk_action_chunk_to_step_dicts(
    actions_dict: Mapping[str, list[list[float]]],
) -> list[dict[str, Any]]:
    """Convert NorthDirect-style chunk lists into paced per-step actuator dicts."""
    chunk_size: int | None = None
    action_keys: list[str] = []
    for key, values in actions_dict.items():
        if "/action" not in key:
            continue
        action_keys.append(key)
        if chunk_size is None:
            chunk_size = len(values)
        elif len(values) != chunk_size:
            raise ValueError("All action lists must have the same length")
    if not chunk_size:
        return []

    steps: list[dict[str, Any]] = []
    for index in range(chunk_size):
        raw = {key: actions_dict[key][index] for key in action_keys}
        steps.append(_prepare_actuator_dict(raw))
    return steps


def encode_uhr_action_bundle(bundle: pb.UhrActionBundle) -> bytes:
    """Serialize an action bundle for Zenoh publish."""
    return bundle.SerializeToString()


def build_synthetic_north_observation(
    *,
    timestamp_s: float = 1.0,
    language: str = "probe",
    jpeg_rgb: bytes | None = None,
    arm_dim: int = 7,
    hand_dim: int = 22,
    motor_dim: int = 7,
) -> pb.NorthObservation:
    """Build a minimal ``NorthObservation`` for codec / client smoke tests."""
    obs = pb.NorthObservation()
    obs.timestamp = timestamp_s
    obs.language = language
    rs = obs.robot_state
    for arm_field in ("left_arm", "right_arm"):
        arm = getattr(rs, arm_field)
        arm.joint.position.extend([0.01 * (i + 1) for i in range(arm_dim)])
        arm.wrench.force.x = 0.0
        arm.wrench.force.y = 0.0
        arm.wrench.force.z = 0.0
        arm.wrench.torque.x = 0.0
        arm.wrench.torque.y = 0.0
        arm.wrench.torque.z = 0.0
    for hand_field in ("left_hand", "right_hand"):
        hand = getattr(rs, hand_field)
        hand.joint.position.extend([0.02 * (i + 1) for i in range(hand_dim)])
        hand.joint.effort.extend([0.0] * hand_dim)
    for motor_id in range(motor_dim):
        status = rs.motor.motors.add()
        status.id = motor_id + 1
        status.position = 0.03 * (motor_id + 1)
        status.velocity = 0.0
        status.torque = 0.0

    if jpeg_rgb is not None:
        vision = obs.vision
        for field_name, _ in _VISION_SLOT_TO_SDK:
            image = getattr(vision, field_name)
            image.format = "jpeg"
            image.data = jpeg_rgb
    return obs


def _prepare_actuator_dict(action_values: Mapping[str, Any]) -> dict[str, Any]:
    action_dict: dict[str, Any] = {}
    for key, value in action_values.items():
        parts = key.split("/")
        if len(parts) < 4:
            continue
        actuator = parts[2]
        if actuator not in action_dict:
            action_dict[actuator] = {}
        action_dict[actuator]["position"] = list(value)
    return action_dict


def _fill_joint(joint_proto: pb.Joint, joint_dict: Mapping[str, Any]) -> None:
    if "position" in joint_dict:
        joint_proto.position[:] = [float(x) for x in joint_dict["position"]]
    if "velocity" in joint_dict:
        joint_proto.velocity[:] = [float(x) for x in joint_dict["velocity"]]
    if "effort" in joint_dict:
        joint_proto.effort[:] = [float(x) for x in joint_dict["effort"]]
    if "name" in joint_dict:
        joint_proto.name[:] = list(joint_dict["name"])


def _fill_motor_commands(motor_proto: pb.MotorsCommand, motor_dict: Mapping[str, Any]) -> None:
    if "position" not in motor_dict:
        return
    for index, pos in enumerate(motor_dict["position"]):
        cmd = motor_proto.commands.add()
        cmd.motor_id = index + 1
        cmd.command = "position"
        cmd.value = float(pos)


def _set_header_timestamp(header: pb.Header, timestamp: float) -> None:
    sec = int(timestamp)
    nanosec = int((timestamp - sec) * 1e9)
    header.stamp.sec = sec
    header.stamp.nanosec = nanosec
