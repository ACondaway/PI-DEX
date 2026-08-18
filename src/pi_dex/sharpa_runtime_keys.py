"""Key bridges between Sharpa North live SDK observations and OpenData HDF5 paths.

``examples/sharpa_north_sdk.py`` is a reference snapshot of the Zenoh/protobuf
``NorthDirect`` env used on hardware. It is not yet a ``BimanualController``
adapter (handoff phase 6). This module only records the observation key identity
shared by the live SDK dict and the on-disk SharpaOpenData schema so dataset and
runtime work stay aligned.
"""

from __future__ import annotations

# Live SDK observation keys emitted by NorthDirect._convert_north_observation_to_dict
# and the corresponding HDF5 groups/datasets in data/schema.json.
SDK_TO_HDF5_STATE = {
    "/state/left_arm/joint_angle": "state/left_arm/joint_angle",
    "/state/right_arm/joint_angle": "state/right_arm/joint_angle",
    "/state/left_hand/joint_angle": "state/left_hand/joint_angle",
    "/state/right_hand/joint_angle": "state/right_hand/joint_angle",
    "/state/motor/joint_angle": "state/motor/joint_angle",
    "/state/left_arm/tcp_forces": "state/left_arm/tcp_forces",
    "/state/right_arm/tcp_forces": "state/right_arm/tcp_forces",
    "/state/left_hand/effort": "state/left_hand/torque",
    "/state/right_hand/effort": "state/right_hand/torque",
    "/state/motor/joint_velocity": None,  # live-only unless a matching HDF5 field is reviewed
    "/state/motor/joint_effort": None,
}

SDK_TO_HDF5_VISION = {
    "/observe/vision/head/stereo/lefteye/rgb": "observe/vision/head/stereo/lefteye/rgb",
    "/observe/vision/head/stereo/righteye/rgb": "observe/vision/head/stereo/righteye/rgb",
    "/observe/vision/left_wrist/fisheye/rgb": "observe/vision/left_wrist/fisheye/rgb",
    "/observe/vision/right_wrist/fisheye/rgb": "observe/vision/right_wrist/fisheye/rgb",
}

# UhrActionBundle actuators filled by NorthDirect._send_action. A future
# BimanualController adapter must stage left/right arm+hand against one target
# timestamp; sequential publishes do not satisfy deployment.BimanualController.
SDK_ACTION_ACTUATORS = (
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "motor",
)

# Default Zenoh topics from the reference SDK. Site deployments may override.
# Robot stack (NUC start.sh + F6 inference + F2 moving) consumes action on
# DEFAULT_ACTION_TOPIC; PI-DEX fills it via ``pi_dex.robot_client``.
DEFAULT_OBSERVATION_TOPIC = "north_observation"
DEFAULT_ACTION_TOPIC = "inference/action"
DEFAULT_ACTION_PUB_DURATION_S = 0.01666
