"""PI-DEX extensions for bimanual dexterous manipulation.

Package layout:
  ``core``      action/spec/normalization contracts
  ``data``      Sharpa datasets, splits, norm compute
  ``training``  OpenPI training, checkpoints, launchers
  ``weights``   π0.5 convert / parity
  ``serve``     WebSocket policy server + deployment wire
  ``robot``     real-robot Runtime / North Zenoh client
"""

from pi_dex.core.actions import ARM_JOINT_DIM
from pi_dex.core.actions import CARTESIAN_LOGICAL_ACTION_DIM
from pi_dex.core.actions import HAND_JOINT_DIM
from pi_dex.core.actions import JOINT_ARM_HAND_DIM
from pi_dex.core.actions import JOINT_LOGICAL_ACTION_DIM
from pi_dex.core.actions import MODEL_ACTION_DIM
from pi_dex.core.actions import MOTOR_JOINT_DIM
from pi_dex.core.actions import PRETRAINED_MODEL_ACTION_DIM
from pi_dex.core.actions import WRIST_POSITION_DIM
from pi_dex.core.actions import WRIST_ROTATION_6D_DIM
from pi_dex.core.actions import ActionRepresentation
from pi_dex.core.actions import deinterleave
from pi_dex.core.actions import interleave
from pi_dex.core.actions import pad_action
from pi_dex.core.actions import unpad_action
from pi_dex.core.actions import valid_action_mask
from pi_dex.core.normalization import NORMALIZATION_FINGERPRINT_ALGORITHM
from pi_dex.core.normalization import normalization_state_dim
from pi_dex.core.normalization import normalization_stats_fingerprint
from pi_dex.core.normalization import validate_normalization_stats
from pi_dex.core.spec import ACTION_LAYOUT_VERSION
from pi_dex.core.spec import ACTION_METADATA_SCHEMA_VERSION
from pi_dex.core.spec import ActionMode
from pi_dex.core.spec import ActionTimebase
from pi_dex.core.spec import BimanualActionSpec
from pi_dex.core.spec import HandNormalization
from pi_dex.core.spec import Rotation6DConvention

__all__ = [
    "ACTION_LAYOUT_VERSION",
    "ACTION_METADATA_SCHEMA_VERSION",
    "ARM_JOINT_DIM",
    "CARTESIAN_LOGICAL_ACTION_DIM",
    "HAND_JOINT_DIM",
    "JOINT_ARM_HAND_DIM",
    "JOINT_LOGICAL_ACTION_DIM",
    "MODEL_ACTION_DIM",
    "MOTOR_JOINT_DIM",
    "NORMALIZATION_FINGERPRINT_ALGORITHM",
    "PRETRAINED_MODEL_ACTION_DIM",
    "WRIST_POSITION_DIM",
    "WRIST_ROTATION_6D_DIM",
    "ActionMode",
    "ActionRepresentation",
    "ActionTimebase",
    "BimanualActionSpec",
    "HandNormalization",
    "Rotation6DConvention",
    "deinterleave",
    "interleave",
    "normalization_state_dim",
    "normalization_stats_fingerprint",
    "pad_action",
    "unpad_action",
    "valid_action_mask",
    "validate_normalization_stats",
]
