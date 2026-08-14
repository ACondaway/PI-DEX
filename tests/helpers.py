"""Shared constructors for representation-aware PI-DEX tests."""

import dataclasses

from pi_dex.actions import ActionRepresentation
from pi_dex.spec import BimanualActionSpec


def spec_for_representation(
    spec: BimanualActionSpec,
    representation: ActionRepresentation,
) -> BimanualActionSpec:
    """Return a fixture-derived spec with representation-specific fields."""
    if representation is ActionRepresentation.CARTESIAN_31D:
        return dataclasses.replace(spec, action_representation=representation)
    return dataclasses.replace(
        spec,
        action_representation=representation,
        coordinate_frame=None,
        rotation_6d_convention=None,
        kinematics_calibration_version=None,
        left_wrist_link=None,
        right_wrist_link=None,
    )
