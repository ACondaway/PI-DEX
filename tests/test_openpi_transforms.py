import numpy as np
import pytest

from pi_dex.core.actions import MODEL_ACTION_DIM
from pi_dex.core.actions import ActionRepresentation
from pi_dex.training.openpi_transforms import PackBimanualActions
from pi_dex.training.openpi_transforms import UnpackBimanualActions
from pi_dex.training.openpi_transforms import ValidateBimanualSample
from pi_dex.core.spec import BimanualActionSpec
from tests.helpers import spec_for_representation


def test_sample_validator_accepts_training_and_inference_boundaries(
    action_spec: BimanualActionSpec,
) -> None:
    validator = ValidateBimanualSample(
        physical_horizon=2,
        action_representation=action_spec.action_representation,
    )
    state = np.ones((8,), dtype=np.float32)
    actions = np.zeros((2, action_spec.logical_action_dim), dtype=np.float32)

    training = validator(
        {
            "state": state,
            "left_actions": actions,
            "right_actions": actions.copy(),
        }
    )
    inference = validator({"state": state})

    assert training["left_actions"].shape == (2, action_spec.logical_action_dim)
    assert set(inference) == {"state"}


def test_sample_validator_rejects_state_width_different_from_checkpoint_stats(
    action_spec: BimanualActionSpec,
) -> None:
    validator = ValidateBimanualSample(
        physical_horizon=2,
        action_representation=action_spec.action_representation,
        state_dim=4,
    )

    with pytest.raises(ValueError, match=r"state.*expected \(4,\).+got \(3,\)"):
        validator({"state": np.zeros((3,), dtype=np.float32)})


@pytest.mark.parametrize(
    ("field_name", "value", "error", "message"),
    [
        ("state", np.zeros((2, 4), dtype=np.float32), ValueError, r"state.*ndim"),
        ("state", np.zeros((4,), dtype=np.int32), TypeError, r"state.*dtype"),
        (
            "left_actions",
            np.zeros(
                (1, ActionRepresentation.CARTESIAN_31D.logical_action_dim),
                dtype=np.float32,
            ),
            ValueError,
            r"left_actions.*shape",
        ),
    ],
)
def test_sample_validator_rejects_invalid_unbatched_shapes_and_dtypes(
    action_spec: BimanualActionSpec,
    field_name: str,
    value: np.ndarray,
    error: type[Exception],
    message: str,
) -> None:
    sample = {
        "state": np.zeros((4,), dtype=np.float32),
        "left_actions": np.zeros((2, action_spec.logical_action_dim), dtype=np.float32),
        "right_actions": np.zeros((2, action_spec.logical_action_dim), dtype=np.float32),
    }
    sample[field_name] = value

    with pytest.raises(error, match=message):
        ValidateBimanualSample(
            physical_horizon=2,
            action_representation=action_spec.action_representation,
        )(sample)


@pytest.mark.parametrize("representation", list(ActionRepresentation))
def test_pack_and_unpack_bimanual_actions_round_trip(
    action_spec: BimanualActionSpec,
    representation: ActionRepresentation,
) -> None:
    spec = spec_for_representation(action_spec, representation)
    left_actions = np.arange(
        2 * spec.logical_action_dim,
        dtype=np.float32,
    ).reshape(2, spec.logical_action_dim)
    right_actions = left_actions + 1_000.0
    training_sample = {
        "state": np.ones((8,), dtype=np.float32),
        "left_actions": left_actions,
        "right_actions": right_actions,
    }

    packed_sample = PackBimanualActions(representation)(training_sample)
    restored_sample = UnpackBimanualActions(representation)(packed_sample)

    assert packed_sample["actions"].shape == (4, MODEL_ACTION_DIM)
    np.testing.assert_array_equal(
        packed_sample["actions"][:, spec.logical_action_dim :],
        np.zeros((4, MODEL_ACTION_DIM - spec.logical_action_dim), dtype=np.float32),
    )
    np.testing.assert_array_equal(restored_sample["left_actions"], left_actions)
    np.testing.assert_array_equal(restored_sample["right_actions"], right_actions)
    np.testing.assert_array_equal(restored_sample["state"], training_sample["state"])
    assert "left_actions" in training_sample
    assert "actions" not in training_sample


@pytest.mark.parametrize("representation", list(ActionRepresentation))
def test_unpack_discards_nonzero_model_padding(representation: ActionRepresentation) -> None:
    model_actions = np.zeros((4, MODEL_ACTION_DIM), dtype=np.float32)
    model_actions[:, representation.logical_action_dim :] = np.nan

    unpacked = UnpackBimanualActions(representation)({"actions": model_actions})

    assert np.all(np.isfinite(unpacked["left_actions"]))
    assert np.all(np.isfinite(unpacked["right_actions"]))


def test_pack_rejects_existing_model_actions_field(action_spec: BimanualActionSpec) -> None:
    logical_actions = np.zeros((2, action_spec.logical_action_dim), dtype=np.float32)

    with pytest.raises(ValueError, match=r"already contains.*actions"):
        PackBimanualActions(action_spec.action_representation)(
            {
                "actions": np.zeros((4, MODEL_ACTION_DIM), dtype=np.float32),
                "left_actions": logical_actions,
                "right_actions": logical_actions,
            }
        )


def test_pack_is_noop_for_inference_observation_without_targets(
    action_spec: BimanualActionSpec,
) -> None:
    observation = {"state": np.ones((8,), dtype=np.float32)}

    transformed = PackBimanualActions(action_spec.action_representation)(observation)

    np.testing.assert_array_equal(transformed["state"], observation["state"])
    assert "actions" not in transformed


def test_pack_rejects_only_one_hand_target(action_spec: BimanualActionSpec) -> None:
    with pytest.raises(KeyError, match="right_actions"):
        PackBimanualActions(action_spec.action_representation)(
            {"left_actions": np.zeros((2, action_spec.logical_action_dim), dtype=np.float32)}
        )


def test_pack_rejects_prepacked_targets_without_per_hand_fields(
    action_spec: BimanualActionSpec,
) -> None:
    with pytest.raises(ValueError, match=r"reserved field.*actions"):
        PackBimanualActions(action_spec.action_representation)(
            {"actions": np.zeros((4, MODEL_ACTION_DIM), dtype=np.float32)}
        )


def test_pack_rejects_nonfinite_semantic_targets(action_spec: BimanualActionSpec) -> None:
    left_actions = np.zeros((2, action_spec.logical_action_dim), dtype=np.float32)
    right_actions = np.zeros_like(left_actions)
    left_actions[0, 5] = np.nan

    with pytest.raises(ValueError, match=r"left_actions.*finite"):
        PackBimanualActions(action_spec.action_representation)(
            {"left_actions": left_actions, "right_actions": right_actions}
        )
