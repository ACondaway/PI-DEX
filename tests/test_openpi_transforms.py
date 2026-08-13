import numpy as np
import pytest

from pi_dex.actions import LOGICAL_ACTION_DIM
from pi_dex.actions import MODEL_ACTION_DIM
from pi_dex.openpi_transforms import PackBimanualActions
from pi_dex.openpi_transforms import UnpackBimanualActions
from pi_dex.openpi_transforms import ValidateBimanualSample


def test_sample_validator_accepts_training_and_inference_boundaries() -> None:
    validator = ValidateBimanualSample(physical_horizon=2)
    state = np.ones((8,), dtype=np.float32)
    actions = np.zeros((2, LOGICAL_ACTION_DIM), dtype=np.float32)

    training = validator(
        {
            "state": state,
            "left_actions": actions,
            "right_actions": actions.copy(),
        }
    )
    inference = validator({"state": state})

    assert training["left_actions"].shape == (2, LOGICAL_ACTION_DIM)
    assert set(inference) == {"state"}


def test_sample_validator_rejects_state_width_different_from_checkpoint_stats() -> None:
    validator = ValidateBimanualSample(physical_horizon=2, state_dim=4)

    with pytest.raises(ValueError, match=r"state.*expected \(4,\).+got \(3,\)"):
        validator({"state": np.zeros((3,), dtype=np.float32)})


@pytest.mark.parametrize(
    ("field_name", "value", "error", "message"),
    [
        ("state", np.zeros((2, 4), dtype=np.float32), ValueError, r"state.*ndim"),
        ("state", np.zeros((4,), dtype=np.int32), TypeError, r"state.*dtype"),
        (
            "left_actions",
            np.zeros((1, LOGICAL_ACTION_DIM), dtype=np.float32),
            ValueError,
            r"left_actions.*shape",
        ),
    ],
)
def test_sample_validator_rejects_invalid_unbatched_shapes_and_dtypes(
    field_name: str,
    value: np.ndarray,
    error: type[Exception],
    message: str,
) -> None:
    sample = {
        "state": np.zeros((4,), dtype=np.float32),
        "left_actions": np.zeros((2, LOGICAL_ACTION_DIM), dtype=np.float32),
        "right_actions": np.zeros((2, LOGICAL_ACTION_DIM), dtype=np.float32),
    }
    sample[field_name] = value

    with pytest.raises(error, match=message):
        ValidateBimanualSample(physical_horizon=2)(sample)


def test_pack_and_unpack_bimanual_actions_round_trip() -> None:
    left_actions = np.arange(2 * LOGICAL_ACTION_DIM, dtype=np.float32).reshape(2, LOGICAL_ACTION_DIM)
    right_actions = left_actions + 1_000.0
    training_sample = {
        "state": np.ones((8,), dtype=np.float32),
        "left_actions": left_actions,
        "right_actions": right_actions,
    }

    packed_sample = PackBimanualActions()(training_sample)
    restored_sample = UnpackBimanualActions()(packed_sample)

    assert packed_sample["actions"].shape == (4, MODEL_ACTION_DIM)
    np.testing.assert_array_equal(
        packed_sample["actions"][:, -1],
        np.zeros((4,), dtype=np.float32),
    )
    np.testing.assert_array_equal(restored_sample["left_actions"], left_actions)
    np.testing.assert_array_equal(restored_sample["right_actions"], right_actions)
    np.testing.assert_array_equal(restored_sample["state"], training_sample["state"])
    assert "left_actions" in training_sample
    assert "actions" not in training_sample


def test_unpack_discards_nonzero_model_padding() -> None:
    model_actions = np.zeros((4, MODEL_ACTION_DIM), dtype=np.float32)
    model_actions[:, -1] = np.nan

    unpacked = UnpackBimanualActions()({"actions": model_actions})

    assert np.all(np.isfinite(unpacked["left_actions"]))
    assert np.all(np.isfinite(unpacked["right_actions"]))


def test_pack_rejects_existing_model_actions_field() -> None:
    logical_actions = np.zeros((2, LOGICAL_ACTION_DIM), dtype=np.float32)

    with pytest.raises(ValueError, match=r"already contains.*actions"):
        PackBimanualActions()(
            {
                "actions": np.zeros((4, MODEL_ACTION_DIM), dtype=np.float32),
                "left_actions": logical_actions,
                "right_actions": logical_actions,
            }
        )


def test_pack_is_noop_for_inference_observation_without_targets() -> None:
    observation = {"state": np.ones((8,), dtype=np.float32)}

    transformed = PackBimanualActions()(observation)

    np.testing.assert_array_equal(transformed["state"], observation["state"])
    assert "actions" not in transformed


def test_pack_rejects_only_one_hand_target() -> None:
    with pytest.raises(KeyError, match="right_actions"):
        PackBimanualActions()(
            {"left_actions": np.zeros((2, LOGICAL_ACTION_DIM), dtype=np.float32)}
        )


def test_pack_rejects_prepacked_targets_without_per_hand_fields() -> None:
    with pytest.raises(ValueError, match=r"reserved field.*actions"):
        PackBimanualActions()(
            {"actions": np.zeros((4, MODEL_ACTION_DIM), dtype=np.float32)}
        )


def test_pack_rejects_nonfinite_semantic_targets() -> None:
    left_actions = np.zeros((2, LOGICAL_ACTION_DIM), dtype=np.float32)
    right_actions = np.zeros_like(left_actions)
    left_actions[0, 5] = np.nan

    with pytest.raises(ValueError, match=r"left_actions.*finite"):
        PackBimanualActions()(
            {"left_actions": left_actions, "right_actions": right_actions}
        )
