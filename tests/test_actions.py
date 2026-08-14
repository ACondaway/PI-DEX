import numpy as np
import pytest

from pi_dex.actions import CARTESIAN_LOGICAL_ACTION_DIM
from pi_dex.actions import JOINT_LOGICAL_ACTION_DIM
from pi_dex.actions import LOGICAL_ACTION_DIM
from pi_dex.actions import MODEL_ACTION_DIM
from pi_dex.actions import VALID_ACTION_MASK
from pi_dex.actions import ActionRepresentation
from pi_dex.actions import deinterleave
from pi_dex.actions import interleave
from pi_dex.actions import pad_action
from pi_dex.actions import unpad_action
from pi_dex.actions import valid_action_mask


@pytest.mark.parametrize("representation", list(ActionRepresentation))
def test_pad_and_unpad_actions_round_trip(representation: ActionRepresentation) -> None:
    logical_dim = representation.logical_action_dim
    logical_actions = np.arange(2 * 3 * logical_dim, dtype=np.float32).reshape(2, 3, logical_dim)

    model_actions = pad_action(logical_actions, representation=representation)
    restored_actions = unpad_action(model_actions, representation=representation)

    assert model_actions.shape == (2, 3, MODEL_ACTION_DIM)
    assert model_actions.dtype == logical_actions.dtype
    np.testing.assert_array_equal(model_actions[..., :logical_dim], logical_actions)
    np.testing.assert_array_equal(
        model_actions[..., logical_dim:],
        np.zeros((*model_actions.shape[:-1], MODEL_ACTION_DIM - logical_dim), dtype=model_actions.dtype),
    )
    np.testing.assert_array_equal(restored_actions, logical_actions)


@pytest.mark.parametrize("representation", list(ActionRepresentation))
def test_unpad_action_discards_nonsemantic_model_output(representation: ActionRepresentation) -> None:
    model_actions = np.zeros((2, MODEL_ACTION_DIM), dtype=np.float32)
    model_actions[..., representation.logical_action_dim :] = 123.0

    logical_actions = unpad_action(model_actions, representation=representation)

    assert logical_actions.shape == (2, representation.logical_action_dim)
    np.testing.assert_array_equal(logical_actions, np.zeros_like(logical_actions))


@pytest.mark.parametrize("representation", list(ActionRepresentation))
def test_public_action_conversions_reject_nonfinite_semantic_values(
    representation: ActionRepresentation,
) -> None:
    logical_actions = np.zeros((2, representation.logical_action_dim), dtype=np.float32)
    logical_actions[0, 0] = np.nan

    with pytest.raises(ValueError, match=r"semantic action values.*finite"):
        pad_action(logical_actions, representation=representation)

    model_actions = np.zeros((2, MODEL_ACTION_DIM), dtype=np.float32)
    model_actions[0, 0] = np.inf
    with pytest.raises(ValueError, match=r"semantic action values.*finite"):
        unpad_action(model_actions, representation=representation)
    with pytest.raises(ValueError, match=r"semantic action values.*finite"):
        deinterleave(model_actions, representation=representation)


@pytest.mark.parametrize("representation", list(ActionRepresentation))
def test_unpad_action_ignores_nonfinite_nonsemantic_padding(representation: ActionRepresentation) -> None:
    model_actions = np.zeros((2, MODEL_ACTION_DIM), dtype=np.float32)
    model_actions[:, representation.logical_action_dim :] = np.nan

    logical_actions = unpad_action(model_actions, representation=representation)

    assert np.all(np.isfinite(logical_actions))


def test_cartesian_compatibility_aliases_retain_original_layout() -> None:
    assert LOGICAL_ACTION_DIM == CARTESIAN_LOGICAL_ACTION_DIM == 31
    assert len(VALID_ACTION_MASK) == MODEL_ACTION_DIM
    assert all(VALID_ACTION_MASK[:LOGICAL_ACTION_DIM])
    assert VALID_ACTION_MASK[-1] is False


@pytest.mark.parametrize("representation", list(ActionRepresentation))
def test_valid_action_mask_excludes_representation_padding(representation: ActionRepresentation) -> None:
    mask = valid_action_mask(representation)

    assert len(mask) == MODEL_ACTION_DIM
    assert all(mask[: representation.logical_action_dim])
    assert not any(mask[representation.logical_action_dim :])


def test_representation_widths_are_stable() -> None:
    assert ActionRepresentation.CARTESIAN_31D.logical_action_dim == CARTESIAN_LOGICAL_ACTION_DIM == 31
    assert ActionRepresentation.JOINT_29D.logical_action_dim == JOINT_LOGICAL_ACTION_DIM == 29


def test_valid_action_mask_rejects_untyped_representation() -> None:
    with pytest.raises(TypeError, match="ActionRepresentation"):
        valid_action_mask("joint_29d")


@pytest.mark.parametrize("representation", list(ActionRepresentation))
@pytest.mark.parametrize("conversion", [pad_action, unpad_action])
def test_action_conversion_rejects_wrong_width(
    conversion,
    representation: ActionRepresentation,
) -> None:
    expected_width = representation.logical_action_dim if conversion is pad_action else MODEL_ACTION_DIM
    width = expected_width - 1
    actions = np.zeros((4, width), dtype=np.float32)

    with pytest.raises(ValueError, match=rf"shape\[-1\].*expected {expected_width}.*got {width}"):
        conversion(actions, representation=representation)


@pytest.mark.parametrize("representation", list(ActionRepresentation))
@pytest.mark.parametrize("conversion", [pad_action, unpad_action])
def test_action_conversion_rejects_non_floating_dtype(conversion, representation: ActionRepresentation) -> None:
    width = representation.logical_action_dim if conversion is pad_action else MODEL_ACTION_DIM
    actions = np.zeros((4, width), dtype=np.int32)

    with pytest.raises(TypeError, match=r"dtype.*floating.*int32"):
        conversion(actions, representation=representation)


@pytest.mark.parametrize(
    "conversion_args",
    [
        (pad_action, (np.zeros((1, CARTESIAN_LOGICAL_ACTION_DIM), dtype=np.float32),)),
        (unpad_action, (np.zeros((1, MODEL_ACTION_DIM), dtype=np.float32),)),
        (
            interleave,
            (
                np.zeros((1, MODEL_ACTION_DIM), dtype=np.float32),
                np.zeros((1, MODEL_ACTION_DIM), dtype=np.float32),
            ),
        ),
        (deinterleave, (np.zeros((2, MODEL_ACTION_DIM), dtype=np.float32),)),
    ],
)
def test_action_conversions_require_explicit_representation(conversion_args) -> None:
    conversion, args = conversion_args

    with pytest.raises(TypeError, match="representation"):
        conversion(*args)


def test_interleave_orders_left_then_right_for_each_physical_step() -> None:
    left_actions = np.zeros((2, MODEL_ACTION_DIM), dtype=np.float32)
    right_actions = np.zeros_like(left_actions)
    left_actions[:, 0] = [10.0, 11.0]
    right_actions[:, 0] = [20.0, 21.0]

    interleaved = interleave(
        left_actions,
        right_actions,
        representation=ActionRepresentation.CARTESIAN_31D,
    )

    assert interleaved.shape == (4, MODEL_ACTION_DIM)
    np.testing.assert_array_equal(interleaved[:, 0], [10.0, 20.0, 11.0, 21.0])


@pytest.mark.parametrize("representation", list(ActionRepresentation))
def test_interleave_and_deinterleave_round_trip_batched_actions(
    representation: ActionRepresentation,
) -> None:
    logical_dim = representation.logical_action_dim
    left_logical = np.arange(2 * 3 * logical_dim, dtype=np.float64).reshape(
        2,
        3,
        logical_dim,
    )
    right_logical = left_logical + 1_000.0
    left_actions = pad_action(left_logical, representation=representation)
    right_actions = pad_action(right_logical, representation=representation)

    model_actions = interleave(left_actions, right_actions, representation=representation)
    restored_left, restored_right = deinterleave(model_actions, representation=representation)

    assert model_actions.shape == (2, 6, MODEL_ACTION_DIM)
    assert model_actions.dtype == np.float64
    np.testing.assert_array_equal(restored_left, left_actions)
    np.testing.assert_array_equal(restored_right, right_actions)


@pytest.mark.parametrize("representation", list(ActionRepresentation))
def test_interleave_requires_finite_values_and_neutral_padding(
    representation: ActionRepresentation,
) -> None:
    left_actions = np.zeros((2, MODEL_ACTION_DIM), dtype=np.float32)
    right_actions = np.zeros_like(left_actions)
    left_actions[0, representation.logical_action_dim] = 1.0

    with pytest.raises(ValueError, match=r"padding dimensions.*zero"):
        interleave(
            left_actions,
            right_actions,
            representation=representation,
        )

    left_actions[0, representation.logical_action_dim] = np.nan
    with pytest.raises(ValueError, match=r"model action values.*finite"):
        interleave(
            left_actions,
            right_actions,
            representation=representation,
        )


def test_deinterleave_rejects_odd_model_horizon() -> None:
    model_actions = np.zeros((3, MODEL_ACTION_DIM), dtype=np.float32)

    with pytest.raises(ValueError, match=r"action horizon.*even.*got 3"):
        deinterleave(model_actions, representation=ActionRepresentation.CARTESIAN_31D)


def test_deinterleave_rejects_empty_model_horizon() -> None:
    model_actions = np.zeros((0, MODEL_ACTION_DIM), dtype=np.float32)

    with pytest.raises(ValueError, match="non-empty action horizon"):
        deinterleave(model_actions, representation=ActionRepresentation.CARTESIAN_31D)


def test_interleave_rejects_mismatched_hand_shapes() -> None:
    left_actions = np.zeros((2, MODEL_ACTION_DIM), dtype=np.float32)
    right_actions = np.zeros((3, MODEL_ACTION_DIM), dtype=np.float32)

    with pytest.raises(ValueError, match=r"matching shapes.*left_actions.*right_actions"):
        interleave(
            left_actions,
            right_actions,
            representation=ActionRepresentation.CARTESIAN_31D,
        )


def test_interleave_rejects_mismatched_hand_dtypes() -> None:
    left_actions = np.zeros((2, MODEL_ACTION_DIM), dtype=np.float32)
    right_actions = np.zeros((2, MODEL_ACTION_DIM), dtype=np.float64)

    with pytest.raises(TypeError, match=r"matching dtypes.*float32.*float64"):
        interleave(
            left_actions,
            right_actions,
            representation=ActionRepresentation.CARTESIAN_31D,
        )


def test_interleave_rejects_non_model_action_width() -> None:
    left_actions = np.zeros((2, LOGICAL_ACTION_DIM), dtype=np.float32)
    right_actions = np.zeros_like(left_actions)

    with pytest.raises(ValueError, match=rf"left_actions.shape\[-1\].*expected {MODEL_ACTION_DIM}"):
        interleave(
            left_actions,
            right_actions,
            representation=ActionRepresentation.CARTESIAN_31D,
        )
