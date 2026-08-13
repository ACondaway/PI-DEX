import numpy as np
import pytest

from pi_dex.actions import LOGICAL_ACTION_DIM
from pi_dex.actions import MODEL_ACTION_DIM
from pi_dex.actions import VALID_ACTION_MASK
from pi_dex.actions import deinterleave
from pi_dex.actions import interleave
from pi_dex.actions import pad_action
from pi_dex.actions import unpad_action


def test_pad_and_unpad_actions_round_trip() -> None:
    logical_actions = np.arange(2 * 3 * LOGICAL_ACTION_DIM, dtype=np.float32).reshape(2, 3, LOGICAL_ACTION_DIM)

    model_actions = pad_action(logical_actions)
    restored_actions = unpad_action(model_actions)

    assert model_actions.shape == (2, 3, MODEL_ACTION_DIM)
    assert model_actions.dtype == logical_actions.dtype
    np.testing.assert_array_equal(model_actions[..., :LOGICAL_ACTION_DIM], logical_actions)
    np.testing.assert_array_equal(
        model_actions[..., LOGICAL_ACTION_DIM],
        np.zeros(model_actions.shape[:-1], dtype=model_actions.dtype),
    )
    np.testing.assert_array_equal(restored_actions, logical_actions)


def test_unpad_action_discards_nonsemantic_model_output() -> None:
    model_actions = np.zeros((2, MODEL_ACTION_DIM), dtype=np.float32)
    model_actions[..., -1] = 123.0

    logical_actions = unpad_action(model_actions)

    assert logical_actions.shape == (2, LOGICAL_ACTION_DIM)
    np.testing.assert_array_equal(logical_actions, np.zeros_like(logical_actions))


def test_public_action_conversions_reject_nonfinite_semantic_values() -> None:
    logical_actions = np.zeros((2, LOGICAL_ACTION_DIM), dtype=np.float32)
    logical_actions[0, 0] = np.nan

    with pytest.raises(ValueError, match=r"semantic action values.*finite"):
        pad_action(logical_actions)

    model_actions = np.zeros((2, MODEL_ACTION_DIM), dtype=np.float32)
    model_actions[0, 0] = np.inf
    with pytest.raises(ValueError, match=r"semantic action values.*finite"):
        unpad_action(model_actions)
    with pytest.raises(ValueError, match=r"semantic action values.*finite"):
        deinterleave(model_actions)


def test_unpad_action_ignores_nonfinite_nonsemantic_padding() -> None:
    model_actions = np.zeros((2, MODEL_ACTION_DIM), dtype=np.float32)
    model_actions[:, -1] = np.nan

    logical_actions = unpad_action(model_actions)

    assert np.all(np.isfinite(logical_actions))


def test_valid_action_mask_excludes_only_padding_dimension() -> None:
    assert len(VALID_ACTION_MASK) == MODEL_ACTION_DIM
    assert all(VALID_ACTION_MASK[:LOGICAL_ACTION_DIM])
    assert VALID_ACTION_MASK[-1] is False


@pytest.mark.parametrize(
    ("conversion", "width"),
    [
        (pad_action, LOGICAL_ACTION_DIM - 1),
        (unpad_action, MODEL_ACTION_DIM - 1),
    ],
)
def test_action_conversion_rejects_wrong_width(conversion, width: int) -> None:
    actions = np.zeros((4, width), dtype=np.float32)

    with pytest.raises(ValueError, match=rf"shape\[-1\].*expected {width + 1}.*got {width}"):
        conversion(actions)


@pytest.mark.parametrize("conversion", [pad_action, unpad_action])
def test_action_conversion_rejects_non_floating_dtype(conversion) -> None:
    width = LOGICAL_ACTION_DIM if conversion is pad_action else MODEL_ACTION_DIM
    actions = np.zeros((4, width), dtype=np.int32)

    with pytest.raises(TypeError, match=r"dtype.*floating.*int32"):
        conversion(actions)


def test_interleave_orders_left_then_right_for_each_physical_step() -> None:
    left_actions = np.zeros((2, MODEL_ACTION_DIM), dtype=np.float32)
    right_actions = np.zeros_like(left_actions)
    left_actions[:, 0] = [10.0, 11.0]
    right_actions[:, 0] = [20.0, 21.0]

    interleaved = interleave(left_actions, right_actions)

    assert interleaved.shape == (4, MODEL_ACTION_DIM)
    np.testing.assert_array_equal(interleaved[:, 0], [10.0, 20.0, 11.0, 21.0])


def test_interleave_and_deinterleave_round_trip_batched_actions() -> None:
    left_logical = np.arange(2 * 3 * LOGICAL_ACTION_DIM, dtype=np.float64).reshape(
        2,
        3,
        LOGICAL_ACTION_DIM,
    )
    right_logical = left_logical + 1_000.0
    left_actions = pad_action(left_logical)
    right_actions = pad_action(right_logical)

    model_actions = interleave(left_actions, right_actions)
    restored_left, restored_right = deinterleave(model_actions)

    assert model_actions.shape == (2, 6, MODEL_ACTION_DIM)
    assert model_actions.dtype == np.float64
    np.testing.assert_array_equal(restored_left, left_actions)
    np.testing.assert_array_equal(restored_right, right_actions)


def test_interleave_requires_finite_values_and_neutral_padding() -> None:
    left_actions = np.zeros((2, MODEL_ACTION_DIM), dtype=np.float32)
    right_actions = np.zeros_like(left_actions)
    left_actions[0, -1] = 1.0

    with pytest.raises(ValueError, match=r"padding dimensions.*zero"):
        interleave(left_actions, right_actions)

    left_actions[0, -1] = np.nan
    with pytest.raises(ValueError, match=r"model action values.*finite"):
        interleave(left_actions, right_actions)


def test_deinterleave_rejects_odd_model_horizon() -> None:
    model_actions = np.zeros((3, MODEL_ACTION_DIM), dtype=np.float32)

    with pytest.raises(ValueError, match=r"action horizon.*even.*got 3"):
        deinterleave(model_actions)


def test_deinterleave_rejects_empty_model_horizon() -> None:
    model_actions = np.zeros((0, MODEL_ACTION_DIM), dtype=np.float32)

    with pytest.raises(ValueError, match="non-empty action horizon"):
        deinterleave(model_actions)


def test_interleave_rejects_mismatched_hand_shapes() -> None:
    left_actions = np.zeros((2, MODEL_ACTION_DIM), dtype=np.float32)
    right_actions = np.zeros((3, MODEL_ACTION_DIM), dtype=np.float32)

    with pytest.raises(ValueError, match=r"matching shapes.*left_actions.*right_actions"):
        interleave(left_actions, right_actions)


def test_interleave_rejects_mismatched_hand_dtypes() -> None:
    left_actions = np.zeros((2, MODEL_ACTION_DIM), dtype=np.float32)
    right_actions = np.zeros((2, MODEL_ACTION_DIM), dtype=np.float64)

    with pytest.raises(TypeError, match=r"matching dtypes.*float32.*float64"):
        interleave(left_actions, right_actions)


def test_interleave_rejects_non_model_action_width() -> None:
    left_actions = np.zeros((2, LOGICAL_ACTION_DIM), dtype=np.float32)
    right_actions = np.zeros_like(left_actions)

    with pytest.raises(ValueError, match=rf"left_actions.shape\[-1\].*expected {MODEL_ACTION_DIM}"):
        interleave(left_actions, right_actions)
