import dataclasses

import numpy as np
import pytest

from pi_dex.actions import ActionRepresentation
from pi_dex.normalization import normalization_stats_fingerprint
from pi_dex.normalization import normalization_state_dim
from pi_dex.normalization import validate_normalization_stats
from pi_dex.spec import BimanualActionSpec
from pi_dex.spec import HandNormalization
from tests.helpers import spec_for_representation


@dataclasses.dataclass(frozen=True)
class AttributeStats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None
    q99: np.ndarray | None


def make_stats_entry(
    width: int,
    *,
    offset: float = 0.0,
    dtype: np.dtype | type = np.float32,
) -> AttributeStats:
    mean = (np.arange(width, dtype=np.float64) + offset).astype(dtype)
    return AttributeStats(
        mean=mean,
        std=np.ones(width, dtype=dtype),
        q01=(mean.astype(np.float64) - 1.0).astype(dtype),
        q99=(mean.astype(np.float64) + 1.0).astype(dtype),
    )


def make_norm_stats(
    spec: BimanualActionSpec,
    *,
    dtype: np.dtype | type = np.float32,
) -> dict[str, object]:
    return {
        "state": make_stats_entry(4, dtype=dtype),
        "left_actions": make_stats_entry(spec.logical_action_dim, offset=10.0, dtype=dtype),
        "right_actions": make_stats_entry(spec.logical_action_dim, offset=20.0, dtype=dtype),
    }


def as_mapping(stats: AttributeStats) -> dict[str, np.ndarray | None]:
    return {
        "q99": stats.q99,
        "std": stats.std,
        "mean": stats.mean,
        "q01": stats.q01,
    }


@pytest.mark.parametrize("representation", list(ActionRepresentation))
def test_validation_accepts_attribute_objects_and_stat_mappings(
    action_spec: BimanualActionSpec,
    representation: ActionRepresentation,
) -> None:
    action_spec = spec_for_representation(action_spec, representation)
    norm_stats = make_norm_stats(action_spec)
    norm_stats["left_actions"] = as_mapping(norm_stats["left_actions"])

    validate_normalization_stats(norm_stats, action_spec)


def test_normalization_state_dim_returns_validated_exact_width(
    action_spec: BimanualActionSpec,
) -> None:
    assert normalization_state_dim(make_norm_stats(action_spec), action_spec) == 4


def test_normalization_boundaries_revalidate_bypassed_frozen_spec(
    action_spec: BimanualActionSpec,
) -> None:
    invalid_spec = dataclasses.replace(action_spec)
    object.__setattr__(invalid_spec, "physical_horizon", 0)
    norm_stats = make_norm_stats(action_spec)

    with pytest.raises(ValueError, match="physical_horizon"):
        validate_normalization_stats(norm_stats, invalid_spec)
    with pytest.raises(ValueError, match="physical_horizon"):
        normalization_stats_fingerprint(norm_stats, invalid_spec)


def test_fingerprint_is_stable_across_mapping_order_dtype_and_endianness(
    action_spec: BimanualActionSpec,
) -> None:
    native_stats = make_norm_stats(action_spec, dtype=np.float32)
    portable_stats = make_norm_stats(action_spec, dtype=np.dtype(">f8"))
    reordered_stats = {
        "right_actions": as_mapping(portable_stats["right_actions"]),
        "left_actions": as_mapping(portable_stats["left_actions"]),
        "state": as_mapping(portable_stats["state"]),
    }

    native_fingerprint = normalization_stats_fingerprint(native_stats, action_spec)
    portable_fingerprint = normalization_stats_fingerprint(reordered_stats, action_spec)

    assert portable_fingerprint == native_fingerprint
    assert len(native_fingerprint) == 64
    assert set(native_fingerprint) <= set("0123456789abcdef")


def test_fingerprint_canonicalizes_negative_zero(action_spec: BimanualActionSpec) -> None:
    positive_zero_stats = make_norm_stats(action_spec)
    negative_zero_stats = make_norm_stats(action_spec)
    positive_zero_stats["state"] = dataclasses.replace(
        positive_zero_stats["state"],
        mean=np.zeros(4, dtype=np.float32),
    )
    negative_zero_stats["state"] = dataclasses.replace(
        negative_zero_stats["state"],
        mean=np.full(4, -0.0, dtype=np.float32),
    )

    assert normalization_stats_fingerprint(
        positive_zero_stats,
        action_spec,
    ) == normalization_stats_fingerprint(negative_zero_stats, action_spec)


@pytest.mark.parametrize("bad_key", [None, "extra"])
def test_validation_requires_exact_top_level_keys(action_spec: BimanualActionSpec, bad_key: str | None) -> None:
    norm_stats = make_norm_stats(action_spec)
    if bad_key is None:
        norm_stats.pop("state")
    else:
        norm_stats[bad_key] = make_stats_entry(2)

    with pytest.raises(ValueError, match=r"norm_stats keys.*expected exactly"):
        validate_normalization_stats(norm_stats, action_spec)


def test_validation_rejects_horizon_by_action_width_statistics(
    action_spec: BimanualActionSpec,
) -> None:
    norm_stats = make_norm_stats(action_spec)
    left_stats = norm_stats["left_actions"]
    norm_stats["left_actions"] = dataclasses.replace(
        left_stats,
        mean=np.zeros(
            (action_spec.physical_horizon, action_spec.logical_action_dim),
            dtype=np.float32,
        ),
    )

    with pytest.raises(ValueError, match=r"left_actions.*mean.*one-dimensional"):
        validate_normalization_stats(norm_stats, action_spec)


def test_validation_rejects_inconsistent_state_stat_shapes(action_spec: BimanualActionSpec) -> None:
    norm_stats = make_norm_stats(action_spec)
    state_stats = norm_stats["state"]
    norm_stats["state"] = dataclasses.replace(state_stats, q99=np.ones(5, dtype=np.float32))

    with pytest.raises(ValueError, match=r"state.*q99.*expected \(4,\).*[Gg]ot \(5,\)"):
        validate_normalization_stats(norm_stats, action_spec)


@pytest.mark.parametrize("dtype", [np.int64, object])
def test_validation_rejects_nonfloating_stats(action_spec: BimanualActionSpec, dtype: np.dtype | type) -> None:
    norm_stats = make_norm_stats(action_spec)
    left_stats = norm_stats["left_actions"]
    norm_stats["left_actions"] = dataclasses.replace(
        left_stats,
        mean=np.ones(action_spec.logical_action_dim, dtype=dtype),
    )

    with pytest.raises(TypeError, match=r"left_actions.*mean.*floating dtype"):
        validate_normalization_stats(norm_stats, action_spec)


def test_validation_rejects_nonfinite_stats(action_spec: BimanualActionSpec) -> None:
    norm_stats = make_norm_stats(action_spec)
    right_stats = norm_stats["right_actions"]
    invalid_mean = right_stats.mean.copy()
    invalid_mean[7] = np.inf
    norm_stats["right_actions"] = dataclasses.replace(right_stats, mean=invalid_mean)

    with pytest.raises(ValueError, match=r"right_actions.*mean.*non-finite.*7"):
        validate_normalization_stats(norm_stats, action_spec)


def test_validation_rejects_negative_standard_deviation(action_spec: BimanualActionSpec) -> None:
    norm_stats = make_norm_stats(action_spec)
    state_stats = norm_stats["state"]
    invalid_std = state_stats.std.copy()
    invalid_std[2] = -0.1
    norm_stats["state"] = dataclasses.replace(state_stats, std=invalid_std)

    with pytest.raises(ValueError, match=r"state.*std.*negative.*2"):
        validate_normalization_stats(norm_stats, action_spec)


def test_validation_rejects_reversed_quantiles(action_spec: BimanualActionSpec) -> None:
    norm_stats = make_norm_stats(action_spec)
    left_stats = norm_stats["left_actions"]
    invalid_q01 = left_stats.q01.copy()
    invalid_q01[3] = left_stats.q99[3] + 0.5
    norm_stats["left_actions"] = dataclasses.replace(left_stats, q01=invalid_q01)

    with pytest.raises(ValueError, match=r"left_actions.*q01 exceeds q99.*3"):
        validate_normalization_stats(norm_stats, action_spec)


def test_validation_requires_pi05_quantiles(action_spec: BimanualActionSpec) -> None:
    norm_stats = make_norm_stats(action_spec)
    right_stats = norm_stats["right_actions"]
    norm_stats["right_actions"] = dataclasses.replace(right_stats, q01=None)

    with pytest.raises(TypeError, match=r"right_actions.*q01.*numpy.ndarray"):
        validate_normalization_stats(norm_stats, action_spec)


def test_shared_hand_normalization_requires_all_statistics_equal(action_spec: BimanualActionSpec) -> None:
    shared_spec = dataclasses.replace(action_spec, hand_normalization=HandNormalization.SHARED)
    norm_stats = make_norm_stats(shared_spec)

    with pytest.raises(
        ValueError,
        match=r"shared hand normalization.*left_actions\.mean.*right_actions\.mean",
    ):
        validate_normalization_stats(norm_stats, shared_spec)


def test_shared_hand_normalization_accepts_equal_numeric_stats(action_spec: BimanualActionSpec) -> None:
    shared_spec = dataclasses.replace(action_spec, hand_normalization=HandNormalization.SHARED)
    norm_stats = make_norm_stats(shared_spec)
    norm_stats["right_actions"] = make_stats_entry(
        shared_spec.logical_action_dim,
        offset=10.0,
        dtype=np.float64,
    )

    validate_normalization_stats(norm_stats, shared_spec)
