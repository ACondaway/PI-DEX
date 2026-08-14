import concurrent.futures
import dataclasses
import threading
from collections.abc import Iterator
from collections.abc import Mapping

import numpy as np
import pytest

from pi_dex.actions import JOINT_LOGICAL_ACTION_DIM
from pi_dex.actions import LOGICAL_ACTION_DIM
from pi_dex.actions import MODEL_ACTION_DIM
from pi_dex.actions import ActionRepresentation
from pi_dex.deployment import BimanualActionChunkBroker
from pi_dex.deployment import BimanualBrokerFault
from pi_dex.deployment import BimanualCommandDispatcher
from pi_dex.deployment import BimanualDispatchError
from pi_dex.deployment import BimanualHoldReceipt
from pi_dex.deployment import BimanualPolicyAdapter
from pi_dex.deployment import BimanualSafetyLimits
from pi_dex.deployment import CHUNK_STEP_INDEX_FIELD
from pi_dex.deployment import DEPLOYMENT_WIRE_FORMAT
from pi_dex.deployment import SESSION_ID_FIELD
from pi_dex.deployment import validate_deployment_metadata
from pi_dex.deployment import validate_execution_horizon
from pi_dex.spec import BimanualActionSpec
from tests.helpers import spec_for_representation

CLOCK_DOMAIN = "unix_realtime"
SOURCE_TIMESTAMP_NS = 1_000_000_000
CURRENT_TIMESTAMP_NS = 1_010_000_000
TARGET_TIMESTAMP_NS = 1_020_000_000
SESSION_ID = "0123456789abcdef0123456789abcdef"


class FakeLogicalPolicy:
    def __init__(self, action_spec: BimanualActionSpec) -> None:
        self.calls = 0
        self.reset_calls = 0
        self.observations: list[dict[str, object]] = []
        self.metadata: dict[str, object] = {
            "checkpoint": {"name": "fake", "tags": ["test"]},
            "pi_dex": action_spec.to_metadata(),
        }
        self.left_actions = np.zeros((2, LOGICAL_ACTION_DIM), dtype=np.float64)
        self.right_actions = np.zeros((2, LOGICAL_ACTION_DIM), dtype=np.float64)
        self.left_actions[:, 0] = [10.0, 11.0]
        self.right_actions[:, 0] = [20.0, 21.0]
        self.left_actions[:, 3] = 1.0
        self.left_actions[:, 7] = 1.0
        self.right_actions[:, 3] = 1.0
        self.right_actions[:, 7] = 1.0

    def infer(self, observation: dict[str, object]) -> dict[str, object]:
        self.calls += 1
        self.observations.append(observation)
        return {
            "left_actions": self.left_actions.copy(),
            "right_actions": self.right_actions.copy(),
            "policy_timing": {"infer_ms": 12.0},
        }

    def reset(self) -> None:
        self.reset_calls += 1


class FakeRawPolicy(FakeLogicalPolicy):
    def infer(self, observation: dict[str, object]) -> dict[str, object]:
        self.calls += 1
        self.observations.append(observation)
        return {"actions": np.zeros((4, MODEL_ACTION_DIM), dtype=np.float32)}


class FakeWrongHorizonWirePolicy:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.metadata: dict[str, object] = {}

    def infer(self, observation: dict[str, object]) -> dict[str, object]:
        del observation
        return {
            "actions": {
                "left": np.zeros((1, LOGICAL_ACTION_DIM), dtype=np.float32),
                "right": np.zeros((1, LOGICAL_ACTION_DIM), dtype=np.float32),
            },
            "source_timestamp_ns": SOURCE_TIMESTAMP_NS,
            "clock_domain": CLOCK_DOMAIN,
            "chunk_sequence_id": 1,
            SESSION_ID_FIELD: SESSION_ID,
        }

    def reset(self) -> None:
        self.reset_calls += 1


class FakeController:
    def __init__(
        self,
        action_spec: BimanualActionSpec,
        *,
        fail_apply: bool = False,
        fail_hold: bool = False,
    ) -> None:
        self.action_spec = action_spec
        self.safety_faulted = False
        self.recovery_epoch = 0
        self.clock_timestamp_ns = CURRENT_TIMESTAMP_NS
        self.clock_reads: list[int] = []
        self._dispatch_lease: object | None = None
        self._dispatch_lease_contract: tuple[BimanualActionSpec, str, int] | None = None
        self._backend_lock = threading.RLock()
        self.fail_apply = fail_apply
        self.fail_hold = fail_hold
        self.applied: list[tuple[np.ndarray, np.ndarray, int, str]] = []
        self.hold_reasons: list[str] = []

    def acquire_dispatch_lease(
        self,
        *,
        expected_spec: BimanualActionSpec,
        expected_clock_domain: str,
        expected_recovery_epoch: int,
    ) -> object:
        with self._backend_lock:
            if self.safety_faulted:
                raise RuntimeError("controller is safety faulted")
            if expected_spec != self.action_spec:
                raise RuntimeError("controller action spec mismatch")
            if expected_clock_domain != self.action_spec.clock_domain:
                raise RuntimeError("controller clock domain mismatch")
            if expected_recovery_epoch != self.recovery_epoch:
                raise RuntimeError("controller recovery epoch mismatch")
            if self._dispatch_lease is not None:
                raise RuntimeError("controller dispatch lease is already active")
            self._dispatch_lease = object()
            self._dispatch_lease_contract = (
                expected_spec,
                expected_clock_domain,
                expected_recovery_epoch,
            )
            return self._dispatch_lease

    def validate_dispatch_lease(
        self,
        dispatch_lease: object,
        *,
        expected_spec: BimanualActionSpec,
        expected_clock_domain: str,
        expected_recovery_epoch: int,
    ) -> None:
        with self._backend_lock:
            if self.safety_faulted:
                raise RuntimeError("controller is safety faulted")
            if expected_spec != self.action_spec:
                raise RuntimeError("controller action spec mismatch")
            if expected_clock_domain != self.action_spec.clock_domain:
                raise RuntimeError("controller clock domain mismatch")
            if expected_recovery_epoch != self.recovery_epoch:
                raise RuntimeError("controller recovery epoch mismatch")
            if dispatch_lease is not self._dispatch_lease:
                raise RuntimeError("controller dispatch lease is not active")
            if self._dispatch_lease_contract != (
                expected_spec,
                expected_clock_domain,
                expected_recovery_epoch,
            ):
                raise RuntimeError("controller dispatch lease contract mismatch")

    def read_clock_ns(self, *, dispatch_lease: object) -> int:
        with self._backend_lock:
            self.validate_dispatch_lease(
                dispatch_lease,
                expected_spec=self.action_spec,
                expected_clock_domain=self.action_spec.clock_domain,
                expected_recovery_epoch=self.recovery_epoch,
            )
            self.clock_reads.append(self.clock_timestamp_ns)
            return self.clock_timestamp_ns

    def apply_bimanual_action(
        self,
        left_action: np.ndarray,
        right_action: np.ndarray,
        *,
        target_timestamp_ns: int,
        not_before_timestamp_ns: int,
        not_after_timestamp_ns: int,
        clock_domain: str,
        dispatch_lease: object,
        expected_recovery_epoch: int,
    ) -> None:
        with self._backend_lock:
            self.validate_dispatch_lease(
                dispatch_lease,
                expected_spec=self.action_spec,
                expected_clock_domain=clock_domain,
                expected_recovery_epoch=expected_recovery_epoch,
            )
            if self.clock_timestamp_ns < not_before_timestamp_ns:
                raise RuntimeError("controller rejected dispatch after clock rollback")
            if not self.clock_timestamp_ns <= not_after_timestamp_ns < target_timestamp_ns:
                raise RuntimeError("controller rejected expired dispatch deadline")
            if self.fail_apply:
                raise RuntimeError("controller write failed")
            self.applied.append(
                (left_action.copy(), right_action.copy(), target_timestamp_ns, clock_domain)
            )

    def hold(
        self,
        *,
        reason: str,
        dispatch_lease: object,
        expected_recovery_epoch: int,
    ) -> BimanualHoldReceipt:
        with self._backend_lock:
            self.hold_reasons.append(reason)
            if self.fail_hold:
                raise RuntimeError("hold failed")
            if expected_recovery_epoch != self.recovery_epoch:
                raise RuntimeError("controller recovery epoch mismatch")
            if dispatch_lease is not self._dispatch_lease:
                raise RuntimeError("controller dispatch lease is not active")
            self.safety_faulted = True
            self._dispatch_lease = None
            self._dispatch_lease_contract = None
            return BimanualHoldReceipt(safety_faulted=True, recovery_epoch=self.recovery_epoch)

    def recover(self) -> None:
        with self._backend_lock:
            self._dispatch_lease = None
            self._dispatch_lease_contract = None
            self.safety_faulted = False
            self.recovery_epoch += 1


def make_observation(
    *,
    timestamp_ns: object = SOURCE_TIMESTAMP_NS,
    clock_domain: object = CLOCK_DOMAIN,
) -> dict[str, object]:
    return {
        "state": np.zeros((4,), dtype=np.float32),
        "observation_timestamp_ns": timestamp_ns,
        "clock_domain": clock_domain,
    }


def make_unbounded_limits(spec: BimanualActionSpec) -> BimanualSafetyLimits:
    lower = np.full((spec.logical_action_dim,), np.finfo(np.float32).min, dtype=np.float32)
    upper = np.full((spec.logical_action_dim,), np.finfo(np.float32).max, dtype=np.float32)
    return BimanualSafetyLimits(spec, lower, upper, lower, upper)


def make_dispatcher(
    spec: BimanualActionSpec,
    controller: FakeController,
    *,
    limits: BimanualSafetyLimits | None = None,
    clock_domain: str = CLOCK_DOMAIN,
    max_target_lead_ms: float = 20.0,
    execution_horizon: int | None = None,
    session_id: str = SESSION_ID,
) -> BimanualCommandDispatcher:
    return BimanualCommandDispatcher(
        controller,
        spec,
        limits or make_unbounded_limits(spec),
        execution_horizon=(spec.physical_horizon if execution_horizon is None else execution_horizon),
        session_id=session_id,
        clock_domain=clock_domain,
        max_target_lead_ms=max_target_lead_ms,
    )


def make_step_result(
    *,
    source_timestamp_ns: object = SOURCE_TIMESTAMP_NS,
    clock_domain: object = CLOCK_DOMAIN,
    chunk_sequence_id: object = 1,
    chunk_step_index: object = 0,
    session_id: object = SESSION_ID,
) -> dict[str, object]:
    left_action = np.zeros((LOGICAL_ACTION_DIM,), dtype=np.float32)
    right_action = np.zeros((LOGICAL_ACTION_DIM,), dtype=np.float32)
    left_action[[3, 7]] = 1.0
    right_action[[3, 7]] = 1.0
    return {
        "actions": {
            "left": left_action,
            "right": right_action,
        },
        "source_timestamp_ns": source_timestamp_ns,
        "clock_domain": clock_domain,
        "chunk_sequence_id": chunk_sequence_id,
        "chunk_step_index": chunk_step_index,
        SESSION_ID_FIELD: session_id,
    }


def make_joint_step_result() -> dict[str, object]:
    return {
        "actions": {
            "left": np.zeros((JOINT_LOGICAL_ACTION_DIM,), dtype=np.float32),
            "right": np.zeros((JOINT_LOGICAL_ACTION_DIM,), dtype=np.float32),
        },
        "source_timestamp_ns": SOURCE_TIMESTAMP_NS,
        "clock_domain": CLOCK_DOMAIN,
        "chunk_sequence_id": 1,
        "chunk_step_index": 0,
        SESSION_ID_FIELD: SESSION_ID,
    }


def test_policy_adapter_strips_transport_fields_without_mutating_observation(
    action_spec: BimanualActionSpec,
) -> None:
    policy = FakeLogicalPolicy(action_spec)
    adapter = BimanualPolicyAdapter(policy, action_spec, execution_horizon=1)
    observation = make_observation()

    result = adapter.infer(observation)

    assert observation["observation_timestamp_ns"] == SOURCE_TIMESTAMP_NS
    assert observation["clock_domain"] == CLOCK_DOMAIN
    assert set(policy.observations[0]) == {"state"}
    assert policy.observations[0] is not observation
    assert result["actions"]["left"].shape == (1, LOGICAL_ACTION_DIM)
    assert result["actions"]["right"].shape == (1, LOGICAL_ACTION_DIM)
    assert result["actions"]["left"].dtype == np.float32
    assert result["source_timestamp_ns"] == SOURCE_TIMESTAMP_NS
    assert result["clock_domain"] == CLOCK_DOMAIN
    assert result["chunk_sequence_id"] == 1
    assert result[SESSION_ID_FIELD] == adapter.metadata["pi_dex"][SESSION_ID_FIELD]


def test_joint_policy_adapter_uses_29d_wire_actions(action_spec: BimanualActionSpec) -> None:
    joint_spec = spec_for_representation(action_spec, ActionRepresentation.JOINT_29D)
    policy = FakeLogicalPolicy(joint_spec)
    policy.left_actions = np.zeros((2, JOINT_LOGICAL_ACTION_DIM), dtype=np.float64)
    policy.right_actions = np.zeros((2, JOINT_LOGICAL_ACTION_DIM), dtype=np.float64)

    result = BimanualPolicyAdapter(policy, joint_spec, execution_horizon=1).infer(make_observation())

    assert result["actions"]["left"].shape == (1, JOINT_LOGICAL_ACTION_DIM)
    assert result["actions"]["right"].shape == (1, JOINT_LOGICAL_ACTION_DIM)


def test_joint_safety_limits_require_29d_vectors(action_spec: BimanualActionSpec) -> None:
    joint_spec = spec_for_representation(action_spec, ActionRepresentation.JOINT_29D)
    valid = np.zeros((JOINT_LOGICAL_ACTION_DIM,), dtype=np.float32)
    wrong = np.zeros((LOGICAL_ACTION_DIM,), dtype=np.float32)

    limits = BimanualSafetyLimits(joint_spec, valid, valid, valid, valid)

    assert limits.left_min.shape == (JOINT_LOGICAL_ACTION_DIM,)
    with pytest.raises(ValueError, match=r"expected \(29,\)"):
        BimanualSafetyLimits(joint_spec, wrong, valid, valid, valid)


def test_policy_adapter_snapshots_nested_observation_values(action_spec: BimanualActionSpec) -> None:
    class MutatingPolicy(FakeLogicalPolicy):
        def infer(self, observation: dict[str, object]) -> dict[str, object]:
            nested = observation["nested"]
            assert isinstance(nested, dict)
            image_list = nested["images"]
            assert isinstance(image_list, list)
            image = image_list[0]
            assert isinstance(image, np.ndarray)
            image[0] = 99
            return super().infer(observation)

    policy = MutatingPolicy(action_spec)
    image = np.asarray([1, 2, 3], dtype=np.uint8)
    observation = make_observation()
    observation["nested"] = {"images": [image], "labels": ("left", np.int32(2))}

    BimanualPolicyAdapter(policy, action_spec).infer(observation)

    assert image.tolist() == [1, 2, 3]
    captured_nested = policy.observations[0]["nested"]
    assert isinstance(captured_nested, dict)
    captured_images = captured_nested["images"]
    assert isinstance(captured_images, list)
    captured_image = captured_images[0]
    assert isinstance(captured_image, np.ndarray)
    assert captured_image.tolist() == [99, 2, 3]
    image[1] = 88
    assert captured_image.tolist() == [99, 2, 3]


@pytest.mark.parametrize(
    "unsupported_value",
    [object(), np.asarray([object()], dtype=object), np.asarray([1 + 2j])],
)
def test_policy_adapter_rejects_unsafe_observation_leaf(
    action_spec: BimanualActionSpec,
    unsupported_value: object,
) -> None:
    policy = FakeLogicalPolicy(action_spec)
    observation = make_observation()
    observation["unsafe"] = unsupported_value

    with pytest.raises(TypeError, match="unsupported"):
        BimanualPolicyAdapter(policy, action_spec).infer(observation)

    assert policy.calls == 0


@pytest.mark.parametrize(
    ("observation", "error", "message"),
    [
        ({"clock_domain": CLOCK_DOMAIN}, KeyError, "observation_timestamp_ns"),
        ({"observation_timestamp_ns": SOURCE_TIMESTAMP_NS}, KeyError, "clock_domain"),
        (make_observation(timestamp_ns=True), TypeError, "observation_timestamp_ns"),
        (make_observation(clock_domain=" "), ValueError, "clock_domain"),
    ],
)
def test_policy_adapter_rejects_invalid_transport_observation(
    action_spec: BimanualActionSpec,
    observation: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        BimanualPolicyAdapter(FakeLogicalPolicy(action_spec), action_spec).infer(observation)


def test_policy_adapter_rejects_raw_model_space_actions(action_spec: BimanualActionSpec) -> None:
    adapter = BimanualPolicyAdapter(FakeRawPolicy(action_spec), action_spec)

    with pytest.raises(ValueError, match=r"raw model-space.*inverse-normalized"):
        adapter.infer(make_observation())


def test_policy_adapter_rejects_nonfinite_and_float32_overflow(action_spec: BimanualActionSpec) -> None:
    policy = FakeLogicalPolicy(action_spec)
    policy.left_actions[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        BimanualPolicyAdapter(policy, action_spec).infer(make_observation())

    policy.left_actions[0, 0] = np.finfo(np.float64).max
    with pytest.raises(ValueError, match="overflow float32"):
        BimanualPolicyAdapter(policy, action_spec).infer(make_observation())


def test_policy_adapter_metadata_is_deeply_copied(action_spec: BimanualActionSpec) -> None:
    policy = FakeLogicalPolicy(action_spec)
    adapter = BimanualPolicyAdapter(policy, action_spec)

    first = adapter.metadata
    first["pi_dex"]["hand_order"].reverse()
    first["checkpoint"]["tags"].append("mutated")

    second = adapter.metadata
    assert second["pi_dex"]["hand_order"] == ["left", "right"]
    assert second["checkpoint"]["tags"] == ["test"]
    assert policy.metadata["checkpoint"]["tags"] == ["test"]


def test_validate_metadata_and_construct_broker_with_server_horizon(action_spec: BimanualActionSpec) -> None:
    adapter = BimanualPolicyAdapter(FakeLogicalPolicy(action_spec), action_spec, execution_horizon=1)

    execution_horizon = validate_deployment_metadata(
        adapter.metadata,
        action_spec,
        expected_execution_horizon=1,
    )
    broker = BimanualActionChunkBroker.from_metadata(
        adapter,
        adapter.metadata,
        action_spec,
        expected_execution_horizon=1,
    )

    assert execution_horizon == 1
    assert broker.faulted is False
    assert adapter.metadata["pi_dex"]["wire_format"] == DEPLOYMENT_WIRE_FORMAT


def test_broker_binds_metadata_to_the_exact_policy_session(
    action_spec: BimanualActionSpec,
) -> None:
    first = BimanualPolicyAdapter(FakeLogicalPolicy(action_spec), action_spec)
    second = BimanualPolicyAdapter(FakeLogicalPolicy(action_spec), action_spec)

    with pytest.raises(ValueError, match="policy metadata does not match"):
        BimanualActionChunkBroker.from_metadata(
            second,
            first.metadata,
            action_spec,
        )


def test_broker_compares_policy_metadata_with_numpy_arrays(
    action_spec: BimanualActionSpec,
) -> None:
    policy = FakeLogicalPolicy(action_spec)
    policy.metadata["calibration_vector"] = np.asarray([1.0, np.nan], dtype=np.float32)
    adapter = BimanualPolicyAdapter(policy, action_spec)

    broker = BimanualActionChunkBroker.from_metadata(
        adapter,
        adapter.metadata,
        action_spec,
    )

    assert broker.session_id == adapter.metadata["pi_dex"][SESSION_ID_FIELD]


def test_broker_compares_nonfloating_numpy_metadata_and_mapping_order(
    action_spec: BimanualActionSpec,
) -> None:
    policy = FakeLogicalPolicy(action_spec)
    policy.metadata["labels"] = np.asarray(["left", "right"])
    adapter = BimanualPolicyAdapter(policy, action_spec)
    supplied_metadata = dict(reversed(tuple(adapter.metadata.items())))

    broker = BimanualActionChunkBroker.from_metadata(
        adapter,
        supplied_metadata,
        action_spec,
    )

    assert broker.session_id == adapter.metadata["pi_dex"][SESSION_ID_FIELD]


def test_broker_rejects_changed_numpy_metadata_in_the_same_session(
    action_spec: BimanualActionSpec,
) -> None:
    policy = FakeLogicalPolicy(action_spec)
    policy.metadata["calibration_vector"] = np.asarray([1.0, 2.0], dtype=np.float32)
    adapter = BimanualPolicyAdapter(policy, action_spec)
    supplied_metadata = adapter.metadata
    supplied_metadata["calibration_vector"][0] = 3.0

    with pytest.raises(ValueError, match="policy metadata does not match"):
        BimanualActionChunkBroker.from_metadata(
            adapter,
            supplied_metadata,
            action_spec,
        )


def test_dispatcher_from_metadata_binds_server_session(action_spec: BimanualActionSpec) -> None:
    adapter = BimanualPolicyAdapter(FakeLogicalPolicy(action_spec), action_spec)
    controller = FakeController(action_spec)

    dispatcher = BimanualCommandDispatcher.from_metadata(
        controller,
        adapter.metadata,
        action_spec,
        make_unbounded_limits(action_spec),
    )

    assert dispatcher.session_id == adapter.metadata["pi_dex"][SESSION_ID_FIELD]
    assert dispatcher.execution_horizon == action_spec.physical_horizon


def test_policy_adapter_requires_verified_training_metadata(action_spec: BimanualActionSpec) -> None:
    policy = FakeLogicalPolicy(action_spec)
    del policy.metadata["pi_dex"]

    with pytest.raises(ValueError, match="missing verified 'pi_dex' training contract"):
        BimanualPolicyAdapter(policy, action_spec)


def test_validate_metadata_rejects_execution_horizon_mismatch(action_spec: BimanualActionSpec) -> None:
    metadata = BimanualPolicyAdapter(
        FakeLogicalPolicy(action_spec),
        action_spec,
        execution_horizon=1,
    ).metadata

    with pytest.raises(ValueError, match="expected client horizon 2, got 1"):
        validate_deployment_metadata(metadata, action_spec, expected_execution_horizon=2)


@pytest.mark.parametrize("field_name", ["execution_horizon", "wire_format", SESSION_ID_FIELD])
def test_validate_metadata_requires_deployment_fields(
    action_spec: BimanualActionSpec,
    field_name: str,
) -> None:
    metadata = BimanualPolicyAdapter(FakeLogicalPolicy(action_spec), action_spec).metadata
    del metadata["pi_dex"][field_name]

    with pytest.raises(ValueError, match=field_name):
        validate_deployment_metadata(metadata, action_spec)


@pytest.mark.parametrize(
    ("field_name", "value", "error"),
    [
        ("execution_horizon", None, TypeError),
        ("execution_horizon", 3, ValueError),
        ("wire_format", True, TypeError),
        ("wire_format", "raw_model_actions_v0", ValueError),
        (SESSION_ID_FIELD, True, TypeError),
        (SESSION_ID_FIELD, "not-a-session", ValueError),
    ],
)
def test_validate_metadata_rejects_invalid_deployment_values(
    action_spec: BimanualActionSpec,
    field_name: str,
    value: object,
    error: type[Exception],
) -> None:
    metadata = BimanualPolicyAdapter(FakeLogicalPolicy(action_spec), action_spec).metadata
    metadata["pi_dex"][field_name] = value

    with pytest.raises(error, match=field_name):
        validate_deployment_metadata(metadata, action_spec)


def test_validate_metadata_rejects_unknown_deployment_extension(
    action_spec: BimanualActionSpec,
) -> None:
    metadata = BimanualPolicyAdapter(FakeLogicalPolicy(action_spec), action_spec).metadata
    metadata["pi_dex"]["unverified_extension"] = True

    with pytest.raises(ValueError, match=r"unexpected fields.*unverified_extension"):
        validate_deployment_metadata(metadata, action_spec)


def test_execution_horizon_rejects_impossible_timing_contract(
    action_spec: BimanualActionSpec,
) -> None:
    impossible_spec = dataclasses.replace(
        action_spec,
        physical_horizon=8,
        max_observation_age_ms=1.0,
        max_command_lead_ms=1.0,
        max_control_period_error_ms=1.0,
    )

    with pytest.raises(ValueError, match="cannot complete one chunk"):
        validate_execution_horizon(8, impossible_spec)


def test_execution_horizon_requires_one_nanosecond_of_future_lead(
    action_spec: BimanualActionSpec,
) -> None:
    impossible_spec = dataclasses.replace(action_spec, max_command_lead_ms=0.0000001)

    with pytest.raises(ValueError, match="future nanosecond"):
        validate_execution_horizon(1, impossible_spec)


def test_execution_horizon_does_not_combine_fractional_age_and_lead_nanoseconds(
    action_spec: BimanualActionSpec,
) -> None:
    impossible_spec = dataclasses.replace(
        action_spec,
        physical_horizon=2,
        control_frequency_hz=1_000_000_000.0,
        max_control_period_error_ms=0.0000001,
        max_observation_age_ms=0.0000006,
        max_command_lead_ms=0.0000016,
    )

    with pytest.raises(ValueError, match="cannot complete one chunk"):
        validate_execution_horizon(2, impossible_spec)


def test_execution_horizon_accepts_exact_integer_nanosecond_boundary(
    action_spec: BimanualActionSpec,
) -> None:
    boundary_spec = dataclasses.replace(
        action_spec,
        physical_horizon=2,
        control_frequency_hz=1_000_000_000.0,
        max_control_period_error_ms=0.0000001,
        max_observation_age_ms=0.0000006,
        max_command_lead_ms=0.000002,
    )

    assert validate_execution_horizon(2, boundary_spec) == 2


def test_execution_horizon_requires_a_representable_integer_nanosecond_period(
    action_spec: BimanualActionSpec,
) -> None:
    impossible_spec = dataclasses.replace(
        action_spec,
        physical_horizon=2,
        control_frequency_hz=2_000_000_000.0,
        max_control_period_error_ms=0.0000001,
    )

    with pytest.raises(ValueError, match="no positive integer-nanosecond control period"):
        validate_execution_horizon(2, impossible_spec)


def test_validate_metadata_uses_one_recursive_handshake_snapshot(
    action_spec: BimanualActionSpec,
) -> None:
    first = {
        **action_spec.to_metadata(),
        "wire_format": DEPLOYMENT_WIRE_FORMAT,
        "execution_horizon": 1,
        SESSION_ID_FIELD: SESSION_ID,
    }
    second = {
        **dataclasses.replace(action_spec, coordinate_frame="other_frame").to_metadata(),
        "wire_format": DEPLOYMENT_WIRE_FORMAT,
        "execution_horizon": 2,
        SESSION_ID_FIELD: SESSION_ID,
    }

    class SwitchingMetadata(Mapping[str, object]):
        def __init__(self) -> None:
            self.read_count = 0

        def __getitem__(self, key: str) -> object:
            if key != "pi_dex":
                raise KeyError(key)
            self.read_count += 1
            return first if self.read_count == 1 else second

        def __iter__(self) -> Iterator[str]:
            return iter(("pi_dex",))

        def __len__(self) -> int:
            return 1

    metadata = SwitchingMetadata()

    assert validate_deployment_metadata(metadata, action_spec) == 1
    assert metadata.read_count == 1


def test_broker_peek_is_stable_until_commit_and_then_advances(action_spec: BimanualActionSpec) -> None:
    policy = FakeLogicalPolicy(action_spec)
    adapter = BimanualPolicyAdapter(policy, action_spec)
    broker = BimanualActionChunkBroker.from_metadata(adapter, adapter.metadata, action_spec)

    first = broker.peek(make_observation())
    first["actions"]["left"][0] = -100.0
    repeated = broker.peek(make_observation(timestamp_ns=SOURCE_TIMESTAMP_NS + 1))
    broker.commit()
    second = broker.peek(make_observation(timestamp_ns=SOURCE_TIMESTAMP_NS + 2))
    broker.commit()
    third = broker.peek(make_observation(timestamp_ns=SOURCE_TIMESTAMP_NS + 3))

    assert repeated["actions"]["left"][0] == 10.0
    assert repeated["actions"]["right"][0] == 20.0
    assert second["actions"]["left"][0] == 11.0
    assert second["actions"]["right"][0] == 21.0
    assert repeated["source_timestamp_ns"] == SOURCE_TIMESTAMP_NS
    assert second["source_timestamp_ns"] == SOURCE_TIMESTAMP_NS
    assert repeated[CHUNK_STEP_INDEX_FIELD] == 0
    assert second[CHUNK_STEP_INDEX_FIELD] == 1
    assert "policy_timing" in repeated
    assert "policy_timing" not in second
    assert third["source_timestamp_ns"] == SOURCE_TIMESTAMP_NS + 3
    assert policy.calls == 2


def test_broker_snapshots_observation_before_remote_policy_serialization() -> None:
    class BlockingWirePolicy:
        def __init__(self) -> None:
            self.entered = threading.Event()
            self.release = threading.Event()
            self.captured_image: np.ndarray | None = None

        def infer(self, observation: dict[str, object]) -> dict[str, object]:
            self.entered.set()
            if not self.release.wait(timeout=2.0):
                raise RuntimeError("timed out waiting to inspect broker snapshot")
            nested = observation["nested"]
            assert isinstance(nested, dict)
            image = nested["image"]
            assert isinstance(image, np.ndarray)
            self.captured_image = image.copy()
            return {
                "actions": {
                    "left": np.zeros((2, LOGICAL_ACTION_DIM), dtype=np.float32),
                    "right": np.zeros((2, LOGICAL_ACTION_DIM), dtype=np.float32),
                },
                "source_timestamp_ns": SOURCE_TIMESTAMP_NS,
                "clock_domain": CLOCK_DOMAIN,
                "chunk_sequence_id": 1,
            }

        def reset(self) -> None:
            pass

    policy = BlockingWirePolicy()
    broker = BimanualActionChunkBroker(
        policy,
        execution_horizon=2,
        action_representation=ActionRepresentation.CARTESIAN_31D,
    )
    image = np.asarray([1, 2, 3], dtype=np.uint8)
    observation = {"nested": {"image": image}}

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(broker.peek, observation)
        assert policy.entered.wait(timeout=2.0)
        image[:] = 99
        policy.release.set()
        future.result(timeout=2.0)

    assert policy.captured_image is not None
    assert policy.captured_image.tolist() == [1, 2, 3]


def test_broker_rejects_commit_without_peek(action_spec: BimanualActionSpec) -> None:
    adapter = BimanualPolicyAdapter(FakeLogicalPolicy(action_spec), action_spec)
    broker = BimanualActionChunkBroker.from_metadata(adapter, adapter.metadata, action_spec)

    with pytest.raises(RuntimeError, match="no peeked"):
        broker.commit()


def test_dispatch_next_commits_only_after_success(action_spec: BimanualActionSpec) -> None:
    policy = FakeLogicalPolicy(action_spec)
    adapter = BimanualPolicyAdapter(policy, action_spec)
    broker = BimanualActionChunkBroker.from_metadata(adapter, adapter.metadata, action_spec)
    controller = FakeController(action_spec)
    dispatcher = make_dispatcher(action_spec, controller, session_id=broker.session_id)

    first = broker.dispatch_next(
        make_observation(),
        dispatcher,
        target_timestamp_ns=TARGET_TIMESTAMP_NS,
    )
    second = broker.peek(make_observation(timestamp_ns=SOURCE_TIMESTAMP_NS + 1))

    assert first["actions"]["left"][0] == 10.0
    assert second["actions"]["left"][0] == 11.0
    assert len(controller.applied) == 1
    assert controller.applied[0][2:] == (TARGET_TIMESTAMP_NS, CLOCK_DOMAIN)
    assert broker.faulted is False


def test_slow_inference_cannot_use_a_pre_inference_clock_sample(
    action_spec: BimanualActionSpec,
) -> None:
    controller = FakeController(action_spec)

    class AdvancingPolicy(FakeLogicalPolicy):
        def infer(self, observation: dict[str, object]) -> dict[str, object]:
            controller.clock_timestamp_ns = TARGET_TIMESTAMP_NS + 1
            return super().infer(observation)

    policy = AdvancingPolicy(action_spec)
    adapter = BimanualPolicyAdapter(policy, action_spec)
    broker = BimanualActionChunkBroker.from_metadata(adapter, adapter.metadata, action_spec)
    dispatcher = make_dispatcher(action_spec, controller, session_id=broker.session_id)

    with pytest.raises(BimanualDispatchError, match="future target"):
        broker.dispatch_next(
            make_observation(),
            dispatcher,
            target_timestamp_ns=TARGET_TIMESTAMP_NS,
        )

    assert controller.applied == []
    assert broker.faulted is True


def test_dispatch_next_rejects_dispatcher_horizon_not_from_metadata(
    action_spec: BimanualActionSpec,
) -> None:
    adapter = BimanualPolicyAdapter(FakeLogicalPolicy(action_spec), action_spec, execution_horizon=1)
    broker = BimanualActionChunkBroker.from_metadata(adapter, adapter.metadata, action_spec)
    controller = FakeController(action_spec)
    dispatcher = make_dispatcher(
        action_spec,
        controller,
        execution_horizon=2,
        session_id=broker.session_id,
    )

    with pytest.raises(BimanualDispatchError, match="execution_horizon conflicts"):
        broker.dispatch_next(
            make_observation(),
            dispatcher,
            target_timestamp_ns=TARGET_TIMESTAMP_NS,
        )

    assert broker.faulted is True
    assert controller.safety_faulted is True


def test_dispatch_next_rejects_dispatcher_from_another_server_session(
    action_spec: BimanualActionSpec,
) -> None:
    adapter = BimanualPolicyAdapter(FakeLogicalPolicy(action_spec), action_spec)
    broker = BimanualActionChunkBroker.from_metadata(adapter, adapter.metadata, action_spec)
    controller = FakeController(action_spec)
    dispatcher = make_dispatcher(
        action_spec,
        controller,
        session_id="f" * 32,
    )

    with pytest.raises(BimanualDispatchError, match="session_id conflicts"):
        broker.dispatch_next(
            make_observation(),
            dispatcher,
            target_timestamp_ns=TARGET_TIMESTAMP_NS,
        )

    assert controller.safety_faulted is True


def test_dispatch_next_rejects_dispatcher_with_different_action_semantics(
    action_spec: BimanualActionSpec,
) -> None:
    policy = FakeLogicalPolicy(action_spec)
    adapter = BimanualPolicyAdapter(policy, action_spec)
    broker = BimanualActionChunkBroker.from_metadata(adapter, adapter.metadata, action_spec)
    incompatible_spec = dataclasses.replace(action_spec, coordinate_frame="other_frame")
    controller = FakeController(incompatible_spec)
    dispatcher = make_dispatcher(incompatible_spec, controller, session_id=broker.session_id)

    with pytest.raises(BimanualDispatchError, match="action_spec conflicts with broker"):
        broker.dispatch_next(
            make_observation(),
            dispatcher,
            target_timestamp_ns=TARGET_TIMESTAMP_NS,
        )

    assert policy.calls == 0
    assert broker.faulted is True
    assert controller.safety_faulted is True


def test_direct_broker_cannot_bypass_metadata_before_hardware_dispatch(
    action_spec: BimanualActionSpec,
) -> None:
    policy = FakeWrongHorizonWirePolicy()
    broker = BimanualActionChunkBroker(
        policy,
        execution_horizon=2,
        action_representation=ActionRepresentation.CARTESIAN_31D,
    )
    controller = FakeController(action_spec)
    dispatcher = make_dispatcher(action_spec, controller, session_id=SESSION_ID)

    with pytest.raises(BimanualDispatchError, match="construction through from_metadata"):
        broker.dispatch_next(
            make_observation(),
            dispatcher,
            target_timestamp_ns=TARGET_TIMESTAMP_NS,
        )

    assert broker.faulted is True
    assert controller.safety_faulted is True


def test_dispatch_failure_faults_broker_until_reset(action_spec: BimanualActionSpec) -> None:
    policy = FakeLogicalPolicy(action_spec)
    adapter = BimanualPolicyAdapter(policy, action_spec)
    broker = BimanualActionChunkBroker.from_metadata(adapter, adapter.metadata, action_spec)
    controller = FakeController(action_spec, fail_apply=True)
    dispatcher = make_dispatcher(action_spec, controller, session_id=broker.session_id)

    with pytest.raises(BimanualDispatchError, match="controller write failed"):
        broker.dispatch_next(
            make_observation(),
            dispatcher,
            target_timestamp_ns=TARGET_TIMESTAMP_NS,
        )
    with pytest.raises(BimanualBrokerFault, match="reset is required"):
        broker.peek(make_observation())

    controller.fail_apply = False
    controller.recover()
    broker.reset()
    dispatcher = make_dispatcher(action_spec, controller, session_id=broker.session_id)
    recovered = broker.dispatch_next(
        make_observation(),
        dispatcher,
        target_timestamp_ns=TARGET_TIMESTAMP_NS,
    )

    assert recovered["actions"]["left"][0] == 10.0
    assert broker.faulted is False
    assert policy.calls == 2
    assert policy.reset_calls == 1


def test_wrong_wire_horizon_faults_dispatch_until_reset(action_spec: BimanualActionSpec) -> None:
    policy = FakeWrongHorizonWirePolicy()
    metadata = {
        "pi_dex": {
            **action_spec.to_metadata(),
            "wire_format": DEPLOYMENT_WIRE_FORMAT,
            "execution_horizon": 2,
            SESSION_ID_FIELD: SESSION_ID,
        }
    }
    policy.metadata = metadata
    broker = BimanualActionChunkBroker.from_metadata(policy, metadata, action_spec)
    controller = FakeController(action_spec)
    dispatcher = make_dispatcher(action_spec, controller, session_id=broker.session_id)

    with pytest.raises(BimanualDispatchError, match=r"expected \(2, 31\).+got \(1, 31\)"):
        broker.dispatch_next(
            make_observation(),
            dispatcher,
            target_timestamp_ns=TARGET_TIMESTAMP_NS,
        )
    with pytest.raises(BimanualBrokerFault, match="reset is required"):
        broker.peek(make_observation())
    assert len(controller.hold_reasons) == 1

    broker.reset()
    assert broker.faulted is False
    assert policy.reset_calls == 1


def test_broker_reset_failure_stays_faulted(action_spec: BimanualActionSpec) -> None:
    class FailingResetPolicy(FakeLogicalPolicy):
        def reset(self) -> None:
            raise RuntimeError("policy reset failed")

    adapter = BimanualPolicyAdapter(FailingResetPolicy(action_spec), action_spec)
    broker = BimanualActionChunkBroker.from_metadata(adapter, adapter.metadata, action_spec)

    with pytest.raises(RuntimeError, match="policy reset failed"):
        broker.reset()
    with pytest.raises(BimanualBrokerFault, match="reset is required"):
        broker.peek(make_observation())

    assert broker.faulted is True


def test_safety_limits_are_immutable_and_bound_to_spec(action_spec: BimanualActionSpec) -> None:
    original_minimum = np.full((LOGICAL_ACTION_DIM,), -1.0, dtype=np.float32)
    original_maximum = np.full((LOGICAL_ACTION_DIM,), 1.0, dtype=np.float32)
    limits = BimanualSafetyLimits(
        action_spec,
        original_minimum,
        original_maximum,
        original_minimum,
        original_maximum,
    )
    incompatible_spec = dataclasses.replace(action_spec, coordinate_frame="other_frame")

    with pytest.raises(ValueError, match=r"limits\.spec conflicts"):
        BimanualCommandDispatcher(
            FakeController(incompatible_spec),
            incompatible_spec,
            limits,
            execution_horizon=action_spec.physical_horizon,
            session_id=SESSION_ID,
            clock_domain=CLOCK_DOMAIN,
        )
    with pytest.raises(ValueError, match="read-only"):
        limits.left_min[0] = 0.0
    with pytest.raises(ValueError, match="WRITEABLE"):
        limits.left_min.flags.writeable = True
    original_minimum[0] = -100.0
    assert limits.left_min[0] == -1.0


def test_controller_allows_only_one_active_dispatcher_lease(action_spec: BimanualActionSpec) -> None:
    controller = FakeController(action_spec)
    first = make_dispatcher(action_spec, controller)

    with pytest.raises(RuntimeError, match="lease is already active"):
        make_dispatcher(action_spec, controller)

    assert first.faulted is False


def test_controller_lease_acquisition_is_atomic_under_concurrent_construction(
    action_spec: BimanualActionSpec,
) -> None:
    controller = FakeController(action_spec)
    start = threading.Barrier(2)

    def construct_dispatcher() -> BimanualCommandDispatcher:
        start.wait(timeout=2.0)
        return make_dispatcher(action_spec, controller)

    successes: list[BimanualCommandDispatcher] = []
    failures: list[Exception] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(construct_dispatcher) for _ in range(2)]
        for future in futures:
            try:
                successes.append(future.result(timeout=2.0))
            except Exception as error:
                failures.append(error)

    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    assert "lease is already active" in str(failures[0])


def test_invalid_dispatcher_configuration_does_not_consume_controller_lease(
    action_spec: BimanualActionSpec,
) -> None:
    controller = FakeController(action_spec)

    with pytest.raises(ValueError, match="may not exceed"):
        make_dispatcher(
            action_spec,
            controller,
            max_target_lead_ms=action_spec.max_command_lead_ms + 1.0,
        )

    assert make_dispatcher(action_spec, controller).faulted is False


def test_unrepresentable_dispatcher_lead_does_not_leak_overflow_or_consume_lease(
    action_spec: BimanualActionSpec,
) -> None:
    controller = FakeController(action_spec)

    with pytest.raises(ValueError, match=r"max_target_lead_ms.*positive finite"):
        make_dispatcher(
            action_spec,
            controller,
            max_target_lead_ms=10**400,
        )

    assert make_dispatcher(action_spec, controller).faulted is False


def test_dispatcher_requires_controller_to_bind_exact_spec(action_spec: BimanualActionSpec) -> None:
    incompatible_spec = dataclasses.replace(action_spec, kinematics_calibration_version="other-calibration")

    with pytest.raises(ValueError, match=r"controller\.action_spec conflicts"):
        make_dispatcher(action_spec, FakeController(incompatible_spec))


def test_adapter_and_dispatcher_reject_clock_domain_outside_spec(
    action_spec: BimanualActionSpec,
) -> None:
    with pytest.raises(ValueError, match="observation clock_domain"):
        BimanualPolicyAdapter(FakeLogicalPolicy(action_spec), action_spec).infer(
            make_observation(clock_domain="host_monotonic")
        )

    with pytest.raises(ValueError, match=r"spec\.clock_domain"):
        BimanualCommandDispatcher(
            FakeController(action_spec),
            action_spec,
            make_unbounded_limits(action_spec),
            execution_horizon=action_spec.physical_horizon,
            session_id=SESSION_ID,
            clock_domain="host_monotonic",
        )


def test_dispatcher_lead_override_can_only_tighten_spec(action_spec: BimanualActionSpec) -> None:
    with pytest.raises(ValueError, match=r"may not exceed.*max_command_lead_ms"):
        BimanualCommandDispatcher(
            FakeController(action_spec),
            action_spec,
            make_unbounded_limits(action_spec),
            execution_horizon=action_spec.physical_horizon,
            session_id=SESSION_ID,
            clock_domain=CLOCK_DOMAIN,
            max_target_lead_ms=action_spec.max_command_lead_ms + 1.0,
        )


def test_dispatcher_validates_then_applies_one_paired_command(action_spec: BimanualActionSpec) -> None:
    controller = FakeController(action_spec)
    dispatcher = make_dispatcher(action_spec, controller)

    dispatcher.dispatch(
        make_step_result(),
        target_timestamp_ns=TARGET_TIMESTAMP_NS,
    )

    assert len(controller.applied) == 1
    assert controller.hold_reasons == []


def test_dispatcher_rejects_wire_response_from_another_session(
    action_spec: BimanualActionSpec,
) -> None:
    controller = FakeController(action_spec)
    dispatcher = make_dispatcher(action_spec, controller)

    with pytest.raises(BimanualDispatchError, match="result session_id"):
        dispatcher.dispatch(
            make_step_result(session_id="f" * 32),
            target_timestamp_ns=TARGET_TIMESTAMP_NS,
        )

    assert controller.applied == []
    assert controller.safety_faulted is True


def test_dispatcher_canonicalizes_ndarray_subclass_before_checks(
    action_spec: BimanualActionSpec,
) -> None:
    class MisleadingArray(np.ndarray):
        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            del ufunc, method, inputs, kwargs
            return np.asarray(True)

    controller = FakeController(action_spec)
    dispatcher = make_dispatcher(action_spec, controller)
    result = make_step_result()
    malicious = np.asarray(result["actions"]["right"]).view(MisleadingArray)
    malicious[0] = np.nan
    result["actions"]["right"] = malicious

    with pytest.raises(BimanualDispatchError, match=r"right.*finite"):
        dispatcher.dispatch(result, target_timestamp_ns=TARGET_TIMESTAMP_NS)

    assert controller.applied == []
    assert controller.safety_faulted is True


def test_dispatcher_rechecks_controller_clock_immediately_before_apply(
    action_spec: BimanualActionSpec,
) -> None:
    class AdvancingClockController(FakeController):
        def read_clock_ns(self, *, dispatch_lease: object) -> int:
            timestamp = super().read_clock_ns(dispatch_lease=dispatch_lease)
            if len(self.clock_reads) == 1:
                self.clock_timestamp_ns = TARGET_TIMESTAMP_NS + 1
            return timestamp

    controller = AdvancingClockController(action_spec)
    dispatcher = make_dispatcher(action_spec, controller)

    with pytest.raises(BimanualDispatchError, match="future target"):
        dispatcher.dispatch(make_step_result(), target_timestamp_ns=TARGET_TIMESTAMP_NS)

    assert controller.applied == []
    assert len(controller.clock_reads) == 2


def test_controller_deadline_closes_read_to_apply_timing_window(
    action_spec: BimanualActionSpec,
) -> None:
    class DeadlineController(FakeController):
        def apply_bimanual_action(
            self,
            left_action: np.ndarray,
            right_action: np.ndarray,
            *,
            target_timestamp_ns: int,
            not_before_timestamp_ns: int,
            not_after_timestamp_ns: int,
            clock_domain: str,
            dispatch_lease: object,
            expected_recovery_epoch: int,
        ) -> None:
            self.clock_timestamp_ns = target_timestamp_ns
            super().apply_bimanual_action(
                left_action,
                right_action,
                target_timestamp_ns=target_timestamp_ns,
                not_before_timestamp_ns=not_before_timestamp_ns,
                not_after_timestamp_ns=not_after_timestamp_ns,
                clock_domain=clock_domain,
                dispatch_lease=dispatch_lease,
                expected_recovery_epoch=expected_recovery_epoch,
            )

    controller = DeadlineController(action_spec)
    dispatcher = make_dispatcher(action_spec, controller)

    with pytest.raises(BimanualDispatchError, match="expired dispatch deadline"):
        dispatcher.dispatch(make_step_result(), target_timestamp_ns=TARGET_TIMESTAMP_NS)

    assert controller.applied == []


def test_controller_rejects_clock_rollback_inside_atomic_apply(
    action_spec: BimanualActionSpec,
) -> None:
    class RollbackController(FakeController):
        def read_clock_ns(self, *, dispatch_lease: object) -> int:
            timestamp = super().read_clock_ns(dispatch_lease=dispatch_lease)
            if len(self.clock_reads) == 1:
                self.clock_timestamp_ns += 1_000
            return timestamp

        def apply_bimanual_action(
            self,
            left_action: np.ndarray,
            right_action: np.ndarray,
            *,
            target_timestamp_ns: int,
            not_before_timestamp_ns: int,
            not_after_timestamp_ns: int,
            clock_domain: str,
            dispatch_lease: object,
            expected_recovery_epoch: int,
        ) -> None:
            self.clock_timestamp_ns = not_before_timestamp_ns - 1
            super().apply_bimanual_action(
                left_action,
                right_action,
                target_timestamp_ns=target_timestamp_ns,
                not_before_timestamp_ns=not_before_timestamp_ns,
                not_after_timestamp_ns=not_after_timestamp_ns,
                clock_domain=clock_domain,
                dispatch_lease=dispatch_lease,
                expected_recovery_epoch=expected_recovery_epoch,
            )

    controller = RollbackController(action_spec)
    dispatcher = make_dispatcher(action_spec, controller)

    with pytest.raises(BimanualDispatchError, match="clock rollback"):
        dispatcher.dispatch(make_step_result(), target_timestamp_ns=TARGET_TIMESTAMP_NS)

    assert controller.applied == []
    assert controller.safety_faulted is True


def test_dispatcher_accepts_only_the_next_step_at_the_declared_period(
    action_spec: BimanualActionSpec,
) -> None:
    controller = FakeController(action_spec)
    dispatcher = make_dispatcher(action_spec, controller)
    next_target_timestamp_ns = TARGET_TIMESTAMP_NS + round(1_000_000_000 / action_spec.control_frequency_hz)

    dispatcher.dispatch(
        make_step_result(),
        target_timestamp_ns=TARGET_TIMESTAMP_NS,
    )
    controller.clock_timestamp_ns = TARGET_TIMESTAMP_NS
    dispatcher.dispatch(
        make_step_result(chunk_step_index=1),
        target_timestamp_ns=next_target_timestamp_ns,
    )

    assert len(controller.applied) == 2
    assert dispatcher.faulted is False


@pytest.mark.parametrize(
    ("second_result", "period_offset_ns", "message"),
    [
        (make_step_result(chunk_step_index=1, source_timestamp_ns=SOURCE_TIMESTAMP_NS + 1), 0, "changed inside"),
        (make_step_result(chunk_sequence_id=2), 0, "before the previous execution horizon completed"),
        (make_step_result(chunk_step_index=1), -12_000_000, "target control period"),
    ],
)
def test_dispatcher_rejects_broken_chunk_sequence_invariants(
    action_spec: BimanualActionSpec,
    second_result: dict[str, object],
    period_offset_ns: int,
    message: str,
) -> None:
    controller = FakeController(action_spec)
    dispatcher = make_dispatcher(action_spec, controller)
    expected_period_ns = round(1_000_000_000 / action_spec.control_frequency_hz)

    dispatcher.dispatch(make_step_result(), target_timestamp_ns=TARGET_TIMESTAMP_NS)
    controller.clock_timestamp_ns = TARGET_TIMESTAMP_NS

    with pytest.raises(BimanualDispatchError, match=message):
        dispatcher.dispatch(
            second_result,
            target_timestamp_ns=TARGET_TIMESTAMP_NS + expected_period_ns + period_offset_ns,
        )

    assert len(controller.applied) == 1


def test_dispatcher_requires_first_chunk_to_start_at_step_zero(action_spec: BimanualActionSpec) -> None:
    controller = FakeController(action_spec)
    dispatcher = make_dispatcher(action_spec, controller)

    with pytest.raises(BimanualDispatchError, match="first dispatched step must be 0"):
        dispatcher.dispatch(
            make_step_result(chunk_step_index=1),
            target_timestamp_ns=TARGET_TIMESTAMP_NS,
        )

    assert controller.applied == []


def test_dispatcher_replay_enters_hold_and_latches_fault(action_spec: BimanualActionSpec) -> None:
    controller = FakeController(action_spec)
    dispatcher = make_dispatcher(action_spec, controller)
    next_target_timestamp_ns = TARGET_TIMESTAMP_NS + round(1_000_000_000 / action_spec.control_frequency_hz)

    dispatcher.dispatch(
        make_step_result(),
        target_timestamp_ns=TARGET_TIMESTAMP_NS,
    )
    controller.clock_timestamp_ns = TARGET_TIMESTAMP_NS
    with pytest.raises(BimanualDispatchError, match="expected the next cached-chunk step 1"):
        dispatcher.dispatch(
            make_step_result(),
            target_timestamp_ns=next_target_timestamp_ns,
        )
    with pytest.raises(BimanualDispatchError, match="construct a new dispatcher"):
        dispatcher.dispatch(
            make_step_result(chunk_step_index=1),
            target_timestamp_ns=next_target_timestamp_ns,
        )

    assert len(controller.applied) == 1
    assert len(controller.hold_reasons) == 1
    assert dispatcher.faulted is True


def test_dispatch_rejection_enters_hold_even_when_error_string_is_broken(
    action_spec: BimanualActionSpec,
) -> None:
    class BrokenStrError(Exception):
        def __str__(self) -> str:
            raise RuntimeError("broken exception formatter")

    controller = FakeController(action_spec)
    dispatcher = make_dispatcher(action_spec, controller)

    with pytest.raises(BimanualDispatchError, match="could not be formatted"):
        dispatcher.reject_without_dispatch(BrokenStrError())

    assert controller.safety_faulted is True
    assert len(controller.hold_reasons) == 1


def test_dispatch_rejection_holds_on_keyboard_interrupt(
    action_spec: BimanualActionSpec,
) -> None:
    controller = FakeController(action_spec)
    dispatcher = make_dispatcher(action_spec, controller)

    with pytest.raises(BimanualDispatchError, match="KeyboardInterrupt"):
        dispatcher.reject_without_dispatch(KeyboardInterrupt())

    assert controller.safety_faulted is True


def test_dispatch_rejection_holds_before_reading_hostile_exception_type_metadata(
    action_spec: BimanualActionSpec,
) -> None:
    events: list[str] = []

    class HostileExceptionType(type):
        def __getattribute__(cls, name: str) -> object:
            if name in {"__module__", "__qualname__"}:
                events.append("metadata")
                raise KeyboardInterrupt()
            return super().__getattribute__(name)

    class HostileTypeMetadataError(Exception, metaclass=HostileExceptionType):
        pass

    class RecordingHoldController(FakeController):
        def hold(
            self,
            *,
            reason: str,
            dispatch_lease: object,
            expected_recovery_epoch: int,
        ) -> BimanualHoldReceipt:
            events.append("hold")
            return super().hold(
                reason=reason,
                dispatch_lease=dispatch_lease,
                expected_recovery_epoch=expected_recovery_epoch,
            )

    error = HostileTypeMetadataError()
    events.clear()
    controller = RecordingHoldController(action_spec)
    dispatcher = make_dispatcher(action_spec, controller)

    with pytest.raises(BimanualDispatchError, match="could not be formatted"):
        dispatcher.reject_without_dispatch(error)

    assert events[:2] == ["hold", "metadata"]
    assert controller.safety_faulted is True
    assert len(controller.hold_reasons) == 1


def test_dispatch_rejection_wraps_keyboard_interrupt_from_safe_hold(
    action_spec: BimanualActionSpec,
) -> None:
    class InterruptingHoldController(FakeController):
        def hold(
            self,
            *,
            reason: str,
            dispatch_lease: object,
            expected_recovery_epoch: int,
        ) -> BimanualHoldReceipt:
            del reason, dispatch_lease, expected_recovery_epoch
            raise KeyboardInterrupt()

    controller = InterruptingHoldController(action_spec)
    dispatcher = make_dispatcher(action_spec, controller)

    with pytest.raises(BimanualDispatchError, match=r"safe hold also failed: .*KeyboardInterrupt"):
        dispatcher.reject_without_dispatch(RuntimeError("reject"))

    assert dispatcher.faulted is True


def test_dispatcher_holds_when_post_apply_check_is_interrupted(
    action_spec: BimanualActionSpec,
) -> None:
    class InterruptingController(FakeController):
        def validate_dispatch_lease(
            self,
            dispatch_lease: object,
            *,
            expected_spec: BimanualActionSpec,
            expected_clock_domain: str,
            expected_recovery_epoch: int,
        ) -> None:
            super().validate_dispatch_lease(
                dispatch_lease,
                expected_spec=expected_spec,
                expected_clock_domain=expected_clock_domain,
                expected_recovery_epoch=expected_recovery_epoch,
            )
            if self.applied:
                raise KeyboardInterrupt()

    controller = InterruptingController(action_spec)
    dispatcher = make_dispatcher(action_spec, controller)

    with pytest.raises(BimanualDispatchError, match="KeyboardInterrupt"):
        dispatcher.dispatch(make_step_result(), target_timestamp_ns=TARGET_TIMESTAMP_NS)

    assert len(controller.applied) == 1
    assert controller.safety_faulted is True


def test_dispatcher_serializes_concurrent_duplicate_step(action_spec: BimanualActionSpec) -> None:
    class BlockingController(FakeController):
        def __init__(self, spec: BimanualActionSpec) -> None:
            super().__init__(spec)
            self.apply_entered = threading.Event()
            self.release_apply = threading.Event()

        def apply_bimanual_action(
            self,
            left_action: np.ndarray,
            right_action: np.ndarray,
            *,
            target_timestamp_ns: int,
            not_before_timestamp_ns: int,
            not_after_timestamp_ns: int,
            clock_domain: str,
            dispatch_lease: object,
            expected_recovery_epoch: int,
        ) -> None:
            self.apply_entered.set()
            if not self.release_apply.wait(timeout=2.0):
                raise RuntimeError("timed out waiting to finish fake controller apply")
            super().apply_bimanual_action(
                left_action,
                right_action,
                target_timestamp_ns=target_timestamp_ns,
                not_before_timestamp_ns=not_before_timestamp_ns,
                not_after_timestamp_ns=not_after_timestamp_ns,
                clock_domain=clock_domain,
                dispatch_lease=dispatch_lease,
                expected_recovery_epoch=expected_recovery_epoch,
            )

    controller = BlockingController(action_spec)
    dispatcher = make_dispatcher(action_spec, controller)
    result = make_step_result()

    def dispatch_same_step() -> None:
        dispatcher.dispatch(
            result,
            target_timestamp_ns=TARGET_TIMESTAMP_NS,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(dispatch_same_step)
        assert controller.apply_entered.wait(timeout=2.0)
        duplicate = executor.submit(dispatch_same_step)
        controller.release_apply.set()
        first.result(timeout=2.0)
        with pytest.raises(BimanualDispatchError):
            duplicate.result(timeout=2.0)

    assert len(controller.applied) == 1
    assert controller.safety_faulted is True


def test_dispatcher_rejects_step_outside_execution_horizon(action_spec: BimanualActionSpec) -> None:
    controller = FakeController(action_spec)
    dispatcher = make_dispatcher(action_spec, controller, execution_horizon=1)

    with pytest.raises(BimanualDispatchError, match="expected < execution_horizon 1"):
        dispatcher.dispatch(
            make_step_result(chunk_step_index=1),
            target_timestamp_ns=TARGET_TIMESTAMP_NS,
        )

    assert controller.applied == []
    assert controller.safety_faulted is True


def test_dispatcher_allows_new_chunk_from_same_observation_timestamp(
    action_spec: BimanualActionSpec,
) -> None:
    controller = FakeController(action_spec)
    dispatcher = make_dispatcher(action_spec, controller, execution_horizon=1)
    next_target_timestamp_ns = TARGET_TIMESTAMP_NS + round(1_000_000_000 / action_spec.control_frequency_hz)

    dispatcher.dispatch(
        make_step_result(chunk_sequence_id=1),
        target_timestamp_ns=TARGET_TIMESTAMP_NS,
    )
    controller.clock_timestamp_ns = TARGET_TIMESTAMP_NS
    dispatcher.dispatch(
        make_step_result(chunk_sequence_id=2),
        target_timestamp_ns=next_target_timestamp_ns,
    )

    assert len(controller.applied) == 2


def test_dispatcher_rejects_skipped_chunk_sequence_id(action_spec: BimanualActionSpec) -> None:
    controller = FakeController(action_spec)
    dispatcher = make_dispatcher(action_spec, controller, execution_horizon=1)
    next_target_timestamp_ns = TARGET_TIMESTAMP_NS + round(1_000_000_000 / action_spec.control_frequency_hz)

    dispatcher.dispatch(
        make_step_result(chunk_sequence_id=1),
        target_timestamp_ns=TARGET_TIMESTAMP_NS,
    )
    controller.clock_timestamp_ns = TARGET_TIMESTAMP_NS
    with pytest.raises(BimanualDispatchError, match="expected the next adapter chunk id 2, got 3"):
        dispatcher.dispatch(
            make_step_result(chunk_sequence_id=3),
            target_timestamp_ns=next_target_timestamp_ns,
        )

    assert len(controller.applied) == 1
    assert controller.safety_faulted is True


def test_dispatcher_rechecks_controller_contract_before_apply(action_spec: BimanualActionSpec) -> None:
    controller = FakeController(action_spec)
    dispatcher = make_dispatcher(action_spec, controller)
    controller.action_spec = dataclasses.replace(action_spec, coordinate_frame="mutated_frame")

    with pytest.raises(BimanualDispatchError, match=r"controller\.action_spec conflicts"):
        dispatcher.dispatch(
            make_step_result(),
            target_timestamp_ns=TARGET_TIMESTAMP_NS,
        )

    assert controller.applied == []
    assert controller.safety_faulted is True


def test_dispatcher_rejects_controller_recovery_epoch_change(action_spec: BimanualActionSpec) -> None:
    controller = FakeController(action_spec)
    dispatcher = make_dispatcher(action_spec, controller)
    controller.recover()

    with pytest.raises(BimanualDispatchError, match=r"recovery_epoch changed.*safe hold also failed"):
        dispatcher.dispatch(
            make_step_result(),
            target_timestamp_ns=TARGET_TIMESTAMP_NS,
        )

    assert controller.applied == []
    assert controller.safety_faulted is False
    assert dispatcher.faulted is True
    assert make_dispatcher(action_spec, controller).faulted is False


@pytest.mark.parametrize(
    "rotation_6d",
    [
        np.zeros((6,), dtype=np.float32),
        np.asarray([1.0, 0.0, 0.0, 2.0, 0.0, 0.0], dtype=np.float32),
    ],
)
def test_dispatcher_rejects_degenerate_rotation_6d(
    action_spec: BimanualActionSpec,
    rotation_6d: np.ndarray,
) -> None:
    controller = FakeController(action_spec)
    dispatcher = make_dispatcher(action_spec, controller)
    result = make_step_result()
    result["actions"]["left"][3:9] = rotation_6d

    with pytest.raises(BimanualDispatchError, match="rotation_6d"):
        dispatcher.dispatch(
            result,
            target_timestamp_ns=TARGET_TIMESTAMP_NS,
        )

    assert controller.applied == []
    assert len(controller.hold_reasons) == 1


@pytest.mark.parametrize(
    ("result", "controller_clock_ns", "target_timestamp_ns", "message"),
    [
        (
            make_step_result(source_timestamp_ns=900_000_000),
            CURRENT_TIMESTAMP_NS,
            TARGET_TIMESTAMP_NS,
            "observation age",
        ),
        (
            make_step_result(source_timestamp_ns=CURRENT_TIMESTAMP_NS + 1),
            CURRENT_TIMESTAMP_NS,
            TARGET_TIMESTAMP_NS,
            "no newer than current",
        ),
        (
            make_step_result(clock_domain="host_monotonic"),
            CURRENT_TIMESTAMP_NS,
            TARGET_TIMESTAMP_NS,
            "clock_domain",
        ),
        (
            make_step_result(),
            CURRENT_TIMESTAMP_NS,
            CURRENT_TIMESTAMP_NS,
            "future target",
        ),
        (
            make_step_result(),
            CURRENT_TIMESTAMP_NS,
            CURRENT_TIMESTAMP_NS + 21_000_000,
            "target lead",
        ),
    ],
)
def test_dispatcher_holds_without_apply_on_timing_or_clock_failure(
    action_spec: BimanualActionSpec,
    result: dict[str, object],
    controller_clock_ns: int,
    target_timestamp_ns: int,
    message: str,
) -> None:
    controller = FakeController(action_spec)
    controller.clock_timestamp_ns = controller_clock_ns
    dispatcher = make_dispatcher(action_spec, controller)

    with pytest.raises(BimanualDispatchError, match=message):
        dispatcher.dispatch(
            result,
            target_timestamp_ns=target_timestamp_ns,
        )

    assert controller.applied == []
    assert len(controller.hold_reasons) == 1


def test_dispatcher_validates_both_hands_before_apply(action_spec: BimanualActionSpec) -> None:
    controller = FakeController(action_spec)
    dispatcher = make_dispatcher(action_spec, controller)
    result = make_step_result()
    result["actions"]["right"][0] = np.nan

    with pytest.raises(BimanualDispatchError, match=r"right.*finite"):
        dispatcher.dispatch(
            result,
            target_timestamp_ns=TARGET_TIMESTAMP_NS,
        )

    assert controller.applied == []
    assert len(controller.hold_reasons) == 1


def test_joint_dispatch_skips_cartesian_rotation_check(action_spec: BimanualActionSpec) -> None:
    joint_spec = spec_for_representation(action_spec, ActionRepresentation.JOINT_29D)
    controller = FakeController(joint_spec)
    dispatcher = make_dispatcher(joint_spec, controller)

    dispatcher.dispatch(make_joint_step_result(), target_timestamp_ns=TARGET_TIMESTAMP_NS)

    assert len(controller.applied) == 1
    assert controller.applied[0][0].shape == (JOINT_LOGICAL_ACTION_DIM,)
    assert controller.applied[0][1].shape == (JOINT_LOGICAL_ACTION_DIM,)


def test_joint_metadata_broker_dispatches_and_commits_29d_step(
    action_spec: BimanualActionSpec,
) -> None:
    joint_spec = spec_for_representation(action_spec, ActionRepresentation.JOINT_29D)
    logical_policy = FakeLogicalPolicy(joint_spec)
    logical_policy.left_actions = np.zeros(
        (joint_spec.physical_horizon, JOINT_LOGICAL_ACTION_DIM),
        dtype=np.float64,
    )
    logical_policy.right_actions = np.zeros_like(logical_policy.left_actions)
    logical_policy.left_actions[:, 0] = 11.0
    logical_policy.right_actions[:, 0] = 22.0
    wire_policy = BimanualPolicyAdapter(logical_policy, joint_spec, execution_horizon=1)
    broker = BimanualActionChunkBroker.from_metadata(
        wire_policy,
        wire_policy.metadata,
        joint_spec,
        expected_execution_horizon=1,
    )
    controller = FakeController(joint_spec)
    dispatcher = make_dispatcher(
        joint_spec,
        controller,
        execution_horizon=1,
        session_id=broker.session_id,
    )

    result = broker.dispatch_next(
        make_observation(),
        dispatcher,
        target_timestamp_ns=TARGET_TIMESTAMP_NS,
    )

    assert result["chunk_step_index"] == 0
    assert result["actions"]["left"].shape == (JOINT_LOGICAL_ACTION_DIM,)
    assert result["actions"]["right"].shape == (JOINT_LOGICAL_ACTION_DIM,)
    assert len(controller.applied) == 1
    assert controller.applied[0][0][0] == 11.0
    assert controller.applied[0][1][0] == 22.0

    next_result = broker.peek(make_observation())

    assert logical_policy.calls == 2
    assert next_result["chunk_sequence_id"] == 2
    assert next_result["chunk_step_index"] == 0


def test_dispatcher_holds_without_apply_on_limit_violation(action_spec: BimanualActionSpec) -> None:
    controller = FakeController(action_spec)
    lower = np.full((LOGICAL_ACTION_DIM,), -1.0, dtype=np.float32)
    upper = np.full((LOGICAL_ACTION_DIM,), 1.0, dtype=np.float32)
    limits = BimanualSafetyLimits(action_spec, lower, upper, lower, upper)
    dispatcher = make_dispatcher(action_spec, controller, limits=limits)
    result = make_step_result()
    result["actions"]["right"][5] = 2.0

    with pytest.raises(BimanualDispatchError, match="right action exceeds"):
        dispatcher.dispatch(
            result,
            target_timestamp_ns=TARGET_TIMESTAMP_NS,
        )

    assert controller.applied == []
    assert len(controller.hold_reasons) == 1


def test_dispatcher_reports_when_safe_hold_also_fails(action_spec: BimanualActionSpec) -> None:
    controller = FakeController(action_spec, fail_apply=True, fail_hold=True)
    dispatcher = make_dispatcher(action_spec, controller)

    with pytest.raises(BimanualDispatchError, match=r"safe hold also failed: .*hold failed"):
        dispatcher.dispatch(
            make_step_result(),
            target_timestamp_ns=TARGET_TIMESTAMP_NS,
        )


def test_dispatcher_rejects_invalid_safe_hold_acknowledgement(
    action_spec: BimanualActionSpec,
) -> None:
    class InvalidReceiptController(FakeController):
        def hold(
            self,
            *,
            reason: str,
            dispatch_lease: object,
            expected_recovery_epoch: int,
        ) -> BimanualHoldReceipt:
            super().hold(
                reason=reason,
                dispatch_lease=dispatch_lease,
                expected_recovery_epoch=expected_recovery_epoch,
            )
            return BimanualHoldReceipt(
                safety_faulted=True,
                recovery_epoch=expected_recovery_epoch + 1,
            )

    controller = InvalidReceiptController(action_spec, fail_apply=True)
    dispatcher = make_dispatcher(action_spec, controller)

    with pytest.raises(BimanualDispatchError, match="acknowledgement conflicts"):
        dispatcher.dispatch(make_step_result(), target_timestamp_ns=TARGET_TIMESTAMP_NS)

    assert controller.safety_faulted is True
