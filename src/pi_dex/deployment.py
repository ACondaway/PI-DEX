"""Validated bimanual wire, acknowledgement, and dispatch boundaries.

The preferred serving order is::

    OpenPI Policy -> BimanualPolicyAdapter -> WebSocket server

The wrapped OpenPI policy must already have unpacked and inverse-normalized its
model output into per-hand logical actions. On the robot, a
``BimanualActionChunkBroker`` exposes one matched pair at a time without
consuming it until dispatch succeeds.
"""

from __future__ import annotations

import dataclasses
import math
import threading
import uuid
from collections.abc import Mapping
from typing import Any
from typing import NoReturn
from typing import Protocol

import numpy as np

from pi_dex.actions import LOGICAL_ACTION_DIM
from pi_dex.spec import BimanualActionSpec

DEPLOYMENT_WIRE_FORMAT = "paired_logical_float32_v2"
OBSERVATION_TIMESTAMP_FIELD = "observation_timestamp_ns"
CLOCK_DOMAIN_FIELD = "clock_domain"
SOURCE_TIMESTAMP_FIELD = "source_timestamp_ns"
CHUNK_STEP_INDEX_FIELD = "chunk_step_index"
CHUNK_SEQUENCE_ID_FIELD = "chunk_sequence_id"
SESSION_ID_FIELD = "session_id"

_PASSTHROUGH_RESULT_FIELDS = ("policy_timing", "server_timing")
_WIRE_RESULT_FIELDS = {
    "actions",
    SOURCE_TIMESTAMP_FIELD,
    CLOCK_DOMAIN_FIELD,
    CHUNK_SEQUENCE_ID_FIELD,
    SESSION_ID_FIELD,
    *_PASSTHROUGH_RESULT_FIELDS,
}
_DISPATCH_RESULT_FIELDS = {*_WIRE_RESULT_FIELDS, CHUNK_STEP_INDEX_FIELD}
_DEPLOYMENT_METADATA_FIELDS = frozenset(
    {"wire_format", "execution_horizon", SESSION_ID_FIELD}
)
_ROTATION_6D_START = 3
_ROTATION_6D_STOP = 9
_MIN_ROTATION_VECTOR_NORM = 1e-6
_MAX_ROTATION_VECTOR_COSINE = 0.9999


class PolicyLike(Protocol):
    """Minimal synchronous policy interface used by deployment adapters.

    After ``infer`` returns, the producer must transfer exclusive ownership of
    the complete result tree to the caller until it has been snapshotted. A
    background producer may not reuse or mutate any returned container or array.
    """

    def infer(self, observation: dict[str, Any]) -> Mapping[str, Any]:
        """Return one unbatched policy result."""

    def reset(self) -> None:
        """Reset episode-local state, or safely do nothing when stateless."""


class MetadataPolicyLike(PolicyLike, Protocol):
    """Inference policy whose metadata carries a verified PI-DEX contract."""

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return policy metadata containing the required ``pi_dex`` mapping."""


class BimanualController(Protocol):
    """Robot controller required by ``BimanualCommandDispatcher``.

    ``apply_bimanual_action`` is intentionally one paired call. A concrete
    Sharpa adapter must implement it using an atomic bimanual API or stage both
    hands against the same target timestamp before committing. Two sequential
    writes do not satisfy this protocol.
    """

    @property
    def action_spec(self) -> BimanualActionSpec:
        """Return the immutable semantic contract implemented by this controller."""

    @property
    def safety_faulted(self) -> bool:
        """Whether the controller is currently latched in its safe state."""

    @property
    def recovery_epoch(self) -> int:
        """Return a monotonic counter incremented after verified recovery."""

    def acquire_dispatch_lease(
        self,
        *,
        expected_spec: BimanualActionSpec,
        expected_clock_domain: str,
        expected_recovery_epoch: int,
    ) -> object:
        """Atomically acquire the controller's only active dispatch lease.

        The backend must bind the lease to all expected values. If this method
        raises or returns ``None``, it must leave no newly active lease behind.
        """

    def validate_dispatch_lease(
        self,
        dispatch_lease: object,
        *,
        expected_spec: BimanualActionSpec,
        expected_clock_domain: str,
        expected_recovery_epoch: int,
    ) -> None:
        """Atomically validate the active lease, epoch, and nonfaulted state."""

    def read_clock_ns(self, *, dispatch_lease: object) -> int:
        """Read the trusted clock named by ``action_spec.clock_domain``."""

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
        """Validate lease/time atomically, then apply one paired 31D command.

        Immediately before committing, the controller must use its own clock to
        require ``not_before_timestamp_ns <= now <= not_after_timestamp_ns <
        target_timestamp_ns``. The closed time window rejects both forward expiry
        and a clock rollback after the dispatcher's final trusted read.
        """

    def hold(
        self,
        *,
        reason: str,
        dispatch_lease: object,
        expected_recovery_epoch: int,
    ) -> BimanualHoldReceipt:
        """Atomically enter safe hold, invalidate the lease, and acknowledge it."""


@dataclasses.dataclass(frozen=True, slots=True)
class BimanualHoldReceipt:
    """Atomic controller acknowledgement of a safe-hold transition.

    Attributes:
        safety_faulted: Must be exactly ``True`` after the hold transition.
        recovery_epoch: Controller epoch in which the lease was invalidated.
    """

    safety_faulted: bool
    recovery_epoch: int

    def __post_init__(self) -> None:
        if type(self.safety_faulted) is not bool:
            raise TypeError(
                "safety_faulted: expected bool, "
                f"got {type(self.safety_faulted).__name__}"
            )
        _validate_nonnegative_int(self.recovery_epoch, field_name="recovery_epoch")


@dataclasses.dataclass(frozen=True, slots=True, init=False)
class BimanualSafetyLimits:
    """Closed per-dimension limits bound to one semantic action contract.

    Bounds have shape ``[31]`` and float32 dtype. They use the exact units and
    frame declared by ``spec``: wrist position in metres, rotation 6D
    dimensionless, and hand joints in radians. Arrays are copied onto immutable
    byte backing, so callers cannot restore write access after initialization.
    """

    spec: BimanualActionSpec
    _left_min_bytes: bytes
    _left_max_bytes: bytes
    _right_min_bytes: bytes
    _right_max_bytes: bytes

    def __init__(
        self,
        spec: BimanualActionSpec,
        left_min: np.ndarray,
        left_max: np.ndarray,
        right_min: np.ndarray,
        right_max: np.ndarray,
    ) -> None:
        """Copy and validate closed ``[31]`` float32 limits for both hands.

        Raises:
            TypeError: If ``spec`` or an array/dtype is invalid.
            ValueError: If a shape/value is invalid or a minimum exceeds maximum.
        """
        validated_spec = _validated_spec_copy(spec)
        validated_left_min = _validate_action_vector(left_min, field_name="left_min")
        validated_left_max = _validate_action_vector(left_max, field_name="left_max")
        validated_right_min = _validate_action_vector(right_min, field_name="right_min")
        validated_right_max = _validate_action_vector(right_max, field_name="right_max")
        if np.any(validated_left_min > validated_left_max):
            raise ValueError("left safety limits: every minimum must be <= its maximum")
        if np.any(validated_right_min > validated_right_max):
            raise ValueError("right safety limits: every minimum must be <= its maximum")
        object.__setattr__(self, "spec", validated_spec)
        object.__setattr__(self, "_left_min_bytes", validated_left_min.tobytes(order="C"))
        object.__setattr__(self, "_left_max_bytes", validated_left_max.tobytes(order="C"))
        object.__setattr__(self, "_right_min_bytes", validated_right_min.tobytes(order="C"))
        object.__setattr__(self, "_right_max_bytes", validated_right_max.tobytes(order="C"))

    @property
    def left_min(self) -> np.ndarray:
        """Return a fresh read-only float32 ``[31]`` view of left minima."""
        return _action_vector_from_bytes(self._left_min_bytes)

    @property
    def left_max(self) -> np.ndarray:
        """Return a fresh read-only float32 ``[31]`` view of left maxima."""
        return _action_vector_from_bytes(self._left_max_bytes)

    @property
    def right_min(self) -> np.ndarray:
        """Return a fresh read-only float32 ``[31]`` view of right minima."""
        return _action_vector_from_bytes(self._right_min_bytes)

    @property
    def right_max(self) -> np.ndarray:
        """Return a fresh read-only float32 ``[31]`` view of right maxima."""
        return _action_vector_from_bytes(self._right_max_bytes)


class BimanualPolicyAdapter:
    """Expose inverse-normalized per-hand policy output on the wire.

    The wrapped policy must return ``left_actions/right_actions[K,31]`` after
    inverse normalization. Raw model-space ``actions[2*K,32]`` are deliberately
    rejected: dimensional decoding alone cannot establish physical units.

    The caller supplies two transport-only observation fields:
    ``observation_timestamp_ns`` and ``clock_domain``. They are validated and
    removed from a recursive snapshot before invoking OpenPI, because OpenPI
    attempts to convert every observation leaf into a tensor. Arrays and nested
    containers are copied. The producer must synchronize writes while the
    snapshot itself is being captured.
    """

    def __init__(
        self,
        policy: MetadataPolicyLike,
        spec: BimanualActionSpec,
        *,
        execution_horizon: int | None = None,
    ) -> None:
        """Initialize an episode-independent decoded-chunk policy adapter.

        Args:
            policy: Policy returning inverse-normalized per-hand chunks and
                exposing verified ``pi_dex`` training metadata.
            spec: Training and deployment semantic contract.
            execution_horizon: Leading physical steps to expose. It defaults to
                ``K`` and must lie in ``[1, K]``.

        Raises:
            TypeError: If arguments or existing metadata have invalid types.
            ValueError: If the horizon or existing metadata conflicts with the
                requested deployment contract.
        """
        validated_spec = _validated_spec_copy(spec)
        self._policy = policy
        self._spec = validated_spec
        self._execution_horizon = _validate_execution_horizon(execution_horizon, spec=validated_spec)
        self._metadata = _merge_metadata(
            policy,
            spec=validated_spec,
            execution_horizon=self._execution_horizon,
        )
        self._session_id = self._metadata["pi_dex"][SESSION_ID_FIELD]
        self._inference_lock = threading.Lock()
        self._last_chunk_sequence_id = 0

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Validate one observation and return a timestamped paired chunk.

        Args:
            observation: Unbatched policy observation plus positive integer
                ``observation_timestamp_ns`` and non-empty ``clock_domain``.

        Returns:
            A mapping containing ``actions.left/right`` with shape ``[E,31]``
            and float32 dtype, plus ``source_timestamp_ns``, ``clock_domain``,
            an adapter-instance-local monotonic ``chunk_sequence_id``, and the
            adapter-instance ``session_id`` bound into deployment metadata.

        Raises:
            TypeError: If containers, timestamps, clock, or actions have invalid
                types or dtypes.
            KeyError: If a required transport or per-hand action field is absent.
            ValueError: If shapes, values, or the output representation are
                invalid. Raw interleaved model actions are never accepted.
        """
        with self._inference_lock:
            policy_observation, source_timestamp_ns, clock_domain = _extract_transport_observation(observation)
            if clock_domain != self._spec.clock_domain:
                raise ValueError(
                    f"observation clock_domain: expected {self._spec.clock_domain!r}, got {clock_domain!r}"
                )
            raw_result = self._policy.infer(policy_observation)
            if not isinstance(raw_result, Mapping):
                raise TypeError(f"policy result: expected a mapping, got {type(raw_result).__name__}")

            left_actions, right_actions = _extract_inverse_normalized_chunk(raw_result, spec=self._spec)
            left_actions = _to_wire_float32(
                left_actions[: self._execution_horizon],
                field_name="actions.left",
            )
            right_actions = _to_wire_float32(
                right_actions[: self._execution_horizon],
                field_name="actions.right",
            )

            result: dict[str, Any] = {
                "actions": {"left": left_actions, "right": right_actions},
                SOURCE_TIMESTAMP_FIELD: source_timestamp_ns,
                CLOCK_DOMAIN_FIELD: clock_domain,
                SESSION_ID_FIELD: self._session_id,
            }
            for field_name in _PASSTHROUGH_RESULT_FIELDS:
                if field_name in raw_result:
                    timing = raw_result[field_name]
                    if not isinstance(timing, Mapping):
                        raise TypeError(
                            f"policy result {field_name!r}: expected a mapping, "
                            f"got {type(timing).__name__}"
                        )
                    result[field_name] = _snapshot_supported_mapping(
                        timing,
                        field_name=f"policy result {field_name!r}",
                    )
            self._last_chunk_sequence_id += 1
            result[CHUNK_SEQUENCE_ID_FIELD] = self._last_chunk_sequence_id
            return result

    def reset(self) -> None:
        """Delegate policy reset without reusing the monotonic chunk counter.

        Raises:
            TypeError: If the wrapped policy has no callable ``reset`` method.
        """
        with self._inference_lock:
            reset = getattr(self._policy, "reset", None)
            if not callable(reset):
                raise TypeError("policy.reset: expected a callable")
            reset()

    @property
    def metadata(self) -> dict[str, Any]:
        """Return an independent copy of versioned deployment metadata."""
        return _snapshot_supported_mapping(self._metadata, field_name="policy metadata")


def validate_deployment_metadata(
    metadata: object,
    spec: BimanualActionSpec,
    *,
    expected_execution_horizon: int | None = None,
) -> int:
    """Validate a server handshake and return its physical execution horizon.

    In addition to the full ``BimanualActionSpec`` contract, deployment metadata
    must declare ``wire_format``, ``execution_horizon``, and a non-empty
    ``session_id``. Supplying ``expected_execution_horizon`` makes client
    configuration mismatches fail at startup instead of at the first inference
    response.

    Args:
        metadata: Server handshake mapping with exact action and deployment fields.
        spec: Expected local semantic action contract.
        expected_execution_horizon: Optional client-required physical horizon.

    Returns:
        The validated positive physical execution horizon.

    Raises:
        TypeError: If metadata fields have invalid container or scalar types.
        ValueError: If fields are missing, unknown, or conflict with ``spec``.
    """
    validated_spec = _validated_spec_copy(spec)
    if not isinstance(metadata, Mapping):
        raise TypeError(f"metadata: expected a mapping, got {type(metadata).__name__}")
    metadata_snapshot = _snapshot_supported_mapping(metadata, field_name="metadata")
    validated_spec.validate_metadata(
        metadata_snapshot,
        allowed_extra_fields=_DEPLOYMENT_METADATA_FIELDS,
    )
    pi_dex_metadata = metadata_snapshot["pi_dex"]
    assert isinstance(pi_dex_metadata, Mapping)

    if "wire_format" not in pi_dex_metadata:
        raise ValueError("metadata['pi_dex']: missing required field 'wire_format'")
    wire_format = pi_dex_metadata["wire_format"]
    if type(wire_format) is not str:
        raise TypeError(
            "metadata['pi_dex']['wire_format']: expected str, "
            f"got {type(wire_format).__name__}"
        )
    if wire_format != DEPLOYMENT_WIRE_FORMAT:
        raise ValueError(
            f"metadata['pi_dex']['wire_format']: expected {DEPLOYMENT_WIRE_FORMAT!r}, got {wire_format!r}"
        )
    if "execution_horizon" not in pi_dex_metadata:
        raise ValueError("metadata['pi_dex']: missing required field 'execution_horizon'")
    declared_execution_horizon = pi_dex_metadata["execution_horizon"]
    if declared_execution_horizon is None:
        raise TypeError("metadata['pi_dex']['execution_horizon']: expected int, got NoneType")
    execution_horizon = _validate_execution_horizon(declared_execution_horizon, spec=validated_spec)
    _validate_execution_horizon_feasibility(
        execution_horizon,
        spec=validated_spec,
        max_target_lead_ms=validated_spec.max_command_lead_ms,
    )
    if SESSION_ID_FIELD not in pi_dex_metadata:
        raise ValueError(f"metadata['pi_dex']: missing required field {SESSION_ID_FIELD!r}")
    _validate_session_id(
        pi_dex_metadata[SESSION_ID_FIELD],
        field_name=f"metadata['pi_dex'][{SESSION_ID_FIELD!r}]",
    )

    if expected_execution_horizon is not None:
        expected = _validate_execution_horizon(expected_execution_horizon, spec=validated_spec)
        if execution_horizon != expected:
            raise ValueError(
                "metadata['pi_dex']['execution_horizon']: "
                f"expected client horizon {expected}, got {execution_horizon}"
            )
    return execution_horizon


def validate_execution_horizon(
    execution_horizon: int | None,
    spec: BimanualActionSpec,
) -> int:
    """Validate a requested physical execution horizon without side effects."""
    validated_spec = _validated_spec_copy(spec)
    value = _validate_execution_horizon(execution_horizon, spec=validated_spec)
    _validate_execution_horizon_feasibility(
        value,
        spec=validated_spec,
        max_target_lead_ms=validated_spec.max_command_lead_ms,
    )
    return value


@dataclasses.dataclass(frozen=True)
class _DecodedChunk:
    left_actions: np.ndarray
    right_actions: np.ndarray
    source_timestamp_ns: int
    clock_domain: str
    chunk_sequence_id: int
    session_id: str | None
    timing: dict[str, dict[str, Any]]


class BimanualBrokerFault(RuntimeError):
    """Raised when a broker is faulted and requires an explicit reset."""


class BimanualActionChunkBroker:
    """Peek paired steps and consume them only after a successful acknowledgement.

    Before a new synchronous inference request, the broker recursively snapshots
    the supported observation tree. The producer must still synchronize writes
    while that snapshot is captured; mutation after policy entry cannot alter the
    request passed to a WebSocket client or other remote policy.
    """

    def __init__(self, policy: PolicyLike, *, execution_horizon: int) -> None:
        """Initialize a non-dispatching broker for decoded chunks.

        Directly constructed brokers support ``peek``/``commit`` only. Hardware
        dispatch requires :meth:`from_metadata`, which binds the complete
        validated server action contract.
        """
        self._policy = policy
        self._execution_horizon = _validate_positive_int(execution_horizon, field_name="execution_horizon")
        self._action_spec: BimanualActionSpec | None = None
        self._session_id: str | None = None
        self._cached_chunk: _DecodedChunk | None = None
        self._step = 0
        self._awaiting_commit = False
        self._faulted = False
        self._lock = threading.RLock()

    @classmethod
    def from_metadata(
        cls,
        policy: PolicyLike,
        metadata: object,
        spec: BimanualActionSpec,
        *,
        expected_execution_horizon: int | None = None,
    ) -> BimanualActionChunkBroker:
        """Construct from validated server metadata, optionally requiring a client E."""
        validated_spec = _validated_spec_copy(spec)
        supplied_metadata = _snapshot_supported_mapping(metadata, field_name="metadata")
        policy_metadata = _read_policy_metadata(policy)
        if not _transport_values_equal(policy_metadata, supplied_metadata):
            raise ValueError("policy metadata does not match the supplied server-session metadata")
        execution_horizon = validate_deployment_metadata(
            supplied_metadata,
            validated_spec,
            expected_execution_horizon=expected_execution_horizon,
        )
        broker = cls(policy, execution_horizon=execution_horizon)
        broker._action_spec = validated_spec
        broker._session_id = _validate_session_id(
            supplied_metadata["pi_dex"][SESSION_ID_FIELD],
            field_name=f"metadata['pi_dex'][{SESSION_ID_FIELD!r}]",
        )
        return broker

    @property
    def faulted(self) -> bool:
        """Whether dispatch is locked until ``reset`` succeeds."""
        with self._lock:
            return self._faulted

    @property
    def execution_horizon(self) -> int:
        """Number of paired steps that must be acknowledged per decoded chunk."""
        return self._execution_horizon

    @property
    def session_id(self) -> str:
        """Return the server adapter session bound during metadata handshake."""
        if self._session_id is None:
            raise RuntimeError("broker session is unavailable without a metadata handshake")
        return self._session_id

    def peek(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Return the current pair without advancing the cached chunk.

        Repeated peeks return copies of the same physical step. The caller must
        call ``commit`` only after external application succeeds, or use
        ``dispatch_next`` for the fail-safe combined operation. When a new chunk
        is required, supported nested containers and NumPy arrays are copied
        before the policy call.
        """
        with self._lock:
            self._ensure_healthy()
            if not isinstance(observation, Mapping):
                raise TypeError(f"observation: expected a mapping, got {type(observation).__name__}")
            if self._cached_chunk is None:
                observation_snapshot = _snapshot_observation_mapping(observation)
                chunk_result = self._policy.infer(observation_snapshot)
                self._cached_chunk = _validate_decoded_chunk(
                    chunk_result,
                    execution_horizon=self._execution_horizon,
                    expected_session_id=self._session_id,
                )
                self._step = 0

            self._awaiting_commit = True
            return self._build_step_result()

    def commit(self) -> None:
        """Acknowledge successful application and advance exactly one step."""
        with self._lock:
            self._ensure_healthy()
            if self._cached_chunk is None or not self._awaiting_commit:
                raise RuntimeError("commit: no peeked bimanual action is awaiting acknowledgement")

            self._step += 1
            self._awaiting_commit = False
            if self._step == self._execution_horizon:
                self._cached_chunk = None
                self._step = 0

    def dispatch_next(
        self,
        observation: Mapping[str, Any],
        dispatcher: BimanualCommandDispatcher,
        *,
        target_timestamp_ns: int,
    ) -> dict[str, Any]:
        """Peek, dispatch with controller time, then commit one paired step.

        Args:
            observation: Unbatched policy input passed to the wrapped policy when
                a new decoded chunk is needed.
            dispatcher: Lease-holding dispatcher with the same execution horizon.
            target_timestamp_ns: Future target in the controller's trusted clock.

        Returns:
            The exact copied paired step accepted by the dispatcher.

        Raises:
            TypeError: If ``dispatcher`` has an invalid type.
            BimanualDispatchError: If inference, wire validation, timing, safety,
                observation snapshotting, controller application, or
                acknowledgement fails.
            BimanualBrokerFault: If this broker was already faulted.

        Once an operation begins with a valid dispatcher, any failure faults the
        broker. ``peek``, ``commit``, and ``dispatch_next`` then reject all work
        until ``reset`` succeeds and clears the local cache.
        """
        with self._lock:
            self._ensure_healthy()
            if not isinstance(dispatcher, BimanualCommandDispatcher):
                self._faulted = True
                raise TypeError(
                    "dispatcher: expected BimanualCommandDispatcher, "
                    f"got {type(dispatcher).__name__}"
                )
            if self._action_spec is None:
                self._faulted = True
                dispatcher.reject_without_dispatch(
                    ValueError(
                        "broker dispatch requires construction through from_metadata with a verified action spec"
                    )
                )
            if dispatcher.action_spec != self._action_spec:
                self._faulted = True
                dispatcher.reject_without_dispatch(
                    ValueError("dispatcher.action_spec conflicts with broker handshake metadata")
                )
            if dispatcher.session_id != self.session_id:
                self._faulted = True
                dispatcher.reject_without_dispatch(
                    ValueError("dispatcher.session_id conflicts with broker handshake metadata")
                )
            if dispatcher.execution_horizon != self._execution_horizon:
                self._faulted = True
                dispatcher.reject_without_dispatch(
                    ValueError(
                        "dispatcher.execution_horizon conflicts with broker metadata: "
                        f"expected {self._execution_horizon}, got {dispatcher.execution_horizon}"
                    )
                )
            try:
                result = self.peek(observation)
            except BaseException as error:
                self._faulted = True
                dispatcher.reject_without_dispatch(error)

            try:
                dispatcher.dispatch(
                    result,
                    target_timestamp_ns=target_timestamp_ns,
                )
            except BaseException:
                self._faulted = True
                raise

            try:
                self.commit()
            except BaseException as error:
                self._faulted = True
                dispatcher.reject_without_dispatch(error)
            return result

    def reset(self) -> None:
        """Reset policy state, then clear cache and fault state.

        The policy hook must succeed before the broker fault is cleared.

        Raises:
            TypeError: If the wrapped policy has no callable ``reset`` method.
            Exception: Propagates a policy reset failure while preserving the
                broker's cached action and fault state.
        """
        with self._lock:
            self._faulted = True
            reset = getattr(self._policy, "reset", None)
            if not callable(reset):
                raise TypeError("policy.reset: a callable reset is required for broker recovery")
            reset()
            self._cached_chunk = None
            self._step = 0
            self._awaiting_commit = False
            self._faulted = False

    def _build_step_result(self) -> dict[str, Any]:
        assert self._cached_chunk is not None
        result: dict[str, Any] = {
            "actions": {
                "left": self._cached_chunk.left_actions[self._step].copy(),
                "right": self._cached_chunk.right_actions[self._step].copy(),
            },
            SOURCE_TIMESTAMP_FIELD: self._cached_chunk.source_timestamp_ns,
            CLOCK_DOMAIN_FIELD: self._cached_chunk.clock_domain,
            CHUNK_SEQUENCE_ID_FIELD: self._cached_chunk.chunk_sequence_id,
            CHUNK_STEP_INDEX_FIELD: self._step,
        }
        if self._cached_chunk.session_id is not None:
            result[SESSION_ID_FIELD] = self._cached_chunk.session_id
        if self._step == 0:
            result.update(
                _snapshot_supported_mapping(
                    self._cached_chunk.timing,
                    field_name="cached timing",
                )
            )
        return result

    def _ensure_healthy(self) -> None:
        if self._faulted:
            raise BimanualBrokerFault("bimanual broker is faulted; reset is required before further dispatch")


class BimanualDispatchError(RuntimeError):
    """Raised after a command is rejected or the controller enters safe hold."""


class BimanualCommandDispatcher:
    """Validate semantics and timing before one paired controller side effect."""

    def __init__(
        self,
        controller: BimanualController,
        spec: BimanualActionSpec,
        limits: BimanualSafetyLimits,
        *,
        execution_horizon: int,
        session_id: str,
        clock_domain: str | None = None,
        max_target_lead_ms: float | None = None,
    ) -> None:
        """Bind a unique controller lease, safety limits, and time contract.

        Args:
            controller: Hardware adapter exposing a trusted clock and atomic
                single-owner dispatch lease.
            spec: Exact training, wire, and hardware action contract.
            limits: Closed float32 ``[31]`` bounds in the units of ``spec``.
            execution_horizon: Physical horizon obtained from server metadata.
            session_id: Exact server adapter session obtained from the same
                validated metadata handshake as ``execution_horizon``.
            clock_domain: Optional explicit clock name; it must equal ``spec``.
            max_target_lead_ms: Optional tighter controller-specific lead bound.

        Raises:
            TypeError: If controller, limits, or numeric values violate their
                declared interfaces.
            ValueError: If contracts conflict, the controller is faulted, or
                bounds are invalid.
            RuntimeError: If the controller backend rejects lease acquisition,
                including because another dispatcher already owns the lease.

        The acquired lease has no healthy-state release operation. Replacement
        requires safe hold, verified hardware recovery, and a new epoch.
        """
        validated_spec = _validated_spec_copy(spec)
        if not isinstance(limits, BimanualSafetyLimits):
            raise TypeError(f"limits: expected BimanualSafetyLimits, got {type(limits).__name__}")
        if limits.spec != validated_spec:
            raise ValueError("limits.spec conflicts with the dispatcher action spec")
        recovery_epoch = _validate_controller_ready(controller, spec=validated_spec)
        validated_execution_horizon = _validate_execution_horizon(execution_horizon, spec=validated_spec)
        validated_session_id = _validate_session_id(session_id, field_name="session_id")
        validated_clock_domain = (
            validated_spec.clock_domain
            if clock_domain is None
            else _validate_clock_domain(
                clock_domain,
                field_name="clock_domain",
            )
        )
        if validated_clock_domain != validated_spec.clock_domain:
            raise ValueError(
                "clock_domain: expected spec.clock_domain "
                f"{validated_spec.clock_domain!r}, got {validated_clock_domain!r}"
            )
        if max_target_lead_ms is None:
            max_target_lead_ms = validated_spec.max_command_lead_ms
        validated_max_target_lead_ms = _validate_positive_finite(
            max_target_lead_ms,
            field_name="max_target_lead_ms",
        )
        if validated_max_target_lead_ms > validated_spec.max_command_lead_ms:
            raise ValueError(
                "max_target_lead_ms: may not exceed "
                f"spec.max_command_lead_ms={validated_spec.max_command_lead_ms}, "
                f"got {validated_max_target_lead_ms}"
            )
        _validate_execution_horizon_feasibility(
            validated_execution_horizon,
            spec=validated_spec,
            max_target_lead_ms=validated_max_target_lead_ms,
        )

        left_min = _immutable_action_vector(limits.left_min, field_name="limits.left_min")
        left_max = _immutable_action_vector(limits.left_max, field_name="limits.left_max")
        right_min = _immutable_action_vector(limits.right_min, field_name="limits.right_min")
        right_max = _immutable_action_vector(limits.right_max, field_name="limits.right_max")
        self._controller = controller
        self._spec = validated_spec
        self._execution_horizon = validated_execution_horizon
        self._session_id = validated_session_id
        self._controller_recovery_epoch = recovery_epoch
        self._dispatch_lease: object | None = None
        self._clock_domain = validated_clock_domain
        self._max_target_lead_ms = validated_max_target_lead_ms
        self._left_min = left_min
        self._left_max = left_max
        self._right_min = right_min
        self._right_max = right_max
        self._last_source_timestamp_ns: int | None = None
        self._last_chunk_sequence_id: int | None = None
        self._last_chunk_step_index: int | None = None
        self._last_target_timestamp_ns: int | None = None
        self._last_controller_clock_ns: int | None = None
        self._faulted = False
        self._lock = threading.RLock()
        dispatch_lease = _acquire_controller_dispatch_lease(
            controller,
            expected_spec=validated_spec,
            expected_clock_domain=validated_clock_domain,
            expected_recovery_epoch=recovery_epoch,
        )
        object.__setattr__(self, "_dispatch_lease", dispatch_lease)

    @classmethod
    def from_metadata(
        cls,
        controller: BimanualController,
        metadata: object,
        spec: BimanualActionSpec,
        limits: BimanualSafetyLimits,
        *,
        expected_execution_horizon: int | None = None,
        clock_domain: str | None = None,
        max_target_lead_ms: float | None = None,
    ) -> BimanualCommandDispatcher:
        """Construct a dispatcher from one validated server-session handshake."""
        validated_spec = _validated_spec_copy(spec)
        metadata_snapshot = _snapshot_supported_mapping(metadata, field_name="metadata")
        execution_horizon = validate_deployment_metadata(
            metadata_snapshot,
            validated_spec,
            expected_execution_horizon=expected_execution_horizon,
        )
        session_id = _validate_session_id(
            metadata_snapshot["pi_dex"][SESSION_ID_FIELD],
            field_name=f"metadata['pi_dex'][{SESSION_ID_FIELD!r}]",
        )
        return cls(
            controller,
            validated_spec,
            limits,
            execution_horizon=execution_horizon,
            session_id=session_id,
            clock_domain=clock_domain,
            max_target_lead_ms=max_target_lead_ms,
        )

    @property
    def faulted(self) -> bool:
        """Whether this dispatcher is permanently latched after a rejection.

        Recovery deliberately requires a newly constructed dispatcher bound to
        a controller that has already completed its external safe-state recovery.
        """
        with self._lock:
            return self._faulted

    @property
    def execution_horizon(self) -> int:
        """Physical paired-step horizon validated at the server handshake."""
        return self._execution_horizon

    @property
    def action_spec(self) -> BimanualActionSpec:
        """Return an independent copy of the controller-bound action contract."""
        return dataclasses.replace(self._spec)

    @property
    def session_id(self) -> str:
        """Server adapter session bound by the deployment handshake."""
        return self._session_id

    def dispatch(
        self,
        result: Mapping[str, Any],
        *,
        target_timestamp_ns: int,
    ) -> None:
        """Validate and atomically hand one bimanual command to the controller.

        Args:
            result: Strict v2 paired logical-action step with source timestamp,
                clock, chunk identifier, and step index.
            target_timestamp_ns: Future target in the controller's trusted clock.

        Raises:
            BimanualDispatchError: If any wire, timing, sequence, lease, limit,
                clock, or controller check fails. The controller is asked to hold
                and this dispatcher is permanently faulted first.

        The caller must exclusively own the complete ``result`` tree until this
        synchronous method returns. This makes the sequential left/right NumPy
        copies one coherent command snapshot; the broker path already satisfies
        the rule.

        The dispatcher reads controller time after entering its lock and again
        immediately before apply, then reruns the complete freshness/lead check.
        The controller must atomically reject both expiry and clock rollback in
        ``apply_bimanual_action``.
        """
        with self._lock:
            self._ensure_healthy()
            try:
                _validate_controller_ready(
                    self._controller,
                    spec=self._spec,
                    expected_recovery_epoch=self._controller_recovery_epoch,
                )
                _validate_controller_dispatch_lease(
                    self._controller,
                    self._dispatch_lease,
                    expected_spec=self._spec,
                    expected_clock_domain=self._clock_domain,
                    expected_recovery_epoch=self._controller_recovery_epoch,
                )
                target_timestamp_ns = _validate_timestamp_ns(
                    target_timestamp_ns,
                    field_name="target_timestamp_ns",
                )
                (
                    left_action,
                    right_action,
                    source_timestamp_ns,
                    clock_domain,
                    chunk_sequence_id,
                    chunk_step_index,
                    session_id,
                ) = _extract_dispatch_payload(result)
                if clock_domain != self._clock_domain:
                    raise ValueError(
                        f"result clock_domain: expected {self._clock_domain!r}, got {clock_domain!r}"
                    )
                if session_id != self._session_id:
                    raise ValueError(
                        f"result session_id: expected {self._session_id!r}, got {session_id!r}"
                    )
                current_timestamp_ns = _read_controller_clock_ns(
                    self._controller,
                    self._dispatch_lease,
                    previous_timestamp_ns=self._last_controller_clock_ns,
                )
                _validate_dispatch_timing(
                    source_timestamp_ns=source_timestamp_ns,
                    current_timestamp_ns=current_timestamp_ns,
                    target_timestamp_ns=target_timestamp_ns,
                    max_observation_age_ms=self._spec.max_observation_age_ms,
                    max_target_lead_ms=self._max_target_lead_ms,
                )
                self._validate_dispatch_sequence(
                    source_timestamp_ns=source_timestamp_ns,
                    chunk_sequence_id=chunk_sequence_id,
                    chunk_step_index=chunk_step_index,
                    target_timestamp_ns=target_timestamp_ns,
                )
                _validate_limits(
                    left_action,
                    lower=self._left_min,
                    upper=self._left_max,
                    side="left",
                )
                _validate_limits(
                    right_action,
                    lower=self._right_min,
                    upper=self._right_max,
                    side="right",
                )
                _validate_controller_ready(
                    self._controller,
                    spec=self._spec,
                    expected_recovery_epoch=self._controller_recovery_epoch,
                )
                _validate_controller_dispatch_lease(
                    self._controller,
                    self._dispatch_lease,
                    expected_spec=self._spec,
                    expected_clock_domain=self._clock_domain,
                    expected_recovery_epoch=self._controller_recovery_epoch,
                )
                current_timestamp_ns = _read_controller_clock_ns(
                    self._controller,
                    self._dispatch_lease,
                    previous_timestamp_ns=current_timestamp_ns,
                )
                _validate_dispatch_timing(
                    source_timestamp_ns=source_timestamp_ns,
                    current_timestamp_ns=current_timestamp_ns,
                    target_timestamp_ns=target_timestamp_ns,
                    max_observation_age_ms=self._spec.max_observation_age_ms,
                    max_target_lead_ms=self._max_target_lead_ms,
                )
                not_after_timestamp_ns = _dispatch_not_after_timestamp_ns(
                    source_timestamp_ns=source_timestamp_ns,
                    target_timestamp_ns=target_timestamp_ns,
                    max_observation_age_ms=self._spec.max_observation_age_ms,
                )
                self._controller.apply_bimanual_action(
                    left_action,
                    right_action,
                    target_timestamp_ns=target_timestamp_ns,
                    not_before_timestamp_ns=current_timestamp_ns,
                    not_after_timestamp_ns=not_after_timestamp_ns,
                    clock_domain=self._clock_domain,
                    dispatch_lease=self._dispatch_lease,
                    expected_recovery_epoch=self._controller_recovery_epoch,
                )
                _validate_controller_ready(
                    self._controller,
                    spec=self._spec,
                    expected_recovery_epoch=self._controller_recovery_epoch,
                )
                _validate_controller_dispatch_lease(
                    self._controller,
                    self._dispatch_lease,
                    expected_spec=self._spec,
                    expected_clock_domain=self._clock_domain,
                    expected_recovery_epoch=self._controller_recovery_epoch,
                )
                current_timestamp_ns = _read_controller_clock_ns(
                    self._controller,
                    self._dispatch_lease,
                    previous_timestamp_ns=current_timestamp_ns,
                )
                self._last_source_timestamp_ns = source_timestamp_ns
                self._last_chunk_sequence_id = chunk_sequence_id
                self._last_chunk_step_index = chunk_step_index
                self._last_target_timestamp_ns = target_timestamp_ns
                self._last_controller_clock_ns = current_timestamp_ns
            except BaseException as error:
                self._hold_and_raise(error)

    def reject_without_dispatch(self, error: BaseException) -> NoReturn:
        """Request safe hold for a policy, wire, or broker failure.

        This is used when no valid action pair reached ``dispatch``. It ensures
        malformed inference output and acknowledgement failures enter the same
        fail-safe controller state as command validation failures.
        """
        with self._lock:
            self._ensure_healthy()
            if not isinstance(error, BaseException):
                error = TypeError(f"error: expected BaseException, got {type(error).__name__}")
            self._hold_and_raise(error)

    def _validate_dispatch_sequence(
        self,
        *,
        source_timestamp_ns: int,
        chunk_sequence_id: int,
        chunk_step_index: int,
        target_timestamp_ns: int,
    ) -> None:
        if chunk_step_index >= self._execution_horizon:
            raise ValueError(
                f"chunk_step_index: expected < execution_horizon {self._execution_horizon}, "
                f"got {chunk_step_index}"
            )
        if self._last_source_timestamp_ns is None:
            if chunk_step_index != 0:
                raise ValueError(f"chunk_step_index: first dispatched step must be 0, got {chunk_step_index}")
            return

        assert self._last_chunk_sequence_id is not None
        assert self._last_chunk_step_index is not None
        assert self._last_target_timestamp_ns is not None
        if target_timestamp_ns <= self._last_target_timestamp_ns:
            raise ValueError(
                "target_timestamp_ns: expected a strictly increasing dispatch target, "
                f"got previous={self._last_target_timestamp_ns} and current={target_timestamp_ns}"
            )

        target_period_ns = target_timestamp_ns - self._last_target_timestamp_ns
        expected_period_ns = 1_000_000_000 / self._spec.control_frequency_hz
        max_error_ns = self._spec.max_control_period_error_ms * 1_000_000
        if abs(target_period_ns - expected_period_ns) > max_error_ns:
            raise ValueError(
                "target control period: expected "
                f"{expected_period_ns / 1_000_000} ms +/- "
                f"{self._spec.max_control_period_error_ms} ms, got "
                f"{target_period_ns / 1_000_000} ms"
            )

        if chunk_sequence_id == self._last_chunk_sequence_id:
            expected_step_index = self._last_chunk_step_index + 1
            if chunk_step_index != expected_step_index:
                raise ValueError(
                    "chunk_step_index: expected the next cached-chunk step "
                    f"{expected_step_index}, got {chunk_step_index}"
                )
            if source_timestamp_ns != self._last_source_timestamp_ns:
                raise ValueError("source_timestamp_ns changed inside one decoded chunk")
            return

        expected_chunk_sequence_id = self._last_chunk_sequence_id + 1
        if chunk_sequence_id != expected_chunk_sequence_id:
            raise ValueError(
                "chunk_sequence_id: expected the next adapter chunk id "
                f"{expected_chunk_sequence_id}, got {chunk_sequence_id}"
            )
        if self._last_chunk_step_index != self._execution_horizon - 1:
            raise ValueError(
                "chunk_sequence_id: received a new chunk before the previous execution horizon completed"
            )
        if source_timestamp_ns < self._last_source_timestamp_ns:
            raise ValueError(
                "source_timestamp_ns: expected a nondecreasing chunk source, "
                f"got previous={self._last_source_timestamp_ns} and current={source_timestamp_ns}"
            )
        if chunk_step_index != 0:
            raise ValueError(f"chunk_step_index: a new chunk must start at 0, got {chunk_step_index}")

    def _ensure_healthy(self) -> None:
        if self._faulted:
            raise BimanualDispatchError(
                "bimanual dispatcher is faulted; complete external controller recovery "
                "and construct a new dispatcher"
            )

    def _hold_and_raise(self, error: BaseException) -> None:
        self._faulted = True
        stable_reason = "PI-DEX bimanual dispatch rejected"
        try:
            receipt = self._controller.hold(
                reason=stable_reason,
                dispatch_lease=self._dispatch_lease,
                expected_recovery_epoch=self._controller_recovery_epoch,
            )
        except BaseException as hold_error:
            raise BimanualDispatchError(
                f"{stable_reason}; safe hold also failed: {_format_exception_safely(hold_error)}"
            ) from hold_error
        reason = f"{stable_reason}: {_format_exception_safely(error)}"
        if not isinstance(receipt, BimanualHoldReceipt):
            raise BimanualDispatchError(
                f"{reason}; safe hold returned invalid acknowledgement {type(receipt).__name__}"
            ) from error
        if receipt.safety_faulted is not True or receipt.recovery_epoch != self._controller_recovery_epoch:
            raise BimanualDispatchError(
                f"{reason}; safe hold acknowledgement conflicts with the controller epoch"
            ) from error
        raise BimanualDispatchError(reason) from error


def _merge_metadata(
    policy: MetadataPolicyLike,
    *,
    spec: BimanualActionSpec,
    execution_horizon: int,
) -> dict[str, Any]:
    _validate_execution_horizon_feasibility(
        execution_horizon,
        spec=spec,
        max_target_lead_ms=spec.max_command_lead_ms,
    )
    policy_metadata = getattr(policy, "metadata", {})
    if callable(policy_metadata):
        policy_metadata = policy_metadata()
    if policy_metadata is None:
        policy_metadata = {}
    if not isinstance(policy_metadata, Mapping):
        raise TypeError(f"policy.metadata: expected a mapping, got {type(policy_metadata).__name__}")

    metadata = _snapshot_supported_mapping(policy_metadata, field_name="policy.metadata")
    expected_pi_dex = {
        **spec.to_metadata(),
        "wire_format": DEPLOYMENT_WIRE_FORMAT,
        "execution_horizon": execution_horizon,
        SESSION_ID_FIELD: uuid.uuid4().hex,
    }
    existing_pi_dex = metadata.get("pi_dex")
    if existing_pi_dex is None:
        raise ValueError(
            "policy.metadata: missing verified 'pi_dex' training contract; "
            "load the policy through PI-DEX checkpoint integration"
        )
    spec.validate_metadata(
        {"pi_dex": existing_pi_dex},
        allowed_extra_fields=_DEPLOYMENT_METADATA_FIELDS,
    )
    if not isinstance(existing_pi_dex, Mapping):
        raise TypeError("policy.metadata['pi_dex']: expected a mapping")
    if "execution_horizon" in existing_pi_dex:
        declared_execution_horizon = existing_pi_dex["execution_horizon"]
        if declared_execution_horizon is None:
            raise TypeError("policy.metadata['pi_dex']['execution_horizon']: expected int, got NoneType")
        existing_horizon = _validate_execution_horizon(declared_execution_horizon, spec=spec)
        if existing_horizon != execution_horizon:
            raise ValueError(
                "policy.metadata['pi_dex']['execution_horizon'] conflicts with the requested deployment horizon"
            )
    if "wire_format" in existing_pi_dex and existing_pi_dex["wire_format"] != DEPLOYMENT_WIRE_FORMAT:
        raise ValueError("policy.metadata['pi_dex']['wire_format'] conflicts with the PI-DEX wire contract")
    if SESSION_ID_FIELD in existing_pi_dex:
        raise ValueError("policy.metadata['pi_dex'] already belongs to a deployment session")
    metadata["pi_dex"] = expected_pi_dex
    return metadata


def _read_policy_metadata(policy: object) -> dict[str, Any]:
    """Snapshot metadata from the exact policy/session used for inference."""
    get_server_metadata = getattr(policy, "get_server_metadata", None)
    if callable(get_server_metadata):
        metadata = get_server_metadata()
    else:
        metadata = getattr(policy, "metadata", None)
        if callable(metadata):
            metadata = metadata()
    if not isinstance(metadata, Mapping):
        raise TypeError(
            "policy session metadata: expected policy.metadata or "
            "get_server_metadata() to return a mapping"
        )
    return _snapshot_supported_mapping(metadata, field_name="policy session metadata")


def _extract_transport_observation(observation: object) -> tuple[dict[str, Any], int, str]:
    if not isinstance(observation, Mapping):
        raise TypeError(f"observation: expected a mapping, got {type(observation).__name__}")
    policy_observation = _snapshot_observation_mapping(observation)
    try:
        source_timestamp_ns = policy_observation.pop(OBSERVATION_TIMESTAMP_FIELD)
    except KeyError:
        raise KeyError(f"observation: missing required field {OBSERVATION_TIMESTAMP_FIELD!r}") from None
    try:
        clock_domain = policy_observation.pop(CLOCK_DOMAIN_FIELD)
    except KeyError:
        raise KeyError(f"observation: missing required field {CLOCK_DOMAIN_FIELD!r}") from None
    return (
        policy_observation,
        _validate_timestamp_ns(source_timestamp_ns, field_name=OBSERVATION_TIMESTAMP_FIELD),
        _validate_clock_domain(clock_domain, field_name=CLOCK_DOMAIN_FIELD),
    )


def _snapshot_observation_mapping(observation: Mapping[Any, Any]) -> dict[str, Any]:
    """Copy the supported OpenPI observation tree without invoking copy hooks."""
    return _snapshot_supported_mapping(observation, field_name="observation")


def _snapshot_supported_mapping(
    mapping: Mapping[Any, Any],
    *,
    field_name: str,
) -> dict[str, Any]:
    """Copy a supported transport tree without invoking arbitrary copy hooks."""
    active_container_ids: set[int] = set()

    def snapshot(value: Any, *, path: str, depth: int) -> Any:
        if depth > 32:
            raise ValueError(f"{path}: transport nesting exceeds 32 levels")
        if value is None or type(value) in (bool, int, float, str, bytes):
            return value
        if isinstance(value, np.generic):
            if value.dtype.kind not in "buifSU":
                raise TypeError(f"{path}: unsupported numpy scalar dtype {value.dtype}")
            return value.copy()
        if isinstance(value, np.ndarray):
            canonical = np.array(value, copy=True, order="K", subok=False)
            if canonical.dtype.kind not in "buifSU":
                raise TypeError(f"{path}: unsupported numpy array dtype {canonical.dtype}")
            return canonical
        if isinstance(value, Mapping | list | tuple):
            container_id = id(value)
            if container_id in active_container_ids:
                raise ValueError(f"{path}: cyclic transport containers are not supported")
            active_container_ids.add(container_id)
            try:
                if isinstance(value, Mapping):
                    copied_mapping: dict[str, Any] = {}
                    for key, child in value.items():
                        if not isinstance(key, str):
                            raise TypeError(
                                f"{path}: expected string mapping keys, got {type(key).__name__}"
                            )
                        copied_mapping[key] = snapshot(
                            child,
                            path=f"{path}[{key!r}]",
                            depth=depth + 1,
                        )
                    return copied_mapping
                copied_items = [
                    snapshot(child, path=f"{path}[{index}]", depth=depth + 1)
                    for index, child in enumerate(value)
                ]
                return tuple(copied_items) if isinstance(value, tuple) else copied_items
            finally:
                active_container_ids.remove(container_id)
        raise TypeError(f"{path}: unsupported transport value type {type(value).__name__}")

    copied = snapshot(mapping, path=field_name, depth=0)
    assert isinstance(copied, dict)
    return copied


def _extract_inverse_normalized_chunk(
    result: Mapping[str, Any],
    *,
    spec: BimanualActionSpec,
) -> tuple[np.ndarray, np.ndarray]:
    if "actions" in result:
        raise ValueError(
            "policy result contains raw model-space 'actions'; expected inverse-normalized "
            "'left_actions' and 'right_actions'"
        )
    has_left_actions = "left_actions" in result
    has_right_actions = "right_actions" in result
    if has_left_actions != has_right_actions:
        missing_field = "right_actions" if has_left_actions else "left_actions"
        raise KeyError(f"policy result: missing required field {missing_field!r}")
    if not has_left_actions:
        raise KeyError("policy result: expected inverse-normalized 'left_actions' and 'right_actions'")

    left_actions = result["left_actions"]
    right_actions = result["right_actions"]
    expected_shape = (spec.physical_horizon, LOGICAL_ACTION_DIM)
    canonical_actions: list[np.ndarray] = []
    for field_name, value in (("left_actions", left_actions), ("right_actions", right_actions)):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"policy result {field_name!r}: expected numpy.ndarray, got {type(value).__name__}")
        canonical = np.array(value, copy=True, order="C", subok=False)
        if canonical.shape != expected_shape:
            raise ValueError(
                f"policy result {field_name!r}.shape: expected {expected_shape}, got {canonical.shape}"
            )
        if not np.issubdtype(canonical.dtype, np.floating):
            raise TypeError(f"policy result {field_name!r}.dtype: expected floating, got {canonical.dtype}")
        _require_finite(canonical, field_name=f"policy result {field_name!r}")
        canonical_actions.append(canonical)
    left_actions, right_actions = canonical_actions
    if left_actions.dtype != right_actions.dtype:
        raise TypeError(
            f"policy result left/right dtypes must match, got {left_actions.dtype} and {right_actions.dtype}"
        )
    return left_actions, right_actions


def _to_wire_float32(values: np.ndarray, *, field_name: str) -> np.ndarray:
    base_values = np.array(values, copy=True, order="C", subok=False)
    with np.errstate(over="ignore"):
        wire_values = base_values.astype(np.float32, copy=True)
    if not np.all(np.isfinite(wire_values)):
        raise ValueError(f"{field_name}: values overflow float32 wire representation")
    return wire_values


def _validate_decoded_chunk(
    result: object,
    *,
    execution_horizon: int,
    expected_session_id: str | None,
) -> _DecodedChunk:
    if not isinstance(result, Mapping):
        raise TypeError(f"policy result: expected a mapping, got {type(result).__name__}")
    result_snapshot = _snapshot_supported_mapping(result, field_name="policy result")
    unsupported_fields = set(result_snapshot) - _WIRE_RESULT_FIELDS
    if unsupported_fields:
        formatted_fields = sorted(repr(field) for field in unsupported_fields)
        raise ValueError(f"policy result: unsupported wire fields {formatted_fields}")

    actions = result_snapshot.get("actions")
    if not isinstance(actions, Mapping):
        raise TypeError("policy result 'actions': expected a mapping with left/right chunks")
    actions_snapshot = _snapshot_supported_mapping(actions, field_name="policy result 'actions'")
    unsupported_action_fields = set(actions_snapshot) - {"left", "right"}
    if unsupported_action_fields:
        formatted_fields = sorted(repr(field) for field in unsupported_action_fields)
        raise ValueError(f"policy result 'actions': unsupported fields {formatted_fields}")
    try:
        raw_left_actions = actions_snapshot["left"]
        raw_right_actions = actions_snapshot["right"]
    except KeyError as error:
        raise KeyError(f"policy result 'actions': missing required field {error.args[0]!r}") from None

    validated_actions: list[np.ndarray] = []
    for field_name, value in (("actions.left", raw_left_actions), ("actions.right", raw_right_actions)):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{field_name}: expected numpy.ndarray, got {type(value).__name__}")
        value_snapshot = np.array(value, copy=True, order="C", subok=False)
        if value_snapshot.shape != (execution_horizon, LOGICAL_ACTION_DIM):
            raise ValueError(
                f"{field_name}.shape: expected {(execution_horizon, LOGICAL_ACTION_DIM)}, "
                f"got {value_snapshot.shape}"
            )
        if value_snapshot.dtype != np.float32:
            raise TypeError(f"{field_name}.dtype: expected float32, got {value_snapshot.dtype}")
        _require_finite(value_snapshot, field_name=field_name)
        validated_actions.append(value_snapshot)
    left_actions, right_actions = validated_actions

    if SOURCE_TIMESTAMP_FIELD not in result_snapshot:
        raise KeyError(f"policy result: missing required field {SOURCE_TIMESTAMP_FIELD!r}")
    if CLOCK_DOMAIN_FIELD not in result_snapshot:
        raise KeyError(f"policy result: missing required field {CLOCK_DOMAIN_FIELD!r}")
    if CHUNK_SEQUENCE_ID_FIELD not in result_snapshot:
        raise KeyError(f"policy result: missing required field {CHUNK_SEQUENCE_ID_FIELD!r}")
    if expected_session_id is not None and SESSION_ID_FIELD not in result_snapshot:
        raise KeyError(f"policy result: missing required field {SESSION_ID_FIELD!r}")
    source_timestamp_ns = _validate_timestamp_ns(
        result_snapshot[SOURCE_TIMESTAMP_FIELD],
        field_name=SOURCE_TIMESTAMP_FIELD,
    )
    clock_domain = _validate_clock_domain(
        result_snapshot[CLOCK_DOMAIN_FIELD],
        field_name=CLOCK_DOMAIN_FIELD,
    )
    chunk_sequence_id = _validate_positive_int(
        result_snapshot[CHUNK_SEQUENCE_ID_FIELD],
        field_name=CHUNK_SEQUENCE_ID_FIELD,
    )
    session_id: str | None = None
    if SESSION_ID_FIELD in result_snapshot:
        session_id = _validate_session_id(
            result_snapshot[SESSION_ID_FIELD],
            field_name=SESSION_ID_FIELD,
        )
        if expected_session_id is not None and session_id != expected_session_id:
            raise ValueError(
                f"policy result session_id: expected {expected_session_id!r}, got {session_id!r}"
            )

    timing: dict[str, dict[str, Any]] = {}
    for field_name in _PASSTHROUGH_RESULT_FIELDS:
        if field_name in result_snapshot:
            value = result_snapshot[field_name]
            if not isinstance(value, Mapping):
                raise TypeError(f"policy result {field_name!r}: expected a mapping, got {type(value).__name__}")
            timing[field_name] = _snapshot_supported_mapping(
                value,
                field_name=f"policy result {field_name!r}",
            )
    return _DecodedChunk(
        left_actions=left_actions,
        right_actions=right_actions,
        source_timestamp_ns=source_timestamp_ns,
        clock_domain=clock_domain,
        chunk_sequence_id=chunk_sequence_id,
        session_id=session_id,
        timing=timing,
    )


def _extract_dispatch_payload(
    result: object,
) -> tuple[np.ndarray, np.ndarray, int, str, int, int, str]:
    if not isinstance(result, Mapping):
        raise TypeError(f"result: expected a mapping, got {type(result).__name__}")
    result_snapshot = _snapshot_supported_mapping(result, field_name="result")
    unsupported_fields = set(result_snapshot) - _DISPATCH_RESULT_FIELDS
    if unsupported_fields:
        formatted_fields = sorted(repr(field) for field in unsupported_fields)
        raise ValueError(f"result: unsupported dispatch fields {formatted_fields}")
    actions = result_snapshot.get("actions")
    if not isinstance(actions, Mapping):
        raise TypeError("result['actions']: expected a mapping with left/right actions")
    actions_snapshot = _snapshot_supported_mapping(actions, field_name="result['actions']")
    unsupported_action_fields = set(actions_snapshot) - {"left", "right"}
    if unsupported_action_fields:
        formatted_fields = sorted(repr(field) for field in unsupported_action_fields)
        raise ValueError(f"result['actions']: unsupported fields {formatted_fields}")
    try:
        left_action = _validate_action_vector(actions_snapshot["left"], field_name="actions.left")
        right_action = _validate_action_vector(actions_snapshot["right"], field_name="actions.right")
    except KeyError as error:
        raise KeyError(f"result['actions']: missing required field {error.args[0]!r}") from None
    if SOURCE_TIMESTAMP_FIELD not in result_snapshot:
        raise KeyError(f"result: missing required field {SOURCE_TIMESTAMP_FIELD!r}")
    if CLOCK_DOMAIN_FIELD not in result_snapshot:
        raise KeyError(f"result: missing required field {CLOCK_DOMAIN_FIELD!r}")
    if CHUNK_STEP_INDEX_FIELD not in result_snapshot:
        raise KeyError(f"result: missing required field {CHUNK_STEP_INDEX_FIELD!r}")
    if CHUNK_SEQUENCE_ID_FIELD not in result_snapshot:
        raise KeyError(f"result: missing required field {CHUNK_SEQUENCE_ID_FIELD!r}")
    if SESSION_ID_FIELD not in result_snapshot:
        raise KeyError(f"result: missing required field {SESSION_ID_FIELD!r}")
    source_timestamp_ns = _validate_timestamp_ns(
        result_snapshot[SOURCE_TIMESTAMP_FIELD],
        field_name=SOURCE_TIMESTAMP_FIELD,
    )
    clock_domain = _validate_clock_domain(
        result_snapshot[CLOCK_DOMAIN_FIELD],
        field_name=CLOCK_DOMAIN_FIELD,
    )
    chunk_sequence_id = _validate_positive_int(
        result_snapshot[CHUNK_SEQUENCE_ID_FIELD],
        field_name=CHUNK_SEQUENCE_ID_FIELD,
    )
    chunk_step_index = _validate_nonnegative_int(
        result_snapshot[CHUNK_STEP_INDEX_FIELD],
        field_name=CHUNK_STEP_INDEX_FIELD,
    )
    session_id = _validate_session_id(
        result_snapshot[SESSION_ID_FIELD],
        field_name=SESSION_ID_FIELD,
    )
    _validate_rotation_6d(left_action, field_name="actions.left")
    _validate_rotation_6d(right_action, field_name="actions.right")
    return (
        left_action,
        right_action,
        source_timestamp_ns,
        clock_domain,
        chunk_sequence_id,
        chunk_step_index,
        session_id,
    )


def _validate_dispatch_timing(
    *,
    source_timestamp_ns: int,
    current_timestamp_ns: int,
    target_timestamp_ns: int,
    max_observation_age_ms: float,
    max_target_lead_ms: float,
) -> None:
    if source_timestamp_ns > current_timestamp_ns:
        raise ValueError(
            "source_timestamp_ns: expected an observation no newer than current_timestamp_ns, "
            f"got source={source_timestamp_ns} and current={current_timestamp_ns}"
        )
    observation_age_ns = current_timestamp_ns - source_timestamp_ns
    if observation_age_ns > max_observation_age_ms * 1_000_000:
        raise ValueError(
            f"observation age: expected <= {max_observation_age_ms} ms, got {observation_age_ns / 1_000_000} ms"
        )
    if target_timestamp_ns <= current_timestamp_ns:
        raise ValueError(
            "target_timestamp_ns: expected a future target after current_timestamp_ns, "
            f"got target={target_timestamp_ns} and current={current_timestamp_ns}"
        )
    target_lead_ns = target_timestamp_ns - current_timestamp_ns
    if target_lead_ns > max_target_lead_ms * 1_000_000:
        raise ValueError(
            f"target lead: expected <= {max_target_lead_ms} ms, got {target_lead_ns / 1_000_000} ms"
        )


def _dispatch_not_after_timestamp_ns(
    *,
    source_timestamp_ns: int,
    target_timestamp_ns: int,
    max_observation_age_ms: float,
) -> int:
    maximum_age_ns = math.floor(max_observation_age_ms * 1_000_000)
    return min(source_timestamp_ns + maximum_age_ns, target_timestamp_ns - 1)


def _validate_action_vector(value: object, *, field_name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{field_name}: expected numpy.ndarray, got {type(value).__name__}")
    value_snapshot = np.array(value, copy=True, order="C", subok=False)
    if value_snapshot.shape != (LOGICAL_ACTION_DIM,):
        raise ValueError(
            f"{field_name}.shape: expected {(LOGICAL_ACTION_DIM,)}, got {value_snapshot.shape}"
        )
    if value_snapshot.dtype != np.float32:
        raise TypeError(f"{field_name}.dtype: expected float32, got {value_snapshot.dtype}")
    _require_finite(value_snapshot, field_name=field_name)
    return value_snapshot


def _immutable_action_vector(value: object, *, field_name: str) -> np.ndarray:
    validated = _validate_action_vector(value, field_name=field_name)
    return _action_vector_from_bytes(validated.tobytes(order="C"))


def _action_vector_from_bytes(values: bytes) -> np.ndarray:
    return np.frombuffer(values, dtype=np.float32, count=LOGICAL_ACTION_DIM)


def _validate_limits(action: np.ndarray, *, lower: np.ndarray, upper: np.ndarray, side: str) -> None:
    violating_dimensions = np.flatnonzero((action < lower) | (action > upper))
    if violating_dimensions.size:
        dimensions = violating_dimensions.tolist()
        raise ValueError(f"{side} action exceeds configured safety limits at dimensions {dimensions}")


def _validate_rotation_6d(action: np.ndarray, *, field_name: str) -> None:
    """Reject 6D rotations for which Gram-Schmidt conversion is ill-defined."""
    first_axis = action[_ROTATION_6D_START : _ROTATION_6D_START + 3].astype(np.float64)
    second_axis = action[_ROTATION_6D_START + 3 : _ROTATION_6D_STOP].astype(np.float64)
    first_norm = float(np.linalg.norm(first_axis))
    second_norm = float(np.linalg.norm(second_axis))
    if first_norm < _MIN_ROTATION_VECTOR_NORM or second_norm < _MIN_ROTATION_VECTOR_NORM:
        raise ValueError(f"{field_name} rotation_6d: both basis vectors must be nonzero")
    cosine = abs(float(np.dot(first_axis, second_axis)) / (first_norm * second_norm))
    if cosine > _MAX_ROTATION_VECTOR_COSINE:
        raise ValueError(f"{field_name} rotation_6d: basis vectors are degenerate or nearly parallel")


def _require_finite(values: np.ndarray, *, field_name: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{field_name}: expected all 31 semantic values to be finite")


def _validate_execution_horizon(execution_horizon: object, *, spec: BimanualActionSpec) -> int:
    if execution_horizon is None:
        return spec.physical_horizon
    value = _validate_positive_int(execution_horizon, field_name="execution_horizon")
    if value > spec.physical_horizon:
        raise ValueError(f"execution_horizon: expected <= {spec.physical_horizon}, got {value}")
    return value


def _validate_execution_horizon_feasibility(
    execution_horizon: int,
    *,
    spec: BimanualActionSpec,
    max_target_lead_ms: float,
) -> None:
    nominal_period_ns = 1_000_000_000 / spec.control_frequency_hz
    period_tolerance_ns = spec.max_control_period_error_ms * 1_000_000
    minimum_period_ns = max(
        1,
        math.ceil(nominal_period_ns - period_tolerance_ns),
    )
    maximum_period_ns = math.floor(nominal_period_ns + period_tolerance_ns)
    if execution_horizon > 1 and minimum_period_ns > maximum_period_ns:
        raise ValueError(
            "execution_horizon: no positive integer-nanosecond control period satisfies "
            "the frequency and period-tolerance contract"
        )
    required_span_ns = 1 + (execution_horizon - 1) * minimum_period_ns
    lead_budget_ns = math.floor(max_target_lead_ms * 1_000_000)
    if lead_budget_ns < 1:
        raise ValueError("max_target_lead_ms: must allow at least one future nanosecond")
    available_span_ns = (
        math.floor(spec.max_observation_age_ms * 1_000_000) + lead_budget_ns
    )
    if required_span_ns > available_span_ns:
        raise ValueError(
            "execution_horizon: cannot complete one chunk within the observation-age, "
            "target-lead, frequency, and period-tolerance contract; "
            f"minimum span is {required_span_ns} ns, available span is {available_span_ns} ns"
        )


def _validate_positive_int(value: object, *, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name}: expected int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{field_name}: expected a positive integer, got {value}")
    return value


def _validate_nonnegative_int(value: object, *, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name}: expected int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{field_name}: expected a nonnegative integer, got {value}")
    return value


def _validate_timestamp_ns(value: object, *, field_name: str) -> int:
    return _validate_positive_int(value, field_name=field_name)


def _validate_clock_domain(value: object, *, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name}: expected str, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{field_name}: expected a non-empty value")
    return value


def _validate_session_id(value: object, *, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name}: expected str, got {type(value).__name__}")
    if len(value) != 32 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name}: expected 32 lowercase hexadecimal digits")
    return value


def _validate_positive_finite(value: object, *, field_name: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{field_name}: expected a real number, got {type(value).__name__}")
    try:
        converted = float(value)
    except OverflowError:
        raise ValueError(f"{field_name}: expected a positive finite value") from None
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{field_name}: expected a positive finite value, got {value}")
    return converted


def _validate_controller_ready(
    controller: object,
    *,
    spec: BimanualActionSpec,
    expected_recovery_epoch: int | None = None,
) -> int:
    controller_spec = getattr(controller, "action_spec", None)
    if not isinstance(controller_spec, BimanualActionSpec):
        raise TypeError("controller.action_spec: expected BimanualActionSpec")
    if controller_spec != spec:
        raise ValueError("controller.action_spec conflicts with the dispatcher action spec")

    safety_faulted = getattr(controller, "safety_faulted", None)
    if not isinstance(safety_faulted, bool):
        raise TypeError("controller.safety_faulted: expected bool")
    if safety_faulted:
        raise ValueError("controller.safety_faulted: verified external recovery is required")

    recovery_epoch = _validate_nonnegative_int(
        getattr(controller, "recovery_epoch", None),
        field_name="controller.recovery_epoch",
    )
    if expected_recovery_epoch is not None and recovery_epoch != expected_recovery_epoch:
        raise ValueError(
            "controller.recovery_epoch changed after dispatcher construction; "
            "construct a new dispatcher from the recovered controller"
        )
    return recovery_epoch


def _acquire_controller_dispatch_lease(
    controller: object,
    *,
    expected_spec: BimanualActionSpec,
    expected_clock_domain: str,
    expected_recovery_epoch: int,
) -> object:
    acquire = getattr(controller, "acquire_dispatch_lease", None)
    if not callable(acquire):
        raise TypeError("controller.acquire_dispatch_lease: expected a callable")
    dispatch_lease = acquire(
        expected_spec=expected_spec,
        expected_clock_domain=expected_clock_domain,
        expected_recovery_epoch=expected_recovery_epoch,
    )
    if dispatch_lease is None:
        raise TypeError("controller.acquire_dispatch_lease: expected an opaque non-None lease")
    return dispatch_lease


def _validate_controller_dispatch_lease(
    controller: object,
    dispatch_lease: object,
    *,
    expected_spec: BimanualActionSpec,
    expected_clock_domain: str,
    expected_recovery_epoch: int,
) -> None:
    validate = getattr(controller, "validate_dispatch_lease", None)
    if not callable(validate):
        raise TypeError("controller.validate_dispatch_lease: expected a callable")
    validate(
        dispatch_lease,
        expected_spec=expected_spec,
        expected_clock_domain=expected_clock_domain,
        expected_recovery_epoch=expected_recovery_epoch,
    )


def _read_controller_clock_ns(
    controller: object,
    dispatch_lease: object,
    *,
    previous_timestamp_ns: int | None,
) -> int:
    read_clock = getattr(controller, "read_clock_ns", None)
    if not callable(read_clock):
        raise TypeError("controller.read_clock_ns: expected a callable")
    current_timestamp_ns = _validate_timestamp_ns(
        read_clock(dispatch_lease=dispatch_lease),
        field_name="controller.read_clock_ns result",
    )
    if previous_timestamp_ns is not None and current_timestamp_ns < previous_timestamp_ns:
        raise ValueError(
            "controller clock moved backwards: "
            f"previous={previous_timestamp_ns}, current={current_timestamp_ns}"
        )
    return current_timestamp_ns


def _validated_spec_copy(spec: object) -> BimanualActionSpec:
    if not isinstance(spec, BimanualActionSpec):
        raise TypeError(f"spec: expected BimanualActionSpec, got {type(spec).__name__}")
    return dataclasses.replace(spec)


def _format_exception_safely(error: BaseException) -> str:
    try:
        error_type = type(error)
        type_name = f"{error_type.__module__}.{error_type.__qualname__}"
        if error_type is KeyboardInterrupt:
            return type_name
        if error_type not in (Exception, TypeError, ValueError, RuntimeError, KeyError, OSError):
            return f"<{type_name} could not be formatted>"
        return f"{type_name}: {error}"
    except BaseException:
        return "<exception could not be formatted>"


def _transport_values_equal(left: object, right: object) -> bool:
    """Compare two snapshotted transport trees without ndarray truth coercion."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            return False
        if left.dtype != right.dtype or left.shape != right.shape:
            return False
        return bool(
            np.array_equal(left, right, equal_nan=True)
            if left.dtype.kind == "f"
            else np.array_equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if set(left) != set(right):
            return False
        return all(_transport_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, list | tuple) or isinstance(right, list | tuple):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(_transport_values_equal(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, np.generic) or isinstance(right, np.generic):
        if not isinstance(left, np.generic) or not isinstance(right, np.generic):
            return False
        if left.dtype != right.dtype:
            return False
        return bool(
            np.array_equal(left, right, equal_nan=True)
            if left.dtype.kind == "f"
            else np.array_equal(left, right)
        )
    if type(left) is float and type(right) is float and math.isnan(left) and math.isnan(right):
        return True
    return type(left) is type(right) and left == right
