"""Vectorized joint_29d normalization-stat extraction.

``compute-norm-stats`` used to build a full ``build_sample_index`` (per-window
HDF5 copies plus cadence probes) and then iterate ``SharpaJoint29dDataset``,
which also JPEG-decodes three cameras. Stats only need ``state`` and the two
``[K,29]`` action chunks, so this module:

* opens each HDF5 once and loads action/state arrays once (no images)
* vectorizes valid-start selection (horizon, cadence, skew, alignment)
* gathers all windows of one episode as ``[S,K,29]`` / ``[S,D]``
* fans episodes across CPU processes; the parent updates one OpenPI
  ``RunningStats`` (histograms are not merged across processes)

Window semantics match training: overlapping length-``K`` windows, each
physical step counted as one vector (OpenPI reshapes ``-1, last_dim``).
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import os
from typing import Any

import numpy as np

from pi_dex.actions import ActionRepresentation
from pi_dex.observation_contract import SharpaObservationContract
from pi_dex.sharpa_data import CANONICAL_ALIGNED_TIME_FIELD
from pi_dex.sharpa_data import AlignedTimeline
from pi_dex.sharpa_data import CommandedJointGroup
from pi_dex.sharpa_data import EpisodeActionProvenance
from pi_dex.sharpa_data import _validate_absolute_action_contract
from pi_dex.sharpa_data import _validate_provenance
from pi_dex.sharpa_dataset import EpisodeRef
from pi_dex.sharpa_dataset import _command_group
from pi_dex.spec import ActionTimebase
from pi_dex.spec import BimanualActionSpec
from pi_dex.spec import HandNormalization

_ACTION_FIELDS = (
    "action/left_arm/joint_angle",
    "action/left_hand/joint_angle",
    "action/right_arm/joint_angle",
    "action/right_hand/joint_angle",
)
_DEFAULT_WORKER_CAP = 64
_PROGRESS_EVERY = 50

_WORKER_STATE: dict[str, Any] = {}


@dataclasses.dataclass(frozen=True)
class EpisodeNormBatch:
    """All valid joint_29d windows from one episode, without images."""

    episode_id: str
    state: np.ndarray
    left_actions: np.ndarray
    right_actions: np.ndarray

    def __post_init__(self) -> None:
        if self.state.ndim != 2:
            raise ValueError(f"state: expected [S,D], got {self.state.shape}")
        if self.left_actions.ndim != 3 or self.right_actions.ndim != 3:
            raise ValueError("actions: expected [S,K,29] left and right")
        if self.state.shape[0] != self.left_actions.shape[0] or self.state.shape[0] != self.right_actions.shape[0]:
            raise ValueError("episode batch: state/action window counts must match")


def resolve_norm_workers(requested: int | None) -> int:
    """Return process count: CLI, else ``NORM_WORKERS``, else ``min(cpu, 64)``."""
    if requested is not None:
        if requested < 1:
            raise ValueError(f"norm workers: expected >= 1, got {requested}")
        return requested
    env = os.environ.get("NORM_WORKERS", "").strip()
    if env:
        try:
            value = int(env)
        except ValueError as error:
            raise ValueError(f"NORM_WORKERS: expected an integer, got {env!r}") from error
        if value < 1:
            raise ValueError(f"NORM_WORKERS: expected >= 1, got {value}")
        return value
    cpu = os.cpu_count() or 1
    return max(1, min(int(cpu), _DEFAULT_WORKER_CAP))


def valid_aligned_starts(
    *,
    timeline: AlignedTimeline,
    groups: tuple[CommandedJointGroup, CommandedJointGroup, CommandedJointGroup, CommandedJointGroup],
    spec: BimanualActionSpec,
    sub_state_ok: np.ndarray | None = None,
) -> np.ndarray:
    """Return sorted aligned-frame starts that pass the training cadence contract.

    Matches ``build_sample_index`` with ``provenance``: horizon fit, per-group
    raw (or canonical) cadence, cross-group skew, and canonical alignment.
    """
    if spec.action_representation is not ActionRepresentation.JOINT_29D:
        raise ValueError("valid_aligned_starts: only joint_29d is supported")
    k = spec.physical_horizon
    aligned_length = timeline.time.shape[0]
    for group in groups:
        if group.aligned_index.shape[0] != aligned_length:
            raise ValueError(
                f"{group.field_name.rsplit('/', 1)[0]}/aligned_index.shape[0]: "
                f"expected canonical N={aligned_length}, got {group.aligned_index.shape[0]}"
            )
    mask = np.ones(aligned_length, dtype=bool)
    if sub_state_ok is not None:
        if sub_state_ok.shape != (aligned_length,):
            raise ValueError(
                f"mode/sub_state length {sub_state_ok.shape[0]} != aligned N={aligned_length}"
            )
        mask &= sub_state_ok
    if spec.timebase is ActionTimebase.ALIGNED_30_HZ:
        if k > aligned_length:
            return np.zeros(0, dtype=np.int64)
        mask[aligned_length - k + 1 :] = False
    elif spec.timebase is ActionTimebase.RAW_CONTROL_60_HZ:
        for group in groups:
            first_raw = group.aligned_index.astype(np.int64, copy=False)
            mask &= (first_raw + k) <= group.joint_angles.shape[0]
    else:
        raise ValueError(f"spec.timebase: unsupported value {spec.timebase!r}")

    starts = np.flatnonzero(mask).astype(np.int64, copy=False)
    if starts.size == 0:
        return starts

    timestamps = np.stack(
        tuple(_window_timestamps(group, starts, spec) for group in groups),
        axis=0,
    )
    ok = np.ones(starts.shape[0], dtype=bool)
    expected_period_s = 1.0 / spec.control_frequency_hz
    if spec.timebase is ActionTimebase.RAW_CONTROL_60_HZ:
        if k >= 2:
            period_error_ms = np.abs(np.diff(timestamps, axis=-1) - expected_period_s) * 1_000.0
            ok &= period_error_ms.max(axis=(0, 2)) <= spec.max_control_period_error_ms
        canonical = timeline.time[starts, 0]
        alignment_error_ms = np.abs(timestamps[:, :, 0] - canonical[np.newaxis, :]) * 1_000.0
        ok &= alignment_error_ms.max(axis=0) <= spec.max_alignment_timestamp_error_ms
    else:
        frame_index = starts[:, np.newaxis] + np.arange(k, dtype=np.int64)
        canonical = timeline.time[frame_index, 0]
        if k >= 2:
            period_error_ms = np.abs(np.diff(canonical, axis=1) - expected_period_s) * 1_000.0
            ok &= period_error_ms.max(axis=1) <= spec.max_control_period_error_ms
        alignment_error_ms = np.abs(timestamps - canonical[np.newaxis, :, :]) * 1_000.0
        ok &= alignment_error_ms.max(axis=(0, 2)) <= spec.max_alignment_timestamp_error_ms

    timestamp_spread_ms = np.ptp(timestamps, axis=0) * 1_000.0
    ok &= timestamp_spread_ms.max(axis=1) <= spec.max_group_timestamp_skew_ms
    return starts[ok]


def extract_episode_norm_batch(
    episode: EpisodeRef,
    spec: BimanualActionSpec,
    contract: SharpaObservationContract,
    *,
    stride: int = 1,
) -> EpisodeNormBatch | None:
    """Load one episode's valid windows. Returns ``None`` when none survive."""
    loaded = _load_episode_windows(episode, spec, contract, stride=stride)
    if loaded is None:
        return None
    _starts, state, left_actions, right_actions = loaded
    del _starts
    return EpisodeNormBatch(
        episode_id=episode.episode_id,
        state=state,
        left_actions=left_actions,
        right_actions=right_actions,
    )


def episode_valid_start_frames(
    episode: EpisodeRef,
    spec: BimanualActionSpec,
    contract: SharpaObservationContract,
) -> np.ndarray:
    """Aligned starts that pass cadence plus finite state/action checks."""
    loaded = _load_episode_windows(episode, spec, contract, stride=1)
    if loaded is None:
        return np.zeros(0, dtype=np.int64)
    starts, _state, _left, _right = loaded
    return starts


def _load_episode_windows(
    episode: EpisodeRef,
    spec: BimanualActionSpec,
    contract: SharpaObservationContract,
    *,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if stride < 1:
        raise ValueError(f"stride: expected >= 1, got {stride}")
    if spec.action_representation is not ActionRepresentation.JOINT_29D:
        raise ValueError("_load_episode_windows: only joint_29d is supported")
    if contract.action_representation is not ActionRepresentation.JOINT_29D:
        raise ValueError("observation contract action_representation must be joint_29d")

    import h5py

    with h5py.File(episode.hdf5_path, "r") as handle:
        timeline = AlignedTimeline(
            CANONICAL_ALIGNED_TIME_FIELD,
            np.asarray(handle[CANONICAL_ALIGNED_TIME_FIELD], dtype=np.float64),
        )
        groups = (
            _command_group(handle, _ACTION_FIELDS[0], spec.left_arm_joint_order),
            _command_group(handle, _ACTION_FIELDS[1], spec.left_hand_joint_order),
            _command_group(handle, _ACTION_FIELDS[2], spec.right_arm_joint_order),
            _command_group(handle, _ACTION_FIELDS[3], spec.right_hand_joint_order),
        )
        sub_state_ok = None
        aligned_length = timeline.time.shape[0]
        if "mode/sub_state" in handle:
            sub_state = np.asarray(handle["mode/sub_state"], dtype=np.int16)
            if sub_state.shape[0] != aligned_length:
                raise ValueError(
                    f"{episode.episode_id}: mode/sub_state length {sub_state.shape[0]} "
                    f"!= aligned N={aligned_length}"
                )
            sub_state_ok = sub_state == 1
        starts = valid_aligned_starts(
            timeline=timeline,
            groups=groups,
            spec=spec,
            sub_state_ok=sub_state_ok,
        )
        if stride > 1:
            starts = starts[::stride]
        if starts.size == 0:
            return None
        state_sources = _load_state_sources(handle, contract, aligned_length=aligned_length)
        state = _gather_state(state_sources, starts, state_dim=contract.state_dim)
        left_actions, right_actions = _gather_joint_actions(groups, starts, spec)

    finite = (
        np.isfinite(state).all(axis=1)
        & np.isfinite(left_actions).all(axis=(1, 2))
        & np.isfinite(right_actions).all(axis=(1, 2))
    )
    if not bool(finite.all()):
        starts = starts[finite]
        state = state[finite]
        left_actions = left_actions[finite]
        right_actions = right_actions[finite]
    if starts.shape[0] == 0:
        return None
    return starts, state, left_actions, right_actions


def compute_joint29d_normalization_stats(
    episodes: Sequence[EpisodeRef],
    *,
    spec: BimanualActionSpec,
    contract: SharpaObservationContract,
    provenance: EpisodeActionProvenance,
    max_samples: int | None = None,
    workers: int = 1,
    stride: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute OpenPI quantile stats from episode HDF5 files without images.

    Parent process owns ``RunningStats``. Workers only return per-episode arrays.
    Episode order is preserved (``imap``) so histogram binning is deterministic
    given the same episode list.
    """
    if spec.action_representation is not ActionRepresentation.JOINT_29D:
        raise ValueError("compute_joint29d_normalization_stats: only joint_29d is supported")
    contract.validate_against_action_spec(spec)
    _validate_absolute_action_contract(spec)
    _validate_provenance(provenance, spec=spec)
    if not episodes:
        raise ValueError("compute-norm-stats: expected a non-empty episode list")
    if workers < 1:
        raise ValueError(f"workers: expected >= 1, got {workers}")
    if stride < 1:
        raise ValueError(f"stride: expected >= 1, got {stride}")
    if max_samples is not None:
        if isinstance(max_samples, bool) or not isinstance(max_samples, int):
            raise TypeError(f"max_samples: expected int or None, got {type(max_samples).__name__}")
        if max_samples <= 0:
            raise ValueError(f"max_samples: expected a positive integer, got {max_samples}")
    workers = min(workers, max(1, len(episodes)))

    from openpi.shared import normalize

    from pi_dex.normalization import validate_normalization_stats

    state_stats = normalize.RunningStats()
    if spec.hand_normalization is HandNormalization.PER_HAND:
        left_stats = normalize.RunningStats()
        right_stats = normalize.RunningStats()
    else:
        left_stats = right_stats = normalize.RunningStats()

    sample_count = 0
    episodes_used = 0
    episodes_skipped = 0
    total = len(episodes)
    print(
        f"compute-norm-stats: vectorized workers={workers} stride={stride} "
        f"episodes={total} max_samples={max_samples}",
        flush=True,
    )

    for done, batch in enumerate(_iter_episode_batches(episodes, spec, contract, workers=workers, stride=stride), start=1):
        if batch is None:
            episodes_skipped += 1
        else:
            state = batch.state
            left_actions = batch.left_actions
            right_actions = batch.right_actions
            window_count = int(state.shape[0])
            if max_samples is not None:
                remaining = max_samples - sample_count
                if remaining <= 0:
                    break
                if window_count > remaining:
                    state = state[:remaining]
                    left_actions = left_actions[:remaining]
                    right_actions = right_actions[:remaining]
                    window_count = remaining
            state_stats.update(state)
            left_stats.update(left_actions)
            right_stats.update(right_actions)
            sample_count += window_count
            episodes_used += 1
        if done == 1 or done % _PROGRESS_EVERY == 0 or done == total:
            print(
                f"compute-norm-stats: episodes {done}/{total} "
                f"windows={sample_count} used={episodes_used} skipped={episodes_skipped}",
                flush=True,
            )
        if max_samples is not None and sample_count >= max_samples:
            break

    if sample_count <= 0:
        raise ValueError(
            f"compute-norm-stats: no valid start frames (episodes={total}, horizon={spec.physical_horizon})"
        )

    stats = {
        "state": state_stats.get_statistics(),
        "left_actions": left_stats.get_statistics(),
        "right_actions": right_stats.get_statistics(),
    }
    validate_normalization_stats(stats, spec, require_state=True)
    meta = {
        "samples": sample_count,
        "episodes_used": episodes_used,
        "episodes_skipped": episodes_skipped,
        "episodes_listed": total,
        "workers": workers,
        "stride": stride,
        "path": "vectorized_multiprocess",
    }
    return stats, meta


def _iter_episode_batches(
    episodes: Sequence[EpisodeRef],
    spec: BimanualActionSpec,
    contract: SharpaObservationContract,
    *,
    workers: int,
    stride: int,
):
    if workers == 1:
        for episode in episodes:
            yield _extract_episode_or_skip(episode, spec, contract, stride)
        return

    ctx = _multiprocessing_context()
    chunksize = max(1, min(8, len(episodes) // (workers * 4) or 1))
    with ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(spec, contract, stride),
        maxtasksperchild=64,
    ) as pool:
        yield from pool.imap(_extract_worker, episodes, chunksize=chunksize)


def _extract_episode_or_skip(
    episode: EpisodeRef,
    spec: BimanualActionSpec,
    contract: SharpaObservationContract,
    stride: int,
) -> EpisodeNormBatch | None:
    try:
        return extract_episode_norm_batch(episode, spec, contract, stride=stride)
    except ValueError:
        return None


def _init_worker(spec: BimanualActionSpec, contract: SharpaObservationContract, stride: int) -> None:
    _limit_blas_threads()
    _WORKER_STATE["spec"] = spec
    _WORKER_STATE["contract"] = contract
    _WORKER_STATE["stride"] = stride


def _extract_worker(episode: EpisodeRef) -> EpisodeNormBatch | None:
    spec = _WORKER_STATE.get("spec")
    contract = _WORKER_STATE.get("contract")
    stride = _WORKER_STATE.get("stride")
    if spec is None or contract is None or stride is None:
        raise RuntimeError("norm worker: missing initializer state")
    return _extract_episode_or_skip(episode, spec, contract, stride)


def _multiprocessing_context():
    import multiprocessing as mp

    requested = os.environ.get("NORM_MP_START", "fork").strip() or "fork"
    available = mp.get_all_start_methods()
    start = requested if requested in available else ("spawn" if "spawn" in available else available[0])
    return mp.get_context(start)


def _limit_blas_threads() -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    try:
        import threadpoolctl

        threadpoolctl.threadpool_limits(1)
    except Exception:
        pass


def _window_raw_indices(
    aligned_index: np.ndarray,
    starts: np.ndarray,
    horizon: int,
    timebase: ActionTimebase,
) -> np.ndarray:
    if timebase is ActionTimebase.RAW_CONTROL_60_HZ:
        first = aligned_index[starts].astype(np.int64, copy=False)
        return first[:, np.newaxis] + np.arange(horizon, dtype=np.int64)
    frame_index = starts[:, np.newaxis] + np.arange(horizon, dtype=np.int64)
    return aligned_index[frame_index].astype(np.int64, copy=False)


def _window_timestamps(group: CommandedJointGroup, starts: np.ndarray, spec: BimanualActionSpec) -> np.ndarray:
    indices = _window_raw_indices(group.aligned_index, starts, spec.physical_horizon, spec.timebase)
    return group.time[indices, 0]


def _gather_joint_actions(
    groups: tuple[CommandedJointGroup, CommandedJointGroup, CommandedJointGroup, CommandedJointGroup],
    starts: np.ndarray,
    spec: BimanualActionSpec,
) -> tuple[np.ndarray, np.ndarray]:
    left_arm, left_hand, right_arm, right_hand = groups

    def _angles(group: CommandedJointGroup) -> np.ndarray:
        indices = _window_raw_indices(group.aligned_index, starts, spec.physical_horizon, spec.timebase)
        return np.asarray(group.joint_angles[indices], dtype=np.float32)

    left = np.concatenate((_angles(left_arm), _angles(left_hand)), axis=-1)
    right = np.concatenate((_angles(right_arm), _angles(right_hand)), axis=-1)
    expected = (starts.shape[0], spec.physical_horizon, spec.logical_action_dim)
    if left.shape != expected or right.shape != expected:
        raise ValueError(f"gathered actions: expected {expected}, got left={left.shape} right={right.shape}")
    return left, right


def _load_state_sources(handle, contract: SharpaObservationContract, *, aligned_length: int) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    sources: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    for column in contract.state_columns:
        group_root = column.source_path.rsplit("/", 1)[0]
        aligned_path = f"{group_root}/aligned_index"
        if aligned_path not in handle:
            raise KeyError(f"state column missing aligned_index: {aligned_path}")
        aligned = np.asarray(handle[aligned_path], dtype=np.int32)
        if aligned.shape[0] != aligned_length:
            raise ValueError(
                f"{aligned_path}.shape[0]: expected canonical N={aligned_length}, got {aligned.shape[0]}"
            )
        values = np.asarray(handle[column.source_path], dtype=np.float32)
        sources.append((aligned, values, column.slice_start, column.slice_stop))
    return sources


def _gather_state(
    sources: Sequence[tuple[np.ndarray, np.ndarray, int, int]],
    starts: np.ndarray,
    *,
    state_dim: int,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for aligned, values, slice_start, slice_stop in sources:
        raw_index = aligned[starts]
        rows = values[raw_index]
        if rows.ndim == 1:
            rows = rows[:, np.newaxis]
        pieces.append(np.asarray(rows[:, slice_start:slice_stop], dtype=np.float32))
    state = np.concatenate(pieces, axis=1)
    expected = (starts.shape[0], state_dim)
    if state.shape != expected:
        raise ValueError(f"state: expected {expected}, got {state.shape}")
    return state
