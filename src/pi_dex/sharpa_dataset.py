"""First-party Sharpa HDF5 observation dataset for PI-DEX joint_29d training.

Samples are unbatched dictionaries with:

* ``image`` / ``image_mask`` — OpenPI keys from the reviewed observation contract
* ``state`` — 1-D float32 vector built from reviewed state columns
* ``prompt`` — language string from the reviewed prompt policy
* ``left_actions`` / ``right_actions`` — ``[K, 29]`` absolute commanded joints

The dataset never constructs FK. Cartesian mode is intentionally unsupported here.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import json
import pathlib
import re
from typing import Any

import numpy as np

from pi_dex.actions import ActionRepresentation
from pi_dex.jpeg_eoi import trim_padded_jpeg
from pi_dex.observation_contract import OPENPI_IMAGE_KEYS
from pi_dex.observation_contract import ImageDtypeRange
from pi_dex.observation_contract import ImageLayout
from pi_dex.observation_contract import MissingPromptPolicy
from pi_dex.observation_contract import SharpaObservationContract
from pi_dex.sharpa_data import CANONICAL_ALIGNED_TIME_FIELD
from pi_dex.sharpa_data import AlignedTimeline
from pi_dex.sharpa_data import CommandedJointGroup
from pi_dex.sharpa_data import EpisodeActionProvenance
from pi_dex.sharpa_data import derive_bimanual_logical_action_chunk
from pi_dex.spec import BimanualActionSpec

_HDF5_NAME_RE = re.compile(r"^train_.*\.hdf5$")


@dataclasses.dataclass(frozen=True)
class EpisodeRef:
    """One on-disk Sharpa episode directory."""

    episode_id: str
    episode_dir: pathlib.Path
    hdf5_path: pathlib.Path
    anno_path: pathlib.Path


@dataclasses.dataclass(frozen=True)
class SampleIndex:
    """One random-access training window."""

    episode_index: int
    start_aligned_frame: int


def discover_episodes(dataset_root: str | pathlib.Path) -> tuple[EpisodeRef, ...]:
    """Discover episode directories under a SharpaOpenData-style root."""
    root = pathlib.Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset_root: not a directory: {root}")
    episodes: list[EpisodeRef] = []
    for anno_path in sorted(root.rglob("anno.json")):
        episode_dir = anno_path.parent
        hdf5_candidates = sorted(
            path for path in episode_dir.iterdir() if path.is_file() and _HDF5_NAME_RE.match(path.name)
        )
        if len(hdf5_candidates) != 1:
            continue
        episode_id = str(episode_dir.relative_to(root))
        episodes.append(
            EpisodeRef(
                episode_id=episode_id,
                episode_dir=episode_dir,
                hdf5_path=hdf5_candidates[0],
                anno_path=anno_path,
            )
        )
    if not episodes:
        raise ValueError(f"dataset_root: no episodes with anno.json + train_*.hdf5 under {root}")
    return tuple(episodes)


def build_sample_index(
    episodes: Sequence[EpisodeRef],
    *,
    spec: BimanualActionSpec,
    contract: SharpaObservationContract,
    provenance: EpisodeActionProvenance | None = None,
    max_episodes: int | None = None,
) -> tuple[SampleIndex, ...]:
    """Build start-frame indices that can host a full physical horizon.

    When ``provenance`` is provided, each candidate start is fully probed with
    the same cadence/finite contract as training (vectorized per episode).
    """
    selected = list(episodes) if max_episodes is None else list(episodes[:max_episodes])
    if provenance is not None:
        return _build_sample_index_vectorized(selected, spec=spec, contract=contract)
    return _build_sample_index_horizon_only(selected, spec=spec, contract=contract)


def _build_sample_index_vectorized(
    selected: Sequence[EpisodeRef],
    *,
    spec: BimanualActionSpec,
    contract: SharpaObservationContract,
) -> tuple[SampleIndex, ...]:
    from pi_dex.norm_compute import episode_valid_start_frames

    indexes: list[SampleIndex] = []
    total = len(selected)
    print(
        f"build_sample_index: vectorized episodes={total} horizon={spec.physical_horizon}",
        flush=True,
    )
    for episode_index, episode in enumerate(selected):
        starts = episode_valid_start_frames(episode, spec, contract)
        for start in starts.tolist():
            indexes.append(SampleIndex(episode_index=episode_index, start_aligned_frame=int(start)))
        done = episode_index + 1
        if done == 1 or done % 50 == 0 or done == total:
            print(
                f"build_sample_index: episodes {done}/{total} windows={len(indexes)}",
                flush=True,
            )
    if not indexes:
        raise ValueError(
            f"build_sample_index: no valid start frames (episodes={total}, horizon={spec.physical_horizon})"
        )
    return tuple(indexes)


def _build_sample_index_horizon_only(
    selected: Sequence[EpisodeRef],
    *,
    spec: BimanualActionSpec,
    contract: SharpaObservationContract,
) -> tuple[SampleIndex, ...]:
    import h5py

    indexes: list[SampleIndex] = []
    horizon = spec.physical_horizon
    for episode_index, episode in enumerate(selected):
        with h5py.File(episode.hdf5_path, "r") as handle:
            aligned_length = int(handle[CANONICAL_ALIGNED_TIME_FIELD].shape[0])
            valid_mask = None
            if "mode/sub_state" in handle:
                sub_state = np.asarray(handle["mode/sub_state"], dtype=np.int16)
                if sub_state.shape[0] != aligned_length:
                    raise ValueError(
                        f"{episode.episode_id}: mode/sub_state length {sub_state.shape[0]} "
                        f"!= aligned N={aligned_length}"
                    )
                valid_mask = sub_state == 1
            for start in range(aligned_length):
                if valid_mask is not None and not bool(valid_mask[start]):
                    continue
                try:
                    _probe_action_horizon_fits(handle, start_aligned_frame=start, spec=spec)
                except ValueError:
                    continue
                indexes.append(SampleIndex(episode_index=episode_index, start_aligned_frame=start))
        if contract.episode_tail_policy.value == "reject_incomplete_horizon" and not indexes:
            continue
    if not indexes:
        raise ValueError(f"build_sample_index: no valid start frames (episodes={len(selected)}, horizon={horizon})")
    return tuple(indexes)


class SharpaJoint29dDataset:
    """Random-access joint_29d dataset over discovered Sharpa episodes.

    Opens HDF5 files lazily. With ``num_workers=0`` a single process may reuse the
    last opened handle; workers must never inherit parent handles.
    """

    def __init__(
        self,
        *,
        episodes: Sequence[EpisodeRef],
        sample_index: Sequence[SampleIndex],
        spec: BimanualActionSpec,
        contract: SharpaObservationContract,
        provenance: EpisodeActionProvenance,
        require_reviewed: bool = True,
    ) -> None:
        if spec.action_representation is not ActionRepresentation.JOINT_29D:
            raise ValueError("SharpaJoint29dDataset: only joint_29d is supported")
        if contract.action_representation is not ActionRepresentation.JOINT_29D:
            raise ValueError("observation contract action_representation must be joint_29d")
        contract.validate_against_action_spec(spec)
        if require_reviewed:
            contract.require_reviewed_for_training()
        if not episodes:
            raise ValueError("episodes: expected a non-empty sequence")
        if not sample_index:
            raise ValueError("sample_index: expected a non-empty sequence")
        self._episodes = tuple(episodes)
        self._sample_index = tuple(sample_index)
        self._spec = dataclasses.replace(spec)
        self._contract = contract
        self._provenance = provenance
        self._handle = None
        self._handle_path: pathlib.Path | None = None

    @property
    def spec(self) -> BimanualActionSpec:
        return self._spec

    @property
    def episodes(self) -> tuple[EpisodeRef, ...]:
        return self._episodes

    @property
    def sample_index(self) -> tuple[SampleIndex, ...]:
        return self._sample_index

    def __len__(self) -> int:
        return len(self._sample_index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if type(index) is not int:
            raise TypeError(f"index: expected int, got {type(index).__name__}")
        if index < 0:
            index += len(self._sample_index)
        if index < 0 or index >= len(self._sample_index):
            raise IndexError(f"index: out of range {index} for length {len(self._sample_index)}")
        sample_ref = self._sample_index[index]
        episode = self._episodes[sample_ref.episode_index]
        handle = self._open(episode.hdf5_path)
        prompt = _load_prompt(episode.anno_path, self._contract)
        state = _load_state(handle, sample_ref.start_aligned_frame, self._contract)
        images, image_masks = _load_images(handle, sample_ref.start_aligned_frame, self._contract)
        chunk = _load_action_chunk(
            handle,
            start_aligned_frame=sample_ref.start_aligned_frame,
            spec=self._spec,
            provenance=self._provenance,
        )
        return {
            "image": images,
            "image_mask": image_masks,
            "state": state,
            "prompt": prompt,
            "left_actions": chunk.left_actions,
            "right_actions": chunk.right_actions,
            "episode_id": episode.episode_id,
            "start_aligned_frame": sample_ref.start_aligned_frame,
        }

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._handle_path = None

    def _open(self, path: pathlib.Path):
        import h5py

        if self._handle is not None and self._handle_path == path:
            return self._handle
        self.close()
        self._handle = h5py.File(path, "r")
        self._handle_path = path
        return self._handle


@dataclasses.dataclass(frozen=True)
class SyntheticJoint29dDataset:
    """In-memory joint_29d dataset for CPU-only pipeline tests."""

    physical_horizon: int
    state_dim: int
    length: int = 8
    image_hw: tuple[int, int] = (32, 32)
    seed: int = 0

    def __post_init__(self) -> None:
        if self.physical_horizon <= 0 or self.state_dim <= 0 or self.length <= 0:
            raise ValueError("SyntheticJoint29dDataset: expected positive sizes")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if type(index) is not int:
            raise TypeError(f"index: expected int, got {type(index).__name__}")
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError(index)
        rng = np.random.default_rng(self.seed + index)
        height, width = self.image_hw
        image = {key: rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8) for key in OPENPI_IMAGE_KEYS}
        image_mask = dict.fromkeys(OPENPI_IMAGE_KEYS, np.True_)
        state = rng.normal(size=(self.state_dim,)).astype(np.float32)
        left = rng.normal(size=(self.physical_horizon, 29)).astype(np.float32)
        right = rng.normal(size=(self.physical_horizon, 29)).astype(np.float32)
        return {
            "image": image,
            "image_mask": image_mask,
            "state": state,
            "prompt": f"synthetic joint task {index}",
            "left_actions": left,
            "right_actions": right,
            "episode_id": "synthetic",
            "start_aligned_frame": index,
        }


def _probe_action_horizon_fits(handle, *, start_aligned_frame: int, spec: BimanualActionSpec) -> None:
    for field_name in (
        "action/left_arm/joint_angle",
        "action/left_hand/joint_angle",
        "action/right_arm/joint_angle",
        "action/right_hand/joint_angle",
    ):
        group_root = field_name.rsplit("/", 1)[0]
        aligned = np.asarray(handle[f"{group_root}/aligned_index"], dtype=np.int32)
        values = handle[field_name]
        if start_aligned_frame >= aligned.shape[0]:
            raise ValueError("start exceeds aligned length")
        first = int(aligned[start_aligned_frame])
        if first + spec.physical_horizon > values.shape[0]:
            raise ValueError("raw horizon exceeds M")


def _load_prompt(anno_path: pathlib.Path, contract: SharpaObservationContract) -> str:
    payload = json.loads(anno_path.read_text(encoding="utf-8"))
    source = contract.prompt_policy.source
    if source != "anno.json:tags.task_instruction":
        raise ValueError(f"prompt_policy.source: unsupported {source!r}")
    try:
        prompt = payload["tags"]["task_instruction"]
    except KeyError as error:
        if contract.prompt_policy.missing_policy is MissingPromptPolicy.REJECT:
            raise KeyError(f"{anno_path}: missing tags.task_instruction") from error
        raise
    if type(prompt) is not str or not prompt.strip():
        if contract.prompt_policy.missing_policy is MissingPromptPolicy.REJECT:
            raise ValueError(f"{anno_path}: empty task_instruction")
        raise ValueError(f"{anno_path}: invalid task_instruction")
    if contract.prompt_policy.normalize_whitespace:
        prompt = " ".join(prompt.split())
    return prompt


def _load_state(handle, start_aligned_frame: int, contract: SharpaObservationContract) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for column in contract.state_columns:
        group_root = column.source_path.rsplit("/", 1)[0]
        aligned_path = f"{group_root}/aligned_index"
        if aligned_path not in handle:
            raise KeyError(f"state column missing aligned_index: {aligned_path}")
        aligned = np.asarray(handle[aligned_path], dtype=np.int32)
        raw_index = int(aligned[start_aligned_frame])
        values = np.asarray(handle[column.source_path][raw_index], dtype=np.float32)
        if values.ndim != 1:
            raise ValueError(f"{column.source_path}: expected 1-D row, got shape {values.shape}")
        pieces.append(values[column.slice_start : column.slice_stop].astype(np.float32, copy=True))
    state = np.concatenate(pieces, axis=0)
    if state.shape != (contract.state_dim,):
        raise ValueError(f"state: expected width {contract.state_dim}, got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("state: expected finite values")
    return state


def _load_images(
    handle,
    start_aligned_frame: int,
    contract: SharpaObservationContract,
) -> tuple[dict[str, np.ndarray], dict[str, np.bool_]]:
    if contract.image_layout is not ImageLayout.HWC:
        raise ValueError(f"image_layout: unsupported {contract.image_layout}")
    if contract.image_dtype_range is not ImageDtypeRange.UINT8_0_255:
        raise ValueError(f"image_dtype_range: unsupported {contract.image_dtype_range}")
    import io

    from PIL import Image

    images: dict[str, np.ndarray] = {}
    masks: dict[str, np.bool_] = {}
    reference_shape: tuple[int, int, int] | None = None
    for slot in contract.image_slots:
        if slot.sharpa_group is None:
            if reference_shape is None:
                raise ValueError("image_slots: cannot pad missing camera before a present one")
            images[slot.openpi_key] = np.zeros(reference_shape, dtype=np.uint8)
            masks[slot.openpi_key] = np.False_
            continue
        rgb_path = f"{slot.sharpa_group}/rgb"
        row = np.asarray(handle[rgb_path][start_aligned_frame], dtype=np.uint8)
        jpeg_bytes = trim_padded_jpeg(row)
        image = np.asarray(Image.open(io.BytesIO(jpeg_bytes)).convert("RGB"), dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{rgb_path}: expected HWC RGB, got {image.shape}")
        if slot.crop_roi_xyxy is not None:
            x0, y0, x1, y1 = slot.crop_roi_xyxy
            image = image[y0:y1, x0:x1]
        if slot.resize_hw is not None:
            image = np.asarray(
                Image.fromarray(image).resize((slot.resize_hw[1], slot.resize_hw[0]), Image.BILINEAR),
                dtype=np.uint8,
            )
        if reference_shape is None:
            reference_shape = image.shape
        images[slot.openpi_key] = image
        masks[slot.openpi_key] = np.bool_(slot.mask_when_present)
    return images, masks


def _load_action_chunk(
    handle,
    *,
    start_aligned_frame: int,
    spec: BimanualActionSpec,
    provenance: EpisodeActionProvenance,
):
    timeline = AlignedTimeline(
        CANONICAL_ALIGNED_TIME_FIELD,
        np.asarray(handle[CANONICAL_ALIGNED_TIME_FIELD], dtype=np.float64),
    )
    return derive_bimanual_logical_action_chunk(
        aligned_timeline=timeline,
        provenance=provenance,
        left_arm=_command_group(handle, "action/left_arm/joint_angle", spec.left_arm_joint_order),
        left_hand=_command_group(handle, "action/left_hand/joint_angle", spec.left_hand_joint_order),
        right_arm=_command_group(handle, "action/right_arm/joint_angle", spec.right_arm_joint_order),
        right_hand=_command_group(handle, "action/right_hand/joint_angle", spec.right_hand_joint_order),
        start_aligned_frame=start_aligned_frame,
        spec=spec,
        kinematics=None,
    )


def _command_group(handle, field_name: str, joint_order: tuple[str, ...]) -> CommandedJointGroup:
    group_root = field_name.rsplit("/", 1)[0]
    return CommandedJointGroup(
        field_name=field_name,
        joint_order=joint_order,
        joint_angles=np.asarray(handle[field_name], dtype=np.float32),
        time=np.asarray(handle[f"{group_root}/time"], dtype=np.float64),
        aligned_index=np.asarray(handle[f"{group_root}/aligned_index"], dtype=np.int32),
    )


def dataset_manifest(
    *,
    episodes: Sequence[EpisodeRef],
    sample_index: Sequence[SampleIndex],
    contract: SharpaObservationContract,
) -> dict[str, Any]:
    """Serializable manifest for lineage / bad-sample reporting."""
    return {
        "contract_id": contract.contract_id,
        "review_status": contract.review_status.value,
        "episode_count": len(episodes),
        "sample_count": len(sample_index),
        "episodes": [episode.episode_id for episode in episodes],
    }
