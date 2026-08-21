"""Versioned Sharpa observation and dataset semantics that must be site-reviewed.

Handoff phase 1 requires freezing camera slots, state columns, prompt rules, and
episode splits before any HDF5 training loader is treated as ready. This module
stores those decisions as an immutable contract and refuses to load configs that
are still marked unreviewed or that leave required fields empty.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import enum
import json
import pathlib
from typing import Any

from pi_dex.core.actions import ActionRepresentation
from pi_dex.core.spec import ActionTimebase
from pi_dex.core.spec import BimanualActionSpec

OPENPI_IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

# Schema-documented Sharpa camera groups. Mapping onto OPENPI_IMAGE_KEYS is a
# site decision; the fourth head view must not be guessed by the loader.
SHARPA_VISION_GROUPS = (
    "observe/vision/head/stereo/lefteye",
    "observe/vision/head/stereo/righteye",
    "observe/vision/left_wrist/fisheye",
    "observe/vision/right_wrist/fisheye",
)

DATASET_CONTRACT_SCHEMA_VERSION = 1


class ReviewStatus(enum.StrEnum):
    """Whether a site dataset contract may be used for training code paths."""

    UNREVIEWED = "unreviewed"
    REVIEWED = "reviewed"


class EpisodeTailPolicy(enum.StrEnum):
    """How samples near the end of an episode are handled."""

    REJECT_INCOMPLETE_HORIZON = "reject_incomplete_horizon"
    REQUIRE_VALID_ANNOTATION_COVER = "require_valid_annotation_cover"


class MissingPromptPolicy(enum.StrEnum):
    """Behavior when a sample has no reviewed language prompt."""

    REJECT = "reject"
    USE_TASK_INSTRUCTION = "use_task_instruction"


class ImageChannelOrder(enum.StrEnum):
    """Channel order produced by the Sharpa image decoder before OpenPI."""

    RGB = "rgb"


class ImageLayout(enum.StrEnum):
    """Array layout of decoded images before OpenPI transforms."""

    HWC = "hwc"


class ImageDtypeRange(enum.StrEnum):
    """Pixel dtype and numeric range after decoding."""

    UINT8_0_255 = "uint8_0_255"


@dataclasses.dataclass(frozen=True)
class StateColumn:
    """One column contributed to the one-dimensional OpenPI ``state`` vector.

    Args:
        source_path: HDF5 dataset path relative to the episode file, using
            ``aligned_index`` when the source is on the 60 Hz timeline.
        slice_start: Inclusive start index within the source vector.
        slice_stop: Exclusive stop index within the source vector.
        unit: Physical unit string, for example ``rad`` or ``m``.
        semantics: Absolute/delta meaning of the column values.
    """

    source_path: str
    slice_start: int
    slice_stop: int
    unit: str
    semantics: str

    def __post_init__(self) -> None:
        _require_nonempty(self.source_path, field_name="source_path")
        _require_nonempty(self.unit, field_name="unit")
        _require_nonempty(self.semantics, field_name="semantics")
        if type(self.slice_start) is not int or type(self.slice_stop) is not int:
            raise TypeError("state column slice bounds must be plain ints")
        if self.slice_start < 0 or self.slice_stop <= self.slice_start:
            raise ValueError(
                "state column slice: expected slice_stop > slice_start >= 0, "
                f"got [{self.slice_start}, {self.slice_stop})"
            )

    @property
    def width(self) -> int:
        return self.slice_stop - self.slice_start


@dataclasses.dataclass(frozen=True)
class ImageSlotMapping:
    """Map one OpenPI image key onto a Sharpa vision group and mask policy."""

    openpi_key: str
    sharpa_group: str | None
    mask_when_present: bool
    resize_hw: tuple[int, int] | None
    crop_roi_xyxy: tuple[int, int, int, int] | None

    def __post_init__(self) -> None:
        if self.openpi_key not in OPENPI_IMAGE_KEYS:
            raise ValueError(f"openpi_key: unsupported OpenPI image key {self.openpi_key!r}")
        if self.sharpa_group is not None:
            if self.sharpa_group not in SHARPA_VISION_GROUPS:
                raise ValueError(f"sharpa_group: unsupported vision group {self.sharpa_group!r}")
            if type(self.mask_when_present) is not bool:
                raise TypeError("mask_when_present: expected bool")
        elif self.mask_when_present:
            raise ValueError(
                f"{self.openpi_key}: missing Sharpa group cannot be marked present "
                "(mask_when_present must be False for explicit padding slots)"
            )
        if self.resize_hw is not None:
            height, width = self.resize_hw
            if type(height) is not int or type(width) is not int or height <= 0 or width <= 0:
                raise ValueError(f"resize_hw: expected positive ints, got {self.resize_hw!r}")
        if self.crop_roi_xyxy is not None:
            if len(self.crop_roi_xyxy) != 4 or any(type(value) is not int for value in self.crop_roi_xyxy):
                raise ValueError(f"crop_roi_xyxy: expected four ints, got {self.crop_roi_xyxy!r}")
            x0, y0, x1, y1 = self.crop_roi_xyxy
            if not (0 <= x0 < x1 and 0 <= y0 < y1):
                raise ValueError(f"crop_roi_xyxy: invalid box {self.crop_roi_xyxy!r}")


@dataclasses.dataclass(frozen=True)
class PromptPolicy:
    """Language prompt extraction and missing-prompt behavior."""

    source: str
    missing_policy: MissingPromptPolicy
    normalize_whitespace: bool
    annotation_interval_closed: bool

    def __post_init__(self) -> None:
        _require_nonempty(self.source, field_name="prompt_policy.source")
        if not isinstance(self.missing_policy, MissingPromptPolicy):
            raise TypeError(f"missing_policy: expected MissingPromptPolicy, got {type(self.missing_policy).__name__}")
        if type(self.normalize_whitespace) is not bool:
            raise TypeError("normalize_whitespace: expected bool")
        if type(self.annotation_interval_closed) is not bool:
            raise TypeError("annotation_interval_closed: expected bool")


@dataclasses.dataclass(frozen=True)
class SplitPolicy:
    """Episode-level split and leakage controls."""

    strategy: str
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    seed: int
    dedupe_by_episode_id: bool

    def __post_init__(self) -> None:
        _require_nonempty(self.strategy, field_name="split_policy.strategy")
        if type(self.seed) is not int:
            raise TypeError("split_policy.seed: expected int")
        if type(self.dedupe_by_episode_id) is not bool:
            raise TypeError("dedupe_by_episode_id: expected bool")
        for name in ("train_fraction", "validation_fraction", "test_fraction"):
            value = getattr(self, name)
            if type(value) is not float and type(value) is not int:
                raise TypeError(f"{name}: expected a number")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name}: expected value in [0, 1], got {value!r}")
        total = float(self.train_fraction) + float(self.validation_fraction) + float(self.test_fraction)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split fractions must sum to 1.0, got {total}")


@dataclasses.dataclass(frozen=True)
class SharpaObservationContract:
    """Site-reviewed observation/dataset semantics for one representation path.

    Loading a contract with ``review_status != reviewed`` is allowed for editing
    and inspection, but ``require_reviewed_for_training`` rejects it. Training
    and dataset construction must call that gate before touching CUDA or HDF5
    sample iteration intended for a closed loop.
    """

    schema_version: int
    contract_id: str
    review_status: ReviewStatus
    reviewed_by: str | None
    review_notes: str
    action_representation: ActionRepresentation
    physical_horizon: int
    timebase: ActionTimebase
    control_frequency_hz: float
    max_group_timestamp_skew_ms: float
    max_alignment_timestamp_error_ms: float
    max_control_period_error_ms: float
    episode_tail_policy: EpisodeTailPolicy
    state_columns: tuple[StateColumn, ...]
    image_slots: tuple[ImageSlotMapping, ...]
    image_channel_order: ImageChannelOrder
    image_layout: ImageLayout
    image_dtype_range: ImageDtypeRange
    unused_sharpa_vision_groups: tuple[str, ...]
    prompt_policy: PromptPolicy
    split_policy: SplitPolicy
    num_workers: int
    bad_sample_report_path: str

    def __post_init__(self) -> None:
        if self.schema_version != DATASET_CONTRACT_SCHEMA_VERSION:
            raise ValueError(f"schema_version: expected {DATASET_CONTRACT_SCHEMA_VERSION}, got {self.schema_version!r}")
        _require_nonempty(self.contract_id, field_name="contract_id")
        if not isinstance(self.review_status, ReviewStatus):
            raise TypeError(f"review_status: expected ReviewStatus, got {type(self.review_status).__name__}")
        if not isinstance(self.action_representation, ActionRepresentation):
            raise TypeError(
                f"action_representation: expected ActionRepresentation, got {type(self.action_representation).__name__}"
            )
        if not isinstance(self.timebase, ActionTimebase):
            raise TypeError(f"timebase: expected ActionTimebase, got {type(self.timebase).__name__}")
        if not isinstance(self.episode_tail_policy, EpisodeTailPolicy):
            raise TypeError(
                f"episode_tail_policy: expected EpisodeTailPolicy, got {type(self.episode_tail_policy).__name__}"
            )
        if type(self.physical_horizon) is not int or self.physical_horizon <= 0:
            raise ValueError(f"physical_horizon: expected positive int, got {self.physical_horizon!r}")
        if type(self.control_frequency_hz) is not float and type(self.control_frequency_hz) is not int:
            raise TypeError("control_frequency_hz: expected a number")
        if float(self.control_frequency_hz) <= 0.0:
            raise ValueError(f"control_frequency_hz: expected positive, got {self.control_frequency_hz!r}")
        for field_name in (
            "max_group_timestamp_skew_ms",
            "max_alignment_timestamp_error_ms",
            "max_control_period_error_ms",
        ):
            value = getattr(self, field_name)
            if type(value) is not float and type(value) is not int:
                raise TypeError(f"{field_name}: expected a number")
            if float(value) <= 0.0:
                raise ValueError(f"{field_name}: expected positive, got {value!r}")
        if not self.state_columns:
            raise ValueError("state_columns: expected at least one reviewed column")
        if len(self.image_slots) != len(OPENPI_IMAGE_KEYS):
            raise ValueError(f"image_slots: expected {len(OPENPI_IMAGE_KEYS)} entries matching OPENPI_IMAGE_KEYS")
        slot_keys = tuple(slot.openpi_key for slot in self.image_slots)
        if slot_keys != OPENPI_IMAGE_KEYS:
            raise ValueError(f"image_slots: expected keys {OPENPI_IMAGE_KEYS}, got {slot_keys}")
        mapped_groups = [slot.sharpa_group for slot in self.image_slots if slot.sharpa_group is not None]
        if len(mapped_groups) != len(set(mapped_groups)):
            raise ValueError("image_slots: each Sharpa vision group may map to at most one OpenPI key")
        unused = set(self.unused_sharpa_vision_groups)
        if len(unused) != len(self.unused_sharpa_vision_groups):
            raise ValueError("unused_sharpa_vision_groups: duplicate entries are not allowed")
        for group in unused:
            if group not in SHARPA_VISION_GROUPS:
                raise ValueError(f"unused_sharpa_vision_groups: unknown group {group!r}")
            if group in mapped_groups:
                raise ValueError(f"unused_sharpa_vision_groups: {group!r} is already mapped")
        expected_unused = set(SHARPA_VISION_GROUPS) - set(mapped_groups)
        if unused != expected_unused:
            raise ValueError(
                "unused_sharpa_vision_groups: must exactly list unmapped Sharpa cameras; "
                f"expected {sorted(expected_unused)}, got {sorted(unused)}"
            )
        if not isinstance(self.image_channel_order, ImageChannelOrder):
            raise TypeError("image_channel_order: expected ImageChannelOrder")
        if not isinstance(self.image_layout, ImageLayout):
            raise TypeError("image_layout: expected ImageLayout")
        if not isinstance(self.image_dtype_range, ImageDtypeRange):
            raise TypeError("image_dtype_range: expected ImageDtypeRange")
        if type(self.num_workers) is not int or self.num_workers < 0:
            raise ValueError(f"num_workers: expected non-negative int, got {self.num_workers!r}")
        _require_nonempty(self.bad_sample_report_path, field_name="bad_sample_report_path")
        _require_nonempty(self.review_notes, field_name="review_notes")
        if self.review_status is ReviewStatus.REVIEWED and (self.reviewed_by is None or not self.reviewed_by.strip()):
            raise ValueError("reviewed_by: required when review_status is reviewed")

    @property
    def state_dim(self) -> int:
        return sum(column.width for column in self.state_columns)

    def require_reviewed_for_training(self) -> None:
        """Fail closed unless a data owner marked this contract reviewed."""
        if self.review_status is not ReviewStatus.REVIEWED:
            raise ValueError(
                f"observation contract {self.contract_id!r} is {self.review_status.value}; "
                "training/dataset closed-loop use requires review_status='reviewed'"
            )
        if self.num_workers != 0:
            raise ValueError(
                "num_workers: first closed loop must use 0 until per-worker HDF5 lifecycle "
                f"is implemented and tested; got {self.num_workers}"
            )

    def validate_against_action_spec(self, spec: BimanualActionSpec) -> None:
        """Ensure dataset horizon/timebase/representation match the bound action spec."""
        if self.action_representation is not spec.action_representation:
            raise ValueError(
                "action_representation mismatch: "
                f"contract={self.action_representation.value} spec={spec.action_representation.value}"
            )
        if self.physical_horizon != spec.physical_horizon:
            raise ValueError(
                f"physical_horizon mismatch: contract={self.physical_horizon} spec={spec.physical_horizon}"
            )
        if self.timebase is not spec.timebase:
            raise ValueError(f"timebase mismatch: contract={self.timebase.value} spec={spec.timebase.value}")
        if abs(float(self.control_frequency_hz) - float(spec.control_frequency_hz)) > 1e-9:
            raise ValueError(
                f"control_frequency_hz mismatch: contract={self.control_frequency_hz} spec={spec.control_frequency_hz}"
            )
        for field_name in (
            "max_group_timestamp_skew_ms",
            "max_alignment_timestamp_error_ms",
            "max_control_period_error_ms",
        ):
            contract_value = float(getattr(self, field_name))
            spec_value = float(getattr(spec, field_name))
            if abs(contract_value - spec_value) > 1e-9:
                raise ValueError(f"{field_name} mismatch: contract={contract_value} spec={spec_value}")


def load_observation_contract(path: str | pathlib.Path) -> SharpaObservationContract:
    """Load a versioned observation contract JSON file."""
    contract_path = pathlib.Path(path)
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"observation contract not found: {contract_path}") from error
    if not isinstance(payload, Mapping):
        raise TypeError(f"observation contract root: expected mapping, got {type(payload).__name__}")
    return observation_contract_from_mapping(payload)


def observation_contract_from_mapping(payload: Mapping[str, Any]) -> SharpaObservationContract:
    """Parse and validate one observation-contract mapping."""
    state_columns = tuple(_state_column_from_mapping(item) for item in _require_list(payload, "state_columns"))
    image_slots = tuple(_image_slot_from_mapping(item) for item in _require_list(payload, "image_slots"))
    unused = tuple(_require_list(payload, "unused_sharpa_vision_groups"))
    prompt_payload = _require_mapping(payload, "prompt_policy")
    split_payload = _require_mapping(payload, "split_policy")
    reviewed_by = payload.get("reviewed_by")
    if reviewed_by is not None and type(reviewed_by) is not str:
        raise TypeError("reviewed_by: expected str or null")
    return SharpaObservationContract(
        schema_version=_require_int(payload, "schema_version"),
        contract_id=_require_str(payload, "contract_id"),
        review_status=ReviewStatus(_require_str(payload, "review_status")),
        reviewed_by=reviewed_by,
        review_notes=_require_str(payload, "review_notes"),
        action_representation=ActionRepresentation(_require_str(payload, "action_representation")),
        physical_horizon=_require_int(payload, "physical_horizon"),
        timebase=ActionTimebase(_require_str(payload, "timebase")),
        control_frequency_hz=float(_require_number(payload, "control_frequency_hz")),
        max_group_timestamp_skew_ms=float(_require_number(payload, "max_group_timestamp_skew_ms")),
        max_alignment_timestamp_error_ms=float(_require_number(payload, "max_alignment_timestamp_error_ms")),
        max_control_period_error_ms=float(_require_number(payload, "max_control_period_error_ms")),
        episode_tail_policy=EpisodeTailPolicy(_require_str(payload, "episode_tail_policy")),
        state_columns=state_columns,
        image_slots=image_slots,
        image_channel_order=ImageChannelOrder(_require_str(payload, "image_channel_order")),
        image_layout=ImageLayout(_require_str(payload, "image_layout")),
        image_dtype_range=ImageDtypeRange(_require_str(payload, "image_dtype_range")),
        unused_sharpa_vision_groups=tuple(str(group) for group in unused),
        prompt_policy=PromptPolicy(
            source=_require_str(prompt_payload, "source"),
            missing_policy=MissingPromptPolicy(_require_str(prompt_payload, "missing_policy")),
            normalize_whitespace=_require_bool(prompt_payload, "normalize_whitespace"),
            annotation_interval_closed=_require_bool(prompt_payload, "annotation_interval_closed"),
        ),
        split_policy=SplitPolicy(
            strategy=_require_str(split_payload, "strategy"),
            train_fraction=float(_require_number(split_payload, "train_fraction")),
            validation_fraction=float(_require_number(split_payload, "validation_fraction")),
            test_fraction=float(_require_number(split_payload, "test_fraction")),
            seed=_require_int(split_payload, "seed"),
            dedupe_by_episode_id=_require_bool(split_payload, "dedupe_by_episode_id"),
        ),
        num_workers=_require_int(payload, "num_workers"),
        bad_sample_report_path=_require_str(payload, "bad_sample_report_path"),
    )


def observation_contract_to_mapping(contract: SharpaObservationContract) -> dict[str, Any]:
    """Serialize a contract to a JSON-compatible mapping."""
    return {
        "schema_version": contract.schema_version,
        "contract_id": contract.contract_id,
        "review_status": contract.review_status.value,
        "reviewed_by": contract.reviewed_by,
        "review_notes": contract.review_notes,
        "action_representation": contract.action_representation.value,
        "physical_horizon": contract.physical_horizon,
        "timebase": contract.timebase.value,
        "control_frequency_hz": contract.control_frequency_hz,
        "max_group_timestamp_skew_ms": contract.max_group_timestamp_skew_ms,
        "max_alignment_timestamp_error_ms": contract.max_alignment_timestamp_error_ms,
        "max_control_period_error_ms": contract.max_control_period_error_ms,
        "episode_tail_policy": contract.episode_tail_policy.value,
        "state_columns": [
            {
                "source_path": column.source_path,
                "slice_start": column.slice_start,
                "slice_stop": column.slice_stop,
                "unit": column.unit,
                "semantics": column.semantics,
            }
            for column in contract.state_columns
        ],
        "image_slots": [
            {
                "openpi_key": slot.openpi_key,
                "sharpa_group": slot.sharpa_group,
                "mask_when_present": slot.mask_when_present,
                "resize_hw": list(slot.resize_hw) if slot.resize_hw is not None else None,
                "crop_roi_xyxy": list(slot.crop_roi_xyxy) if slot.crop_roi_xyxy is not None else None,
            }
            for slot in contract.image_slots
        ],
        "image_channel_order": contract.image_channel_order.value,
        "image_layout": contract.image_layout.value,
        "image_dtype_range": contract.image_dtype_range.value,
        "unused_sharpa_vision_groups": list(contract.unused_sharpa_vision_groups),
        "prompt_policy": {
            "source": contract.prompt_policy.source,
            "missing_policy": contract.prompt_policy.missing_policy.value,
            "normalize_whitespace": contract.prompt_policy.normalize_whitespace,
            "annotation_interval_closed": contract.prompt_policy.annotation_interval_closed,
        },
        "split_policy": {
            "strategy": contract.split_policy.strategy,
            "train_fraction": contract.split_policy.train_fraction,
            "validation_fraction": contract.split_policy.validation_fraction,
            "test_fraction": contract.split_policy.test_fraction,
            "seed": contract.split_policy.seed,
            "dedupe_by_episode_id": contract.split_policy.dedupe_by_episode_id,
        },
        "num_workers": contract.num_workers,
        "bad_sample_report_path": contract.bad_sample_report_path,
    }


def _state_column_from_mapping(payload: object) -> StateColumn:
    if not isinstance(payload, Mapping):
        raise TypeError(f"state_columns item: expected mapping, got {type(payload).__name__}")
    return StateColumn(
        source_path=_require_str(payload, "source_path"),
        slice_start=_require_int(payload, "slice_start"),
        slice_stop=_require_int(payload, "slice_stop"),
        unit=_require_str(payload, "unit"),
        semantics=_require_str(payload, "semantics"),
    )


def _image_slot_from_mapping(payload: object) -> ImageSlotMapping:
    if not isinstance(payload, Mapping):
        raise TypeError(f"image_slots item: expected mapping, got {type(payload).__name__}")
    resize_hw_raw = payload.get("resize_hw")
    crop_raw = payload.get("crop_roi_xyxy")
    sharpa_group = payload.get("sharpa_group")
    if sharpa_group is not None and type(sharpa_group) is not str:
        raise TypeError("sharpa_group: expected str or null")
    resize_hw = None
    if resize_hw_raw is not None:
        if not isinstance(resize_hw_raw, list | tuple) or len(resize_hw_raw) != 2:
            raise ValueError(f"resize_hw: expected [H, W], got {resize_hw_raw!r}")
        resize_hw = (int(resize_hw_raw[0]), int(resize_hw_raw[1]))
    crop_roi_xyxy: tuple[int, int, int, int] | None = None
    if crop_raw is not None:
        if not isinstance(crop_raw, list | tuple) or len(crop_raw) != 4:
            raise ValueError(f"crop_roi_xyxy: expected [x0,y0,x1,y1], got {crop_raw!r}")
        x0, y0, x1, y1 = (int(value) for value in crop_raw)
        crop_roi_xyxy = (x0, y0, x1, y1)
    return ImageSlotMapping(
        openpi_key=_require_str(payload, "openpi_key"),
        sharpa_group=sharpa_group,
        mask_when_present=_require_bool(payload, "mask_when_present"),
        resize_hw=resize_hw,
        crop_roi_xyxy=crop_roi_xyxy,
    )


def _require_nonempty(value: object, *, field_name: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name}: expected a non-empty string")


def _require_mapping(payload: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    value = payload.get(field_name)
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name}: expected mapping, got {type(value).__name__}")
    return value


def _require_list(payload: Mapping[str, Any], field_name: str) -> list[Any]:
    value = payload.get(field_name)
    if not isinstance(value, list):
        raise TypeError(f"{field_name}: expected list, got {type(value).__name__}")
    return value


def _require_str(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if type(value) is not str:
        raise TypeError(f"{field_name}: expected str, got {type(value).__name__}")
    return value


def _require_int(payload: Mapping[str, Any], field_name: str) -> int:
    value = payload.get(field_name)
    if type(value) is not int:
        raise TypeError(f"{field_name}: expected int, got {type(value).__name__}")
    return value


def _require_number(payload: Mapping[str, Any], field_name: str) -> float:
    value = payload.get(field_name)
    if type(value) is not int and type(value) is not float:
        raise TypeError(f"{field_name}: expected number, got {type(value).__name__}")
    return float(value)


def _require_bool(payload: Mapping[str, Any], field_name: str) -> bool:
    value = payload.get(field_name)
    if type(value) is not bool:
        raise TypeError(f"{field_name}: expected bool, got {type(value).__name__}")
    return value
