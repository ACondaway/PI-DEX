"""Map Sharpa North live SDK observations onto OpenPI / PI-DEX policy inputs.

Live SDK keys follow ``examples/sharpa_north_sdk.py`` (leading slash). HDF5
training paths omit the leading slash; ``pi_dex.sharpa_runtime_keys`` records
the identity bridge. This module is pure NumPy and does not import Zenoh or
protobuf.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from pi_dex.deployment import CLOCK_DOMAIN_FIELD
from pi_dex.deployment import OBSERVATION_TIMESTAMP_FIELD
from pi_dex.observation_contract import ImageDtypeRange
from pi_dex.observation_contract import ImageLayout
from pi_dex.observation_contract import OPENPI_IMAGE_KEYS
from pi_dex.observation_contract import SharpaObservationContract
from pi_dex.sharpa_runtime_keys import SDK_TO_HDF5_STATE
from pi_dex.sharpa_runtime_keys import SDK_TO_HDF5_VISION


def hdf5_path_to_sdk_key(hdf5_path: str) -> str:
    """Convert an on-disk HDF5 relative path to the live SDK dict key."""
    if not hdf5_path or hdf5_path.startswith("/"):
        raise ValueError(f"hdf5_path: expected relative path without leading '/', got {hdf5_path!r}")
    return f"/{hdf5_path}"


def sdk_key_to_hdf5_path(sdk_key: str) -> str:
    """Strip the live SDK leading slash to recover the HDF5-relative path."""
    if not sdk_key.startswith("/"):
        raise ValueError(f"sdk_key: expected leading '/', got {sdk_key!r}")
    return sdk_key[1:]


def build_policy_observation_from_sdk(
    sdk_observation: Mapping[str, Any],
    contract: SharpaObservationContract,
    *,
    prompt: str,
    observation_timestamp_ns: int,
    clock_domain: str,
) -> dict[str, Any]:
    """Build an unbatched OpenPI observation plus PI-DEX transport fields.

    Args:
        sdk_observation: Dict produced by NorthDirect-style conversion (SDK keys).
        contract: Reviewed joint_29d observation contract.
        prompt: Language instruction (live deployments often pass a fixed task
            string when ``/language`` is absent).
        observation_timestamp_ns: Positive transport timestamp in nanoseconds.
        clock_domain: Must match the deployment ``BimanualActionSpec.clock_domain``.

    Returns:
        Mapping with ``image``, ``image_mask``, ``state``, ``prompt``, plus
        ``observation_timestamp_ns`` and ``clock_domain`` for
        :class:`~pi_dex.deployment.BimanualPolicyAdapter`.
    """
    if type(prompt) is not str or not prompt.strip():
        raise ValueError("prompt: expected a non-empty string")
    if type(observation_timestamp_ns) is not int or observation_timestamp_ns <= 0:
        raise ValueError("observation_timestamp_ns: expected a positive int")
    if type(clock_domain) is not str or not clock_domain.strip():
        raise ValueError("clock_domain: expected a non-empty string")
    if contract.image_layout is not ImageLayout.HWC:
        raise ValueError(f"image_layout: unsupported {contract.image_layout}")
    if contract.image_dtype_range is not ImageDtypeRange.UINT8_0_255:
        raise ValueError(f"image_dtype_range: unsupported {contract.image_dtype_range}")

    state = _load_state_from_sdk(sdk_observation, contract)
    images, masks = _load_images_from_sdk(sdk_observation, contract)
    return {
        "image": images,
        "image_mask": masks,
        "state": state,
        "prompt": " ".join(prompt.split()) if contract.prompt_policy.normalize_whitespace else prompt,
        OBSERVATION_TIMESTAMP_FIELD: observation_timestamp_ns,
        CLOCK_DOMAIN_FIELD: clock_domain,
    }


def resolve_live_prompt(sdk_observation: Mapping[str, Any], *, fallback: str | None = None) -> str:
    """Prefer ``/language`` from the SDK payload, else a caller-supplied fallback."""
    language = sdk_observation.get("/language")
    if isinstance(language, str) and language.strip():
        return language.strip()
    if fallback is not None and fallback.strip():
        return fallback.strip()
    raise KeyError("sdk observation missing /language and no fallback prompt was provided")


def resolve_observation_timestamp_ns(sdk_observation: Mapping[str, Any]) -> int:
    """Derive a positive nanosecond timestamp from the SDK payload."""
    raw = sdk_observation.get("timestamp")
    if raw is None:
        raise KeyError("sdk observation missing timestamp")
    if isinstance(raw, (np.integer, int)):
        value = int(raw)
    elif isinstance(raw, float):
        # SDK reference uses time.time()-like seconds for some action paths;
        # observations may already be integer ns. Treat values < 1e12 as seconds.
        value = int(raw * 1_000_000_000) if raw < 1_000_000_000_000 else int(raw)
    else:
        raise TypeError(f"timestamp: expected numeric, got {type(raw).__name__}")
    if value <= 0:
        raise ValueError(f"timestamp: expected positive, got {value}")
    return value


def _load_state_from_sdk(
    sdk_observation: Mapping[str, Any],
    contract: SharpaObservationContract,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for column in contract.state_columns:
        sdk_key = hdf5_path_to_sdk_key(column.source_path)
        if sdk_key not in sdk_observation:
            # Allow documented aliases from the runtime key table.
            aliases = [key for key, hdf5 in SDK_TO_HDF5_STATE.items() if hdf5 == column.source_path]
            for alias in aliases:
                if alias in sdk_observation:
                    sdk_key = alias
                    break
            else:
                raise KeyError(f"sdk observation missing state key {sdk_key!r}")
        values = np.asarray(sdk_observation[sdk_key], dtype=np.float32)
        if values.ndim != 1:
            raise ValueError(f"{sdk_key}: expected 1-D vector, got shape {values.shape}")
        pieces.append(values[column.slice_start : column.slice_stop].astype(np.float32, copy=True))
    state = np.concatenate(pieces, axis=0)
    if state.shape != (contract.state_dim,):
        raise ValueError(f"state: expected width {contract.state_dim}, got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("state: expected finite values")
    return state


def _load_images_from_sdk(
    sdk_observation: Mapping[str, Any],
    contract: SharpaObservationContract,
) -> tuple[dict[str, np.ndarray], dict[str, np.bool_]]:
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
        sdk_key = hdf5_path_to_sdk_key(f"{slot.sharpa_group}/rgb")
        if sdk_key not in sdk_observation:
            vision_aliases = [
                key for key, hdf5 in SDK_TO_HDF5_VISION.items() if hdf5 == f"{slot.sharpa_group}/rgb"
            ]
            for alias in vision_aliases:
                if alias in sdk_observation:
                    sdk_key = alias
                    break
            else:
                raise KeyError(f"sdk observation missing image key {sdk_key!r}")
        image = np.asarray(sdk_observation[sdk_key])
        if image.dtype != np.uint8:
            raise TypeError(f"{sdk_key}: expected uint8, got {image.dtype}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{sdk_key}: expected HWC RGB, got {image.shape}")
        if slot.crop_roi_xyxy is not None:
            x0, y0, x1, y1 = slot.crop_roi_xyxy
            image = image[y0:y1, x0:x1]
        if slot.resize_hw is not None:
            # Avoid hard Pillow dependency in the pure conversion path for tests;
            # deployments that need resize should set resize_hw in the contract
            # and provide Pillow in the runtime env.
            from PIL import Image

            image = np.asarray(
                Image.fromarray(image).resize((slot.resize_hw[1], slot.resize_hw[0]), Image.BILINEAR),
                dtype=np.uint8,
            )
        if reference_shape is None:
            reference_shape = image.shape
        elif image.shape != reference_shape:
            raise ValueError(
                f"{sdk_key}: shape {image.shape} conflicts with reference {reference_shape}"
            )
        images[slot.openpi_key] = np.asarray(image, dtype=np.uint8, order="C")
        masks[slot.openpi_key] = np.bool_(slot.mask_when_present)
    for key in OPENPI_IMAGE_KEYS:
        if key not in images:
            raise KeyError(f"contract image_slots omitted required OpenPI key {key!r}")
    return images, masks
