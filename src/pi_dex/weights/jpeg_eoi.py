"""JPEG EOI trimming for Sharpa padded image rows.

Sharpa stores each camera frame as a fixed-width ``uint8`` row containing a
JPEG bytestream zero-padded to the episode maximum. Valid JPEG payloads may
contain ``0x00`` bytes, so trailing-zero trimming is incorrect. Decode paths
must cut at the final EOI marker ``FF D9``.
"""

from __future__ import annotations

import numpy as np

JPEG_EOI = b"\xff\xd9"


def trim_padded_jpeg(payload: bytes | bytearray | memoryview | np.ndarray) -> bytes:
    """Return the JPEG prefix ending at the last EOI marker.

    Args:
        payload: Raw padded JPEG bytes, or a 1-D ``uint8`` NumPy row.

    Returns:
        Bytes that include the final ``FF D9`` marker and exclude padding after
        it.

    Raises:
        TypeError: If ``payload`` is not bytes-like or a 1-D uint8 array.
        ValueError: If no EOI marker is present.
    """
    data = _as_bytes(payload)
    end = data.rfind(JPEG_EOI)
    if end < 0:
        raise ValueError("padded JPEG payload: missing EOI marker 0xFFD9")
    return data[: end + len(JPEG_EOI)]


def _as_bytes(payload: bytes | bytearray | memoryview | np.ndarray) -> bytes:
    if isinstance(payload, np.ndarray):
        if payload.ndim != 1:
            raise TypeError(f"padded JPEG array: expected 1-D row, got shape {payload.shape}")
        if payload.dtype != np.uint8:
            raise TypeError(f"padded JPEG array: expected uint8, got dtype {payload.dtype}")
        return payload.tobytes()
    if isinstance(payload, memoryview):
        return payload.tobytes()
    if isinstance(payload, bytearray):
        return bytes(payload)
    if isinstance(payload, bytes):
        return payload
    raise TypeError(f"padded JPEG payload: expected bytes-like or 1-D uint8 ndarray, got {type(payload).__name__}")
