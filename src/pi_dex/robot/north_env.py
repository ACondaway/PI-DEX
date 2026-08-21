"""harobotsDL-style North Zenoh hardware I/O (no policy / no OpenPI Runtime)."""

from __future__ import annotations

import logging
import pathlib
import threading
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from pi_dex.robot.north_codec import encode_uhr_action_bundle
from pi_dex.robot.north_codec import parse_north_observation
from pi_dex.robot.north_codec import sdk_action_chunk_to_step_dicts
from pi_dex.robot.north_codec import sdk_step_action_to_uhr_bundle
from pi_dex.robot.realtime_actions import SDK_LEFT_ARM_ACTION
from pi_dex.robot.realtime_actions import SDK_LEFT_HAND_ACTION
from pi_dex.robot.realtime_actions import SDK_MOTOR_ACTION
from pi_dex.robot.realtime_actions import SDK_RIGHT_ARM_ACTION
from pi_dex.robot.realtime_actions import SDK_RIGHT_HAND_ACTION
from pi_dex.robot.sharpa_runtime_keys import DEFAULT_ACTION_PUB_DURATION_S
from pi_dex.robot.sharpa_runtime_keys import DEFAULT_ACTION_TOPIC
from pi_dex.robot.sharpa_runtime_keys import DEFAULT_OBSERVATION_TOPIC

logger = logging.getLogger(__name__)

_ACTION_TO_STATE = {
    SDK_LEFT_ARM_ACTION: "/state/left_arm/joint_angle",
    SDK_LEFT_HAND_ACTION: "/state/left_hand/joint_angle",
    SDK_RIGHT_ARM_ACTION: "/state/right_arm/joint_angle",
    SDK_RIGHT_HAND_ACTION: "/state/right_hand/joint_angle",
    SDK_MOTOR_ACTION: "/state/motor/joint_angle",
}


def _sample_payload_bytes(sample: Any) -> bytes:
    payload = sample.payload
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    to_bytes = getattr(payload, "to_bytes", None)
    if callable(to_bytes):
        return bytes(to_bytes())
    return bytes(payload)


def smooth_action_accel(
    action_chunk: np.ndarray,
    start_state: np.ndarray,
    smooth_chunk_size: int,
    *,
    accel_ratio: float = 0.25,
) -> np.ndarray:
    """harobotsDL ``smooth_action_accel`` (joint positions)."""
    chunk = np.asarray(action_chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[0] <= 0:
        raise ValueError(f"action_chunk: expected [T, D], got {chunk.shape}")
    start = np.asarray(start_state, dtype=np.float32).reshape(-1)
    if start.shape[0] != chunk.shape[1]:
        raise ValueError(
            f"start_state width {start.shape[0]} != action_chunk width {chunk.shape[1]}"
        )
    n = min(int(smooth_chunk_size), int(chunk.shape[0]))
    if n <= 0:
        return chunk.copy()
    if not (0.0 < accel_ratio < 1.0):
        raise ValueError(f"accel_ratio: expected in (0, 1), got {accel_ratio}")
    end = chunk[n - 1]
    v_max = 1.0 / (1.0 - accel_ratio / 2.0)
    blended = np.empty((n, chunk.shape[1]), dtype=np.float32)
    for index in range(n):
        t = (index + 1) / n
        if t <= accel_ratio:
            alpha = v_max * (t**2) / (2.0 * accel_ratio)
        else:
            alpha = v_max * accel_ratio / 2.0 + v_max * (t - accel_ratio)
        blended[index] = start + (end - start) * alpha
    if n >= chunk.shape[0]:
        return blended
    return np.concatenate([blended, chunk[n:]], axis=0)


def apply_first_chunk_smooth(
    actions_dict: Mapping[str, list[list[float]]],
    observation: Mapping[str, Any],
    *,
    smooth_chunk_size: int,
) -> dict[str, list[list[float]]]:
    if smooth_chunk_size <= 0:
        return {key: list(values) for key, values in actions_dict.items()}
    out: dict[str, list[list[float]]] = {}
    for key, values in actions_dict.items():
        if "/action" not in key or key.endswith("_raw"):
            out[key] = list(values)
            continue
        state_key = _ACTION_TO_STATE.get(key, key.replace("/action/", "/state/", 1))
        current = observation.get(state_key)
        if current is None:
            out[key] = list(values)
            continue
        smoothed = smooth_action_accel(
            np.asarray(values, dtype=np.float32),
            np.asarray(current, dtype=np.float32),
            smooth_chunk_size,
        )
        out[key] = smoothed.tolist()
    return out


class NorthZmqEnv:
    """Zenoh North I/O mirroring harobotsDL ``NorthZmqEnv`` (joint publish path).

    Receive: ``NorthObservation`` on ``observation_topic``.
    Send: paced ``UhrActionBundle`` on ``action_topic`` via a dedicated thread.
    """

    def __init__(
        self,
        *,
        observation_topic: str = DEFAULT_OBSERVATION_TOPIC,
        action_topic: str = DEFAULT_ACTION_TOPIC,
        action_pub_duration: float = DEFAULT_ACTION_PUB_DURATION_S,
        zenoh_config: Any | None = None,
        decode_images: bool = True,
        first_chunk_smooth_size: int = 0,
        language: str | None = None,
    ) -> None:
        if action_pub_duration <= 0:
            raise ValueError("action_pub_duration: expected positive")
        if first_chunk_smooth_size < 0:
            raise ValueError("first_chunk_smooth_size: expected >= 0")
        self.observation_topic = observation_topic
        self.action_topic = action_topic
        self.action_pub_duration = float(action_pub_duration)
        self._zenoh_config = zenoh_config
        self._decode_images = decode_images
        self.first_chunk_smooth_size = int(first_chunk_smooth_size)
        self.language = language

        self._session: Any | None = None
        self._subscriber: Any | None = None
        self._publisher: Any | None = None
        self._obs_lock = threading.Lock()
        self._latest_obs: dict[str, Any] | None = None
        self._obs_seq = 0
        self._action_lock = threading.Lock()
        self._action_buffer: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._action_thread: threading.Thread | None = None
        self._action_thread_stop_event = self._stop
        self.error_count = 0
        self.is_connected = False

    def connect(self) -> None:
        try:
            import zenoh
        except ImportError as error:
            raise ImportError(
                "NorthZmqEnv requires eclipse-zenoh (pip install eclipse-zenoh)"
            ) from error
        config = self._zenoh_config
        if config is None:
            config = zenoh.Config()
        elif isinstance(config, (str, pathlib.Path)):
            config = zenoh.Config.from_file(str(config))
        self._session = zenoh.open(config)
        self._publisher = self._session.declare_publisher(self.action_topic)
        self._subscriber = self._session.declare_subscriber(
            self.observation_topic,
            self._on_observation,
        )
        self.is_connected = True
        logger.info(
            "NorthZmqEnv connected obs=%s action=%s",
            self.observation_topic,
            self.action_topic,
        )

    def disconnect(self) -> None:
        self.shutdown()
        for handle in (self._subscriber, self._publisher, self._session):
            if handle is None:
                continue
            close = getattr(handle, "close", None) or getattr(handle, "undeclare", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass
        self._subscriber = None
        self._publisher = None
        self._session = None
        self.is_connected = False

    def _on_observation(self, sample: Any) -> None:
        try:
            obs = parse_north_observation(
                _sample_payload_bytes(sample),
                decode_images=self._decode_images,
            )
            with self._obs_lock:
                self._latest_obs = obs
                self._obs_seq += 1
        except Exception:  # noqa: BLE001
            self.error_count += 1
            logger.exception("failed to parse NorthObservation")

    def get_observation(self) -> dict[str, Any] | None:
        with self._obs_lock:
            return None if self._latest_obs is None else dict(self._latest_obs)

    def action_buffer_len(self) -> int:
        with self._action_lock:
            return len(self._action_buffer)

    def clear_action_and_history(self) -> None:
        with self._action_lock:
            self._action_buffer.clear()

    def _start_action_thread(self) -> None:
        if self._action_thread is not None and self._action_thread.is_alive():
            return
        self._stop.clear()
        self._action_thread = threading.Thread(target=self._action_sender_loop, daemon=True)
        self._action_thread.start()

    def _action_sender_loop(self) -> None:
        last_send_time = None
        while not self._action_thread_stop_event.is_set():
            if last_send_time is not None:
                sleep_time = self.action_pub_duration - (time.perf_counter() - last_send_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            last_send_time = time.perf_counter()
            with self._action_lock:
                if not self._action_buffer:
                    continue
                raw_action = self._action_buffer.pop(0)
            try:
                self._send_action(raw_action)
            except Exception:  # noqa: BLE001
                self.error_count += 1
                logger.exception("failed to publish UhrActionBundle")

    def _send_action(self, step_action: Mapping[str, Any]) -> None:
        if not self.is_connected or self._publisher is None:
            raise RuntimeError("Zenoh not connected")
        bundle = sdk_step_action_to_uhr_bundle(step_action, language=self.language)
        self._publisher.put(encode_uhr_action_bundle(bundle))

    def _send_actions(
        self,
        actions_dict: Mapping[str, list[list[float]]],
        *,
        observation: Mapping[str, Any] | None = None,
    ) -> int:
        """Replace the paced buffer (harobotsDL ``_send_actions``)."""
        payload = dict(actions_dict)
        if (
            self.first_chunk_smooth_size > 0
            and observation is not None
            and self.action_buffer_len() == 0
        ):
            payload = apply_first_chunk_smooth(
                payload,
                observation,
                smooth_chunk_size=self.first_chunk_smooth_size,
            )
        steps = sdk_action_chunk_to_step_dicts(payload)
        with self._action_lock:
            self._action_buffer = steps
            return len(steps)

    def enqueue_single_step(self, step_sdk: Mapping[str, list[float]]) -> None:
        """Enqueue one control tick (OpenPI Runtime applies one broker step)."""
        chunk = {key: [list(values)] for key, values in step_sdk.items()}
        steps = sdk_action_chunk_to_step_dicts(chunk)
        with self._action_lock:
            self._action_buffer.extend(steps)

    def reset(self) -> None:
        self.clear_action_and_history()
        self._start_action_thread()

    def shutdown(self) -> None:
        self._action_thread_stop_event.set()
        if self._action_thread is not None:
            self._action_thread.join(timeout=2.0)
        self._action_thread = None

    def step(self, in_step_info: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """harobotsDL ``step``: clear and/or replace action buffer."""
        info = dict(in_step_info or {})
        clear = bool(info.get("clear", False))
        if clear:
            self.clear_action_and_history()
        if not clear:
            action_keys = [k for k in info if "/action" in k and not str(k).endswith("_raw")]
            if action_keys:
                actions = {k: info[k] for k in action_keys}
                obs = info.get("obs")
                self._send_actions(actions, observation=obs if isinstance(obs, Mapping) else None)
        return {"clear_action": clear, "buffer_len": self.action_buffer_len()}
