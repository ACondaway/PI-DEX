"""Minimal WebSocket probe for a running ``pi-dex-serve`` instance."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

import numpy as np


def probe_once(
    *,
    host: str,
    port: int,
    api_key: str | None = None,
    clock_domain: str = "unix_realtime",
    prompt: str = "probe",
) -> dict[str, Any]:
    """Connect, read metadata, send one synthetic observation, return summary."""
    import websockets.sync.client
    from openpi_client import msgpack_numpy

    uri = f"ws://{host}:{port}"
    headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None
    packer = msgpack_numpy.Packer()
    with websockets.sync.client.connect(
        uri,
        compression=None,
        max_size=32 * 1024 * 1024,
        additional_headers=headers,
        open_timeout=10,
    ) as conn:
        metadata = msgpack_numpy.unpackb(conn.recv())
        observation = {
            "state": np.zeros((4,), dtype=np.float32),
            "prompt": prompt,
            "observation_timestamp_ns": 1_000_000_000,
            "clock_domain": clock_domain,
        }
        # Prefer clock_domain declared by the server when present.
        pi_dex = metadata.get("pi_dex") if isinstance(metadata, dict) else None
        if isinstance(pi_dex, dict) and type(pi_dex.get("clock_domain")) is str:
            observation["clock_domain"] = pi_dex["clock_domain"]
        conn.send(packer.pack(observation))
        raw = conn.recv()
        if isinstance(raw, str):
            raise RuntimeError(f"server error: {raw}")
        result = msgpack_numpy.unpackb(raw)
    left = result["actions"]["left"]
    right = result["actions"]["right"]
    return {
        "ok": True,
        "host": host,
        "port": port,
        "session_id": (pi_dex or {}).get("session_id") if isinstance(pi_dex, dict) else None,
        "execution_horizon": (pi_dex or {}).get("execution_horizon") if isinstance(pi_dex, dict) else None,
        "left_shape": list(np.asarray(left).shape),
        "right_shape": list(np.asarray(right).shape),
        "server_timing": result.get("server_timing"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pi-dex-serve-probe")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--clock-domain", default="unix_realtime")
    parser.add_argument("--prompt", default="probe")
    args = parser.parse_args(list(argv) if argv is not None else None)
    summary = probe_once(
        host=args.host,
        port=args.port,
        api_key=args.api_key or None,
        clock_domain=args.clock_domain,
        prompt=args.prompt,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
