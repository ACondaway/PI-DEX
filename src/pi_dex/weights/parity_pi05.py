"""JAX↔PyTorch vector-field parity harness for converted ``pi05_base``.

Registers tolerances **before** computing errors. Compares ``sample_actions``
outputs under identical post-transform observations and noise (eval /
no-augmentation). This is required for validation A.2; conversion alone is not
PASS.

Example::

    python -m pi_dex.weights.parity_pi05 \\
      --jax-checkpoint-root .../pi05_base \\
      --pytorch-weight-path .../pi05_base-pytorch-bfloat16-K8 \\
      --physical-horizon 8 \\
      --output-json .../parity.json
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import pathlib
import time
from typing import Any

import numpy as np

from pi_dex.core.actions import ActionRepresentation
from pi_dex.training.openpi_integration import create_pi05_model_config
from pi_dex.weights.pi05_weights import load_verified_pi05_base
from pi_dex.core.spec import ActionMode
from pi_dex.core.spec import ActionTimebase
from pi_dex.core.spec import BimanualActionSpec
from pi_dex.core.spec import HandNormalization

# Registered before any numeric comparison (handoff / A.2). Do not loosen after seeing results.
DEFAULT_ATOL = 5.0e-2
DEFAULT_RTOL = 5.0e-2
DEFAULT_MAX_ABS = 1.5e-1
FIXTURE_SEED = 0
PROMPT = "parity harness fixed prompt"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_parity(
        jax_checkpoint_root=pathlib.Path(args.jax_checkpoint_root),
        pytorch_weight_path=pathlib.Path(args.pytorch_weight_path),
        physical_horizon=args.physical_horizon,
        precision=args.precision,
        device=args.device,
        num_steps=args.num_steps,
        atol=args.atol,
        rtol=args.rtol,
        max_abs=args.max_abs,
        expected_base_sha256=args.expected_base_sha256 or None,
    )
    if args.output_json:
        path = pathlib.Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


def run_parity(
    *,
    jax_checkpoint_root: pathlib.Path,
    pytorch_weight_path: pathlib.Path,
    physical_horizon: int,
    precision: str = "bfloat16",
    device: str = "cuda:0",
    num_steps: int = 10,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    max_abs: float = DEFAULT_MAX_ABS,
    expected_base_sha256: str | None = None,
) -> dict[str, Any]:
    """Compare JAX source and PyTorch converted ``sample_actions`` trajectories."""
    # Freeze acceptance criteria before any forward pass.
    criteria = {
        "atol": float(atol),
        "rtol": float(rtol),
        "max_abs": float(max_abs),
        "num_steps": int(num_steps),
        "fixture_seed": FIXTURE_SEED,
        "prompt": PROMPT,
        "precision": precision,
    }

    import jax
    import jax.numpy as jnp
    from openpi.models import model as openpi_model
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    import torch

    started = time.time()
    spec = _joint_spec(physical_horizon)
    model_config = create_pi05_model_config(spec, dtype=precision, pytorch_compile_mode=None)

    params_dir = jax_checkpoint_root / "params"
    if not params_dir.is_dir():
        raise FileNotFoundError(f"jax_checkpoint_root: expected params/ under {jax_checkpoint_root}")

    # --- fixtures (shared numpy, then framework copies) ---
    rng = np.random.default_rng(FIXTURE_SEED)
    batch_size = 1
    horizon = model_config.action_horizon
    action_dim = model_config.action_dim
    # Model fake-obs state width is action_dim (32); conversion parity is not Sharpa state_dim.
    state_np = rng.normal(size=(batch_size, action_dim)).astype(np.float32) * 0.1
    noise_np = rng.normal(size=(batch_size, horizon, action_dim)).astype(np.float32)
    images_np = {
        key: (rng.random(size=(batch_size, 224, 224, 3), dtype=np.float32) * 2.0 - 1.0)
        for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    }
    masks_np = {key: np.ones((batch_size,), dtype=np.bool_) for key in images_np}
    tokenizer = PaligemmaTokenizer(model_config.max_token_len)
    tokens, token_mask = tokenizer.tokenize(PROMPT, state_np[0])
    tokens_np = np.stack([tokens], axis=0).astype(np.int32)
    token_mask_np = np.stack([token_mask], axis=0).astype(np.bool_)

    fixture_hash = hashlib.sha256(
        json.dumps(
            {
                "state": state_np.tolist(),
                "noise": noise_np.tolist(),
                "tokens": tokens_np.tolist(),
                "token_mask": token_mask_np.tolist(),
                "images_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in images_np.items()},
            },
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    # --- JAX model ---
    params = openpi_model.restore_params(params_dir, dtype=jnp.bfloat16 if precision == "bfloat16" else jnp.float32)
    jax_model = model_config.load(params)
    jax_obs = openpi_model.Observation(
        images={k: jnp.asarray(v) for k, v in images_np.items()},
        image_masks={k: jnp.asarray(v) for k, v in masks_np.items()},
        state=jnp.asarray(state_np),
        tokenized_prompt=jnp.asarray(tokens_np),
        tokenized_prompt_mask=jnp.asarray(token_mask_np),
    )
    jax_noise = jnp.asarray(noise_np)
    jax_out = np.asarray(
        jax_model.sample_actions(jax.random.key(0), jax_obs, noise=jax_noise, num_steps=num_steps),
        dtype=np.float32,
    )
    # Free JAX model before loading the PyTorch twin (A.2 allows sequential runs for VRAM).
    del jax_model, params, jax_obs, jax_noise
    with contextlib.suppress(Exception):
        jax.clear_caches()

    # --- PyTorch model ---
    torch_device = torch.device(device)
    pt_model = PI0Pytorch(model_config).to(torch_device)
    load_meta = load_verified_pi05_base(
        pt_model,
        pytorch_weight_path,
        expected_weights_sha256=expected_base_sha256,
    )
    pt_model.eval()
    pt_images = {k: torch.as_tensor(np.transpose(v, (0, 3, 1, 2)), device=torch_device) for k, v in images_np.items()}
    pt_obs = openpi_model.Observation(
        images=pt_images,
        image_masks={k: torch.as_tensor(v, device=torch_device) for k, v in masks_np.items()},
        state=torch.as_tensor(state_np, device=torch_device),
        tokenized_prompt=torch.as_tensor(tokens_np, device=torch_device),
        tokenized_prompt_mask=torch.as_tensor(token_mask_np, device=torch_device),
    )
    pt_noise = torch.as_tensor(noise_np, device=torch_device)
    with torch.no_grad():
        pt_out_t = pt_model.sample_actions(torch_device, pt_obs, noise=pt_noise, num_steps=num_steps)
    pt_out = pt_out_t.detach().float().cpu().numpy()

    abs_err = np.abs(jax_out - pt_out)
    rel_err = abs_err / np.maximum(np.abs(jax_out), 1e-6)
    metrics = {
        "max_abs_error": float(abs_err.max()),
        "mean_abs_error": float(abs_err.mean()),
        "max_rel_error": float(rel_err.max()),
        "mean_rel_error": float(rel_err.mean()),
        "jax_finite": bool(np.isfinite(jax_out).all()),
        "pytorch_finite": bool(np.isfinite(pt_out).all()),
        "allclose": bool(np.allclose(jax_out, pt_out, atol=atol, rtol=rtol)),
    }
    passed = (
        metrics["jax_finite"]
        and metrics["pytorch_finite"]
        and metrics["allclose"]
        and metrics["max_abs_error"] <= max_abs
    )

    return {
        "schema_version": 1,
        "passed": passed,
        "criteria": criteria,
        "metrics": metrics,
        "fixture_sha256": fixture_hash,
        "jax_checkpoint_root": str(jax_checkpoint_root.resolve()),
        "pytorch_weight_path": str(pytorch_weight_path.resolve()),
        "pytorch_weights_sha256": load_meta["weights_sha256"],
        "model_config": {
            "action_dim": action_dim,
            "action_horizon": horizon,
            "max_token_len": model_config.max_token_len,
            "pi05": True,
            "dtype": precision,
            "paligemma_variant": model_config.paligemma_variant,
            "action_expert_variant": model_config.action_expert_variant,
        },
        "elapsed_seconds": time.time() - started,
        "vendored_note": "Compares sample_actions trajectories (integrated vector field) under train=False.",
    }


def _joint_spec(physical_horizon: int) -> BimanualActionSpec:
    left_arm = tuple(f"left_arm_j{i}" for i in range(7))
    right_arm = tuple(f"right_arm_j{i}" for i in range(7))
    left_hand = tuple(f"left_hand_j{i}" for i in range(22))
    right_hand = tuple(f"right_hand_j{i}" for i in range(22))
    return BimanualActionSpec(
        physical_horizon=physical_horizon,
        timebase=ActionTimebase.RAW_CONTROL_60_HZ,
        control_frequency_hz=59.4,
        robot_id="parity",
        embodiment_version="parity",
        coordinate_frame=None,
        action_mode=ActionMode.ABSOLUTE,
        action_representation=ActionRepresentation.JOINT_29D,
        hand_normalization=HandNormalization.PER_HAND,
        rotation_6d_convention=None,
        kinematics_calibration_version=None,
        command_semantics_version="parity",
        left_arm_joint_order=left_arm,
        right_arm_joint_order=right_arm,
        left_hand_joint_order=left_hand,
        right_hand_joint_order=right_hand,
        hand_mapping_version="parity",
        left_wrist_link=None,
        right_wrist_link=None,
        clock_domain="unix_realtime",
        max_group_timestamp_skew_ms=2.0,
        max_alignment_timestamp_error_ms=2.0,
        max_control_period_error_ms=8.0,
        max_observation_age_ms=50.0,
        max_command_lead_ms=25.0,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m pi_dex.weights.parity_pi05")
    parser.add_argument("--jax-checkpoint-root", required=True)
    parser.add_argument("--pytorch-weight-path", required=True)
    parser.add_argument("--physical-horizon", type=int, required=True)
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--max-abs", type=float, default=DEFAULT_MAX_ABS)
    parser.add_argument("--expected-base-sha256", default="")
    parser.add_argument("--output-json", default="")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
