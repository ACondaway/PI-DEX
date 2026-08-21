"""Controlled JAX/Orbax → PyTorch converter wrapper for official ``pi05_base``.

This is the first-party wrapper required by handoff phase 2 / validation A.2.
It does **not** claim parity PASS by itself: callers must still run a registered
JAX↔PyTorch vector-field parity harness after conversion.

Usage (from the OpenPI uv environment with ``pi-dex`` editable-installed)::

    python -m pi_dex.weights.convert_pi05 \\
      --checkpoint-root /path/to/pi05_base \\
      --output /path/to/pi05_base-pytorch-bfloat16 \\
      --physical-horizon 8 \\
      --precision bfloat16
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import tempfile
import time
from typing import Any, Literal

from pi_dex.core.actions import ActionRepresentation
from pi_dex.training.checkpoints import MODEL_WEIGHTS_FILENAME
from pi_dex.training.openpi_integration import create_pi05_model_config
from pi_dex.weights.pi05_weights import file_sha256
from pi_dex.weights.pi05_weights import load_verified_pi05_base
from pi_dex.core.spec import ActionMode
from pi_dex.core.spec import ActionTimebase
from pi_dex.core.spec import BimanualActionSpec
from pi_dex.core.spec import HandNormalization

_OFFICIAL_URI = "gs://openpi-assets/checkpoints/pi05_base"
_EXPECTED_PALIGEMMA = "gemma_2b"
_EXPECTED_EXPERT = "gemma_300m"


def main(argv: list[str] | None = None) -> int:
    """CLI entry for controlled ``pi05_base`` conversion."""
    args = _parse_args(argv)
    convert_pi05_base(
        checkpoint_root=pathlib.Path(args.checkpoint_root),
        output_dir=pathlib.Path(args.output),
        physical_horizon=args.physical_horizon,
        precision=args.precision,
        timebase=ActionTimebase(args.timebase),
        control_frequency_hz=args.control_frequency_hz,
        expected_source_manifest=pathlib.Path(args.expected_source_manifest) if args.expected_source_manifest else None,
        skip_parity=args.skip_parity,
    )
    return 0


def convert_pi05_base(
    *,
    checkpoint_root: pathlib.Path,
    output_dir: pathlib.Path,
    physical_horizon: int,
    precision: Literal["bfloat16", "float32"] = "bfloat16",
    timebase: ActionTimebase = ActionTimebase.RAW_CONTROL_60_HZ,
    control_frequency_hz: float = 59.4,
    expected_source_manifest: pathlib.Path | None = None,
    skip_parity: bool = False,
) -> pathlib.Path:
    """Convert a verified JAX ``pi05_base`` root into a new PyTorch directory.

    Args:
        checkpoint_root: Official checkpoint root containing ``params/`` (not the
            ``params`` directory itself).
        output_dir: Final publish path. Must not already exist.
        physical_horizon: Site ``K`` used to build ``create_pi05_model_config``.
        precision: Target PyTorch dtype for saved weights.
        timebase / control_frequency_hz: Spec fields required to construct a
            joint_29d model config fingerprint matching later training.
        expected_source_manifest: Optional JSON manifest of relative path →
            ``{size, sha256}`` that must match the on-disk source before convert.
        skip_parity: If true, record that parity was skipped. Conversion still
            requires strict reload; A.2 remains incomplete until parity runs.

    Returns:
        The published ``output_dir``.

    Raises:
        FileExistsError: If ``output_dir`` already exists.
        FileNotFoundError: If ``params/`` is missing.
        ValueError: If variants, manifests, or reload coverage fail.
    """
    if output_dir.exists():
        raise FileExistsError(f"output: refuse to overwrite existing path {output_dir}")
    params_dir = checkpoint_root / "params"
    if not params_dir.is_dir():
        raise FileNotFoundError(
            f"checkpoint_root: expected params/ under {checkpoint_root} (pass the pi05_base root, not .../params)"
        )
    if expected_source_manifest is not None:
        _verify_source_manifest(checkpoint_root, expected_source_manifest)

    spec = _joint_spec_for_convert(
        physical_horizon=physical_horizon,
        timebase=timebase,
        control_frequency_hz=control_frequency_hz,
    )
    model_config = create_pi05_model_config(
        spec,
        dtype=precision,
        paligemma_variant=_EXPECTED_PALIGEMMA,
        action_expert_variant=_EXPECTED_EXPERT,
        pytorch_compile_mode=None,
    )
    if model_config.paligemma_variant != _EXPECTED_PALIGEMMA:
        raise ValueError("converter refuses non-gemma_2b paligemma variants")
    if model_config.action_expert_variant != _EXPECTED_EXPERT:
        raise ValueError("converter refuses non-gemma_300m action expert variants")

    converter_path = _locate_vendored_converter()
    converter = _load_converter_module(converter_path)
    _patch_adarms_selection(converter)

    staging_parent = output_dir.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(tempfile.mkdtemp(prefix=".pi05_convert_staging_", dir=str(staging_parent)))
    started = time.time()
    try:
        converter.convert_pi0_checkpoint(
            str(checkpoint_root.resolve()),
            precision,
            str(staging),
            model_config,
        )
        # Upstream may copy unrelated assets; PI-DEX converted base must not carry them.
        assets_dir = staging / "assets"
        if assets_dir.exists():
            shutil.rmtree(assets_dir)

        reload_model = _instantiate_pytorch_model(model_config)
        load_meta = load_verified_pi05_base(reload_model, staging)
        from safetensors.torch import save_model

        save_model(reload_model, str(staging / MODEL_WEIGHTS_FILENAME))
        weights_sha256 = file_sha256(staging / MODEL_WEIGHTS_FILENAME)

        provenance = {
            "schema_version": 1,
            "artifact_kind": "pi05_base_pytorch_init",
            "official_uri": _OFFICIAL_URI,
            "checkpoint_root": str(checkpoint_root.resolve()),
            "params_dir": str(params_dir.resolve()),
            "precision": precision,
            "physical_horizon": physical_horizon,
            "model_action_horizon": spec.model_action_horizon,
            "model_config": {
                "dtype": getattr(model_config, "dtype", None),
                "pi05": True,
                "discrete_state_input": getattr(model_config, "discrete_state_input", None),
                "action_dim": getattr(model_config, "action_dim", None),
                "action_horizon": getattr(model_config, "action_horizon", None),
                "max_token_len": getattr(model_config, "max_token_len", None),
                "paligemma_variant": model_config.paligemma_variant,
                "action_expert_variant": model_config.action_expert_variant,
            },
            "vendored_converter": {
                "path": str(converter_path),
                "sha256": file_sha256(converter_path),
            },
            "weights_file": MODEL_WEIGHTS_FILENAME,
            "weights_sha256": weights_sha256,
            "source_tree_sha256": _directory_manifest_digest(checkpoint_root),
            "elapsed_seconds": time.time() - started,
            "parity": {
                "status": "skipped" if skip_parity else "required_not_run",
                "note": (
                    "Conversion wrapper completed strict reload only. "
                    "A.2 PASS still requires registered JAX↔PyTorch vector-field parity."
                ),
            },
            "not_a_deployable_checkpoint": True,
        }
        (staging / "pi_dex_convert_provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "manifest.sha256.json").write_text(
            json.dumps(_file_manifest(staging), indent=2) + "\n",
            encoding="utf-8",
        )
        os.rename(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(json.dumps({"status": "converted", "output": str(output_dir), "weights_sha256": weights_sha256}, indent=2))
    if not skip_parity:
        print(
            "WARNING: parity harness was not run; do not mark validation A.2 as PASS.",
            file=sys.stderr,
        )
    return output_dir


def _joint_spec_for_convert(
    *, physical_horizon: int, timebase: ActionTimebase, control_frequency_hz: float
) -> BimanualActionSpec:
    left_arm = tuple(f"left_arm_j{i}" for i in range(7))
    right_arm = tuple(f"right_arm_j{i}" for i in range(7))
    left_hand = tuple(f"left_hand_j{i}" for i in range(22))
    right_hand = tuple(f"right_hand_j{i}" for i in range(22))
    return BimanualActionSpec(
        physical_horizon=physical_horizon,
        timebase=timebase,
        control_frequency_hz=control_frequency_hz,
        robot_id="convert_placeholder",
        embodiment_version="convert_placeholder",
        coordinate_frame=None,
        action_mode=ActionMode.ABSOLUTE,
        action_representation=ActionRepresentation.JOINT_29D,
        hand_normalization=HandNormalization.PER_HAND,
        rotation_6d_convention=None,
        kinematics_calibration_version=None,
        command_semantics_version="convert_placeholder",
        left_arm_joint_order=left_arm,
        right_arm_joint_order=right_arm,
        left_hand_joint_order=left_hand,
        right_hand_joint_order=right_hand,
        hand_mapping_version="convert_placeholder",
        left_wrist_link=None,
        right_wrist_link=None,
        clock_domain="unix_realtime",
        max_group_timestamp_skew_ms=2.0,
        max_alignment_timestamp_error_ms=2.0,
        max_control_period_error_ms=8.0,
        max_observation_age_ms=50.0,
        max_command_lead_ms=25.0,
    )


def _locate_vendored_converter() -> pathlib.Path:
    repo_openpi = (
        pathlib.Path(__file__).resolve().parents[2] / "openpi" / "examples" / "convert_jax_model_to_pytorch.py"
    )
    if repo_openpi.is_file():
        return repo_openpi
    raise FileNotFoundError("vendored converter: openpi/examples/convert_jax_model_to_pytorch.py not found")


def _load_converter_module(path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location("pi_dex_vendored_jax_to_pytorch", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load converter module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _patch_adarms_selection(converter: Any) -> None:
    """Force AdaRMS mapping from ``pi05`` bool, not path substring heuristics."""
    original = converter.slice_gemma_state_dict

    def patched(state_dict, config, *, num_expert, checkpoint_dir, pi05):
        del checkpoint_dir
        # Upstream checks `"pi05" in checkpoint_dir`; pass a synthetic path that
        # matches the boolean so Dense vs scale branches stay consistent.
        synthetic = "pi05" if pi05 else "pi0"
        return original(state_dict, config, num_expert=num_expert, checkpoint_dir=synthetic, pi05=pi05)

    converter.slice_gemma_state_dict = patched


def _instantiate_pytorch_model(model_config: Any) -> Any:
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    return PI0Pytorch(model_config)


def _verify_source_manifest(root: pathlib.Path, manifest_path: pathlib.Path) -> None:
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(expected, dict):
        raise ValueError("expected_source_manifest: expected a JSON object")
    actual = _file_manifest(root)
    if actual != expected:
        raise ValueError(
            f"source manifest mismatch versus expected_source_manifest ({manifest_path}); refusing conversion"
        )


def _file_manifest(root: pathlib.Path) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        manifest[relative] = {"size": path.stat().st_size, "sha256": file_sha256(path)}
    return manifest


def _directory_manifest_digest(root: pathlib.Path) -> str:
    payload = json.dumps(_file_manifest(root), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m pi_dex.weights.convert_pi05")
    parser.add_argument("--checkpoint-root", required=True, help="JAX pi05_base root containing params/")
    parser.add_argument("--output", required=True, help="New directory for model.safetensors publish")
    parser.add_argument("--physical-horizon", type=int, required=True, help="Site K for model config")
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--timebase", default=ActionTimebase.RAW_CONTROL_60_HZ.value)
    parser.add_argument("--control-frequency-hz", type=float, default=59.4)
    parser.add_argument("--expected-source-manifest", default="")
    parser.add_argument(
        "--skip-parity",
        action="store_true",
        help="Allow conversion without running parity; A.2 remains incomplete",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
