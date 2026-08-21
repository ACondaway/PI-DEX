"""Validated command-line boundary for representation-selectable PyTorch training.

This module deliberately does not implement a first-party Sharpa HDF5 dataset or
training loop yet. It validates the action representation and its FK dependency,
then delegates to an explicitly configured ``module:callable`` runner. A
mandatory cooperative completion handshake prevents an accidental successful
return unless the runner attests to a matching spec and, in Cartesian mode,
obtains FK through this context. Dynamic import remains behind argument validation
so invalid calls fail before OpenPI, PyTorch/CUDA, or server data initialization.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import sys
import threading
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any

from pi_dex.core.actions import ActionRepresentation
from pi_dex.core.spec import BimanualActionSpec

ForwardKinematicsFactory = Callable[[], object]


@dataclasses.dataclass
class _LaunchBinding:
    action_spec: BimanualActionSpec | None = None
    kinematics: object | None = None
    lock: threading.RLock = dataclasses.field(
        default_factory=threading.RLock,
        repr=False,
        compare=False,
    )


@dataclasses.dataclass(frozen=True)
class PytorchTrainingLaunchContext:
    """Stable representation contract passed to an external training runner.

    Args:
        action_representation: Selected 31D Cartesian or 29D joint layout.
        runner_args: Unparsed arguments that appeared after the CLI ``--``
            separator. The external runner owns their schema and interpretation.
        fk_provider_factory: Zero-argument factory for a calibrated FK provider.
            It is required for Cartesian actions and forbidden for joint actions.

    The repository does not yet provide a complete HDF5 training runner. A
    configured runner must build the dataset, OpenPI config, model, optimizer,
    checkpoint lifecycle, and server-side validation around this context. Before
    returning, it must bind the exact spec used by the pipeline with
    :meth:`bind_action_spec`. Cartesian runners must then obtain their provider
    with :meth:`create_kinematics`; the launcher rejects a success return when
    either handshake is missing.

    Raises:
        TypeError: If fields have invalid exact types or the FK factory is not
            callable.
        ValueError: If FK configuration conflicts with the representation.

    The internal completion binding is excluded from equality and remains
    launcher-owned mutable state despite the frozen public dataclass surface.
    """

    action_representation: ActionRepresentation
    runner_args: tuple[str, ...] = ()
    fk_provider_factory: ForwardKinematicsFactory | None = None
    _binding: _LaunchBinding = dataclasses.field(
        default_factory=_LaunchBinding,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.action_representation, ActionRepresentation):
            raise TypeError(
                "action_representation: expected ActionRepresentation, "
                f"got {type(self.action_representation).__name__}"
            )
        if type(self.runner_args) is not tuple:
            raise TypeError(
                f"runner_args: expected tuple[str, ...], got {type(self.runner_args).__name__}"
            )
        for index, value in enumerate(self.runner_args):
            if type(value) is not str:
                raise TypeError(
                    f"runner_args[{index}]: expected str, got {type(value).__name__}"
                )
        if self.requires_forward_kinematics:
            if self.fk_provider_factory is None:
                raise ValueError(
                    "fk_provider_factory: required for cartesian_31d training"
                )
            if not callable(self.fk_provider_factory):
                raise TypeError("fk_provider_factory: expected a callable")
        elif self.fk_provider_factory is not None:
            raise ValueError("fk_provider_factory: forbidden for joint_29d training")

    @property
    def logical_action_dim(self) -> int:
        """Return the selected unpadded per-hand action width, 31 or 29."""
        return self.action_representation.logical_action_dim

    @property
    def requires_forward_kinematics(self) -> bool:
        """Return whether the selected action layout requires calibrated FK."""
        return self.action_representation is ActionRepresentation.CARTESIAN_31D

    @property
    def action_spec(self) -> BimanualActionSpec:
        """Return the representation-matched spec bound by the runner.

        Raises:
            RuntimeError: If the runner has not called :meth:`bind_action_spec`.
        """
        with self._binding.lock:
            action_spec = self._binding.action_spec
            if action_spec is None:
                raise RuntimeError(
                    "runner must bind its BimanualActionSpec through "
                    "context.bind_action_spec(spec)"
                )
            return action_spec

    def bind_action_spec(self, spec: BimanualActionSpec) -> BimanualActionSpec:
        """Bind and return the validated spec that every training layer must use.

        The returned frozen copy is the canonical spec for dataset derivation,
        OpenPI transforms, PyTorch loss, checkpoint metadata, and deployment.

        Raises:
            TypeError: If ``spec`` is not a :class:`BimanualActionSpec`.
            ValueError: If its representation differs from the CLI selection or
                a different spec was already bound.
        """
        if not isinstance(spec, BimanualActionSpec):
            raise TypeError(
                f"spec: expected BimanualActionSpec, got {type(spec).__name__}"
            )
        validated_spec = dataclasses.replace(spec)
        if validated_spec.action_representation is not self.action_representation:
            raise ValueError(
                "spec.action_representation conflicts with launcher selection: "
                f"expected {self.action_representation.value!r}, "
                f"got {validated_spec.action_representation.value!r}"
            )
        with self._binding.lock:
            bound_spec = self._binding.action_spec
            if bound_spec is not None:
                if bound_spec != validated_spec:
                    raise ValueError("runner attempted to bind a different BimanualActionSpec")
                return bound_spec
            self._binding.action_spec = validated_spec
            return validated_spec

    def create_kinematics(self) -> object:
        """Create at most one Cartesian FK provider after binding the spec.

        Returns:
            The non-``None`` object returned by ``fk_provider_factory``. The data
            boundary subsequently validates its full calibration metadata.

        Raises:
            RuntimeError: If joint training attempts to create an FK provider.
            RuntimeError: If the Cartesian runner has not bound its action spec.
            ValueError: If the Cartesian factory returns ``None``.
        """
        with self._binding.lock:
            if not self.requires_forward_kinematics:
                raise RuntimeError("joint_29d training must not create a forward-kinematics provider")
            if self._binding.action_spec is None:
                raise RuntimeError(
                    "cartesian_31d training must bind its BimanualActionSpec before "
                    "creating the forward-kinematics provider"
                )
            if self._binding.kinematics is not None:
                return self._binding.kinematics
            factory = self.fk_provider_factory
            if factory is None:
                raise AssertionError("validated Cartesian launch context lost its FK factory")
            kinematics = factory()
            if kinematics is None:
                raise ValueError("fk_provider_factory: expected a non-None provider")
            self._binding.kinematics = kinematics
            return kinematics

    def validate_runner_completion(self) -> None:
        """Reject a successful runner return without the required handshakes."""
        with self._binding.lock:
            if self._binding.action_spec is None:
                raise RuntimeError(
                    "runner returned without binding its BimanualActionSpec through "
                    "context.bind_action_spec(spec)"
                )
            if self.requires_forward_kinematics and self._binding.kinematics is None:
                raise RuntimeError(
                    "cartesian_31d runner returned without obtaining the configured "
                    "forward-kinematics provider through context.create_kinematics()"
                )


TrainingRunner = Callable[[PytorchTrainingLaunchContext], int | None]


@dataclasses.dataclass(frozen=True)
class _CallableReference:
    module_name: str
    attribute_path: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class _ParsedLaunchArguments:
    action_representation: ActionRepresentation
    runner_reference: _CallableReference
    fk_factory_reference: _CallableReference | None
    runner_args: tuple[str, ...]


def main(argv: Sequence[str] | None = None) -> int:
    """Validate launcher arguments, dynamically load a runner, and execute it.

    Args:
        argv: Optional argument sequence excluding the executable name. ``None``
            reads ``sys.argv[1:]``. Arguments after ``--`` are passed unchanged
            to the runner through :class:`PytorchTrainingLaunchContext`.

    Returns:
        Zero when the runner returns ``None``; otherwise its exact integer exit
        code in ``[0, 255]``. ``SystemExit(None)`` and ``SystemExit(code)`` use
        the same rules. Any path observable as zero is accepted only after the
        runner binds a matching action spec and, for Cartesian mode, obtains FK.

    Raises:
        SystemExit: If launcher arguments are missing, unknown, or inconsistent.
        ImportError: If a configured module cannot be imported.
        AttributeError: If a configured callable path does not exist.
        TypeError: If a configured target is not callable or the runner returns
            anything other than ``None`` or an exact ``int`` exit code.
        ValueError: If a runner exit code is outside ``[0, 255]``.
        RuntimeError: If a successful runner return omits a required handshake.
    """
    parsed = _parse_launch_arguments(argv)
    fk_provider_factory: ForwardKinematicsFactory | None = None
    if parsed.fk_factory_reference is not None:
        fk_provider_factory = _resolve_callable(
            parsed.fk_factory_reference,
            field_name="--fk-provider-factory",
        )
    runner = _resolve_callable(parsed.runner_reference, field_name="--runner")
    context = PytorchTrainingLaunchContext(
        action_representation=parsed.action_representation,
        runner_args=parsed.runner_args,
        fk_provider_factory=fk_provider_factory,
    )
    try:
        result = runner(context)
    except SystemExit as error:
        result = _runner_system_exit_code(error)
    if result is None:
        context.validate_runner_completion()
        return 0
    _validate_runner_exit_code(result, field_name="runner return")
    if result == 0:
        context.validate_runner_completion()
    return result


def _runner_system_exit_code(error: SystemExit) -> int:
    code = error.code
    if code is None:
        return 0
    _validate_runner_exit_code(code, field_name="runner SystemExit code")
    return code


def _validate_runner_exit_code(value: object, *, field_name: str) -> None:
    if type(value) is not int:
        raise TypeError(
            f"{field_name}: expected an exact int in [0, 255], "
            f"got {type(value).__name__}"
        )
    if not 0 <= value <= 255:
        raise ValueError(f"{field_name}: expected an integer in [0, 255], got {value}")


def _parse_launch_arguments(argv: Sequence[str] | None) -> _ParsedLaunchArguments:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    launcher_arguments, runner_args = _split_runner_arguments(raw_arguments)
    parser = _create_argument_parser()
    namespace = parser.parse_args(launcher_arguments)
    action_representation = ActionRepresentation(namespace.action_representation)

    if (
        action_representation is ActionRepresentation.CARTESIAN_31D
        and namespace.fk_provider_factory is None
    ):
        parser.error(
            "--fk-provider-factory is required when "
            "--action-representation=cartesian_31d"
        )
    if (
        action_representation is ActionRepresentation.JOINT_29D
        and namespace.fk_provider_factory is not None
    ):
        parser.error(
            "--fk-provider-factory is forbidden when "
            "--action-representation=joint_29d"
        )

    try:
        runner_reference = _parse_callable_reference(namespace.runner, field_name="--runner")
        fk_factory_reference = (
            None
            if namespace.fk_provider_factory is None
            else _parse_callable_reference(
                namespace.fk_provider_factory,
                field_name="--fk-provider-factory",
            )
        )
    except ValueError as error:
        parser.error(str(error))
    return _ParsedLaunchArguments(
        action_representation=action_representation,
        runner_reference=runner_reference,
        fk_factory_reference=fk_factory_reference,
        runner_args=runner_args,
    )


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pi-dex-train-pytorch",
        description=(
            "Validate a PI-DEX action representation and delegate to an external "
            "PyTorch training runner. This repository does not yet ship a complete "
            "Sharpa HDF5 runner."
        ),
    )
    parser.add_argument(
        "--action-representation",
        required=True,
        choices=tuple(representation.value for representation in ActionRepresentation),
        help="Per-hand training target layout.",
    )
    parser.add_argument(
        "--runner",
        required=True,
        help="External training entry point as module:callable.",
    )
    parser.add_argument(
        "--fk-provider-factory",
        help=(
            "Zero-argument module:callable factory required only for "
            "cartesian_31d training."
        ),
    )
    return parser


def _split_runner_arguments(arguments: list[str]) -> tuple[list[str], tuple[str, ...]]:
    try:
        separator_index = arguments.index("--")
    except ValueError:
        return arguments, ()
    return arguments[:separator_index], tuple(arguments[separator_index + 1 :])


def _parse_callable_reference(value: str, *, field_name: str) -> _CallableReference:
    if value.count(":") != 1:
        raise ValueError(f"{field_name}: expected module:callable, got {value!r}")
    module_name, attribute_name = value.split(":", maxsplit=1)
    module_parts = module_name.split(".")
    attribute_path = tuple(attribute_name.split("."))
    if not module_parts or not all(part.isidentifier() for part in module_parts):
        raise ValueError(f"{field_name}: invalid module path {module_name!r}")
    if not attribute_path or not all(part.isidentifier() for part in attribute_path):
        raise ValueError(f"{field_name}: invalid callable path {attribute_name!r}")
    return _CallableReference(module_name=module_name, attribute_path=attribute_path)


def _resolve_callable(reference: _CallableReference, *, field_name: str) -> Any:
    target: Any = importlib.import_module(reference.module_name)
    resolved_path = reference.module_name
    for attribute_name in reference.attribute_path:
        resolved_path = f"{resolved_path}.{attribute_name}"
        target = _resolve_attribute(target, attribute_name, field_name=field_name, resolved_path=resolved_path)
    if not callable(target):
        raise TypeError(f"{field_name}: resolved target {resolved_path!r} is not callable")
    return target


def _resolve_attribute(
    target: Any,
    attribute_name: str,
    *,
    field_name: str,
    resolved_path: str,
) -> Any:
    try:
        return getattr(target, attribute_name)
    except AttributeError:
        raise AttributeError(f"{field_name}: callable path not found: {resolved_path}") from None


if __name__ == "__main__":
    raise SystemExit(main())
