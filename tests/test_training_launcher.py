import concurrent.futures
import dataclasses
import threading
import types
from collections.abc import Callable

import pytest

import pi_dex.training.training_launcher as training_launcher
from pi_dex.core.actions import ActionRepresentation
from pi_dex.core.spec import BimanualActionSpec
from pi_dex.training.training_launcher import PytorchTrainingLaunchContext
from pi_dex.training.training_launcher import main
from tests.helpers import spec_for_representation


def install_import_registry(monkeypatch, modules: dict[str, types.ModuleType], calls: list[str]) -> None:
    def import_module(module_name: str) -> types.ModuleType:
        calls.append(module_name)
        try:
            return modules[module_name]
        except KeyError:
            raise AssertionError(f"unexpected dynamic import: {module_name}") from None

    monkeypatch.setattr(training_launcher.importlib, "import_module", import_module)


def make_provider_factory(provider: object) -> Callable[[], object]:
    def create_kinematics() -> object:
        return provider

    return create_kinematics


@pytest.mark.parametrize(
    "arguments",
    [
        [
            "--action-representation",
            "cartesian_31d",
            "--runner",
            "runner_module:run",
        ],
        [
            "--action-representation",
            "joint_29d",
            "--runner",
            "runner_module:run",
            "--fk-provider-factory",
            "fk_module:create",
        ],
    ],
)
def test_representation_fk_combination_fails_before_dynamic_import(
    monkeypatch,
    arguments: list[str],
) -> None:
    import_calls: list[str] = []
    install_import_registry(monkeypatch, {}, import_calls)

    with pytest.raises(SystemExit) as error:
        main(arguments)

    assert error.value.code == 2
    assert import_calls == []


@pytest.mark.parametrize(
    "arguments",
    [
        [
            "--action-representation",
            "joint_29d",
            "--runner",
            "missing_separator",
        ],
        [
            "--action-representation",
            "unsupported",
            "--runner",
            "runner_module:run",
        ],
        [
            "--action-representation",
            "joint_29d",
            "--runner",
            "runner_module:run",
            "--steps",
            "10",
        ],
    ],
)
def test_launcher_syntax_errors_fail_before_dynamic_import(
    monkeypatch,
    arguments: list[str],
) -> None:
    import_calls: list[str] = []
    install_import_registry(monkeypatch, {}, import_calls)

    with pytest.raises(SystemExit) as error:
        main(arguments)

    assert error.value.code == 2
    assert import_calls == []


def test_cartesian_launch_binds_spec_and_defers_fk_creation(
    monkeypatch,
    action_spec: BimanualActionSpec,
) -> None:
    provider = object()
    events: list[object] = []
    captured: dict[str, object] = {}
    fk_module = types.ModuleType("fk_module")
    runner_module = types.ModuleType("runner_module")

    def create_kinematics() -> object:
        events.append("create_kinematics")
        return provider

    def run(context: PytorchTrainingLaunchContext) -> None:
        captured["context"] = context
        events.append("runner")
        assert context.bind_action_spec(action_spec) == action_spec
        assert context.create_kinematics() is provider
        assert context.create_kinematics() is provider

    fk_module.create = create_kinematics
    runner_module.entrypoints = types.SimpleNamespace(run=run)
    import_calls: list[str] = []
    install_import_registry(
        monkeypatch,
        {"fk_module": fk_module, "runner_module": runner_module},
        import_calls,
    )

    exit_code = main(
        [
            "--action-representation",
            "cartesian_31d",
            "--runner",
            "runner_module:entrypoints.run",
            "--fk-provider-factory",
            "fk_module:create",
            "--",
            "--dataset",
            "/server/data.hdf5",
            "--steps=10",
        ]
    )

    context = captured["context"]
    assert isinstance(context, PytorchTrainingLaunchContext)
    assert exit_code == 0
    assert import_calls == ["fk_module", "runner_module"]
    assert events == ["runner", "create_kinematics"]
    assert context.action_representation is ActionRepresentation.CARTESIAN_31D
    assert context.logical_action_dim == 31
    assert context.requires_forward_kinematics is True
    assert context.runner_args == (
        "--dataset",
        "/server/data.hdf5",
        "--steps=10",
    )


def test_joint_launch_does_not_load_or_create_fk(
    monkeypatch,
    action_spec: BimanualActionSpec,
) -> None:
    captured: dict[str, PytorchTrainingLaunchContext] = {}
    runner_module = types.ModuleType("runner_module")
    joint_spec = spec_for_representation(action_spec, ActionRepresentation.JOINT_29D)

    def run(context: PytorchTrainingLaunchContext) -> int:
        captured["context"] = context
        context.bind_action_spec(joint_spec)
        return 9

    runner_module.run = run
    import_calls: list[str] = []
    install_import_registry(monkeypatch, {"runner_module": runner_module}, import_calls)

    exit_code = main(
        [
            "--action-representation",
            "joint_29d",
            "--runner",
            "runner_module:run",
        ]
    )

    context = captured["context"]
    assert exit_code == 9
    assert import_calls == ["runner_module"]
    assert context.action_representation is ActionRepresentation.JOINT_29D
    assert context.logical_action_dim == 36
    assert context.requires_forward_kinematics is False
    assert context.action_spec == joint_spec
    assert context.runner_args == ()
    with pytest.raises(RuntimeError, match="must not create"):
        context.create_kinematics()


def test_cartesian_context_rejects_factory_returning_none(
    action_spec: BimanualActionSpec,
) -> None:
    def return_none() -> object:
        return None

    context = PytorchTrainingLaunchContext(
        action_representation=ActionRepresentation.CARTESIAN_31D,
        fk_provider_factory=return_none,
    )
    context.bind_action_spec(action_spec)

    with pytest.raises(ValueError, match="non-None provider"):
        context.create_kinematics()


def test_launch_context_strictly_validates_fk_and_runner_args() -> None:
    with pytest.raises(TypeError, match="expected ActionRepresentation"):
        PytorchTrainingLaunchContext(
            action_representation="joint_29d",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="required for cartesian_31d"):
        PytorchTrainingLaunchContext(
            action_representation=ActionRepresentation.CARTESIAN_31D,
        )
    with pytest.raises(TypeError, match="fk_provider_factory.*callable"):
        PytorchTrainingLaunchContext(
            action_representation=ActionRepresentation.CARTESIAN_31D,
            fk_provider_factory=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="forbidden for joint_29d"):
        PytorchTrainingLaunchContext(
            action_representation=ActionRepresentation.JOINT_29D,
            fk_provider_factory=make_provider_factory(object()),
        )
    with pytest.raises(TypeError, match="runner_args.*tuple"):
        PytorchTrainingLaunchContext(
            action_representation=ActionRepresentation.JOINT_29D,
            runner_args=[],  # type: ignore[arg-type]
        )


def test_launcher_rejects_resolved_noncallable_runner(monkeypatch) -> None:
    runner_module = types.ModuleType("runner_module")
    runner_module.run = object()
    install_import_registry(monkeypatch, {"runner_module": runner_module}, [])

    with pytest.raises(TypeError, match=r"--runner.*not callable"):
        main(
            [
                "--action-representation",
                "joint_29d",
                "--runner",
                "runner_module:run",
            ]
        )


def test_launcher_rejects_noncallable_fk_before_importing_runner(monkeypatch) -> None:
    fk_module = types.ModuleType("fk_module")
    fk_module.create = object()
    import_calls: list[str] = []
    install_import_registry(monkeypatch, {"fk_module": fk_module}, import_calls)

    with pytest.raises(TypeError, match=r"--fk-provider-factory.*not callable"):
        main(
            [
                "--action-representation",
                "cartesian_31d",
                "--runner",
                "runner_module:run",
                "--fk-provider-factory",
                "fk_module:create",
            ]
        )

    assert import_calls == ["fk_module"]


@pytest.mark.parametrize(("runner_result", "expected_exit_code"), [(None, 0), (7, 7)])
def test_launcher_accepts_none_or_exact_integer_runner_return(
    monkeypatch,
    action_spec: BimanualActionSpec,
    runner_result: int | None,
    expected_exit_code: int,
) -> None:
    runner_module = types.ModuleType("runner_module")
    joint_spec = spec_for_representation(action_spec, ActionRepresentation.JOINT_29D)

    def run(context: PytorchTrainingLaunchContext) -> int | None:
        context.bind_action_spec(joint_spec)
        return runner_result

    runner_module.run = run
    install_import_registry(monkeypatch, {"runner_module": runner_module}, [])

    exit_code = main(
        [
            "--action-representation",
            "joint_29d",
            "--runner",
            "runner_module:run",
        ]
    )

    assert exit_code == expected_exit_code


@pytest.mark.parametrize("runner_result", [True, "0", object()])
def test_launcher_rejects_non_exact_integer_runner_return(
    monkeypatch,
    action_spec: BimanualActionSpec,
    runner_result: object,
) -> None:
    runner_module = types.ModuleType("runner_module")
    joint_spec = spec_for_representation(action_spec, ActionRepresentation.JOINT_29D)

    def run(context: PytorchTrainingLaunchContext) -> object:
        context.bind_action_spec(joint_spec)
        return runner_result

    runner_module.run = run
    install_import_registry(monkeypatch, {"runner_module": runner_module}, [])

    with pytest.raises(TypeError, match=r"exact int in \[0, 255\]"):
        main(
            [
                "--action-representation",
                "joint_29d",
                "--runner",
                "runner_module:run",
            ]
        )


def test_launcher_rejects_runner_that_does_not_bind_action_spec(monkeypatch) -> None:
    runner_module = types.ModuleType("runner_module")

    def run(_context: PytorchTrainingLaunchContext) -> None:
        return None

    runner_module.run = run
    install_import_registry(monkeypatch, {"runner_module": runner_module}, [])

    with pytest.raises(RuntimeError, match="returned without binding"):
        main(
            [
                "--action-representation",
                "joint_29d",
                "--runner",
                "runner_module:run",
            ]
        )


def test_context_rejects_spec_from_another_representation(
    action_spec: BimanualActionSpec,
) -> None:
    joint_context = PytorchTrainingLaunchContext(
        action_representation=ActionRepresentation.JOINT_29D,
    )

    with pytest.raises(ValueError, match="conflicts with launcher selection"):
        joint_context.bind_action_spec(action_spec)


def test_context_reuses_one_canonical_spec_and_rejects_rebinding(
    action_spec: BimanualActionSpec,
) -> None:
    context = PytorchTrainingLaunchContext(
        action_representation=ActionRepresentation.CARTESIAN_31D,
        fk_provider_factory=make_provider_factory(object()),
    )

    first = context.bind_action_spec(action_spec)
    second = context.bind_action_spec(action_spec)

    assert first is second is context.action_spec
    with pytest.raises(ValueError, match="different BimanualActionSpec"):
        context.bind_action_spec(
            dataclasses.replace(action_spec, robot_id="another-station")
        )


def test_context_concurrent_initialization_reuses_spec_and_fk(
    action_spec: BimanualActionSpec,
) -> None:
    provider = object()
    factory_calls = 0
    start = threading.Barrier(2)

    def create_kinematics() -> object:
        nonlocal factory_calls
        factory_calls += 1
        return provider

    context = PytorchTrainingLaunchContext(
        action_representation=ActionRepresentation.CARTESIAN_31D,
        fk_provider_factory=create_kinematics,
    )

    def initialize(_index: int) -> tuple[BimanualActionSpec, object]:
        start.wait(timeout=2.0)
        spec = context.bind_action_spec(action_spec)
        return spec, context.create_kinematics()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(initialize, range(2)))

    assert results[0][0] is results[1][0] is context.action_spec
    assert results[0][1] is results[1][1] is provider
    assert factory_calls == 1


def test_cartesian_launcher_requires_runner_to_obtain_fk(
    monkeypatch,
    action_spec: BimanualActionSpec,
) -> None:
    runner_module = types.ModuleType("runner_module")
    fk_module = types.ModuleType("fk_module")

    def run(context: PytorchTrainingLaunchContext) -> None:
        context.bind_action_spec(action_spec)

    def create_kinematics() -> object:
        return object()

    runner_module.run = run
    fk_module.create = create_kinematics
    install_import_registry(
        monkeypatch,
        {"fk_module": fk_module, "runner_module": runner_module},
        [],
    )

    with pytest.raises(RuntimeError, match="returned without obtaining"):
        main(
            [
                "--action-representation",
                "cartesian_31d",
                "--runner",
                "runner_module:run",
                "--fk-provider-factory",
                "fk_module:create",
            ]
        )


def test_launcher_preserves_nonzero_runner_failure_before_binding(monkeypatch) -> None:
    runner_module = types.ModuleType("runner_module")

    def fail_before_configuring(_context: PytorchTrainingLaunchContext) -> int:
        return 7

    runner_module.run = fail_before_configuring
    install_import_registry(monkeypatch, {"runner_module": runner_module}, [])

    assert (
        main(
            [
                "--action-representation",
                "joint_29d",
                "--runner",
                "runner_module:run",
            ]
        )
        == 7
    )


@pytest.mark.parametrize("runner_result", [-1, 256])
def test_launcher_rejects_runner_code_that_can_wrap_at_process_boundary(
    monkeypatch,
    runner_result: int,
) -> None:
    runner_module = types.ModuleType("runner_module")

    def run(_context: PytorchTrainingLaunchContext) -> int:
        return runner_result

    runner_module.run = run
    install_import_registry(monkeypatch, {"runner_module": runner_module}, [])

    with pytest.raises(ValueError, match=r"integer in \[0, 255\]"):
        main(
            [
                "--action-representation",
                "joint_29d",
                "--runner",
                "runner_module:run",
            ]
        )


@pytest.mark.parametrize("exit_code", [None, 0])
def test_launcher_validates_handshake_for_system_exit_success(
    monkeypatch,
    exit_code: int | None,
) -> None:
    runner_module = types.ModuleType("runner_module")

    def run(_context: PytorchTrainingLaunchContext) -> None:
        raise SystemExit(exit_code)

    runner_module.run = run
    install_import_registry(monkeypatch, {"runner_module": runner_module}, [])

    with pytest.raises(RuntimeError, match="returned without binding"):
        main(
            [
                "--action-representation",
                "joint_29d",
                "--runner",
                "runner_module:run",
            ]
        )


@pytest.mark.parametrize("exit_code", [None, 0, 7])
def test_launcher_accepts_valid_system_exit_code(
    monkeypatch,
    action_spec: BimanualActionSpec,
    exit_code: int | None,
) -> None:
    runner_module = types.ModuleType("runner_module")
    joint_spec = spec_for_representation(action_spec, ActionRepresentation.JOINT_29D)

    def run(context: PytorchTrainingLaunchContext) -> None:
        context.bind_action_spec(joint_spec)
        raise SystemExit(exit_code)

    runner_module.run = run
    install_import_registry(monkeypatch, {"runner_module": runner_module}, [])

    assert (
        main(
            [
                "--action-representation",
                "joint_29d",
                "--runner",
                "runner_module:run",
            ]
        )
        == (0 if exit_code is None else exit_code)
    )


@pytest.mark.parametrize("exit_code", [-1, 256])
def test_launcher_rejects_system_exit_code_that_can_wrap(
    monkeypatch,
    exit_code: int,
) -> None:
    runner_module = types.ModuleType("runner_module")

    def run(_context: PytorchTrainingLaunchContext) -> None:
        raise SystemExit(exit_code)

    runner_module.run = run
    install_import_registry(monkeypatch, {"runner_module": runner_module}, [])

    with pytest.raises(ValueError, match=r"integer in \[0, 255\]"):
        main(
            [
                "--action-representation",
                "joint_29d",
                "--runner",
                "runner_module:run",
            ]
        )
