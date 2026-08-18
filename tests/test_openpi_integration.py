import dataclasses
import pathlib
import sys
import types

import numpy as np
import pytest

import pi_dex.openpi_integration as openpi_integration
from pi_dex.actions import ActionRepresentation
from pi_dex.openpi_integration import BimanualDataConfigFactory
from pi_dex.openpi_integration import compute_bimanual_normalization_stats
from pi_dex.openpi_integration import configure_bimanual_data
from pi_dex.openpi_integration import configure_bimanual_train_config
from pi_dex.openpi_integration import create_bimanual_trained_policy
from pi_dex.openpi_integration import create_pi05_model_config
from pi_dex.openpi_integration import create_pytorch_data_loader_from_dataset
from pi_dex.openpi_transforms import PackBimanualActions
from pi_dex.openpi_transforms import UnpackBimanualActions
from pi_dex.openpi_transforms import ValidateBimanualSample
from pi_dex.spec import BimanualActionSpec
from pi_dex.spec import HandNormalization
from tests.helpers import spec_for_representation


@dataclasses.dataclass(frozen=True)
class FakeGroup:
    inputs: tuple[object, ...] = ()
    outputs: tuple[object, ...] = ()

    def push(self, *, inputs=(), outputs=()) -> "FakeGroup":
        return FakeGroup(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class FakeDataConfig:
    model_transforms: FakeGroup = dataclasses.field(default_factory=FakeGroup)
    data_transforms: FakeGroup = dataclasses.field(default_factory=FakeGroup)
    repack_transforms: FakeGroup = dataclasses.field(default_factory=FakeGroup)
    asset_id: str | None = "fake_asset"
    norm_stats: dict[str, object] | None = None
    repo_id: str = "sharpa_local"
    use_quantile_norm: bool = True


class FakeDataFactory:
    def __init__(
        self,
        *,
        use_quantile_norm: bool = True,
        data_config: FakeDataConfig | None = None,
    ) -> None:
        self.use_quantile_norm = use_quantile_norm
        self.data_config = data_config
        self.repo_id = "fake_asset"
        self.assets = types.SimpleNamespace(asset_id="fake_asset")
        self.create_calls = 0

    def create(self, assets_dirs, model_config):
        del assets_dirs, model_config
        self.create_calls += 1
        if self.data_config is not None:
            return self.data_config
        return FakeDataConfig(FakeGroup(), use_quantile_norm=self.use_quantile_norm)


def make_bimanual_data_factory(
    action_spec: BimanualActionSpec,
    data_config: FakeDataConfig,
) -> BimanualDataConfigFactory:
    return BimanualDataConfigFactory(
        FakeDataFactory(data_config=data_config),
        action_spec,
    )


def make_valid_norm_stats(action_spec: BimanualActionSpec) -> dict[str, dict[str, np.ndarray]]:
    def make_entry(width: int) -> dict[str, np.ndarray]:
        return {
            "mean": np.zeros((width,), dtype=np.float32),
            "std": np.ones((width,), dtype=np.float32),
            "q01": np.zeros((width,), dtype=np.float32),
            "q99": np.ones((width,), dtype=np.float32),
        }

    return {
        "state": make_entry(4),
        "left_actions": make_entry(action_spec.logical_action_dim),
        "right_actions": make_entry(action_spec.logical_action_dim),
    }


@dataclasses.dataclass(frozen=True)
class FakeTrainConfig:
    model: object
    data: object
    policy_metadata: dict | None = None
    assets_dirs: pathlib.Path = pathlib.Path("/config-assets")


def make_model_config(**overrides: object) -> object:
    values: dict[str, object] = {
        "pi05": True,
        "action_dim": 32,
        "action_horizon": 4,
        "dtype": "bfloat16",
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "max_token_len": 200,
        "discrete_state_input": True,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def install_fake_pi0_config(monkeypatch: pytest.MonkeyPatch) -> type:
    """Install a lightweight ``Pi0Config`` stub for boundary-only tests."""

    class FakePi0Config:
        def __init__(self, **kwargs: object) -> None:
            for field_name, value in kwargs.items():
                setattr(self, field_name, value)

    pi0_config_module = types.ModuleType("openpi.models.pi0_config")
    pi0_config_module.Pi0Config = FakePi0Config
    models_module = types.ModuleType("openpi.models")
    models_module.pi0_config = pi0_config_module
    openpi_module = types.ModuleType("openpi")
    openpi_module.models = models_module
    monkeypatch.setitem(sys.modules, "openpi", openpi_module)
    monkeypatch.setitem(sys.modules, "openpi.models", models_module)
    monkeypatch.setitem(sys.modules, "openpi.models.pi0_config", pi0_config_module)
    return FakePi0Config


def test_create_pi05_model_config_builds_validated_config_before_openpi_use(
    action_spec: BimanualActionSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_config_type = install_fake_pi0_config(monkeypatch)

    model_config = create_pi05_model_config(
        action_spec,
        dtype="float32",
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        max_token_len=128,
        pytorch_compile_mode=None,
    )

    assert isinstance(model_config, fake_config_type)
    assert model_config.dtype == "float32"
    assert model_config.paligemma_variant == "gemma_2b_lora"
    assert model_config.action_expert_variant == "gemma_300m_lora"
    assert model_config.action_dim == 32
    assert model_config.action_horizon == action_spec.model_action_horizon
    assert model_config.max_token_len == 128
    assert model_config.pi05 is True
    assert model_config.discrete_state_input is True
    assert model_config.pytorch_compile_mode is None


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("dtype", 1, TypeError),
        ("dtype", "float16", ValueError),
        ("paligemma_variant", None, TypeError),
        ("paligemma_variant", "unknown", ValueError),
        ("action_expert_variant", 1, TypeError),
        ("action_expert_variant", "unknown", ValueError),
        ("max_token_len", True, TypeError),
        ("max_token_len", 0, ValueError),
        ("pytorch_compile_mode", 1, TypeError),
        ("pytorch_compile_mode", "fast", ValueError),
    ],
)
def test_create_pi05_model_config_validates_fields_before_importing_openpi(
    action_spec: BimanualActionSpec,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    monkeypatch.setitem(sys.modules, "openpi.models.pi0_config", None)

    with pytest.raises(error_type, match=field_name):
        create_pi05_model_config(action_spec, **{field_name: value})


def test_create_pi05_model_config_revalidates_spec_before_importing_openpi(
    action_spec: BimanualActionSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "openpi.models.pi0_config", None)
    invalid_spec = dataclasses.replace(action_spec)
    object.__setattr__(invalid_spec, "physical_horizon", 0)

    with pytest.raises(ValueError, match="physical_horizon"):
        create_pi05_model_config(invalid_spec)

    with pytest.raises(TypeError, match="spec"):
        create_pi05_model_config(object())


def test_configure_bimanual_data_preserves_openpi_transform_order(action_spec: BimanualActionSpec) -> None:
    tokenize = object()
    data_config = FakeDataConfig(model_transforms=FakeGroup(inputs=(tokenize,)))
    model_config = make_model_config()

    configured = configure_bimanual_data(data_config, model_config, action_spec)

    assert configured is not data_config
    assert configured.model_transforms.inputs[0] is tokenize
    assert isinstance(configured.model_transforms.inputs[-1], PackBimanualActions)
    assert isinstance(configured.model_transforms.outputs[0], UnpackBimanualActions)
    assert (
        configured.model_transforms.inputs[-1].action_representation
        is action_spec.action_representation
    )
    assert (
        configured.model_transforms.outputs[0].action_representation
        is action_spec.action_representation
    )
    assert len(configured.model_transforms.outputs) == 1
    assert isinstance(configured.data_transforms.inputs[-1], ValidateBimanualSample)
    assert configured.data_transforms.inputs[-1].physical_horizon == action_spec.physical_horizon
    assert (
        configured.data_transforms.inputs[-1].action_representation
        is action_spec.action_representation
    )


def test_configure_bimanual_data_binds_state_width_from_stats(
    action_spec: BimanualActionSpec,
) -> None:
    data_config = FakeDataConfig(norm_stats=make_valid_norm_stats(action_spec))

    configured = configure_bimanual_data(data_config, make_model_config(), action_spec)

    validator = configured.data_transforms.inputs[-1]
    assert isinstance(validator, ValidateBimanualSample)
    assert validator.state_dim == 4


def test_bimanual_data_factory_validates_model_and_asset_before_base_factory_io(
    action_spec: BimanualActionSpec,
) -> None:
    base_factory = FakeDataFactory()
    factory = BimanualDataConfigFactory(base_factory, action_spec)

    with pytest.raises(ValueError, match="discrete_state_input"):
        factory.create(pathlib.Path("/assets"), make_model_config(discrete_state_input=False))
    assert base_factory.create_calls == 0

    base_factory.assets.asset_id = "../unsafe"
    with pytest.raises(ValueError, match="directory name"):
        factory.create(pathlib.Path("/assets"), make_model_config())
    assert base_factory.create_calls == 0


def test_configure_train_config_adds_data_factory_and_wire_metadata(action_spec: BimanualActionSpec) -> None:
    model_config = make_model_config()
    train_config = FakeTrainConfig(model=model_config, data=FakeDataFactory(), policy_metadata={"owner": "test"})

    configured = configure_bimanual_train_config(train_config, action_spec)

    assert isinstance(configured.data, BimanualDataConfigFactory)
    assert configured.policy_metadata["owner"] == "test"
    assert configured.policy_metadata["pi_dex"]["model_action_horizon"] == 4
    assert configured.policy_metadata["openpi_model"]["max_token_len"] == 200


def test_configure_train_config_rejects_conflicting_model_metadata(
    action_spec: BimanualActionSpec,
) -> None:
    train_config = FakeTrainConfig(
        model=make_model_config(),
        data=FakeDataFactory(),
        policy_metadata={"openpi_model": {"max_token_len": 128}},
    )

    with pytest.raises(ValueError, match=r"policy_metadata\['openpi_model'\].*conflicts"):
        configure_bimanual_train_config(train_config, action_spec)


def test_configure_train_config_requires_a_dataclass_instance(
    action_spec: BimanualActionSpec,
) -> None:
    train_config = types.SimpleNamespace(
        model=types.SimpleNamespace(pi05=True, action_dim=32, action_horizon=4),
        data=FakeDataFactory(),
        policy_metadata=None,
    )

    with pytest.raises(TypeError, match=r"train_config.*dataclass instance"):
        configure_bimanual_train_config(train_config, action_spec)


@pytest.mark.parametrize("policy_metadata", [[], ""])
def test_configure_train_config_rejects_falsey_non_dict_metadata(
    action_spec: BimanualActionSpec,
    policy_metadata: object,
) -> None:
    model_config = make_model_config()
    train_config = FakeTrainConfig(model=model_config, data=FakeDataFactory(), policy_metadata=policy_metadata)

    with pytest.raises(TypeError, match="policy_metadata"):
        configure_bimanual_train_config(train_config, action_spec)


def test_configure_bimanual_data_is_idempotent(action_spec: BimanualActionSpec) -> None:
    model_config = make_model_config()
    once = configure_bimanual_data(FakeDataConfig(FakeGroup()), model_config, action_spec)

    twice = configure_bimanual_data(once, model_config, action_spec)

    assert twice is once


def test_configure_bimanual_data_rejects_existing_transforms_for_another_representation(
    action_spec: BimanualActionSpec,
) -> None:
    configured = configure_bimanual_data(
        FakeDataConfig(FakeGroup()),
        make_model_config(),
        action_spec,
    )
    other_representation = (
        ActionRepresentation.JOINT_29D
        if action_spec.action_representation is ActionRepresentation.CARTESIAN_31D
        else ActionRepresentation.CARTESIAN_31D
    )
    other_spec = spec_for_representation(action_spec, other_representation)

    with pytest.raises(ValueError, match="action representation conflicts"):
        configure_bimanual_data(configured, make_model_config(), other_spec)


def test_configure_bimanual_data_rejects_validator_for_another_horizon(
    action_spec: BimanualActionSpec,
) -> None:
    data_config = FakeDataConfig(
        data_transforms=FakeGroup(
            inputs=(
                ValidateBimanualSample(
                    physical_horizon=1,
                    action_representation=action_spec.action_representation,
                ),
            )
        )
    )

    with pytest.raises(ValueError, match="validator physical horizon conflicts"):
        configure_bimanual_data(
            data_config,
            make_model_config(),
            action_spec,
        )


def test_configure_bimanual_data_upgrades_or_rejects_bound_state_width(
    action_spec: BimanualActionSpec,
) -> None:
    unbound = FakeDataConfig(
        data_transforms=FakeGroup(
            inputs=(
                ValidateBimanualSample(
                    physical_horizon=action_spec.physical_horizon,
                    action_representation=action_spec.action_representation,
                ),
            ),
        ),
    )

    upgraded = configure_bimanual_data(
        unbound,
        make_model_config(),
        action_spec,
        state_dim=4,
    )
    assert upgraded.data_transforms.inputs[-1].state_dim == 4

    with pytest.raises(ValueError, match="state width conflicts"):
        configure_bimanual_data(
            upgraded,
            make_model_config(),
            action_spec,
            state_dim=3,
        )


@pytest.mark.parametrize(
    "model_transforms",
    [
        FakeGroup(
            inputs=(PackBimanualActions(ActionRepresentation.CARTESIAN_31D), object()),
            outputs=(UnpackBimanualActions(ActionRepresentation.CARTESIAN_31D),),
        ),
        FakeGroup(inputs=(PackBimanualActions(ActionRepresentation.CARTESIAN_31D),), outputs=()),
        FakeGroup(
            inputs=(
                PackBimanualActions(ActionRepresentation.CARTESIAN_31D),
                PackBimanualActions(ActionRepresentation.CARTESIAN_31D),
            ),
            outputs=(UnpackBimanualActions(ActionRepresentation.CARTESIAN_31D),),
        ),
    ],
)
def test_configure_bimanual_data_rejects_malformed_existing_pair(
    action_spec: BimanualActionSpec,
    model_transforms: FakeGroup,
) -> None:
    model_config = make_model_config()

    with pytest.raises(ValueError, match="pack/unpack"):
        configure_bimanual_data(
            FakeDataConfig(model_transforms=model_transforms),
            model_config,
            action_spec,
        )


@pytest.mark.parametrize("group_name", ["model_transforms", "data_transforms", "repack_transforms"])
def test_configure_bimanual_data_rejects_existing_outer_output_transforms(
    action_spec: BimanualActionSpec,
    group_name: str,
) -> None:
    model_config = make_model_config()
    data_config = dataclasses.replace(
        FakeDataConfig(),
        **{group_name: FakeGroup(outputs=(object(),))},
    )

    with pytest.raises(ValueError, match=rf"data_config\.{group_name}\.outputs"):
        configure_bimanual_data(data_config, model_config, action_spec)


class FakeRunningStats:
    def __init__(self) -> None:
        self.values = []

    def update(self, values: np.ndarray) -> None:
        self.values.append(values.reshape(-1, values.shape[-1]))

    def get_statistics(self):
        values = np.concatenate(self.values, axis=0)
        return types.SimpleNamespace(
            mean=values.mean(axis=0),
            std=values.std(axis=0),
            q01=np.quantile(values, 0.01, axis=0),
            q99=np.quantile(values, 0.99, axis=0),
        )


def install_fake_openpi(monkeypatch, validation_calls: list[tuple[object, object, bool]] | None = None) -> None:
    transforms_module = types.ModuleType("openpi.transforms")
    transforms_module.compose = lambda transforms: lambda sample: _apply_transforms(transforms, sample)
    normalize_module = types.ModuleType("openpi.shared.normalize")
    normalize_module.RunningStats = FakeRunningStats
    pi_dex_normalization_module = types.ModuleType("pi_dex.normalization")

    def validate_normalization_stats(stats, spec, *, require_state=False) -> None:
        if validation_calls is not None:
            validation_calls.append((stats, spec, require_state))

    pi_dex_normalization_module.validate_normalization_stats = validate_normalization_stats
    shared_module = types.ModuleType("openpi.shared")
    shared_module.normalize = normalize_module
    openpi_module = types.ModuleType("openpi")
    openpi_module.transforms = transforms_module
    openpi_module.shared = shared_module
    monkeypatch.setitem(sys.modules, "openpi", openpi_module)
    monkeypatch.setitem(sys.modules, "openpi.transforms", transforms_module)
    monkeypatch.setitem(sys.modules, "openpi.shared", shared_module)
    monkeypatch.setitem(sys.modules, "openpi.shared.normalize", normalize_module)
    monkeypatch.setitem(sys.modules, "pi_dex.normalization", pi_dex_normalization_module)


def _apply_transforms(transforms, sample):
    result = sample
    for transform in transforms:
        result = transform(result)
    return result


@pytest.mark.parametrize("representation", list(ActionRepresentation))
def test_compute_stats_keeps_logical_actions_until_after_normalization(
    action_spec: BimanualActionSpec,
    monkeypatch,
    representation: ActionRepresentation,
) -> None:
    action_spec = spec_for_representation(action_spec, representation)
    validation_calls = []
    install_fake_openpi(monkeypatch, validation_calls)
    dataset = [
        {
            "state": np.full((4,), index, dtype=np.float32),
            "left_actions": np.full(
                (2, action_spec.logical_action_dim),
                index + 1,
                dtype=np.float32,
            ),
            "right_actions": np.full(
                (2, action_spec.logical_action_dim),
                index + 11,
                dtype=np.float32,
            ),
        }
        for index in range(2)
    ]
    data_config = types.SimpleNamespace(
        repack_transforms=FakeGroup(),
        data_transforms=FakeGroup(),
    )

    stats = compute_bimanual_normalization_stats(dataset, data_config, action_spec)

    assert stats["left_actions"].mean.shape == (action_spec.logical_action_dim,)
    assert stats["right_actions"].mean.shape == (action_spec.logical_action_dim,)
    assert stats["left_actions"].mean[0] == 1.5
    assert stats["right_actions"].mean[0] == 11.5
    assert len(validation_calls) == 1
    validated_stats, validated_spec, require_state = validation_calls[0]
    assert validated_stats is stats
    assert validated_spec == action_spec
    assert require_state is True


def test_compute_stats_can_pool_left_and_right_explicitly(
    action_spec: BimanualActionSpec,
    monkeypatch,
) -> None:
    install_fake_openpi(monkeypatch)
    shared_spec = dataclasses.replace(action_spec, hand_normalization=HandNormalization.SHARED)
    dataset = [
        {
            "state": np.zeros((4,), dtype=np.float32),
            "left_actions": np.zeros((2, shared_spec.logical_action_dim), dtype=np.float32),
            "right_actions": np.full(
                (2, shared_spec.logical_action_dim),
                10.0,
                dtype=np.float32,
            ),
        },
        {
            "state": np.ones((4,), dtype=np.float32),
            "left_actions": np.zeros((2, shared_spec.logical_action_dim), dtype=np.float32),
            "right_actions": np.full(
                (2, shared_spec.logical_action_dim),
                10.0,
                dtype=np.float32,
            ),
        },
    ]
    data_config = types.SimpleNamespace(repack_transforms=FakeGroup(), data_transforms=FakeGroup())

    stats = compute_bimanual_normalization_stats(dataset, data_config, shared_spec)

    np.testing.assert_array_equal(
        stats["left_actions"].mean,
        np.full((shared_spec.logical_action_dim,), 5.0),
    )
    np.testing.assert_array_equal(
        stats["right_actions"].mean,
        np.full((shared_spec.logical_action_dim,), 5.0),
    )


def test_compute_stats_requires_exact_unbatched_hand_shapes(
    action_spec: BimanualActionSpec,
    monkeypatch,
) -> None:
    install_fake_openpi(monkeypatch)
    dataset = [
        {
            "state": np.zeros((4,), dtype=np.float32),
            "left_actions": np.zeros(
                (1, 2, action_spec.logical_action_dim),
                dtype=np.float32,
            ),
            "right_actions": np.zeros((2, action_spec.logical_action_dim), dtype=np.float32),
        }
    ]
    data_config = types.SimpleNamespace(repack_transforms=FakeGroup(), data_transforms=FakeGroup())

    with pytest.raises(
        ValueError,
        match=rf"left_actions.*expected \(2, {action_spec.logical_action_dim}\)",
    ):
        compute_bimanual_normalization_stats(dataset, data_config, action_spec)


def test_compute_stats_requires_matching_hand_dtypes(
    action_spec: BimanualActionSpec,
    monkeypatch,
) -> None:
    install_fake_openpi(monkeypatch)
    dataset = [
        {
            "state": np.zeros((4,), dtype=np.float32),
            "left_actions": np.zeros((2, action_spec.logical_action_dim), dtype=np.float32),
            "right_actions": np.zeros((2, action_spec.logical_action_dim), dtype=np.float64),
        }
    ]
    data_config = types.SimpleNamespace(repack_transforms=FakeGroup(), data_transforms=FakeGroup())

    with pytest.raises(TypeError, match="action dtypes"):
        compute_bimanual_normalization_stats(dataset, data_config, action_spec)


def test_compute_stats_requires_unbatched_one_dimensional_state(
    action_spec: BimanualActionSpec,
    monkeypatch,
) -> None:
    install_fake_openpi(monkeypatch)
    dataset = [
        {
            "state": np.zeros((2, 4), dtype=np.float32),
            "left_actions": np.zeros((2, action_spec.logical_action_dim), dtype=np.float32),
            "right_actions": np.zeros((2, action_spec.logical_action_dim), dtype=np.float32),
        }
    ]
    data_config = types.SimpleNamespace(repack_transforms=FakeGroup(), data_transforms=FakeGroup())

    with pytest.raises(ValueError, match=r"state.*ndim.*expected 1.*got 2"):
        compute_bimanual_normalization_stats(dataset, data_config, action_spec)


def test_custom_dataset_loader_requires_false_shuffle_when_distributed(
    action_spec: BimanualActionSpec,
    monkeypatch,
) -> None:
    torch_module = types.ModuleType("torch")
    torch_module.distributed = types.SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        get_world_size=lambda: 2,
        get_rank=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    model_config = make_model_config()

    with pytest.raises(ValueError, match=r"shuffle must be False under DDP"):
        create_pytorch_data_loader_from_dataset(
            object(),
            make_bimanual_data_factory(action_spec, FakeDataConfig()),
            pathlib.Path("/assets"),
            model_config,
            action_spec,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            seed=0,
            skip_norm_stats=True,
        )


def test_custom_dataset_loader_rejects_distributed_false_when_initialized(
    action_spec: BimanualActionSpec,
    monkeypatch,
) -> None:
    torch_module = types.ModuleType("torch")
    torch_module.distributed = types.SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    model_config = make_model_config()

    with pytest.raises(ValueError, match=r"distributed=False conflicts"):
        create_pytorch_data_loader_from_dataset(
            object(),
            make_bimanual_data_factory(action_spec, FakeDataConfig()),
            pathlib.Path("/assets"),
            model_config,
            action_spec,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            seed=0,
            skip_norm_stats=True,
            distributed=False,
        )


def test_custom_dataset_loader_rejects_factory_for_another_action_spec(
    action_spec: BimanualActionSpec,
) -> None:
    other_spec = dataclasses.replace(action_spec, coordinate_frame="other_frame")
    factory = make_bimanual_data_factory(other_spec, FakeDataConfig())

    with pytest.raises(ValueError, match=r"data_factory\.spec"):
        create_pytorch_data_loader_from_dataset(
            object(),
            factory,
            pathlib.Path("/assets"),
            make_model_config(),
            action_spec,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            seed=0,
            skip_norm_stats=True,
        )


@pytest.mark.parametrize(
    ("override", "error_type", "message"),
    [
        ({"shuffle": 1}, TypeError, "shuffle"),
        ({"skip_norm_stats": 1}, TypeError, "skip_norm_stats"),
        ({"seed": True}, TypeError, "seed"),
        ({"num_batches": True}, TypeError, "num_batches"),
        ({"num_batches": 0}, ValueError, "num_batches"),
        ({"batch_size": True}, TypeError, "batch_size"),
        ({"batch_size": "2"}, TypeError, "batch_size"),
        ({"batch_size": 0}, ValueError, "batch_size"),
        ({"num_workers": True}, TypeError, "num_workers"),
        ({"num_workers": "0"}, TypeError, "num_workers"),
        ({"num_workers": -1}, ValueError, "num_workers"),
    ],
)
def test_custom_dataset_loader_rejects_invalid_control_arguments(
    action_spec: BimanualActionSpec,
    override: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    model_config = make_model_config()
    arguments: dict[str, object] = {
        "batch_size": 2,
        "shuffle": True,
        "num_workers": 0,
        "seed": 0,
        "skip_norm_stats": True,
    }
    arguments.update(override)

    with pytest.raises(error_type, match=message):
        create_pytorch_data_loader_from_dataset(
            object(),
            make_bimanual_data_factory(action_spec, FakeDataConfig()),
            pathlib.Path("/assets"),
            model_config,
            action_spec,
            **arguments,
        )


def test_custom_dataset_loader_requires_normalization_stats(
    action_spec: BimanualActionSpec,
) -> None:
    model_config = make_model_config()

    with pytest.raises(ValueError, match=r"data_config\.norm_stats"):
        create_pytorch_data_loader_from_dataset(
            object(),
            make_bimanual_data_factory(action_spec, FakeDataConfig()),
            pathlib.Path("/assets"),
            model_config,
            action_spec,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            seed=0,
        )


def test_custom_dataset_loader_rejects_openpi_fake_repo_normalization_sentinel(
    action_spec: BimanualActionSpec,
) -> None:
    model_config = make_model_config()
    data_config = FakeDataConfig(norm_stats=make_valid_norm_stats(action_spec), repo_id="fake")

    with pytest.raises(ValueError, match="'fake' disables normalization"):
        create_pytorch_data_loader_from_dataset(
            object(),
            make_bimanual_data_factory(action_spec, data_config),
            pathlib.Path("/assets"),
            model_config,
            action_spec,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            seed=0,
        )


def test_custom_dataset_loader_requires_pi05_quantile_normalization(
    action_spec: BimanualActionSpec,
) -> None:
    model_config = make_model_config()
    data_config = FakeDataConfig(
        norm_stats=make_valid_norm_stats(action_spec),
        use_quantile_norm=False,
    )

    with pytest.raises(ValueError, match="use_quantile_norm"):
        create_pytorch_data_loader_from_dataset(
            object(),
            make_bimanual_data_factory(action_spec, data_config),
            pathlib.Path("/assets"),
            model_config,
            action_spec,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            seed=0,
        )


def test_custom_dataset_loader_wires_openpi_pytorch_loader(
    action_spec: BimanualActionSpec,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    torch_module = types.ModuleType("torch")
    torch_module.distributed = types.SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: False,
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    data_loader_module = types.ModuleType("openpi.training.data_loader")

    def transform_dataset(dataset, data_config, *, skip_norm_stats):
        captured["transform_dataset"] = dataset
        captured["transform_data_config"] = data_config
        captured["skip_norm_stats"] = skip_norm_stats
        return "transformed-dataset"

    class FakeTorchDataLoader:
        def __init__(self, dataset, **kwargs) -> None:
            captured["torch_dataset"] = dataset
            captured["torch_kwargs"] = kwargs

    def data_loader_impl(data_config, loader):
        captured["wrapper_data_config"] = data_config
        captured["wrapper_loader"] = loader
        return "wrapped-loader"

    data_loader_module.transform_dataset = transform_dataset
    data_loader_module.TorchDataLoader = FakeTorchDataLoader
    data_loader_module.DataLoaderImpl = data_loader_impl
    training_module = types.ModuleType("openpi.training")
    training_module.data_loader = data_loader_module
    openpi_module = types.ModuleType("openpi")
    openpi_module.training = training_module
    monkeypatch.setitem(sys.modules, "openpi", openpi_module)
    monkeypatch.setitem(sys.modules, "openpi.training", training_module)
    monkeypatch.setitem(sys.modules, "openpi.training.data_loader", data_loader_module)

    dataset = object()
    data_config = FakeDataConfig(norm_stats=make_valid_norm_stats(action_spec))
    model_config = make_model_config()
    loader = create_pytorch_data_loader_from_dataset(
        dataset,
        make_bimanual_data_factory(action_spec, data_config),
        pathlib.Path("/assets"),
        model_config,
        action_spec,
        batch_size=3,
        shuffle=True,
        num_workers=2,
        seed=7,
        num_batches=5,
    )

    assert loader == "wrapped-loader"
    assert captured["transform_dataset"] is dataset
    assert captured["skip_norm_stats"] is False
    assert captured["torch_dataset"] == "transformed-dataset"
    assert captured["torch_kwargs"] == {
        "local_batch_size": 3,
        "shuffle": True,
        "sampler": None,
        "num_batches": 5,
        "num_workers": 2,
        "seed": 7,
        "framework": "pytorch",
    }
    assert captured["wrapper_data_config"] is captured["transform_data_config"]
    assert isinstance(captured["wrapper_loader"], FakeTorchDataLoader)


@pytest.mark.parametrize(
    ("use_quantile_norm", "has_pytorch_weights", "model_dtype"),
    [
        (True, True, "bfloat16"),
        (False, True, "bfloat16"),
        (True, False, "bfloat16"),
        (True, True, "float32"),
    ],
)
def test_create_trained_policy_requires_quantiles_then_validates_checkpoint_stats(
    action_spec: BimanualActionSpec,
    monkeypatch,
    tmp_path: pathlib.Path,
    use_quantile_norm: bool,
    has_pytorch_weights: bool,
    model_dtype: str,
) -> None:
    captured: dict[str, object] = {}
    norm_stats = make_valid_norm_stats(action_spec)

    download_module = types.ModuleType("openpi.shared.download")

    def maybe_download(path: str) -> pathlib.Path:
        captured["download_path"] = path
        return pathlib.Path(path)

    download_module.maybe_download = maybe_download
    shared_module = types.ModuleType("openpi.shared")
    shared_module.download = download_module

    checkpoints_module = types.ModuleType("openpi.training.checkpoints")

    def load_norm_stats(assets_dir: pathlib.Path, asset_id: str):
        captured["norm_stats_path"] = assets_dir
        captured["norm_stats_exists_during_load"] = (
            assets_dir / asset_id / "norm_stats.json"
        ).is_file()
        captured["asset_id"] = asset_id
        return norm_stats

    checkpoints_module.load_norm_stats = load_norm_stats
    training_module = types.ModuleType("openpi.training")
    training_module.checkpoints = checkpoints_module

    policy_config_module = types.ModuleType("openpi.policies.policy_config")

    def create_trained_policy(
        train_config,
        checkpoint_dir,
        *,
        repack_transforms=None,
        sample_kwargs=None,
        default_prompt=None,
        norm_stats=None,
        pytorch_device=None,
    ):
        captured["policy_train_config"] = train_config
        captured["policy_checkpoint_dir"] = checkpoint_dir
        captured["policy_repack_transforms"] = repack_transforms
        captured["policy_sample_kwargs"] = sample_kwargs
        captured["policy_default_prompt"] = default_prompt
        captured["policy_norm_stats"] = norm_stats
        captured["policy_pytorch_device"] = pytorch_device
        captured["policy_data_config"] = train_config.data.create(
            train_config.assets_dirs,
            train_config.model,
        )
        captured["snapshot_weights_exist_during_policy_load"] = (
            pathlib.Path(checkpoint_dir) / "model.safetensors"
        ).is_file()
        return types.SimpleNamespace(metadata=train_config.policy_metadata, _model=object())

    policy_config_module.create_trained_policy = create_trained_policy
    policies_module = types.ModuleType("openpi.policies")
    policies_module.policy_config = policy_config_module

    openpi_module = types.ModuleType("openpi")
    openpi_module.policies = policies_module
    openpi_module.shared = shared_module
    openpi_module.training = training_module

    normalization_module = types.ModuleType("pi_dex.normalization")

    def validate_normalization_stats(stats, spec, *, require_state=False) -> None:
        captured["validated_stats"] = stats
        captured["validated_spec"] = spec
        captured["require_state"] = require_state

    def normalization_state_dim(stats, spec) -> int:
        captured["state_dim_stats"] = stats
        captured["state_dim_spec"] = spec
        return 4

    normalization_module.validate_normalization_stats = validate_normalization_stats
    normalization_module.normalization_state_dim = normalization_state_dim

    pytorch_training_module = types.ModuleType("pi_dex.pytorch_training")

    def neutralize_model(model, spec) -> None:
        captured["neutralized_model"] = model
        captured["neutralized_spec"] = spec

    pytorch_training_module.neutralize_openpi_dense_action_io = neutralize_model

    monkeypatch.setitem(sys.modules, "openpi", openpi_module)
    monkeypatch.setitem(sys.modules, "openpi.policies", policies_module)
    monkeypatch.setitem(sys.modules, "openpi.policies.policy_config", policy_config_module)
    monkeypatch.setitem(sys.modules, "openpi.shared", shared_module)
    monkeypatch.setitem(sys.modules, "openpi.shared.download", download_module)
    monkeypatch.setitem(sys.modules, "openpi.training", training_module)
    monkeypatch.setitem(sys.modules, "openpi.training.checkpoints", checkpoints_module)
    monkeypatch.setitem(sys.modules, "pi_dex.normalization", normalization_module)
    monkeypatch.setitem(sys.modules, "pi_dex.pytorch_training", pytorch_training_module)

    def validate_contract(
        checkpoint_dir: pathlib.Path,
        spec: BimanualActionSpec,
        *,
        model_config: object,
        norm_stats,
        asset_id: str,
    ) -> None:
        contract_calls = captured.setdefault("contract_calls", [])
        assert isinstance(contract_calls, list)
        contract_calls.append(
            {
                "checkpoint_dir": checkpoint_dir,
                "sidecar_exists": (checkpoint_dir / "pi_dex.json").is_file(),
            }
        )
        captured["contract_checkpoint_dir"] = checkpoint_dir
        captured["contract_spec"] = spec
        captured["contract_model_config"] = model_config
        captured["contract_norm_stats"] = norm_stats
        captured["contract_asset_id"] = asset_id

    monkeypatch.setattr(openpi_integration, "load_and_validate_training_contract", validate_contract)
    configure_bimanual_data_impl = openpi_integration.configure_bimanual_data

    def capture_materialized_data_config(data_config, model_config, spec, *, state_dim=None):
        configured_data = configure_bimanual_data_impl(
            data_config,
            model_config,
            spec,
            state_dim=state_dim,
        )
        if state_dim is not None:
            captured["configured_state_dim"] = state_dim
            captured["materialized_data_config"] = configured_data
        return configured_data

    monkeypatch.setattr(
        openpi_integration,
        "configure_bimanual_data",
        capture_materialized_data_config,
    )
    model_config = make_model_config(dtype=model_dtype)
    data_factory = FakeDataFactory(use_quantile_norm=use_quantile_norm)
    train_config = FakeTrainConfig(
        model=model_config,
        data=data_factory,
    )
    if has_pytorch_weights:
        (tmp_path / "model.safetensors").touch()
        (tmp_path / "pi_dex.json").write_text("{}", encoding="utf-8")
        normalization_asset = tmp_path / "assets" / "fake_asset" / "norm_stats.json"
        normalization_asset.parent.mkdir(parents=True)
        normalization_asset.write_text("{}", encoding="utf-8")
    else:
        with pytest.raises(FileNotFoundError, match=r"model\.safetensors"):
            create_bimanual_trained_policy(train_config, tmp_path, action_spec)
        return

    if not use_quantile_norm:
        with pytest.raises(ValueError, match="use_quantile_norm"):
            create_bimanual_trained_policy(train_config, tmp_path, action_spec)
        return

    if model_dtype != "bfloat16":
        with pytest.raises(ValueError, match=r"model\.dtype.*only with 'bfloat16'"):
            create_bimanual_trained_policy(train_config, tmp_path, action_spec)
        return

    create_bimanual_trained_policy(
        train_config,
        tmp_path,
        action_spec,
        default_prompt="pick up the test object",
        pytorch_device="cuda:0",
        sample_kwargs={"num_steps": 12},
    )

    norm_stats_path = captured["norm_stats_path"]
    assert isinstance(norm_stats_path, pathlib.Path)
    assert norm_stats_path != tmp_path / "assets"
    assert norm_stats_path.name == "assets"
    assert captured["norm_stats_exists_during_load"] is True
    assert captured["snapshot_weights_exist_during_policy_load"] is True
    contract_calls = captured["contract_calls"]
    assert isinstance(contract_calls, list)
    assert len(contract_calls) == 2
    assert all(call["sidecar_exists"] is True for call in contract_calls)
    assert contract_calls[0]["checkpoint_dir"] == contract_calls[1]["checkpoint_dir"]
    assert captured["asset_id"] == "fake_asset"
    assert captured["validated_stats"] is norm_stats
    assert captured["validated_spec"] == action_spec
    assert captured["require_state"] is True
    assert captured["contract_norm_stats"] is norm_stats
    assert captured["contract_asset_id"] == "fake_asset"
    assert captured["contract_model_config"] is model_config
    assert captured["configured_state_dim"] == 4
    assert captured["neutralized_spec"] == action_spec
    assert data_factory.create_calls == 1
    assert captured["policy_data_config"] is captured["materialized_data_config"]
    assert captured["policy_data_config"].use_quantile_norm is True
    policy_train_config = captured["policy_train_config"]
    assert policy_train_config.data.create(
        policy_train_config.assets_dirs,
        policy_train_config.model,
    ) is captured["policy_data_config"]
    assert captured["policy_repack_transforms"] is None
    assert captured["policy_sample_kwargs"] == {"num_steps": 12}
    assert captured["policy_default_prompt"] == "pick up the test object"
    assert captured["policy_norm_stats"] is norm_stats
    assert captured["policy_pytorch_device"] == "cuda:0"


@pytest.mark.parametrize(
    ("sample_kwargs", "error_type", "message"),
    [
        ({"noise": object()}, ValueError, "num_steps"),
        ({"num_steps": True}, TypeError, "num_steps"),
        ({"num_steps": 0}, ValueError, r"\[1, 1000\]"),
        ({"num_steps": 1001}, ValueError, r"\[1, 1000\]"),
    ],
)
def test_create_trained_policy_rejects_unsafe_sample_kwargs_before_openpi_import(
    action_spec: BimanualActionSpec,
    monkeypatch: pytest.MonkeyPatch,
    sample_kwargs: object,
    error_type: type[Exception],
    message: str,
) -> None:
    monkeypatch.setitem(sys.modules, "openpi.policies", None)
    train_config = FakeTrainConfig(model=make_model_config(), data=FakeDataFactory())

    with pytest.raises(error_type, match=message):
        create_bimanual_trained_policy(
            train_config,
            pathlib.Path("/checkpoint"),
            action_spec,
            sample_kwargs=sample_kwargs,
        )


@pytest.mark.parametrize("checkpoint_dir", [None, object(), 1])
def test_create_trained_policy_rejects_checkpoint_type_before_openpi_import(
    action_spec: BimanualActionSpec,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_dir: object,
) -> None:
    monkeypatch.setitem(sys.modules, "openpi.policies", None)
    train_config = FakeTrainConfig(model=make_model_config(), data=FakeDataFactory())

    with pytest.raises(TypeError, match="checkpoint_dir"):
        create_bimanual_trained_policy(train_config, checkpoint_dir, action_spec)


@pytest.mark.parametrize(
    ("asset_id", "message"),
    [
        (None, "declared assets.asset_id or repo_id"),
        ("../unsafe", "directory name"),
    ],
)
def test_create_trained_policy_rejects_declared_asset_before_openpi_import_or_download(
    action_spec: BimanualActionSpec,
    monkeypatch: pytest.MonkeyPatch,
    asset_id: str | None,
    message: str,
) -> None:
    data_factory = FakeDataFactory()
    data_factory.assets.asset_id = asset_id
    data_factory.repo_id = asset_id
    monkeypatch.setitem(sys.modules, "openpi.policies", None)
    train_config = FakeTrainConfig(model=make_model_config(), data=data_factory)

    with pytest.raises(ValueError, match=message):
        create_bimanual_trained_policy(
            train_config,
            pathlib.Path("gs://checkpoint-that-must-not-be-downloaded"),
            action_spec,
        )

    assert data_factory.create_calls == 0
