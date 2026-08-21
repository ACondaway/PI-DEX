import dataclasses
import types

import pytest

torch = pytest.importorskip("torch")

from pi_dex.core.actions import MODEL_ACTION_DIM  # noqa: E402
from pi_dex.core.actions import PRETRAINED_MODEL_ACTION_DIM  # noqa: E402
from pi_dex.core.spec import BimanualActionSpec  # noqa: E402
from pi_dex.training.pytorch_training import PiDexPytorchTrainer  # noqa: E402
from pi_dex.training.pytorch_training import compute_semantic_flow_matching_loss  # noqa: E402
from pi_dex.training.pytorch_training import neutralize_model_padding  # noqa: E402
from pi_dex.core.actions import ActionRepresentation  # noqa: E402
from pi_dex.training.pytorch_training import reduce_semantic_action_loss  # noqa: E402
from pi_dex.training.pytorch_training import expand_action_projections_from_pretrained  # noqa: E402
from pi_dex.training.pytorch_training import neutralize_openpi_dense_action_io  # noqa: E402
from tests.helpers import spec_for_representation  # noqa: E402


class FakePi05(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(
            pi05=True,
            action_dim=MODEL_ACTION_DIM,
            action_horizon=4,
            dtype="bfloat16",
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            max_token_len=200,
            discrete_state_input=True,
        )
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
        self.forwarded_actions = None
        self.forwarded_noise = None
        self.forwarded_time = None

    def forward(self, observation, actions, noise=None, time=None):
        del observation
        self.forwarded_actions = actions.detach().clone()
        self.forwarded_noise = noise.detach().clone()
        self.forwarded_time = time
        return torch.ones_like(actions) * self.scale.square()


class NonfiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        del ctx
        return value.square()

    @staticmethod
    def backward(ctx, gradient):
        del ctx
        return gradient * float("nan")


class FakeNonfiniteGradientPi05(FakePi05):
    def forward(self, observation, actions, noise=None, time=None):
        del observation, noise, time
        return torch.ones_like(actions) * NonfiniteGradient.apply(self.scale)


@pytest.mark.parametrize(
    "representation",
    [ActionRepresentation.CARTESIAN_31D, ActionRepresentation.JOINT_29D],
)
def test_reduce_semantic_loss_excludes_padding_dimensions(
    action_spec: BimanualActionSpec,
    representation: ActionRepresentation,
) -> None:
    action_spec = spec_for_representation(action_spec, representation)
    elementwise_loss = torch.ones((2, 4, MODEL_ACTION_DIM), dtype=torch.float32)
    elementwise_loss[..., action_spec.logical_action_dim :] = 1_000_000.0

    loss = reduce_semantic_action_loss(elementwise_loss, action_spec)

    torch.testing.assert_close(loss, torch.tensor(1.0))


@pytest.mark.parametrize(
    "representation",
    [ActionRepresentation.CARTESIAN_31D, ActionRepresentation.JOINT_29D],
)
def test_neutralize_padding_does_not_mutate_input(
    action_spec: BimanualActionSpec,
    representation: ActionRepresentation,
) -> None:
    action_spec = spec_for_representation(action_spec, representation)
    values = torch.ones((2, 4, MODEL_ACTION_DIM), dtype=torch.float32)

    neutralized = neutralize_model_padding(values, action_spec)

    torch.testing.assert_close(values, torch.ones_like(values))
    torch.testing.assert_close(
        neutralized[..., : action_spec.logical_action_dim],
        values[..., : action_spec.logical_action_dim],
    )
    torch.testing.assert_close(
        neutralized[..., action_spec.logical_action_dim :],
        torch.zeros_like(neutralized[..., action_spec.logical_action_dim :]),
    )


def test_compute_flow_loss_neutralizes_action_and_noise_padding(action_spec: BimanualActionSpec) -> None:
    model = FakePi05()
    actions = torch.ones((2, 4, MODEL_ACTION_DIM), dtype=torch.float32)
    noise = torch.full_like(actions, 3.0)

    loss = compute_semantic_flow_matching_loss(model, object(), actions, action_spec, noise=noise)

    torch.testing.assert_close(loss, torch.tensor(4.0))
    torch.testing.assert_close(model.forwarded_actions[..., -1], torch.zeros((2, 4)))
    torch.testing.assert_close(model.forwarded_noise[..., -1], torch.zeros((2, 4)))
    torch.testing.assert_close(actions[..., -1], torch.ones((2, 4)))
    torch.testing.assert_close(noise[..., -1], torch.full((2, 4), 3.0))


def test_joint_flow_loss_keeps_all_semantic_dimensions(
    action_spec: BimanualActionSpec,
) -> None:
    joint_spec = spec_for_representation(action_spec, ActionRepresentation.JOINT_29D)
    model = FakePi05()
    actions = torch.ones((2, 4, MODEL_ACTION_DIM), dtype=torch.float32)
    noise = torch.full_like(actions, 3.0)

    compute_semantic_flow_matching_loss(model, object(), actions, joint_spec, noise=noise)

    torch.testing.assert_close(model.forwarded_actions, torch.ones((2, 4, MODEL_ACTION_DIM)))
    torch.testing.assert_close(model.forwarded_noise, torch.full((2, 4, MODEL_ACTION_DIM), 3.0))


def test_neutralize_openpi_dense_action_io_zeros_only_padding_parameters(
    action_spec: BimanualActionSpec,
) -> None:
    class FakeProjectionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = FakePi05().config
            self.action_in_proj = torch.nn.Linear(MODEL_ACTION_DIM, 8)
            self.action_out_proj = torch.nn.Linear(8, MODEL_ACTION_DIM)

    # Fixture action_spec is Cartesian (31D semantic, 5D pad to MODEL_ACTION_DIM).
    pad = MODEL_ACTION_DIM - action_spec.logical_action_dim
    assert pad > 0
    model = FakeProjectionModel()
    input_semantic = model.action_in_proj.weight[:, :-pad].detach().clone()
    output_semantic = model.action_out_proj.weight[:-pad].detach().clone()
    bias_semantic = model.action_out_proj.bias[:-pad].detach().clone()

    neutralize_openpi_dense_action_io(model, action_spec)

    torch.testing.assert_close(model.action_in_proj.weight[:, :-pad], input_semantic)
    torch.testing.assert_close(model.action_out_proj.weight[:-pad], output_semantic)
    torch.testing.assert_close(model.action_out_proj.bias[:-pad], bias_semantic)
    torch.testing.assert_close(
        model.action_in_proj.weight[:, -pad:], torch.zeros((8, pad))
    )
    torch.testing.assert_close(
        model.action_out_proj.weight[-pad:], torch.zeros((pad, 8))
    )
    torch.testing.assert_close(model.action_out_proj.bias[-pad:], torch.zeros((pad,)))


def test_neutralize_openpi_dense_action_io_is_noop_for_full_width_joint(
    action_spec: BimanualActionSpec,
) -> None:
    class FakeProjectionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = FakePi05().config
            self.action_in_proj = torch.nn.Linear(MODEL_ACTION_DIM, 8)
            self.action_out_proj = torch.nn.Linear(8, MODEL_ACTION_DIM)

    joint_spec = spec_for_representation(action_spec, ActionRepresentation.JOINT_29D)
    model = FakeProjectionModel()
    input_semantic = model.action_in_proj.weight.detach().clone()
    output_semantic = model.action_out_proj.weight.detach().clone()
    bias_semantic = model.action_out_proj.bias.detach().clone()

    neutralize_openpi_dense_action_io(model, joint_spec)

    torch.testing.assert_close(model.action_in_proj.weight, input_semantic)
    torch.testing.assert_close(model.action_out_proj.weight, output_semantic)
    torch.testing.assert_close(model.action_out_proj.bias, bias_semantic)


def test_expand_action_projections_from_pretrained_copies_and_zero_inits_motor_block() -> None:
    class FakeProjectionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = FakePi05().config
            self.action_in_proj = torch.nn.Linear(PRETRAINED_MODEL_ACTION_DIM, 8)
            self.action_out_proj = torch.nn.Linear(8, PRETRAINED_MODEL_ACTION_DIM)

    model = FakeProjectionModel()
    input_semantic = model.action_in_proj.weight.detach().clone()
    output_semantic = model.action_out_proj.weight.detach().clone()
    bias_semantic = model.action_out_proj.bias.detach().clone()

    expanded = expand_action_projections_from_pretrained(model)

    assert expanded is True
    assert model.action_in_proj.in_features == MODEL_ACTION_DIM
    assert model.action_out_proj.out_features == MODEL_ACTION_DIM
    torch.testing.assert_close(
        model.action_in_proj.weight[:, :PRETRAINED_MODEL_ACTION_DIM],
        input_semantic,
    )
    torch.testing.assert_close(
        model.action_in_proj.weight[:, PRETRAINED_MODEL_ACTION_DIM:],
        torch.zeros((8, MODEL_ACTION_DIM - PRETRAINED_MODEL_ACTION_DIM)),
    )
    torch.testing.assert_close(
        model.action_out_proj.weight[:PRETRAINED_MODEL_ACTION_DIM, :],
        output_semantic,
    )
    torch.testing.assert_close(
        model.action_out_proj.weight[PRETRAINED_MODEL_ACTION_DIM:, :],
        torch.zeros((MODEL_ACTION_DIM - PRETRAINED_MODEL_ACTION_DIM, 8)),
    )
    torch.testing.assert_close(
        model.action_out_proj.bias[:PRETRAINED_MODEL_ACTION_DIM],
        bias_semantic,
    )
    torch.testing.assert_close(
        model.action_out_proj.bias[PRETRAINED_MODEL_ACTION_DIM:],
        torch.zeros((MODEL_ACTION_DIM - PRETRAINED_MODEL_ACTION_DIM,)),
    )


def test_compute_flow_loss_rejects_model_output_dtype(action_spec: BimanualActionSpec) -> None:
    class WrongDtypeModel(FakePi05):
        def forward(self, observation, actions, noise=None, time=None):
            del observation, noise, time
            return torch.ones(actions.shape, dtype=torch.float64, device=actions.device)

    actions = torch.zeros((2, 4, MODEL_ACTION_DIM), dtype=torch.float32)

    with pytest.raises(TypeError, match="model output dtype"):
        compute_semantic_flow_matching_loss(
            WrongDtypeModel(),
            object(),
            actions,
            action_spec,
            noise=torch.zeros_like(actions),
        )


def test_compute_flow_loss_rejects_model_output_device(action_spec: BimanualActionSpec) -> None:
    class WrongDeviceModel(FakePi05):
        def forward(self, observation, actions, noise=None, time=None):
            del observation, noise, time
            return torch.empty(actions.shape, dtype=actions.dtype, device="meta")

    actions = torch.zeros((2, 4, MODEL_ACTION_DIM), dtype=torch.float32)

    with pytest.raises(ValueError, match="model output device"):
        compute_semantic_flow_matching_loss(
            WrongDeviceModel(),
            object(),
            actions,
            action_spec,
            noise=torch.zeros_like(actions),
        )


def test_compute_flow_loss_rejects_nonfinite_semantic_loss(action_spec: BimanualActionSpec) -> None:
    class FakeNonfiniteLossPi05(FakePi05):
        def forward(self, observation, actions, noise=None, time=None):
            del observation, noise, time
            return torch.full_like(actions, float("nan"))

    actions = torch.zeros((2, 4, MODEL_ACTION_DIM), dtype=torch.float32)

    with pytest.raises(FloatingPointError, match="finite scalar before backward"):
        compute_semantic_flow_matching_loss(
            FakeNonfiniteLossPi05(),
            object(),
            actions,
            action_spec,
            noise=torch.zeros_like(actions),
        )


@pytest.mark.parametrize(
    ("time", "error_type", "message"),
    [
        (torch.zeros((2, 1), dtype=torch.float32), ValueError, "time.shape"),
        (torch.zeros((2,), dtype=torch.float64), TypeError, "time.dtype"),
        (torch.tensor((float("nan"), 0.5), dtype=torch.float32), ValueError, "finite"),
        (torch.tensor((-0.1, 0.5), dtype=torch.float32), ValueError, r"\[0, 1\]"),
        (torch.tensor((0.5, 1.1), dtype=torch.float32), ValueError, r"\[0, 1\]"),
        (object(), TypeError, "time: expected torch.Tensor"),
    ],
)
def test_compute_flow_loss_validates_explicit_time(
    action_spec: BimanualActionSpec,
    time,
    error_type,
    message: str,
) -> None:
    actions = torch.zeros((2, 4, MODEL_ACTION_DIM), dtype=torch.float32)

    with pytest.raises(error_type, match=message):
        compute_semantic_flow_matching_loss(
            FakePi05(),
            object(),
            actions,
            action_spec,
            noise=torch.zeros_like(actions),
            time=time,
        )


def test_compute_flow_loss_requires_time_on_action_device(action_spec: BimanualActionSpec) -> None:
    actions = torch.zeros((2, 4, MODEL_ACTION_DIM), dtype=torch.float32)
    time = torch.empty((2,), dtype=torch.float32, device="meta")

    with pytest.raises(ValueError, match=r"time\.device"):
        compute_semantic_flow_matching_loss(
            FakePi05(),
            object(),
            actions,
            action_spec,
            noise=torch.zeros_like(actions),
            time=time,
        )


def test_compute_flow_loss_forwards_valid_time(action_spec: BimanualActionSpec) -> None:
    model = FakePi05()
    actions = torch.zeros((2, 4, MODEL_ACTION_DIM), dtype=torch.float32)
    time = torch.tensor((0.25, 0.75), dtype=torch.float32)

    compute_semantic_flow_matching_loss(
        model,
        object(),
        actions,
        action_spec,
        noise=torch.zeros_like(actions),
        time=time,
    )

    assert model.forwarded_time is time


def test_compute_flow_loss_requires_generator_on_action_device(action_spec: BimanualActionSpec) -> None:
    actions = torch.empty((2, 4, MODEL_ACTION_DIM), dtype=torch.float32, device="meta")

    with pytest.raises(ValueError, match=r"generator\.device"):
        compute_semantic_flow_matching_loss(
            FakePi05(),
            object(),
            actions,
            action_spec,
            generator=torch.Generator(device="cpu"),
        )


def test_compute_flow_loss_ignores_generator_when_noise_is_explicit(
    action_spec: BimanualActionSpec,
) -> None:
    actions = torch.zeros((2, 4, MODEL_ACTION_DIM), dtype=torch.float32)

    loss = compute_semantic_flow_matching_loss(
        FakePi05(),
        object(),
        actions,
        action_spec,
        noise=torch.zeros_like(actions),
        generator=object(),  # type: ignore[arg-type]
    )

    assert loss.ndim == 0


def test_compute_flow_loss_accepts_default_cpu_generator(action_spec: BimanualActionSpec) -> None:
    actions = torch.zeros((2, 4, MODEL_ACTION_DIM), dtype=torch.float32)

    loss = compute_semantic_flow_matching_loss(
        FakePi05(),
        object(),
        actions,
        action_spec,
        generator=torch.Generator(),
    )

    assert loss.ndim == 0


def test_trainer_updates_original_model_without_wrapping(action_spec: BimanualActionSpec) -> None:
    model = FakePi05()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = PiDexPytorchTrainer(model, optimizer, action_spec, gradient_clip_norm=1.0)
    actions = torch.zeros((2, 4, MODEL_ACTION_DIM), dtype=torch.float32)

    result = trainer.train_step(object(), actions, noise=torch.zeros_like(actions))

    assert result.loss.ndim == 0
    assert not result.loss.requires_grad
    assert result.loss.grad_fn is None
    assert isinstance(result.gradient_norm, torch.Tensor)
    assert not result.gradient_norm.requires_grad
    assert result.gradient_norm.grad_fn is None
    assert model.scale.item() < 2.0
    assert set(model.state_dict()) == {"scale"}


def test_trainer_rejects_nonfinite_gradients_before_optimizer_step(
    action_spec: BimanualActionSpec,
) -> None:
    model = FakeNonfiniteGradientPi05()
    initial_scale = model.scale.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = PiDexPytorchTrainer(model, optimizer, action_spec, gradient_clip_norm=None)
    actions = torch.zeros((2, 4, MODEL_ACTION_DIM), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="non-finite"):
        trainer.train_step(object(), actions, noise=torch.zeros_like(actions))

    torch.testing.assert_close(model.scale.detach(), initial_scale)


def test_trainer_owns_a_revalidated_spec_copy(action_spec: BimanualActionSpec) -> None:
    model = FakePi05()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = PiDexPytorchTrainer(model, optimizer, action_spec, gradient_clip_norm=None)
    object.__setattr__(action_spec, "physical_horizon", 1)
    actions = torch.zeros((2, 4, MODEL_ACTION_DIM), dtype=torch.float32)

    result = trainer.train_step(object(), actions, noise=torch.zeros_like(actions))

    assert result.loss.ndim == 0


def test_trainer_rejects_a_bypassed_invalid_spec(action_spec: BimanualActionSpec) -> None:
    invalid_spec = dataclasses.replace(action_spec)
    object.__setattr__(invalid_spec, "robot_id", " ")
    model = FakePi05()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="robot_id"):
        PiDexPytorchTrainer(model, optimizer, invalid_spec, gradient_clip_norm=None)


def test_trainer_rejects_incomplete_model_contract(action_spec: BimanualActionSpec) -> None:
    model = FakePi05()
    model.config.discrete_state_input = False
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="discrete_state_input"):
        PiDexPytorchTrainer(model, optimizer, action_spec, gradient_clip_norm=None)


def test_trainer_rejects_optimizer_for_foreign_model(action_spec: BimanualActionSpec) -> None:
    model = FakePi05()
    foreign_model = FakePi05()
    optimizer = torch.optim.SGD(foreign_model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="foreign parameter"):
        PiDexPytorchTrainer(model, optimizer, action_spec, gradient_clip_norm=None)


def test_trainer_rejects_optimizer_omitting_trainable_parameter(
    action_spec: BimanualActionSpec,
) -> None:
    model = FakePi05()
    model.extra = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([model.scale], lr=0.1)

    with pytest.raises(ValueError, match="omitted 1 trainable"):
        PiDexPytorchTrainer(model, optimizer, action_spec, gradient_clip_norm=None)


def test_trainer_rejects_duplicate_optimizer_parameter(action_spec: BimanualActionSpec) -> None:
    model = FakePi05()
    optimizer = torch.optim.SGD([model.scale], lr=0.1)
    optimizer.param_groups[0]["params"].append(model.scale)

    with pytest.raises(ValueError, match="duplicate model parameter"):
        PiDexPytorchTrainer(model, optimizer, action_spec, gradient_clip_norm=None)


@pytest.mark.parametrize("gradient_clip_norm", [True, float("nan"), float("inf"), 0.0])
def test_trainer_rejects_invalid_gradient_clip_norm(
    action_spec: BimanualActionSpec,
    gradient_clip_norm: object,
) -> None:
    model = FakePi05()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises((TypeError, ValueError), match="gradient_clip_norm"):
        PiDexPytorchTrainer(
            model,
            optimizer,
            action_spec,
            gradient_clip_norm=gradient_clip_norm,
        )


def test_trainer_rejects_unrepresentable_integer_gradient_clip_norm(
    action_spec: BimanualActionSpec,
) -> None:
    model = FakePi05()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match=r"gradient_clip_norm.*positive finite"):
        PiDexPytorchTrainer(
            model,
            optimizer,
            action_spec,
            gradient_clip_norm=10**400,
        )
