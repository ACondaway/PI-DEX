import dataclasses
import types

import pytest

from pi_dex.spec import BimanualActionSpec
from pi_dex.training_contract import openpi_model_contract_metadata
from pi_dex.training_contract import training_contract_metadata


def make_model_config() -> object:
    return types.SimpleNamespace(
        pi05=True,
        action_dim=32,
        action_horizon=4,
        dtype="bfloat16",
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        max_token_len=200,
        discrete_state_input=True,
    )


def test_training_contract_metadata_revalidates_spec(action_spec: BimanualActionSpec) -> None:
    invalid_spec = dataclasses.replace(action_spec)
    object.__setattr__(invalid_spec, "physical_horizon", 0)
    model_config = make_model_config()

    with pytest.raises(ValueError, match="physical_horizon"):
        training_contract_metadata(invalid_spec, model_config)

    with pytest.raises(TypeError, match="spec"):
        training_contract_metadata(object(), model_config)


def test_openpi_model_contract_binds_tokenizer_variants_and_precision(
    action_spec: BimanualActionSpec,
) -> None:
    contract = openpi_model_contract_metadata(make_model_config(), action_spec)

    assert contract == {
        "pi05": True,
        "action_dim": 32,
        "action_horizon": 4,
        "dtype": "bfloat16",
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "max_token_len": 200,
        "discrete_state_input": True,
    }
