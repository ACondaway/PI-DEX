import types

import numpy as np
import pytest

from pi_dex.actions import ActionRepresentation
from pi_dex.checkpoints import load_and_validate_training_contract
from pi_dex.checkpoints import save_training_contract
from pi_dex.spec import BimanualActionSpec
from tests.helpers import spec_for_representation

openpi_normalize = pytest.importorskip("openpi.shared.normalize")
openpi_checkpoints = pytest.importorskip("openpi.training.checkpoints")

ASSET_ID = "sharpa_north_train_v1"


@pytest.mark.parametrize("representation", list(ActionRepresentation))
def test_openpi_normalization_file_round_trip_preserves_checkpoint_contract(
    tmp_path,
    action_spec: BimanualActionSpec,
    representation: ActionRepresentation,
) -> None:
    action_spec = spec_for_representation(action_spec, representation)

    def stats(width: int, offset: float):
        mean = np.arange(width, dtype=np.float32) + offset
        return openpi_normalize.NormStats(
            mean=mean,
            std=np.ones(width, dtype=np.float32),
            q01=mean - 0.5,
            q99=mean + 0.5,
        )

    norm_stats = {
        "state": stats(4, 0.0),
        "left_actions": stats(action_spec.logical_action_dim, 10.0),
        "right_actions": stats(action_spec.logical_action_dim, 20.0),
    }
    model_config = types.SimpleNamespace(
        pi05=True,
        action_dim=32,
        action_horizon=4,
        dtype="bfloat16",
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        max_token_len=200,
        discrete_state_input=True,
    )
    (tmp_path / "model.safetensors").write_bytes(b"fixture weights")
    openpi_normalize.save(tmp_path / "assets" / ASSET_ID, norm_stats)
    save_training_contract(
        tmp_path,
        action_spec,
        model_config=model_config,
        norm_stats=norm_stats,
        asset_id=ASSET_ID,
    )

    loaded_stats = openpi_checkpoints.load_norm_stats(tmp_path / "assets", ASSET_ID)
    metadata = load_and_validate_training_contract(
        tmp_path,
        action_spec,
        model_config=model_config,
        norm_stats=loaded_stats,
        asset_id=ASSET_ID,
    )

    assert metadata["normalization"]["asset_id"] == ASSET_ID
