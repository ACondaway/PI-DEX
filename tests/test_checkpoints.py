import dataclasses
import json
import types

import numpy as np
import pytest

from pi_dex.actions import LOGICAL_ACTION_DIM
from pi_dex.checkpoints import load_and_validate_training_contract
from pi_dex.checkpoints import save_training_contract
from pi_dex.checkpoints import validate_normalization_asset_id
from pi_dex.normalization import NORMALIZATION_FINGERPRINT_ALGORITHM
from pi_dex.normalization import normalization_stats_fingerprint
from pi_dex.spec import BimanualActionSpec

ASSET_ID = "sharpa_north_train_v1"


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


def make_norm_stats() -> dict[str, dict[str, np.ndarray]]:
    def stats(width: int, offset: float) -> dict[str, np.ndarray]:
        mean = np.arange(width, dtype=np.float32) + offset
        return {
            "mean": mean,
            "std": np.ones(width, dtype=np.float32),
            "q01": mean - 1.0,
            "q99": mean + 1.0,
        }

    return {
        "state": stats(4, 0.0),
        "left_actions": stats(LOGICAL_ACTION_DIM, 10.0),
        "right_actions": stats(LOGICAL_ACTION_DIM, 20.0),
    }


def write_norm_stats_json(path, norm_stats) -> None:
    payload = {
        "norm_stats": {
            key: {
                field_name: values.tolist()
                for field_name, values in entry.items()
            }
            for key, entry in norm_stats.items()
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def stage_training_contract(
    checkpoint_dir,
    action_spec: BimanualActionSpec,
    *,
    norm_stats=None,
    model_config=None,
):
    if norm_stats is None:
        norm_stats = make_norm_stats()
    if model_config is None:
        model_config = make_model_config()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "model.safetensors").write_bytes(b"fixture weights")
    asset_dir = checkpoint_dir / "assets" / ASSET_ID
    asset_dir.mkdir(parents=True, exist_ok=True)
    write_norm_stats_json(asset_dir / "norm_stats.json", norm_stats)
    return save_training_contract(
        checkpoint_dir,
        action_spec,
        model_config=model_config,
        norm_stats=norm_stats,
        asset_id=ASSET_ID,
    )


def load_training_contract(
    checkpoint_dir,
    action_spec: BimanualActionSpec,
    *,
    norm_stats=None,
    model_config=None,
    asset_id: str = ASSET_ID,
):
    return load_and_validate_training_contract(
        checkpoint_dir,
        action_spec,
        model_config=model_config or make_model_config(),
        norm_stats=norm_stats or make_norm_stats(),
        asset_id=asset_id,
    )


def test_training_contract_round_trip(tmp_path, action_spec: BimanualActionSpec) -> None:
    norm_stats = make_norm_stats()
    metadata_path = stage_training_contract(
        tmp_path,
        action_spec,
        norm_stats=norm_stats,
    )

    metadata = load_training_contract(
        tmp_path,
        action_spec,
        norm_stats=norm_stats,
    )

    assert metadata_path.name == "pi_dex.json"
    assert metadata["pi_dex"]["model_action_horizon"] == 4
    assert metadata["pytorch_training"]["padding_loss_policy"] == "exclude_invalid_dimensions_v1"
    assert metadata["pytorch_training"]["padding_inference_policy"] == "zero_dense_action_io_parameters_v1"
    assert metadata["normalization"]["asset_id"] == ASSET_ID
    assert metadata["normalization"]["fingerprint_algorithm"] == NORMALIZATION_FINGERPRINT_ALGORITHM
    assert metadata["normalization"]["fingerprint"] == normalization_stats_fingerprint(
        norm_stats,
        action_spec,
    )
    assert metadata["normalization"]["state_dim"] == 4
    assert metadata["normalization"]["asset_file"] == f"assets/{ASSET_ID}/norm_stats.json"
    assert metadata["openpi_model"]["max_token_len"] == 200
    assert len(metadata["pytorch_training"]["weights_fingerprint"]) == 64
    assert not list(tmp_path.glob(".pi_dex.json.*.tmp"))


def test_training_contract_rejects_changed_model_or_tokenizer_contract(
    tmp_path,
    action_spec: BimanualActionSpec,
) -> None:
    stage_training_contract(tmp_path, action_spec)

    with pytest.raises(ValueError, match=r"openpi_model.*max_token_len"):
        load_training_contract(
            tmp_path,
            action_spec,
            model_config=make_model_config(max_token_len=128),
        )


def test_checkpoint_load_revalidates_the_supplied_spec(
    tmp_path,
    action_spec: BimanualActionSpec,
) -> None:
    stage_training_contract(tmp_path, action_spec)
    invalid_spec = dataclasses.replace(action_spec)
    object.__setattr__(invalid_spec, "physical_horizon", 0)

    with pytest.raises(ValueError, match="physical_horizon"):
        load_training_contract(tmp_path, invalid_spec)


def test_training_contract_rejects_resume_with_different_horizon(
    tmp_path,
    action_spec: BimanualActionSpec,
) -> None:
    norm_stats = make_norm_stats()
    metadata_path = stage_training_contract(
        tmp_path,
        action_spec,
        norm_stats=norm_stats,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["pi_dex"]["physical_horizon"] = 3
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="physical_horizon"):
        load_training_contract(
            tmp_path,
            action_spec,
            norm_stats=norm_stats,
        )


def test_training_contract_rejects_unknown_action_metadata_field(
    tmp_path,
    action_spec: BimanualActionSpec,
) -> None:
    metadata_path = stage_training_contract(tmp_path, action_spec)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["pi_dex"]["unversioned_extension"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=r"unexpected fields.*unversioned_extension"):
        load_training_contract(tmp_path, action_spec)


def test_training_contract_rejects_unknown_root_metadata_field(
    tmp_path,
    action_spec: BimanualActionSpec,
) -> None:
    metadata_path = stage_training_contract(tmp_path, action_spec)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["unversioned_extension"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=r"checkpoint metadata fields.*unversioned_extension"):
        load_training_contract(tmp_path, action_spec)


def test_training_contract_rejects_different_normalization_asset_id(
    tmp_path,
    action_spec: BimanualActionSpec,
) -> None:
    norm_stats = make_norm_stats()
    stage_training_contract(
        tmp_path,
        action_spec,
        norm_stats=norm_stats,
    )

    with pytest.raises(ValueError, match=r"normalization\.asset_id"):
        load_training_contract(
            tmp_path,
            action_spec,
            norm_stats=norm_stats,
            asset_id="different_asset",
        )


def test_training_contract_rejects_changed_normalization_values(
    tmp_path,
    action_spec: BimanualActionSpec,
) -> None:
    norm_stats = make_norm_stats()
    stage_training_contract(
        tmp_path,
        action_spec,
        norm_stats=norm_stats,
    )
    changed_stats = make_norm_stats()
    changed_stats["left_actions"]["mean"][0] += 0.25

    with pytest.raises(ValueError, match=r"normalization\.fingerprint"):
        load_training_contract(
            tmp_path,
            action_spec,
            norm_stats=changed_stats,
        )


def test_training_contract_rejects_changed_state_width(
    tmp_path,
    action_spec: BimanualActionSpec,
) -> None:
    stage_training_contract(tmp_path, action_spec)
    changed_stats = make_norm_stats()
    for field_name in changed_stats["state"]:
        changed_stats["state"][field_name] = np.zeros((3,), dtype=np.float32)

    with pytest.raises(ValueError, match=r"normalization\.state_dim|normalization\.fingerprint"):
        load_training_contract(tmp_path, action_spec, norm_stats=changed_stats)


def test_training_contract_rejects_tampered_recorded_state_width(
    tmp_path,
    action_spec: BimanualActionSpec,
) -> None:
    metadata_path = stage_training_contract(tmp_path, action_spec)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["normalization"]["state_dim"] = 3
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=r"normalization\.state_dim"):
        load_training_contract(tmp_path, action_spec)


def test_training_contract_rejects_changed_weight_bytes(
    tmp_path,
    action_spec: BimanualActionSpec,
) -> None:
    stage_training_contract(tmp_path, action_spec)
    (tmp_path / "model.safetensors").write_bytes(b"different weights")

    with pytest.raises(ValueError, match=r"weights_fingerprint.*weights changed"):
        load_training_contract(tmp_path, action_spec)


def test_training_contract_rejects_changed_serialized_normalization_asset(
    tmp_path,
    action_spec: BimanualActionSpec,
) -> None:
    stage_training_contract(tmp_path, action_spec)
    (tmp_path / "assets" / ASSET_ID / "norm_stats.json").write_text(
        '{"changed": true}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"asset_file_fingerprint.*asset changed"):
        load_training_contract(tmp_path, action_spec)


def test_save_rejects_serialized_stats_that_differ_from_supplied_stats(
    tmp_path,
    action_spec: BimanualActionSpec,
) -> None:
    supplied_stats = make_norm_stats()
    serialized_stats = make_norm_stats()
    serialized_stats["right_actions"]["mean"][0] += 0.5
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "model.safetensors").write_bytes(b"fixture weights")
    asset_dir = tmp_path / "assets" / ASSET_ID
    asset_dir.mkdir(parents=True, exist_ok=True)
    write_norm_stats_json(asset_dir / "norm_stats.json", serialized_stats)

    with pytest.raises(ValueError, match="asset content does not match supplied"):
        save_training_contract(
            tmp_path,
            action_spec,
            model_config=make_model_config(),
            norm_stats=supplied_stats,
            asset_id=ASSET_ID,
        )


def test_save_training_contract_requires_staged_checkpoint_artifacts(
    tmp_path,
    action_spec: BimanualActionSpec,
) -> None:
    with pytest.raises(FileNotFoundError, match=r"model\.safetensors"):
        save_training_contract(
            tmp_path,
            action_spec,
            model_config=make_model_config(),
            norm_stats=make_norm_stats(),
            asset_id=ASSET_ID,
        )

    (tmp_path / "model.safetensors").write_bytes(b"fixture weights")
    with pytest.raises(FileNotFoundError, match=r"norm_stats\.json"):
        save_training_contract(
            tmp_path,
            action_spec,
            model_config=make_model_config(),
            norm_stats=make_norm_stats(),
            asset_id=ASSET_ID,
        )


def test_training_contract_rejects_malformed_recorded_fingerprint(
    tmp_path,
    action_spec: BimanualActionSpec,
) -> None:
    metadata_path = stage_training_contract(tmp_path, action_spec)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["normalization"]["fingerprint"] = "not-a-sha256"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"normalization\.fingerprint.*64 lowercase hexadecimal",
    ):
        load_training_contract(tmp_path, action_spec)


@pytest.mark.parametrize("asset_id", ["../other", "/absolute", "nested/name", " ", ".", ".."])
def test_normalization_asset_id_rejects_unsafe_directory_names(asset_id: str) -> None:
    with pytest.raises(ValueError, match="directory name"):
        validate_normalization_asset_id(asset_id)
