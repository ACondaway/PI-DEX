"""Tests for converted pi05_base weight loading helpers."""

from __future__ import annotations

import pathlib

import pytest
import torch
from torch import nn

from pi_dex.pi05_weights import file_sha256
from pi_dex.pi05_weights import load_verified_pi05_base
from pi_dex.pi05_weights import require_converted_base_dir


class _TinyConfig:
    pi05 = True
    paligemma_variant = "gemma_2b"
    action_expert_variant = "gemma_300m"


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _TinyConfig()
        self.weight = nn.Parameter(torch.zeros(2, 2))


def test_require_converted_base_dir_and_sha(tmp_path: pathlib.Path) -> None:
    weights = tmp_path / "model.safetensors"
    from safetensors.torch import save_file

    save_file({"weight": torch.ones(2, 2)}, str(weights))
    require_converted_base_dir(tmp_path)
    digest = file_sha256(weights)
    assert len(digest) == 64


def test_load_verified_pi05_base_strict(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    model = _TinyModel()
    payload = {"weight": torch.ones(2, 2)}
    save_file(payload, str(tmp_path / "model.safetensors"))
    meta = load_verified_pi05_base(model, tmp_path)
    assert torch.allclose(model.weight, torch.ones(2, 2))
    assert meta["weights_sha256"] == file_sha256(tmp_path / "model.safetensors")


def test_load_verified_rejects_sha_mismatch(tmp_path: pathlib.Path) -> None:
    from safetensors.torch import save_file

    save_file({"weight": torch.ones(2, 2)}, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_verified_pi05_base(_TinyModel(), tmp_path, expected_weights_sha256="0" * 64)
