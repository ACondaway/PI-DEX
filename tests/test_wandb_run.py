"""Tests for wandb helpers and checkpoint interval policy."""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from pi_dex.training.wandb_run import finish_train_wandb
from pi_dex.training.wandb_run import init_train_wandb
from pi_dex.training.wandb_run import log_train_metrics
from pi_dex.training.wandb_run import require_wandb_api_key
from pi_dex.training.wandb_run import should_save_checkpoint


def test_should_save_checkpoint_interval_and_final() -> None:
    assert not should_save_checkpoint(global_step=0, save_interval=100, is_final=False)
    assert should_save_checkpoint(global_step=100, save_interval=100, is_final=False)
    assert not should_save_checkpoint(global_step=50, save_interval=100, is_final=False)
    assert should_save_checkpoint(global_step=50, save_interval=100, is_final=True)
    assert not should_save_checkpoint(global_step=50, save_interval=0, is_final=False)
    assert should_save_checkpoint(global_step=50, save_interval=0, is_final=True)


def test_init_train_wandb_disabled_is_noop(tmp_path: pathlib.Path) -> None:
    run = init_train_wandb(
        enabled=False,
        project="pi-dex",
        run_name="t",
        config={"a": 1},
        run_dir=tmp_path,
    )
    assert run is None
    log_train_metrics(None, {"loss": 1.0}, step=1)
    finish_train_wandb(None)


def test_require_wandb_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(OSError, match="WANDB_API_KEY"):
        require_wandb_api_key()
    monkeypatch.setenv("WANDB_API_KEY", "  secret-key  ")
    assert require_wandb_api_key() == "secret-key"


def test_init_train_wandb_writes_id(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    class _FakeRun:
        id = "abc123"

        def log(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def finish(self) -> None:
            return None

    class _FakeSettings:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    calls: list[dict[str, Any]] = []

    def _fake_init(**kwargs: Any) -> _FakeRun:
        calls.append(kwargs)
        return _FakeRun()

    fake_wandb = type("W", (), {"init": staticmethod(_fake_init), "Settings": _FakeSettings})()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")

    run = init_train_wandb(
        enabled=True,
        project="pi-dex",
        run_name="demo",
        config={"lr": 1e-5},
        run_dir=tmp_path,
        resume=False,
    )
    assert run is not None
    assert (tmp_path / "wandb_id.txt").read_text(encoding="utf-8").strip() == "abc123"
    assert calls[0]["project"] == "pi-dex"
    assert calls[0]["name"] == "demo"
    assert isinstance(calls[0]["settings"], _FakeSettings)
    assert calls[0]["settings"].kwargs["api_key"] == "test-key"
