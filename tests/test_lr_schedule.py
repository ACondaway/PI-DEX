"""Warmup + cosine LR schedule used by the joint_29d training runner."""

from __future__ import annotations

import math

import pytest

from pi_dex.training_runner import _MODEL_INPUT_KEYS
from pi_dex.training_runner import _OrderedModelInputDataset
from pi_dex.training_runner import cosine_warmup_decay_lr
from pi_dex.training_runner import resolve_lr_schedule


def test_cosine_warmup_matches_openpi_endpoints() -> None:
    peak = 1e-5
    end = 1e-6
    warmup = 1000
    decay = 80_000

    step0 = cosine_warmup_decay_lr(
        0, peak_lr=peak, warmup_steps=warmup, decay_steps=decay, end_lr=end
    )
    assert step0 == pytest.approx(peak / (warmup + 1))

    at_peak = cosine_warmup_decay_lr(
        warmup, peak_lr=peak, warmup_steps=warmup, decay_steps=decay, end_lr=end
    )
    assert at_peak == pytest.approx(peak)

    mid = warmup + (decay - warmup) // 2
    mid_lr = cosine_warmup_decay_lr(
        mid, peak_lr=peak, warmup_steps=warmup, decay_steps=decay, end_lr=end
    )
    expected_mid = end + (peak - end) * 0.5 * (1.0 + math.cos(math.pi * 0.5))
    assert mid_lr == pytest.approx(expected_mid)
    assert end < mid_lr < peak

    at_end = cosine_warmup_decay_lr(
        decay, peak_lr=peak, warmup_steps=warmup, decay_steps=decay, end_lr=end
    )
    assert at_end == pytest.approx(end)

    after = cosine_warmup_decay_lr(
        decay + 5000, peak_lr=peak, warmup_steps=warmup, decay_steps=decay, end_lr=end
    )
    assert after == pytest.approx(end)


def test_constant_lr_when_decay_disabled() -> None:
    peak = 1e-5
    for step in (0, 10, 10_000):
        assert cosine_warmup_decay_lr(
            step, peak_lr=peak, warmup_steps=0, decay_steps=0, end_lr=1e-6
        ) == pytest.approx(peak)


def test_resolve_80k_uses_openpi_cosine() -> None:
    peak, warmup, decay, end = resolve_lr_schedule(
        peak_lr=1e-5,
        warmup_steps=1000,
        decay_steps=None,
        end_lr=None,
        max_steps=80_000,
    )
    assert peak == pytest.approx(1e-5)
    assert warmup == 1000
    assert decay == 80_000
    assert end == pytest.approx(1e-6)


def test_resolve_short_run_stays_constant() -> None:
    peak, warmup, decay, end = resolve_lr_schedule(
        peak_lr=1e-5,
        warmup_steps=1000,
        decay_steps=None,
        end_lr=None,
        max_steps=2,
    )
    assert peak == pytest.approx(1e-5)
    assert warmup == 0
    assert decay == 0
    assert end == pytest.approx(1e-6)


def test_wrapped_order_cycles_from_resume_cursor() -> None:
    class _Fake:
        def __getitem__(self, index: int) -> dict[str, int]:
            return {key: index for key in _MODEL_INPUT_KEYS}

    dataset = _OrderedModelInputDataset(_Fake(), order=(10, 20, 30), start_index=1, wrap=True)
    assert len(dataset) == 3
    assert dataset[0]["state"] == 20
    assert dataset[1]["state"] == 30
    assert dataset[2]["state"] == 10
