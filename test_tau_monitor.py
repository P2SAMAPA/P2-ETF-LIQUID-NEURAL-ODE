"""test_tau_monitor.py — Tests for τ distribution regime monitor."""
import torch
import pytest
from tau_monitor import compute_regime_labels, FAST_THRESHOLD, SLOW_THRESHOLD


def test_fractions_sum_to_one():
    tau = torch.rand(4, 16) * 8.0 + 0.1
    out = compute_regime_labels(tau)
    total = out["fast_frac"] + out["slow_frac"] + out["mixed_frac"]
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)


def test_all_fast_neurons():
    tau = torch.full((2, 16), FAST_THRESHOLD * 0.5)
    out = compute_regime_labels(tau)
    assert (out["fast_frac"] > 0.99).all()
    assert (out["slow_frac"] < 0.01).all()


def test_all_slow_neurons():
    tau = torch.full((2, 16), SLOW_THRESHOLD * 2.0)
    out = compute_regime_labels(tau)
    assert (out["slow_frac"] > 0.99).all()
    assert (out["fast_frac"] < 0.01).all()


def test_1d_input():
    tau = torch.rand(16) * 3.0 + 0.1
    out = compute_regime_labels(tau)
    assert out["tau_mean"].shape == torch.Size([1])


def test_dtype_float32():
    tau = torch.rand(4, 16)
    out = compute_regime_labels(tau)
    assert out["fast_frac"].dtype == torch.float32
