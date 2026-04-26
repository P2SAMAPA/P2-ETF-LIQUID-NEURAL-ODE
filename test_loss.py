"""test_loss.py — Tests for the differentiable Sharpe + IC loss."""

import pytest
import torch

from loss import SharpeLoss


@pytest.fixture
def loss_fn():
    return SharpeLoss(lam=0.1)


def test_loss_is_scalar(loss_fn, batch_tensors):
    x, dt, y = batch_tensors
    pred = torch.randn_like(y)
    loss, _ = loss_fn(pred, y)
    assert loss.shape == torch.Size([])


def test_gradient_non_zero(loss_fn, batch_tensors):
    _, _, y = batch_tensors
    pred = torch.randn_like(y, requires_grad=True)
    loss, _ = loss_fn(pred, y)
    loss.backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()
    assert pred.grad.abs().sum() > 0


def test_ic_in_bounds(loss_fn, batch_tensors):
    _, _, y = batch_tensors
    pred = torch.randn_like(y)
    _, info = loss_fn(pred, y)
    assert -1.0 <= info["ic"] <= 1.0


def test_loss_decreases_on_perfect_pred(loss_fn):
    """Loss with perfectly correlated pred should be lower than random."""
    n_assets = 20
    actual = torch.randn(4, n_assets)
    perfect = actual + 0.01 * torch.randn(4, n_assets)
    random = torch.randn(4, n_assets)
    l_perf, _ = loss_fn(perfect, actual)
    l_rand, _ = loss_fn(random, actual)
    assert l_perf.item() < l_rand.item()
