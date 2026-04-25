"""test_ltc_cell.py — Unit tests for LTCCell and ClosedFormLTCCell."""
import torch
import pytest
from closed_form import ClosedFormLTCCell


INPUT_DIM  = 13
HIDDEN_DIM = 8
BATCH      = 4


@pytest.fixture
def cell():
    return ClosedFormLTCCell(INPUT_DIM, HIDDEN_DIM, tau_min=0.1, tau_max=5.0)


def test_tau_strictly_positive(cell):
    x  = torch.randn(BATCH, INPUT_DIM)
    h  = torch.zeros(BATCH, HIDDEN_DIM)
    dt = torch.ones(BATCH)
    _, tau = cell(x, h, dt)
    assert (tau > 0).all(), "τ must be strictly positive after softplus"


def test_output_shape(cell):
    x  = torch.randn(BATCH, INPUT_DIM)
    h  = torch.zeros(BATCH, HIDDEN_DIM)
    dt = torch.ones(BATCH)
    h_next, tau = cell(x, h, dt)
    assert h_next.shape == (BATCH, HIDDEN_DIM)
    assert tau.shape    == (BATCH, HIDDEN_DIM)


def test_gradient_flows(cell):
    x  = torch.randn(BATCH, INPUT_DIM, requires_grad=False)
    h  = torch.zeros(BATCH, HIDDEN_DIM, requires_grad=True)
    dt = torch.ones(BATCH)
    h_next, _ = cell(x, h, dt)
    loss = h_next.sum()
    loss.backward()
    assert h.grad is not None, "Gradient should flow through closed-form cell"
    assert not torch.isnan(h.grad).any()


def test_tau_responds_to_input():
    """τ should vary when input changes."""
    cell = ClosedFormLTCCell(INPUT_DIM, HIDDEN_DIM)
    h  = torch.zeros(1, HIDDEN_DIM)
    dt = torch.ones(1)
    x1 = torch.zeros(1, INPUT_DIM)
    x2 = torch.ones(1,  INPUT_DIM) * 10.0
    _, tau1 = cell(x1, h, dt)
    _, tau2 = cell(x2, h, dt)
    assert not torch.allclose(tau1, tau2), "τ must respond to different inputs"


def test_init_hidden(cell):
    h0 = cell.init_hidden(BATCH, torch.device("cpu"))
    assert h0.shape == (BATCH, HIDDEN_DIM)
    assert (h0 == 0).all()
