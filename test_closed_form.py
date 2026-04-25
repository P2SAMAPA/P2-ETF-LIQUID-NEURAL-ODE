"""test_closed_form.py — Tests for closed-form LTC approximation."""
import torch

from closed_form import ClosedFormLTCCell

INPUT_DIM  = 8
HIDDEN_DIM = 8


def test_closed_form_output_range():
    cell = ClosedFormLTCCell(INPUT_DIM, HIDDEN_DIM)
    x    = torch.randn(2, INPUT_DIM)
    h    = torch.zeros(2, HIDDEN_DIM)
    dt   = torch.tensor([1.0, 2.0])
    h_next, tau = cell(x, h, dt)
    assert torch.isfinite(h_next).all()
    assert torch.isfinite(tau).all()


def test_larger_dt_changes_state_more():
    """With all else equal, a larger Δt should produce a more different h."""
    torch.manual_seed(0)
    cell = ClosedFormLTCCell(INPUT_DIM, HIDDEN_DIM)
    x    = torch.randn(1, INPUT_DIM)
    h    = torch.randn(1, HIDDEN_DIM)

    h1, _ = cell(x, h.clone(), torch.tensor([0.1]))
    h2, _ = cell(x, h.clone(), torch.tensor([5.0]))

    diff1 = (h1 - h).abs().mean().item()
    diff2 = (h2 - h).abs().mean().item()
    assert diff2 > diff1, "Larger Δt should change hidden state more"


def test_no_nan_with_extreme_dt():
    cell = ClosedFormLTCCell(INPUT_DIM, HIDDEN_DIM)
    x    = torch.randn(1, INPUT_DIM)
    h    = torch.zeros(1, HIDDEN_DIM)
    for dt_val in [0.001, 1.0, 100.0]:
        h_next, tau = cell(x, h, torch.tensor([dt_val]))
        assert torch.isfinite(h_next).all(), f"NaN at dt={dt_val}"
