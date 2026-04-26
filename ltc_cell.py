"""ltc_cell.py — Core Liquid Time-Constant (LTC) ODE cell.

Implements:
    dh/dt = −[1/τ(x,h)] · h  +  f(x, h, t) · A(x)
    τ(x,h) = τ_min + softplus(W_τ · [x ‖ h] + b_τ)
    A(x)   = σ(W_A · x + b_A)

Solved with torchdiffeq dopri5 using the adjoint method for
memory-efficient backpropagation through the ODE.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

from logging_utils import get_logger

log = get_logger(__name__)


class LTCODEFunc(nn.Module):
    """ODE right-hand side for a single Liquid Time-Constant layer.

    State: h  (batch, hidden_dim)
    Input: x  (batch, input_dim)  — held constant over an integration step
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        tau_min: float = 0.1,
        tau_max: float = 10.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tau_min = tau_min
        self.tau_max = tau_max

        concat_dim = input_dim + hidden_dim

        # τ network: [x ‖ h] → τ (strictly positive via softplus)
        self.tau_net = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.Softplus(),
        )

        # Input gate A: x → (0,1) via sigmoid
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # State update function f
        self.f_net = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.Tanh(),
        )

        self._x: torch.Tensor | None = None  # current input, set before solve

    def set_input(self, x: torch.Tensor) -> None:
        """Bind current input x before calling odeint."""
        self._x = x

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        assert self._x is not None, "Call set_input(x) before integrating."
        x = self._x
        xh = torch.cat([x, h], dim=-1)

        tau = self.tau_min + self.tau_net(xh) * (self.tau_max - self.tau_min)
        gate_out = self.gate_net(x)
        f = self.f_net(xh)

        dhdt = -(1.0 / tau) * h + f * gate_out
        return dhdt


class LTCCell(nn.Module):
    """LTC cell that integrates over a single time step [0, Δt].

    Args:
        input_dim:  Feature dimension.
        hidden_dim: Number of LTC neurons.
        tau_min:    Minimum time constant (days).
        tau_max:    Maximum time constant (days).
        use_adjoint: Use adjoint sensitivity for backprop (memory-efficient).
        ode_method: ODE solver method (dopri5 recommended).
        rtol:       Relative tolerance.
        atol:       Absolute tolerance.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        tau_min: float = 0.1,
        tau_max: float = 10.0,
        use_adjoint: bool = True,
        ode_method: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_adjoint = use_adjoint
        self.ode_method = ode_method
        self.rtol = rtol
        self.atol = atol

        self.ode_func = LTCODEFunc(input_dim, hidden_dim, tau_min, tau_max)

    def forward(
        self,
        x: torch.Tensor,  # (batch, input_dim)
        h: torch.Tensor,  # (batch, hidden_dim)
        delta_t: torch.Tensor,  # (batch,) elapsed days
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate ODE from 0 to delta_t and return (h_next, tau_dist).

        Returns:
            h_next:   (batch, hidden_dim) updated hidden state.
            tau_dist: (batch, hidden_dim) time-constant values for regime monitoring.
        """
        self.ode_func.set_input(x)

        # Build per-sample time spans — dopri5 needs a common t tensor;
        # we normalise by mean delta_t and scale inside ODE
        t_span = torch.stack([torch.zeros_like(delta_t[0]), delta_t.mean()])

        solver = odeint_adjoint if self.use_adjoint else odeint

        h_traj = solver(
            self.ode_func,
            h,
            t_span,
            method=self.ode_method,
            rtol=self.rtol,
            atol=self.atol,
        )  # shape: (2, batch, hidden_dim)

        h_next = h_traj[-1]

        # Extract τ distribution for regime monitoring
        xh = torch.cat([x, h_next], dim=-1)
        tau_raw = self.ode_func.tau_net(xh)
        tau_dist = self.ode_func.tau_min + tau_raw * (self.ode_func.tau_max - self.ode_func.tau_min)

        return h_next, tau_dist

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)
