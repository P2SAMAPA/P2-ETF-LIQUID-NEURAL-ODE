"""closed_form.py — Closed-form LTC approximation (Hasani et al. 2021).

h(t+Δt) ≈ h(t)·exp(−Δt/τ) + τ·(1−exp(−Δt/τ))·f(x,h,t)·A(x)

~10× faster than ODE solve at inference. Toggled via config.use_closed_form.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from logging_utils import get_logger

log = get_logger(__name__)


class ClosedFormLTCCell(nn.Module):
    """Closed-form approximation of the LTC ODE for fast inference.

    Implements the same tau(x,h), A(x), f(x,h) networks as LTCCell
    but replaces the ODE solver with the analytical solution.
    Compatible drop-in replacement for LTCCell at inference time.
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

        # FIX: Sigmoid instead of Softplus so tau is strictly bounded in
        # (tau_min, tau_max). Softplus outputs [0, inf) which makes tau
        # unbounded -> tau * (1 - decay) explodes to infinity -> NaN in h_next.
        self.tau_net = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.Sigmoid(),  # output in (0,1) -> tau in (tau_min, tau_max)
        )

        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.f_net = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        x: torch.Tensor,  # (batch, input_dim)
        h: torch.Tensor,  # (batch, hidden_dim)
        delta_t: torch.Tensor,  # (batch,) elapsed days
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Closed-form LTC step.

        Returns:
            h_next:   (batch, hidden_dim)
            tau_dist: (batch, hidden_dim)
        """
        xh = torch.cat([x, h], dim=-1)

        # tau strictly in (tau_min, tau_max) -- never zero, never infinite
        tau = self.tau_min + self.tau_net(xh) * (self.tau_max - self.tau_min)

        gate_out = self.gate_net(x)
        f = self.f_net(xh)

        # Broadcast delta_t: (batch,) -> (batch, hidden_dim)
        dt = delta_t.unsqueeze(-1).expand_as(tau)

        # Clamp dt/tau: avoids exp underflow/overflow
        decay = torch.exp(-torch.clamp(dt / tau, min=1e-8, max=20.0))

        # tau is bounded so tau * (1 - decay) is always finite
        h_next = h * decay + tau * (1.0 - decay) * f * gate_out

        # Final clamp to prevent any residual explosion propagating across steps
        h_next = torch.clamp(h_next, min=-10.0, max=10.0)

        return h_next, tau

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)
