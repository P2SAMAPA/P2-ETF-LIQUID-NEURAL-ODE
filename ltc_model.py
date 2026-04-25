"""ltc_model.py — Full Liquid Time-Constant model for ETF ranking.

Architecture:
    Input projection → LTC recurrence (window steps) → Dropout → Linear head

Forward pass processes a rolling window of (x, delta_t) pairs and
returns (scores, tau_dist) where tau_dist feeds the regime monitor.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from closed_form import ClosedFormLTCCell
from logging_utils import get_logger
from ltc_cell import LTCCell

log = get_logger(__name__)


class LTCModel(nn.Module):
    """Liquid Time-Constant model for cross-sectional ETF scoring.

    Args:
        input_dim:       Raw feature dimension.
        hidden_dim:      LTC hidden state size.
        n_etf:           Output dimension (number of ETFs in universe).
        tau_min:         Minimum time constant.
        tau_max:         Maximum time constant.
        dropout:         MC Dropout probability.
        use_closed_form: Use fast closed-form cell instead of ODE solver.
        use_adjoint:     Use adjoint backprop (ODE solver path only).
        ode_method:      ODE solver method.
        rtol, atol:      ODE tolerances.
    """

    def __init__(
        self,
        input_dim:       int,
        hidden_dim:      int,
        n_etf:           int,
        tau_min:         float = 0.1,
        tau_max:         float = 10.0,
        dropout:         float = 0.1,
        use_closed_form: bool  = False,
        use_adjoint:     bool  = True,
        ode_method:      str   = "dopri5",
        rtol:            float = 1e-3,
        atol:            float = 1e-4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_etf      = n_etf

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )

        # LTC core (ODE or closed-form)
        if use_closed_form:
            log.info("Using closed-form LTC cell (fast inference mode)")
            self.ltc_cell = ClosedFormLTCCell(hidden_dim, hidden_dim, tau_min, tau_max)
        else:
            log.info("Using ODE LTC cell (dopri5 + adjoint)")
            self.ltc_cell = LTCCell(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                tau_min=tau_min,
                tau_max=tau_max,
                use_adjoint=use_adjoint,
                ode_method=ode_method,
                rtol=rtol,
                atol=atol,
            )

        # Output head
        self.dropout    = nn.Dropout(p=dropout)
        self.output_head = nn.Linear(hidden_dim, n_etf)

        log.info("LTCModel: input=%d → hidden=%d → output=%d", input_dim, hidden_dim, n_etf)

    def forward(
        self,
        x:       torch.Tensor,   # (batch, window, input_dim)
        delta_t: torch.Tensor,   # (batch, window)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass over a rolling window.

        Args:
            x:        (B, W, D) feature sequences.
            delta_t:  (B, W) elapsed-day sequences.

        Returns:
            scores:   (B, N_etf) raw predicted returns.
            tau_dist: (B, hidden_dim) time-constant distribution at last step.
        """
        batch_size = x.size(0)
        device     = x.device
        h = self.ltc_cell.init_hidden(batch_size, device)
        tau_dist = torch.zeros(batch_size, self.hidden_dim, device=device)

        # Unroll over window
        for t in range(x.size(1)):
            x_t  = self.input_proj(x[:, t, :])   # (B, hidden_dim)
            dt_t = delta_t[:, t]                  # (B,)
            h, tau_dist = self.ltc_cell(x_t, h, dt_t)

        h_drop = self.dropout(h)
        scores = self.output_head(h_drop)         # (B, N_etf)
        return scores, tau_dist
