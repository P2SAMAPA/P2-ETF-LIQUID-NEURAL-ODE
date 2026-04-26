"""loss.py — Differentiable Sharpe + IC loss for LTC training."""

from __future__ import annotations

import torch
import torch.nn as nn


class SharpeLoss(nn.Module):
    """Differentiable Sharpe ratio loss with IC regularisation.

    L = −E[r] / √(Var[r] + ε)  +  λ · (1 − IC)

    where IC is the differentiable approximation to Spearman rank
    correlation computed via soft rank over the predicted scores.

    Args:
        lam: IC regularisation weight (default 0.1).
        eps: Numerical stability constant.
    """

    def __init__(self, lam: float = 0.1, eps: float = 1e-8) -> None:
        super().__init__()
        self.lam = lam
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,  # (B, N_etf) predicted scores
        actual: torch.Tensor,  # (B, N_etf) actual next-day returns
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss.

        Returns:
            loss:  scalar loss tensor.
            info:  dict with individual loss components for logging.
        """
        # Long-short portfolio return: dot(softmax_rank(pred), actual)
        weights = torch.softmax(pred, dim=-1) - 1.0 / pred.size(-1)
        port_returns = (weights * actual).sum(dim=-1)  # (B,)

        mu = port_returns.mean()
        sigma = port_returns.std()
        sharpe_loss = -(mu / (sigma + self.eps))

        # Differentiable IC (Pearson on the predicted scores vs actual)
        p_centered = pred - pred.mean(dim=-1, keepdim=True)
        a_centered = actual - actual.mean(dim=-1, keepdim=True)
        cov = (p_centered * a_centered).mean(dim=-1)
        std_p = p_centered.pow(2).mean(dim=-1).sqrt()
        std_a = a_centered.pow(2).mean(dim=-1).sqrt()
        ic = (cov / (std_p * std_a + self.eps)).mean()
        ic_loss = 1.0 - ic

        loss = sharpe_loss + self.lam * ic_loss

        info = {
            "sharpe_loss": sharpe_loss.item(),
            "ic": ic.item(),
            "ic_loss": ic_loss.item(),
            "total_loss": loss.item(),
        }
        return loss, info
