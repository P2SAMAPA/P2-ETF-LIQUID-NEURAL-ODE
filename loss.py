"""loss.py — Differentiable Sharpe + IC loss for LTC training."""

from __future__ import annotations

import torch
import torch.nn as nn


class SharpeLoss(nn.Module):
    """Differentiable Sharpe ratio loss with IC regularisation.

    L = −E[r] / sqrt(Var[r] + eps)  +  lam * (1 - IC)

    where IC is the differentiable Pearson correlation between
    predicted scores and actual returns.

    Args:
        lam: IC regularisation weight (default 0.1).
        eps: Numerical stability constant.
    """

    def __init__(self, lam: float = 0.1, eps: float = 1e-6) -> None:
        # FIX: eps raised from 1e-8 to 1e-6 — at 1e-8 the division by
        # (std + eps) still explodes when std is ~1e-7 at initialisation.
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
        # Long-short portfolio return
        weights = torch.softmax(pred, dim=-1) - 1.0 / pred.size(-1)
        port_returns = (weights * actual).sum(dim=-1)  # (B,)

        mu = port_returns.mean()

        # FIX: unbiased=False — std() with unbiased=True (default) returns NaN
        # for batch_size=1 because it divides by (N-1)=0.
        sigma = port_returns.std(unbiased=False).clamp(min=self.eps)
        sharpe_loss = -(mu / sigma)

        # Differentiable IC (Pearson correlation)
        p_centered = pred - pred.mean(dim=-1, keepdim=True)
        a_centered = actual - actual.mean(dim=-1, keepdim=True)
        cov = (p_centered * a_centered).mean(dim=-1)

        # FIX: clamp avoids NaN when all scores are identical at init (std=0)
        std_p = p_centered.pow(2).mean(dim=-1).sqrt().clamp(min=self.eps)
        std_a = a_centered.pow(2).mean(dim=-1).sqrt().clamp(min=self.eps)

        ic = (cov / (std_p * std_a)).mean().clamp(-1.0, 1.0)
        ic_loss = 1.0 - ic

        loss = sharpe_loss + self.lam * ic_loss

        # Final guard — return zero-grad loss if still non-finite
        if not torch.isfinite(loss):
            loss = torch.tensor(0.0, requires_grad=True, device=pred.device)

        info = {
            "sharpe_loss": sharpe_loss.item() if torch.isfinite(sharpe_loss) else 0.0,
            "ic": ic.item() if torch.isfinite(ic) else 0.0,
            "ic_loss": ic_loss.item() if torch.isfinite(ic_loss) else 0.0,
            "total_loss": loss.item(),
        }
        return loss, info
