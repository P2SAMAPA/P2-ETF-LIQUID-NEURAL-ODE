"""loss.py — Differentiable Sharpe + IC loss for LTC training."""

from __future__ import annotations

import torch
import torch.nn as nn


class SharpeLoss(nn.Module):
    """Differentiable Sharpe ratio loss with IC regularisation.

    L = −Sharpe(portfolio)  +  lam * (1 − IC)

    Args:
        lam: IC regularisation weight.
        eps: Numerical stability constant.
    """

    def __init__(self, lam: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        self.lam = lam
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,  # (B, N_etf) predicted scores
        actual: torch.Tensor,  # (B, N_etf) actual next-day returns
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss.

        Note: NaN returns should be handled upstream in LTCDataset.
        This function assumes actual is clean (no NaN).
        """
        # Defensive scrub — should be a no-op if dataset.py is correct
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=-1.0)
        actual = torch.nan_to_num(actual, nan=0.0, posinf=0.5, neginf=-0.5)

        # ── Long-short portfolio return ───────────────────────────────────────
        weights = torch.softmax(pred, dim=-1) - 1.0 / pred.size(-1)
        port_returns = (weights * actual).sum(dim=-1)  # (B,)

        mu = port_returns.mean()
        sigma = port_returns.std(unbiased=False).clamp(min=self.eps)
        sharpe_loss = -(mu / sigma)

        # ── Differentiable IC (Pearson) ───────────────────────────────────────
        p_centered = pred - pred.mean(dim=-1, keepdim=True)
        a_centered = actual - actual.mean(dim=-1, keepdim=True)
        cov = (p_centered * a_centered).mean(dim=-1)
        std_p = p_centered.pow(2).mean(dim=-1).sqrt().clamp(min=self.eps)
        std_a = a_centered.pow(2).mean(dim=-1).sqrt().clamp(min=self.eps)
        ic = (cov / (std_p * std_a)).mean().clamp(-1.0, 1.0)
        ic_loss = 1.0 - ic

        loss = sharpe_loss + self.lam * ic_loss

        if not torch.isfinite(loss):
            loss = ic_loss.clamp(0.0, 2.0)

        info = {
            "sharpe_loss": float(sharpe_loss.item()) if torch.isfinite(sharpe_loss) else 0.0,
            "ic": float(ic.item()) if torch.isfinite(ic) else 0.0,
            "ic_loss": float(ic_loss.item()) if torch.isfinite(ic_loss) else 0.0,
            "total_loss": float(loss.item()),
        }
        return loss, info
