"""trainer.py — Training loop for LTC model.

Uses torchdiffeq adjoint method for memory-efficient backprop.
Early stopping on val Sharpe. Cosine LR with warmup.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from callbacks import EarlyStopping, ModelCheckpoint, TauDistributionLogger
from config import LTCConfig
from logging_utils import get_logger, log_epoch, try_wandb_log
from loss import SharpeLoss
from metrics import sharpe
from scheduler import build_scheduler
from tau_monitor import compute_regime_labels

log = get_logger(__name__)


def train(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    cfg:          LTCConfig,
    device:       torch.device | None = None,
) -> dict[str, list[float]]:
    """Train the LTC model.

    Args:
        model:        LTCModel instance.
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader.
        cfg:          LTCConfig.
        device:       Torch device (auto-detected if None).

    Returns:
        history: dict of lists — train_loss, val_sharpe, tau_mean per epoch.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on %s", device)
    model = model.to(device)

    criterion = SharpeLoss(lam=cfg.training.sharpe_lambda)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler     = build_scheduler(optimizer, cfg.training.epochs, cfg.training.warmup_epochs)
    early_stop    = EarlyStopping(patience=cfg.training.patience)
    checkpointer  = ModelCheckpoint(cfg.output.checkpoint_dir)
    tau_logger    = TauDistributionLogger(f"{cfg.output.results_dir}/tau_log.json")

    history: dict[str, list[float]] = {"train_loss": [], "val_sharpe": [], "tau_mean": []}

    for epoch in range(1, cfg.training.epochs + 1):

        # ── Training pass ────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        tau_accum  = 0.0
        n_batches  = 0

        for x, dt, y in train_loader:
            x, dt, y = x.to(device), dt.to(device), y.to(device)
            optimizer.zero_grad()

            scores, tau_dist = model(x, dt)
            loss, info       = criterion(scores, y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            tau_accum  += tau_dist.mean().item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_tau  = tau_accum  / max(n_batches, 1)

        # ── Validation pass ──────────────────────────────────────────────
        val_sharpe, val_fast_frac, val_slow_frac = _evaluate(model, val_loader, device)

        scheduler.step()

        # ── Logging ──────────────────────────────────────────────────────
        log_epoch(log, epoch, avg_loss, val_sharpe, avg_tau, val_fast_frac, universe="?")
        try_wandb_log({"train_loss": avg_loss, "val_sharpe": val_sharpe,
                       "tau_mean": avg_tau, "lr": scheduler.get_last_lr()[0]}, step=epoch)

        history["train_loss"].append(avg_loss)
        history["val_sharpe"].append(val_sharpe)
        history["tau_mean"].append(avg_tau)

        tau_logger.log(epoch, avg_tau, val_fast_frac, val_slow_frac)
        checkpointer(model, val_sharpe, epoch)

        if early_stop(val_sharpe):
            log.info("Early stop at epoch %d.", epoch)
            break

    tau_logger.save()
    log.info("Training complete. Best val Sharpe: %.4f", early_stop.best_score)
    return history


def _evaluate(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """Run validation pass. Returns (val_sharpe, fast_frac, slow_frac)."""
    from tau_monitor import compute_regime_labels

    model.eval()
    all_scores   = []
    all_actuals  = []
    all_tau_dist = []

    with torch.no_grad():
        for x, dt, y in loader:
            x, dt, y = x.to(device), dt.to(device), y.to(device)
            scores, tau_dist = model(x, dt)
            all_scores.append(scores.cpu())
            all_actuals.append(y.cpu())
            all_tau_dist.append(tau_dist.cpu())

    scores_cat  = torch.cat(all_scores,   dim=0)
    actuals_cat = torch.cat(all_actuals,  dim=0)
    tau_cat     = torch.cat(all_tau_dist, dim=0)

    # Portfolio returns from scores
    weights = torch.softmax(scores_cat, dim=-1) - 1.0 / scores_cat.size(-1)
    port_r  = (weights * actuals_cat).sum(dim=-1).numpy()
    val_s   = sharpe(port_r)

    regime = compute_regime_labels(tau_cat)
    fast_f = regime["fast_frac"].mean().item()
    slow_f = regime["slow_frac"].mean().item()

    return val_s, fast_f, slow_f
