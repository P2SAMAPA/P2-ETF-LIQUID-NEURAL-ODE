"""trainer.py — Training loop for LTC model.

Uses torchdiffeq adjoint method for memory-efficient backprop.
Early stopping on val Sharpe. Cosine LR with warmup.
"""

from __future__ import annotations

import math

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


def _is_finite(val: float) -> bool:
    return math.isfinite(val)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: LTCConfig,
    device: torch.device | None = None,
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
    scheduler = build_scheduler(optimizer, cfg.training.epochs, cfg.training.warmup_epochs)
    early_stop = EarlyStopping(patience=cfg.training.patience)
    checkpointer = ModelCheckpoint(cfg.output.checkpoint_dir)
    tau_logger = TauDistributionLogger(f"{cfg.output.results_dir}/tau_log.json")

    history: dict[str, list[float]] = {"train_loss": [], "val_sharpe": [], "tau_mean": []}

    # Save initial weights so we can roll back if NaN occurs
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    nan_streak = 0
    MAX_NAN_STREAK = 3  # abort training if NaN persists this many epochs

    for epoch in range(1, cfg.training.epochs + 1):

        # ── Training pass ────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        tau_accum = 0.0
        n_batches = 0
        skipped_batches = 0

        for x, dt, y in train_loader:
            x, dt, y = x.to(device), dt.to(device), y.to(device)
            optimizer.zero_grad()

            scores, tau_dist = model(x, dt)
            loss, info = criterion(scores, y)

            # ── NaN/Inf loss guard ───────────────────────────────────────
            if not torch.isfinite(loss):
                skipped_batches += 1
                log.warning(
                    "Non-finite loss (%.4g) at epoch %d — skipping batch", loss.item(), epoch
                )
                optimizer.zero_grad()
                continue

            loss.backward()

            # ── Gradient NaN guard ───────────────────────────────────────
            grad_ok = all(
                p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
            )
            if not grad_ok:
                skipped_batches += 1
                log.warning("NaN gradient detected at epoch %d — skipping batch", epoch)
                optimizer.zero_grad()
                continue

            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optimizer.step()

            # ── Parameter NaN guard — roll back if weights exploded ──────
            param_ok = all(torch.isfinite(p).all() for p in model.parameters())
            if not param_ok:
                log.warning("NaN parameters after step at epoch %d — rolling back", epoch)
                model.load_state_dict(best_state)
                optimizer.zero_grad()
                skipped_batches += 1
                continue

            epoch_loss += loss.item()
            tau_val = tau_dist.mean().item()
            tau_accum += tau_val if _is_finite(tau_val) else 0.0
            n_batches += 1

        if skipped_batches > 0:
            log.warning(
                "Epoch %d: skipped %d/%d batches due to NaN",
                epoch,
                skipped_batches,
                skipped_batches + n_batches,
            )

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_tau = tau_accum / max(n_batches, 1)

        # ── Validation pass ──────────────────────────────────────────────
        val_sharpe, val_fast_frac, val_slow_frac = _evaluate(model, val_loader, device)

        scheduler.step()

        # ── NaN epoch guard ──────────────────────────────────────────────
        if not _is_finite(val_sharpe) or not _is_finite(avg_loss):
            nan_streak += 1
            log.warning(
                "Epoch %d: NaN metrics (loss=%.4g, val_sharpe=%.4g) — streak %d/%d",
                epoch,
                avg_loss,
                val_sharpe,
                nan_streak,
                MAX_NAN_STREAK,
            )
            if nan_streak >= MAX_NAN_STREAK:
                log.error("NaN persisted for %d epochs — aborting training.", MAX_NAN_STREAK)
                break
            # Use 0.0 as sentinel so early stopping / checkpointer skip this epoch
            val_sharpe = 0.0
            avg_loss = 0.0
        else:
            nan_streak = 0
            # Update best_state only when weights are healthy
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # ── Logging ──────────────────────────────────────────────────────
        log_epoch(log, epoch, avg_loss, val_sharpe, avg_tau, val_fast_frac, universe="?")
        try_wandb_log(
            {
                "train_loss": avg_loss,
                "val_sharpe": val_sharpe,
                "tau_mean": avg_tau,
                "lr": scheduler.get_last_lr()[0],
            },
            step=epoch,
        )

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
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """Run validation pass. Returns (val_sharpe, fast_frac, slow_frac)."""
    model.eval()
    all_scores = []
    all_actuals = []
    all_tau_dist = []

    with torch.no_grad():
        for x, dt, y in loader:
            x, dt, y = x.to(device), dt.to(device), y.to(device)
            scores, tau_dist = model(x, dt)
            all_scores.append(scores.cpu())
            all_actuals.append(y.cpu())
            all_tau_dist.append(tau_dist.cpu())

    scores_cat = torch.cat(all_scores, dim=0)
    actuals_cat = torch.cat(all_actuals, dim=0)
    tau_cat = torch.cat(all_tau_dist, dim=0)

    # Guard: if scores are all NaN return 0.0 instead of propagating NaN
    if not torch.isfinite(scores_cat).any():
        log.warning("All validation scores are non-finite — returning val_sharpe=0.0")
        return 0.0, 0.0, 0.0

    # Replace any residual NaN scores with 0 before computing portfolio weights
    scores_cat = torch.nan_to_num(scores_cat, nan=0.0, posinf=0.0, neginf=0.0)

    # Portfolio returns from scores
    weights = torch.softmax(scores_cat, dim=-1) - 1.0 / scores_cat.size(-1)
    port_r = (weights * actuals_cat).sum(dim=-1).numpy()
    val_s = sharpe(port_r)
    val_s = val_s if _is_finite(val_s) else 0.0

    regime = compute_regime_labels(tau_cat)
    fast_f = regime["fast_frac"].mean().item()
    slow_f = regime["slow_frac"].mean().item()
    fast_f = fast_f if _is_finite(fast_f) else 0.0
    slow_f = slow_f if _is_finite(slow_f) else 0.0

    return val_s, fast_f, slow_f
