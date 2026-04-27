"""callbacks.py — Training callbacks: EarlyStopping, ModelCheckpoint, TauLogger."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn

from logging_utils import get_logger

log = get_logger(__name__)


class EarlyStopping:
    """Stop training when val_sharpe has not improved for `patience` epochs."""

    def __init__(self, patience: int = 25, min_delta: float = 1e-3) -> None:
        # FIX: min_delta raised from 1e-4 to 1e-3 — Sharpe oscillates at the
        # noise level early in training; 1e-4 caused early stopping after ~26
        # epochs before the model had a chance to find real signal.
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float("inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_sharpe: float) -> bool:
        # Treat NaN/inf as no improvement
        if not math.isfinite(val_sharpe):
            val_sharpe = 0.0

        if val_sharpe > self.best_score + self.min_delta:
            self.best_score = val_sharpe
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                log.info(
                    "Early stopping triggered after %d epochs without improvement.", self.patience
                )
                self.should_stop = True
        return self.should_stop


class ModelCheckpoint:
    """Save the model whenever val_sharpe improves."""

    def __init__(self, checkpoint_dir: str = "checkpoints") -> None:
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.best_score = -float("inf")
        self.best_path = self.dir / "best_val_sharpe.pt"

    def __call__(self, model: nn.Module, val_sharpe: float, epoch: int) -> bool:
        """Save if improved, or on the very first epoch as a safety baseline.

        Returns:
            True if a new checkpoint was saved.
        """
        # FIX: don't save NaN checkpoints — they poison evaluation
        if not math.isfinite(val_sharpe):
            return False

        if val_sharpe > self.best_score or epoch == 1:
            self.best_score = max(val_sharpe, self.best_score)
            torch.save(
                {"epoch": epoch, "val_sharpe": val_sharpe, "state_dict": model.state_dict()},
                self.best_path,
            )
            log.info(
                "Checkpoint saved  epoch=%d  val_sharpe=%.4f  → %s",
                epoch,
                val_sharpe,
                self.best_path,
            )
            return True
        return False


class TauDistributionLogger:
    """Record mean τ per epoch for post-training regime analysis."""

    def __init__(self, output_path: str = "results/tau_log.json") -> None:
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []

    def log(self, epoch: int, tau_mean: float, fast_frac: float, slow_frac: float) -> None:
        # FIX: sanitise values before logging — NaN/inf break JSON serialisation
        def _safe(v: float) -> float:
            return float(v) if math.isfinite(v) else 0.0

        self.records.append(
            {
                "epoch": epoch,
                "tau_mean": _safe(tau_mean),
                "fast_frac": _safe(fast_frac),
                "slow_frac": _safe(slow_frac),
            }
        )

    def save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self.records, f, indent=2)
        log.info("τ distribution log saved → %s", self.path)
