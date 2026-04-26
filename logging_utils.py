"""logging_utils.py — Structured logging for LTC engine.

Named logging_utils to avoid shadowing stdlib logging module in flat root.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

_LOG_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "ltc_engine", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger. Idempotent — safe to call multiple times."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def log_epoch(
    logger: logging.Logger,
    epoch: int,
    train_loss: float,
    val_sharpe: float,
    tau_mean: float,
    fast_frac: float,
    universe: str,
) -> None:
    """Log a structured epoch summary line."""
    logger.info(
        "epoch=%d | train_loss=%.4f | val_sharpe=%.4f | "
        "tau_mean=%.3f | fast_frac=%.3f | universe=%s",
        epoch,
        train_loss,
        val_sharpe,
        tau_mean,
        fast_frac,
        universe,
    )


def try_wandb_log(metrics: dict[str, Any], step: int | None = None) -> None:
    """Optionally log to wandb if it is installed and initialised."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass
