"""export_onnx.py — Export closed-form LTC model to ONNX.

Uses the closed-form path only (no ODE solver needed at export).
Suitable for low-latency inference environments.

Usage:
    python export_onnx.py --universe combined --output ltc_model.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import load_config
from loader import UNIVERSE_MAP
from logging_utils import get_logger
from ltc_model import LTCModel

log = get_logger("ltc_engine.export")


def export(universe: str = "combined", config_path: str = "ltc_config.toml",
           output_path: str = "ltc_model.onnx") -> None:
    cfg    = load_config(config_path)
    device = torch.device("cpu")

    tickers = UNIVERSE_MAP[universe]
    n_etf   = len(tickers)
    m       = cfg.model

    model = LTCModel(
        input_dim=m.input_dim, hidden_dim=m.n_neurons, n_etf=n_etf,
        tau_min=m.tau_min, tau_max=m.tau_max, dropout=0.0,
        use_closed_form=True,
    ).to(device)

    ckpt = torch.load(Path(cfg.output.checkpoint_dir) / "best_val_sharpe.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Dummy inputs
    dummy_x  = torch.randn(1, cfg.data.window, m.input_dim)
    dummy_dt = torch.ones(1, cfg.data.window)

    torch.onnx.export(
        model,
        (dummy_x, dummy_dt),
        output_path,
        input_names=["features", "delta_t"],
        output_names=["scores", "tau_dist"],
        dynamic_axes={
            "features":  {0: "batch", 1: "window"},
            "delta_t":   {0: "batch", 1: "window"},
            "scores":    {0: "batch"},
            "tau_dist":  {0: "batch"},
        },
        opset_version=17,
    )
    log.info("ONNX model exported → %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="combined")
    parser.add_argument("--config",   default="ltc_config.toml")
    parser.add_argument("--output",   default="ltc_model.onnx")
    args = parser.parse_args()
    export(args.universe, args.config, args.output)
