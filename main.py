"""main.py — CLI orchestrator for P2-ETF-LIQUID-NEURAL-ODE engine.

Usage:
    python main.py --universe combined --mode train --config ltc_config.toml
    python main.py --universe equity   --mode eval  --checkpoint checkpoints/best_val_sharpe.pt
    python main.py --universe fi       --mode infer
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import load_config
from loader import load_master_data, get_universe_data, UNIVERSE_MAP
from logging_utils import get_logger
from ltc_model import LTCModel
from preprocessor import preprocess
from seed import set_seed
from splitter import make_dataloaders

log = get_logger("ltc_engine.main")


def build_model(cfg, n_etf: int) -> LTCModel:
    m = cfg.model
    return LTCModel(
        input_dim       = m.input_dim,
        hidden_dim      = m.n_neurons,
        n_etf           = n_etf,
        tau_min         = m.tau_min,
        tau_max         = m.tau_max,
        dropout         = m.dropout,
        use_closed_form = m.use_closed_form,
        use_adjoint     = cfg.ode.adjoint,
        ode_method      = cfg.ode.method,
        rtol            = cfg.ode.rtol,
        atol            = cfg.ode.atol,
    )


def cmd_train(args) -> None:
    cfg = load_config(args.config)
    set_seed(cfg.training.seed)

    df = load_master_data(cfg.data.hf_repo, cfg.data.parquet_file)
    returns_df, macro_df = get_universe_data(df, args.universe)
    features, delta_t    = preprocess(returns_df, macro_df, cfg.data.window)

    tickers = UNIVERSE_MAP[args.universe]
    n_etf   = len([t for t in tickers if f"{t}_log_return" in returns_df.columns])

    train_loader, val_loader, _ = make_dataloaders(
        features, returns_df, delta_t,
        window     = cfg.data.window,
        batch_size = cfg.training.batch_size,
        train_end  = cfg.data.train_end,
        val_end    = cfg.data.val_end,
    )

    model = build_model(cfg, n_etf)
    log.info("Model parameters: %s", sum(p.numel() for p in model.parameters()))

    from trainer import train
    train(model, train_loader, val_loader, cfg)


def cmd_eval(args) -> None:
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_master_data(cfg.data.hf_repo, cfg.data.parquet_file)
    returns_df, macro_df = get_universe_data(df, args.universe)
    features, delta_t    = preprocess(returns_df, macro_df, cfg.data.window)

    tickers = UNIVERSE_MAP[args.universe]
    n_etf   = len([t for t in tickers if f"{t}_log_return" in returns_df.columns])

    _, _, test_loader = make_dataloaders(
        features, returns_df, delta_t,
        window=cfg.data.window, batch_size=cfg.training.batch_size,
        train_end=cfg.data.train_end, val_end=cfg.data.val_end,
    )

    model = build_model(cfg, n_etf)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)

    test_dates = features.index[features.index > cfg.data.val_end].tolist()

    from backtester import run_backtest
    run_backtest(model, test_loader, tickers[:n_etf], test_dates,
                 device, cfg.scoring.mc_passes, cfg.scoring.alpha_amp,
                 cfg.output.results_dir)


def cmd_infer(args) -> None:
    from infer_daily import run_daily_inference
    run_daily_inference(args.universe, args.config)


def cli_train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe",   default="combined")
    parser.add_argument("--config",     default="ltc_config.toml")
    cmd_train(parser.parse_args())


def cli_eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe",   default="combined")
    parser.add_argument("--config",     default="ltc_config.toml")
    parser.add_argument("--checkpoint", default="checkpoints/best_val_sharpe.pt")
    cmd_eval(parser.parse_args())


def cli_infer():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="combined")
    parser.add_argument("--config",   default="ltc_config.toml")
    cmd_infer(parser.parse_args())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P2-ETF-LIQUID-NEURAL-ODE")
    parser.add_argument("--universe",   default="combined",                     choices=["fi","equity","combined"])
    parser.add_argument("--mode",       default="train",                         choices=["train","eval","infer"])
    parser.add_argument("--config",     default="ltc_config.toml")
    parser.add_argument("--checkpoint", default="checkpoints/best_val_sharpe.pt")
    args = parser.parse_args()

    {"train": cmd_train, "eval": cmd_eval, "infer": cmd_infer}[args.mode](args)
