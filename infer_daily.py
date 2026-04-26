"""infer_daily.py — Daily inference: score today's ETFs and publish to HF.

Uses the closed-form LTC path for low-latency inference.
Appends one row per ETF to P2SAMAPA/p2-etf-liquid-neural-ode-results.

Usage:
    python infer_daily.py --universe combined
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import load_config
from loader import get_universe_data, load_master_data
from logging_utils import get_logger
from ltc_model import LTCModel
from preprocessor import preprocess
from publisher import push_results, save_results_locally
from ranker import build_daily_ranking
from regime_score import apply_regime_adjustment
from tau_monitor import compute_regime_labels
from uncertainty import uncertainty_pass

log = get_logger("ltc_engine.infer")


def run_daily_inference(universe: str = "combined", config_path: str = "ltc_config.toml") -> None:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    df = load_master_data(cfg.data.hf_repo, cfg.data.parquet_file)
    returns_df, macro_df = get_universe_data(df, universe)
    features, delta_t = preprocess(returns_df, macro_df, cfg.data.window)

    n_etf = len(returns_df.columns)
    input_dim = features.shape[1]
    active_tickers = list(returns_df.columns)
    log.info("n_etf=%d  input_dim=%d", n_etf, input_dim)

    # Load model — override to use closed_form for speed
    m = cfg.model
    model = LTCModel(
        input_dim=input_dim,
        hidden_dim=m.n_neurons,
        n_etf=n_etf,
        tau_min=m.tau_min,
        tau_max=m.tau_max,
        dropout=m.dropout,
        use_closed_form=True,  # fast inference
    ).to(device)

    ckpt_path = Path(cfg.output.checkpoint_dir) / "best_val_sharpe.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run train.py first.")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    log.info("Loaded checkpoint from epoch %d (val_sharpe=%.4f)", ckpt["epoch"], ckpt["val_sharpe"])

    # Build last window
    last_window = features.iloc[-cfg.data.window :].values.astype("float32")
    last_dt = delta_t.iloc[-cfg.data.window :].values.astype("float32")
    today = features.index[-1]

    x = torch.tensor(last_window, device=device).unsqueeze(0)  # (1, W, D)
    dt = torch.tensor(last_dt, device=device).unsqueeze(0)  # (1, W)

    # Inference
    unc = uncertainty_pass(model, x, dt, cfg.scoring.mc_passes, cfg.scoring.ci_level)
    _, tau_d = model(x, dt)

    scores = unc["mean"].squeeze(0).cpu().numpy()
    ci_lower = unc["ci_lower"].squeeze(0).cpu().numpy()
    ci_upper = unc["ci_upper"].squeeze(0).cpu().numpy()

    regime = compute_regime_labels(tau_d)
    fast_frac = regime["fast_frac"].item()
    tau_mean = regime["tau_mean"].item()

    scores_adj = apply_regime_adjustment(
        torch.tensor(scores), torch.tensor(fast_frac), cfg.scoring.alpha_amp
    ).numpy()

    results_df = build_daily_ranking(
        scores=scores_adj,
        tickers=active_tickers,
        date=today,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        tau_mean=tau_mean,
        fast_frac=fast_frac,
        universe=universe,
    )

    log.info(
        "Top 3 ETFs for %s:\n%s",
        today.date(),
        results_df[["ticker", "score_adj"]].head(3).to_string(),
    )

    try:
        push_results(results_df, cfg.output.hf_results_repo)
    except Exception as e:
        log.warning("HF push failed (%s) — saving locally.", e)
        save_results_locally(results_df, cfg.output.results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="combined", choices=["fi", "equity", "combined"])
    parser.add_argument("--config", default="ltc_config.toml")
    args = parser.parse_args()
    run_daily_inference(args.universe, args.config)
