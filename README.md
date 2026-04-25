# P2-ETF-LIQUID-NEURAL-ODE

**P2Quant Engine #88 · Liquid Time-Constant Network (LTC-NN) · Neural Circuit Policy (NCP)**

> Adaptive-τ continuous-time ETF ranking engine. The network's internal time constants adjust to market conditions — fast during volatility spikes, slow during calm trending regimes.

---

## Engine Summary

| Field | Value |
|---|---|
| Engine ID | LIQUID-NEURAL-ODE |
| Category | Liquid Time-Constant Networks |
| Core Algorithm | LTC-NN + Neural Circuit Policy (AutoNCP) + torchdiffeq dopri5 |
| Universes | FI/Commodities · Equity Sectors · Combined |
| Output | Cross-sectional z-score ranking per ETF, daily |
| Input Data | `P2SAMAPA/fi-etf-macro-signal-master-data` |
| Results Dataset | `P2SAMAPA/p2-etf-liquid-neural-ode-results` |
| Train Period | 2008–2019 |
| Val Period | 2020–2022 |
| Test Period | 2023–2026 |

---

## Key Innovation

Unlike NCDE/NSDE (fixed ODE architecture), the Liquid Time-Constant Network has **learnable, input-dependent time constants τ(x,h)** per neuron:

```
dh/dt = −[1/τ(x,h)] · h  +  f(x, h, t) · A(x)
τ(x,h) = τ_min + softplus(W_τ · [x ‖ h] + b_τ)
```

- **VIX spike** → τ collapses toward 0.2 days → network processes at day-trader speed
- **Quiet trend** → τ expands toward 7+ days → network processes at position-trade speed
- τ distribution per day doubles as a **soft regime label**, compatible with HHMM-REGIME

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Train (combined universe)
python train.py --universe combined --config ltc_config.toml

# 3. Evaluate
python evaluate.py --universe combined --checkpoint checkpoints/best_val_sharpe.pt

# 4. Daily inference
python infer_daily.py --universe combined
```

Or use the Makefile:
```bash
make train UNIVERSE=combined
make test
make infer
```

---

## Repository Structure

All files are in the root directory.

| File | Purpose |
|---|---|
| `main.py` | CLI orchestrator |
| `config.py` | Pydantic config loader |
| `loader.py` | HF data loading |
| `preprocessor.py` | MODWT + z-score + Δt encoding |
| `dataset.py` | PyTorch Dataset |
| `splitter.py` | Walk-forward splits |
| `ltc_cell.py` | **Core LTC ODE cell** |
| `closed_form.py` | Fast closed-form approximation |
| `ncp_wiring.py` | AutoNCP wiring policy |
| `ltc_model.py` | Full model assembly |
| `uncertainty.py` | MC Dropout inference |
| `tau_monitor.py` | τ distribution → regime labels |
| `regime_score.py` | Regime-adjusted scoring |
| `loss.py` | Differentiable Sharpe loss |
| `trainer.py` | Training loop (adjoint backprop) |
| `scheduler.py` | Cosine LR with warmup |
| `callbacks.py` | EarlyStopping / Checkpoint |
| `ranker.py` | Cross-sectional z-score ranker |
| `publisher.py` | HF Dataset publisher |
| `backtester.py` | Walk-forward backtest |
| `wavelet.py` | MODWT wrapper |
| `metrics.py` | Sharpe, IC, drawdown, etc. |
| `logging_utils.py` | Structured logging |
| `seed.py` | Reproducibility seed |

---

## Data

**Input:** [`P2SAMAPA/fi-etf-macro-signal-master-data`](https://huggingface.co/datasets/P2SAMAPA/fi-etf-macro-signal-master-data)
- `master_data.parquet` — daily OHLCV, log returns, VIX, DXY, T10Y2Y, HY Spreads, WTI, DTB3 (2008–2026 YTD)

**Output:** [`P2SAMAPA/p2-etf-liquid-neural-ode-results`](https://huggingface.co/datasets/P2SAMAPA/p2-etf-liquid-neural-ode-results)
- Daily scores: `date`, `ticker`, `score`, `score_adj`, `ci_lower`, `ci_upper`, `tau_mean`, `fast_frac`, `universe`

---

## Universe

**FI/Commodities:** TLT, VCIT, LQD, HYG, VNQ, GLD, SLV (benchmark: AGG)
**Equity Sectors:** SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWM (benchmark: SPY)
**Combined:** All of the above

---

## References

- Lechner et al. (2020) — *Liquid Time-Constant Networks* (NeurIPS 2020)
- Hasani et al. (2021) — *Closed-form Continuous-time Neural Networks* (NeurIPS 2021)
- Chen et al. (2018) — *Neural Ordinary Differential Equations* (NeurIPS 2018)
- Kidger et al. (2020) — *Neural Controlled Differential Equations* (NeurIPS 2020)

---

*P2Quant Engine Master Map v8 · April 2026 · P2SAMAPA*
