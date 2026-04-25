---
annotations_creators: []
language: []
license: mit
multilinguality: []
pretty_name: P2-ETF Liquid Neural ODE Results
size_categories:
  - 10K<n<100K
source_datasets: []
tags:
  - finance
  - etf
  - quant
  - liquid-neural-network
  - time-series
task_categories:
  - tabular-regression
task_ids: []
---

# P2-ETF Liquid Neural ODE Results

Daily ETF ranking scores from the **LIQUID-NEURAL-ODE** engine.  
Part of the [P2Quant Engine Suite](https://github.com/P2SAMAPA) — Engine #88, v8 April 2026.

## Dataset Description

Each row represents one ETF's score for one trading day.

### Schema

| Column      | Type    | Description |
|-------------|---------|-------------|
| `date`      | string  | Trading date (YYYY-MM-DD) |
| `ticker`    | string  | ETF ticker symbol |
| `score_raw` | float   | Raw model output (predicted return) |
| `score_adj` | float   | Cross-sectional z-score after regime adjustment |
| `ci_lower`  | float   | 95% credible interval lower bound (MC Dropout) |
| `ci_upper`  | float   | 95% credible interval upper bound (MC Dropout) |
| `tau_mean`  | float   | Mean time constant across neurons (regime proxy) |
| `fast_frac` | float   | Fraction of fast neurons (τ < 0.3d) — high = volatile regime |
| `rank`      | int     | Cross-sectional rank (1 = highest score) |
| `universe`  | string  | Universe: `fi`, `equity`, or `combined` |

### Usage

```python
from datasets import load_dataset
import pandas as pd

ds = load_dataset("P2SAMAPA/p2-etf-liquid-neural-ode-results", split="train")
df = ds.to_pandas()
df["date"] = pd.to_datetime(df["date"])

# Top-ranked ETF per day (combined universe)
top = df[df["universe"] == "combined"].sort_values(["date","rank"])
print(top.groupby("date").first()[["ticker","score_adj","fast_frac"]].tail(10))
```

## Source

- Engine code: [P2SAMAPA/P2-ETF-LIQUID-NEURAL-ODE](https://github.com/P2SAMAPA/P2-ETF-LIQUID-NEURAL-ODE)
- Input data:  [P2SAMAPA/fi-etf-macro-signal-master-data](https://huggingface.co/datasets/P2SAMAPA/fi-etf-macro-signal-master-data)
- Updated: daily, Mon–Fri after US market close (22:30 UTC)

## Citation

```bibtex
@misc{p2quant_ltc_2026,
  title  = {P2Quant LIQUID-NEURAL-ODE Engine},
  author = {P2SAMAPA},
  year   = {2026},
  url    = {https://github.com/P2SAMAPA/P2-ETF-LIQUID-NEURAL-ODE}
}
```
