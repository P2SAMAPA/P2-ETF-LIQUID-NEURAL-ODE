"""app.py — Streamlit dashboard for P2-ETF-LIQUID-NEURAL-ODE engine.

Reads daily scores from:
    P2SAMAPA/p2-etf-liquid-neural-ode-results  (HuggingFace Dataset)

Deploy:  streamlit run app.py
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LIQUID-NEURAL-ODE · P2Quant",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
HF_RESULTS_REPO = "P2SAMAPA/p2-etf-liquid-neural-ode-results"
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
FAST_THRESHOLD = 2.0  # matches tau_monitor.py — sigmoid tau range (0.1, 10.0)
SLOW_THRESHOLD = 7.0

FI_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQ_TICKERS = [
    "SPY",
    "QQQ",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLY",
    "XLP",
    "XLU",
    "GDX",
    "XME",
    "IWM",
    "IWF",
    "XSD",
    "XBI",
    "XLB",
    "XLRE",
]

UNIVERSE_COLOURS = {
    "TLT": "#1B4F8A",
    "VCIT": "#2E86C1",
    "LQD": "#148F77",
    "HYG": "#B7950B",
    "VNQ": "#6C3483",
    "GLD": "#CA6F1E",
    "SLV": "#717D7E",
    "SPY": "#C0392B",
    "QQQ": "#922B21",
    "XLK": "#1A5276",
    "XLF": "#117A65",
    "XLE": "#784212",
    "XLV": "#1D8348",
    "XLI": "#2471A3",
    "XLY": "#7D6608",
    "XLP": "#6E2F83",
    "XLU": "#17202A",
    "GDX": "#B7950B",
    "XME": "#5D6D7E",
    "IWM": "#E74C3C",
    "IWF": "#1ABC9C",
    "XSD": "#8E44AD",
    "XBI": "#E67E22",
    "XLB": "#2ECC71",
    "XLRE": "#F39C12",
}

# ── Helpers ───────────────────────────────────────────────────────────────────


@st.cache_data(ttl=3600, show_spinner="Loading results from HuggingFace…")
def load_results() -> tuple[pd.DataFrame, bool]:
    """Load scores via direct parquet URL — no datasets/pyarrow dependency.

    Uses huggingface_hub to list files then reads parquet with pandas directly.
    This avoids the datasets library entirely, fixing Python 3.14 / PyArrow
    build failures on Streamlit Cloud where cmake is unavailable.
    """
    try:
        import io
        import requests
        from huggingface_hub import HfApi

        hf_token = os.environ.get("HF_TOKEN")
        api = HfApi()

        # List all parquet files in the dataset repo
        files = list(
            api.list_repo_files(
                HF_RESULTS_REPO,
                repo_type="dataset",
                token=hf_token if hf_token else None,
            )
        )
        parquet_files = [f for f in files if f.endswith(".parquet")]

        if not parquet_files:
            raise ValueError("No parquet files found in HF dataset repo.")

        # Download and concatenate all parquet shards directly
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
        dfs = []
        for fname in parquet_files:
            url = f"https://huggingface.co/datasets/{HF_RESULTS_REPO}/resolve/main/{fname}"
            resp = requests.get(url, headers=headers, timeout=60)
            resp.raise_for_status()
            dfs.append(pd.read_parquet(io.BytesIO(resp.content)))

        df = pd.concat(dfs, ignore_index=True)

        if df.empty:
            raise ValueError("Dataset loaded but contains no rows.")

        df["date"] = pd.to_datetime(df["date"])

        # Ensure universe column exists
        if "universe" not in df.columns:
            df["universe"] = df["ticker"].apply(lambda t: "fi" if t in FI_TICKERS else "equity")

        # Deduplicate — parallel CI jobs can push duplicate (date, ticker, universe) rows
        dedup_cols = [c for c in ["date", "ticker", "universe"] if c in df.columns]
        if dedup_cols:
            df = df.drop_duplicates(subset=dedup_cols, keep="last")

        return df.sort_values("date"), False  # (data, is_demo=False)

    except Exception as e:
        st.warning(f"Could not load HF dataset ({e}). Showing synthetic demo data.")
        return _synthetic_demo(), True  # (data, is_demo=True)


def _synthetic_demo() -> pd.DataFrame:
    """Generate synthetic demo data when HF dataset is unavailable."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=120)
    tickers = FI_TICKERS + EQ_TICKERS
    rows = []
    for date in dates:
        scores = rng.normal(0, 1, len(tickers))
        tau_m = rng.uniform(0.5, 6.0)
        ff = float(rng.beta(2, 5))
        for i, ticker in enumerate(tickers):
            universe = "fi" if ticker in FI_TICKERS else "equity"
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "score_raw": float(scores[i]),
                    "score_adj": float(scores[i]),
                    "ci_lower": float(scores[i] - rng.uniform(0.2, 0.5)),
                    "ci_upper": float(scores[i] + rng.uniform(0.2, 0.5)),
                    "tau_mean": tau_m,
                    "fast_frac": ff,
                    "rank": int(np.argsort(np.argsort(-scores))[i] + 1),
                    "universe": universe,
                }
            )
    return pd.DataFrame(rows)


def regime_label(fast_frac: float, slow_frac: float) -> tuple[str, str]:
    """Return (label, colour) for the current regime."""
    if fast_frac > 0.5:
        return "⚡ FAST — Volatile", "#E74C3C"
    if slow_frac > 0.5:
        return "🐢 SLOW — Trending", "#27AE60"
    return "〰️ MIXED", "#F39C12"


def equity_curve(daily_returns: pd.Series) -> pd.Series:
    return (1 + daily_returns).cumprod()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    (
        st.image(
            "https://huggingface.co/datasets/P2SAMAPA/p2-etf-liquid-neural-ode-results/resolve/main/banner.png",
            use_column_width=True,
        )
        if False
        else None
    )  # skip if banner not present

    st.markdown("## 💧 LIQUID-NEURAL-ODE")
    st.markdown(
        "**P2Quant Engine #88**  \n"
        "Liquid Time-Constant Network  \n"
        "Neural Circuit Policy (AutoNCP)"
    )
    st.divider()

    universe_opt = st.selectbox(
        "Universe",
        ["combined", "fi", "equity"],
        index=0,
        help="FI/Commodities, Equity Sectors, or Combined",
    )

    lookback = st.slider(
        "Lookback (trading days)",
        min_value=5,
        max_value=120,
        value=60,
        step=5,
    )

    top_n = st.slider("Top N ETFs to highlight", min_value=1, max_value=10, value=5)

    show_ci = st.toggle("Show 95% credible intervals", value=True)

    st.divider()
    st.markdown(
        "**Data source**  \n"
        f"[{HF_RESULTS_REPO}](https://huggingface.co/datasets/{HF_RESULTS_REPO})  \n\n"
        "**Engine repo**  \n"
        "[P2SAMAPA/P2-ETF-LIQUID-NEURAL-ODE](https://github.com/P2SAMAPA/P2-ETF-LIQUID-NEURAL-ODE)"
    )
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

# ── Load & filter data ────────────────────────────────────────────────────────
df_all, is_demo = load_results()

if is_demo:
    st.info(
        "📊 Displaying **synthetic demo data**. "
        "Live results will appear automatically after the next GitHub Actions training run "
        "pushes scores to the HuggingFace dataset."
    )

# Universe filter
if universe_opt != "combined":
    df_all = df_all[df_all["universe"] == universe_opt]

# Date filter
max_date = df_all["date"].max()
cutoff = max_date - pd.Timedelta(days=lookback * 1.5)
df = df_all[df_all["date"] >= cutoff].copy()
dates_avail = sorted(df["date"].unique())

# Latest day
latest_date = df["date"].max()
df_today = df[df["date"] == latest_date].sort_values("rank")

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("# 💧 Liquid Neural ODE · ETF Rankings")
st.caption(
    f"Engine: **LIQUID-NEURAL-ODE** · Universe: **{universe_opt.upper()}** · "
    f"Latest date: **{latest_date.date()}** · "
    f"{len(df_today)} ETFs scored" + (" · ⚠️ DEMO DATA" if is_demo else " · ✅ Live")
)

# ── KPI row ───────────────────────────────────────────────────────────────────
tau_today = df_today["tau_mean"].mean()
fast_today = df_today["fast_frac"].mean()
slow_today = (
    1.0 - fast_today - max(0, 1.0 - fast_today - df_today.get("slow_frac", pd.Series([0.0])).mean())
)
rlabel, rcolour = regime_label(fast_today, slow_today)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Today's Regime", rlabel)
col2.metric("Mean τ (days)", f"{tau_today:.2f}")
col3.metric("Fast Neurons", f"{fast_today*100:.1f}%")
col4.metric("ETFs Ranked", len(df_today))
col5.metric("Universe", universe_opt.upper())

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Today's Rankings",
        "📈 Score History",
        "🌡️ Regime Monitor",
        "⚖️ Signal Confidence",
        "ℹ️ Engine Info",
    ]
)

# ═══════════════════════════════════════════════════════════════
# TAB 1 — Today's Rankings
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.subheader(f"Rankings for {latest_date.date()}")

    try:
        import plotly.express as px
        import plotly.graph_objects as go

        # Sort by score descending so best ETFs are on the left
        df_plot = df_today.sort_values("score_adj", ascending=False).reset_index(drop=True)
        colours_sorted = [UNIVERSE_COLOURS.get(t, "#888") for t in df_plot["ticker"]]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=df_plot["ticker"],
                y=df_plot["score_adj"],
                marker_color=colours_sorted,
                marker_line_width=0,
                text=df_plot.apply(lambda r: f"#{int(r['rank'])}  {r['score_adj']:.2f}", axis=1),
                textposition="outside",
                textfont=dict(size=10),
                name="Score (z)",
            )
        )
        if show_ci and "ci_lower" in df_plot.columns:
            fig.update_traces(
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=(df_plot["ci_upper"] - df_plot["score_adj"]).clip(lower=0).tolist(),
                    arrayminus=(df_plot["score_adj"] - df_plot["ci_lower"]).clip(lower=0).tolist(),
                    color="rgba(80,80,80,0.45)",
                    thickness=1.5,
                    width=4,
                )
            )
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(128,128,128,0.5)", line_width=1)
        fig.update_layout(
            title=dict(
                text=f"ETF Rankings — {latest_date.date()} · sorted best → worst",
                font=dict(size=14),
            ),
            xaxis=dict(title="ETF", tickangle=-35, tickfont=dict(size=11)),
            yaxis=dict(title="Score (z-score)", zeroline=False),
            showlegend=False,
            height=540,
            margin=dict(t=70, b=90, l=60, r=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
            bargap=0.3,
        )
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.bar_chart(df_today.set_index("ticker")["score_adj"])

    # Data table
    display_cols = [
        "rank",
        "ticker",
        "score_adj",
        "score_raw",
        "ci_lower",
        "ci_upper",
        "tau_mean",
        "fast_frac",
    ]
    display_cols = [c for c in display_cols if c in df_today.columns]
    st.dataframe(
        df_today[display_cols]
        .rename(
            columns={
                "score_adj": "Score (z)",
                "score_raw": "Score (raw)",
                "ci_lower": "CI Lower",
                "ci_upper": "CI Upper",
                "tau_mean": "τ mean",
                "fast_frac": "Fast %",
            }
        )
        .style.format(
            {
                "Score (z)": "{:.3f}",
                "Score (raw)": "{:.4f}",
                "CI Lower": "{:.3f}",
                "CI Upper": "{:.3f}",
                "τ mean": "{:.2f}",
                "Fast %": "{:.1%}",
            }
        ),
        use_container_width=True,
        height=350,
    )

# ═══════════════════════════════════════════════════════════════
# TAB 2 — Score History
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Score History — Top ETFs")

    # Pivot to wide format
    pivot = df.pivot_table(index="date", columns="ticker", values="score_adj")
    pivot = pivot.sort_index()

    # Identify top N by mean absolute score
    top_tickers = pivot.abs().mean().nlargest(top_n).index.tolist()

    try:
        import plotly.graph_objects as go

        fig2 = go.Figure()
        for ticker in top_tickers:
            if ticker not in pivot.columns:
                continue
            s = pivot[ticker].dropna()
            fig2.add_trace(
                go.Scatter(
                    x=s.index,
                    y=s.values,
                    mode="lines",
                    name=ticker,
                    line=dict(width=2, color=UNIVERSE_COLOURS.get(ticker, "#888")),
                )
            )
        fig2.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
        fig2.update_layout(
            title=f"Top {top_n} ETFs by Mean |Score| — Last {lookback} Trading Days",
            xaxis_title="Date",
            yaxis_title="Score (z-score)",
            height=420,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig2, use_container_width=True)
    except ImportError:
        st.line_chart(pivot[top_tickers])

    # Rank heatmap
    st.subheader("Rank Heatmap")
    rank_pivot = df.pivot_table(index="date", columns="ticker", values="rank")
    rank_pivot = rank_pivot.sort_index().tail(lookback)

    try:
        import plotly.express as px

        fig_heat = px.imshow(
            rank_pivot.T,
            color_continuous_scale="RdYlGn_r",
            labels=dict(x="Date", y="ETF", color="Rank"),
            title="ETF Rank Over Time (1 = Best)",
            aspect="auto",
            height=400,
        )
        fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_heat, use_container_width=True)
    except ImportError:
        st.dataframe(rank_pivot.T)

# ═══════════════════════════════════════════════════════════════
# TAB 3 — Regime Monitor
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Adaptive τ Regime Monitor")
    st.caption(
        "The LTC network's mean time constant τ is a natural regime indicator.  \n"
        "**Fast (τ < 0.3d):** volatile, intraday speed.  "
        "**Slow (τ > 5d):** calm, position-trade speed."
    )

    # Daily aggregated regime stats
    regime_df = (
        df.groupby("date")
        .agg(tau_mean=("tau_mean", "mean"), fast_frac=("fast_frac", "mean"))
        .reset_index()
        .sort_values("date")
        .tail(lookback)
    )

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig3 = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=(
                "Mean τ (days) — Network Tempo",
                "Fast-Neuron Fraction — Volatility Proxy",
            ),
            vertical_spacing=0.12,
        )

        # τ trace
        fig3.add_trace(
            go.Scatter(
                x=regime_df["date"],
                y=regime_df["tau_mean"],
                mode="lines",
                fill="tozeroy",
                name="τ mean",
                line=dict(color="#2E86C1", width=1.5),
                fillcolor="rgba(46,134,193,0.15)",
            ),
            row=1,
            col=1,
        )
        fig3.add_hline(
            y=FAST_THRESHOLD,
            line_dash="dash",
            line_color="#E74C3C",
            annotation_text="Fast threshold",
            row=1,
            col=1,
        )
        fig3.add_hline(
            y=SLOW_THRESHOLD,
            line_dash="dash",
            line_color="#27AE60",
            annotation_text="Slow threshold",
            row=1,
            col=1,
        )

        # Fast fraction trace
        fast_colours = [
            "#E74C3C" if v > 0.5 else ("#27AE60" if v < 0.2 else "#F39C12")
            for v in regime_df["fast_frac"]
        ]
        fig3.add_trace(
            go.Bar(
                x=regime_df["date"],
                y=regime_df["fast_frac"],
                name="Fast frac",
                marker_color=fast_colours,
            ),
            row=2,
            col=1,
        )

        fig3.update_layout(
            height=520,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig3.update_yaxes(title_text="τ (days)", row=1, col=1)
        fig3.update_yaxes(title_text="Fast fraction", tickformat=".0%", row=2, col=1)
        st.plotly_chart(fig3, use_container_width=True)

    except ImportError:
        st.line_chart(regime_df.set_index("date")[["tau_mean", "fast_frac"]])

    # Regime summary table
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg τ (period)", f"{regime_df['tau_mean'].mean():.2f}d")
    c2.metric("% Fast-regime days", f"{(regime_df['fast_frac'] > 0.5).mean()*100:.1f}%")
    c3.metric("Regime today", rlabel)

# ═══════════════════════════════════════════════════════════════
# TAB 4 — Signal Confidence
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Signal Confidence — MC Dropout 95% Credible Intervals")
    st.caption(
        "Wider intervals = higher epistemic uncertainty (model less confident).  \n"
        "Narrow intervals on large |score| = high-conviction signals."
    )

    if "ci_lower" in df_today.columns and "ci_upper" in df_today.columns:
        df_ci = df_today.copy()
        df_ci["ci_width"] = df_ci["ci_upper"] - df_ci["ci_lower"]
        df_ci["conviction"] = df_ci["score_adj"].abs() / (df_ci["ci_width"] + 1e-8)
        df_ci = df_ci.sort_values("conviction", ascending=False)

        try:
            import plotly.graph_objects as go

            fig4 = go.Figure()
            for _, row in df_ci.iterrows():
                colour = UNIVERSE_COLOURS.get(row["ticker"], "#888")
                fig4.add_trace(
                    go.Scatter(
                        x=[row["ticker"], row["ticker"]],
                        y=[row["ci_lower"], row["ci_upper"]],
                        mode="lines",
                        line=dict(color=colour, width=3),
                        showlegend=False,
                    )
                )
                fig4.add_trace(
                    go.Scatter(
                        x=[row["ticker"]],
                        y=[row["score_adj"]],
                        mode="markers",
                        marker=dict(size=10, color=colour, symbol="diamond"),
                        name=row["ticker"],
                        showlegend=False,
                    )
                )
            fig4.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
            fig4.update_layout(
                title="Score ± 95% CI per ETF",
                xaxis_title="ETF",
                yaxis_title="Score (z-score)",
                height=420,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig4, use_container_width=True)
        except ImportError:
            st.dataframe(
                df_ci[
                    [
                        "ticker",
                        "score_adj",
                        "ci_lower",
                        "ci_upper",
                        "ci_width",
                        "conviction",
                    ]
                ]
            )

        # Conviction table
        st.subheader("Conviction Rankings")
        st.dataframe(
            df_ci[["ticker", "score_adj", "ci_width", "conviction"]]
            .rename(
                columns={
                    "score_adj": "Score (z)",
                    "ci_width": "CI Width",
                    "conviction": "Conviction",
                }
            )
            .style.format({"Score (z)": "{:.3f}", "CI Width": "{:.3f}", "Conviction": "{:.2f}"}),
            use_container_width=True,
            height=320,
        )
    else:
        st.info("Credible interval columns not found in dataset.")

# ═══════════════════════════════════════════════════════════════
# TAB 5 — Engine Info
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Engine Specification")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
**Engine ID:** LIQUID-NEURAL-ODE
**Category:** Liquid Time-Constant Networks
**Suite Version:** P2Quant v8 · April 2026

**Core Algorithm**
- Liquid Time-Constant Network (LTC-NN)
- Neural Circuit Policy (AutoNCP) wiring
- torchdiffeq dopri5 + adjoint backprop
- Closed-form approximation for inference

**Architecture**
- Sensory neurons: 32
- Inter neurons: 24
- Command neurons: 16
- Motor neurons: N_etf
- Wiring sparsity: 0.6

**Training**
- Loss: Differentiable Sharpe + IC
- Optimiser: AdamW, cosine LR
- Epochs: 300, early stopping (patience 25)
- Train: 2008–2019 · Val: 2020–2022 · Test: 2023–2026
        """)

    with col_b:
        st.markdown("""
**Key Innovation**

The network's time constants τ(x,h) adapt to market conditions:

```
dh/dt = −[1/τ(x,h)] · h  +  f(x,h,t) · A(x)
τ(x,h) = τ_min + softplus(W_τ · [x ‖ h] + b_τ)
```

- **VIX spike** → τ → 0.2d (day-trader mode)
- **Calm trend** → τ → 7d (position-trade mode)

**Scoring**
- Raw: motor neuron output per ETF
- Regime-adjusted: × (1 + 0.3 × fast_frac)
- Final: cross-sectional z-score

**Uncertainty**
- MC Dropout × 50 passes
- 95% credible intervals per ETF

**Links**
        """)
        st.link_button(
            "📦 GitHub Repo",
            "https://github.com/P2SAMAPA/P2-ETF-LIQUID-NEURAL-ODE",
        )
        st.link_button(
            "🤗 Results Dataset",
            f"https://huggingface.co/datasets/{HF_RESULTS_REPO}",
        )
        st.link_button(
            "🤗 Input Data",
            f"https://huggingface.co/datasets/{HF_DATA_REPO}",
        )

    st.divider()
    st.subheader("Universe Coverage")
    uni_col1, uni_col2 = st.columns(2)
    with uni_col1:
        st.markdown("**FI / Commodities** (benchmark: AGG)")
        st.code("  ".join(FI_TICKERS))
    with uni_col2:
        st.markdown("**Equity Sectors** (benchmark: SPY)")
        st.code("  ".join(EQ_TICKERS))

    st.divider()
    st.subheader("References")
    st.markdown("""
- Lechner et al. (2020) — *Liquid Time-Constant Networks* · NeurIPS 2020
- Hasani et al. (2021) — *Closed-form Continuous-time Neural Networks* · NeurIPS 2021
- Chen et al. (2018) — *Neural Ordinary Differential Equations* · NeurIPS 2018
- Kidger et al. (2020) — *Neural Controlled Differential Equations* · NeurIPS 2020
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "P2Quant Engine Master Map v8 · April 2026 · "
    "[P2SAMAPA](https://github.com/P2SAMAPA) · "
    f"Last refreshed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
)
