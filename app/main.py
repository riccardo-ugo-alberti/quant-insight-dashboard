from __future__ import annotations

# === Bootstrap path to import "src" both locally and on Streamlit Cloud ===
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]     # .../quant-insight-dashboard
SRC  = ROOT / "src"                             # .../quant-insight-dashboard/src
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)
# ==============================================================================

import io
import json
import datetime as dt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --- Backtest engine & configs ---
try:
    from src.backtest.engine import RollingEWMAEngine
    from src.backtest import (
        BacktestConfig,
        EstimatorConfig,
        ShrinkageConfig,
        MVConfig,
        CostConfig,
        RebalanceConfig,
        RollingEWMAEngine,
    )
except Exception as e:
    st.error(f"Backtest imports failed: {e}")

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Quant Insight Dashboard",
    page_icon=None,
    layout="wide",
)

# --- UI polish: typography & help components ---
st.markdown("""
<style>
/* Subtle, clean typography */
html, body, [class*="css"] { letter-spacing: 0.1px; }
.block-container { padding-top: 2rem; }

/* Section headings */
.qid-h1 { font-size: 1.55rem; font-weight: 700; margin: .25rem 0 .5rem 0; }
.qid-sub { color: #b9c1d9; font-size: .95rem; margin-top: -.35rem; }

/* Cards look (used in Backtest controls) */
.qid-card {
  border: 1px solid rgba(160, 185, 255, 0.18);
  border-radius: 14px;
  padding: .75rem .9rem;
  background: linear-gradient(180deg, rgba(20,20,30,0.55) 0%, rgba(20,20,30,0.35) 100%);
  box-shadow: 0 8px 22px rgba(0,0,0,.25);
  margin-bottom: .75rem;
}
.qid-gap { height: .5rem; }

/* Help badge + popover content */
.qid-badge-wrap { display: flex; justify-content: flex-end; margin-top: .2rem; }
.qid-badge {
  background: rgba(180, 200, 255, 0.10);
  border: 1px solid rgba(160, 185, 255, 0.25);
  border-radius: 10px;
  padding: .25rem .55rem;
  font-size: .80rem; cursor: pointer; user-select: none;
}
.qid-badge:hover { background: rgba(180, 200, 255, 0.16); border-color: rgba(160, 185, 255, 0.45); }
.qid-help-title { font-weight: 600; font-size: .95rem; margin: 0 0 .25rem 0; }
.qid-help-list  { margin: .2rem 0 0 .2rem; padding-left: 1rem; line-height: 1.35; font-size: .90rem; opacity: .95; }

/* Tighten metric spacing */
[data-testid="stMetricValue"] { font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Popover helper (with graceful fallback)
# --------------------------------------------------------------------------
SUPPORTS_POPOVER = hasattr(st, "popover")

def info_popover(label: str, title_md: str, points_md: list[str]):
    """
    Render a small info badge that opens a popover (or caption fallback).
    `title_md` is a short heading; `points_md` is a list of bullets (markdown).
    """
    content = f'<div class="qid-help-title">{title_md}</div><ul class="qid-help-list">' + \
              "".join([f"<li>{p}</li>" for p in points_md]) + "</ul>"
    if SUPPORTS_POPOVER:
        with st.popover(label, use_container_width=True):
            st.markdown(content, unsafe_allow_html=True)
    else:
        # fallback: a compact caption (no layout jumps)
        st.caption(title_md + " — " + " • ".join(points_md))

# --------------------------------------------------------------------------
# Compat helpers
# --------------------------------------------------------------------------
def _try_compute_frontier(px_df, rf, n_points, allow_short=None, max_weight=None):
    """
    Try calling compute_frontier with different known signatures.
    Return the frontier DataFrame (or raise for a real error).
    """
    from src.optimizer import compute_frontier
    from src.analytics import to_returns

    try:
        return compute_frontier(
            prices=px_df, rf=rf, n_points=n_points,
            **({} if allow_short is None else {"allow_short": allow_short}),
            **({} if max_weight is None else {"max_weight": max_weight}),
        )
    except TypeError:
        pass
    try:
        return compute_frontier(px_df, rf, n_points)
    except TypeError:
        pass
    try:
        _rets = to_returns(px_df)
        return compute_frontier(returns=_rets, rf=rf, n_points=n_points)
    except TypeError:
        pass
    _rets = to_returns(px_df)
    return compute_frontier(_rets)

def _try_build_html_report(prices, summary, corr_df):
    """Try different kw for build_html_report and return bytes."""
    from io import StringIO
    from src.report import build_html_report
    try:
        res = build_html_report(prices=prices, summary=summary, cm=corr_df)
        return (res.getvalue() if isinstance(res, StringIO) else (res if isinstance(res, str) else str(res))).encode("utf-8")
    except TypeError:
        pass
    try:
        res = build_html_report(prices=prices, summary=summary, corr=corr_df)
        return (res.getvalue() if isinstance(res, StringIO) else (res if isinstance(res, str) else str(res))).encode("utf-8")
    except TypeError:
        pass
    res = build_html_report(prices=prices, summary=summary, correlation=corr_df)
    return (res.getvalue() if isinstance(res, StringIO) else (res if isinstance(res, str) else str(res))).encode("utf-8")

# --- Optimizer utilities ---
def _normalize_frontier(obj) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame(columns=["Return", "Vol", "Sharpe"])
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    elif isinstance(obj, tuple) and len(obj) >= 2 and isinstance(obj[1], pd.DataFrame):
        df = obj[1].copy()
    elif isinstance(obj, dict):
        if "frontier" in obj and isinstance(obj["frontier"], pd.DataFrame):
            df = obj["frontier"].copy()
        elif "df" in obj and isinstance(obj["df"], pd.DataFrame):
            df = obj["df"].copy()
        else:
            dfs = [v for v in obj.values() if isinstance(v, pd.DataFrame)]
            df = dfs[0].copy() if dfs else pd.DataFrame()
    else:
        df = pd.DataFrame()

    cols = {c.lower(): c for c in df.columns}
    def _pick(*names):
        for n in names:
            if n in df.columns: return n
            if n.lower() in cols: return cols[n.lower()]
        return None

    c_ret = _pick("Return", "ret", "mu", "expected_return")
    c_vol = _pick("Vol", "sigma", "risk", "std")
    c_shp = _pick("Sharpe", "sr", "sharpe_ratio")

    out = pd.DataFrame()
    if c_ret is not None: out["Return"] = df[c_ret].astype(float)
    if c_vol is not None: out["Vol"]    = df[c_vol].astype(float)
    if c_shp is not None and c_shp in df: out["Sharpe"] = df[c_shp].astype(float)
    return out

def _as_weights(res):
    import pandas as _pd
    if res is None:
        return None
    if isinstance(res, dict):
        w = res.get("weights", None)
        if w is None and len(res):
            first = next(iter(res.values()))
            if isinstance(first, (_pd.Series, _pd.DataFrame)):
                w = first
        res = w
    if isinstance(res, tuple) and len(res) > 0:
        res = res[0]
    if isinstance(res, _pd.Series):
        s = res.copy()
        s = s / s.sum() if s.sum() != 0 else s
        return s
    if isinstance(res, _pd.DataFrame):
        df = res.copy()
        if "weight" in df.columns:
            s = df["weight"]
        elif df.shape[1] == 1:
            s = df.iloc[:, 0]
        else:
            num_cols = df.select_dtypes("number").columns
            if len(num_cols) == 1:
                s = df[num_cols[0]]
            else:
                raise ValueError("Cannot infer weights from DataFrame.")
        s.index = s.index.astype(str)
        s = s / s.sum() if s.sum() != 0 else s
        return s
    raise ValueError("Unrecognized weights format from optimizer.")

def _mv_kpi_from_weights(px_df, weights_series, rf_annual):
    from src.analytics import to_returns
    rets = to_returns(px_df)
    mu = rets.mean() * 252.0
    cov = rets.cov() * 252.0
    w = weights_series.reindex(mu.index).fillna(0.0).values
    port_ret = float(np.dot(w, mu.values))
    port_vol = float((np.dot(w, cov.values @ w)) ** 0.5)
    sharpe = (port_ret - rf_annual) / port_vol if port_vol > 0 else float("nan")
    return dict(return_=port_ret, vol=port_vol, sharpe=sharpe)

def _with_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index()
    first_col = out.columns[0]
    if first_col != "Date":
        out = out.rename(columns={first_col: "Date"})
    return out

# ===== App modules =============================================================
from src.data_loader import fetch_prices
from src.analytics   import compute_summary, to_returns, rolling_vol
from src.optimizer   import optimize_portfolio, optimize_cvar
from src.visuals     import (
    prices_chart, perf_cum_chart, rr_scatter, vol_chart, corr_heatmap,
    frontier_chart, weights_donut, weights_pie,
)
# ==============================================================================

# --------------------------------------------------------------------------
# Sidebar / Controls
# --------------------------------------------------------------------------
st.sidebar.header("Parameters")

# Language is fixed to English for help text (by your request)
def t(en: str) -> str:
    return en

# (a) tickers & date range
default_tickers = "AAPL, MSFT, NVDA, TSLA, GOOG"
tickers_str = st.sidebar.text_input(
    "Tickers (comma-separated) — Yahoo Finance",
    value=default_tickers,
    key="tickers_str",
    help=t("Enter comma-separated tickers (e.g., AAPL, MSFT). We use Yahoo Finance Adjusted Close.")
)
tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

date_range = st.sidebar.date_input(
    "Date range",
    value=(dt.date.today() - dt.timedelta(days=365*5), dt.date.today()),
    help=t("Analysis window. Calculations use trading days available in the period.")
)

# (b) risk-free & rolling window (default 90)
rf = st.sidebar.number_input(
    "Risk-free (annual, %)",
    min_value=0.0, value=2.00, step=0.05,
    help=t("Annual risk-free rate in percent (e.g., 2.00 = 2%). Used in Sharpe.")
)
col_win, col_win_info = st.sidebar.columns([3,1])
with col_win:
    window = st.sidebar.slider(
        "Rolling window (days)",
        min_value=30, max_value=252, value=90, step=1,
        help=t("Window length for rolling statistics (e.g., rolling volatility).")
    )
with col_win_info:
    info_popover("ℹ️", "Rolling window",
                 [t("Number of days for the rolling stats."),
                  t("Larger = smoother, less reactive."),
                  t("Typical equity: 60–126 days.")])

# (c) optional: save/load config (JSON only)
with st.sidebar.expander("Save / Load setup", expanded=False):
    col_s, col_l = st.columns(2)
    cfg = {
        "tickers": tickers,
        "start": str(date_range[0]),
        "end": str(date_range[1]),
        "rf": rf,
        "window": window,
    }
    col_s.download_button(
        "Download JSON",
        data=json.dumps(cfg).encode(),
        file_name="qid-config.json",
        mime="application/json",
        use_container_width=True,
    )
    up = col_l.file_uploader("Load JSON", type=["json"], label_visibility="collapsed")
    if up is not None:
        try:
            cfg_in = json.load(io.BytesIO(up.read()))
            st.session_state["tickers_str"] = ", ".join(cfg_in.get("tickers", tickers))
            if "rf" in cfg_in: rf = float(cfg_in["rf"])
            if "window" in cfg_in: window = int(cfg_in["window"])
            st.success("Configuration loaded. Adjust other widgets if needed and rerun.")
        except Exception as e:
            st.warning(f"Could not load JSON: {e}")

# --------------------------------------------------------------------------
# Data fetch
# --------------------------------------------------------------------------
st.title("Quant Insight Dashboard")
st.caption("Performance, risk and correlations for selected tickers/ETFs.")

if len(tickers) < 2:
    st.info("Enter at least two tickers to start.")
    st.stop()

try:
    start = pd.to_datetime(date_range[0])
    end = pd.to_datetime(date_range[1])
    px_df = fetch_prices(tickers, start=start, end=end)   # Adj Close, index=Date
    if px_df is None or px_df.empty:
        st.error("No price data returned.")
        st.stop()
    px_df = px_df.sort_index()
except Exception as e:
    st.error(f"Data error: {e}")
    st.stop()

# Pre-compute series
ret = to_returns(px_df)                         # daily returns
summary = compute_summary(px_df, rf=rf/100.0)   # annualized
cm = ret.corr().astype(float)                   # correlation matrix
vol_roll = rolling_vol(ret, window=window)      # rolling annualized vol

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_prices, tab_perf, tab_vol, tab_corr, tab_opt, tab_backtest, tab_export = st.tabs(
    ["Prices", "Performance", "Volatility", "Correlation", "Optimizer", "Dynamic Backtest", "Data / Export"]
)

# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------
with tab_prices:
    st.subheader("Adjusted Prices")
    st.plotly_chart(prices_chart(px_df), use_container_width=True)
    st.caption("Adjusted for dividends/splits. Table shows the latest rows.")
    st.dataframe(px_df.tail(), use_container_width=True)

# --------------------------------------------------------------------------
# Performance
# --------------------------------------------------------------------------
with tab_perf:
    st.subheader("Cumulative Returns")
    st.plotly_chart(perf_cum_chart(px_df), use_container_width=True)
    st.caption("Prices normalized at start (t=0): Performance_t = Price_t / Price_0.")

    st.subheader("Risk vs Return (annualized)")
    cols_rr = st.columns([6,1])
    with cols_rr[0]:
        st.plotly_chart(rr_scatter(summary), use_container_width=True)
    with cols_rr[1]:
        info_popover("ℹ️", "Risk/Return metrics",
                     ["Return (ann.) ≈ daily mean × 252 (or compounding)",
                      "Volatility (ann.) = daily std × √252",
                      "Sharpe = (Return − rf) / Vol"])

    st.markdown("**Summary (annualized)**")
    st.dataframe(
        summary.style.format({"Return": "{:.2%}", "Vol": "{:.2%}", "Sharpe": "{:.2f}"}),
        use_container_width=True,
    )
    with st.expander("How are Return/Vol/Sharpe computed?"):
        st.markdown(
            "- **Return (annualized)**: compounding the average daily return over ~252 trading days.\n"
            "- **Volatility (annualized)**: daily standard deviation × √252.\n"
            "- **Sharpe**: (Return − risk-free) / Volatility."
        )

# --------------------------------------------------------------------------
# Volatility
# --------------------------------------------------------------------------
with tab_vol:
    vol_head = st.columns([6,1])
    with vol_head[0]:
        st.subheader(f"Rolling Volatility (window = {window} days, annualized)")
    with vol_head[1]:
        info_popover("ℹ️", "Rolling volatility",
                     ["Based on daily returns (not prices)",
                      "Annualized via √252",
                      "Larger window = smoother curve"])

    st.plotly_chart(vol_chart(vol_roll), use_container_width=True)
    st.caption("Annualized volatility = std of daily returns within the window × √252 (data in decimals, chart shown in %).")
    with st.expander("Volatility details"):
        st.markdown(
            f"- Computed on **daily returns**.\n"
            f"- **{window}**-day rolling window.\n"
            "- Annualization uses **√252** trading days."
        )

# --------------------------------------------------------------------------
# Correlation
# --------------------------------------------------------------------------
with tab_corr:
    corr_head = st.columns([6,1])
    with corr_head[0]:
        st.subheader("Correlation matrix (ordered)")
    with corr_head[1]:
        info_popover("ℹ️", "Pearson correlation",
                     ["Daily return correlations",
                      "Range: −1 (inverse) to +1 (direct)"])
    st.plotly_chart(corr_heatmap(cm), use_container_width=True)
    st.caption("Pearson correlations of daily returns.")

    st.markdown("**Pairwise rolling correlation**")
    pairs = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            pairs.append(f"{tickers[i]} — {tickers[j]}")
    cols_pair = st.columns([4,1])
    with cols_pair[0]:
        pair_choice = st.selectbox("Select pair", options=pairs, help="Pick two assets to inspect time-varying correlation.")
    with cols_pair[1]:
        info_popover("ℹ️", "Rolling corr",
                     [f"Window: **{window}** days",
                      "Useful to spot decoupling phases"])

    if pair_choice:
        a, b = [t.strip() for t in pair_choice.split("—")]
        r_pair = ret[a].rolling(window).corr(ret[b]).dropna().to_frame(name=f"{a}-{b}")
        fig_pair = px.line(_with_date(r_pair), x="Date", y=f"{a}-{b}", template="plotly_dark")
        fig_pair.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig_pair, use_container_width=True)

# --------------------------------------------------------------------------
# Optimizer
# --------------------------------------------------------------------------
with tab_opt:
    st.subheader("Optimizer")

    # Controls (with compact popovers)
    c1, c2, c3, c4, cinfo = st.columns([1, 1, 1, 1, 0.8])
    allow_short = c1.checkbox("Allow shorting", value=False, help="Permit negative weights (short selling).")
    max_cap = c2.slider("Max weight cap", min_value=0.05, max_value=1.00, value=0.30, step=0.05, help="Per-asset allocation cap.")
    frontier_points = c3.slider("Frontier points", min_value=5, max_value=50, value=25, step=1, help="How many portfolios to sample on the frontier.")
    show_frontier = c4.checkbox("Show frontier", value=True, help="Display the efficient frontier plot.")
    with cinfo:
        info_popover("ℹ️", "Mean–Variance (MV)",
                     ["Annualized mean/covariance of returns",
                      "Objectives: max Sharpe / min Vol / target return",
                      "Constraints: shorting, per-asset caps"])

    cols_mode = st.columns([2,1,0.8])
    with cols_mode[0]:
        mode = st.selectbox(
            "Optimization mode",
            options=["max_sharpe", "min_vol", "target_return"],
            index=0,
            help="Choose the mean–variance objective."
        )
    with cols_mode[1]:
        if mode == "target_return":
            target_ret = st.slider("Target annual return (%)", 0.0, 40.0, 10.0, 0.5,
                                   help="Desired annual return (percentage).") / 100.0
        else:
            target_ret = None
    with cols_mode[2]:
        info_popover("ℹ️", "Target return",
                     ["Expressed in annual percent",
                      "Solver searches weights achieving at least that return"])

    with st.expander("How MV optimization works"):
        st.markdown(
            "- Estimate **mean** and **covariance** of daily returns (annualized).\n"
            "- Solve for weights that **maximize Sharpe**, **minimize Vol**, or achieve a **target return**.\n"
            "- Apply selected constraints: shorting, caps, etc."
        )

    # ===== MV optimize =====
    try:
        mv_kwargs = dict(
            prices=px_df,
            rf=rf/100.0,
            mode=mode,
            allow_short=allow_short,
            max_weight=max_cap,
        )
        if mode == "target_return" and target_ret is not None:
            mv_kwargs["target_return"] = target_ret

        mv_out = optimize_portfolio(**mv_kwargs)
        mv_weights = _as_weights(mv_out)
        st.plotly_chart(
            weights_donut(mv_weights, title="Portfolio Weights (MV)"),
            use_container_width=True
        )

        # KPI (if not provided, compute from weights)
        if isinstance(mv_out, dict) and all(k in mv_out for k in ("return", "vol", "sharpe")):
            mv_kpi = {"return_": mv_out["return"], "vol": mv_out["vol"], "sharpe": mv_out["sharpe"]}
        else:
            mv_kpi = _mv_kpi_from_weights(px_df, mv_weights, rf_annual=rf/100.0)

        k1, k2, k3 = st.columns(3)
        k1.metric("Return (MV)", f"{mv_kpi['return_']:.2%}")
        k2.metric("Volatility (MV)", f"{mv_kpi['vol']:.2%}")
        k3.metric("Sharpe (MV)", f"{mv_kpi['sharpe']:.2f}")

        fr_info_cols = st.columns([6,1])
        with fr_info_cols[0]:
            if show_frontier:
                fr_raw = _try_compute_frontier(
                    px_df, rf=rf/100.0, n_points=frontier_points,
                    allow_short=allow_short, max_weight=max_cap
                )
                fr = _normalize_frontier(fr_raw)
                st.plotly_chart(frontier_chart(fr), use_container_width=True)
                st.caption("Estimated efficient frontier: portfolios at different risk/return levels.")
            else:
                st.caption("Frontier hidden.")
        with fr_info_cols[1]:
            info_popover("ℹ️", "Frontier",
                         ["Sampled in N points",
                          "Each point = feasible efficient portfolio",
                          "Color = Sharpe (if available)"])

    except Exception as e:
        st.warning(f"MV optimization error: {e}")

    st.markdown("---")
    st.subheader("CVaR Optimizer (Expected Shortfall)")
    cols_cvar = st.columns([2,0.8])
    with cols_cvar[0]:
        alpha = st.slider("Confidence (1−α)", min_value=0.80, max_value=0.99, value=0.95, step=0.01,
                          help="Confidence level for CVaR: 95% = average loss in the worst 5% of cases.")
    with cols_cvar[1]:
        info_popover("ℹ️", "What is CVaR?",
                     ["Average loss in the tail beyond VaR",
                      "More conservative than VaR",
                      "Useful for downside-risk control"])

    # ===== CVaR optimize =====
    try:
        cv_kwargs = dict(
            prices=px_df,
            alpha=alpha,
            allow_short=allow_short,
            max_weight=max_cap,
        )
        cv_out = optimize_cvar(**cv_kwargs)
        cv_weights = _as_weights(cv_out)
        st.plotly_chart(
            weights_donut(cv_weights, title="Portfolio Weights (CVaR)"),
            use_container_width=True
        )

        if isinstance(cv_out, dict) and "curve" in cv_out and isinstance(cv_out["curve"], pd.DataFrame):
            st.plotly_chart(px.line(cv_out["curve"], x="Vol", y="Return"), use_container_width=True)
            st.caption("Risk/return curve using CVaR (Expected Shortfall).")

    except NotImplementedError:
        st.info("CVaR optimization not available in this build.")
    except Exception as e:
        st.warning(f"CVaR optimizer error: {e}")

# --------------------------------------------------------------------------
# Dynamic Backtest (polished UI)
# --------------------------------------------------------------------------
with tab_backtest:
    st.markdown('<div class="qid-h1">DYNAMIC BACKTEST</div>', unsafe_allow_html=True)
    st.markdown('<div class="qid-sub">Rolling / EWMA engine with covariance shrinkage & trading costs</div>', unsafe_allow_html=True)
    st.divider()

    # Controls grouped in sleek cards
    c_est, c_shr, c_reb = st.columns([1.35, 1.0, 1.0])

    # === Estimation ===
    with c_est:
        st.markdown('<div class="qid-card">', unsafe_allow_html=True)
        head_cols = st.columns([4,1])
        with head_cols[0]:
            st.markdown("**Estimation**")
        with head_cols[1]:
            info_popover("ℹ️", "EWMA vs Rolling",
                         ["EWMA: exponential decay (more weight on recent data)",
                          "Rolling: simple moving window",
                          "Pick based on regime reactivity"])

        use_ewma = st.checkbox("Use EWMA (else Rolling)", value=True, key="bt_use_ewma",
                               help="EWMA uses exponential weighting; Rolling uses a fixed-size window.")
        col_e1, col_e2, col_ei = st.columns([2,2,1])
        with col_e1:
            ewma_lam = st.slider("EWMA λ", 0.80, 0.995, 0.97, 0.001, key="bt_ewma",
                                 help="Decay factor (higher = longer memory).")
        with col_e2:
            roll_win  = st.slider("Rolling window (days)", 30, 252, 90, 1, key="bt_win",
                                  help="Window length for mean/covariance estimation.")
        with col_ei:
            info_popover("ℹ️", "Tuning tips",
                         ["Higher λ → slower response",
                          "Larger window → more stable, less reactive"])
        st.markdown("</div>", unsafe_allow_html=True)

    # === Shrinkage ===
    with c_shr:
        st.markdown('<div class="qid-card">', unsafe_allow_html=True)
        head_cols = st.columns([4,1])
        with head_cols[0]:
            st.markdown("**Shrinkage**")
        with head_cols[1]:
            info_popover("ℹ️", "Why shrink Σ?",
                         ["Reduces covariance noise",
                          "Converges to a target structure",
                          "Stabilizes optimization"])

        row1 = st.columns([2,2,1])
        with row1[0]:
            shrink_method = st.selectbox(
                "Target",
                options=["none", "const-cor", "diag", "identity"],
                index=1,
                help="Shrink the covariance matrix towards a target.",
                key="bt_shr_m",
            )
        with row1[1]:
            shrink_intensity = st.slider("Intensity γ", 0.0, 1.0, 0.25, 0.05, key="bt_shr_g",
                                         help="0 = no shrink; 1 = fully to target.")
        with row1[2]:
            info_popover("ℹ️", "γ guidance",
                         ["Typical equity multi-asset: 0.1–0.4"])
        st.markdown("</div>", unsafe_allow_html=True)

    # === Rebalance ===
    with c_reb:
        st.markdown('<div class="qid-card">', unsafe_allow_html=True)
        head_cols = st.columns([4,1])
        with head_cols[0]:
            st.markdown("**Rebalance**")
        with head_cols[1]:
            info_popover("ℹ️", "Frequency & shorts",
                         ["k days between rebalances",
                          "Shorts allow negative weights"])

        row1 = st.columns([2,2,1])
        with row1[0]:
            reb_k       = st.slider("Every k days", 1, 63, 21, 1, key="bt_reb_k",
                                    help="Rebalance frequency in trading days.")
        with row1[1]:
            allow_short_bt = st.checkbox("Allow short", value=False, key="bt_short",
                                         help="Allow negative weights in backtest.")
        with row1[2]:
            info_popover("ℹ️", "Trade-offs",
                         ["Lower k → more trading & costs",
                          "Shorts may amplify risk/turnover"])
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="qid-gap"></div>', unsafe_allow_html=True)

    # Risk model & costs
    c_risk, c_costs = st.columns([1.0, 1.0])

    with c_risk:
        st.markdown('<div class="qid-card">', unsafe_allow_html=True)
        head_cols = st.columns([4,1])
        with head_cols[0]:
            st.markdown("**Risk model**")
        with head_cols[1]:
            info_popover("ℹ️", "γ & Ridge",
                         ["γ: risk aversion (higher = more conservative)",
                          "Ridge: regularizes Σ (conditioning)"])

        row1 = st.columns([2,2,1])
        with row1[0]:
            gamma = st.number_input("Risk aversion γ", min_value=0.1, value=5.0, step=0.1, key="bt_gamma",
                                    help="Higher γ emphasizes risk penalty.")
        with row1[1]:
            ridge  = st.number_input("Ridge on Σ", min_value=0.0, value=1e-3, step=1e-3, format="%.4f", key="bt_ridge",
                                     help="Tikhonov regularization for covariance matrix.")
        with row1[2]:
            info_popover("ℹ️", "Guidance",
                         ["Higher γ → more defensive weights",
                          "Ridge helps when Σ is ill-conditioned"])
        st.markdown("</div>", unsafe_allow_html=True)

    with c_costs:
        st.markdown('<div class="qid-card">', unsafe_allow_html=True)
        head_cols = st.columns([4,1])
        with head_cols[0]:
            st.markdown("**Trading costs**")
        with head_cols[1]:
            info_popover("ℹ️", "bps & slippage",
                         ["1 bps = 0.01%",
                          "Slippage = execution cost"])

        row1 = st.columns([2,2,2,1])
        with row1[0]:
            tx_bps  = st.number_input("Commissions (bps)", min_value=0.0, value=5.0, step=0.5, key="bt_tx",
                                      help="Per-trade commissions, in basis points.")
        with row1[1]:
            slp_bps = st.number_input("Slippage (bps)",   min_value=0.0, value=1.0, step=0.5, key="bt_slip",
                                      help="Execution cost (spread/impact), in bps.")
        with row1[2]:
            turn_L2 = st.number_input("Turnover penalty λ (L2)", min_value=0.0, value=5.0, step=0.5, key="bt_turn",
                                      help="Quadratic penalty on weight changes to limit turnover.")
        with row1[3]:
            info_popover("ℹ️", "Cost control",
                         ["Higher turnover → higher costs",
                          "λ helps smooth allocation changes"])
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # === Run engine & plots ===
    try:
        est_cfg = EstimatorConfig(
            window=int(roll_win),
            ewma_lambda=float(ewma_lam),
            use_ewma=bool(use_ewma),
            shrink=ShrinkageConfig(method=shrink_method, intensity=float(shrink_intensity)),
        )
        mv_cfg = MVConfig(
            risk_aversion=float(gamma),
            ridge=float(ridge),
            turnover_L2=float(turn_L2),
            leverage=1.0,
            allow_short=bool(allow_short_bt),
        )
        cost_cfg = CostConfig(proportional_bps=float(tx_bps), slippage_bps=float(slp_bps))
        reb_cfg  = RebalanceConfig(every_k_days=int(reb_k), offset=0)
        bt_cfg   = BacktestConfig(estimator=est_cfg, mv=mv_cfg, costs=cost_cfg, rebalance=reb_cfg)

        eng = RollingEWMAEngine(ret, bt_cfg)
        out = eng.run(initial_nav=1.0)

        nav_df     = out.get("nav", pd.DataFrame())
        turn_df    = out.get("turnover", pd.DataFrame())
        cost_df    = out.get("costs", pd.DataFrame())
        weights_df = out.get("weights", pd.DataFrame())

        st.session_state["bt_out"] = {"nav": nav_df, "turnover": turn_df, "costs": cost_df, "weights": weights_df}

        if nav_df.empty:
            st.info("No NAV produced (short history or first rebalance out of range). Try a smaller window or wider dates.")
            st.stop()

        from src.visuals import nav_chart, turnover_bar, costs_bar, costs_cum_chart, weights_area

        st.plotly_chart(nav_chart(nav_df, title="Backtest NAV"), use_container_width=True)
        st.caption("NAV = cumulative portfolio value (base 1.0).")

        cA, cB = st.columns(2)
        with cA:
            st.plotly_chart(turnover_bar(turn_df, title="Turnover (rebalance dates)"), use_container_width=True)
            st.caption("Turnover = sum of absolute weight changes at each rebalance.")
        with cB:
            st.plotly_chart(costs_bar(cost_df, title="Transaction costs"), use_container_width=True)
            st.caption("Transaction costs = commissions + slippage for each rebalance.")

        st.plotly_chart(costs_cum_chart(cost_df, title="Cumulative transaction costs"), use_container_width=True)
        st.caption("Cumulative cost drag over time.")

        st.plotly_chart(weights_area(weights_df, title="Weights over time (stacked)"), use_container_width=True)
        st.caption("Allocation evolution over time (weights in %).")

        with st.expander("How to read the backtest results"):
            st.markdown(
                "- **NAV**: portfolio value starting at 1.0.\n"
                "- **Turnover**: sum of absolute weight changes per rebalance.\n"
                "- **Transaction costs**: commissions + slippage; the cumulative plot shows cost drag.\n"
                "- **KPI** are annualized from NAV daily returns."
            )

        # KPI
        nav_vals = nav_df["nav"].astype(float)
        if len(nav_vals) > 1:
            ret_series = nav_vals.pct_change().dropna()
            ann_ret = (1 + ret_series.mean())**252 - 1
            ann_vol = ret_series.std() * np.sqrt(252)
            rf_annual = rf/100.0
            sharpe = (ann_ret - rf_annual) / ann_vol if ann_vol > 0 else float("nan")
            k1, k2, k3 = st.columns(3)
            k1.metric("Backtest Return (ann.)", f"{ann_ret:.2%}")
            k2.metric("Backtest Vol (ann.)", f"{ann_vol:.2%}")
            k3.metric("Sharpe (ann.)", f"{sharpe:.2f}")

        # Downloads
        cD1, cD2, cD3, cD4 = st.columns(4)
        cD1.download_button("Download NAV CSV",      nav_df.to_csv().encode(),     "nav.csv",      "text/csv", use_container_width=True)
        cD2.download_button("Download Turnover CSV", turn_df.to_csv().encode(),    "turnover.csv", "text/csv", use_container_width=True)
        cD3.download_button("Download Costs CSV",    cost_df.to_csv().encode(),    "costs.csv",    "text/csv", use_container_width=True)
        cD4.download_button("Download Weights CSV",  weights_df.to_csv().encode(), "weights.csv",  "text/csv", use_container_width=True)

    except Exception as e:
        st.error("Dynamic Backtest error")
        st.exception(e)

# --------------------------------------------------------------------------
# Data / Export
# --------------------------------------------------------------------------
with tab_export:
    st.subheader("Data / Export")
    st.caption("Download raw session data (prices, correlations) and a one-click HTML report.")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Prices CSV**")
        st.download_button(
            "Download prices",
            data=px_df.to_csv().encode(),
            file_name="prices.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_prices",
        )

        st.markdown("**Correlation CSV**")
        st.download_button(
            "Download correlation",
            data=cm.to_csv().encode(),
            file_name="correlation.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_corr",
        )

    with c2:
        st.markdown("**One-click HTML report**")
        try:
            data_bytes = _try_build_html_report(prices=px_df, summary=summary, corr_df=cm)
            st.download_button(
                "Download HTML report",
                data=data_bytes,
                file_name="report.html",
                mime="text/html",
                use_container_width=True,
                key="dl_html",
            )
        except Exception as e:
            st.warning(f"Unable to build HTML report: {e}")
