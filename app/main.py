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
from collections import OrderedDict
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

# --- Minimal UI polish (no cards, no popovers, no help tooltips) ---
st.markdown("""
<style>
html, body, [class*="css"] { letter-spacing: 0.05px; }
.block-container { padding-top: 1.4rem; }
.qid-h1 { font-size: 1.55rem; font-weight: 700; margin: .25rem 0 .5rem 0; }
.qid-sub { color: #b9c1d9; font-size: .95rem; margin-top: -.35rem; }
[data-testid="stMetricValue"] { font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* Safe styling only on closed controls */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
  background: #141821 !important;
  border-color: #2a2f3a !important;
}
div[data-baseweb="select"] svg { fill: #e6edf3 !important; }

/* Minimal open-menu text fix (global): keep option labels visible */
div[role="listbox"] [role="option"],
div[role="listbox"] [role="option"] *,
ul[role="listbox"] li,
ul[role="listbox"] li *,
ul[data-testid="stSelectboxVirtualDropdown"] li,
ul[data-testid="stSelectboxVirtualDropdown"] li *,
div[data-testid="stMultiSelectPopover"] [role="option"],
div[data-testid="stMultiSelectPopover"] [role="option"] * {
  color: #f8fafc !important;
  -webkit-text-fill-color: #f8fafc !important;
  opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _compute_frontier(px_df, rf, n_points, shorting=False, max_weight=0.30):
    """Compute efficient frontier using canonical optimizer API."""
    from src.optimizer import frontier_from_prices
    return frontier_from_prices(
        prices=px_df,
        rf=rf,
        points=n_points,
        shorting=shorting,
        max_weight=max_weight,
    )

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

def _normalize_frontier(obj) -> pd.DataFrame:
    """Normalize frontier outputs to a DataFrame with Return/Vol/Sharpe if possible."""
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
    """Normalize weights output (dict/tuple/Series/DataFrame) to a Series."""
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
    """Compute annualized return/vol/sharpe from weights."""
    from src.analytics import to_returns
    rets = to_returns(px_df)
    mu = rets.mean() * 252.0
    cov = rets.cov() * 252.0
    w = weights_series.reindex(mu.index).fillna(0.0).values
    port_ret = float(np.dot(w, mu.values))
    port_vol = float((np.dot(w, cov.values @ w)) ** 0.5)
    sharpe = (port_ret - rf_annual) / port_vol if port_vol > 0 else float("nan")
    return dict(return_=port_ret, vol=port_vol, sharpe=sharpe)

def _historical_nav_from_weights(px_df, weights_series, initial_value: float = 100.0) -> pd.DataFrame:
    """Static-weights historical NAV built from asset daily returns."""
    rets = to_returns(px_df).dropna(how="all")
    if rets.empty:
        return pd.DataFrame(columns=["nav"])
    w = weights_series.reindex(rets.columns).fillna(0.0).astype(float)
    if float(w.sum()) == 0.0:
        return pd.DataFrame(columns=["nav"])
    w = w / w.sum()
    port_rets = rets.fillna(0.0).mul(w, axis=1).sum(axis=1)
    nav = initial_value * (1.0 + port_rets).cumprod()
    return nav.to_frame(name="nav")

def _with_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index()
    first_col = out.columns[0]
    if first_col != "Date":
        out = out.rename(columns={first_col: "Date"})
    return out

# ===== App modules =============================================================
from src.data_loader import fetch_prices
from src.analytics   import compute_summary, to_returns, rolling_vol, simulate_portfolio_paths
from src.optimizer   import optimize_portfolio, optimize_cvar
from src.visuals     import (
    prices_chart, perf_cum_chart, rr_scatter, vol_chart, corr_heatmap,
    frontier_chart, weights_donut, weights_pie, monte_carlo_paths_chart,
)
# ==============================================================================

# --------------------------------------------------------------------------
# Sidebar / Controls
# --------------------------------------------------------------------------
st.sidebar.header("Parameters")

TICKER_PRESETS = OrderedDict({
    "Balanced Multi-Asset": ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "LQD", "GLD", "USO", "AAPL", "MSFT"],
    "Global Equity Core": ["SPY", "QQQ", "IWM", "VEA", "VWO", "AAPL", "MSFT", "GOOG", "AMZN"],
    "Defensive Mix": ["SPY", "XLV", "XLP", "XLU", "TLT", "IEF", "LQD", "GLD"],
    "Growth & Tech": ["QQQ", "XLK", "SMH", "AAPL", "MSFT", "GOOG", "AMZN", "META"],
})

def t(en: str) -> str:
    return en


def section_header(title: str, info_md: str) -> None:
    """Render a section title with a small info icon + popover."""
    col_title, col_info = st.columns([0.94, 0.06])
    with col_title:
        st.subheader(title)
    with col_info:
        with st.popover("ⓘ"):
            st.markdown(info_md)

def _sanitize_tickers(raw: str) -> tuple[list[str], list[str]]:
    parsed = [t.strip().upper() for t in raw.split(",") if t.strip()]
    seen = set()
    clean = []
    duplicates = []
    for t in parsed:
        if t in seen:
            duplicates.append(t)
            continue
        seen.add(t)
        clean.append(t)
    return clean, duplicates


def _queue_tickers_update(tickers_value: str | list[str]) -> None:
    """Queue ticker text update to apply before the sidebar widget is created."""
    if isinstance(tickers_value, list):
        st.session_state["pending_tickers_str"] = ", ".join(tickers_value)
    else:
        st.session_state["pending_tickers_str"] = str(tickers_value)

# (a) tickers & date range
default_tickers = "SPY, QQQ, IWM, EFA, EEM, TLT, LQD, GLD, USO, AAPL, MSFT"
if "pending_tickers_str" in st.session_state:
    st.session_state["tickers_str"] = st.session_state.pop("pending_tickers_str")

with st.sidebar.expander("Quick ticker presets", expanded=False):
    preset_name = st.selectbox("Preset", list(TICKER_PRESETS.keys()), index=0)
    st.caption(", ".join(TICKER_PRESETS[preset_name]))
    if st.button("Apply preset", use_container_width=True):
        _queue_tickers_update(TICKER_PRESETS[preset_name])
        st.rerun()

tickers_str = st.sidebar.text_input(
    "Tickers (comma-separated) from Yahoo Finance",
    value=default_tickers,
    key="tickers_str",
)
tickers, duplicate_tickers = _sanitize_tickers(tickers_str)
if duplicate_tickers:
    st.sidebar.info(f"Removed duplicate tickers: {', '.join(sorted(set(duplicate_tickers)))}")
if len(tickers) > 16:
    st.sidebar.warning("More than 16 tickers can make charts crowded and slower.")

date_range = st.sidebar.date_input(
    "Date range",
    value=(dt.date.today() - dt.timedelta(days=365*5), dt.date.today()),
)

# (b) risk-free & rolling window (default 90)
rf = st.sidebar.number_input(
    "Risk-free (annual, %)",
    min_value=0.0, value=2.00, step=0.05,
)
window = st.sidebar.slider(
    "Rolling window (days)",
    min_value=30, max_value=252, value=90, step=1,
)

# (c) Save/Load config (JSON)
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
            _queue_tickers_update(cfg_in.get("tickers", tickers))
            if "rf" in cfg_in: rf = float(cfg_in["rf"])
            if "window" in cfg_in: window = int(cfg_in["window"])
            st.success("Configuration loaded.")
            st.rerun()
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
    with st.spinner("Loading market data..."):
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

# Quick context KPIs
missing_pct = float(px_df.isna().sum().sum() / max(px_df.shape[0] * px_df.shape[1], 1))
days_loaded = int(px_df.shape[0])
assets_loaded = int(px_df.shape[1])
kpi_a, kpi_b, kpi_c = st.columns(3)
kpi_a.metric("Assets loaded", f"{assets_loaded}")
kpi_b.metric("History length", f"{days_loaded} days")
kpi_c.metric("Missing data", f"{missing_pct:.2%}")

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_guide, tab_prices, tab_perf, tab_vol, tab_corr, tab_opt, tab_backtest, tab_export = st.tabs(
    [
        "Build Your First Portfolio",
        "Prices",
        "Performance",
        "Volatility",
        "Correlation",
        "Optimizer",
        "Dynamic Backtest",
        "Data / Export",
    ]
)

# --------------------------------------------------------------------------
# Build your first portfolio
# --------------------------------------------------------------------------
with tab_guide:
    def _fmt_eur(value: float) -> str:
        return f"€{value:,.0f}"

    section_header(
        "Build Your First Portfolio (Beginner Guide)",
        """
A practical workflow for users without a finance background.
Set your plan, tune constraints, then apply a curated starter universe in one click.
""",
    )

    st.markdown("### Portfolio Builder Wizard 2.0")
    st.caption("Structured setup: define profile, define capital plan, then add optional tilts.")

    st.markdown("#### 1) Investor Profile")
    w1, w2 = st.columns(2)
    with w1:
        horizon_years = st.selectbox("Investment horizon", ["0-3 years", "3-7 years", "7+ years"], index=2)
        risk_profile = st.selectbox("Risk profile", ["Conservative", "Balanced", "Growth"], index=1)
    with w2:
        objective = st.selectbox("Main objective", ["Capital preservation", "Steady growth", "Max growth"], index=1)
        concentration_style = st.selectbox(
            "Portfolio style",
            ["Diversified (8-12 positions)", "Focused (5-8 positions)", "Core-only (3-5 positions)"],
            index=0,
        )
        rebalance_preference = st.selectbox(
            "Rebalance frequency",
            ["Monthly", "Quarterly", "Semi-annual"],
            index=1,
        )

    st.markdown("#### 2) Capital Plan")
    c_plan_1, c_plan_2 = st.columns(2)
    with c_plan_1:
        initial_capital = st.number_input("Initial capital (€)", min_value=0, value=20000, step=1000)
        monthly_contribution = st.number_input("Monthly contribution (€)", min_value=0, value=600, step=50)
    with c_plan_2:
        target_goal = st.number_input("Target value (€)", min_value=1000, value=150000, step=5000)
        min_bond_floor = st.slider("Min bond floor (%)", min_value=5, max_value=60, value=20, step=5)

    st.markdown("#### 3) Optional Tilts")
    try:
        tilts = st.multiselect(
            "Select up to two",
            options=["Technology tilt (+5%)", "Income tilt (+5%)", "Inflation hedge (+5%)"],
            max_selections=2,
        )
    except TypeError:
        tilts = st.multiselect(
            "Select up to two",
            options=["Technology tilt (+5%)", "Income tilt (+5%)", "Inflation hedge (+5%)"],
        )[:2]

    starter_models = {
        "Conservative": {
            "allocation": {"Global Equity": 40, "Bonds": 50, "Gold": 10, "Thematic": 0, "Dividend": 0},
            "example_tickers": ["VT", "BND", "GLD", "IEF"],
            "rebalance": "Every 6 months",
            "max_single": 20,
            "ret": 0.055,
            "vol": 0.10,
        },
        "Balanced": {
            "allocation": {"Global Equity": 60, "Bonds": 30, "Gold": 10, "Thematic": 0, "Dividend": 0},
            "example_tickers": ["VT", "QQQ", "BND", "GLD", "SPY"],
            "rebalance": "Every 3 months",
            "max_single": 25,
            "ret": 0.075,
            "vol": 0.14,
        },
        "Growth": {
            "allocation": {"Global Equity": 78, "Bonds": 12, "Gold": 5, "Thematic": 5, "Dividend": 0},
            "example_tickers": ["VT", "QQQ", "SPY", "BND", "SMH"],
            "rebalance": "Every 3 months",
            "max_single": 30,
            "ret": 0.092,
            "vol": 0.19,
        },
    }

    model = starter_models[risk_profile]
    allocation = {k: float(v) for k, v in model["allocation"].items()}

    horizon_adjust = {"0-3 years": -10, "3-7 years": 0, "7+ years": 5}[horizon_years]
    objective_adjust = {"Capital preservation": -5, "Steady growth": 0, "Max growth": 7}[objective]
    equity_shift = horizon_adjust + objective_adjust
    allocation["Global Equity"] += equity_shift
    allocation["Bonds"] -= equity_shift

    if "Technology tilt (+5%)" in tilts:
        allocation["Thematic"] += 5
        allocation["Global Equity"] -= 3
        allocation["Bonds"] -= 2
    if "Income tilt (+5%)" in tilts:
        allocation["Dividend"] += 5
        allocation["Global Equity"] -= 5
    if "Inflation hedge (+5%)" in tilts:
        allocation["Gold"] += 5
        allocation["Bonds"] -= 5

    for bucket in ["Global Equity", "Gold", "Thematic", "Dividend"]:
        allocation[bucket] = max(allocation[bucket], 0.0)

    if allocation["Bonds"] < float(min_bond_floor):
        need = float(min_bond_floor) - allocation["Bonds"]
        allocation["Bonds"] += need
        for bucket in ["Global Equity", "Gold", "Thematic", "Dividend"]:
            take = min(allocation[bucket], need)
            allocation[bucket] -= take
            need -= take
            if need <= 0:
                break

    total_alloc = max(sum(allocation.values()), 1e-9)
    allocation = {k: round(100.0 * v / total_alloc, 1) for k, v in allocation.items()}
    alloc_gap = round(100.0 - sum(allocation.values()), 1)
    allocation["Bonds"] = round(allocation["Bonds"] + alloc_gap, 1)

    concentration_map = {
        "Diversified (8-12 positions)": (10, -4),
        "Focused (5-8 positions)": (7, 0),
        "Core-only (3-5 positions)": (5, 4),
    }
    n_positions, max_single_shift = concentration_map[concentration_style]
    max_single = max(10, min(40, model["max_single"] + max_single_shift))

    return_adj = {"Capital preservation": -0.01, "Steady growth": 0.0, "Max growth": 0.012}[objective]
    est_return = max(0.02, model["ret"] + return_adj)
    est_vol = model["vol"] + {"Diversified (8-12 positions)": -0.01, "Focused (5-8 positions)": 0.0, "Core-only (3-5 positions)": 0.012}[concentration_style]

    years = {"0-3 years": 3, "3-7 years": 6, "7+ years": 10}[horizon_years]
    months = years * 12
    monthly_rate = (1.0 + est_return) ** (1.0 / 12.0) - 1.0
    if monthly_rate > 0:
        projected_value = initial_capital * (1.0 + monthly_rate) ** months + monthly_contribution * (((1.0 + monthly_rate) ** months - 1.0) / monthly_rate)
        required_monthly = max(
            (
                (float(target_goal) - (float(initial_capital) * (1.0 + monthly_rate) ** months))
                * monthly_rate
            ) / max(((1.0 + monthly_rate) ** months - 1.0), 1e-9),
            0.0,
        )
    else:
        projected_value = initial_capital + (monthly_contribution * months)
        required_monthly = max((float(target_goal) - float(initial_capital)) / max(months, 1), 0.0)
    goal_coverage = projected_value / max(float(target_goal), 1.0)
    monthly_gap = float(monthly_contribution) - required_monthly
    preferred_rebalance = {"Monthly": "every month", "Quarterly": "every 3 months", "Semi-annual": "every 6 months"}[rebalance_preference]

    sg1, sg2 = st.columns([0.82, 0.18])
    with sg1:
        st.success(
            f"Suggested setup for **{risk_profile}** profile ({objective.lower()}, {horizon_years}): "
            f"rebalance {preferred_rebalance}, target {n_positions} positions, keep max single position around {max_single}%."
        )
    with sg2:
        apply_suggested_now = st.button(
            "Use these options",
            key="apply_suggested_now",
            use_container_width=True,
        )

    kk1, kk2, kk3, kk4 = st.columns(4)
    kk1.metric("Est. annual return", f"{est_return:.1%}")
    kk2.metric("Est. annual volatility", f"{est_vol:.1%}")
    kk3.metric(f"Projection ({years}y)", _fmt_eur(projected_value))
    kk4.metric("Goal coverage", f"{goal_coverage:.1%}")
    kg1, kg2, kg3 = st.columns(3)
    kg1.metric("Required monthly", _fmt_eur(required_monthly))
    kg2.metric("Your monthly gap", _fmt_eur(monthly_gap))
    kg3.metric("Rebalance plan", preferred_rebalance.replace("every ", ""))
    st.progress(float(min(max(goal_coverage, 0.0), 1.0)))
    st.caption("Goal progress under base-return assumptions.")

    if goal_coverage < 0.9:
        st.warning(
            f"Current plan may miss the target. To improve probability, increase monthly contribution to about {_fmt_eur(required_monthly)}, extend horizon, or lower target."
        )
    elif goal_coverage > 1.2:
        st.info("Current plan is ahead of target assumptions. You can keep this setup or reduce risk concentration.")

    if max_single > 30:
        st.warning("Concentration is high. Consider tighter max single-position limits in Optimizer.")
    if horizon_years == "0-3 years" and risk_profile == "Growth":
        st.warning("Short horizon with growth profile can create high drawdown risk.")

    st.markdown("#### Stress Test Scenarios")
    ss1, ss2, ss3 = st.columns(3)
    with ss1:
        bear_return = st.slider("Bear return (annual %)", min_value=-25.0, max_value=10.0, value=float(np.clip((est_return - 0.06) * 100.0, -25.0, 10.0)), step=0.5) / 100.0
    with ss2:
        base_return = st.slider("Base return (annual %)", min_value=-5.0, max_value=20.0, value=float(np.clip(est_return * 100.0, -5.0, 20.0)), step=0.5) / 100.0
    with ss3:
        bull_return = st.slider("Bull return (annual %)", min_value=0.0, max_value=30.0, value=float(np.clip((est_return + 0.05) * 100.0, 0.0, 30.0)), step=0.5) / 100.0

    def _scenario_path(annual_return: float, n_years: int, start_value: float, monthly_add: float) -> tuple[list[dict], float]:
        value = float(start_value)
        path = [{"Year": 0, "Value": value}]
        monthly_r = (1.0 + annual_return) ** (1.0 / 12.0) - 1.0
        for month_idx in range(1, n_years * 12 + 1):
            value = value * (1.0 + monthly_r) + monthly_add
            if month_idx % 12 == 0:
                path.append({"Year": month_idx // 12, "Value": value})
        return path, value

    bear_path, bear_final = _scenario_path(bear_return, years, initial_capital, monthly_contribution)
    base_path, base_final = _scenario_path(base_return, years, initial_capital, monthly_contribution)
    bull_path, bull_final = _scenario_path(bull_return, years, initial_capital, monthly_contribution)

    scenario_curve = pd.concat(
        [
            pd.DataFrame(bear_path).assign(Scenario="Bear"),
            pd.DataFrame(base_path).assign(Scenario="Base"),
            pd.DataFrame(bull_path).assign(Scenario="Bull"),
        ],
        ignore_index=True,
    )
    scenario_fig = px.line(
        scenario_curve,
        x="Year",
        y="Value",
        color="Scenario",
        markers=True,
        template="plotly_dark",
        color_discrete_map={"Bear": "#ff6b6b", "Base": "#f8f9fa", "Bull": "#51cf66"},
    )
    scenario_fig.update_layout(margin=dict(l=10, r=10, t=35, b=10), title="Projected portfolio paths by scenario")
    st.plotly_chart(scenario_fig, use_container_width=True)

    scenario_outcomes = pd.DataFrame(
        [
            {
                "Scenario": "Bear",
                "Assumed return": f"{bear_return:.1%}",
                "Projected value": _fmt_eur(bear_final),
                "Goal coverage": f"{(bear_final / max(float(target_goal), 1.0)):.1%}",
            },
            {
                "Scenario": "Base",
                "Assumed return": f"{base_return:.1%}",
                "Projected value": _fmt_eur(base_final),
                "Goal coverage": f"{(base_final / max(float(target_goal), 1.0)):.1%}",
            },
            {
                "Scenario": "Bull",
                "Assumed return": f"{bull_return:.1%}",
                "Projected value": _fmt_eur(bull_final),
                "Goal coverage": f"{(bull_final / max(float(target_goal), 1.0)):.1%}",
            },
        ]
    )
    st.dataframe(scenario_outcomes, use_container_width=True, hide_index=True)

    # Criteria-driven starter universe (deterministic, no random picks)
    ticker_meta = {
        "VT": {"bucket": "Global Equity"},
        "SPY": {"bucket": "Global Equity"},
        "QQQ": {"bucket": "Thematic"},
        "VEA": {"bucket": "Global Equity"},
        "VWO": {"bucket": "Global Equity"},
        "BND": {"bucket": "Bonds"},
        "IEF": {"bucket": "Bonds"},
        "TLT": {"bucket": "Bonds"},
        "GLD": {"bucket": "Gold"},
        "IAU": {"bucket": "Gold"},
        "SMH": {"bucket": "Thematic"},
        "XLK": {"bucket": "Thematic"},
        "SCHD": {"bucket": "Dividend"},
        "VIG": {"bucket": "Dividend"},
        "DBC": {"bucket": "Gold"},
    }
    bucket_candidates = {
        "Global Equity": ["VT", "SPY", "VEA", "VWO"],
        "Bonds": ["BND", "IEF", "TLT"],
        "Gold": ["GLD", "IAU", "DBC"],
        "Thematic": ["QQQ", "SMH", "XLK"],
        "Dividend": ["SCHD", "VIG"],
    }

    score = {t: 0.0 for t in ticker_meta}
    # Profile priorities
    if risk_profile == "Conservative":
        for t in ["BND", "IEF", "TLT", "GLD", "IAU", "SCHD", "VIG", "VT"]:
            score[t] += 2.0
        for t in ["QQQ", "SMH", "XLK", "VWO"]:
            score[t] -= 2.0
    elif risk_profile == "Balanced":
        for t in ["VT", "SPY", "BND", "GLD", "VEA"]:
            score[t] += 1.5
    else:  # Growth
        for t in ["QQQ", "SMH", "XLK", "SPY", "VT", "VWO"]:
            score[t] += 2.0
        for t in ["TLT", "IEF"]:
            score[t] -= 1.0

    # Objective and horizon consistency
    if objective == "Capital preservation":
        for t in ["BND", "IEF", "TLT", "GLD", "IAU", "SCHD"]:
            score[t] += 2.0
    elif objective == "Steady growth":
        for t in ["VT", "SPY", "BND", "VEA"]:
            score[t] += 1.2
    else:  # Max growth
        for t in ["QQQ", "SMH", "XLK", "VWO", "SPY"]:
            score[t] += 2.2

    if horizon_years == "0-3 years":
        for t in ["BND", "IEF", "TLT", "GLD"]:
            score[t] += 1.8
        for t in ["SMH", "QQQ", "VWO"]:
            score[t] -= 1.0
    elif horizon_years == "7+ years":
        for t in ["VT", "SPY", "QQQ", "VEA", "VWO", "SMH"]:
            score[t] += 1.0

    # Tilt alignment
    if "Technology tilt (+5%)" in tilts:
        for t in ["QQQ", "SMH", "XLK"]:
            score[t] += 4.0
    if "Income tilt (+5%)" in tilts:
        for t in ["SCHD", "VIG", "BND"]:
            score[t] += 3.0
    if "Inflation hedge (+5%)" in tilts:
        for t in ["GLD", "IAU", "DBC"]:
            score[t] += 3.0

    # Build target counts by bucket from dynamic allocation
    target_by_bucket = {}
    for bucket, w in allocation.items():
        if w <= 0:
            continue
        base_n = int(round((w / 100.0) * n_positions))
        target_by_bucket[bucket] = max(1, base_n)

    # Enforce bond floor directly in ticker count
    min_bond_positions = int(np.ceil((float(min_bond_floor) / 100.0) * n_positions))
    target_by_bucket["Bonds"] = max(target_by_bucket.get("Bonds", 1), min_bond_positions)

    # Conservative profile avoids EM by default unless max growth selected
    if risk_profile == "Conservative" and objective != "Max growth":
        score["VWO"] -= 3.0

    selected = []
    # First pass: fill by bucket priority
    bucket_priority = sorted(
        target_by_bucket.keys(),
        key=lambda b: target_by_bucket.get(b, 0),
        reverse=True,
    )
    for bucket in bucket_priority:
        candidates = bucket_candidates.get(bucket, [])
        ranked = sorted(candidates, key=lambda t: score.get(t, 0.0), reverse=True)
        need = target_by_bucket.get(bucket, 0)
        for t in ranked:
            if need <= 0 or len(selected) >= n_positions:
                break
            if t not in selected:
                selected.append(t)
                need -= 1

    # Second pass: fill remaining slots by global score
    ranked_all = sorted(score.keys(), key=lambda t: score[t], reverse=True)
    for t in ranked_all:
        if len(selected) >= n_positions:
            break
        if t not in selected:
            selected.append(t)

    starter_universe = selected[:n_positions]

    # Explain criteria used for transparency
    selection_df = pd.DataFrame(
        [
            {
                "Ticker": t,
                "Bucket": ticker_meta[t]["bucket"],
                "Score": round(score[t], 1),
            }
            for t in starter_universe
        ]
    ).sort_values("Score", ascending=False)

    mcol1, mcol2 = st.columns([0.62, 0.38])
    with mcol1:
        alloc_df = pd.DataFrame(
            [{"Bucket": k, "Weight %": v} for k, v in allocation.items() if v > 0]
        )
        alloc_series = pd.Series(
            {row["Bucket"]: row["Weight %"] / 100.0 for row in alloc_df.to_dict("records")}
        )
        st.plotly_chart(
            weights_donut(alloc_series, title="Suggested Allocation Mix"),
            use_container_width=True,
        )
        st.dataframe(alloc_df, use_container_width=True, hide_index=True)
        st.caption("Allocation is dynamically adjusted by horizon, objective, tilts, and bond floor.")
    with mcol2:
        st.info("**Starter universe**\n\n" + ", ".join(starter_universe))
        if st.button("Use this starter universe in sidebar", key="apply_starter_universe", use_container_width=True):
            _queue_tickers_update(starter_universe)
            st.rerun()
        if st.button("Use starter + keep max cap guidance", key="apply_starter_and_guidance", use_container_width=True):
            _queue_tickers_update(starter_universe)
            st.session_state["guide_max_weight_hint"] = max_single / 100.0
            st.rerun()
    if apply_suggested_now:
        _queue_tickers_update(starter_universe)
        st.session_state["guide_max_weight_hint"] = max_single / 100.0
        st.rerun()
    with st.expander("Starter universe criteria (why these tickers)"):
        st.caption(
            "Selection is deterministic and driven by your profile, objective, horizon, tilts, concentration target, and bond-floor constraint."
        )
        st.dataframe(selection_df, use_container_width=True, hide_index=True)

    progress_df = pd.DataFrame(
        {
            "Step": ["Capital plan", "Allocation design", "Risk constraints", "Backtest readiness"],
            "Status": [
                "Done" if (initial_capital > 0 and monthly_contribution >= 0) else "Pending",
                "Done",
                "Done" if max_single <= 35 else "Review",
                "Ready",
            ],
            "Action": [
                f"Target {_fmt_eur(target_goal)} in about {years} years",
                f"{risk_profile} base with {len(tilts)} optional tilts",
                f"Max single {max_single}%, min bonds {min_bond_floor}%",
                "Open Optimizer and Dynamic Backtest tabs",
            ],
        }
    )
    st.dataframe(progress_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown(
        """
### Step-by-step roadmap
1. **Set your constraints first**
   - Define horizon, contributions, and goal value.
   - Choose max tolerable drawdown and concentration style.
2. **Build the core**
   - Start with broad ETFs before adding thematic sleeves.
   - Keep rules explicit: max single weight and bond floor.
3. **Control overlap**
   - Use the **Correlation** tab to remove highly redundant holdings.
4. **Stress test your idea**
   - Use **Optimizer** for weight sanity checks and caps.
   - Use **Dynamic Backtest** to test rebalancing + costs.
5. **Automate discipline**
   - Rebalance on a fixed schedule, not based on emotions.
   - Review quarterly and only change rules when assumptions change.
"""
    )

    with st.expander("Beginner checklist before investing"):
        st.markdown(
            """
- [ ] I know my time horizon and can leave this capital invested.
- [ ] I chose a risk profile that matches my tolerance for losses.
- [ ] No single position is above my max weight rule.
- [ ] I set a bond floor and concentration style.
- [ ] I defined a rebalancing rule (calendar or threshold based).
- [ ] I understand fees, taxes, and expected volatility.
"""
        )

    st.caption(
        "Educational use only: this dashboard provides learning support, not personalized investment advice."
    )

# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------
with tab_prices:
    section_header(
        "Adjusted Prices",
        """
**What is it?**  
Historical prices adjusted for corporate actions.

**How it is computed**  
- `Adjusted Price_t` = `Close_t` corrected for splits and dividends.
- Source: Yahoo Finance adjusted close series.

This makes returns more comparable over long periods.
""",
    )
    st.plotly_chart(prices_chart(px_df), use_container_width=True)
    st.caption("Adjusted for dividends/splits. Table shows the latest rows (most recent on top).")

    # 1) garantisci indice datetime
    df_show = px_df.copy()
    df_show.index = pd.to_datetime(df_show.index)

    # 2) ordina crescente -> prendi le ultime N -> ribalta l'ordine
    N = 5  # cambia se vuoi più/meno righe
    df_show = df_show.sort_index(ascending=True).tail(N).iloc[::-1]

    # 3) (opzionale) porta la data come colonna visibile
    df_show = df_show.reset_index().rename(columns={"index": "Date"})

    st.dataframe(df_show, use_container_width=True)

# --------------------------------------------------------------------------
# Performance
# --------------------------------------------------------------------------
with tab_perf:
    section_header(
        "Cumulative Returns",
        """
**What is it?**  
Normalized performance of each asset from the start date.

**Formula**  
`Cumulative_t = Price_t / Price_0`

Interpretation: value of 1.30 means +30% vs initial date.
""",
    )
    st.plotly_chart(perf_cum_chart(px_df), use_container_width=True)
    st.caption("Prices normalized at start (t=0): Performance_t = Price_t / Price_0.")

    section_header(
        "Risk vs Return (annualized)",
        """
**What is it?**  
Scatter plot of expected annual return vs annual volatility.

**Formulas**  
- `Return_ann ≈ (1 + mean(daily_return))^252 - 1`
- `Vol_ann = std(daily_return) * sqrt(252)`

Each point is one asset/ticker.
""",
    )
    st.plotly_chart(rr_scatter(summary, show_ols=False), use_container_width=True)

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
    section_header(
        f"Rolling Volatility (window = {window} days, annualized)",
        f"""
**What is it?**  
Time-varying risk computed on a moving window.

**Formula**  
`RollingVol_t = std(daily_returns[t-{window}:t]) * sqrt(252)`

Higher values indicate more unstable returns.
""",
    )
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
    section_header(
        "Correlation matrix (ordered)",
        """
**What is it?**  
Pearson correlation between daily returns of each pair of assets.

**Formula**  
`corr(X, Y) = cov(X, Y) / (std(X) * std(Y))`

Values range from -1 (opposite moves) to +1 (same direction).
""",
    )
    st.plotly_chart(corr_heatmap(cm), use_container_width=True)
    st.caption("Pearson correlations of daily returns.")

    st.markdown("**Pairwise rolling correlation**")
    with st.popover("ⓘ Pairwise correlation"):
        st.markdown(
            f"Computed as rolling Pearson correlation on **{window}** daily observations for the selected pair."
        )
    pairs = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            pairs.append(f"{tickers[i]} — {tickers[j]}")
    pair_choice = st.selectbox("Select pair", options=pairs)
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
    section_header(
        "Optimizer",
        """
**What is it?**  
Mean-Variance optimization under weight constraints.

**Objective (Max Sharpe)**  
maximize `(w^T μ - r_f) / sqrt(w^T Σ w)`

with constraints such as `sum(w)=1` and per-asset weight bounds.
""",
    )

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    allow_short = c1.checkbox("Allow shorting", value=False)
    max_cap_default = float(st.session_state.get("guide_max_weight_hint", 0.30))
    max_cap_default = float(min(max(max_cap_default, 0.05), 1.00))
    max_cap = c2.slider("Max weight cap", min_value=0.05, max_value=1.00, value=max_cap_default, step=0.05)
    frontier_points = c3.slider("Frontier points", min_value=5, max_value=50, value=25, step=1)
    show_frontier = c4.checkbox("Show frontier", value=True)
    if "guide_max_weight_hint" in st.session_state:
        st.caption(f"Guide hint applied: max weight cap suggested at {max_cap_default:.0%}.")

    # === Optimization mode (dark-mode friendly) ===
    _options = ["max_sharpe", "min_vol", "target_return"]
    # etichetta palese per capire se il codice è quello nuovo
    try:
       mode = st.segmented_control("Optimization mode", options=_options, default="max_sharpe")
    except Exception:
       mode = st.radio("Optimization mode", options=_options, index=0, horizontal=True)


    if mode == "target_return":
        target_ret = st.slider("Target annual return (%)", 0.0, 40.0, 10.0, 0.5) / 100.0
    else:
        target_ret = None

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
            shorting=allow_short,
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
        st.caption("Slices below 0.01% are grouped into 'Other' to keep labels readable.")

        st.markdown("**Historical performance of this optimized portfolio**")
        hist_nav = _historical_nav_from_weights(px_df, mv_weights, initial_value=100.0)
        if not hist_nav.empty:
            bench_col = "SPY" if "SPY" in px_df.columns else px_df.columns[0]
            bench = (px_df[bench_col].dropna() / px_df[bench_col].dropna().iloc[0]) * 100.0
            compare = pd.concat(
                [hist_nav["nav"].rename("MV Portfolio"), bench.rename(bench_col)],
                axis=1,
                join="inner",
            ).dropna()
            fig_hist = px.line(_with_date(compare), x="Date", y=compare.columns, template="plotly_dark")
            fig_hist.update_layout(
                title=dict(text=f"Historical NAV (vs {bench_col})", font=dict(size=16)),
                margin=dict(l=10, r=10, t=50, b=10),
            )
            fig_hist.update_xaxes(title="")
            fig_hist.update_yaxes(title="NAV (base 100)")
            st.plotly_chart(fig_hist, use_container_width=True)
            st.caption("Static-weight backtest from selected start date. Benchmark is SPY if included, otherwise first ticker.")

            nav_s = hist_nav["nav"].astype(float)
            years = max(len(nav_s) / 252.0, 1e-9)
            total_ret = nav_s.iloc[-1] / nav_s.iloc[0] - 1.0
            cagr = (nav_s.iloc[-1] / nav_s.iloc[0]) ** (1.0 / years) - 1.0
            drawdown = nav_s / nav_s.cummax() - 1.0
            max_dd = float(drawdown.min())
            hh1, hh2, hh3 = st.columns(3)
            hh1.metric("Total return", f"{total_ret:.2%}")
            hh2.metric("CAGR", f"{cagr:.2%}")
            hh3.metric("Max drawdown", f"{max_dd:.2%}")
            dd_fig = px.area(
                _with_date(drawdown.to_frame(name="drawdown")),
                x="Date",
                y="drawdown",
                template="plotly_dark",
            )
            dd_fig.update_layout(
                title=dict(text="Portfolio drawdown", font=dict(size=15)),
                margin=dict(l=10, r=10, t=45, b=10),
            )
            dd_fig.update_yaxes(tickformat=".1%")
            st.plotly_chart(dd_fig, use_container_width=True)
        else:
            st.info("Not enough data to compute historical portfolio performance.")

        st.markdown("**Monte Carlo simulation (future portfolio paths)**")
        mc_col1, mc_col2, mc_col3 = st.columns([1, 1, 1])
        mc_horizon = mc_col1.slider("Horizon (trading days)", min_value=21, max_value=756, value=252, step=21)
        mc_sims = mc_col2.slider("Number of simulations", min_value=50, max_value=1500, value=300, step=50)
        mc_seed = mc_col3.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)

        if st.button("Run Monte Carlo", key="run_mc_optimizer"):
            try:
                sim_paths = simulate_portfolio_paths(
                    returns=ret,
                    weights=mv_weights,
                    horizon_days=int(mc_horizon),
                    n_sims=int(mc_sims),
                    initial_value=100.0,
                    random_seed=int(mc_seed),
                )
                st.plotly_chart(
                    monte_carlo_paths_chart(sim_paths, title="Monte Carlo Projection (MV Portfolio)"),
                    use_container_width=True,
                )
                end_vals = sim_paths.iloc[-1]
                p10, p50, p90 = np.percentile(end_vals.values, [10, 50, 90])
                mc_k1, mc_k2, mc_k3 = st.columns(3)
                mc_k1.metric("P10 final value", f"{p10:.2f}")
                mc_k2.metric("Median final value", f"{p50:.2f}")
                mc_k3.metric("P90 final value", f"{p90:.2f}")
            except Exception as mc_err:
                st.warning(f"Monte Carlo simulation error: {mc_err}")

        # KPI (if not provided, compute from weights)
        if isinstance(mv_out, dict) and all(k in mv_out for k in ("return", "vol", "sharpe")):
            mv_kpi = {"return_": mv_out["return"], "vol": mv_out["vol"], "sharpe": mv_out["sharpe"]}
        else:
            mv_kpi = _mv_kpi_from_weights(px_df, mv_weights, rf_annual=rf/100.0)

        k1, k2, k3 = st.columns(3)
        k1.metric("Return (MV)", f"{mv_kpi['return_']:.2%}")
        k2.metric("Volatility (MV)", f"{mv_kpi['vol']:.2%}")
        k3.metric("Sharpe (MV)", f"{mv_kpi['sharpe']:.2f}")

        mv_tbl = mv_weights.sort_values(ascending=False).rename("Weight").to_frame()
        st.markdown("**MV weights table**")
        st.dataframe(
            (mv_tbl * 100.0).rename(columns={"Weight": "Weight (%)"}).style.format({"Weight (%)": "{:.2f}"}),
            use_container_width=True,
        )
        st.download_button(
            "Download MV weights CSV",
            data=mv_tbl.to_csv().encode(),
            file_name="mv_weights.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_mv_weights",
        )

        if show_frontier:
            fr_raw = _compute_frontier(
                px_df, rf=rf/100.0, n_points=frontier_points,
                shorting=allow_short, max_weight=max_cap
            )
            fr = _normalize_frontier(fr_raw)
            st.plotly_chart(frontier_chart(fr), use_container_width=True)
            st.caption("Estimated efficient frontier: portfolios at different risk/return levels.")
        else:
            st.caption("Frontier hidden.")
    except Exception as e:
        st.warning(f"MV optimization error: {e}")

    st.markdown("---")
    section_header(
        "CVaR Optimizer (Expected Shortfall)",
        """
**What is it?**  
Optimization focused on tail risk rather than variance.

**Intuition**  
CVaR measures the average loss in the worst `α%` scenarios.

Useful when downside risk matters more than symmetric volatility.
""",
    )
    alpha = st.slider("Confidence (1−α)", min_value=0.80, max_value=0.99, value=0.95, step=0.01)

    # ===== CVaR optimize =====
    try:
        cv_kwargs = dict(
            prices=px_df,
            alpha=alpha,
            shorting=allow_short,
            max_weight=max_cap,
        )
        cv_out = optimize_cvar(**cv_kwargs)
        cv_weights = _as_weights(cv_out)
        st.plotly_chart(
            weights_donut(cv_weights, title="Portfolio Weights (CVaR)"),
            use_container_width=True
        )
        cv_tbl = cv_weights.sort_values(ascending=False).rename("Weight").to_frame()
        st.download_button(
            "Download CVaR weights CSV",
            data=cv_tbl.to_csv().encode(),
            file_name="cvar_weights.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_cvar_weights",
        )

        if isinstance(cv_out, dict) and "curve" in cv_out and isinstance(cv_out["curve"], pd.DataFrame):
            st.plotly_chart(px.line(cv_out["curve"], x="Vol", y="Return"), use_container_width=True)
            st.caption("Risk/return curve using CVaR (Expected Shortfall).")

    except NotImplementedError:
        st.info("CVaR optimization not available in this build.")
    except Exception as e:
        st.warning(f"CVaR optimizer error: {e}")

# --------------------------------------------------------------------------
# Dynamic Backtest (clean layout)
# --------------------------------------------------------------------------
with tab_backtest:
    bt_title_col, bt_info_col = st.columns([0.94, 0.06])
    with bt_title_col:
        st.markdown('<div class="qid-h1">DYNAMIC BACKTEST</div>', unsafe_allow_html=True)
    with bt_info_col:
        with st.popover("ⓘ"):
            st.markdown(
                """
**What is it?**  
Out-of-sample simulation with periodic rebalancing.

**How it works**  
- Estimate moments with Rolling/EWMA.
- Build weights with MV objective and constraints.
- Apply turnover, commissions and slippage at each rebalance.

Outputs include NAV, turnover, costs and evolving weights.
"""
            )
    st.markdown('<div class="qid-sub">Rolling / EWMA engine with covariance shrinkage & trading costs</div>', unsafe_allow_html=True)
    st.divider()

    # Estimation controls
    st.markdown("**Estimation**")
    col_est1, col_est2, col_est3 = st.columns([1,1,1])
    with col_est1:
        use_ewma = st.checkbox("Use EWMA (else Rolling)", value=True, key="bt_use_ewma")
    with col_est2:
        ewma_lam = st.slider("EWMA λ", 0.80, 0.995, 0.97, 0.001, key="bt_ewma")
    with col_est3:
        roll_win  = st.slider("Rolling window (days)", 30, 252, 90, 1, key="bt_win")

    # Shrinkage controls
    st.markdown("**Shrinkage**")
    col_shr1, col_shr2 = st.columns([1,1])
    with col_shr1:
        shrink_method = st.selectbox(
            "Target",
            options=["none", "const-cor", "diag", "identity"],
            index=1,
            key="bt_shr_m",
        )
    with col_shr2:
        shrink_intensity = st.slider("Intensity γ", 0.0, 1.0, 0.25, 0.05, key="bt_shr_g")

    # Rebalance controls
    st.markdown("**Rebalance**")
    col_reb1, col_reb2 = st.columns([1,1])
    with col_reb1:
        reb_k = st.slider("Every k days", 1, 63, 21, 1, key="bt_reb_k")
    with col_reb2:
        allow_short_bt = st.checkbox("Allow short", value=False, key="bt_short")

    # Risk & Costs
    st.markdown("**Risk model**")
    col_r1, col_r2 = st.columns([1,1])
    with col_r1:
        gamma = st.number_input("Risk aversion γ", min_value=0.1, value=5.0, step=0.1, key="bt_gamma")
    with col_r2:
        ridge = st.number_input("Ridge on Σ", min_value=0.0, value=1e-3, step=1e-3, format="%.4f", key="bt_ridge")

    st.markdown("**Trading costs**")
    col_c1, col_c2, col_c3 = st.columns([1,1,1])
    with col_c1:
        tx_bps  = st.number_input("Commissions (bps)", min_value=0.0, value=5.0, step=0.5, key="bt_tx")
    with col_c2:
        slp_bps = st.number_input("Slippage (bps)", min_value=0.0, value=1.0, step=0.5, key="bt_slip")
    with col_c3:
        turn_L2 = st.number_input("Turnover penalty λ (L2)", min_value=0.0, value=5.0, step=0.5, key="bt_turn")

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
    section_header(
        "Data / Export",
        """
Download the current analysis outputs (tables/reports) for documentation and reuse.
""",
    )
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
