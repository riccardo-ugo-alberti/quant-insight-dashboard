# app/main.py
from __future__ import annotations
import os, sys, io, json
import streamlit as st
import pandas as pd
import numpy as np

# --- ensure project root is on sys.path for Streamlit Cloud ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- app imports (coerenti con i file condivisi) ---
from src.data_loader import fetch_prices
from src.analytics import (
    to_returns, cum_returns, summary_table, simulate_portfolios
)
from src.optimizer import mean_variance_opt, cvar_opt
from src.factors import get_fama_french, regress_ff
from src.econ import default_macro
from src.visuals import style_table, mc_scatter, factor_bars, macro_lines
from src.report import build_html_report

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Quant Insight Dashboard", layout="wide")
st.title("Quant Insight Dashboard")
st.caption("Performance, risk, correlation, factor models and portfolio optimization for equities/ETFs.")
st.markdown("---")

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("Parameters")

    tickers_input = st.text_input("Tickers", "AAPL, MSFT, SPY")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    window = st.slider("Rolling window (days)", 30, 252, 90, 1)
    rf = st.number_input("Risk-free (annual, %)", value=2.0, step=0.1) / 100.0
    bench = st.selectbox("Beta benchmark", options=["SPY", "QQQ", "^GSPC"], index=0)

    st.subheader("Correlation tools")
    # (Per semplicità: visualizziamo la matrice; rolling-pair è opzionale)
    st.caption("Use the Correlation tab to inspect the matrix; rolling pair can be added later.")

    st.markdown("---")
    st.subheader("Save / Load setup")
    cfg = {"tickers": tickers}
    st.download_button("Download configuration (JSON)",
                       data=json.dumps(cfg).encode("utf-8"),
                       file_name="qid-config.json",
                       use_container_width=True)
    uploaded = st.file_uploader("Upload configuration (JSON)", type=["json"], label_visibility="collapsed")
    if uploaded:
        try:
            data = json.load(uploaded)
            if isinstance(data.get("tickers"), list):
                tickers[:] = [t.strip().upper() for t in data["tickers"] if t.strip()]
                st.success("Configuration loaded. Check tickers above.")
        except Exception as e:
            st.error(f"Invalid config: {e}")

# ------------------ LOAD DATA ------------------
if len(tickers) == 0:
    st.info("Add at least one ticker in the sidebar to start.")
    st.stop()

with st.spinner("Downloading market data..."):
    prices = fetch_prices(tickers, start="2020-01-01")

returns_d = to_returns(prices, "D")
summary = summary_table(returns_d, rf=rf)

# -------------- LAYOUT TABS -------------------
tabs = st.tabs([
    "Prices", "Performance", "Volatility", "Correlation",
    "Optimizer", "Simulator", "Factors", "Macro", "Data / Export"
])

# -------------- PRICES ------------------------
with tabs[0]:
    st.subheader("Adjusted prices (normalized to 1)")
    eq = prices / prices.iloc[0]
    st.line_chart(eq, use_container_width=True)
    st.caption("Prices normalized to 1 at the first date.")

# -------------- PERFORMANCE -------------------
with tabs[1]:
    st.subheader("Cumulative returns")
    st.line_chart(cum_returns(returns_d), use_container_width=True)

    st.subheader("Performance summary")
    st.plotly_chart(style_table(summary.round(3), title="Key metrics"), use_container_width=True)

# -------------- VOLATILITY --------------------
with tabs[2]:
    st.subheader("Rolling annualized volatility")
    ann = returns_d.rolling(window).std().dropna() * np.sqrt(252)
    st.line_chart(ann, use_container_width=True)
    st.caption(f"Rolling window = {window} days; annualized with √252.")

# -------------- CORRELATION -------------------
with tabs[3]:
    st.subheader("Correlation matrix (Pearson)")
    cm = returns_d.corr()
    st.dataframe(cm.round(2), use_container_width=True)
    st.caption("Ordered heatmap and rolling pair charts can be added as enhancements.")

# -------------- OPTIMIZER ---------------------
with tabs[4]:
    left, right = st.columns([1.2, 1.0], gap="large")

    with left:
        st.subheader("Mean–Variance Optimizer")
        allow_short = st.checkbox("Allow shorting (MV & CVaR)", value=False)
        weight_cap = st.slider("Max weight cap", 0.1, 1.0, 0.3, 0.05)

        mode = st.radio("Optimization mode", ["Max Sharpe", "Min Vol", "Target Return"], horizontal=True)
        target = None
        if mode == "Target Return":
            target = st.slider("Target annual return", 0.00, 0.50, 0.15, 0.01)

        try:
            mv = mean_variance_opt(
                returns_d,
                mode="min_vol" if mode == "Min Vol" else ("target" if mode == "Target Return" else "max_sharpe"),
                target_return=target,
                rf=rf,
                allow_short=allow_short,
                weight_cap=weight_cap,
            )
            st.success(f"MV: μ={mv['mu']:.2%}, σ={mv['sigma']:.2%}, Sharpe={mv['sharpe']:.2f}")
        except Exception as e:
            st.error(f"MV optimization error: {e}")
            mv = None

        st.subheader("CVaR Optimizer (Expected Shortfall)")
        alpha = st.slider("Confidence (1-α)", 0.80, 0.99, 0.95, 0.01)
        try:
            cv = cvar_opt(returns_d, alpha=alpha, allow_short=allow_short, weight_cap=weight_cap)
            st.info(f"CVaR (ES {alpha:.0%}) minimized to: {cv['es']:.3%}")
        except Exception as e:
            st.warning(f"CVaR optimization not available: {e}")
            cv = None

    with right:
        st.subheader("Weights")
        if mv is not None:
            w = pd.Series(mv["weights"], index=returns_d.columns, name="MV Weights")
            st.dataframe(w.to_frame().style.format("{:.2%}"), use_container_width=True)
        if cv is not None:
            w2 = pd.Series(cv["weights"], index=returns_d.columns, name="CVaR Weights")
            st.dataframe(w2.to_frame().style.format("{:.2%}"), use_container_width=True)

# -------------- SIMULATOR (Monte Carlo) -------
with tabs[5]:
    st.subheader("Monte Carlo portfolios")
    n_sims = st.slider("Number of simulations", 100, 10000, 3000, step=100)
    allow_short_mc = st.checkbox("Allow shorting (MC)", value=False, key="mc_short")
    weight_cap_mc = st.slider("Weight cap (MC)", 0.1, 1.0, 0.3, 0.05, key="mc_cap")

    sims = simulate_portfolios(returns_d, n_sims=n_sims, allow_short=allow_short_mc,
                               weight_cap=weight_cap_mc, rf=rf)
    st.plotly_chart(mc_scatter(sims, opt_point=None), use_container_width=True)
    st.caption("Each dot is a random feasible portfolio. Color = Sharpe. Use Optimizer tab for optimal weights.")

# -------------- FACTORS (Fama–French) ---------
with tabs[6]:
    st.subheader("Fama–French factor decomposition")
    freq = st.radio("Frequency", ["Monthly", "Daily"], horizontal=True, index=0)
    five = st.checkbox("Use 5-Factor model (otherwise 3-Factor)", value=True)
    try:
        ff = get_fama_french(freq="M" if freq == "Monthly" else "D", five=five)
        # Align returns frequency to factors
        r = to_returns(prices, "M" if freq == "Monthly" else "D")
        # Excess returns (over RF)
        ex = r.sub(ff["RF"], axis=0, fill_value=0.0)
        cols = [c for c in returns_d.columns if c in ex.columns]
        if not cols:
            st.warning("No overlap between selected tickers and factor data. Try different frequency.")
        else:
            res = regress_ff(ex[cols], ff, five=five)
            st.plotly_chart(style_table(res.round(3), "Loadings / Alpha / R²"), use_container_width=True)
            st.plotly_chart(factor_bars(res, five=five), use_container_width=True)
    except Exception as e:
        st.error(f"Factor regression error: {e}")
        st.caption("Tip: Fama–French endpoint can throttle. Retry or switch Daily/Monthly.")

# -------------- MACRO (FRED) ------------------
with tabs[7]:
    st.subheader("Macro indicators (FRED)")
    start = st.date_input("Start date", value=pd.to_datetime("2010-01-01")).strftime("%Y-%m-%d")
    try:
        macro = default_macro(start=start)
        if macro.empty:
            st.warning("Could not fetch FRED data. Try again later.")
        else:
            st.plotly_chart(macro_lines(macro), use_container_width=True)
            st.dataframe(macro.tail(), use_container_width=True)
            st.caption("Series: GDPC1 (real GDP, quarterly), CPIAUCSL (CPI), DGS10 (10Y Treasury yield).")
    except Exception as e:
        st.error(f"FRED error: {e}")

# -------------- DATA / EXPORT -----------------
with tabs[8]:
    st.subheader("Data snapshots")
    st.dataframe(prices.tail(), use_container_width=True)
    st.dataframe(returns_d.tail(), use_container_width=True)

    st.subheader("Export")
    csv_buf = io.StringIO()
    prices.to_csv(csv_buf)
    st.download_button("Download Prices (CSV)",
                       data=csv_buf.getvalue().encode("utf-8"),
                       file_name="prices.csv",
                       mime="text/csv",
                       use_container_width=True)

    # HTML report: build_html_report -> StringIO; convert to bytes
    html_bytes = build_html_report(prices, summary, returns_d.corr()).getvalue().encode("utf-8")
    st.download_button("Download HTML Report",
                       data=html_bytes,
                       file_name="report.html",
                       mime="text/html",
                       use_container_width=True)
