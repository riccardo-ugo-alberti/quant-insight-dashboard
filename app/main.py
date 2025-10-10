# app/main.py
from __future__ import annotations
import io
import json
import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import fetch_prices
from src.analytics import (
    to_returns, cum_returns, summary_table, simulate_portfolios
)
from src.optimizer import mean_variance_opt, cvar_opt
from src.factors import get_fama_french, regress_ff
from src.econ import default_macro
from src.visuals import style_table, mc_scatter, factor_bars, macro_lines
from src.report import build_html_report

st.set_page_config(page_title="Quant Insight Dashboard", layout="wide")

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("Parameters")
    tickers_input = st.text_input("Tickers", "AAPL, MSFT, SPY")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    window = st.slider("Rolling window (days)", 30, 252, 90)
    rf = st.number_input("Risk-free (annual, %)", value=2.0, step=0.1) / 100.0
    bench = st.selectbox("Beta benchmark", options=["SPY", "QQQ", "^GSPC"], index=0)
    st.checkbox("Color Risk-Return by sector (when available)", value=True, key="sector_color")

    st.subheader("Correlation tools")
    pair_choice = st.selectbox("Pair for rolling correlation", options=["— none —"] + tickers, index=0)

    st.button("Run Analysis", type="primary", use_container_width=True)

# ------------------ LOAD DATA ------------------
if len(tickers) == 0:
    st.stop()

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
    st.subheader("Adjusted prices (normalized)")
    eq = prices / prices.iloc[0]
    st.line_chart(eq, use_container_width=True)
    st.caption("Prices normalized to 1 at the first date. Use the app's controls to change window and risk-free.")

# -------------- PERFORMANCE -------------------
with tabs[1]:
    st.subheader("Cumulative returns")
    st.line_chart(cum_returns(returns_d), use_container_width=True)

    st.subheader("Performance summary")
    st.plotly_chart(style_table(summary.round(3)), use_container_width=True)

# -------------- VOLATILITY --------------------
with tabs[2]:
    st.subheader("Rolling annualized volatility")
    ann = returns_d.rolling(window).std().dropna() * np.sqrt(252)
    st.line_chart(ann, use_container_width=True)
    st.caption(f"Rolling window = {window} days; annualization via √252.")

# -------------- CORRELATION -------------------
with tabs[3]:
    st.subheader("Correlation matrix (Pearson)")
    cm = returns_d.corr()
    st.dataframe(cm.round(2), use_container_width=True)

# -------------- OPTIMIZER ---------------------
with tabs[4]:
    colL, colR = st.columns([1.2, 1.0])
    with colL:
        st.subheader("Mean-Variance Optimizer")
        allow_short = st.checkbox("Allow shorting", value=False)
        weight_cap = st.slider("Max weight cap", 0.1, 1.0, 0.3, 0.05)
        mode = st.radio("Optimization mode", ["Max Sharpe", "Min Vol", "Target Return"], horizontal=True)
        target = None
        if mode == "Target Return":
            target = st.slider("Target annual return", 0.0, 0.5, 0.15, 0.01)

        try:
            mv = mean_variance_opt(
                returns_d, mode="min_vol" if mode=="Min Vol" else ("target" if mode=="Target Return" else "max_sharpe"),
                target_return=target, rf=rf, allow_short=allow_short, weight_cap=weight_cap
            )
            st.success(f"MV: μ={mv['mu']:.2%}, σ={mv['sigma']:.2%}, Sharpe={mv['sharpe']:.2f}")
        except Exception as e:
            st.error(f"MV optimization error: {e}")
            mv = None

        st.subheader("CVaR Optimizer (ES)")
        alpha = st.slider("Confidence (1-α)", 0.80, 0.99, 0.95, 0.01)
        try:
            cv = cvar_opt(returns_d, alpha=alpha, allow_short=allow_short, weight_cap=weight_cap)
            st.info(f"CVaR (ES {alpha:.0%}) minimized to: {cv['es']:.3%}")
        except Exception as e:
            st.error(f"CVaR optimization error: {e}")
            cv = None

    with colR:
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
    allow_short = st.checkbox("Allow shorting (MC)", value=False, key="mc_short")
    weight_cap = st.slider("Weight cap (MC)", 0.1, 1.0, 0.3, 0.05, key="mc_cap")

    sims = simulate_portfolios(returns_d, n_sims=n_sims, allow_short=allow_short,
                               weight_cap=weight_cap, rf=rf)
    st.plotly_chart(mc_scatter(sims, opt_point=None), use_container_width=True)
    st.caption("Each dot is a random feasible portfolio. Color = Sharpe. Use Optimizer tab for optimal weights.")

# -------------- FACTORS (Fama–French) ---------
with tabs[6]:
    st.subheader("Fama–French factor decomposition")
    freq = st.radio("Frequency", ["Monthly", "Daily"], horizontal=True)
    five = st.checkbox("Use 5-Factor model (otherwise 3-Factor)", value=True)
    try:
        ff = get_fama_french(freq="M" if freq=="Monthly" else "D", five=five)
        # Convert asset returns to same frequency
        r = to_returns(prices, "M" if freq=="Monthly" else "D")
        # Excess returns (over RF)
        ex = r.sub(ff["RF"], axis=0, fill_value=0.0)
        res = regress_ff(ex[returns_d.columns], ff, five=five)
        st.plotly_chart(style_table(res.round(3), "Loadings / Alpha / R2"), use_container_width=True)
        st.plotly_chart(factor_bars(res, five=five), use_container_width=True)
    except Exception as e:
        st.error(f"Factor regression error: {e}")
        st.caption("Tip: pandas_datareader 'famafrench' sometimes throttles — retry or switch frequency.")

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
        st.caption("If needed, set a FRED API key in Secrets as FRED_API_KEY.")

# -------------- DATA / EXPORT -----------------
with tabs[8]:
    st.subheader("Data snapshots")
    st.dataframe(prices.tail(), use_container_width=True)
    st.dataframe(returns_d.tail(), use_container_width=True)

    st.subheader("Export")
    csv_buf = io.StringIO()
    prices.to_csv(csv_buf)
    st.download_button("Download Prices (CSV)", data=csv_buf.getvalue().encode("utf-8"),
                       file_name="prices.csv", mime="text/csv", use_container_width=True)

    html_bytes = build_html_report(prices, summary, returns_d.corr()).getvalue().encode("utf-8")
    st.download_button("Download HTML Report", data=html_bytes,
                       file_name="report.html", mime="text/html", use_container_width=True)
