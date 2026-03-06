from __future__ import annotations

import datetime as dt
import io
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Bootstrap path so `src` is importable locally and on Streamlit Cloud.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.analytics import compute_summary, rolling_vol, simulate_portfolio_paths, to_returns
from src.data_loader import fetch_prices
from src.optimizer import frontier_from_prices, optimize_cvar, optimize_portfolio
from src.report import build_html_report
from src.visuals import corr_heatmap, frontier_chart, perf_cum_chart, prices_chart, vol_chart, weights_donut

st.set_page_config(page_title="Quant Insight Dashboard", layout="wide")

PRESETS = OrderedDict(
    {
        "Balanced Core": ["SPY", "VEA", "VWO", "BND", "IEF", "GLD"],
        "Conservative": ["VT", "BND", "IEF", "LQD", "GLD"],
        "Growth": ["QQQ", "SPY", "SMH", "IWM", "VEA"],
    }
)

STARTER_MODELS = {
    "Conservative": {"weights": {"VT": 0.35, "BND": 0.35, "IEF": 0.20, "GLD": 0.10}},
    "Balanced": {"weights": {"VT": 0.50, "BND": 0.25, "IEF": 0.15, "GLD": 0.10}},
    "Growth": {"weights": {"VT": 0.55, "QQQ": 0.25, "BND": 0.10, "GLD": 0.10}},
}


def _sanitize_tickers(raw: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in raw.split(","):
        tk = t.strip().upper()
        if not tk or tk in seen:
            continue
        seen.add(tk)
        out.append(tk)
    return out


def _normalize_weights(weights: pd.Series) -> pd.Series:
    clean = weights.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    clean = clean[clean > 0]
    if clean.empty:
        return clean
    return clean / clean.sum()


def _portfolio_kpi(returns: pd.DataFrame, weights: pd.Series, rf_annual: float) -> dict[str, float]:
    if returns.empty:
        return {"ret": np.nan, "vol": np.nan, "sharpe": np.nan}
    w = _normalize_weights(weights).reindex(returns.columns).fillna(0.0)
    if w.sum() == 0:
        return {"ret": np.nan, "vol": np.nan, "sharpe": np.nan}

    mu = returns.mean() * 252.0
    cov = returns.cov() * 252.0
    wv = w.values
    exp_ret = float(np.dot(wv, mu.values))
    vol = float(np.sqrt(wv @ cov.values @ wv))
    sharpe = (exp_ret - rf_annual) / vol if vol > 0 else np.nan
    return {"ret": exp_ret, "vol": vol, "sharpe": sharpe}


def _portfolio_nav(returns: pd.DataFrame, weights: pd.Series, start_value: float = 100.0) -> pd.Series:
    w = _normalize_weights(weights).reindex(returns.columns).fillna(0.0)
    if w.sum() == 0 or returns.empty:
        return pd.Series(dtype=float)
    port_ret = returns.fillna(0.0).mul(w, axis=1).sum(axis=1)
    return start_value * (1.0 + port_ret).cumprod()


@st.cache_data(ttl=3600, show_spinner=False)
def _load_prices_cached(tickers: tuple[str, ...], start: dt.date, end: dt.date) -> pd.DataFrame:
    return fetch_prices(list(tickers), start=str(start), end=str(end))


st.title("Quant Insight Dashboard")
st.caption("Focus: beginner-friendly portfolio construction and ticker/portfolio analysis.")

with st.sidebar:
    st.header("Setup")
    preset = st.selectbox("Ticker preset", list(PRESETS.keys()))
    if st.button("Apply preset", use_container_width=True):
        st.session_state["tickers_str"] = ", ".join(PRESETS[preset])
        st.rerun()

    tickers_input = st.text_input(
        "Tickers (comma-separated)",
        value=st.session_state.get("tickers_str", ", ".join(PRESETS["Balanced Core"])),
        key="tickers_str",
    )
    tickers = _sanitize_tickers(tickers_input)

    date_range = st.date_input(
        "Date range",
        value=(dt.date.today() - dt.timedelta(days=365 * 5), dt.date.today()),
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = dt.date.today() - dt.timedelta(days=365 * 5)
        end_date = dt.date.today()

    rf_pct = st.number_input("Annual risk-free rate (%)", min_value=0.0, max_value=20.0, value=2.0, step=0.1)
    roll_window = st.slider("Rolling volatility window (days)", min_value=21, max_value=252, value=90)

    st.divider()
    st.caption("Configuration")
    setup_blob = {
        "tickers": tickers,
        "start": str(start_date),
        "end": str(end_date),
        "risk_free_pct": rf_pct,
        "rolling_window": int(roll_window),
    }
    st.download_button(
        "Download setup JSON",
        data=json.dumps(setup_blob, indent=2).encode("utf-8"),
        file_name="portfolio_setup.json",
        mime="application/json",
        use_container_width=True,
    )

if len(tickers) < 2:
    st.info("Enter at least 2 tickers to start.")
    st.stop()

prices_raw = _load_prices_cached(tuple(tickers), start_date, end_date)
if prices_raw is None or prices_raw.empty:
    st.error("No price data returned. Check your tickers and date range.")
    st.stop()

prices = prices_raw.sort_index().dropna(how="all")
prices = prices.loc[:, [c for c in prices.columns if prices[c].notna().sum() > 10]]
if prices.shape[1] < 2:
    st.error("Not enough tickers with valid history. Keep at least 2 valid assets.")
    st.stop()

returns = to_returns(prices).dropna(how="all")
summary = compute_summary(prices, rf=rf_pct / 100.0)
corr = returns.corr().astype(float)
vol_roll = rolling_vol(returns, window=roll_window)

k1, k2, k3 = st.columns(3)
k1.metric("Valid assets", str(prices.shape[1]))
k2.metric("Observations", str(prices.shape[0]))
k3.metric("Missing data", f"{prices.isna().sum().sum() / max(prices.size, 1):.2%}")


tab_builder, tab_tickers, tab_portfolio, tab_export = st.tabs(
    [
        "Portfolio Builder",
        "Ticker Analysis",
        "Portfolio Analysis",
        "Export",
    ]
)

with tab_builder:
    st.subheader("Build a Starter Portfolio")
    st.caption("Beginner workflow: select a profile, define your capital plan, and get a starter allocation.")

    c1, c2, c3 = st.columns(3)
    with c1:
        profile = st.selectbox("Profile", list(STARTER_MODELS.keys()), index=1)
    with c2:
        initial_capital = st.number_input("Initial capital (EUR)", min_value=0, value=10000, step=500)
    with c3:
        monthly_contribution = st.number_input("Monthly contribution (EUR)", min_value=0, value=300, step=50)

    horizon_years = st.slider("Horizon (years)", min_value=1, max_value=30, value=10)

    model_weights = pd.Series(STARTER_MODELS[profile]["weights"], name="Weight")
    model_weights = _normalize_weights(model_weights)

    available = model_weights[model_weights.index.isin(prices.columns)]
    missing = [t for t in model_weights.index if t not in available.index]

    if available.empty:
        st.warning("The selected model has no tickers available in the current dataset.")
    else:
        st.plotly_chart(weights_donut(available, title=f"Starter allocation ({profile})"), use_container_width=True)
        if missing:
            st.info(f"Tickers not available in the current history: {', '.join(missing)}")

        starter_kpi = _portfolio_kpi(returns[available.index], available, rf_annual=rf_pct / 100.0)
        m1, m2, m3 = st.columns(3)
        m1.metric("Expected annual return", f"{starter_kpi['ret']:.2%}" if np.isfinite(starter_kpi["ret"]) else "n/a")
        m2.metric("Expected annual volatility", f"{starter_kpi['vol']:.2%}" if np.isfinite(starter_kpi["vol"]) else "n/a")
        m3.metric("Sharpe", f"{starter_kpi['sharpe']:.2f}" if np.isfinite(starter_kpi["sharpe"]) else "n/a")

        annual_ret = starter_kpi["ret"] if np.isfinite(starter_kpi["ret"]) else 0.0
        months = horizon_years * 12
        monthly_rate = (1.0 + annual_ret) ** (1.0 / 12.0) - 1.0
        fv_initial = float(initial_capital) * (1.0 + monthly_rate) ** months
        if monthly_rate != 0:
            fv_annuity = float(monthly_contribution) * (((1.0 + monthly_rate) ** months - 1.0) / monthly_rate)
        else:
            fv_annuity = float(monthly_contribution) * months
        projected_value = fv_initial + fv_annuity
        st.metric(f"Projected value in {horizon_years} years", f"EUR {projected_value:,.0f}")

        if st.button("Use this portfolio in Portfolio Analysis", use_container_width=True):
            st.session_state["portfolio_lab_weights"] = available.to_dict()
            st.success("Starter allocation loaded in Portfolio Analysis.")

with tab_tickers:
    st.subheader("Ticker Analysis")

    st.plotly_chart(prices_chart(prices, title="Prices"), use_container_width=True)
    st.plotly_chart(perf_cum_chart(prices, title="Cumulative performance (base 1)"), use_container_width=True)

    st.markdown("**Ticker metrics table**")
    summary_show = summary.copy().sort_values("Sharpe", ascending=False)
    st.dataframe(
        (summary_show * pd.Series({"Return": 100, "Vol": 100, "Sharpe": 1}))
        .rename(columns={"Return": "Return %", "Vol": "Vol %", "Sharpe": "Sharpe"})
        .style.format({"Return %": "{:.2f}", "Vol %": "{:.2f}", "Sharpe": "{:.2f}"}),
        use_container_width=True,
    )

    st.plotly_chart(vol_chart(vol_roll, title=f"Rolling volatility ({roll_window} days)"), use_container_width=True)
    st.plotly_chart(corr_heatmap(corr, title="Correlation matrix"), use_container_width=True)

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().sort_values(ascending=False)
    if not upper.empty:
        cpos, cneg = st.columns(2)
        with cpos:
            st.markdown("**Top 5 positive correlations**")
            for (a, b), v in upper.head(5).items():
                st.write(f"{a} - {b}: {v:.2f}")
        with cneg:
            st.markdown("**Top 5 negative correlations**")
            for (a, b), v in upper.tail(5).sort_values().items():
                st.write(f"{a} - {b}: {v:.2f}")

with tab_portfolio:
    st.subheader("Portfolio Analysis")
    st.caption("Compare manual allocation and optimization with a clean, focused workflow.")

    assets = st.multiselect("Included assets", options=list(prices.columns), default=list(prices.columns))
    if len(assets) < 2:
        st.info("Select at least 2 assets.")
    else:
        method = st.radio("Method", ["Manual", "Optimized Max Sharpe", "Optimized Min Vol", "Optimized CVaR"], horizontal=True)
        sub_prices = prices[assets]
        sub_returns = returns[assets]

        final_weights = pd.Series(dtype=float)

        if method == "Manual":
            seed = st.session_state.get("portfolio_lab_weights", {a: 1.0 / len(assets) for a in assets})
            seed_series = pd.Series(seed).reindex(assets).fillna(0.0)
            if seed_series.sum() <= 0:
                seed_series = pd.Series({a: 1.0 / len(assets) for a in assets})
            seed_series = _normalize_weights(seed_series)

            edit_df = pd.DataFrame({"Ticker": assets, "Weight %": (seed_series.reindex(assets).fillna(0.0) * 100).round(2)})
            edited = st.data_editor(edit_df, hide_index=True, num_rows="fixed", use_container_width=True)
            final_weights = pd.Series(edited["Weight %"].values, index=edited["Ticker"].values, dtype=float) / 100.0
            final_weights = _normalize_weights(final_weights)

        else:
            max_weight = st.slider("Max weight per asset", min_value=0.15, max_value=1.0, value=0.35, step=0.05)
            allow_short = st.checkbox("Allow shorting", value=False)

            if method in ("Optimized Max Sharpe", "Optimized Min Vol"):
                mode = "max_sharpe" if method.endswith("Sharpe") else "min_vol"
                opt_w, _, _ = optimize_portfolio(
                    prices=sub_prices,
                    rf=rf_pct / 100.0,
                    mode=mode,
                    shorting=allow_short,
                    max_weight=float(max_weight),
                )
                final_weights = _normalize_weights(opt_w)

            if method == "Optimized CVaR":
                alpha = st.slider("CVaR confidence", min_value=0.80, max_value=0.99, value=0.95, step=0.01)
                cv_w = optimize_cvar(
                    prices=sub_prices,
                    alpha=float(alpha),
                    shorting=allow_short,
                    max_weight=float(max_weight),
                )
                if cv_w is None:
                    st.warning("CVaR is not available in this environment. Use Max Sharpe or Min Vol.")
                else:
                    final_weights = _normalize_weights(cv_w)

        if final_weights.empty:
            st.info("No valid portfolio was computed.")
        else:
            final_weights = final_weights.reindex(assets).fillna(0.0)
            final_weights = _normalize_weights(final_weights)

            st.plotly_chart(weights_donut(final_weights, title="Portfolio weights"), use_container_width=True)
            st.dataframe(
                (final_weights.sort_values(ascending=False) * 100.0)
                .rename("Weight %")
                .to_frame()
                .style.format({"Weight %": "{:.2f}"}),
                use_container_width=True,
            )

            kpi = _portfolio_kpi(sub_returns, final_weights, rf_annual=rf_pct / 100.0)
            kk1, kk2, kk3 = st.columns(3)
            kk1.metric("Expected annual return", f"{kpi['ret']:.2%}" if np.isfinite(kpi["ret"]) else "n/a")
            kk2.metric("Expected annual volatility", f"{kpi['vol']:.2%}" if np.isfinite(kpi["vol"]) else "n/a")
            kk3.metric("Sharpe", f"{kpi['sharpe']:.2f}" if np.isfinite(kpi["sharpe"]) else "n/a")

            nav = _portfolio_nav(sub_returns, final_weights, start_value=100.0)
            if not nav.empty:
                bench = prices[assets[0]].dropna()
                bench_norm = (bench / bench.iloc[0]) * 100.0
                compare = pd.concat([nav.rename("Portfolio"), bench_norm.rename(assets[0])], axis=1).dropna()

                fig_nav = px.line(compare.reset_index().rename(columns={compare.index.name or "index": "Date"}), x="Date", y=compare.columns)
                fig_nav.update_layout(title="Historical NAV (base 100)", margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_nav, use_container_width=True)

                dd_df = (nav / nav.cummax() - 1.0).rename("Drawdown").reset_index()
                dd_df = dd_df.rename(columns={dd_df.columns[0]: "Date"})
                fig_dd = px.area(dd_df, x="Date", y="Drawdown")
                fig_dd.update_layout(title="Portfolio drawdown", margin=dict(l=10, r=10, t=40, b=10))
                fig_dd.update_yaxes(tickformat=".1%")
                st.plotly_chart(fig_dd, use_container_width=True)

            with st.expander("Efficient frontier (selected assets)"):
                frontier_points = st.slider("Number of points", min_value=10, max_value=60, value=25)
                fr = frontier_from_prices(
                    prices=sub_prices,
                    rf=rf_pct / 100.0,
                    points=int(frontier_points),
                    shorting=False,
                    max_weight=0.60,
                )
                st.plotly_chart(frontier_chart(fr), use_container_width=True)

            with st.expander("Monte Carlo (optional)"):
                mc_h = st.slider("Horizon (days)", min_value=21, max_value=756, value=252, step=21)
                mc_n = st.slider("Simulations", min_value=100, max_value=1200, value=300, step=100)
                if st.button("Run Monte Carlo simulation", use_container_width=True):
                    paths = simulate_portfolio_paths(
                        returns=sub_returns,
                        weights=final_weights,
                        horizon_days=int(mc_h),
                        n_sims=int(mc_n),
                        initial_value=100.0,
                        random_seed=42,
                    )
                    fig_mc = px.line(paths, x=paths.index, y=paths.columns)
                    fig_mc.update_traces(line=dict(width=1), opacity=0.15, hoverinfo="skip", showlegend=False)
                    fig_mc.update_layout(title="Monte Carlo paths", xaxis_title="Day", yaxis_title="Value")
                    st.plotly_chart(fig_mc, use_container_width=True)
                    end_vals = paths.iloc[-1]
                    p10, p50, p90 = np.percentile(end_vals, [10, 50, 90])
                    p1, p2, p3 = st.columns(3)
                    p1.metric("P10", f"{p10:.2f}")
                    p2.metric("Median", f"{p50:.2f}")
                    p3.metric("P90", f"{p90:.2f}")

            st.session_state["latest_portfolio_weights"] = final_weights.to_dict()

with tab_export:
    st.subheader("Export")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Prices CSV",
            data=prices.to_csv().encode("utf-8"),
            file_name="prices.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Returns CSV",
            data=returns.to_csv().encode("utf-8"),
            file_name="returns.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Ticker summary CSV",
            data=summary.to_csv().encode("utf-8"),
            file_name="summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with c2:
        st.download_button(
            "Correlation CSV",
            data=corr.to_csv().encode("utf-8"),
            file_name="correlation.csv",
            mime="text/csv",
            use_container_width=True,
        )

        latest_weights = st.session_state.get("latest_portfolio_weights")
        if latest_weights:
            w_df = pd.Series(latest_weights, name="weight").to_frame()
            st.download_button(
                "Portfolio weights CSV",
                data=w_df.to_csv().encode("utf-8"),
                file_name="portfolio_weights.csv",
                mime="text/csv",
                use_container_width=True,
            )

        try:
            html = build_html_report(prices=prices, summary=summary, corr=corr)
            payload = html.getvalue().encode("utf-8") if isinstance(html, io.StringIO) else str(html).encode("utf-8")
            st.download_button(
                "HTML report",
                data=payload,
                file_name="quant_insight_report.html",
                mime="text/html",
                use_container_width=True,
            )
        except Exception as err:
            st.warning(f"HTML report is unavailable: {err}")
