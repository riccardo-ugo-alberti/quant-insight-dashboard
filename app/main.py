from __future__ import annotations
import sys, io, json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# --------- PATH / IMPORTS ----------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import fetch_prices
from src.analytics import (
    to_returns, to_cum_returns, rolling_vol,
    corr_matrix, rolling_corr, summary_table, order_corr_matrix,
    top_corr_pairs, rolling_beta, rolling_sharpe, rolling_drawdown,
)
from src.visuals import (
    # price & perf
    price_lines, cumret_lines, sharpe_vol_scatter, kpi_card,
    bar_annual_return, bar_annual_vol,
    # risk & beta
    rolling_vol_lines, beta_lines, rolling_sharpe_lines, drawdown_area,
    # correlation
    corr_heatmap, rolling_corr_line, style_corr_pairs_table,
    # tables / styles
    style_summary_table, style_prices_preview,
    # optimizer visuals
    weights_pie, efficient_frontier_plot, equity_curve_plot, weights_area_plot,
)
from src.meta import get_sectors
from src.report import build_html_report
from src.optimizer import (
    optimize_portfolios, efficient_frontier, solve_target_return,
    backtest_rebalance, annualize_returns_and_cov, OptResult,
)

# --------- PAGE CONFIG ----------
st.set_page_config(page_title="Quant Insight Dashboard", layout="wide")
st.title("Quant Insight Dashboard")
st.caption("Performance, risk and correlation across equities/ETFs with interpretable, interactive charts.")
st.markdown("---")

# --------- SIDEBAR CONTROLS ---------
with st.sidebar:
    st.header("Controls")

    with st.form(key="controls"):
        st.markdown("**Universe**")
        presets = {
            "US Tech": ["AAPL", "MSFT", "NVDA", "GOOGL"],
            "Benchmarks": ["SPY", "QQQ", "IWM", "TLT", "GLD"],
            "FAANG+": ["META", "AMZN", "AAPL", "NFLX", "GOOGL", "MSFT"],
        }
        preset = st.selectbox("Preset", ["— none —", *presets.keys()])
        default = presets.get(preset, ["AAPL", "MSFT", "SPY"])
        base_options = sorted(set(sum(presets.values(), [])) | set(default))

        tickers = st.multiselect(
            "Tickers",
            options=base_options,
            default=default,
            help="You can type any Yahoo Finance ticker and press Enter to add it."
        )
        extra = st.text_input("Add tickers (comma-separated)", "")
        if extra.strip():
            tickers += [t.strip().upper() for t in extra.split(",") if t.strip()]
            tickers = list(dict.fromkeys(tickers))  # dedupe

        st.markdown("---")
        st.markdown("**Period**")
        start_date, end_date = st.date_input(
            "Date range", value=(pd.to_datetime("2020-01-01"), pd.to_datetime("today"))
        )

        st.markdown("---")
        st.markdown("**Parameters**")
        window = st.slider("Rolling window (days)", 20, 252, 90, 1)
        rf = st.number_input("Risk-free (annual, %)", min_value=0.0, value=2.0, step=0.1) / 100.0
        benchmark = st.selectbox("Beta benchmark", options=(["SPY"] + [t for t in tickers if t != "SPY"]), index=0)
        color_sectors = st.checkbox("Color Risk-Return by sector (when available)", value=True)

        st.markdown("---")
        st.markdown("**Correlation tools**")
        pair = ["— none —"]
        if len(tickers) >= 2:
            pair += [f"{a} — {b}" for i, a in enumerate(tickers) for b in tickers[i+1:]]
        pair_choice = st.selectbox("Pair for rolling correlation", pair)

        run = st.form_submit_button("Run Analysis", type="primary")

    with st.expander("Save / Load setup"):
        cfg = {"tickers": tickers}
        st.download_button("Download configuration (JSON)",
                           data=json.dumps(cfg).encode(),
                           file_name="qid-config.json")
        uploaded = st.file_uploader("Upload configuration (JSON)", type=["json"], label_visibility="collapsed")
        if uploaded:
            try:
                data = json.load(uploaded)
                if isinstance(data.get("tickers"), list):
                    tickers[:] = [t.strip().upper() for t in data["tickers"] if t.strip()]
                    st.success("Configuration loaded. Check controls and click Run.")
            except Exception as e:
                st.error(f"Invalid config: {e}")

# --------- DATA FETCH (CACHE) ---------
@st.cache_data(show_spinner=False)
def _fetch_cached(tickers_tuple, start_str, end_str):
    prices_all = fetch_prices(list(tickers_tuple), start=start_str)
    return prices_all.loc[(prices_all.index >= start_str) & (prices_all.index <= end_str)]

start = start_date.strftime("%Y-%m-%d")
end = end_date.strftime("%Y-%m-%d")

if not run:
    st.info("Adjust the controls in the sidebar and click **Run Analysis**.")
    st.stop()

if not tickers:
    st.warning("Please select at least one ticker.")
    st.stop()

with st.spinner("Downloading market data..."):
    prices = _fetch_cached(tuple(tickers), start, end)

# --------- CORE METRICS ---------
rets = to_returns(prices)
cum = to_cum_returns(rets)
vol = rolling_vol(rets, window=window)

cm_raw = corr_matrix(rets)
cm = order_corr_matrix(cm_raw)
summary = summary_table(rets, rf=rf)

sectors = get_sectors(list(prices.columns)) if color_sectors else {}

rs = rolling_sharpe(rets, window=window, rf_annual=rf)
dd = rolling_drawdown(rets)

# --------- SNAPSHOT (KPI + CUM) ---------
with st.expander("Snapshot (key metrics and cumulative curve)", expanded=False):
    if not summary.empty:
        top_ret = summary["Ann. Return"].idxmax()
        best_sharpe = summary["Sharpe"].idxmax()
        worst_dd = summary["Max Drawdown"].idxmin()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(**kpi_card(f"Top Return — {top_ret}",
                                 f"{summary.loc[top_ret,'Ann. Return']:.2%}",
                                 f"Vol {summary.loc[top_ret,'Ann. Vol']:.2%}",
                                 summary.loc[top_ret,'Ann. Return']))
        with c2:
            st.metric(**kpi_card(f"Best Sharpe — {best_sharpe}",
                                 f"{summary.loc[best_sharpe,'Sharpe']:.2f}",
                                 f"Vol {summary.loc[best_sharpe,'Ann. Vol']:.2%}",
                                 summary.loc[best_sharpe,'Sharpe']))
        with c3:
            st.metric(**kpi_card(f"Worst MDD — {worst_dd}",
                                 f"{summary.loc[worst_dd,'Max Drawdown']:.2%}",
                                 f"Return {summary.loc[worst_dd,'Ann. Return']:.2%}",
                                 -abs(summary.loc[worst_dd,'Max Drawdown'])))

        st.plotly_chart(cumret_lines(cum, "Cumulative performance (base=100)", height=520),
                        use_container_width=True, key="cum_snapshot")

st.markdown("### Sections")
tab_prices, tab_perf, tab_analytics, tab_opt, tab_vol, tab_corr, tab_data = st.tabs([
    "Prices", "Performance", "Analytics", "Optimizer", "Volatility", "Correlation", "Data / Export"
])

# --------- PRICES ---------
with tab_prices:
    st.plotly_chart(price_lines(prices, height=520), use_container_width=True, key="price_chart")
    st.caption("Adjusted close prices for each selected asset across the chosen period.")

# --------- PERFORMANCE ---------
with tab_perf:
    # grafico grande sopra
    st.plotly_chart(cumret_lines(cum, height=560), use_container_width=True, key="cum_perf")
    st.caption("Cumulative return per asset (base=100).")

    # scatter con regressione (linea sottile)
    st.plotly_chart(sharpe_vol_scatter(summary, sectors, height=560, trendline=True),
                    use_container_width=True, key="sharpe_scatter")
    st.caption("Annualized return vs volatility. The dashed line is an OLS regression line.")

    b1, b2 = st.columns([1, 1])
    with b1:
        st.plotly_chart(bar_annual_return(summary), use_container_width=True, key="bar_ret")
    with b2:
        st.plotly_chart(bar_annual_vol(summary), use_container_width=True, key="bar_vol")

    st.markdown("Summary table")
    st.dataframe(style_summary_table(summary), use_container_width=True)

# --------- ANALYTICS ---------
with tab_analytics:
    st.subheader("Metrics Summary")
    st.dataframe(style_summary_table(summary), use_container_width=True)
    st.markdown("—")

    st.subheader("Rolling Sharpe")
    st.plotly_chart(rolling_sharpe_lines(rs, height=520), use_container_width=True, key="roll_sharpe")
    st.caption(f"Rolling Sharpe with {window}-day window.")

    st.subheader("Drawdown (selected ticker)")
    focus = st.selectbox("Select ticker for drawdown view", options=list(rets.columns))
    st.plotly_chart(drawdown_area(dd, focus, height=420), use_container_width=True, key="drawdown")

# --------- OPTIMIZER ---------
with tab_opt:
    st.subheader("Mean–Variance Optimizer")

    c1, c2, c3 = st.columns(3)
    with c1:
        allow_short = st.checkbox("Allow shorting", value=False)
        w_max = st.slider("Max weight cap", 0.10, 1.00, 0.30, 0.05)
    with c2:
        ef_points = st.slider("Frontier points", 10, 60, 25, 1)
        rebalance = st.selectbox("Rebalance frequency", ["Monthly", "Quarterly"], index=0)
        rb_code = "M" if rebalance == "Monthly" else "Q"
    with c3:
        turnover_limit = st.slider("Turnover limit per rebalance", 0.0, 1.0, 0.5, 0.05)
        tc_bps = st.number_input("Transaction costs (bps)", min_value=0.0, value=5.0, step=1.0)

    sect_on = st.checkbox("Apply sector exposure cap", value=False)
    sector_cap = st.slider("Sector cap (each)", 0.20, 1.00, 0.40, 0.05, disabled=not sect_on)
    sectors_opt = get_sectors(list(prices.columns)) if sect_on else {}

    mode = st.radio("Optimization mode", ["Max Sharpe", "Min Vol", "Target Return"], horizontal=True)
    mu_annual = rets.mean() * 252
    tr_lo, tr_hi = float(np.percentile(mu_annual.values, 10)), float(np.percentile(mu_annual.values, 90))
    target_ret = st.slider("Target annual return",
                           float(tr_lo), float(tr_hi),
                           float(np.clip(mu_annual.mean(), tr_lo, tr_hi)), 0.005,
                           disabled=(mode != "Target Return"))
    st.markdown("---")

    try:
        ef = efficient_frontier(
            rets, rf=rf,
            allow_short=allow_short, w_max=w_max,
            sectors=sectors_opt if sect_on else None,
            sector_cap=sector_cap if sect_on else None,
            points=ef_points
        )
        results = optimize_portfolios(
            rets, rf=rf, allow_short=allow_short, w_max=w_max,
            sectors=sectors_opt if sect_on else None,
            sector_cap=sector_cap if sect_on else None
        )
        res_ms, res_mv = results["max_sharpe"], results["min_vol"]
        res_tr = None
        if mode == "Target Return":
            mu, cov = annualize_returns_and_cov(rets)
            w_tr = solve_target_return(
                mu, cov, target_ret, allow_short, w_max,
                sectors=sectors_opt if sect_on else None,
                sector_cap=sector_cap if sect_on else None
            )
            r_tr = float(w_tr.values @ mu.values)
            v_tr = float(np.sqrt(max(w_tr.values @ cov.values @ w_tr.values, 0.0)))
            res_tr = OptResult(weights=w_tr, ann_return=r_tr, ann_vol=v_tr, sharpe=(r_tr - rf) / (v_tr + 1e-12))

        # KPI tables
        k1, k2, k3 = st.columns(3)
        for res, name, col in zip([res_ms, res_mv, res_tr],
                                  ["Max Sharpe", "Min Vol", "Target Return"], [k1, k2, k3]):
            with col:
                if res:
                    st.markdown(f"**{name}**")
                    st.metric("Return", f"{res.ann_return:.2%}")
                    st.metric("Volatility", f"{res.ann_vol:.2%}")
                    st.metric("Sharpe", f"{res.sharpe:.2f}")
                    st.dataframe(res.weights.rename("Weight").to_frame().style.format("{:.2%}"),
                                 use_container_width=True)

        # Weights pies
        p1, p2, p3 = st.columns(3)
        with p1:
            st.plotly_chart(weights_pie(res_ms.weights), use_container_width=True, key="pie_ms")
        with p2:
            st.plotly_chart(weights_pie(res_mv.weights), use_container_width=True, key="pie_mv")
        with p3:
            if res_tr:
                st.plotly_chart(weights_pie(res_tr.weights), use_container_width=True, key="pie_tr")

        # Efficient frontier
        st.plotly_chart(
            efficient_frontier_plot(
                ef,
                point_ms=(res_ms.ann_vol, res_ms.ann_return),
                point_mv=(res_mv.ann_vol, res_mv.ann_return),
                point_tr=((res_tr.ann_vol, res_tr.ann_return) if res_tr else None)
            ),
            use_container_width=True,
            key="ef_plot"
        )

        # Backtest
        st.markdown("### Backtest (rebalance)")
        bt_mode = {"Max Sharpe": "max_sharpe", "Min Vol": "min_vol", "Target Return": "target"}[mode]
        eq, W = backtest_rebalance(
            prices, rets, rf,
            mode=bt_mode, allow_short=allow_short, w_max=w_max,
            sectors=sectors_opt if sect_on else None,
            sector_cap=sector_cap if sect_on else None,
            rebalance=rb_code,
            target_ret=(target_ret if mode == "Target Return" else None),
            turnover_limit=turnover_limit,
            tc_bps=tc_bps
        )

        daily_eq_ret = eq.pct_change().fillna(0.0)
        cagr = (eq.iloc[-1]) ** (252 / len(eq)) - 1.0
        vol_bt = daily_eq_ret.std() * np.sqrt(252)
        sharpe_bt = (daily_eq_ret.mean() * 252 - rf) / (vol_bt + 1e-12)
        dd_bt = (eq / eq.cummax() - 1.0).min()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CAGR", f"{cagr:.2%}")
        m2.metric("Volatility", f"{vol_bt:.2%}")
        m3.metric("Sharpe", f"{sharpe_bt:.2f}")
        m4.metric("Max Drawdown", f"{dd_bt:.2%}")

        g1, g2 = st.columns([1.2, 1.0])
        with g1:
            st.plotly_chart(equity_curve_plot(eq), use_container_width=True, key="eq_curve")
        with g2:
            st.plotly_chart(weights_area_plot(W), use_container_width=True, key="weights_area")

    except Exception as e:
        st.error(f"Optimizer error: {e}")

# --------- VOLATILITY ---------
with tab_vol:
    st.plotly_chart(rolling_vol_lines(vol, window, height=520),
                    use_container_width=True, key="vol_chart")
    st.caption("Rolling annualized volatility with selected window.")
    try:
        betas = rolling_beta(rets, benchmark=benchmark, window=window)
        st.plotly_chart(beta_lines(betas, benchmark, window, height=500),
                        use_container_width=True, key="beta_chart")
    except Exception as e:
        st.info(f"Beta not available: {e}")

# --------- CORRELATION ---------
with tab_corr:
    st.plotly_chart(corr_heatmap(cm, height=560),
                    use_container_width=True, key="corr_heat")
    st.caption("Correlation heatmap (blue=positive, red=negative).")
    pos, neg = top_corr_pairs(cm_raw, k=5)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("Top positive pairs")
        st.dataframe(style_corr_pairs_table(pos), use_container_width=True)
    with c2:
        st.markdown("Top negative pairs")
        st.dataframe(style_corr_pairs_table(neg), use_container_width=True)

    if (pair_choice and pair_choice != "— none —"):
        a, b = [p.strip() for p in pair_choice.split("—")]
        if a in rets.columns and b in rets.columns:
            rc = rolling_corr(rets, a, b, window=window)
            st.plotly_chart(rolling_corr_line(rc, a, b, window, height=480),
                            use_container_width=True, key="roll_corr")

# --------- DATA / EXPORT ---------
with tab_data:
    st.subheader("Preview (last rows)")
    st.dataframe(style_prices_preview(prices), use_container_width=True)

    # CSV
    buf = io.BytesIO()
    prices.to_csv(buf)
    buf.seek(0)
    st.download_button(
        label="Download CSV",
        data=buf,
        file_name="prices.csv",
        mime="text/csv",
        use_container_width=True,
        key="dl_csv"
    )

    # ---- HTML report download ----
html_bytes = build_html_report(prices, summary, cm).getvalue().encode("utf-8")
st.download_button(
    label="Download HTML Report",
    data=html_bytes,
    file_name="report.html",
    mime="text/html",
    use_container_width=True,
    key="dl_html",
)
