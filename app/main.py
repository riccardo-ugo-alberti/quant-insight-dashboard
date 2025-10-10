# app/main.py
from __future__ import annotations
import sys, json, io
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from src.data_loader import fetch_prices
from src.analytics import (
    to_returns, to_cum_returns, rolling_vol,
    corr_matrix, rolling_corr, summary_table, order_corr_matrix,
    top_corr_pairs, rolling_beta
)
from src.visuals import (
    price_lines, cumret_lines, rolling_vol_lines, beta_lines,
    corr_heatmap, rolling_corr_line, sharpe_vol_scatter,
    kpi_card, bar_annual_return, bar_annual_vol,
    style_summary_table, style_corr_pairs_table, style_prices_preview
)
from src.meta import get_sectors
from src.report import build_html_report

st.set_page_config(page_title="Quant Insight Dashboard", page_icon=None, layout="wide")
st.title("Quant Insight Dashboard")
st.caption("Performance, risk and correlation across equities/ETFs with interpretable, interactive charts.")
st.markdown("---")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Controls")

    # FORM — only widgets that participate in submit
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
            tickers = list(dict.fromkeys(tickers))  # deduplicate

        st.markdown("---")
        st.markdown("**Period**")
        start_date, end_date = st.date_input(
            "Date range", value=(pd.to_datetime("2020-01-01"), pd.to_datetime("today"))
        )

        st.markdown("---")
        st.markdown("**Parameters**")
        window = st.slider("Rolling window (days)", 20, 252, 90, 1,  # <-- default 90
                           help="Used to compute rolling volatility/beta/correlation; plots still span the whole period.")
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

    # OUTSIDE the form: save/load
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

@st.cache_data(show_spinner=False)
def _fetch_cached(tickers_tuple, start_str, end_str):
    prices = fetch_prices(list(tickers_tuple), start=start_str)
    return prices.loc[(prices.index >= start_str) & (prices.index <= end_str)]

start = start_date.strftime("%Y-%m-%d")
end = end_date.strftime("%Y-%m-%d")

# ---------------- Main ----------------
if not run:
    st.info("Adjust the controls in the sidebar and click **Run Analysis**.")
    st.stop()

if not tickers:
    st.warning("Please select at least one ticker.")
    st.stop()

try:
    with st.spinner("Downloading market data..."):
        prices = _fetch_cached(tuple(tickers), start, end)

    rets = to_returns(prices)
    cum = to_cum_returns(rets)
    vol = rolling_vol(rets, window=window)
    cm_raw = corr_matrix(rets)
    cm = order_corr_matrix(cm_raw)
    summary = summary_table(rets, rf=rf)
    sectors = get_sectors(list(prices.columns)) if color_sectors else {}

    # ---- Snapshot (collapsed by default) ----
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
                            use_container_width=True)
            st.caption("Cumulative total return indexed to 100 at the start date.")

    # ---- Tabs ----
    st.markdown("### Sections")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Prices", "Performance", "Volatility", "Correlation", "Data / Export"])

    with tab1:
        st.plotly_chart(price_lines(prices, height=520), use_container_width=True)
        st.caption("Adjusted close prices for each selected asset across the chosen period.")

    with tab2:
        st.plotly_chart(cumret_lines(cum, height=560), use_container_width=True)
        st.caption("Cumulative return per asset (base=100), allowing a clean relative performance comparison.")

        st.plotly_chart(sharpe_vol_scatter(summary, sectors, height=560, trendline=True), use_container_width=True)
        st.caption("Annualized return vs annualized volatility. The dashed line is a proper OLS regression fitted on all points.")

        b1, b2 = st.columns([1, 1])
        with b1:
            st.plotly_chart(bar_annual_return(summary), use_container_width=True)
            st.caption("Ranking by annualized return since the beginning of the period.")
        with b2:
            st.plotly_chart(bar_annual_vol(summary), use_container_width=True)
            st.caption("Ranking by annualized volatility (risk).")

        st.markdown("Summary table")
        st.dataframe(style_summary_table(summary), use_container_width=True)

    with tab3:
        st.plotly_chart(rolling_vol_lines(vol, window, height=520), use_container_width=True)
        st.caption("Rolling annualized volatility computed with the selected window; the plot spans the full date range.")
        try:
            betas = rolling_beta(rets, benchmark=benchmark, window=window)
            st.plotly_chart(beta_lines(betas, benchmark, window, height=500), use_container_width=True)
            st.caption(f"Rolling CAPM beta vs {benchmark} (window = {window} days).")
        except Exception as e:
            st.info(f"Beta not available: {e}")

    with tab4:
        st.markdown("Correlation matrix (daily returns)")
        st.plotly_chart(corr_heatmap(cm, height=560), use_container_width=True)
        st.caption("Pearson correlation of daily log-returns—ordered to reveal clusters. Blue = positive, red = negative (classic vivid palette).")

        pos, neg = top_corr_pairs(cm_raw, k=5)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("Top positive pairs")
            st.dataframe(style_corr_pairs_table(pos), use_container_width=True)
        with c2:
            st.markdown("Top negative pairs")
            st.dataframe(style_corr_pairs_table(neg), use_container_width=True)

        if pair_choice := pair_choice if (pair_choice := pair_choice) != "— none —" else None:
            a, b = [p.strip() for p in pair_choice.split("—")]
            if a in rets.columns and b in rets.columns and a != b:
                rc = rolling_corr(rets, a, b, window=window)
                st.plotly_chart(rolling_corr_line(rc, a, b, window, height=480), use_container_width=True)
                st.caption(f"Time-varying correlation between {a} and {b} using a {window}-day rolling window.")
            else:
                st.info("Please choose two distinct tickers present in the dataset.")

    with tab5:
        st.subheader("Preview (last rows)")
        st.dataframe(style_prices_preview(prices), use_container_width=True)
        st.caption("Tail of the cleaned price table used in the analysis.")

        buf = io.BytesIO(); prices.to_csv(buf); buf.seek(0)
        st.download_button("Download prices (CSV)", data=buf, file_name="prices.csv", mime="text/csv")

        figs = {
            "Prices": price_lines(prices),
            "Cumulative Returns": cumret_lines(cum),
            "Volatility": rolling_vol_lines(vol, window),
            "Correlation": corr_heatmap(cm),
        }
        html = build_html_report(prices, summary, figs)
        st.download_button("Download HTML report",
                           data=html.encode("utf-8"),
                           file_name="quant_insight_report.html",
                           mime="text/html")

except Exception as e:
    st.error(f"Error: {e}")
