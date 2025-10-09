# app/main.py
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import io
import pandas as pd
import streamlit as st

from src.data_loader import fetch_prices
from src.analytics import (
    to_returns, to_cum_returns, rolling_vol,
    corr_matrix, rolling_corr, summary_table,
    order_corr_matrix, top_corr_pairs, rolling_beta
)
from src.visuals import (
    price_lines, cumret_lines, rolling_vol_lines,
    corr_heatmap, rolling_corr_line, sharpe_vol_scatter, beta_lines
)
from src.meta import get_sectors
from src.report import build_html_report

st.set_page_config(page_title="Quant Insight Dashboard", page_icon="📈", layout="wide")
st.title("📈 Quant Insight Dashboard")
st.caption("Explore performance, volatility, correlation and risk metrics across assets.")

# ----------------------- SIDEBAR ---------------------------
with st.sidebar:
    st.header("Controls")

    presets = {
        "US Tech": ["AAPL", "MSFT", "NVDA", "GOOGL"],
        "Benchmarks": ["SPY", "QQQ", "IWM", "TLT", "GLD"],
        "FAANG+": ["META", "AMZN", "AAPL", "NFLX", "GOOGL", "MSFT"],
    }
    preset = st.selectbox("Preset (optional)", ["— none —", *presets.keys()])
    default_list = presets.get(preset, ["AAPL", "MSFT", "SPY"])

    tickers = st.multiselect(
        "Tickers", options=sorted(set(sum(presets.values(), [])) | set(default_list)),
        default=default_list, help="Pick 2–15 symbols. Type to add any Yahoo ticker."
    )
    manual = st.text_input("Add tickers (comma-separated)", "", help="e.g. TSLA, META")
    if manual.strip():
        tickers.extend([t.strip().upper() for t in manual.split(",") if t.strip()])
        tickers = list(dict.fromkeys(tickers))

    # Config save/load
    with st.expander("Save / Load setup"):
        cfg = {"tickers": tickers}
        st.download_button("⬇️ Download config (JSON)", data=json.dumps(cfg).encode(), file_name="qid-config.json")
        uploaded = st.file_uploader("Upload config", type=["json"], label_visibility="collapsed")
        if uploaded:
            try:
                data = json.load(uploaded)
                if isinstance(data.get("tickers"), list):
                    tickers = [t.strip().upper() for t in data["tickers"] if t.strip()]
                    st.success("Config loaded. Adjust if needed, then Run Analysis.")
            except Exception as e:
                st.error(f"Invalid config: {e}")

    start_date, end_date = st.date_input("Date range",
        value=(pd.to_datetime("2020-01-01"), pd.to_datetime("today")))

    window = st.slider("Rolling window (days)", min_value=20, max_value=252, value=63, step=1)

    with st.expander("Advanced"):
        rf = st.number_input("Risk-free (annual, %)", min_value=0.0, value=2.0, step=0.1) / 100.0
        use_benchmark = st.checkbox("Overlay SPY as benchmark", value=False)
        benchmark = st.selectbox("Beta benchmark", options=["SPY", *tickers] if tickers else ["SPY"], index=0)
        color_by_sector = st.checkbox("Color scatter by sector (best-effort metadata)", value=True)
        st.caption("RF used for Sharpe; SPY overlay and beta use chosen benchmark.")

    pair_options = []
    if len(tickers) >= 2:
        for i in range(len(tickers)):
            for j in range(i+1, len(tickers)):
                pair_options.append(f"{tickers[i]} — {tickers[j]}")
    pair_choice = st.selectbox("Pairwise rolling correlation", options=["— none —", *pair_options])

    run = st.button("Run Analysis", type="primary")

start = start_date.strftime("%Y-%m-%d")
end = end_date.strftime("%Y-%m-%d")

@st.cache_data(show_spinner=False)
def _fetch_cached(tickers_tuple, start_str, end_str):
    prices = fetch_prices(list(tickers_tuple), start=start_str)
    return prices.loc[(prices.index >= start_str) & (prices.index <= end_str)]

# ----------------------- MAIN ------------------------------
if run and tickers:
    try:
        with st.spinner("Fetching market data..."):
            prices = _fetch_cached(tuple(tickers), start, end)

        rets = to_returns(prices)
        cum = to_cum_returns(rets)
        vol = rolling_vol(rets, window=window)
        cm_raw = corr_matrix(rets)
        cm = order_corr_matrix(cm_raw)
        summary = summary_table(rets, rf=rf)

        sectors = get_sectors(list(prices.columns)) if color_by_sector else {}

        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ["Prices", "Performance", "Volatility", "Correlation", "Beta", "Data / Export"]
        )

        with tab1:
            st.plotly_chart(price_lines(prices), use_container_width=True)

        with tab2:
            # KPI cards
            if not summary.empty:
                top3 = summary.head(3).reset_index().rename(columns={"index": "Ticker"})
                cols = st.columns(min(3, len(top3)))
                for i, (_, row) in enumerate(top3.iterrows()):
                    with cols[i]:
                        st.metric(
                            label=f"{row['Ticker']} — Ann. Return",
                            value=f"{row['Ann. Return']:.2%}",
                            delta=f"Sharpe {row['Sharpe']:.2f}"
                        )

            to_plot = cum.copy()
            if use_benchmark and "SPY" not in to_plot.columns:
                spy = _fetch_cached(tuple(["SPY"]), start, end)
                spy_rets = to_returns(spy)
                spy_cum = to_cum_returns(spy_rets).rename(columns={"SPY": "SPY (benchmark)"})
                to_plot = to_plot.join(spy_cum, how="left")

            c1, c2 = st.columns([2, 1])
            with c1:
                st.plotly_chart(cumret_lines(to_plot), use_container_width=True)
            with c2:
                st.markdown("**Summary metrics**")
                st.dataframe(
                    summary.style.format({
                        "Ann. Return":"{:.2%}","Ann. Vol":"{:.2%}",
                        "Sharpe":"{:.2f}","Max Drawdown":"{:.2%}"
                    }),
                    use_container_width=True, height=420
                )
                st.plotly_chart(sharpe_vol_scatter(summary, sectors if color_by_sector else None),
                                use_container_width=True)

        with tab3:
            st.plotly_chart(rolling_vol_lines(vol), use_container_width=True)

        with tab4:
            st.markdown("**Correlation matrix (daily returns, grouped)**")
            st.plotly_chart(corr_heatmap(cm), use_container_width=True)
            from src.analytics import top_corr_pairs
            pos, neg = top_corr_pairs(cm_raw, k=5)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Top positive pairs**")
                st.dataframe(pos.style.format({"rho":"{:.2f}"}), use_container_width=True, hide_index=True)
            with c2:
                st.markdown("**Top negative pairs**")
                st.dataframe(neg.style.format({"rho":"{:.2f}"}), use_container_width=True, hide_index=True)

            if pair_choice != "— none —":
                a, b = [p.strip() for p in pair_choice.split("—")]
                if a in rets.columns and b in rets.columns and a != b:
                    rc = rolling_corr(rets, a, b, window=window)
                    st.markdown(f"**Rolling correlation** ({window}d) — {a} vs {b}")
                    st.plotly_chart(rolling_corr_line(rc, a, b, window), use_container_width=True)
                else:
                    st.info("Select two different tickers present in the dataset.")
            st.caption("Heatmap shows daily-return correlations. Blue=positive, red=negative. Values in each cell.")

        with tab5:
            try:
                betas = rolling_beta(rets, benchmark=benchmark, window=window)
                st.plotly_chart(beta_lines(betas, benchmark), use_container_width=True)
                st.caption(f"Rolling CAPM beta vs **{benchmark}** (window={window}).")
            except Exception as e:
                st.info(f"Beta not available: {e}")

        with tab6:
            st.subheader("Preview (last rows)")
            st.dataframe(prices.tail(), use_container_width=True)

            # CSV export
            buf = io.BytesIO(); prices.to_csv(buf); buf.seek(0)
            st.download_button("⬇️ Download prices (CSV)", data=buf, file_name="prices.csv", mime="text/csv")

            # HTML report
            st.subheader("One-click HTML report")
            figs = {
                "Prices": price_lines(prices),
                "Cumulative Returns": cumret_lines(cum),
                "Volatility": rolling_vol_lines(vol),
                "Correlation": corr_heatmap(cm),
            }
            html = build_html_report(prices, summary, figs)
            st.download_button("🧾 Download HTML report",
                data=html.encode("utf-8"), file_name="quant_insight_report.html", mime="text/html")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Choose tickers and click **Run Analysis** in the sidebar.")
