from __future__ import annotations

# === Bootstrap path per importare "src" sia in locale sia su Streamlit Cloud ===
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

# ==== Compat helpers ==========================================================
def _try_compute_frontier(px_df, rf, n_points, allow_short=None, max_weight=None):
    """
    Prova a chiamare compute_frontier con varie firme note.
    Ritorna il DataFrame della frontiera (oppure solleva in caso di errore reale).
    """
    from src.optimizer import compute_frontier
    from src.analytics import to_returns

    # 1) kw classici
    try:
        return compute_frontier(
            prices=px_df, rf=rf, n_points=n_points,
            **({} if allow_short is None else {"allow_short": allow_short}),
            **({} if max_weight is None else {"max_weight": max_weight}),
        )
    except TypeError:
        pass
    # 2) solo posizionali
    try:
        return compute_frontier(px_df, rf, n_points)
    except TypeError:
        pass
    # 3) vuole i returns
    try:
        _rets = to_returns(px_df)
        return compute_frontier(returns=_rets, rf=rf, n_points=n_points)
    except TypeError:
        pass
    # 4) fallback: compute_frontier(returns)
    _rets = to_returns(px_df)
    return compute_frontier(_rets)


def _try_build_html_report(prices, summary, corr_df):
    """
    Prova vari nomi-argomento per build_html_report e ritorna bytes da scaricare.
    Accetta ritorni di tipo str o io.StringIO.
    """
    from io import StringIO
    from src.report import build_html_report

    # cm=
    try:
        res = build_html_report(prices=prices, summary=summary, cm=corr_df)
        return (res.getvalue() if isinstance(res, StringIO) else (res if isinstance(res, str) else str(res))).encode("utf-8")
    except TypeError:
        pass
    # corr=
    try:
        res = build_html_report(prices=prices, summary=summary, corr=corr_df)
        return (res.getvalue() if isinstance(res, StringIO) else (res if isinstance(res, str) else str(res))).encode("utf-8")
    except TypeError:
        pass
    # correlation=
    res = build_html_report(prices=prices, summary=summary, correlation=corr_df)
    return (res.getvalue() if isinstance(res, StringIO) else (res if isinstance(res, str) else str(res))).encode("utf-8")
# ==============================================================================

# --- Helpers per normalizzare gli output degli optimizer e calcolare KPI ---
def _normalize_frontier(obj) -> pd.DataFrame:
    """
    Accetta vari formati (DataFrame diretto, tuple, dict) e restituisce
    un DataFrame con colonne ['Return','Vol','Sharpe'] se possibile.
    """
    if obj is None:
        return pd.DataFrame(columns=["Return", "Vol", "Sharpe"])

    # DataFrame già pronto
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    # Tuple: spesso (weights, frontier_df) o (kpi, frontier_df)
    elif isinstance(obj, tuple) and len(obj) >= 2 and isinstance(obj[1], pd.DataFrame):
        df = obj[1].copy()
    # Dict: prova chiavi comuni
    elif isinstance(obj, dict):
        if "frontier" in obj and isinstance(obj["frontier"], pd.DataFrame):
            df = obj["frontier"].copy()
        elif "df" in obj and isinstance(obj["df"], pd.DataFrame):
            df = obj["df"].copy()
        else:
            # se l’unico DataFrame è in qualche valore
            df_vals = [v for v in obj.values() if isinstance(v, pd.DataFrame)]
            df = df_vals[0].copy() if df_vals else pd.DataFrame()
    else:
        df = pd.DataFrame()

    # rinomina colonne in modo robusto
    cols = {c.lower(): c for c in df.columns}
    def _pick(*names):
        for n in names:
            if n in df.columns:
                return n
            if n.lower() in cols:
                return cols[n.lower()]
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
    """Normalizza vari formati (dict/tuple/Series/DF) in una Series di pesi indicizzata per ticker."""
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
                raise ValueError("Impossibile inferire i pesi dal DataFrame (nessuna colonna unica numerica).")
        s.index = s.index.astype(str)
        s = s / s.sum() if s.sum() != 0 else s
        return s
    raise ValueError("Formato pesi non riconosciuto dall'optimizer.")


def _mv_kpi_from_weights(px_df, weights_series, rf_annual):
    """Return, Vol, Sharpe annualizzati dai pesi (rf_annual in frazione)."""
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

# ===== Import moduli applicativi ==============================================
from src.data_loader import fetch_prices
from src.analytics   import compute_summary, to_returns, rolling_vol
from src.optimizer   import optimize_portfolio, optimize_cvar
from src.visuals     import (
    prices_chart, perf_cum_chart, rr_scatter, vol_chart, corr_heatmap,
    frontier_chart, weights_donut, weights_pie,
)
# ==============================================================================

# --------------------------------------------------------------------------
# Page setup (no emoji icon)
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Quant Insight Dashboard",
    page_icon=None,
    layout="wide",
)

# --------------------------------------------------------------------------
# Sidebar / Controls
# --------------------------------------------------------------------------
st.sidebar.header("Parameters")

# (a) tickers & date range
default_tickers = "AAPL, MSFT, SPY, NVDA"
tickers_str = st.sidebar.text_input(
    "Tickers (comma-separated) Yahoo Finance",
    value=default_tickers,
    key="tickers_str",
)
tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

date_range = st.sidebar.date_input(
    "Date range",
    value=(dt.date.today() - dt.timedelta(days=365*5), dt.date.today()),
)

# (b) risk free & rolling window (default 90)
rf = st.sidebar.number_input("Risk-free (annual, %)", min_value=0.0, value=2.00, step=0.05)
window = st.sidebar.slider("Rolling window (days)", min_value=30, max_value=252, value=90, step=1)

# (c) optional: save/load config (solo JSON)
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

# precompute derived series
ret = to_returns(px_df)                         # daily returns
summary = compute_summary(px_df, rf=rf/100.0)   # annualized
cm = ret.corr().astype(float)                   # correlation matrix (robusta)
vol_roll = rolling_vol(ret, window=window)      # rolling annualized vol

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_prices, tab_perf, tab_vol, tab_corr, tab_opt, tab_export = st.tabs(
    ["Prices", "Performance", "Volatility", "Correlation", "Optimizer", "Data / Export"]
)

# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------
with tab_prices:
    st.subheader("Adjusted Prices")
    # Grafico SOPRA la tabella
    st.plotly_chart(prices_chart(px_df), use_container_width=True)
    st.dataframe(px_df.tail(), use_container_width=True)

# --------------------------------------------------------------------------
# Performance
# --------------------------------------------------------------------------
with tab_perf:
    st.subheader("Cumulative Returns")
    st.plotly_chart(perf_cum_chart(px_df), use_container_width=True)

    st.subheader("Risk vs Return (annualized)")
    st.plotly_chart(rr_scatter(summary), use_container_width=True)

    st.markdown("**Summary (annualized)**")
    st.dataframe(
        summary.style.format({"Return": "{:.2%}", "Vol": "{:.2%}", "Sharpe": "{:.2f}"}),
        use_container_width=True,
    )

# --------------------------------------------------------------------------
# Volatility
# --------------------------------------------------------------------------
with tab_vol:
    st.subheader(f"Rolling Volatility (window = {window} days, annualized)")
    st.plotly_chart(vol_chart(vol_roll), use_container_width=True)

# --------------------------------------------------------------------------
# Correlation
# --------------------------------------------------------------------------
with tab_corr:
    st.subheader("Correlation matrix (ordered)")
    st.plotly_chart(corr_heatmap(cm), use_container_width=True)

    st.markdown("**Pairwise rolling correlation**")
    pairs = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            pairs.append(f"{tickers[i]} — {tickers[j]}")
    pair_choice = st.selectbox("Select pair", options=pairs)
    if pair_choice:
        a, b = [t.strip() for t in pair_choice.split("—")]
        # Rolling correlation diretta (più stabile)
        r_pair = ret[a].rolling(window).corr(ret[b]).dropna().to_frame(name=f"{a}-{b}")
        fig_pair = px.line(_with_date(r_pair), x="Date", y=f"{a}-{b}", template="plotly_dark")
        fig_pair.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig_pair, use_container_width=True)

# --------------------------------------------------------------------------
# Optimizer
# --------------------------------------------------------------------------
with tab_opt:
    st.subheader("Optimizer")

    # Controlli
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    allow_short = c1.checkbox("Allow shorting", value=False)
    max_cap = c2.slider("Max weight cap", min_value=0.05, max_value=1.00, value=0.30, step=0.05)
    frontier_points = c3.slider("Frontier points", min_value=5, max_value=50, value=25, step=1)
    show_frontier = c4.checkbox("Show frontier", value=True)

    # Modalità MV come nelle versioni che ti piacevano
    mode = st.selectbox(
        "Optimization mode",
        options=["max_sharpe", "min_vol", "target_return"],
        index=0,
        help="Choose the mean-variance objective.",
    )
    target_ret = None
    if mode == "target_return":
        target_ret = st.slider("Target annual return (%)", 0.0, 40.0, 10.0, 0.5) / 100.0

    # ===== Mean-Variance =====
    try:
        # PASSIAMO davvero i vincoli allo scheduler (prima erano commentati)
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

        # KPI (se non forniti, li ricalcoliamo)
        if isinstance(mv_out, dict) and all(k in mv_out for k in ("return", "vol", "sharpe")):
            mv_kpi = {"return_": mv_out["return"], "vol": mv_out["vol"], "sharpe": mv_out["sharpe"]}
        else:
            mv_kpi = _mv_kpi_from_weights(px_df, mv_weights, rf_annual=rf/100.0)

        k1, k2, k3 = st.columns(3)
        k1.metric("Return (MV)", f"{mv_kpi['return_']:.2%}")
        k2.metric("Volatility (MV)", f"{mv_kpi['vol']:.2%}")
        k3.metric("Sharpe (MV)", f"{mv_kpi['sharpe']:.2f}")

        # ===== Frontier =====
        if show_frontier:
            fr_raw = _try_compute_frontier(
                px_df, rf=rf/100.0, n_points=frontier_points,
                allow_short=allow_short, max_weight=max_cap
            )
            fr = _normalize_frontier(fr_raw)
            st.plotly_chart(frontier_chart(fr), use_container_width=True)
        else:
            st.caption("Frontier hidden.")

    except Exception as e:
        st.warning(f"MV optimization error: {e}")

    st.markdown("---")
    st.subheader("CVaR Optimizer (Expected Shortfall)")
    alpha = st.slider("Confidence (1-α)", min_value=0.80, max_value=0.99, value=0.95, step=0.01)

    # ===== CVaR =====
    try:
        cv_kwargs = dict(
            prices=px_df,
            alpha=alpha,
            allow_short=allow_short,   # se il tuo optimizer li ignora, non fa danni
            max_weight=max_cap,
        )
        cv_out = optimize_cvar(**cv_kwargs)
        cv_weights = _as_weights(cv_out)
        st.plotly_chart(
            weights_donut(cv_weights, title="Portfolio Weights (CVaR)"),
            use_container_width=True
        )

        # Se l'optimizer fornisce una curva rischio/rendimento la mostriamo
        if isinstance(cv_out, dict) and "curve" in cv_out and isinstance(cv_out["curve"], pd.DataFrame):
            st.plotly_chart(px.line(cv_out["curve"], x="Vol", y="Return"), use_container_width=True)

    except NotImplementedError:
        st.info("CVaR optimization not available in this build.")
    except Exception as e:
        st.warning(f"CVaR optimizer error: {e}")

# --------------------------------------------------------------------------
# Data / Export
# --------------------------------------------------------------------------
with tab_export:
    st.subheader("Data / Export")
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
