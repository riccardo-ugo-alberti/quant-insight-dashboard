from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------
# Theme & small utils (coerenti con UI scura)
# ---------------------------------------------------------------------
_DEF_TEMPLATE = "plotly_dark"
_AXIS_FONT = dict(size=12)
_TITLE_FONT = dict(size=18, color="#E8E8E8")


def _fmt_pct(x: float) -> str:
    try:
        return f"{x:.2%}"
    except Exception:
        return ""


def _with_date(df: pd.DataFrame) -> pd.DataFrame:
    """Ritorna un df con prima colonna 'Date' (per Plotly)."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Date"])
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index)
        except Exception:
            pass
    out = out.reset_index()
    first_col = out.columns[0]
    if first_col != "Date":
        out = out.rename(columns={first_col: "Date"})
    return out


# ---------------------------------------------------------------------
# 1) Prices (line chart)
# ---------------------------------------------------------------------
def prices_chart(prices: pd.DataFrame, title: str = "Adjusted prices") -> go.Figure:
    df = prices.sort_index().reset_index().rename(columns={"index": "Date"})
    xcol = df.columns[0]  # Date
    fig = px.line(df, x=xcol, y=prices.columns, template=_DEF_TEMPLATE)
    fig.update_layout(
        title=dict(text=title, font=_TITLE_FONT),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(title="", tickfont=_AXIS_FONT)
    fig.update_yaxes(title="", tickfont=_AXIS_FONT)
    return fig


# ---------------------------------------------------------------------
# 2) Volatility (rolling annualized) - con leggera EMA di smoothing
# ---------------------------------------------------------------------
def vol_chart(rolling_vol: pd.DataFrame, title: str = "Rolling annualized volatility") -> go.Figure:
    # smoothing leggero (opzionale)
    try:
        smoothed = rolling_vol.astype(float).ewm(span=5, adjust=False).mean()
    except Exception:
        smoothed = rolling_vol

    df = smoothed.sort_index().reset_index().rename(columns={"index": "Date"})
    xcol = df.columns[0]

    # linee per tutte le colonne (dati in DECIMALI: 0.25 = 25%)
    fig = px.line(df, x=xcol, y=smoothed.columns, template=_DEF_TEMPLATE)

    # hover in percento, asse Y in percento (nessun *100 sui dati!)
    fig.update_traces(mode="lines", hovertemplate="%{y:.2%}<extra>%{fullData.name}</extra>")
    fig.update_layout(
        title=dict(text=title, font=_TITLE_FONT),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(title="", tickfont=_AXIS_FONT)
    fig.update_yaxes(title="", tickfont=_AXIS_FONT, tickformat=".0%")  # << percento via formattazione
    return fig

# ---------------------------------------------------------------------
# 3) Risk vs Return scatter (df: ['Ticker','Return','Vol']) + OLS neutra
# ---------------------------------------------------------------------
def rr_scatter(df_rr: pd.DataFrame, show_ols: bool = True, title: str = "Risk vs Return") -> go.Figure:
    # Copia e normalizza
    df = df_rr.copy()

    # colonne attese: Return, Vol (+ opzionale Ticker)
    if "Return" not in df.columns or "Vol" not in df.columns:
        fig = go.Figure()
        fig.update_layout(template=_DEF_TEMPLATE,
                          title=dict(text=f"{title} (no data)", font=_TITLE_FONT))
        return fig

    # Etichette
    if "Ticker" in df.columns:
        labels = df["Ticker"].astype(str)
    else:
        labels = df.index.astype(str)

    # Coercizione a numerico + drop dei NaN
    df2 = pd.DataFrame({
        "Return": pd.to_numeric(df["Return"], errors="coerce"),
        "Vol":    pd.to_numeric(df["Vol"], errors="coerce"),
        "Label":  labels,
    }).dropna(subset=["Return", "Vol"]).reset_index(drop=True)

    if df2.empty:
        fig = go.Figure()
        fig.update_layout(template=_DEF_TEMPLATE,
                          title=dict(text=f"{title} (no valid rows)", font=_TITLE_FONT))
        return fig

    df2["Return %"] = df2["Return"] * 100.0
    df2["Vol %"]    = df2["Vol"] * 100.0

    # Scatter
    fig = px.scatter(df2, x="Vol %", y="Return %", text=df2["Label"], template=_DEF_TEMPLATE)
    fig.update_traces(textposition="top center")

    # Regressione OLS (facoltativa, con guardie)
    if show_ols and len(df2) >= 2:
        x = df2["Vol %"].to_numpy(dtype=float)
        y = df2["Return %"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) >= 2:
            try:
                m, c = np.polyfit(x, y, 1)
                x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100)
                y_line = m * x_line + c
                fig.add_trace(
                    go.Scatter(
                        x=x_line, y=y_line, mode="lines", name="OLS fit",
                        line=dict(width=1.5, color="#9AA0A6", dash="dash"),
                        hoverinfo="skip", showlegend=True
                    )
                )
            except Exception:
                # se il fit fallisce, saltiamo senza rompere il grafico
                pass

    fig.update_layout(
        title=dict(text=title, font=_TITLE_FONT),
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
    )
    fig.update_xaxes(title="Volatility (%)", tickfont=_AXIS_FONT)
    fig.update_yaxes(title="Return (%)", tickfont=_AXIS_FONT)
    return fig


# ---------------------------------------------------------------------
# 4) Efficient Frontier (df: ['Return','Vol','Sharpe'])
# ---------------------------------------------------------------------
def frontier_chart(frontier_df: pd.DataFrame, title: str = "Efficient Frontier", mv_point: dict | None = None) -> go.Figure:
    if frontier_df is None or frontier_df.empty:
        fig = go.Figure()
        fig.update_layout(
            template=_DEF_TEMPLATE,
            title=dict(text=f"{title} (no feasible points)", font=_TITLE_FONT),
            margin=dict(l=10, r=10, t=60, b=10),
        )
        return fig

    dfp = frontier_df.copy()
    if "Return %" not in dfp and "Return" in dfp:
        dfp["Return %"] = dfp["Return"] * 100.0
    if "Vol %" not in dfp and "Vol" in dfp:
        dfp["Vol %"] = dfp["Vol"] * 100.0

    fig = px.scatter(
        dfp,
        x="Vol %",
        y="Return %",
        color="Sharpe" if "Sharpe" in dfp.columns else None,
        color_continuous_scale="Plasma",
        template=_DEF_TEMPLATE,
    )
    # linea di collegamento
    fig.add_trace(
        go.Scatter(
            x=dfp["Vol %"],
            y=dfp["Return %"],
            mode="lines",
            line=dict(width=1.5),
            name="Frontier",
            hoverinfo="skip",
        )
    )

    # Punto Max Sharpe opzionale
    if isinstance(mv_point, dict) and all(k in mv_point for k in ("vol", "return")):
        fig.add_trace(
            go.Scatter(
                x=[mv_point["vol"] * 100.0],
                y=[mv_point["return"] * 100.0],
                mode="markers+text",
                text=["Max Sharpe"],
                textposition="top center",
                marker=dict(size=10, color="#00E5FF", line=dict(color="white", width=1)),
                name="Max Sharpe",
            )
        )

    fig.update_layout(
        title=dict(text=title, font=_TITLE_FONT),
        coloraxis_colorbar=dict(title="Sharpe"),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(title="Volatility (%)", tickfont=_AXIS_FONT)
    fig.update_yaxes(title="Return (%)", tickfont=_AXIS_FONT)
    return fig


# ---------------------------------------------------------------------
# 5) Portfolio weights (donut)
# ---------------------------------------------------------------------
def weights_donut(weights: pd.Series, title: str = "Portfolio Weights") -> go.Figure:
    s = weights.copy().sort_values(ascending=False)
    fig = go.Figure(
        data=[go.Pie(labels=s.index, values=(s.values * 100.0), hole=0.55, textinfo="percent+label")]
    )
    fig.update_layout(
        template=_DEF_TEMPLATE,
        title=dict(text=title, font=_TITLE_FONT),
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
    )
    return fig


# ---------------------------------------------------------------------
# 6) Correlation heatmap (con valori numerici ben leggibili)
# ---------------------------------------------------------------------
def corr_heatmap(cm: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    z = cm.values
    text = np.vectorize(lambda v: f"{v:.3f}")(z)  # 3 decimali

    colorscale = [
        [0.0, "#7A1E1E"],  # rosso scuro
        [0.5, "#222222"],  # neutro
        [1.0, "#1E90FF"],  # blu
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=z, x=cm.columns, y=cm.index, zmin=-1, zmax=1, colorscale=colorscale, colorbar=dict(title="ρ")
        )
    )
    for i, r in enumerate(cm.index):
        for j, c in enumerate(cm.columns):
            fig.add_annotation(x=c, y=r, text=text[i, j], showarrow=False, font=dict(color="white", size=11))
    fig.update_layout(
        template=_DEF_TEMPLATE,
        title=dict(text=title, font=_TITLE_FONT),
        xaxis=dict(tickfont=_AXIS_FONT),
        yaxis=dict(tickfont=_AXIS_FONT, autorange="reversed"),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


# ---------------------------------------------------------------------
# 7) Cumulative performance (normalized)
# ---------------------------------------------------------------------
def perf_cum_chart(prices: pd.DataFrame, title: str = "Cumulative Returns (normalized)") -> go.Figure:
    base = prices / prices.iloc[0]
    df = base.reset_index().rename(columns={"index": "Date"})
    xcol = df.columns[0]
    fig = px.line(df, x=xcol, y=prices.columns, template=_DEF_TEMPLATE)
    fig.update_layout(
        title=dict(text=title, font=_TITLE_FONT),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(title="", tickfont=_AXIS_FONT)
    fig.update_yaxes(title="", tickfont=_AXIS_FONT)
    return fig


# ---------------------------------------------------------------------
# 8) Pie alternativo robusto (accetta Series o DataFrame)
# ---------------------------------------------------------------------
def weights_pie(weights, title="Portfolio Weights"):
    if isinstance(weights, pd.Series):
        names = weights.index.astype(str)
        values = weights.values
        fig = px.pie(names=names, values=values, hole=0.55, title=title, template=_DEF_TEMPLATE)
    else:
        df = weights.copy()
        if "Ticker" not in df.columns:
            if "ticker" in df.columns:
                df = df.rename(columns={"ticker": "Ticker"})
            elif df.index.name:
                df = df.reset_index().rename(columns={df.columns[0]: "Ticker"})
            else:
                df = df.reset_index().rename(columns={"index": "Ticker"})
        if "weight" not in df.columns:
            num_cols = df.select_dtypes("number").columns.tolist()
            if len(num_cols) == 1:
                df = df.rename(columns={num_cols[0]: "weight"})
        fig = px.pie(df, names="Ticker", values="weight", hole=0.55, title=title, template=_DEF_TEMPLATE)

    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0))
    return fig


# =========================
# NUOVI GRAFICI BACKTEST
# =========================

# 9) NAV time series (output: out['nav'] dal backtest)
def nav_chart(nav_df: pd.DataFrame, title: str = "Backtest NAV") -> go.Figure:
    if nav_df is None or nav_df.empty:
        fig = go.Figure()
        fig.update_layout(template=_DEF_TEMPLATE, title=dict(text=f"{title} (no data)", font=_TITLE_FONT))
        return fig
    df = _with_date(nav_df)
    ycol = "nav" if "nav" in df.columns else df.columns[1]
    fig = px.line(df, x="Date", y=ycol, template=_DEF_TEMPLATE)
    fig.update_layout(
        title=dict(text=title, font=_TITLE_FONT),
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
    )
    fig.update_xaxes(title="", tickfont=_AXIS_FONT)
    fig.update_yaxes(title="NAV", tickfont=_AXIS_FONT)
    return fig


# 10) Turnover per rebalance date (out['turnover'])
def turnover_bar(turn_df: pd.DataFrame, title: str = "Turnover per rebalance") -> go.Figure:
    if turn_df is None or turn_df.empty:
        fig = go.Figure()
        fig.update_layout(template=_DEF_TEMPLATE, title=dict(text=f"{title} (no trades)", font=_TITLE_FONT))
        return fig
    df = _with_date(turn_df)
    ycol = "turnover" if "turnover" in df.columns else df.columns[1]
    fig = px.bar(df, x="Date", y=ycol, template=_DEF_TEMPLATE)
    fig.update_traces(hovertemplate="%{y:.2f}<extra></extra>")
    fig.update_layout(
        title=dict(text=title, font=_TITLE_FONT),
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(tickfont=_AXIS_FONT),
        yaxis=dict(tickfont=_AXIS_FONT, title="Turnover (|Δw| sum)"),
    )
    return fig


# 11) Transaction costs (out['costs'])
def costs_bar(cost_df: pd.DataFrame, title: str = "Transaction costs") -> go.Figure:
    if cost_df is None or cost_df.empty:
        fig = go.Figure()
        fig.update_layout(template=_DEF_TEMPLATE, title=dict(text=f"{title} (no costs)", font=_TITLE_FONT))
        return fig
    df = _with_date(cost_df)
    ycol = "cost" if "cost" in df.columns else df.columns[1]
    fig = px.bar(df, x="Date", y=ycol, template=_DEF_TEMPLATE)
    fig.update_traces(hovertemplate="%{y:.6f}<extra></extra>")
    fig.update_layout(
        title=dict(text=title, font=_TITLE_FONT),
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(tickfont=_AXIS_FONT),
        yaxis=dict(tickfont=_AXIS_FONT, title="Cost (currency)"),
    )
    return fig


# 12) Cumulative costs line (utile per visualizzare il cost drag cumulato)
def costs_cum_chart(cost_df: pd.DataFrame, title: str = "Cumulative transaction costs") -> go.Figure:
    if cost_df is None or cost_df.empty:
        fig = go.Figure()
        fig.update_layout(template=_DEF_TEMPLATE, title=dict(text=f"{title} (no costs)", font=_TITLE_FONT))
        return fig
    df = cost_df.copy().sort_index()
    if "cost" not in df.columns:
        first_num = df.select_dtypes("number").columns
        if len(first_num):
            df = df.rename(columns={first_num[0]: "cost"})
        else:
            df["cost"] = 0.0
    cum = df["cost"].cumsum().to_frame("cum_cost")
    dfp = _with_date(cum)
    fig = px.line(dfp, x="Date", y="cum_cost", template=_DEF_TEMPLATE)
    fig.update_layout(
        title=dict(text=title, font=_TITLE_FONT),
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(tickfont=_AXIS_FONT),
        yaxis=dict(tickfont=_AXIS_FONT, title="Cumulative cost"),
    )
    return fig


# 13) Weights over time (stacked area) — input: out['weights']
def weights_area(weights_df: pd.DataFrame, title: str = "Weights over time (stacked)") -> go.Figure:
    if weights_df is None or weights_df.empty:
        fig = go.Figure()
        fig.update_layout(template=_DEF_TEMPLATE, title=dict(text=f"{title} (no data)", font=_TITLE_FONT))
        return fig

    df = _with_date(weights_df)
    # individua colonne numeriche (gli asset)
    num_cols = [c for c in df.columns if c != "Date" and np.issubdtype(df[c].dtype, np.number)]
    if not num_cols:
        fig = go.Figure()
        fig.update_layout(template=_DEF_TEMPLATE, title=dict(text=f"{title} (no numeric columns)", font=_TITLE_FONT))
        return fig

    # long format per area stacked
    long_df = df.melt(id_vars="Date", value_vars=num_cols, var_name="Asset", value_name="Weight")
    long_df["Weight %"] = long_df["Weight"] * 100.0

    fig = px.area(long_df, x="Date", y="Weight %", color="Asset", template=_DEF_TEMPLATE)
    fig.update_layout(
        title=dict(text=title, font=_TITLE_FONT),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(title="", tickfont=_AXIS_FONT)
    fig.update_yaxes(title="Weight (%)", tickfont=_AXIS_FONT)
    return fig

