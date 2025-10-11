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


# ---------------------------------------------------------------------
# 1) Prices (line chart)
# ---------------------------------------------------------------------
def prices_chart(prices: pd.DataFrame, title: str = "Adjusted prices") -> go.Figure:
    df = prices.sort_index().reset_index().rename(columns={"index": "Date"})
    xcol = df.columns[0]  # Date (dopo rename sopra)
    fig = px.line(df, x=xcol, y=prices.columns, template=_DEF_TEMPLATE)
    fig.update_layout(
        title=dict(text=title, font=_TITLE_FONT),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),  # legenda sotto
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(title="", tickfont=_AXIS_FONT)
    fig.update_yaxes(title="", tickfont=_AXIS_FONT)
    return fig


# ---------------------------------------------------------------------
# 2) Volatility (rolling annualized) - NO spline (compat con ScatterGL)
#     Applichiamo una lieve levigatura EMA per estetica (solo plotting).
# ---------------------------------------------------------------------
def vol_chart(rolling_vol: pd.DataFrame, title: str = "Rolling annualized volatility") -> go.Figure:
    try:
        smoothed = rolling_vol.ewm(span=5, adjust=False).mean()
    except Exception:
        smoothed = rolling_vol

    df = smoothed.sort_index().reset_index().rename(columns={"index": "Date"})
    xcol = df.columns[0]
    fig = px.line(df, x=xcol, y=smoothed.columns, template=_DEF_TEMPLATE)
    fig.update_traces(mode="lines", hovertemplate="%{y:.2%}<extra>%{fullData.name}</extra>")
    fig.update_layout(
        title=dict(text=title, font=_TITLE_FONT),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(title="", tickfont=_AXIS_FONT)
    fig.update_yaxes(title="", tickfont=_AXIS_FONT)
    return fig


# ---------------------------------------------------------------------
# 3) Risk vs Return scatter (df: ['Ticker','Return','Vol']) + OLS neutra
# ---------------------------------------------------------------------
def rr_scatter(df_rr: pd.DataFrame, show_ols: bool = True, title: str = "Risk vs Return") -> go.Figure:
    df = df_rr.copy()

    # Colonne tolleranti (Ticker può stare nell'indice)
    if "Ticker" in df.columns:
        labels = df["Ticker"].astype(str)
    else:
        labels = df.index.astype(str)
    if "Return %" not in df.columns and "Return" in df.columns:
        df["Return %"] = df["Return"] * 100.0
    if "Vol %" not in df.columns and "Vol" in df.columns:
        df["Vol %"] = df["Vol"] * 100.0

    fig = px.scatter(df, x="Vol %", y="Return %", text=labels, template=_DEF_TEMPLATE)
    fig.update_traces(textposition="top center")

    # Regressione OLS grigio neutro, hover disattivato
    if show_ols and len(df) >= 2 and df["Vol %"].notna().any() and df["Return %"].notna().any():
        x = df["Vol %"].to_numpy()
        y = df["Return %"].to_numpy()
        m, c = np.polyfit(x, y, 1)
        x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        y_line = m * x_line + c
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="OLS fit",
                line=dict(width=1.5, color="#9AA0A6", dash="dash"),  # grigio neutro
                hoverinfo="skip",
                showlegend=True,
            )
        )

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
    text = np.vectorize(lambda v: f"{v:.3f}")(z)  # 3 decimali per evitare troppi 0.00

    colorscale = [
        [0.0, "#7A1E1E"],  # rosso scuro
        [0.5, "#222222"],  # neutro su tema scuro
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
