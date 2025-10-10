# src/visuals.py
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ---------- Tabelle sobrie (grigi/bordi) ----------
TABLE_STYLE = dict(
    header=dict(fill_color="#f2f2f2", align="center"),
    cells=dict(align="right"),
)

def style_table(df: pd.DataFrame, title: str | None = None) -> go.Figure:
    fig = go.Figure(data=[go.Table(
        header=dict(values=["<b>"+c+"</b>" for c in [""]+df.columns.tolist()],
                    line_color="#e5e7eb", fill_color="#f8fafc", align="left"),
        cells=dict(values=[[*df.index.tolist()], *[df[c].tolist() for c in df.columns]],
                   line_color="#e5e7eb", align="right")
    )])
    if title:
        fig.update_layout(title=title, margin=dict(l=0, r=0, t=40, b=0))
    else:
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig

# ---------- Monte Carlo scatter ----------
def mc_scatter(sim_df: pd.DataFrame, opt_point: dict | None = None) -> go.Figure:
    fig = px.scatter(sim_df, x="sigma", y="mu", color="sharpe",
                     labels={"sigma":"Annualized Vol", "mu":"Annualized Return", "color":"Sharpe"},
                     opacity=0.6)
    if opt_point is not None:
        fig.add_trace(go.Scatter(x=[opt_point["sigma"]], y=[opt_point["mu"]],
                                 mode="markers", marker=dict(size=12, symbol="x", line=dict(width=2)),
                                 name="Optimal"))
    fig.update_layout(margin=dict(l=0, r=0, t=20, b=0))
    return fig

# ---------- Factor bar ----------
def factor_bars(ff_res: pd.DataFrame, five: bool = True) -> go.Figure:
    fac = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"] if five else ["Mkt-RF", "SMB", "HML"]
    bars = []
    for col in fac:
        if col in ff_res.columns:
            bars.append(go.Bar(name=col, x=ff_res.index, y=ff_res[col]))
    fig = go.Figure(data=bars)
    fig.update_layout(barmode="group", xaxis_title="Ticker", yaxis_title="Loading",
                      margin=dict(l=0, r=0, t=30, b=0), legend_title="Factor")
    return fig

# ---------- Macro lines ----------
def macro_lines(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for c in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[c], name=c, mode="lines"))
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), xaxis_title="", yaxis_title="")
    return fig
