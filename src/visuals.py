# src/visuals.py
from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# ============================================================
#  Generic, clean visuals for the dashboard
#  - style_table(df, title)
#  - mc_scatter(sim_df, opt_point)
#  - factor_bars(ff_result, five=True)
#  - macro_lines(df)
# ============================================================

def style_table(df: pd.DataFrame, title: str | None = None) -> go.Figure:
    """
    Render a clean table (Plotly Table) with subtle borders and compact spacing.
    - Index in first column
    - Numeric values displayed as-is (pre-format before calling if needed)
    """
    df = df.copy()
    df.index = df.index.astype(str)

    header_vals = ["<b></b>"] + [f"<b>{c}</b>" for c in df.columns]
    cell_vals = [[*df.index.tolist()], *[df[c].tolist() for c in df.columns]]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=header_vals,
                    fill_color="#f8fafc",
                    line_color="#e5e7eb",
                    align="left",
                    font=dict(size=12),
                    height=30,
                ),
                cells=dict(
                    values=cell_vals,
                    fill_color="white",
                    line_color="#e5e7eb",
                    align="right",
                    font=dict(size=12),
                    height=28,
                ),
                columnwidth=[140] + [120] * len(df.columns),
            )
        ]
    )
    fig.update_layout(
        title=title or None,
        margin=dict(l=0, r=0, t=40 if title else 6, b=0),
    )
    return fig


def mc_scatter(sim_df: pd.DataFrame, opt_point: dict | None = None) -> go.Figure:
    """
    Monte Carlo scatter of portfolios:
      x = sigma (annualized vol), y = mu (annualized return), color = Sharpe.
    Optionally highlight an optimal point with an 'X'.
    Expected columns: ['sigma', 'mu', 'sharpe'].
    """
    base = px.scatter(
        sim_df,
        x="sigma",
        y="mu",
        color="sharpe",
        labels={"sigma": "Annualized Vol", "mu": "Annualized Return", "color": "Sharpe"},
        opacity=0.7,
        render_mode="webgl",
    )
    base.update_traces(marker=dict(size=6))
    base.update_coloraxes(colorbar_title="Sharpe")

    if opt_point is not None and "sigma" in opt_point and "mu" in opt_point:
        base.add_trace(
            go.Scatter(
                x=[opt_point["sigma"]],
                y=[opt_point["mu"]],
                name="Optimal",
                mode="markers",
                marker=dict(symbol="x", size=14, line=dict(width=2)),
            )
        )

    base.update_layout(
        template="plotly_white",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Volatility (σ)",
        yaxis_title="Return (μ)",
    )
    return base


def factor_bars(ff_res: pd.DataFrame, five: bool = True) -> go.Figure:
    """
    Grouped bar chart of factor loadings per ticker.
    ff_res columns expected (typical): alpha, R2, and factors:
      - 5 factors: Mkt-RF, SMB, HML, RMW, CMA
      - 3 factors: Mkt-RF, SMB, HML
    """
    if ff_res is None or ff_res.empty:
        return go.Figure()

    factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"] if five else ["Mkt-RF", "SMB", "HML"]
    factors = [f for f in factors if f in ff_res.columns]

    bars = []
    for fac in factors:
        bars.append(go.Bar(name=fac, x=ff_res.index.astype(str), y=ff_res[fac].values))

    fig = go.Figure(data=bars)
    fig.update_layout(
        template="plotly_white",
        barmode="group",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Ticker",
        yaxis_title="Factor Loading",
        legend_title="Factor",
    )
    return fig


def macro_lines(df: pd.DataFrame) -> go.Figure:
    """
    Simple multi-line chart for macro series (e.g., GDP, CPI, Yields).
    """
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=str(col), mode="lines"))

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="",
        yaxis_title="",
    )
    return fig
