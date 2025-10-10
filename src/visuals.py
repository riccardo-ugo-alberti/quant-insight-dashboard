# src/visuals.py
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

PALETTE_SEQ = px.colors.qualitative.Set2
TEMPLATE = "plotly_white"

# Classic vivid RdBu colorscale (stronger blues, standard reds)
RDBU_CLASSIC = [
    [0.00, "#67001f"], [0.10, "#b2182b"], [0.20, "#d6604d"],
    [0.30, "#f4a582"], [0.40, "#fddbc7"], [0.50, "#f7f7f7"],
    [0.60, "#d1e5f0"], [0.70, "#92c5de"], [0.80, "#4393c3"],
    [0.90, "#2166ac"], [1.00, "#053061"],
]

def _tweak(fig: go.Figure, *, height: int | None = None, show_slider: bool = True) -> go.Figure:
    fig.update_layout(
        template=TEMPLATE,
        margin=dict(l=10, r=10, t=50, b=10),
        legend_title_text="",
        hovermode="x unified",
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(rangeslider_visible=show_slider)
    return fig

# ---------- KPI helpers ----------
def kpi_delta_color(delta_value: float) -> str:
    return "normal" if delta_value >= 0 else "inverse"

def kpi_card(label: str, value_str: str, delta_str: str | None = None, delta_val: float | None = None):
    kw = {"label": label, "value": value_str}
    if delta_str is not None:
        kw["delta"] = delta_str
        if delta_val is not None:
            kw["delta_color"] = kpi_delta_color(delta_val)
    return kw

# ---------- Price / Performance ----------
def price_lines(prices: pd.DataFrame, height: int = 520) -> go.Figure:
    fig = px.line(
        prices, x=prices.index, y=prices.columns,
        color_discrete_sequence=PALETTE_SEQ,
        labels={"x": "Date", "value": "Adj Close", "variable": "Ticker"},
        title="Adjusted Close Prices"
    )
    return _tweak(fig, height=height)

def cumret_lines(cum: pd.DataFrame, title: str = "Cumulative Returns (base=100)", height: int = 560) -> go.Figure:
    fig = px.line(
        cum, x=cum.index, y=cum.columns,
        color_discrete_sequence=PALETTE_SEQ,
        labels={"x": "Date", "value": "Index", "variable": "Ticker"},
        title=title
    )
    return _tweak(fig, height=height)

# ---------- Risk ----------
def rolling_vol_lines(vol: pd.DataFrame, window: int, height: int = 520) -> go.Figure:
    fig = px.line(
        vol, x=vol.index, y=vol.columns,
        color_discrete_sequence=PALETTE_SEQ,
        labels={"x": "Date", "value": "Annualized Volatility", "variable": "Ticker"},
        title=f"Rolling Annualized Volatility (window = {window} days)"
    )
    return _tweak(fig, height=height)

def beta_lines(betas: pd.DataFrame, benchmark: str, window: int, height: int = 500) -> go.Figure:
    df = betas.drop(columns=[benchmark], errors="ignore")
    fig = px.line(
        df, x=df.index, y=df.columns,
        color_discrete_sequence=PALETTE_SEQ,
        labels={"x": "Date", "value": f"β vs {benchmark}", "variable": "Ticker"},
        title=f"Rolling CAPM Beta vs {benchmark} (window = {window} days)"
    )
    return _tweak(fig, height=height)

# ---------- Correlation ----------
def corr_heatmap(cm: pd.DataFrame, title: str = "Correlation (daily returns)", height: int = 560) -> go.Figure:
    z = cm.values
    text = [[f"{v:.2f}" for v in row] for row in z]
    fig = go.Figure(data=go.Heatmap(
        z=z, x=cm.columns, y=cm.index,
        colorscale=RDBU_CLASSIC,
        reversescale=False, zmin=-1, zmax=1, zmid=0,
        colorbar=dict(title="ρ"),
        text=text, texttemplate="%{text}", textfont=dict(size=10)
    ))
    fig.update_layout(title=title, margin=dict(l=50, r=10, t=50, b=10), template=TEMPLATE, height=height)
    return fig

def rolling_corr_line(rc: pd.Series, a: str, b: str, window: int, height: int = 480) -> go.Figure:
    df = rc.to_frame(name=f"{a}–{b}")
    fig = px.line(
        df, x=df.index, y=df.columns,
        color_discrete_sequence=PALETTE_SEQ,
        labels={"x": "Date", "value": f"Rolling corr ({window}d)", "variable": "Pair"},
        title=f"Rolling Correlation — {a} vs {b}"
    )
    return _tweak(fig, height=height)

# ---------- Scatter + Bars ----------
def _ols_line(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    X = sm.add_constant(x)
    model = sm.OLS(y, X, missing="drop").fit()
    x_line = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    y_line = model.params[0] + model.params[1] * x_line
    return x_line, y_line

def sharpe_vol_scatter(summary: pd.DataFrame, sectors: dict[str, str] | None = None,
                       height: int = 560, trendline: bool = True) -> go.Figure:
    df = summary.reset_index().rename(columns={"index": "Ticker"})
    color = "Sector" if sectors else None
    if sectors:
        df["Sector"] = df["Ticker"].map(sectors).fillna("Other")

    fig = px.scatter(
        df, x="Ann. Vol", y="Ann. Return",
        color=color, text="Ticker",
        size=(df["Sharpe"].abs() + 0.2),
        labels={"Ann. Vol": "Annualized Volatility", "Ann. Return": "Annualized Return"},
        title="Risk vs Return (marker size ~ |Sharpe|)"
    )
    fig.update_traces(textposition="top center")

    if trendline and len(df) >= 2 and df["Ann. Vol"].notna().sum() >= 2 and df["Ann. Return"].notna().sum() >= 2:
        x_line, y_line = _ols_line(df["Ann. Vol"].values, df["Ann. Return"].values)
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines",
            line=dict(color="rgba(44,62,80,0.4)", width=2, dash="dash"),
            name="OLS regression",
            hoverinfo="skip",
            showlegend=True
        ))

    return _tweak(fig, height=height, show_slider=False)

def bar_annual_return(summary: pd.DataFrame, height: int = 420) -> go.Figure:
    df = summary["Ann. Return"].sort_values(ascending=False).to_frame().reset_index()
    df.columns = ["Ticker", "Ann. Return"]
    fig = px.bar(df, x="Ticker", y="Ann. Return", color="Ticker",
                 color_discrete_sequence=PALETTE_SEQ,
                 text=df["Ann. Return"].map(lambda v: f"{v:.1%}"),
                 title="Annualized Return (ranking)")
    fig.update_traces(textposition="outside")
    fig.update_yaxes(tickformat=".0%")
    return _tweak(fig, height=height, show_slider=False)

def bar_annual_vol(summary: pd.DataFrame, height: int = 420) -> go.Figure:
    df = summary["Ann. Vol"].sort_values(ascending=False).to_frame().reset_index()
    df.columns = ["Ticker", "Ann. Vol"]
    fig = px.bar(df, x="Ticker", y="Ann. Vol", color="Ticker",
                 color_discrete_sequence=PALETTE_SEQ,
                 text=df["Ann. Vol"].map(lambda v: f"{v:.1%}"),
                 title="Annualized Volatility (ranking)")
    fig.update_traces(textposition="outside")
    fig.update_yaxes(tickformat=".0%")
    return _tweak(fig, height=height, show_slider=False)

# ---------- NEW: Analytics visuals ----------
def rolling_sharpe_lines(rs: pd.DataFrame, height: int = 520) -> go.Figure:
    fig = px.line(
        rs, x=rs.index, y=rs.columns,
        color_discrete_sequence=PALETTE_SEQ,
        labels={"x": "Date", "value": "Rolling Sharpe", "variable": "Ticker"},
        title="Rolling Sharpe Ratio"
    )
    return _tweak(fig, height=height)

def drawdown_area(dd: pd.DataFrame, ticker: str, height: int = 420) -> go.Figure:
    if ticker not in dd.columns:
        raise ValueError(f"{ticker} not found in drawdown series.")
    s = dd[ticker].fillna(0.0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values, mode="lines",
        line=dict(color="rgba(52, 73, 94, 0.9)", width=1.5),
        name=f"Drawdown — {ticker}"
    ))
    fig.add_trace(go.Scatter(
        x=s.index, y=np.minimum(s.values, 0.0),
        fill="tozeroy", mode="none",
        fillcolor="rgba(52, 73, 94, 0.25)",
        name="Drawdown area"
    ))
    fig.update_layout(
        title=f"Drawdown (peak-to-trough) — {ticker}",
        template=TEMPLATE, height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis=dict(tickformat=".0%")
    )
    return fig

# ---------- Table styling (light, professional) ----------
def _base_style(s: pd.io.formats.style.Styler) -> pd.io.formats.style.Styler:
    s = s.set_properties(**{"border": "1px solid #DDD", "border-collapse": "collapse"})
    s = s.set_table_styles([{"selector": "th", "props": [("background-color", "#F7F7F7")]}])
    return s

def style_summary_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    s = df.style.format({
        "Ann. Return": "{:.2%}",
        "Ann. Vol": "{:.2%}",
        "Sharpe": "{:.2f}",
        "Sortino": "{:.2f}",
        "Max Drawdown": "{:.2%}",
        "Longest DD (days)": "{:.0f}",
        "Recovery (days)": "{:.0f}",
    })
    return _base_style(s)

def style_corr_pairs_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    s = df.style.format({"rho": "{:.2f}"})
    return _base_style(s)

def style_prices_preview(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    s = df.tail().style.format("{:.2f}")
    return _base_style(s)
