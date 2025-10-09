# src/visuals.py
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def _ranges(fig):
    fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=40, b=10), template="plotly_white")
    fig.update_xaxes(rangeslider_visible=True)
    return fig

def price_lines(prices: pd.DataFrame):
    fig = px.line(
        prices, x=prices.index, y=prices.columns,
        labels={"x": "Date", "value": "Price", "variable": "Ticker"}
    )
    return _ranges(fig)

def cumret_lines(cum: pd.DataFrame):
    fig = px.line(
        cum, x=cum.index, y=cum.columns,
        labels={"x": "Date", "value": "Index (start=100)", "variable": "Ticker"}
    )
    return _ranges(fig)

def rolling_vol_lines(vol: pd.DataFrame):
    fig = px.line(
        vol, x=vol.index, y=vol.columns,
        labels={"x": "Date", "value": "Ann. Volatility", "variable": "Ticker"}
    )
    return _ranges(fig)

def corr_heatmap(cm: pd.DataFrame, title: str = "Correlation (daily returns)"):
    z = cm.values
    text = [[f"{v:.2f}" for v in row] for row in z]
    fig = go.Figure(data=go.Heatmap(
        z=z, x=cm.columns, y=cm.index,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title="ρ"), text=text, texttemplate="%{text}", textfont=dict(size=10)
    ))
    fig.update_layout(title=title, margin=dict(l=60, r=10, t=40, b=10), template="plotly_white")
    return fig

def rolling_corr_line(rc: pd.Series, a: str, b: str, window: int):
    df = rc.to_frame(name=f"{a}–{b}")
    fig = px.line(
        df, x=df.index, y=df.columns,
        labels={"x": "Date", "value": f"Rolling corr ({window}d)", "variable": "Pair"}
    )
    return _ranges(fig)

def sharpe_vol_scatter(summary: pd.DataFrame, sectors: dict[str, str] | None = None):
    df = summary.reset_index().rename(columns={"index": "Ticker"})
    if sectors:
        df["Sector"] = df["Ticker"].map(sectors).fillna("Unknown")
        color = "Sector"
    else:
        color = None
    fig = px.scatter(
        df, x="Ann. Vol", y="Ann. Return", text="Ticker",
        size=df["Sharpe"].abs() + 0.2,
        color=color,
        hover_data=["Sharpe", "Max Drawdown"],
        labels={"Ann. Vol": "Annualized Volatility", "Ann. Return": "Annualized Return"}
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), template="plotly_white")
    return fig

def beta_lines(betas: pd.DataFrame, benchmark: str):
    df = betas.drop(columns=[benchmark], errors="ignore")  # hide bench self-beta
    fig = px.line(
        df, x=df.index, y=df.columns,
        labels={"x": "Date", "value": f"Rolling Beta vs {benchmark}", "variable": "Ticker"}
    )
    return _ranges(fig)
