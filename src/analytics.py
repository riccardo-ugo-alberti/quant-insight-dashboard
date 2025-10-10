# src/analytics.py
from __future__ import annotations
import numpy as np
import pandas as pd

# -----------------------
# Core transforms
# -----------------------
def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    rets = prices.sort_index().pct_change().dropna(how="all")
    return rets

def to_cum_returns(rets: pd.DataFrame) -> pd.DataFrame:
    return (1 + rets).cumprod() * 100  # base=100

# -----------------------
# Risk & rolling
# -----------------------
def rolling_vol(rets: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    return rets.rolling(window).std() * np.sqrt(252)

def corr_matrix(rets: pd.DataFrame) -> pd.DataFrame:
    return rets.corr()

def order_corr_matrix(corr: pd.DataFrame) -> pd.DataFrame:
    # simple ordering by mean corr (cluster-ish look without scipy)
    order = corr.mean().sort_values(ascending=False).index
    return corr.loc[order, order]

def rolling_corr(rets: pd.DataFrame, a: str, b: str, window: int = 63) -> pd.Series:
    return rets[a].rolling(window).corr(rets[b])

# -----------------------
# Drawdowns
# -----------------------
def drawdown_series(rets: pd.DataFrame) -> pd.DataFrame:
    """Return drawdown series (cumprod vs running max) for each column."""
    growth = (1 + rets).cumprod()
    peak = growth.cummax()
    dd = (growth / peak) - 1.0
    return dd

def max_drawdown(rets: pd.DataFrame) -> pd.Series:
    dd = drawdown_series(rets)
    return dd.min()

def longest_drawdown_days(rets: pd.DataFrame) -> pd.Series:
    """Longest consecutive days under water (drawdown < 0)."""
    dd = drawdown_series(rets)
    out = {}
    for c in dd.columns:
        under = dd[c] < 0
        # run-length encoding of consecutive True blocks
        max_len = 0
        cur = 0
        for v in under.to_numpy():
            if v:
                cur += 1
                max_len = max(max_len, cur)
            else:
                cur = 0
        out[c] = int(max_len)
    return pd.Series(out)

def recovery_days_from_max_dd(rets: pd.DataFrame) -> pd.Series:
    """
    Days from the trough of the *maximum* drawdown to the next recovery (dd back to 0).
    If never recovered, returns NaN.
    """
    dd = drawdown_series(rets)
    out = {}
    for c in dd.columns:
        s = dd[c].dropna()
        if s.empty:
            out[c] = np.nan
            continue
        trough_idx = s.idxmin()  # date of max drawdown
        # find next date where drawdown returns to (approximately) 0
        recovered = s.loc[trough_idx:].pipe(lambda x: x[np.isclose(x, 0.0, atol=1e-10) | (x >= -1e-12)])
        if recovered.empty:
            out[c] = np.nan
        else:
            out[c] = (recovered.index[0] - trough_idx).days
    return pd.Series(out)

# -----------------------
# Sharpe / Sortino / Rolling metrics
# -----------------------
def annualize_return(rets: pd.Series) -> float:
    mu = rets.mean() * 252
    return float(mu)

def annualize_vol(rets: pd.Series) -> float:
    sigma = rets.std() * np.sqrt(252)
    return float(sigma)

def sharpe_ratio(rets: pd.Series, rf_annual: float = 0.0) -> float:
    mu = annualize_return(rets) - rf_annual
    sig = annualize_vol(rets)
    return float(mu / sig) if sig and not np.isclose(sig, 0) else np.nan

def sortino_ratio(rets: pd.Series, rf_annual: float = 0.0) -> float:
    # downside deviation uses negative *excess* returns
    rf_daily = rf_annual / 252.0
    excess = rets - rf_daily
    negatives = excess[excess < 0]
    if len(negatives) == 0:
        return np.nan
    downside_vol_annual = negatives.std() * np.sqrt(252)
    mu_excess_annual = excess.mean() * 252
    return float(mu_excess_annual / downside_vol_annual) if downside_vol_annual and not np.isclose(downside_vol_annual, 0) else np.nan

def rolling_sharpe(rets: pd.DataFrame, window: int = 63, rf_annual: float = 0.0) -> pd.DataFrame:
    rf_daily = rf_annual / 252.0
    excess = rets - rf_daily
    mu = excess.rolling(window).mean() * 252
    sigma = rets.rolling(window).std() * np.sqrt(252)
    rs = mu / sigma
    return rs

def rolling_drawdown(rets: pd.DataFrame) -> pd.DataFrame:
    """Drawdown level over time (same as drawdown_series, kept for clarity)."""
    return drawdown_series(rets)

# -----------------------
# Rolling beta (CAPM)
# -----------------------
def rolling_beta(rets: pd.DataFrame, benchmark: str, window: int = 63) -> pd.DataFrame:
    """
    Rolling beta via cov/var per window.
    """
    if benchmark not in rets:
        raise ValueError(f"Benchmark {benchmark} not in return columns")
    out = {}
    bench = rets[benchmark]
    for c in rets.columns:
        cov = rets[c].rolling(window).cov(bench)
        var = bench.rolling(window).var()
        out[c] = cov / var
    return pd.DataFrame(out, index=rets.index)

# -----------------------
# Summary table
# -----------------------
def summary_table(rets: pd.DataFrame, rf: float = 0.0) -> pd.DataFrame:
    """
    Returns: Ann. Return, Ann. Vol, Sharpe, Sortino, Max Drawdown, Longest DD (days), Recovery (days)
    """
    cols = rets.columns
    ann_ret = rets.apply(annualize_return, axis=0)
    ann_vol = rets.apply(annualize_vol, axis=0)
    sharpe = rets.apply(lambda s: sharpe_ratio(s, rf), axis=0)
    sortino = rets.apply(lambda s: sortino_ratio(s, rf), axis=0)
    mdd = max_drawdown(rets)
    longest_dd = longest_drawdown_days(rets)
    rec_days = recovery_days_from_max_dd(rets)

    df = pd.DataFrame({
        "Ann. Return": ann_ret,
        "Ann. Vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": mdd,
        "Longest DD (days)": longest_dd.reindex(cols),
        "Recovery (days)": rec_days.reindex(cols),
    })
    return df.sort_index()

# -----------------------
# Helpers for correlation pairs
# -----------------------
def top_corr_pairs(cm: pd.DataFrame, k: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns two dataframes with columns [a, b, rho] for top + and top - correlations (off-diagonal).
    """
    pairs = []
    cols = cm.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], cm.iloc[i, j]))
    df = pd.DataFrame(pairs, columns=["a", "b", "rho"]).dropna()
    df_pos = df.sort_values("rho", ascending=False).head(k).reset_index(drop=True)
    df_neg = df.sort_values("rho", ascending=True).head(k).reset_index(drop=True)
    return df_pos, df_neg
