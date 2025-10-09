# src/analytics.py
import numpy as np
import pandas as pd

TRADING_DAYS = 252

def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.sort_index().pct_change().dropna(how="all")

def to_cum_returns(returns: pd.DataFrame, start_at_100: bool = True) -> pd.DataFrame:
    cum = (1 + returns).cumprod()
    if start_at_100:
        cum = cum * 100
    return cum

def rolling_vol(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)

def corr_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()

def rolling_corr(returns: pd.DataFrame, a: str, b: str, window: int = 63) -> pd.Series:
    return returns[a].rolling(window).corr(returns[b])

def order_corr_matrix(cm: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder correlation matrix to group similar assets.
    Try hierarchical clustering via scipy if available;
    otherwise fallback to sorting by first eigenvector (PCA-like).
    """
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        dist = 1 - cm.values
        z = linkage(squareform(dist, checks=False), method="average")
        order = leaves_list(z)
        ordered = cm.iloc[order, :].iloc[:, order]
        ordered.index = cm.index[order]
        ordered.columns = cm.columns[order]
        return ordered
    except Exception:
        vals, vecs = np.linalg.eigh(cm.values)
        first = vecs[:, -1]
        order = np.argsort(first)
        ordered = cm.iloc[order, :].iloc[:, order]
        ordered.index = cm.index[order]
        ordered.columns = cm.columns[order]
        return ordered

def summary_table(returns: pd.DataFrame, rf: float = 0.0) -> pd.DataFrame:
    """
    Annualized return, volatility, Sharpe, Max Drawdown for each asset.
    rf is annual risk-free rate in decimal (e.g., 0.02 for 2%).
    """
    n = len(returns)
    if n == 0:
        return pd.DataFrame(columns=["Ann. Return","Ann. Vol","Sharpe","Max Drawdown"])

    ann_ret = (1 + returns).prod() ** (TRADING_DAYS / n) - 1
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)

    daily_rf = (1 + rf) ** (1 / TRADING_DAYS) - 1
    excess = returns - daily_rf
    sharpe = (excess.mean() * TRADING_DAYS) / (returns.std() * np.sqrt(TRADING_DAYS))

    equity = (1 + returns).cumprod()
    dd = equity / equity.cummax() - 1
    mdd = dd.min()

    out = pd.DataFrame({
        "Ann. Return": ann_ret,
        "Ann. Vol": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": mdd
    }).round(4)

    return out.sort_values("Ann. Return", ascending=False)

def top_corr_pairs(cm: pd.DataFrame, k: int = 5):
    """Return top + and - correlated pairs from a correlation matrix (upper triangle)."""
    pairs = []
    cols = cm.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            pairs.append((cols[i], cols[j], cm.iloc[i, j]))
    df = pd.DataFrame(pairs, columns=["A", "B", "rho"]).dropna()
    top_pos = df.sort_values("rho", ascending=False).head(k)
    top_neg = df.sort_values("rho", ascending=True).head(k)
    return top_pos, top_neg

def rolling_beta(returns: pd.DataFrame, benchmark: str, window: int = 63) -> pd.DataFrame:
    """
    Rolling CAPM beta for each column versus the chosen benchmark.
    beta = cov(asset, bench) / var(bench)
    """
    if benchmark not in returns.columns:
        raise ValueError(f"Benchmark '{benchmark}' not in returns columns.")
    bench = returns[benchmark]
    var_b = bench.rolling(window).var()
    betas = {}
    for col in returns.columns:
        cov_ab = returns[col].rolling(window).cov(bench)
        betas[col] = cov_ab / var_b
    df = pd.DataFrame(betas, index=returns.index)
    return df
