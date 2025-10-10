# src/analytics.py
from __future__ import annotations
import numpy as np
import pandas as pd

# ----------------------------
# Base analytics on prices
# ----------------------------

def to_returns(prices: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """
    Log-returns (default) daily. If freq='M' resample monthly etc.
    """
    p = prices.ffill().dropna(how="all")
    if freq and freq.upper() != "D":
        p = p.resample(freq).last()
    rets = np.log(p / p.shift(1)).dropna(how="all")
    return rets

def cum_returns(returns: pd.DataFrame) -> pd.DataFrame:
    return np.exp(returns.cumsum())

def annualize_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    mu = returns.mean() * periods_per_year
    return mu

def annualize_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    vol = returns.std(ddof=0) * np.sqrt(periods_per_year)
    return vol

def sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    mu = annualize_return(returns, periods_per_year)
    sig = annualize_vol(returns, periods_per_year)
    rf_ann = rf
    return (mu - rf_ann) / (sig + 1e-12)

def max_drawdown(prices: pd.DataFrame) -> pd.Series:
    """Max drawdown per asset, calcolato su equity curve (cum returns)."""
    eq = prices / prices.iloc[0]
    roll_max = eq.cummax()
    dd = (eq / roll_max - 1.0).min()
    return dd

def summary_table(returns: pd.DataFrame, rf: float = 0.0, periods_per_year: int = 252) -> pd.DataFrame:
    out = []
    for c in returns.columns:
        r = returns[c].dropna()
        if len(r) < 2:
            continue
        row = {
            "Ann. Return": annualize_return(r, periods_per_year),
            "Ann. Vol": annualize_vol(r, periods_per_year),
            "Sharpe": sharpe(r, rf, periods_per_year),
            "Skew": r.skew(),
            "Kurt": r.kurt(),
            "HitRatio": (r > 0).mean(),
        }
        out.append(pd.Series(row, name=c))
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    return df

# ----------------------------
# Monte Carlo Simulations
# ----------------------------

def simulate_portfolios(returns: pd.DataFrame,
                        n_sims: int = 2000,
                        allow_short: bool = False,
                        weight_cap: float | None = 0.3,
                        seed: int | None = 42,
                        rf: float = 0.0,
                        periods_per_year: int = 252) -> pd.DataFrame:
    """
    Random weights -> compute annualized return/vol/sharpe for each portfolio.
    """
    rng = np.random.default_rng(seed)
    tickers = returns.columns.tolist()
    n = len(tickers)

    def sample_weights() -> np.ndarray:
        if allow_short:
            w = rng.normal(0, 1, n)
        else:
            w = rng.random(n)
        w = np.clip(w, -1.0, 1.0)
        if weight_cap is not None:
            w = np.clip(w, -weight_cap, weight_cap)
        if w.sum() == 0:
            w[rng.integers(0, n)] = 1.0
        w = w / np.sum(w)
        return w

    rets = returns.dropna(how="any")
    mu_vec = rets.mean() * periods_per_year
    cov = rets.cov() * periods_per_year

    rows = []
    for _ in range(n_sims):
        w = sample_weights()
        port_mu = float(np.dot(w, mu_vec))
        port_sig = float(np.sqrt(w @ cov.values @ w))
        sh = (port_mu - rf) / (port_sig + 1e-12)
        rows.append({"mu": port_mu, "sigma": port_sig, "sharpe": sh, **{f"w_{t}": w[i] for i, t in enumerate(tickers)}})

    return pd.DataFrame(rows)

# ----------------------------
# Helper for CVaR/ES calculation (historical)
# ----------------------------

def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    return returns.dropna(how="any").values @ weights

def expected_shortfall(returns: np.ndarray, alpha: float = 0.95) -> float:
    """
    ES (CVaR) of portfolio returns array (NOT losses): negative of lower tail mean.
    """
    q = np.quantile(returns, 1 - alpha)
    tail = returns[returns <= q]
    if tail.size == 0:
        return -q
    es = tail.mean()
    return -float(es)  # loss is positive
