# src/analytics.py
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "to_returns",
    "compute_summary",
    "compute_corr_matrix",
    "rolling_vol",
]

# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def to_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    """
    Convert adjusted prices to returns.
    log=False -> simple returns; log=True -> log-returns.
    """
    if prices.empty:
        return prices.copy()
    r = prices.sort_index().pct_change()
    if log:
        # usa log(1+r) ma preserva NaN del primo giorno
        r = np.log1p(r)
    return r

# ---------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------
def compute_summary(
    prices: pd.DataFrame,
    rf: float = 0.02,        # risk-free annuale, es. 2%
    freq: int = 252,         # frequenza trading days
) -> pd.DataFrame:
    """
    Tabella riassuntiva: Return (ann), Vol (ann), Sharpe.
    """
    if prices.empty:
        return pd.DataFrame(columns=["Return", "Vol", "Sharpe"])

    r = to_returns(prices)               # daily returns
    mean_daily = r.mean(skipna=True)     # media giornaliera
    vol_daily = r.std(skipna=True)       # std giornaliera

    ann_ret = (1 + mean_daily) ** freq - 1
    ann_vol = vol_daily * np.sqrt(freq)
    sharpe = (ann_ret - rf) / ann_vol.replace(0, np.nan)

    out = pd.DataFrame({
        "Return": ann_ret,
        "Vol": ann_vol,
        "Sharpe": sharpe,
    })
    # Ordine tickers come in input, se possibile
    out = out.reindex(columns=["Return", "Vol", "Sharpe"])
    return out.sort_index()

def compute_corr_matrix(
    prices: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Matrice di correlazione sui returns.
    """
    if prices.empty:
        return pd.DataFrame()
    r = to_returns(prices)
    return r.corr(method=method)

def rolling_vol(
    prices: pd.DataFrame,
    window: int = 90,
    freq: int = 252,
) -> pd.DataFrame:
    """
    Volatilità annualizzata rolling su `window` giorni.
    Ritorna un DataFrame con stesse colonne dei tickers.
    """
    if prices.empty:
        return pd.DataFrame()

    r = to_returns(prices)
    # std rolling * sqrt(freq) per annualizzare
    vol = r.rolling(window=window, min_periods=max(2, window // 3)).std() * np.sqrt(freq)
    return vol
# ... dentro src/analytics.py

__all__ = [
    "to_returns",
    "compute_summary",
    "compute_corr_matrix",
    "rolling_vol",
    "rolling_volatility",   # <-- AGGIUNTO
]

def rolling_volatility(prices, window: int = 90, freq: int = 252):
    """Alias per compatibilità retro-compatibile."""
    return rolling_vol(prices, window=window, freq=freq)
