# src/analytics.py
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "to_returns",
    "compute_summary",
    "compute_corr_matrix",
    "rolling_vol",
    "rolling_volatility",
]

# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def to_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    """
    Convert adjusted prices to daily returns (DECIMALS).
    log=False -> simple returns; log=True -> log-returns via log(1+r).
    """
    if prices.empty:
        return prices.copy()
    r = prices.sort_index().pct_change()
    if log:
        # usa log(1+r) ma preserva NaN del primo giorno
        r = np.log1p(r)
    # robustezza: togli inf/NaN spuri
    r = r.replace([np.inf, -np.inf], np.nan)
    return r

# ---------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------
def compute_summary(
    prices: pd.DataFrame,
    rf: float = 0.02,        # risk-free annuale (in decimali, 0.02 = 2%)
    freq: int = 252,         # trading days
) -> pd.DataFrame:
    """
    Tabella riassuntiva: Return (ann), Vol (ann), Sharpe.
    Tutti i valori in DECIMALI (es. 0.12 = 12% annuo).
    """
    if prices.empty:
        return pd.DataFrame(columns=["Return", "Vol", "Sharpe"])

    r = to_returns(prices)               # daily returns (decimali)
    r = r.astype(float)
    r = r.replace([np.inf, -np.inf], np.nan)

    mean_daily = r.mean(skipna=True)                 # media giornaliera
    vol_daily  = r.std(skipna=True, ddof=1)          # std giornaliera

    # Annualizzazione (coerente col resto dell'app: decimali)
    ann_ret = (1.0 + mean_daily)**freq - 1.0         # compounding da mean daily
    ann_vol = vol_daily * np.sqrt(freq)
    sharpe  = (ann_ret - rf) / ann_vol.replace(0, np.nan)

    out = pd.DataFrame({
        "Return": ann_ret,
        "Vol": ann_vol,
        "Sharpe": sharpe,
    })
    return out.sort_index()

def compute_corr_matrix(
    prices: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Matrice di correlazione sui returns (decimali).
    """
    if prices.empty:
        return pd.DataFrame()
    r = to_returns(prices).astype(float)
    r = r.replace([np.inf, -np.inf], np.nan)
    return r.corr(method=method)

def _looks_like_prices(df: pd.DataFrame) -> bool:
    """
    Heuristica semplice: se i valori escono chiaramente dal range tipico dei returns
    (|x| > 1.5) o se quasi tutti sono positivi e ben > 1, trattiamo come prezzi.
    """
    if df.empty:
        return False
    x = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    if x.empty:
        return False
    # se c'è qualche valore enorme in valore assoluto, probabilmente sono livelli di prezzo
    if (x.abs() > 1.5).any().any():
        return True
    # se la mediana è molto > 1 e quasi tutto è positivo → prezzi
    med = x.median(numeric_only=True).median()
    frac_pos = (x > 0).sum().sum() / x.size
    return bool((med is not np.nan) and (med > 1.2) and (frac_pos > 0.95))

def rolling_vol(
    data: pd.DataFrame,
    window: int = 90,
    freq: int = 252,
) -> pd.DataFrame:
    """
    Volatilità annualizzata rolling su `window` giorni (DECIMALI).
    Accetta SIA returns SIA prezzi:
      - se gli passi i returns (come fa il tuo main), usa direttamente i returns;
      - se gli passi i prezzi, calcola prima i returns con pct_change().
    """
    if data.empty:
        return pd.DataFrame()

    if _looks_like_prices(data):
        r = to_returns(data)
    else:
        # assume returns già in ingresso
        r = data.copy()

    r = r.astype(float).replace([np.inf, -np.inf], np.nan)

    vol = (
        r.rolling(window=window, min_periods=max(10, window // 3))
         .std(ddof=1)
         * np.sqrt(freq)
    )
    # output in DECIMALI (es. 0.25 = 25% ann.)
    return vol

def rolling_volatility(prices_or_returns: pd.DataFrame, window: int = 90, freq: int = 252):
    """Alias retro-compatibile di rolling_vol."""
    return rolling_vol(prices_or_returns, window=window, freq=freq)
