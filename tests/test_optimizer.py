# tests/test_optimizer.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.optimizer import (
    optimize_portfolio,
    compute_frontier,
    optimize_cvar,
)

TRADING_DAYS = 252


def _make_fake_returns(n_days: int = 750, tickers: list[str] | None = None) -> pd.DataFrame:
    """
    Genera rendimenti giornalieri sintetici (senza internet) per test veloci e deterministici.
    """
    if tickers is None:
        tickers = ["AAPL", "MSFT", "SPY", "NVDA", "GOOG"]
    rng = np.random.default_rng(42)

    # media e covarianza “realistiche”: rendimenti attesi annui 6–12%, vol 15–25%
    mu_ann = rng.uniform(0.06, 0.12, size=len(tickers))
    vol_ann = rng.uniform(0.15, 0.25, size=len(tickers))
    mu_day = mu_ann / TRADING_DAYS
    vol_day = vol_ann / np.sqrt(TRADING_DAYS)

    # matrice di correlazione casuale ma ben condizionata
    A = rng.normal(size=(len(tickers), len(tickers)))
    corr = np.corrcoef(A)
    D = np.diag(vol_day)
    cov_day = D @ corr @ D

    R = rng.multivariate_normal(mean=mu_day, cov=cov_day, size=n_days)
    df = pd.DataFrame(R, columns=tickers)
    return df


def test_max_sharpe_basic():
    returns = _make_fake_returns()
    rf = 0.02
    res = optimize_portfolio(
        returns=returns, rf=rf, mode="max_sharpe",
        allow_short=False, max_weight=0.60
    )
    # pesi validi
    assert np.isfinite(res.weights.values).all()
    assert abs(res.weights.sum() - 1.0) < 1e-6
    assert (res.weights.values >= -1e-8).all() and (res.weights.values <= 0.60 + 1e-8).all()
    # metriche finite
    assert np.isfinite(res.exp_return)
    assert np.isfinite(res.exp_vol) and res.exp_vol >= 0
    assert np.isfinite(res.sharpe)


def test_min_vol_vs_max_sharpe():
    returns = _make_fake_returns()
    rf = 0.02
    maxsh = optimize_portfolio(returns=returns, rf=rf, mode="max_sharpe", allow_short=False, max_weight=0.60)
    minvol = optimize_portfolio(returns=returns, rf=rf, mode="min_vol", allow_short=False, max_weight=0.60)
    # la soluzione min_vol non deve avere volatilità maggiore della max_sharpe (di solito è <=)
    assert minvol.exp_vol <= maxsh.exp_vol + 1e-9


def test_target_return_hits_level():
    returns = _make_fake_returns()
    rf = 0.02
    # scegliamo un target nel range plausibile
    # calcoliamo rapidamente il range dai dati sintetici
    mu_day = returns.mean()
    mu_ann = mu_day * TRADING_DAYS
    t_min, t_max = float(mu_ann.min()), float(mu_ann.max())
    target = (t_min + t_max) / 2.0  # punto intermedio

    res = optimize_portfolio(
        returns=returns, rf=rf, mode="target_return",
        target_return=target, allow_short=False, max_weight=0.60
    )
    # il rendimento deve essere vicino al target (tolleranza 50 bps)
    assert abs(res.exp_return - target) < 0.005 + 1e-9
    assert abs(res.weights.sum() - 1.0) < 1e-6


def test_cvar_optimizer_valid():
    returns = _make_fake_returns()
    res = optimize_cvar(returns=returns, alpha=0.95, allow_short=False, max_weight=0.50)
    # pesi in [0, cap], somma = 1
    w = res.weights.values
    assert (w >= -1e-8).all() and (w <= 0.50 + 1e-8).all()
    assert abs(w.sum() - 1.0) < 1e-6
    # metriche finite
    assert np.isfinite(res.exp_return)
    assert np.isfinite(res.exp_vol) and res.exp_vol >= 0
    # messaggio presente (COBYLA di solito riporta esito)
    assert isinstance(res.message, str) and len(res.message) > 0


def test_frontier_shape():
    returns = _make_fake_returns()
    front = compute_frontier(returns=returns, rf=0.02, n_points=15, allow_short=False, max_weight=0.40)
    assert set(["Return", "Vol", "Sharpe"]).issubset(front.columns)
    assert len(front) > 5
    assert np.isfinite(front["Return"]).all()
    assert np.isfinite(front["Vol"]).all() and (front["Vol"] >= 0).all()
