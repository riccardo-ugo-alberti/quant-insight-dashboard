# tests/test_optimizer.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimizer import (
    optimize_portfolio,
    optimize_portfolio_from_returns,
    compute_frontier,
    frontier_from_returns,
    optimize_cvar,
    optimize_cvar_from_returns,
    prepare_mv_inputs,
)

TRADING_DAYS = 252


def _make_fake_returns(n_days: int = 750, tickers: list[str] | None = None) -> pd.DataFrame:
    if tickers is None:
        tickers = ["AAPL", "MSFT", "SPY", "NVDA", "GOOG"]
    rng = np.random.default_rng(42)

    mu_ann = rng.uniform(0.06, 0.12, size=len(tickers))
    vol_ann = rng.uniform(0.15, 0.25, size=len(tickers))
    mu_day = mu_ann / TRADING_DAYS
    vol_day = vol_ann / np.sqrt(TRADING_DAYS)

    A = rng.normal(size=(len(tickers), len(tickers)))
    corr = np.corrcoef(A)
    D = np.diag(vol_day)
    cov_day = D @ corr @ D

    R = rng.multivariate_normal(mean=mu_day, cov=cov_day, size=n_days)
    return pd.DataFrame(R, columns=tickers)


def _returns_to_prices(returns: pd.DataFrame, start_price: float = 100.0) -> pd.DataFrame:
    return start_price * (1.0 + returns).cumprod()


def test_max_sharpe_basic():
    prices = _returns_to_prices(_make_fake_returns())
    rf = 0.02
    weights, (exp_return, exp_vol, sharpe), _ = optimize_portfolio(
        prices=prices,
        rf=rf,
        mode="max_sharpe",
        shorting=False,
        max_weight=0.60,
    )

    assert np.isfinite(weights.values).all()
    assert abs(weights.sum() - 1.0) < 1e-6
    assert (weights.values >= -1e-8).all() and (weights.values <= 0.60 + 1e-8).all()
    assert np.isfinite(exp_return)
    assert np.isfinite(exp_vol) and exp_vol >= 0
    assert np.isfinite(sharpe)


def test_min_vol_reduces_vol_vs_equal_weight():
    prices = _returns_to_prices(_make_fake_returns())
    rf = 0.02
    _, (_, vol_minvol, _), mv = optimize_portfolio(prices=prices, rf=rf, mode="min_vol", shorting=False, max_weight=0.60)

    w_eq = np.ones(len(mv.mu)) / len(mv.mu)
    vol_equal = float(np.sqrt(w_eq @ mv.cov.values @ w_eq))
    assert vol_minvol <= vol_equal + 1e-9


def test_target_return_hits_level():
    prices = _returns_to_prices(_make_fake_returns())
    rf = 0.02
    mv = prepare_mv_inputs(prices)
    target = float((mv.mu.min() + mv.mu.max()) / 2.0)

    weights, (exp_return, _, _), _ = optimize_portfolio(
        prices=prices,
        rf=rf,
        mode="target_return",
        target_return=target,
        shorting=False,
        max_weight=0.60,
    )

    assert abs(exp_return - target) < 0.005 + 1e-9
    assert abs(weights.sum() - 1.0) < 1e-6


def test_cvar_optimizer_valid_or_unavailable():
    prices = _returns_to_prices(_make_fake_returns())
    res = optimize_cvar(prices=prices, alpha=0.95, shorting=False, max_weight=0.50)
    if res is None:
        pytest.skip("cvxpy not available in this environment")

    w = res.values
    assert (w >= -1e-8).all() and (w <= 0.50 + 1e-8).all()
    assert abs(w.sum() - 1.0) < 1e-6


def test_frontier_shape():
    prices = _returns_to_prices(_make_fake_returns())
    mv = prepare_mv_inputs(prices)
    front = compute_frontier(mv.mu, mv.cov, rf=0.02, points=15, shorting=False, max_weight=0.40)
    assert set(["Return", "Vol", "Sharpe"]).issubset(front.columns)
    assert len(front) > 5
    assert np.isfinite(front["Return"]).all()
    assert np.isfinite(front["Vol"]).all() and (front["Vol"] >= 0).all()


def test_frontier_runtime_regression_guard():
    import time

    prices = _returns_to_prices(_make_fake_returns(n_days=900, tickers=[f"T{i}" for i in range(12)]))
    mv = prepare_mv_inputs(prices)

    t0 = time.perf_counter()
    front = compute_frontier(mv.mu, mv.cov, rf=0.02, points=35, shorting=False, max_weight=0.35)
    elapsed = time.perf_counter() - t0

    assert len(front) >= 10
    # Guardrail for accidental major slowdowns in CI-sized environments.
    assert elapsed < 4.0


def test_infeasible_target_return_has_clear_error():
    prices = _returns_to_prices(_make_fake_returns())
    with pytest.raises(ValueError, match="target_return is infeasible"):
        optimize_portfolio(
            prices=prices,
            rf=0.02,
            mode="target_return",
            target_return=10.0,
            shorting=False,
            max_weight=0.30,
        )


def test_invalid_max_weight_error():
    prices = _returns_to_prices(_make_fake_returns())
    with pytest.raises(ValueError, match="max_weight must be > 0"):
        optimize_portfolio(prices=prices, mode="max_sharpe", max_weight=0.0)


def test_frontier_points_validation():
    prices = _returns_to_prices(_make_fake_returns())
    mv = prepare_mv_inputs(prices)
    with pytest.raises(ValueError, match="points must be >= 2"):
        compute_frontier(mv.mu, mv.cov, points=1)


def test_explicit_returns_entrypoints_match_price_flow():
    returns = _make_fake_returns()
    prices = _returns_to_prices(returns)

    w_p, stats_p, mv_p = optimize_portfolio(prices=prices, mode="max_sharpe", shorting=False, max_weight=0.50)
    w_r, stats_r, mv_r = optimize_portfolio_from_returns(returns=returns, mode="max_sharpe", shorting=False, max_weight=0.50)

    assert list(mv_p.mu.index) == list(mv_r.mu.index)
    assert np.isfinite(w_r.values).all()
    assert abs(w_r.sum() - 1.0) < 1e-6
    assert np.isfinite(stats_r[0]) and np.isfinite(stats_r[1])

    f_r = frontier_from_returns(returns=returns, rf=0.01, points=12, shorting=False, max_weight=0.50)
    assert len(f_r) >= 5
    assert set(["Return", "Vol", "Sharpe"]).issubset(f_r.columns)

    cvar_r = optimize_cvar_from_returns(returns=returns, alpha=0.95, shorting=False, max_weight=0.50)
    if cvar_r is not None:
        assert abs(cvar_r.sum() - 1.0) < 1e-6
