# src/optimizer.py
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional, Literal, List
from scipy.optimize import minimize

try:
    import cvxpy as cp  # optional, used for CVaR
    _HAS_CVXPY = True
except Exception:
    _HAS_CVXPY = False


# ---------- Helpers

def _to_numpy(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _annualize_mean(mu_daily: pd.Series) -> pd.Series:
    # geometric annualization of mean daily return
    return (1.0 + mu_daily).pow(252).sub(1.0)


def _annualize_cov(cov_daily: pd.DataFrame) -> pd.DataFrame:
    return cov_daily * 252.0


@dataclass
class MVInputs:
    """Inputs for mean-variance."""
    mu: pd.Series          # annualized expected returns
    cov: pd.DataFrame      # annualized covariance
    tickers: List[str]


def prepare_mv_inputs(prices: pd.DataFrame) -> MVInputs:
    """
    From a price DataFrame (index: date, cols: tickers) create annualized
    expected returns and covariance for MV optimization.
    """
    # daily % returns (drop the first NaN row)
    rets = prices.pct_change().dropna(how="all")
    mu_d = rets.mean()
    cov_d = rets.cov()

    mu = _annualize_mean(mu_d)
    cov = _annualize_cov(cov_d)

    tickers = list(prices.columns)
    # guard: drop columns with all-NaN or zero variance
    ok = cov.columns[(np.diag(cov.values) > 0)]
    mu = mu.loc[ok]
    cov = cov.loc[ok, ok]
    return MVInputs(mu=mu, cov=cov, tickers=list(ok))


def _portfolio_stats(w: np.ndarray, mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0) -> Tuple[float, float, float]:
    """Return (exp_return, vol, sharpe). rf in decimal, annualized."""
    w = _to_numpy(w)
    exp_return = float(np.dot(w, mu.values))
    vol = float(np.sqrt(w @ cov.values @ w))
    sharpe = (exp_return - rf) / vol if vol > 0 else -np.inf
    return exp_return, vol, sharpe


def _mv_optimize(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float = 0.0,
    mode: Literal["max_sharpe", "min_vol", "target_return"] = "max_sharpe",
    target_return: Optional[float] = None,
    shorting: bool = False,
    max_weight: float = 0.30,
    w0: Optional[np.ndarray] = None,
) -> Tuple[pd.Series, Tuple[float, float, float]]:
    """
    Core MV optimizer (SLSQP). Returns (weights Series, (ret, vol, sharpe)).
    """
    n = len(mu)
    if n == 0:
        raise ValueError("No assets available after preprocessing.")

    # bounds
    lo = -max_weight if shorting else 0.0
    hi = max_weight
    bnds = [(lo, hi) for _ in range(n)]

    # constraints: sum w = 1
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if mode == "target_return":
        if target_return is None:
            target_return = float(mu.mean())
        # add return constraint
        cons.append({"type": "eq", "fun": lambda w, mu=mu, tr=target_return: np.dot(w, mu.values) - tr})

        # objective = variance
        def obj(w):
            return float(w @ cov.values @ w)

    elif mode == "min_vol":
        def obj(w):
            return float(w @ cov.values @ w)

    else:  # max_sharpe
        def obj(w):
            ret, vol, _ = _portfolio_stats(w, mu, cov, rf)
            # maximize sharpe => minimize -sharpe
            return -((ret - rf) / vol if vol > 0 else -1e6)

    if w0 is None:
        w0 = np.ones(n) / n

    res = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 1000})
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    w = res.x
    ret, vol, sr = _portfolio_stats(w, mu, cov, rf)
    weights = pd.Series(w, index=mu.index, name="weight")
    return weights, (ret, vol, sr)


def compute_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float = 0.0,
    points: int = 25,
    shorting: bool = False,
    max_weight: float = 0.30,
) -> pd.DataFrame:
    """
    Compute the mean-variance efficient frontier as a DataFrame with
    columns ['Return','Vol','Sharpe'] and one row per target point.
    """
    mu_vals = mu.values
    min_tr = float(np.percentile(mu_vals, 10))
    max_tr = float(np.percentile(mu_vals, 90))
    targets = np.linspace(min_tr, max_tr, points)

    rows = []
    for tr in targets:
        try:
            _, stats = _mv_optimize(mu, cov, rf, mode="target_return",
                                    target_return=float(tr),
                                    shorting=shorting, max_weight=max_weight)
            ret, vol, sr = stats
            rows.append((ret, vol, sr))
        except Exception:
            # skip infeasible targets
            continue

    if not rows:
        # fall back to max sharpe single point to avoid UI breaking
        try:
            _, stats = _mv_optimize(mu, cov, rf, mode="max_sharpe",
                                    shorting=shorting, max_weight=max_weight)
            ret, vol, sr = stats
            rows = [(ret, vol, sr)]
        except Exception:
            rows = []

    df = pd.DataFrame(rows, columns=["Return", "Vol", "Sharpe"])
    return df


def optimize_portfolio(
    prices: pd.DataFrame,
    rf: float = 0.0,
    mode: Literal["max_sharpe", "min_vol", "target_return"] = "max_sharpe",
    target_return: Optional[float] = None,
    shorting: bool = False,
    max_weight: float = 0.30,
) -> Tuple[pd.Series, Tuple[float, float, float], MVInputs]:
    """
    High-level: prepare inputs from prices, run MV optimization.
    Returns (weights, (ret, vol, sharpe), mv_inputs).
    """
    mv = prepare_mv_inputs(prices)
    w, stats = _mv_optimize(mv.mu, mv.cov, rf, mode, target_return, shorting, max_weight)
    return w, stats, mv


# ---------- CVaR optimizer (optional)

def optimize_cvar(
    prices: pd.DataFrame,
    alpha: float = 0.95,
    shorting: bool = False,
    max_weight: float = 0.30,
) -> Optional[pd.Series]:
    """
    CVaR (Expected Shortfall) portfolio via historical simulation + LP.
    Requires cvxpy. Returns weights Series or None if unavailable.
    """
    if not _HAS_CVXPY:
        return None

    rets = prices.pct_change().dropna(how="any")
    if rets.empty:
        return None

    T, n = rets.shape
    R = rets.values  # (T, n)

    w = cp.Variable(n)
    z = cp.Variable(T)         # losses beyond VaR
    var = cp.Variable(1)       # VaR

    lo = -max_weight if shorting else 0.0
    hi = max_weight
    constraints = [
        cp.sum(w) == 1,
        w >= lo,
        w <= hi,
        z >= 0,
        z >= -(R @ w) - var,   # losses beyond VaR (negative returns)
    ]
    # Minimize CVaR = VaR + (1 / ((1-alpha)*T)) * sum(z)
    objective = cp.Minimize(var + (1.0 / ((1.0 - alpha) * T)) * cp.sum(z))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=False, max_iters=2000)

    if w.value is None:
        return None

    wv = np.asarray(w.value).ravel()
    weights = pd.Series(wv, index=rets.columns, name="weight")
    return weights


# ---------- Convenience for the UI (one consistent API)

def frontier_from_prices(
    prices: pd.DataFrame,
    rf: float = 0.0,
    points: int = 25,
    shorting: bool = False,
    max_weight: float = 0.30,
) -> pd.DataFrame:
    mv = prepare_mv_inputs(prices)
    return compute_frontier(mv.mu, mv.cov, rf=rf, points=points, shorting=shorting, max_weight=max_weight)
