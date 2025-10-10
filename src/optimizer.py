# src/optimizer.py
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .analytics import portfolio_returns, expected_shortfall

def _constraints(n: int, allow_short: bool, weight_cap: float | None):
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = None
    if allow_short:
        lo, hi = (-1.0, 1.0)
    else:
        lo, hi = (0.0, 1.0)
    if weight_cap is not None:
        hi = min(hi, weight_cap)
    bounds = [(lo, hi)] * n
    return cons, bounds

def mean_variance_opt(returns: pd.DataFrame, mode: str = "max_sharpe",
                      target_return: float | None = None,
                      rf: float = 0.0, allow_short: bool = False,
                      weight_cap: float | None = 0.3,
                      periods_per_year: int = 252) -> dict:
    """
    Simple MV optimizer via SLSQP. mode in {"max_sharpe","min_vol","target"}
    """
    rets = returns.dropna(how="any")
    mu = rets.mean() * periods_per_year
    cov = rets.cov() * periods_per_year
    n = rets.shape[1]
    cons, bounds = _constraints(n, allow_short, weight_cap)
    w0 = np.repeat(1/n, n)

    if mode == "min_vol":
        obj = lambda w: np.sqrt(w @ cov.values @ w)
    elif mode == "target":
        assert target_return is not None, "target_return required for mode='target'"
        cons = cons + [{"type": "eq", "fun": lambda w, mu=mu: float(w @ mu.values) - target_return}]
        obj = lambda w: np.sqrt(w @ cov.values @ w)
    else:  # max_sharpe
        def neg_sharpe(w):
            sig = np.sqrt(w @ cov.values @ w)
            mu_p = float(w @ mu.values)
            return - (mu_p - rf) / (sig + 1e-12)
        obj = neg_sharpe

    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 100})
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    w = res.x
    mu_p = float(w @ mu.values)
    vol_p = float(np.sqrt(w @ cov.values @ w))
    sh = (mu_p - rf) / (vol_p + 1e-12)
    return {"weights": w, "mu": mu_p, "sigma": vol_p, "sharpe": sh, "success": res.success}

def cvar_opt(returns: pd.DataFrame, alpha: float = 0.95,
             allow_short: bool = False, weight_cap: float | None = 0.3) -> dict:
    """
    CVaR (ES) minimization via SLSQP on historical returns (black-box).
    Minimize Expected Shortfall of portfolio returns at level alpha.
    """
    R = returns.dropna(how="any").values
    n = R.shape[1]
    cons, bounds = _constraints(n, allow_short, weight_cap)
    w0 = np.repeat(1/n, n)

    def obj(w):
        pr = R @ w
        return expected_shortfall(pr, alpha=alpha)

    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 150})
    if not res.success:
        raise RuntimeError(f"CVaR optimization failed: {res.message}")

    w = res.x
    pr = R @ w
    es = expected_shortfall(pr, alpha=alpha)
    return {"weights": w, "es": es, "success": res.success}
