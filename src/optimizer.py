# src/optimizer.py
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# -------------------------------------------------
# Mean–Variance (Markowitz) + CVaR (Expected Shortfall) via SLSQP
# Nessuna dipendenza pesante: funziona su Streamlit Cloud "as is".
# -------------------------------------------------

def _constraints(n: int, allow_short: bool, weight_cap: float | None):
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
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
    Simple MV optimizer via SLSQP. mode ∈ {"max_sharpe","min_vol","target"}.
    """
    rets = returns.dropna(how="any")
    mu = rets.mean() * periods_per_year
    cov = rets.cov() * periods_per_year
    n = rets.shape[1]

    cons, bounds = _constraints(n, allow_short, weight_cap)
    w0 = np.repeat(1 / n, n)

    if mode == "min_vol":
        obj = lambda w: np.sqrt(np.maximum(w @ cov.values @ w, 0.0))
    elif mode == "target":
        if target_return is None:
            raise ValueError("target_return required for mode='target'")
        cons = cons + [{"type": "eq", "fun": lambda w, mu=mu: float(w @ mu.values) - target_return}]
        obj = lambda w: np.sqrt(np.maximum(w @ cov.values @ w, 0.0))
    else:  # max_sharpe
        def neg_sharpe(w):
            var = np.maximum(w @ cov.values @ w, 0.0)
            sig = np.sqrt(var)
            mu_p = float(w @ mu.values)
            return - (mu_p - rf) / (sig + 1e-12)
        obj = neg_sharpe

    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-8})

    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    w = res.x
    mu_p = float(w @ mu.values)
    vol_p = float(np.sqrt(np.maximum(w @ cov.values @ w, 0.0)))
    sh = (mu_p - rf) / (vol_p + 1e-12)
    return {"weights": w, "mu": mu_p, "sigma": vol_p, "sharpe": sh, "success": res.success}

def _portfolio_returns(returns: pd.DataFrame, w: np.ndarray) -> np.ndarray:
    R = returns.dropna(how="any").values
    return R @ w

def _expected_shortfall(port_ret: np.ndarray, alpha: float = 0.95) -> float:
    """
    CVaR/ES of returns: positive = loss. We minimize ES => more conservative.
    """
    q = np.quantile(port_ret, 1 - alpha)
    tail = port_ret[port_ret <= q]
    if tail.size == 0:
        return -q
    return -float(tail.mean())

def cvar_opt(returns: pd.DataFrame, alpha: float = 0.95,
             allow_short: bool = False, weight_cap: float | None = 0.3) -> dict:
    """
    Minimize Expected Shortfall (historical) using SLSQP.
    """
    rets = returns.dropna(how="any")
    n = rets.shape[1]
    cons, bounds = _constraints(n, allow_short, weight_cap)
    w0 = np.repeat(1 / n, n)

    def obj(w):
        pr = _portfolio_returns(rets, w)
        return _expected_shortfall(pr, alpha=alpha)

    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-8})

    if not res.success:
        raise RuntimeError(f"CVaR optimization failed: {res.message}")

    w = res.x
    es = _expected_shortfall(_portfolio_returns(rets, w), alpha=alpha)
    return {"weights": w, "es": es, "success": res.success}
