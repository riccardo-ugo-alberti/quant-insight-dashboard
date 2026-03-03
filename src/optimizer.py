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


def _validate_covariance_psd(cov_vals: np.ndarray, tol: float = 1e-10) -> None:
    eigvals = np.linalg.eigvalsh(cov_vals)
    min_eig = float(np.min(eigvals)) if eigvals.size else 0.0
    if min_eig < -tol:
        raise ValueError(
            f"Covariance matrix is not PSD (min eigenvalue={min_eig:.3e}). "
            "Check input data quality or apply covariance regularization."
        )


def _feasible_return_bounds(mu_vals: np.ndarray, lo: float, hi: float) -> Tuple[float, float]:
    order_desc = np.argsort(-mu_vals)
    order_asc = np.argsort(mu_vals)

    def _extreme(idx_order: np.ndarray) -> float:
        w = np.full(mu_vals.shape[0], lo, dtype=float)
        remaining = 1.0 - float(np.sum(w))
        for i in idx_order:
            if remaining <= 0:
                break
            room = hi - w[i]
            if room <= 0:
                continue
            add = min(room, remaining)
            w[i] += add
            remaining -= add
        if remaining > 1e-10:
            raise ValueError(
                f"Weight constraints infeasible: sum(w)=1 cannot be satisfied with bounds [{lo:.3f}, {hi:.3f}]."
            )
        return float(np.dot(w, mu_vals))

    r_max = _extreme(order_desc)
    r_min = _extreme(order_asc)
    return r_min, r_max


@dataclass
class MVInputs:
    """Inputs for mean-variance."""
    mu: pd.Series          # annualized expected returns
    cov: pd.DataFrame      # annualized covariance
    tickers: List[str]


def _mv_inputs_from_returns(returns: pd.DataFrame) -> MVInputs:
    """Build MV inputs from a daily returns DataFrame."""
    rets = returns.dropna(how="all")
    if rets.empty:
        raise ValueError("No return observations available after dropping NaNs.")

    mu_d = rets.mean()
    cov_d = rets.cov()
    mu = _annualize_mean(mu_d)
    cov = _annualize_cov(cov_d)

    ok = cov.columns[(np.diag(cov.values) > 0)]
    if len(ok) == 0:
        raise ValueError("No assets with positive variance available for optimization.")

    mu = mu.loc[ok]
    cov = cov.loc[ok, ok]
    return MVInputs(mu=mu, cov=cov, tickers=list(ok))


def prepare_mv_inputs(prices: pd.DataFrame) -> MVInputs:
    """
    From a price DataFrame (index: date, cols: tickers) create annualized
    expected returns and covariance for MV optimization.
    """
    rets = prices.pct_change().dropna(how="all")
    return _mv_inputs_from_returns(rets)


def prepare_mv_inputs_from_returns(returns: pd.DataFrame) -> MVInputs:
    """Explicit variant of MV input preparation for daily returns."""
    return _mv_inputs_from_returns(returns)


def _portfolio_stats(
    w: np.ndarray,
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float = 0.0,
    mu_vals: Optional[np.ndarray] = None,
    cov_vals: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Return (exp_return, vol, sharpe). rf in decimal, annualized."""
    w = _to_numpy(w)
    mu_vals = mu.values if mu_vals is None else mu_vals
    cov_vals = cov.values if cov_vals is None else cov_vals
    exp_return = float(np.dot(w, mu_vals))
    vol = float(np.sqrt(w @ cov_vals @ w))
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
    mu_vals: Optional[np.ndarray] = None,
    cov_vals: Optional[np.ndarray] = None,
) -> Tuple[pd.Series, Tuple[float, float, float]]:
    """
    Core MV optimizer (SLSQP). Returns (weights Series, (ret, vol, sharpe)).
    """
    n = len(mu)
    if n == 0:
        raise ValueError("No assets available after preprocessing.")

    if not np.isfinite(max_weight) or max_weight <= 0:
        raise ValueError(f"max_weight must be > 0. Got {max_weight}.")

    mu_vals = mu.values if mu_vals is None else mu_vals
    cov_vals = cov.values if cov_vals is None else cov_vals
    _validate_covariance_psd(cov_vals)

    lo = -max_weight if shorting else 0.0
    hi = max_weight

    if not shorting and max_weight * n < 1.0 - 1e-12:
        raise ValueError(
            f"Infeasible long-only constraints: n_assets={n}, max_weight={max_weight:.4f}, "
            f"requires at least {1.0 / n:.4f}."
        )

    min_feasible_ret, max_feasible_ret = _feasible_return_bounds(mu_vals, lo=lo, hi=hi)

    bnds = [(lo, hi) for _ in range(n)]
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if mode == "target_return":
        if target_return is None:
            target_return = float(mu.mean())
        if target_return < min_feasible_ret - 1e-8 or target_return > max_feasible_ret + 1e-8:
            raise ValueError(
                "target_return is infeasible under current constraints: "
                f"target={target_return:.6f}, feasible=[{min_feasible_ret:.6f}, {max_feasible_ret:.6f}], "
                f"shorting={shorting}, max_weight={max_weight:.4f}."
            )
        cons.append({"type": "eq", "fun": lambda w, mu_vals=mu_vals, tr=target_return: np.dot(w, mu_vals) - tr})

        def obj(w):
            return float(w @ cov_vals @ w)

    elif mode == "min_vol":
        def obj(w):
            return float(w @ cov_vals @ w)

    else:  # max_sharpe
        def obj(w):
            ret, vol, _ = _portfolio_stats(w, mu, cov, rf, mu_vals=mu_vals, cov_vals=cov_vals)
            return -((ret - rf) / vol if vol > 0 else -1e6)

    if w0 is None:
        w0 = np.ones(n) / n

    starts = [w0]
    if mode == "min_vol":
        rng = np.random.default_rng(7)
        for _ in range(6):
            v = rng.random(n)
            starts.append(v / v.sum())

    best_res = None
    last_fail = None
    for w_start in starts:
        cand = minimize(
            obj,
            w_start,
            method="SLSQP",
            bounds=bnds,
            constraints=cons,
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        if not cand.success:
            last_fail = cand
            continue
        if best_res is None or cand.fun < best_res.fun:
            best_res = cand

    if best_res is None:
        fail_msg = (
            f"status={last_fail.status}, message={last_fail.message}"
            if last_fail is not None
            else "no candidate solution"
        )
        raise RuntimeError(
            "Optimization failed "
            f"(mode={mode}, shorting={shorting}, max_weight={max_weight:.4f}, "
            f"target_return={target_return}, {fail_msg})"
        )

    w = best_res.x
    ret, vol, sr = _portfolio_stats(w, mu, cov, rf, mu_vals=mu_vals, cov_vals=cov_vals)
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
    if points < 2:
        raise ValueError(f"points must be >= 2. Got {points}.")

    mu_vals = mu.values
    cov_vals = cov.values
    min_tr = float(np.percentile(mu_vals, 10))
    max_tr = float(np.percentile(mu_vals, 90))
    targets = np.linspace(min_tr, max_tr, points)

    rows = []
    w0: Optional[np.ndarray] = None
    for tr in targets:
        try:
            weights, stats = _mv_optimize(
                mu,
                cov,
                rf,
                mode="target_return",
                target_return=float(tr),
                shorting=shorting,
                max_weight=max_weight,
                w0=w0,
                mu_vals=mu_vals,
                cov_vals=cov_vals,
            )
            w0 = weights.values
            ret, vol, sr = stats
            rows.append((ret, vol, sr))
        except Exception:
            continue

    if not rows:
        try:
            _, stats = _mv_optimize(
                mu,
                cov,
                rf,
                mode="max_sharpe",
                shorting=shorting,
                max_weight=max_weight,
                mu_vals=mu_vals,
                cov_vals=cov_vals,
            )
            ret, vol, sr = stats
            rows = [(ret, vol, sr)]
        except Exception as exc:
            raise RuntimeError(
                "Frontier computation failed: no feasible target-return point and max_sharpe fallback failed. "
                f"points={points}, shorting={shorting}, max_weight={max_weight:.4f}."
            ) from exc

    return pd.DataFrame(rows, columns=["Return", "Vol", "Sharpe"])


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


def optimize_portfolio_from_returns(
    returns: pd.DataFrame,
    rf: float = 0.0,
    mode: Literal["max_sharpe", "min_vol", "target_return"] = "max_sharpe",
    target_return: Optional[float] = None,
    shorting: bool = False,
    max_weight: float = 0.30,
) -> Tuple[pd.Series, Tuple[float, float, float], MVInputs]:
    """Explicit MV optimizer entrypoint for daily returns input."""
    mv = prepare_mv_inputs_from_returns(returns)
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
    R = rets.values

    w = cp.Variable(n)
    z = cp.Variable(T)
    var = cp.Variable(1)

    lo = -max_weight if shorting else 0.0
    hi = max_weight
    constraints = [
        cp.sum(w) == 1,
        w >= lo,
        w <= hi,
        z >= 0,
        z >= -(R @ w) - var,
    ]
    objective = cp.Minimize(var + (1.0 / ((1.0 - alpha) * T)) * cp.sum(z))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=False, max_iters=2000)

    if w.value is None:
        return None

    wv = np.asarray(w.value).ravel()
    weights = pd.Series(wv, index=rets.columns, name="weight")
    return weights




def optimize_cvar_from_returns(
    returns: pd.DataFrame,
    alpha: float = 0.95,
    shorting: bool = False,
    max_weight: float = 0.30,
) -> Optional[pd.Series]:
    """Explicit CVaR optimizer entrypoint for daily returns input."""
    if returns.empty:
        return None
    prices_proxy = (1.0 + returns).cumprod()
    return optimize_cvar(prices=prices_proxy, alpha=alpha, shorting=shorting, max_weight=max_weight)


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

def frontier_from_returns(
    returns: pd.DataFrame,
    rf: float = 0.0,
    points: int = 25,
    shorting: bool = False,
    max_weight: float = 0.30,
) -> pd.DataFrame:
    """Explicit efficient frontier entrypoint for daily returns input."""
    mv = prepare_mv_inputs_from_returns(returns)
    return compute_frontier(mv.mu, mv.cov, rf=rf, points=points, shorting=shorting, max_weight=max_weight)

