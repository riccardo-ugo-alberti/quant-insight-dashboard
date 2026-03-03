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

<<<<<<< ours
    res = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 1000})
    if not res.success:
        raise RuntimeError(
            "Optimization failed "
            f"(mode={mode}, shorting={shorting}, max_weight={max_weight:.4f}, "
            f"target_return={target_return}, status={res.status}): {res.message}"
        )

    w = res.x
    ret, vol, sr = _portfolio_stats(w, mu, cov, rf, mu_vals=mu_vals, cov_vals=cov_vals)
=======
    starts = [w0]
    if mode == "min_vol":
        rng = np.random.default_rng(7)
        for _ in range(6):
            v = rng.random(n)
            starts.append(v / v.sum())

    best_res = None
    for w_start in starts:
        cand = minimize(obj, w_start, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 2000, "ftol": 1e-12})
        if not cand.success:
            continue
        if best_res is None or cand.fun < best_res.fun:
            best_res = cand

    if best_res is None:
        raise RuntimeError("Optimization failed: no feasible solution")

    w = best_res.x
    ret, vol, sr = _portfolio_stats(w, mu, cov, rf)
>>>>>>> theirs
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
<<<<<<< ours
=======
# =============================================================================
# COMPATIBILITY SHIM – garantisce coerenza dei nomi tra versioni diverse
# =============================================================================
import inspect
import pandas as pd



class CompatResult(dict):
    """Dict-like result with attribute access for backward compatibility."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _as_compat_result_from_mv(out):
    if isinstance(out, CompatResult):
        return out
    if isinstance(out, tuple) and len(out) >= 2:
        weights = out[0]
        stats = out[1] if isinstance(out[1], tuple) and len(out[1]) >= 3 else (np.nan, np.nan, np.nan)
        exp_return, exp_vol, sharpe = float(stats[0]), float(stats[1]), float(stats[2])
        data = CompatResult(
            weights=weights,
            exp_return=exp_return,
            exp_vol=exp_vol,
            sharpe=sharpe,
            ret=exp_return,
            vol=exp_vol,
            message="ok",
        )
        data["return"] = exp_return
        if len(out) >= 3:
            data["mv_inputs"] = out[2]
        return data
    tmp = CompatResult(weights=out, exp_return=np.nan, exp_vol=np.nan, sharpe=np.nan, ret=np.nan, vol=np.nan, message="ok")
    tmp["return"] = np.nan
    return tmp


def _as_compat_result_from_cvar(out, prices=None, message="ok"):
    weights = out
    exp_return = np.nan
    exp_vol = np.nan
    if isinstance(weights, pd.Series) and prices is not None and not prices.empty:
        rets = prices.pct_change().dropna(how="all")
        if not rets.empty:
            mu = rets.mean() * 252.0
            cov = rets.cov() * 252.0
            w = weights.reindex(mu.index).fillna(0.0).values
            exp_return = float(np.dot(w, mu.values))
            exp_vol = float(np.sqrt(np.dot(w, cov.values @ w)))
    tmp = CompatResult(weights=weights, exp_return=exp_return, exp_vol=exp_vol, sharpe=np.nan, ret=exp_return, vol=exp_vol, message=message)
    tmp["return"] = exp_return
    return tmp
def _prices_from_returns(rets: pd.DataFrame) -> pd.DataFrame:
    """Build a synthetic price path from returns for compatibility wrappers."""
    if rets is None or rets.empty:
        raise ValueError("returns cannot be empty")
    clean = rets.dropna(how="all").fillna(0.0)
    return (1.0 + clean).cumprod()


def _extract_returns_or_prices(kwargs):
    prices = kwargs.get("prices")
    returns = kwargs.get("returns")
    if prices is None and returns is not None:
        prices = _prices_from_returns(returns)
    return prices, returns


def _compat_call(func, **kwargs):
    """
    Chiama una funzione provando ad adattare i nomi degli argomenti.
    Esempio: allow_short -> allow_shorting, max_weight -> weight_cap, n_points -> n_portfolios, ecc.
    """
    sig = inspect.signature(func)
    valid = set(sig.parameters.keys())
    alt = {
        "allow_short": "shorting",
<<<<<<< ours
        "max_weight": "max_weight",
        "n_points": "points",
        "prices": "px_df",
        "returns": "rets",
=======
        "n_points": "points",
>>>>>>> theirs
    }

    # Mappa i nomi noti
    fixed_kwargs = {}
    for k, v in kwargs.items():
        if k in valid:
            fixed_kwargs[k] = v
        elif k in alt and alt[k] in valid:
            fixed_kwargs[alt[k]] = v
        # ignora quelli sconosciuti

    return func(**fixed_kwargs)


# ---------------------------------------------------------------------------
# Wrappers di sicurezza: garantiscono che le chiamate non vadano in errore
# ---------------------------------------------------------------------------
try:
    if "optimize_portfolio" in globals():
        _orig_optimize_portfolio = optimize_portfolio

        def optimize_portfolio(**kwargs):
            prices, _ = _extract_returns_or_prices(kwargs)
            if prices is not None:
                kwargs["prices"] = prices
            try:
                return _as_compat_result_from_mv(_compat_call(_orig_optimize_portfolio, **kwargs))
            except TypeError:
                # fallback diretto, se non serve compat
                direct = {k: v for k, v in kwargs.items() if k != "returns"}
                return _as_compat_result_from_mv(_orig_optimize_portfolio(**direct))
except Exception:
    pass


try:
    if "compute_frontier" in globals():
        _orig_compute_frontier = compute_frontier

        def compute_frontier(*args, **kwargs):
<<<<<<< ours
            # UI/backward-compat: compute_frontier(prices_df, rf, n_points, ...)
            if args and isinstance(args[0], pd.DataFrame):
                first = args[0]
                if len(args) < 2 or not isinstance(args[1], pd.DataFrame):
                    rf = float(args[1]) if len(args) > 1 else float(kwargs.pop("rf", 0.0))
                    points = int(args[2]) if len(args) > 2 else int(kwargs.pop("n_points", kwargs.pop("points", 25)))
                    shorting = bool(kwargs.pop("allow_short", kwargs.pop("shorting", False)))
                    max_weight = float(kwargs.pop("max_weight", 0.30))
                    return frontier_from_prices(
                        prices=first,
                        rf=rf,
                        points=points,
                        shorting=shorting,
                        max_weight=max_weight,
                    )

=======
            prices, returns = _extract_returns_or_prices(kwargs)
            if returns is not None and ("mu" not in kwargs and "cov" not in kwargs):
                kwargs["mu"] = _annualize_mean(returns.mean())
                kwargs["cov"] = _annualize_cov(returns.cov())
            elif prices is not None and ("mu" not in kwargs and "cov" not in kwargs):
                mv = prepare_mv_inputs(prices)
                kwargs["mu"] = mv.mu
                kwargs["cov"] = mv.cov
>>>>>>> theirs
            try:
                res = _compat_call(_orig_compute_frontier, **kwargs)
            except TypeError:
                direct = {k: v for k, v in kwargs.items() if k not in {"returns", "prices"}}
                res = _orig_compute_frontier(*args, **direct)

            # Assicura sempre un DataFrame come output
            if isinstance(res, (tuple, list)) and any(isinstance(x, pd.DataFrame) for x in res):
                for x in res:
                    if isinstance(x, pd.DataFrame):
                        return x
            if isinstance(res, dict):
                for v in res.values():
                    if isinstance(v, pd.DataFrame):
                        return v
            if isinstance(res, pd.DataFrame):
                return res
            return pd.DataFrame()
except Exception:
    pass
>>>>>>> theirs


<<<<<<< ours
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
=======
try:
    if "optimize_cvar" in globals():
        _orig_optimize_cvar = optimize_cvar

        def optimize_cvar(**kwargs):
            prices, _ = _extract_returns_or_prices(kwargs)
            if prices is not None:
                kwargs["prices"] = prices
            try:
                return _as_compat_result_from_cvar(_compat_call(_orig_optimize_cvar, **kwargs), prices=kwargs.get("prices"))
            except TypeError:
                direct = {k: v for k, v in kwargs.items() if k != "returns"}
                return _as_compat_result_from_cvar(_orig_optimize_cvar(**direct), prices=direct.get("prices"))
except Exception:
    pass
>>>>>>> theirs
