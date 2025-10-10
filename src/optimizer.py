# src/optimizer.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Iterable, Dict, Tuple, Optional
from scipy.optimize import minimize

TRADING_DAYS = 252

@dataclass
class OptResult:
    weights: pd.Series
    ann_return: float
    ann_vol: float
    sharpe: float

# -------------------- utilities --------------------
def annualize_returns_and_cov(rets: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    mu = rets.mean() * TRADING_DAYS
    cov = rets.cov() * TRADING_DAYS
    return mu, cov

def _regularize_cov(cov: pd.DataFrame, floor: float = 1e-8, ridge: float = 1e-6) -> pd.DataFrame:
    """Rende PSD e ben condizionata la matrice di covarianza."""
    vals, vecs = np.linalg.eigh(cov.values)
    vals = np.clip(vals, floor, None)
    cov_psd = (vecs @ np.diag(vals) @ vecs.T)
    cov_psd += ridge * np.eye(cov_psd.shape[0])
    cov_psd = (cov_psd + cov_psd.T) * 0.5
    return pd.DataFrame(cov_psd, index=cov.index, columns=cov.columns)

def _bounds(n: int, allow_short: bool, w_max: float) -> list[tuple[float, float]]:
    if allow_short:
        return [(-1.0, 1.0)] * n
    return [(0.0, float(min(1.0, w_max)))] * n

def _sector_constraints(tickers: Iterable[str], sectors: dict[str, str] | None, sector_cap: float | None):
    """I vincoli settoriali: somma pesi di ciascun settore <= cap."""
    cons = []
    if not sectors or sector_cap is None:
        return tuple()
    tickers = list(tickers)
    sec = pd.Series([sectors.get(t, "Other") for t in tickers], index=range(len(tickers)))
    for s, idxs in sec.groupby(sec).groups.items():
        idx = np.fromiter(idxs, dtype=int)
        def f(w, idx=idx, cap=sector_cap):  # cap - sum(w_idx) >= 0
            return float(cap - np.sum(w[idx]))
        cons.append({"type": "ineq", "fun": f})
    return tuple(cons)

def _project_simplex_with_cap(w: np.ndarray, cap: float) -> np.ndarray:
    """
    Proietta su {w_i >= 0, sum w = 1, w_i <= cap}. Algoritmo semplice:
    cap first, poi proiezione su simplex.
    """
    w = np.maximum(w, 0)
    if cap < 1.0:
        w = np.minimum(w, cap)
    s = w.sum()
    if s <= 0:
        w = np.ones_like(w) / len(w)
        s = 1.0
    w = w / s
    # se qualche componente eccede cap dopo normalizzazione, iteriamo
    for _ in range(10):
        over = w > cap
        if not np.any(over) or cap >= 1.0:
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        w[~over] = np.maximum(w[~over] - excess * w[~over].sum() / max(w[~over].sum(), 1e-12), 0)
        w = w / max(w.sum(), 1e-12)
    return w

def portfolio_stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> tuple[float, float, float]:
    ret = float(w @ mu)
    var = float(w @ cov @ w)
    var = max(var, 0.0)
    vol = float(np.sqrt(var))
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return ret, vol, sharpe

# -------------------- core solvers (robusti) --------------------
def _solve_internal(mu: pd.Series, cov: pd.DataFrame, rf: float, objective: str,
                    allow_short: bool, w_max: float,
                    sectors: dict[str, str] | None = None, sector_cap: float | None = None) -> OptResult:

    n = len(mu)
    mu_v, cov_m = mu.values, cov.values
    bounds = _bounds(n, allow_short, w_max)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    cons += list(_sector_constraints(mu.index, sectors, sector_cap))

    # objective
    if objective == "min_vol":
        def obj(w):  # ridge L2 per aiutare il linesearch
            return float(w @ cov_m @ w + 1e-8 * (w @ w))
    elif objective == "max_sharpe":
        def obj(w):
            r, v, _ = portfolio_stats(w, mu_v, cov_m, rf)
            return -(r - rf) / (v + 1e-12)
    else:
        raise ValueError("unknown objective")

    # warm starts
    starts = []
    # equal weight proiettato
    if allow_short:
        starts.append(np.ones(n) / n)
    else:
        starts.append(_project_simplex_with_cap(np.ones(n) / n, w_max))
    # 2 dirichlet random
    rng = np.random.default_rng(123)
    for _ in range(2):
        s = rng.dirichlet(np.ones(n))
        if not allow_short:
            s = _project_simplex_with_cap(s, w_max)
        starts.append(s)

    last_err = None
    for ridge in [1e-6, 5e-6, 1e-5]:
        cov_reg = cov_m + ridge * np.eye(n)
        for w0 in starts:
            try:
                res = minimize(
                    obj, w0, method="SLSQP",
                    bounds=bounds, constraints=tuple(cons),
                    options={"maxiter": 600, "ftol": 1e-12, "eps": 1e-8}
                )
                if res.success and np.isfinite(res.fun):
                    w = res.x
                    r, v, s = portfolio_stats(w, mu_v, cov_reg, rf)
                    return OptResult(pd.Series(w, index=mu.index), r, v, s)
                last_err = res.message
            except Exception as e:
                last_err = str(e)

    # fallback: equal weight fattibile
    if allow_short:
        w = np.ones(n) / n
    else:
        w = _project_simplex_with_cap(np.ones(n) / n, w_max)
    r, v, s = portfolio_stats(w, mu_v, cov_m, rf)
    return OptResult(pd.Series(w, index=mu.index), r, v, s)

def _achievable_return_range(mu: pd.Series, allow_short: bool, w_max: float,
                             sectors: dict[str, str] | None, sector_cap: float | None) -> tuple[float, float]:
    """Range approssimato di rendimento annuo fattibile coi vincoli."""
    # max return = peso massimo su titolo con mu più alto (rispettando cap/settori) – euristica
    mu_sorted = mu.sort_values(ascending=False)
    n = len(mu)
    if allow_short:
        return (mu.min(), mu.max())
    # packing greedy con cap
    max_ret = 0.0
    w_left = 1.0
    sector_w = {}
    for t, m in mu_sorted.items():
        add = min(w_max, w_left)
        if sectors and sector_cap is not None:
            s = sectors.get(t, "Other")
            already = sector_w.get(s, 0.0)
            add = min(add, sector_cap - already)
            if add <= 0:
                continue
            sector_w[s] = already + add
        max_ret += add * m
        w_left -= add
        if w_left <= 1e-12:
            break
    # min return ~ tutto sul peggiore
    min_ret = mu.sort_values(ascending=True).iloc[0] if not allow_short else mu.min()
    return (float(min_ret), float(max_ret))

def solve_target_return(mu: pd.Series, cov: pd.DataFrame, target_ret: float,
                        allow_short: bool, w_max: float,
                        sectors: dict[str, str] | None = None, sector_cap: float | None = None) -> pd.Series:
    """Min var dato ritorno target (con clamp del target nel range fattibile)."""
    mu_v, cov_m = mu.values, cov.values
    n = len(mu)
    bounds = _bounds(n, allow_short, w_max)

    # clamp target
    lo, hi = _achievable_return_range(mu, allow_short, w_max, sectors, sector_cap)
    target_ret = float(np.clip(target_ret, lo, hi))

    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w, tr=target_ret: float(w @ mu_v) - tr},
    ]
    cons += list(_sector_constraints(mu.index, sectors, sector_cap))

    def obj(w): return float(w @ cov_m @ w + 1e-8 * (w @ w))

    # warm start
    if allow_short:
        w0 = np.ones(n) / n
    else:
        w0 = _project_simplex_with_cap(np.ones(n) / n, w_max)

    last_err = None
    for ridge in [1e-6, 5e-6, 1e-5]:
        cov_reg = cov_m + ridge * np.eye(n)
        try:
            res = minimize(obj, w0, method="SLSQP",
                           bounds=bounds, constraints=tuple(cons),
                           options={"maxiter": 600, "ftol": 1e-12, "eps": 1e-8})
            if res.success and np.isfinite(res.fun):
                return pd.Series(res.x, index=mu.index)
            last_err = res.message
        except Exception as e:
            last_err = str(e)

    # fallback: proporzionale a mu positiva (cap + normalizza)
    w = np.maximum(mu_v, 0)
    if w.sum() <= 0:
        w = np.ones(n) / n
    w = w / w.sum()
    if not allow_short:
        w = _project_simplex_with_cap(w, w_max)
    return pd.Series(w, index=mu.index)

def optimize_portfolios(rets: pd.DataFrame, rf: float,
                        allow_short: bool, w_max: float,
                        sectors: dict[str, str] | None = None, sector_cap: float | None = None) -> dict[str, OptResult]:
    mu, cov = annualize_returns_and_cov(rets.dropna(how="any"))
    cov = _regularize_cov(cov)
    return {
        "max_sharpe": _solve_internal(mu, cov, rf, "max_sharpe", allow_short, w_max, sectors, sector_cap),
        "min_vol":    _solve_internal(mu, cov, rf, "min_vol",    allow_short, w_max, sectors, sector_cap),
    }

def efficient_frontier(rets: pd.DataFrame, rf: float,
                       allow_short: bool, w_max: float,
                       sectors: dict[str, str] | None = None, sector_cap: float | None = None,
                       points: int = 25) -> pd.DataFrame:
    mu, cov = annualize_returns_and_cov(rets.dropna(how="any"))
    cov = _regularize_cov(cov)
    lo, hi = _achievable_return_range(mu, allow_short, w_max, sectors, sector_cap)
    targets = np.linspace(lo, hi, points)

    vols = []
    for tr in targets:
        try:
            w = solve_target_return(mu, cov, float(tr), allow_short, w_max, sectors, sector_cap)
            v = float(np.sqrt(max(w.values @ cov.values @ w.values, 0.0)))
        except Exception:
            v = np.nan
        vols.append(v)

    out = pd.DataFrame({"target_ret": targets, "vol": vols})
    out["sharpe"] = (out["target_ret"] - rf) / out["vol"]
    return out.dropna()

# -------------------- backtest con turnover & costi --------------------
def backtest_rebalance(prices: pd.DataFrame, rets: pd.DataFrame, rf: float,
                        mode: str, allow_short: bool, w_max: float,
                        sectors: dict[str, str] | None, sector_cap: float | None,
                        rebalance: str = "M", target_ret: float | None = None,
                        turnover_limit: float = 0.5, tc_bps: float = 5.0
                        ) -> tuple[pd.Series, pd.DataFrame]:

    mu_full, cov_full = annualize_returns_and_cov(rets)
    cov_full = _regularize_cov(cov_full)

    if mode == "target" and target_ret is None:
        target_ret = float(np.median(mu_full.values))

    dates = prices.index.intersection(rets.index)
    if len(dates) < 2:
        raise ValueError("Not enough data for backtest.")

    rb_mask = pd.Series(False, index=dates)
    rb_mask.loc[dates.to_period(rebalance).to_timestamp().unique().intersection(dates)] = True
    rb_mask.iloc[0] = True

    tickers = list(rets.columns)
    n = len(tickers)
    w = np.ones(n) / n if allow_short else _project_simplex_with_cap(np.ones(n) / n, w_max)
    W = []
    equity = [1.0]

    for i in range(1, len(dates)):
        d = dates[i]
        r = rets.loc[d].values
        equity.append(equity[-1] * (1.0 + float(w @ r)))

        if rb_mask.loc[d]:
            hist = rets.loc[:d].dropna()
            mu, cov = annualize_returns_and_cov(hist)
            cov = _regularize_cov(cov)

            if mode == "max_sharpe":
                res = _solve_internal(mu, cov, rf, "max_sharpe", allow_short, w_max, sectors, sector_cap)
                w_star = res.weights.values
            elif mode == "min_vol":
                res = _solve_internal(mu, cov, rf, "min_vol", allow_short, w_max, sectors, sector_cap)
                w_star = res.weights.values
            else:
                w_star = solve_target_return(mu, cov, float(target_ret), allow_short, w_max, sectors, sector_cap).values

            diff = w_star - w
            l1 = float(np.sum(np.abs(diff)))
            alpha = 1.0 if l1 <= 1e-12 else min(1.0, float(turnover_limit) / l1)
            w_new = w + alpha * diff
            if not allow_short:
                w_new = np.clip(w_new, 0.0, 1.0)
                w_new = _project_simplex_with_cap(w_new, w_max)
            else:
                w_new = w_new / max(np.sum(w_new), 1e-12)

            tc = float(np.sum(np.abs(w_new - w))) * (tc_bps / 10000.0)
            equity[-1] *= (1.0 - tc)
            w = w_new

        W.append(pd.Series(w, index=tickers, name=d))

    equity_series = pd.Series(equity, index=dates, name="Equity")
    weights_df = pd.DataFrame(W)
    return equity_series, weights_df
