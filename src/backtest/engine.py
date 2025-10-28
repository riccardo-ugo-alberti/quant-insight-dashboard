# src/backtest/engine.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import BacktestConfig, EstimatorConfig, MVConfig, CostConfig, RebalanceConfig, ShrinkageConfig


# =============================================================================
# Utility
# =============================================================================
def _to_numpy(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _annualize(mu_daily: pd.Series, cov_daily: pd.DataFrame, freq: int = 252) -> Tuple[pd.Series, pd.DataFrame]:
    mu = mu_daily * freq
    cov = cov_daily * freq
    return mu, cov


def _rolling_mean_cov(returns: pd.DataFrame, end_idx: int, window: int) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Stima rolling sui giorni [end_idx - window, end_idx) (escluso end_idx).
    """
    start = max(0, end_idx - window)
    r = returns.iloc[start:end_idx]
    mu_d = r.mean()
    cov_d = r.cov()
    return mu_d, cov_d


def _ewma_mean_cov(returns: pd.DataFrame, end_idx: int, window: int, lam: float) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Stima EWMA su [end_idx - window, end_idx) con pesi λ^k (k=0 per il giorno più recente).
    """
    start = max(0, end_idx - window)
    r = returns.iloc[start:end_idx]
    if r.empty:
        return r.mean(), r.cov()

    n = len(r)
    w = np.array([lam ** (n - 1 - i) for i in range(n)], dtype=float)
    w = w / w.sum()

    # media pesata
    mu = pd.Series((r.values * w[:, None]).sum(axis=0), index=r.columns)

    # cov pesata (biased, ma stabile per il backtest)
    x = r.values - mu.values  # broadcast
    cov = (w[:, None, None] * (x[:, :, None] * x[:, None, :])).sum(axis=0)
    cov = pd.DataFrame(cov, index=r.columns, columns=r.columns)
    return mu, cov


def _const_cor_target(cov: pd.DataFrame) -> pd.DataFrame:
    """
    Bersaglio 'constant-correlation': varianze originali e unica correlazione media.
    """
    std = np.sqrt(np.diag(cov.values))
    if np.any(std == 0):
        # fallback: identità
        return np.eye(cov.shape[0]) * np.mean(np.diag(cov.values) + 1e-12)

    Dinv = np.diag(1.0 / std)
    corr = Dinv @ cov.values @ Dinv
    # media delle correlazioni off-diagonal
    m = corr.shape[0]
    rho_bar = (corr.sum() - np.trace(corr)) / (m * (m - 1) + 1e-12)
    target_corr = np.full_like(corr, rho_bar)
    np.fill_diagonal(target_corr, 1.0)
    target_cov = np.diag(std) @ target_corr @ np.diag(std)
    return pd.DataFrame(target_cov, index=cov.index, columns=cov.columns)


def _apply_shrinkage(cov: pd.DataFrame, method: str, gamma: float) -> pd.DataFrame:
    if gamma <= 0 or method == "none":
        return cov.copy()

    if method == "diag":
        target = np.diag(np.diag(cov.values))
    elif method == "identity":
        avg_var = float(np.mean(np.diag(cov.values)))
        target = np.eye(cov.shape[0]) * avg_var
    elif method == "const-cor":
        target = _const_cor_target(cov).values
    else:
        target = cov.values  # fallback

    shrunk = (1.0 - gamma) * cov.values + gamma * target
    return pd.DataFrame(shrunk, index=cov.index, columns=cov.columns)


def _mv_optimize(mu: pd.Series,
                 cov: pd.DataFrame,
                 prev_w: Optional[np.ndarray],
                 mv: MVConfig) -> np.ndarray:
    """
    Mean-Variance con penalità L2 su turnover rispetto a prev_w.
    Obiettivo (da minimizzare):
        f(w) = 0.5*γ * w'Σw - μ'w + λ * ||w - prev_w||^2
    vincoli: sum w = 1; bounds dipende da allow_short.
    """
    n = len(mu)
    mu_v = mu.values
    C = cov.values + np.eye(n) * float(max(0.0, mv.ridge))

    # bounds
    if mv.allow_short:
        bnds = [(-mv.leverage, mv.leverage) for _ in range(n)]
    else:
        bnds = [(0.0, mv.leverage) for _ in range(n)]

    # vincolo somma 1
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # init
    if prev_w is None:
        w0 = np.ones(n) / n
    else:
        w0 = prev_w.copy()

    # obiettivo
    def obj(w):
        w = _to_numpy(w)
        quad = 0.5 * mv.risk_aversion * (w @ C @ w)
        lin = - np.dot(w, mu_v)
        turn = 0.0
        if prev_w is not None and mv.turnover_L2 > 0:
            d = w - prev_w
            turn = mv.turnover_L2 * float(np.dot(d, d))
        return quad + lin + turn

    res = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 1000})
    if not res.success:
        # fallback: normalizza w0
        s = float(np.sum(np.clip(w0, bnds[0][0], bnds[0][1])))
        w = (w0 / s) if s != 0 else (np.ones(n) / n)
    else:
        w = res.x

    # normalizza hard alla somma 1 (robustezza)
    w = np.asarray(w, dtype=float)
    if np.sum(np.abs(w)) == 0:
        w = np.ones(n) / n
    w = w / np.sum(w)
    return w


def _turnover(prev_w: np.ndarray, new_w: np.ndarray) -> float:
    """
    Turnover come somma dei notional scambiati (L1 distance).
    """
    return float(np.sum(np.abs(new_w - prev_w)))


def _cost_from_turnover(turnover: float, costs: CostConfig) -> float:
    # costi in frazione del NAV
    bps = float(costs.proportional_bps) + float(costs.slippage_bps)
    return turnover * (bps / 10_000.0)


# =============================================================================
# Motore
# =============================================================================
@dataclass
class RollingEWMAEngine:
    """
    Motore di backtest dinamico su rendimenti giornalieri.
    - returns: DataFrame (index=Date, columns=tickers), valori giornalieri (non log).
    - cfg: BacktestConfig

    Output di run():
      {
        "nav":      DataFrame[['nav']],
        "turnover": DataFrame[['turnover']], ribilanciamenti
        "costs":    DataFrame[['cost']],     ribilanciamenti
        "weights":  DataFrame[tickers],      pesi giornalieri (post-ritorno e dopo rib.)
      }
    """
    returns: pd.DataFrame
    cfg: BacktestConfig

    def run(self, initial_nav: float = 1.0) -> Dict[str, pd.DataFrame]:
        r = self.returns.sort_index().dropna(how="all")
        if r.empty or r.shape[1] < 2:
            # output vuoto coerente
            idx = r.index if not r.empty else pd.Index([], name="Date")
            return {
                "nav": pd.DataFrame({"nav": []}, index=idx),
                "turnover": pd.DataFrame({"turnover": []}, index=idx),
                "costs": pd.DataFrame({"cost": []}, index=idx),
                "weights": pd.DataFrame(columns=[]),
            }

        est: EstimatorConfig = self.cfg.estimator
        mv: MVConfig = self.cfg.mv
        costs: CostConfig = self.cfg.costs
        reb: RebalanceConfig = self.cfg.rebalance

        dates = r.index
        n = r.shape[1]
        tickers = list(r.columns)

        nav_vals = []
        w_hist = np.zeros((len(dates), n), dtype=float)

        nav = float(initial_nav)
        w_prev: Optional[np.ndarray] = None  # pesi “attivi” per il giorno t (prima del ritorno)
        last_reb_idx: Optional[int] = None

        turnovers = {}
        costs_rec = {}

        for t in range(len(dates)):
            # 1) decidi se ribilanciare all'inizio del giorno t
            do_reb = False
            if t >= est.window:  # serve abbastanza storia
                if (t - reb.offset) % max(1, reb.every_k_days) == 0:
                    do_reb = True

            # 2) se ribilanci e hai storia, stima mu e cov
            if do_reb:
                if est.use_ewma:
                    mu_d, cov_d = _ewma_mean_cov(r, t, est.window, est.ewma_lambda)
                else:
                    mu_d, cov_d = _rolling_mean_cov(r, t, est.window)

                # annualizza
                mu_ann, cov_ann = _annualize(mu_d, cov_d, freq=252)

                # shrinkage e ridge
                cov_ann = _apply_shrinkage(cov_ann, est.shrink.method, float(est.shrink.intensity))
                cov_ann.values[range(n), range(n)] += float(max(0.0, mv.ridge))

                # ottimizza (se non hai pesi precedenti, prev_w=None)
                w_tgt = _mv_optimize(mu_ann, cov_ann, prev_w=w_prev, mv=mv)

                # turnover e costi (applicati PRIMA del ritorno di oggi)
                if w_prev is None:
                    trn = float(np.sum(np.abs(w_tgt)))  # prima allocazione
                else:
                    trn = _turnover(w_prev, w_tgt)

                c = _cost_from_turnover(trn, costs)
                nav *= max(1.0 - c, 0.0)

                turnovers[dates[t]] = trn
                costs_rec[dates[t]] = c

                w_prev = w_tgt.copy()
                last_reb_idx = t

            # 3) applica il ritorno del giorno t usando i pesi correnti (se nulli, equal-weight)
            wt = w_prev if w_prev is not None else (np.ones(n) / n)
            day_ret = float(np.dot(wt, r.iloc[t].fillna(0.0).values))
            nav *= (1.0 + day_ret)

            # 4) aggiorna i pesi “driftati” a fine giornata (senza costi)
            #    w_{t+} = w_t * (1 + r_i) / (1 + port_ret)
            denom = (1.0 + day_ret) if (1.0 + day_ret) != 0 else 1.0
            w_after = wt * (1.0 + r.iloc[t].fillna(0.0).values) / denom
            # normalizza numericamente
            if np.sum(np.abs(w_after)) == 0:
                w_after = np.ones(n) / n
            w_prev = w_after / np.sum(w_after)

            # 5) registra
            nav_vals.append(nav)
            w_hist[t, :] = w_prev

        # --- costruisci output ---
        nav_df = pd.DataFrame({"nav": nav_vals}, index=dates)
        weights_df = pd.DataFrame(w_hist, index=dates, columns=tickers)

        turn_df = pd.DataFrame({"turnover": pd.Series(turnovers)}).sort_index()
        cost_df = pd.DataFrame({"cost": pd.Series(costs_rec)}).sort_index()

        return {
            "nav": nav_df,
            "turnover": turn_df,
            "costs": cost_df,
            "weights": weights_df,
        }
