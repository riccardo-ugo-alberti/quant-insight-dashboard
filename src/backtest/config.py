# src/backtest/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ShrinkMethod = Literal["none", "const-cor", "diag", "identity"]


@dataclass(frozen=True)
class ShrinkageConfig:
    """
    Parametri per lo shrinkage della matrice di covarianza.
    intensity in [0,1]; 0 = nessuno, 1 = solo target.
    """
    method: ShrinkMethod = "const-cor"
    intensity: float = 0.25


@dataclass(frozen=True)
class EstimatorConfig:
    """
    Parametri di stima (rolling o EWMA) su rendimenti giornalieri.
    - window: dimensione finestra rolling (giorni)
    - use_ewma: True usa EWMA, False usa rolling semplice
    - ewma_lambda: lambda EWMA (tipico 0.94–0.99)
    - shrink: come sopra
    """
    window: int = 90
    use_ewma: bool = True
    ewma_lambda: float = 0.97
    shrink: ShrinkageConfig = ShrinkageConfig()


@dataclass(frozen=True)
class MVConfig:
    """
    Mean-Variance “alla Markowitz” con:
    - risk_aversion γ : più alto => portafogli più prudenti
    - ridge: aggiunta a diag(Σ) per stabilizzare
    - turnover_L2: penalità quadratica rispetto ai pesi precedenti
    - leverage: leva massima (1.0 = fully invested)
    - allow_short: se True ammessi pesi negativi (short)
    """
    risk_aversion: float = 5.0
    ridge: float = 1e-3
    turnover_L2: float = 5.0
    leverage: float = 1.0
    allow_short: bool = False


@dataclass(frozen=True)
class CostConfig:
    """
    Costi di transazione applicati sul notional scambiato (turnover):
    entrambe in basis points (1 bps = 0.01%).
    """
    proportional_bps: float = 5.0
    slippage_bps: float = 1.0


@dataclass(frozen=True)
class RebalanceConfig:
    """
    Ribilanciamento ogni k giorni, con eventuale offset (0 = primo giorno utile).
    """
    every_k_days: int = 21
    offset: int = 0


@dataclass(frozen=True)
class BacktestConfig:
    """
    Config unificata per il motore di backtest.
    """
    estimator: EstimatorConfig = EstimatorConfig()
    mv: MVConfig = MVConfig()
    costs: CostConfig = CostConfig()
    rebalance: RebalanceConfig = RebalanceConfig()
