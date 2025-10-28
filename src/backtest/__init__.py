# src/__init__.py

# Espone utility principali SENZA import circolari/rotti
from .data_loader import fetch_prices
from .analytics import (
    to_returns, compute_summary, compute_corr_matrix, rolling_vol, rolling_volatility
)
from .optimizer import optimize_portfolio, compute_frontier, optimize_cvar
from .visuals import (
    prices_chart, perf_cum_chart, rr_scatter, vol_chart, corr_heatmap,
    frontier_chart, weights_donut, weights_pie
)

# Backtest: import dai moduli corretti dentro il sottopackage
from .backtest.engine import RollingEWMAEngine
from .backtest.config import (
    BacktestConfig, EstimatorConfig, ShrinkageConfig,
    MVConfig, CostConfig, RebalanceConfig
)

__all__ = [
    # data / analytics
    "fetch_prices", "to_returns", "compute_summary", "compute_corr_matrix",
    "rolling_vol", "rolling_volatility",
    # optimizer
    "optimize_portfolio", "compute_frontier", "optimize_cvar",
    # visuals
    "prices_chart", "perf_cum_chart", "rr_scatter", "vol_chart", "corr_heatmap",
    "frontier_chart", "weights_donut", "weights_pie",
    # backtest
    "RollingEWMAEngine", "BacktestConfig", "EstimatorConfig", "ShrinkageConfig",
    "MVConfig", "CostConfig", "RebalanceConfig",
]
