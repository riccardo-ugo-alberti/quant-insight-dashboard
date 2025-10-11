# src/__init__.py

from .analytics import (
    to_returns,
    compute_summary,
    compute_corr_matrix,
    rolling_vol,
    rolling_volatility,  # alias esposto
)

from .data_loader import fetch_prices

__all__ = [
    "to_returns",
    "compute_summary",
    "compute_corr_matrix",
    "rolling_vol",
    "rolling_volatility",
    "fetch_prices",
]
