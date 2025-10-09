# tests/test_analytics.py
import pandas as pd
import numpy as np
from src.analytics import to_returns, rolling_beta

def test_to_returns_shape():
    idx = pd.date_range("2023-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"A":[100,101,102,101,103], "B":[50,52,51,52,53]}, index=idx)
    rets = to_returns(prices)
    assert rets.shape == (4,2)
    assert np.isfinite(rets.values).all()

def test_rolling_beta_basic():
    idx = pd.date_range("2023-01-01", periods=120, freq="D")
    rng = np.random.default_rng(0)
    bench = np.cumsum(rng.normal(0, 0.01, size=len(idx)))
    a = 1.5*bench + rng.normal(0,0.01,len(idx))  # asset correlated with beta ~1.5
    prices = pd.DataFrame({"SPY":100*np.exp(bench), "A":100*np.exp(a)}, index=idx)
    rets = prices.pct_change().dropna()
    betas = rolling_beta(rets, benchmark="SPY", window=30)
    assert "A" in betas.columns and "SPY" in betas.columns
    assert betas.dropna().shape[0] > 0
