from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics import simulate_portfolio_paths


def test_simulate_portfolio_paths_shape_and_start_value():
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(400, 3)),
        columns=["AAPL", "MSFT", "NVDA"],
    )
    weights = pd.Series({"AAPL": 0.4, "MSFT": 0.4, "NVDA": 0.2})

    out = simulate_portfolio_paths(
        returns=returns,
        weights=weights,
        horizon_days=126,
        n_sims=200,
        initial_value=100.0,
        random_seed=11,
    )

    assert out.shape == (127, 200)
    assert np.allclose(out.iloc[0].values, 100.0)
    assert np.isfinite(out.values).all()
