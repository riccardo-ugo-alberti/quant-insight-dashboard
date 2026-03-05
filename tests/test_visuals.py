from __future__ import annotations

import numpy as np
import pandas as pd

from src.visuals import monte_carlo_paths_chart


def test_monte_carlo_chart_hides_cloud_hover_and_shows_bands_only():
    rng = np.random.default_rng(1)
    paths = pd.DataFrame(rng.normal(100, 5, size=(40, 25)))

    fig = monte_carlo_paths_chart(paths)

    # All simulation cloud traces should skip hover events.
    cloud = fig.data[:-3]
    assert len(cloud) == 25
    assert all(getattr(t, "hoverinfo", None) == "skip" for t in cloud)

    # Percentile traces should be the only hoverable labels.
    labels = [t.name for t in fig.data[-3:]]
    assert labels == ["Bottom 10%", "Median", "Top 10%"]
