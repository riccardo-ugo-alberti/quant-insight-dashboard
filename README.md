# Quant Insight Dashboard

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://quant-insight-dashboard-ing3uxvvrggk7hntnbcg5r.streamlit.app/)

Quant Insight Dashboard is a Streamlit app focused on two main jobs:
- beginner-friendly portfolio construction
- practical ticker and portfolio analysis

## Main Features

- Guided `Build Your First Portfolio` workflow (profile, capital plan, goal check, starter allocation)
- Ticker analytics: prices, cumulative performance, risk/return summary, rolling volatility, correlations
- Portfolio optimization: Max Sharpe, Min Vol, Target Return, CVaR
- Efficient frontier visualization and allocation diagnostics
- Dynamic backtest (rolling/EWMA estimation, shrinkage, turnover, costs)
- Monte Carlo simulation for forward scenario ranges
- Data export: prices, correlation, optimizer outputs, HTML report

## Local Quickstart

```bash
git clone https://github.com/riccardo-ugo-alberti/quant-insight-dashboard.git
cd quant-insight-dashboard
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
streamlit run app/main.py
```

## Streamlit Cloud

- Repository: `riccardo-ugo-alberti/quant-insight-dashboard`
- Branch: `main`
- Main file path: `streamlit_app.py`

## Project Structure

- `app/main.py`: main Streamlit app
- `src/data_loader.py`: Yahoo Finance data loader
- `src/analytics.py`: metrics, rolling stats, simulations
- `src/optimizer.py`: MV and CVaR optimization
- `src/backtest/`: dynamic backtest engine and configs
- `src/visuals.py`: Plotly chart builders
- `src/report.py`: HTML report generation

## Notes

- Use price-based optimizer entrypoints with price data (`optimize_portfolio`, `frontier_from_prices`, `optimize_cvar`).
- Use returns-based entrypoints only with daily returns data.
- If CVaR is unavailable in your environment, the app automatically falls back to MV options.
