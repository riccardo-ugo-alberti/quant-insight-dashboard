# Quant Insight Dashboard

Streamlit app focused on two core tasks:
- beginner-friendly portfolio construction
- practical ticker and portfolio analysis

Goal: clean UX, minimal controls, and no conflicting features in the main flow.

## Core Features

- Quick ticker presets + manual setup
- Ticker analysis: prices, cumulative performance, return/volatility/Sharpe summary, rolling volatility, correlations
- Portfolio Builder with `Conservative`, `Balanced`, and `Growth` starter profiles
- Portfolio Analysis with 4 approaches:
  - manual allocation
  - optimized max Sharpe
  - optimized min volatility
  - optimized CVaR (when `cvxpy` is available)
- Historical NAV and drawdown view
- Efficient frontier and optional Monte Carlo simulation
- CSV export + HTML report export

## Local Run

```bash
git clone https://github.com/riccardo-ugo-alberti/quant-insight-dashboard.git
cd quant-insight-dashboard
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
streamlit run app/main.py
```

## Streamlit Cloud Deploy

- Main file path: `streamlit_app.py`
- `streamlit_app.py` forwards execution to `app/main.py`

## Project Structure

- `app/main.py`: simplified primary UI
- `src/data_loader.py`: Yahoo Finance price loading
- `src/analytics.py`: metrics and simulation logic
- `src/optimizer.py`: MV/CVaR optimizers + efficient frontier
- `src/visuals.py`: Plotly chart builders
- `src/report.py`: HTML report generation

## Notes

- If CVaR is not available in your environment, the app falls back to max Sharpe/min volatility options.
- Setup JSON export and analysis CSV exports are included.
