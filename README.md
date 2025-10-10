# Quant Insight Dashboard

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://quant-insight-dashboard-cq4skayqidv5vnmysogrcz.streamlit.app)
[![Open App](https://img.shields.io/badge/Open%20App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://quant-insight-dashboard-cq4skayqidv5vnmysogrcz.streamlit.app)

Interactive Streamlit app to analyze **performance, volatility, correlation, portfolio optimization, and beta** for any set of tickers (stocks/ETFs).

---

## ✨ Features

- **Performance**: Cumulative returns, KPI cards, Sharpe–Volatility scatter  
- **Volatility**: Rolling annualized volatility (custom window)  
- **Correlation**: Ordered heatmap + Top Positive/Negative pairs + Pairwise rolling correlation  
- **Portfolio Optimization**: Efficient frontier, weight caps, sector limits, rebalancing options  
- **Beta**: Rolling CAPM beta vs selected benchmark (default: SPY)  
- **Export**: CSV and one-click HTML report  
- **Presets**: Quick ticker sets (US Tech, Benchmarks, FAANG+)  
- **Save/Load**: Export/import configuration JSON  
- **Caching**: Fast reloads with Streamlit cache  
- **Testing & CI**: `pytest` + GitHub Actions for reproducibility  

---

## 🚀 Quickstart

```bash
git clone https://github.com/riccardo-ugo-alberti/quant-insight-dashboard.git
cd quant-insight-dashboard
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
streamlit run app/main.py
