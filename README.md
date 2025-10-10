# Quant Insight Dashboard

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://quant-insight-dashboard-cq4skayqidv5vnmysogrcz.streamlit.app)

Interactive Streamlit app to analyze **performance, volatility, correlation, and portfolio optimization** for any set of tickers (stocks/ETFs).

---

## ✨ Features

- **Performance**: Cumulative returns, KPI cards, Sharpe–Volatility scatter  
- **Volatility**: Rolling annualized volatility (window selectable)  
- **Correlation**: Ordered heatmap + Top Positive/Negative pairs + Pairwise rolling correlation  
- **Portfolio Optimization**: Efficient frontier, weight constraints, sector caps  
- **Beta**: Rolling CAPM beta vs benchmark (default: SPY)  
- **Export**: CSV + one-click HTML report  
- **Presets**: Quick ticker sets (Tech, Benchmarks, FAANG+)  
- **Save/Load**: Configuration JSON  
- **Caching**: Fast reloads with Streamlit cache  
- **Testing & CI**: `pytest` + GitHub Actions  

---

## 🚀 Quickstart

```bash
git clone https://github.com/riccardo-ugo-alberti/quant-insight-dashboard.git
cd quant-insight-dashboard
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
streamlit run app/main.py


![Dashboard (Performance)](docs/screenshot_main.png)

![Correlation Heatmap](docs/screenshot_snapshot.png)

[![Release](https://img.shields.io/github/v/release/riccardo-ugo-alberti/quant-insight-dashboard?label=Latest%20release)](https://github.com/riccardo-ugo-alberti/quant-insight-dashboard/releases)
