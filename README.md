# 📈 Quant Insight Dashboard

Interactive Streamlit app to analyze **performance, volatility, correlation, and beta** for any set of tickers (stocks/ETFs).

### Live Demo
Deploy easily on **Streamlit Community Cloud** (connect this repo → set `app/main.py`).  
*(Add your live link here once deployed.)*

---

## ✨ Features

- **Prices**: Adjusted close prices with range slider  
- **Performance**: Cumulative returns, KPI cards (Return & Sharpe), Sharpe–Volatility scatter  
- **Volatility**: Rolling annualized vol (window selectable)  
- **Correlation**: Ordered heatmap + Top Positive/Negative pairs + Pairwise rolling corr  
- **Beta**: Rolling CAPM beta vs selected benchmark (default: SPY)  
- **Export**: CSV and one-click HTML report  
- **Presets**: Quick ticker sets (US Tech, Benchmarks, FAANG+)  
- **Save/Load**: Export/import configuration JSON  
- **Optional**: Sector coloring on scatter (best-effort via Yahoo)  
- **Caching**: Fast reloads with Streamlit cache  
- **Tests & CI**: `pytest` + GitHub Actions

---

## 🚀 Quickstart

```bash
git clone <your-repo-url>
cd quant-insight-dashboard
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app/main.py
