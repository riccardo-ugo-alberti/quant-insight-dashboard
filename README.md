[![Live Demo — Streamlit](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://quant-insight-dashboard-cq4skayqidv5vnmysogrcz.streamlit.app)

# Quant Insight Dashboard

Interactive Streamlit web app for analyzing **performance, volatility, correlation, and risk metrics** of financial assets (stocks, ETFs, indices).

---

## ✨ Features

- **Prices** — Adjusted close prices with interactive range slider  
- **Performance** — Cumulative returns, KPI cards (Return & Sharpe), and Risk vs Return scatter with regression line  
- **Volatility** — Rolling annualized volatility (customizable window)  
- **Correlation** — Ordered correlation heatmap + Top positive/negative pairs + Pairwise rolling correlation  
- **Beta** — Rolling CAPM β vs selected benchmark (default = SPY)  
- **Export** — One-click CSV and HTML report download  
- **Presets** — Quick ticker sets (US Tech, Benchmarks, FAANG+)  
- **Save / Load** — Export or import full setup in JSON  
- **Optional** — Sector coloring on scatter (best-effort via Yahoo Finance)  
- **Caching** — Fast reloads using Streamlit cache  
- **Tests & CI** — `pytest` + GitHub Actions integration  

---

## 🚀 Quickstart (run locally)

Clone the repository and launch the app in your local environment.

```bash
git clone https://github.com/riccardo-ugo-alberti/quant-insight-dashboard.git
cd quant-insight-dashboard

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate        # Windows
# source .venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/main.py
## 📊 Screenshots

![Dashboard (Performance)](docs/screenshot_main.png)

![Correlation Heatmap](docs/screenshot_snapshot.png)

