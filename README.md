[![Live Demo — Streamlit](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://quant-insight-dashboard-cq4skayqidv5vnmysogrcz.streamlit.app)

# Quant Insight Dashboard

Interactive Streamlit web app for analyzing **performance, volatility, correlation, and risk metrics** of financial assets (stocks, ETFs, indices).

---

## ✨ Features

- **Performance** — Cumulative returns, KPI cards (Return & Sharpe), Risk–Return scatter with OLS regression
- **Risk** — Rolling annualised volatility and CAPM beta (customisable window)
- **Correlation** — Ordered correlation heatmap, Top ± pairs, pairwise rolling correlation
- **Data** — Prices preview, CSV export, one-click HTML report
- **UX** — Ticker presets (US Tech, Benchmarks, FAANG+), JSON save/load of the setup
- **Styling** — Neutral professional tables (zebra/borders), vivid but readable heatmap
- **Infra** — Streamlit caching, tests with `pytest`, GitHub Actions ready

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

