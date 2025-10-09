# src/report.py
from __future__ import annotations
import pandas as pd
from plotly.graph_objs import Figure

def fig_to_html(fig: Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn")

def build_html_report(
    prices: pd.DataFrame,
    summary: pd.DataFrame,
    figs: dict[str, Figure]
) -> str:
    # Minimal, clean HTML report (self-contained with Plotly CDN)
    parts = []
    parts.append("""
<!doctype html><html><head>
<meta charset="utf-8"/>
<title>Quant Insight Report</title>
<style>
body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; color: #0F172A; }
h1, h2 { margin: 8px 0; }
.card { background:#F6F7FB; padding:12px 16px; border-radius:12px; margin: 12px 0; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { border: 1px solid #E5E7EB; padding: 6px 8px; text-align: right; }
th { background: #F1F5F9; }
td:first-child, th:first-child { text-align: left; }
</style></head><body>
<h1>📈 Quant Insight — Report</h1>
<div class="card">Auto-generated summary with charts.</div>
""")
    # Summary table
    parts.append("<h2>Summary</h2>")
    parts.append(summary.to_html())

    # Charts
    for title, fig in figs.items():
        parts.append(f"<h2>{title}</h2>")
        parts.append(fig_to_html(fig))

    # Preview data tail
    parts.append("<h2>Data — last rows</h2>")
    parts.append(prices.tail().to_html())

    parts.append("</body></html>")
    return "\n".join(parts).strip()
