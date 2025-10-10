# src/report.py
from __future__ import annotations
import io
import pandas as pd

# -------------------------------
# Simple HTML Report Generator
# -------------------------------

_HTML_CSS = """
<style>
  body {
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
    padding: 24px;
    color: #111;
    background-color: #fff;
  }
  h1 {
    font-size: 24px;
    margin-bottom: 6px;
  }
  h2 {
    font-size: 18px;
    margin-top: 24px;
    margin-bottom: 8px;
  }
  p, .note {
    font-size: 13px;
    color: #444;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    font-size: 13px;
  }
  th, td {
    border: 1px solid #e5e7eb;
    padding: 6px 8px;
    text-align: right;
  }
  th {
    background: #f8fafc;
    text-align: left;
  }
  .small {
    font-size: 12px;
    color: #777;
  }
</style>
"""

def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    """Applica formattazione leggibile ai valori numerici."""
    if df.empty:
        return df
    df_fmt = df.copy()
    for col in df_fmt.columns:
        if pd.api.types.is_numeric_dtype(df_fmt[col]):
            df_fmt[col] = df_fmt[col].apply(lambda x: f"{x:.2%}" if abs(x) < 1.5 else f"{x:.2f}")
    return df_fmt

def build_html_report(prices: pd.DataFrame,
                      summary: pd.DataFrame,
                      corr: pd.DataFrame) -> io.StringIO:
    """
    Crea un report HTML statico con prezzi, performance e correlazioni.
    Ritorna un buffer (StringIO) pronto per essere scaricato.
    """
    buf = io.StringIO()
    buf.write("<!doctype html><html><head><meta charset='utf-8'>")
    buf.write(_HTML_CSS)
    buf.write("</head><body>")

    buf.write("<h1>Quant Insight Dashboard — Report</h1>")
    buf.write("<p class='small'>Generated automatically by Quant Insight Dashboard.</p>")

    # Prices (preview)
    buf.write("<h2>Prices — Last Observations</h2>")
    tail = prices.tail().round(4)
    buf.write(tail.to_html(border=0))

    # Summary table
    buf.write("<h2>Performance Summary</h2>")
    buf.write(_fmt(summary).to_html(border=0))

    # Correlation matrix
    buf.write("<h2>Correlation Matrix</h2>")
    buf.write(corr.round(2).to_html(border=0))

    buf.write("<p class='note'>End of report.</p>")
    buf.write("</body></html>")
    buf.seek(0)
    return buf
