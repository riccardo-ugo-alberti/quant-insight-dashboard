# src/data_loader.py
import yfinance as yf
import pandas as pd

def fetch_prices(tickers: list[str], start: str = "2020-01-01") -> pd.DataFrame:
    """
    Download daily prices for tickers, returning a tidy DataFrame:
    index = dates, columns = tickers, values = price.
    Prefers 'Adj Close', falls back to 'Close'. Works with 1+ tickers.
    """
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not tickers:
        raise ValueError("No valid tickers provided")

    raw = yf.download(
        tickers,
        start=start,
        auto_adjust=False,     # keep Close and Adj Close (if available)
        progress=False,
        group_by="column"      # field-first columns for multi-ticker
    )

    if raw is None or raw.empty:
        raise ValueError("No data returned. Check tickers or internet connection.")

    def pick_table(df: pd.DataFrame) -> pd.DataFrame:
        # MultiIndex columns (field, ticker)
        if isinstance(df.columns, pd.MultiIndex):
            lvl0 = df.columns.get_level_values(0)
            if "Adj Close" in set(lvl0):
                out = df["Adj Close"]
            elif "Close" in set(lvl0):
                out = df["Close"]
            else:
                # fallback to first available field
                out = df.xs(df.columns.levels[0][0], axis=1, level=0)
            return out

        # Single-index columns (single ticker)
        cols = list(df.columns)
        if "Adj Close" in cols:
            series = df["Adj Close"]
        elif "Close" in cols:
            series = df["Close"]
        else:
            numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
            if not numeric_cols:
                raise ValueError("No numeric price columns found.")
            series = df[numeric_cols[0]]
        return series.to_frame(name=tickers[0])

    data = pick_table(raw)
    data = data.dropna(how="all").sort_index()
    return data
