# src/meta.py
from functools import lru_cache
import yfinance as yf

@lru_cache(maxsize=512)
def get_sector(ticker: str) -> str | None:
    try:
        info = yf.Ticker(ticker).info or {}
        # yfinance may use different keys over time; try a couple
        return info.get("sector") or info.get("industry") or None
    except Exception:
        return None

def get_sectors(tickers: list[str]) -> dict[str, str]:
    out = {}
    for t in tickers:
        s = get_sector(t)
        if s:
            out[t] = s
    return out
