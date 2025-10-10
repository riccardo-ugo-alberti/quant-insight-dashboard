# src/econ.py
from __future__ import annotations
import os
import pandas as pd
from pandas_datareader import data as pdr

def fetch_fred(series: list[str], start: str = "2000-01-01") -> pd.DataFrame:
    """
    Scarica serie FRED usando pandas_datareader 'fred'.
    Supporta opzionalmente FRED_API_KEY in st.secrets['FRED_API_KEY'] (non strettamente necessario).
    """
    out = []
    for s in series:
        try:
            df = pdr.DataReader(s, "fred", start=start)
            df.columns = [s]
            out.append(df)
        except Exception:
            pass
    if not out:
        return pd.DataFrame()
    data = pd.concat(out, axis=1).ffill()
    return data

def default_macro(start: str = "2000-01-01") -> pd.DataFrame:
    """
    GDP (GDPC1, quarterly), CPI (CPIAUCSL), 10Y yield (DGS10).
    """
    series = ["GDPC1", "CPIAUCSL", "DGS10"]
    df = fetch_fred(series, start=start)
    return df
