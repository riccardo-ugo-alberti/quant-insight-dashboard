# src/data_loader.py
from __future__ import annotations
from typing import List, Optional
import pandas as pd

def _ensure_datetime(s: Optional[str]) -> Optional[str]:
    # yfinance accetta ISO stringhe; lasciamo pass-through
    return s

def fetch_prices(
    tickers: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Scarica prezzi con yfinance e restituisce un DataFrame a colonne semplici
    con 'Adj Close' se disponibile, altrimenti 'Close'.

    Ritorna un DataFrame (index = Date, columns = tickers).
    """
    if not tickers:
        return pd.DataFrame()

    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError("yfinance non installato. Aggiungi 'yfinance' ai requirements.") from e

    start = _ensure_datetime(start)
    end = _ensure_datetime(end)

    df = yf.download(
        tickers=" ".join(tickers),
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        actions=False,
        threads=True,
    )

    # Normalizzazione: estraiamo 'Adj Close' quando presente (multi-index), altrimenti 'Close'
    def _extract_panel(_df) -> pd.DataFrame:
        if isinstance(_df.columns, pd.MultiIndex):
            last = _df.columns.get_level_values(-1)
            if "Adj Close" in last:
                out = _df.xs("Adj Close", axis=1, level=-1)
            elif "Close" in last:
                out = _df.xs("Close", axis=1, level=-1)
            else:
                # prendi l'ultimo livello come fallback
                out = _df.droplevel(0, axis=1)
        else:
            # Colonne semplici (caso singolo ticker). Proviamo a trovare Adj Close
            if "Adj Close" in _df.columns:
                out = _df[["Adj Close"]].copy()
            elif "Close" in _df.columns:
                out = _df[["Close"]].copy()
            else:
                # ultimo tentativo: prendi l’ultima colonna
                out = _df.iloc[:, [-1]].copy()

        # Se solo una colonna, rinomina con il ticker (yfinance mette il nome del campo)
        if out.shape[1] == 1 and len(tickers) == 1:
            out.columns = [tickers[0]]
        return out

    out = _extract_panel(df)

    # Mantieni l’ordine richiesto dove possibile
    cols = [c for c in tickers if c in out.columns]
    # Aggiungi eventuali altri simboli (se yfinance li ha rinominati)
    cols += [c for c in out.columns if c not in cols]
    out = out.loc[:, cols]

    # Pulizia base
    out = out.sort_index().dropna(how="all")
    return out
