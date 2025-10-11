# src/factors.py
from __future__ import annotations
import pandas as pd
import statsmodels.api as sm

# download tramite pandas-datareader
import pandas_datareader.data as web


def get_fama_french(freq: str = "M", five: bool = True) -> pd.DataFrame:
    """
    Scarica Fama–French 3 o 5 fattori.
    freq: "M" (monthly) oppure "D" (daily)
    Restituisce dataframe con colonne fattori + RF in decimali (non percentuali).
    """
    name = (
        "F-F_Research_Data_5_Factors_2x3" if five else "F-F_Research_Data_Factors"
        if freq.upper().startswith("M")
        else "F-F_Research_Data_5_Daily" if five else "F-F_Research_Data_Factors_Daily"
    )

    try:
        ff = web.DataReader(name, "famafrench")[0]
    except Exception:
        raise RuntimeError("Could not fetch Fama–French dataset. Try switching frequency or retry later.")

    # indice -> datetime
    if isinstance(ff.index, pd.PeriodIndex):
        # per i monthly, to_timestamp porta alla fine mese
        ff.index = ff.index.to_timestamp(how="end")
    else:
        ff.index = pd.to_datetime(ff.index)

    ff = ff / 100.0  # % -> decimali
    return ff


def regress_ff(excess_returns: pd.DataFrame, ff: pd.DataFrame, five: bool = True) -> pd.DataFrame:
    """
    OLS per ogni asset: excess returns = alpha + beta*fattori (+ ... ) + eps
    Ritorna DataFrame con alpha, R2 e loadings fattori.
    """
    factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"] if five else ["Mkt-RF", "SMB", "HML"]

    # Allinea indici su datetime
    X = ff.copy()
    X.index = pd.to_datetime(X.index)
    X = X[factors]

    out = []
    for col in excess_returns.columns:
        y = excess_returns[col].copy()

        # se è PeriodIndex (mensile) converti a timestamp
        if isinstance(y.index, pd.PeriodIndex):
            y.index = y.index.to_timestamp(how="end")
        else:
            y.index = pd.to_datetime(y.index)

        aligned = X.reindex(y.index).dropna()
        y = y.reindex(aligned.index).dropna()

        if len(y) < 12:
            continue

        X_ = sm.add_constant(aligned)
        model = sm.OLS(y.astype(float), X_.astype(float)).fit()

        row = {"alpha": model.params.get("const", float("nan")), "R2": model.rsquared}
        for fac in factors:
            row[fac] = model.params.get(fac, float("nan"))

        out.append(pd.Series(row, name=col))

    return pd.DataFrame(out) if out else pd.DataFrame(columns=["alpha", *factors, "R2"])
