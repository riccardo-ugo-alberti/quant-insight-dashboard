# src/factors.py
from __future__ import annotations
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import statsmodels.api as sm

def get_fama_french(freq: str = "M", five: bool = True) -> pd.DataFrame:
    """
    Carica i fattori Fama-French da pandas_datareader.
    freq: 'D' (Daily), 'M' (Monthly). 'famafrench' ufficiale è mensile; per daily usa 'F-F_Research_Data_5_Factors_2x3_daily'.
    """
    if freq.upper().startswith("M"):
        ds = "F-F_Research_Data_5_Factors_2x3" if five else "F-F_Research_Data_Factors"
    else:
        ds = "F-F_Research_Data_5_Factors_2x3_daily" if five else "F-F_Research_Data_Factors_daily"
    ff = pdr.DataReader(ds, "famafrench")[0]
    ff = ff.rename(columns=lambda c: c.replace(" ", ""))  # es: 'Mkt-RF'
    ff = ff / 100.0  # in frazioni
    return ff

def regress_ff(excess_returns: pd.DataFrame, ff: pd.DataFrame, five: bool = True) -> pd.DataFrame:
    """
    Esegue regressione OLS: r_it - rf_t = alpha + b * factors + eps.
    Ritorna betas, alpha e R2 per ogni ticker.
    """
    common = excess_returns.join(ff, how="inner")
    results = []
    fac_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"] if five else ["Mkt-RF", "SMB", "HML"]
    for t in excess_returns.columns:
        y = common[t].dropna()
        X = common.loc[y.index, fac_cols]
        X = sm.add_constant(X)
        model = sm.OLS(y.values, X.values)
        res = model.fit()
        row = {"alpha": res.params[0], "R2": res.rsquared}
        for i, name in enumerate(fac_cols, start=1):
            row[name] = res.params[i]
        results.append(pd.Series(row, name=t))
    return pd.DataFrame(results)
