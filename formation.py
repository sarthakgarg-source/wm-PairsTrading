"""
The signal: a level relationship between each stock and its partner basket, re-fitted monthly on
trailing data only, and the residual z-score day by day.

  Formation (at every month-end session d, on the FORMATION sessions ending at d, for every stock eligible at d):
      log P_i,t = alpha + beta * log B_i,t + e_t          (B_i = the stock's group, equal-weighted, excluding itself)
      z_t = e_t / sd(e)
      A PAIR EXISTS only if the residual has a level:
        Engle-Granger: ADF (constant, ADF_LAG lagged differences) on e, p-value from MacKinnon's
        two-variable surface (the residual is fitted, so the plain ADF table would overstate it); p < ADF_P_MAX
        and the AR(1) half-life of e between HALFLIFE_MIN and HALFLIFE_MAX sessions, and beta > 0.
  Daily:
      z_t under the fit in force = the latest fit dated STRICTLY before t. Nothing after t is used.
      An open position keeps the fit it was entered on (book.py: z_under).

  fit_all(...)     -> dict[refit date] -> DataFrame(index qid: alpha, beta, sd, adf_t, p, halflife, ok, n)
  daily_z(...)     -> Z [session x qid], OK [session x qid] (bool), FIT (Series: the refit date in force at each session)
  z_under(fits, d, qid, logP, logB) -> z of one stock under the fit dated d (for open positions)
  fit_summary(fits) -> per refit date: fitted, passing, pass share, median half-life / beta / p
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.adfvalues import mackinnonp

import config as C


def refit_dates(cal: pd.DatetimeIndex, window: int = C.FORMATION) -> pd.DatetimeIndex:
    """The last session of every month, from the first month-end with a full window of history."""
    s = pd.Series(cal, index=cal)
    me = pd.DatetimeIndex(s.groupby([cal.year, cal.month]).last().values)
    return me[me >= cal[window - 1]]


def _adf_t(e: np.ndarray, lag: int) -> float:
    """t-statistic on rho in  de_t = c + rho e_{t-1} + sum_{j<=lag} phi_j de_{t-j}."""
    de = np.diff(e)
    y = de[lag:]
    X = [np.ones(len(y)), e[lag:-1]]
    for j in range(1, lag + 1):
        X.append(de[lag - j:-j])
    X = np.column_stack(X)
    n, k = X.shape
    if n <= k + 2:
        return np.nan
    coef, res, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if rank < k:
        return np.nan
    resid = y - X @ coef
    s2 = (resid @ resid) / (n - k)
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return np.nan
    se = np.sqrt(cov[1, 1])
    return float(coef[1] / se) if se > 0 else np.nan


def halflife(e: np.ndarray) -> float:
    """AR(1): de_t = c + rho e_{t-1}; half-life = -ln 2 / ln(1 + rho), inf unless -1 < rho < 0."""
    de = np.diff(e)
    X = np.column_stack([np.ones(len(de)), e[:-1]])
    coef = np.linalg.lstsq(X, de, rcond=None)[0]
    rho = coef[1]
    return float(-np.log(2) / np.log(1 + rho)) if -1 < rho < 0 else np.inf


def fit_one(y: np.ndarray, x: np.ndarray, lag: int = C.ADF_LAG) -> dict:
    """OLS of y on x (both already restricted to valid rows), EG level test on the residual, half-life."""
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx <= 0:
        return {}
    beta = ((x - xm) * (y - ym)).sum() / sxx
    alpha = ym - beta * xm
    e = y - alpha - beta * x
    sd = e.std(ddof=2)
    t = _adf_t(e, lag)
    p = float(mackinnonp(t, regression="c", N=2)) if np.isfinite(t) else np.nan
    hl = halflife(e)
    return {"alpha": alpha, "beta": beta, "sd": sd, "adf_t": t, "p": p, "halflife": hl, "n": len(y)}


def fit_all(logP: pd.DataFrame, logB: pd.DataFrame, valid: pd.DataFrame, E: pd.DataFrame, cal: pd.DatetimeIndex,
            window: int = C.FORMATION, min_obs: int | None = None, lag: int = C.ADF_LAG,
            p_max: float = C.ADF_P_MAX, hl_min: float = C.HALFLIFE_MIN, hl_max: float = C.HALFLIFE_MAX,
            verbose: bool = True) -> dict[pd.Timestamp, pd.DataFrame]:
    min_obs = int(C.MIN_FORMATION_SHARE * window) if min_obs is None else min_obs
    dates = refit_dates(cal, window)
    Y, X, V = logP.values, logB.values, valid.values
    Ev = E.values
    cols = list(logP.columns)
    fits = {}
    for d in dates:
        k = cal.get_loc(d)
        rows = slice(k - window + 1, k + 1)
        elig = np.flatnonzero(Ev[k])
        out = {}
        for j in elig:
            m = V[rows, j]
            if m.sum() < min_obs:
                continue
            y = Y[rows, j][m]; x = X[rows, j][m]
            f = fit_one(y, x, lag)
            if f:
                out[cols[j]] = f
        df = pd.DataFrame.from_dict(out, orient="index")
        if len(df):
            df["ok"] = (df["p"] < p_max) & (df["halflife"] >= hl_min) & (df["halflife"] <= hl_max) & (df["beta"] > 0) & (df["sd"] > 0)
        fits[d] = df
        if verbose and (len(fits) % 24 == 0 or d == dates[-1]):
            print(f"  fitted {d.date()}: {len(df)} stocks, {int(df['ok'].sum()) if len(df) else 0} pass the level test")
    return fits


def daily_z(fits: dict, logP: pd.DataFrame, logB: pd.DataFrame, cal: pd.DatetimeIndex):
    """z under the fit in force (the latest refit date strictly before the session)."""
    dates = sorted(fits)
    Z = pd.DataFrame(np.nan, index=cal, columns=logP.columns)
    OK = pd.DataFrame(False, index=cal, columns=logP.columns)
    FIT = pd.Series(pd.NaT, index=cal, dtype="datetime64[ns]")
    for i, d in enumerate(dates):
        nxt = dates[i + 1] if i + 1 < len(dates) else None
        block = cal[(cal > d) & ((cal <= nxt) if nxt is not None else True)]
        if not len(block):
            continue
        f = fits[d]
        if not len(f):
            continue
        qids = [q for q in f.index if q in Z.columns]
        a = f.loc[qids, "alpha"].values; b = f.loc[qids, "beta"].values; s = f.loc[qids, "sd"].values
        z = (logP.loc[block, qids].values - a - b * logB.loc[block, qids].values) / s
        Z.loc[block, qids] = z
        OK.loc[block, qids] = np.broadcast_to(f.loc[qids, "ok"].values.astype(bool), z.shape)
        FIT.loc[block] = d
    return Z, OK, FIT


def z_under(fits: dict, d: pd.Timestamp, qid: str, logP: pd.DataFrame, logB: pd.DataFrame) -> pd.Series:
    """One stock's z over the whole calendar under the fit dated d."""
    f = fits[d].loc[qid]
    return (logP[qid] - f["alpha"] - f["beta"] * logB[qid]) / f["sd"]


def fit_summary(fits: dict) -> pd.DataFrame:
    rows = []
    for d, f in fits.items():
        if not len(f):
            rows.append({"refit_date": d, "fitted": 0}); continue
        ok = f[f["ok"]]
        rows.append({"refit_date": d, "fitted": len(f), "passing": int(f["ok"].sum()), "pass_share": float(f["ok"].mean()),
                     "chance_share": C.ADF_P_MAX, "p_lt_005_share": float((f["p"] < 0.05).mean()),
                     "median_p": f["p"].median(), "median_beta": f["beta"].median(),
                     "median_halflife_passing": ok["halflife"].median() if len(ok) else np.nan,
                     "median_sd_passing": ok["sd"].median() if len(ok) else np.nan,
                     "share_hl_too_long": float((f["halflife"] > C.HALFLIFE_MAX).mean()),
                     "share_hl_too_short": float((f["halflife"] < C.HALFLIFE_MIN).mean())})
    return pd.DataFrame(rows).set_index("refit_date")


def fits_long(fits: dict) -> pd.DataFrame:
    parts = [f.assign(refit_date=d).rename_axis("qid").reset_index() for d, f in fits.items() if len(f)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
