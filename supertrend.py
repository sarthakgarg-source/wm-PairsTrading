"""
The desk's supertrend, as recovered in wm-STstrat (validated there against the
AA report's SCMO_0043 ledger). Reproduced here, not re-derived:

  * bars:        daily here (the base case), or weekly (W-FRI) built from daily OHLC with resample_ohlc
  * ATR:         5-bar LINEAR-WEIGHTED true range, weights 1/15 .. 5/15 (newest heaviest)
                 ("lw"); "ema" = plain EMA span 5; "rma" = Wilder's
  * bands:       (H + L) / 2  +/-  multiplier x ATR, multiplier 2
  * state:       +1 (buy) when close crosses above the upper band from a sell
                 state; -1 (sell) when close crosses below the lower band from a
                 buy state; bands ratchet in the direction of the trend

`panel_states` runs it for every stock in a wide OHLC panel and returns the daily state
matrix; a weekly state is carried forward over the week's sessions (known at the Friday
close, applied from the next session).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def resample_ohlc(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    o = df["open"].resample(freq).first()
    h = df["high"].resample(freq).max()
    l = df["low"].resample(freq).min()
    c = df["close"].resample(freq).last()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna(subset=["close"])
    return out


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, lookback: int = C.LOOKBACK, method: str = C.ATR_METHOD) -> pd.Series:
    tr = true_range(df)
    if method == "lw":
        w = np.arange(1, lookback + 1, dtype=float)
        w /= w.sum()
        return tr.rolling(lookback).apply(lambda x: float(np.dot(x, w)), raw=True)
    if method == "ema":
        return tr.ewm(span=lookback, adjust=False).mean()
    if method == "rma":
        return tr.ewm(alpha=1.0 / lookback, adjust=False).mean()
    raise ValueError(method)


def _lw_atr_np(tr: np.ndarray, lookback: int) -> np.ndarray:
    w = np.arange(1, lookback + 1, dtype=float)
    w /= w.sum()
    out = np.full(len(tr), np.nan)
    if len(tr) >= lookback:
        win = np.lib.stride_tricks.sliding_window_view(tr, lookback)
        out[lookback - 1:] = win @ w
    return out


def supertrend(df: pd.DataFrame, lookback: int = C.LOOKBACK, mult: float = C.MULTIPLIER,
               method: str = C.ATR_METHOD) -> pd.DataFrame:
    """Returns DataFrame[state, upper, lower, stop] on the bar index. state in {+1, -1}."""
    a = atr(df, lookback, method)
    hl2 = (df["high"] + df["low"]) / 2.0
    up_basic = (hl2 + mult * a).values
    dn_basic = (hl2 - mult * a).values
    close = df["close"].values
    n = len(df)
    up = np.full(n, np.nan)
    dn = np.full(n, np.nan)
    state = np.zeros(n, dtype=int)
    first = int(np.argmax(~np.isnan(up_basic))) if (~np.isnan(up_basic)).any() else n
    for i in range(first, n):
        if i == first:
            up[i], dn[i] = up_basic[i], dn_basic[i]
            state[i] = 1 if close[i] > up[i] else -1
            continue
        # ratchet the bands
        up[i] = up_basic[i] if (up_basic[i] < up[i - 1] or close[i - 1] > up[i - 1]) else up[i - 1]
        dn[i] = dn_basic[i] if (dn_basic[i] > dn[i - 1] or close[i - 1] < dn[i - 1]) else dn[i - 1]
        if state[i - 1] == -1 and close[i] > up[i]:
            state[i] = 1
        elif state[i - 1] == 1 and close[i] < dn[i]:
            state[i] = -1
        else:
            state[i] = state[i - 1]
    out = pd.DataFrame({"state": state, "upper": up, "lower": dn}, index=df.index)
    out["stop"] = np.where(out["state"] == 1, out["lower"], out["upper"])
    out.loc[out.index[:first], "state"] = np.nan
    return out


def _state_np(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
              lookback: int, mult: float) -> np.ndarray:
    """The same rule on arrays (NaN rows skipped), for the panel. Returns float state with NaN before the first bar."""
    ok = ~(np.isnan(h) | np.isnan(l) | np.isnan(c))
    idx = np.flatnonzero(ok)
    out = np.full(len(c), np.nan)
    if len(idx) < lookback + 1:
        return out
    hh, ll, cc = h[idx], l[idx], c[idx]
    pc = np.concatenate([[np.nan], cc[:-1]])
    tr = np.nanmax(np.vstack([hh - ll, np.abs(hh - pc), np.abs(ll - pc)]), axis=0)
    tr[0] = hh[0] - ll[0]
    a = _lw_atr_np(tr, lookback)
    hl2 = (hh + ll) / 2.0
    ub, db = hl2 + mult * a, hl2 - mult * a
    n = len(cc)
    st = np.zeros(n)
    up = np.full(n, np.nan); dn = np.full(n, np.nan)
    first = lookback - 1
    up[first], dn[first] = ub[first], db[first]
    st[first] = 1 if cc[first] > up[first] else -1
    for i in range(first + 1, n):
        up[i] = ub[i] if (ub[i] < up[i - 1] or cc[i - 1] > up[i - 1]) else up[i - 1]
        dn[i] = db[i] if (db[i] > dn[i - 1] or cc[i - 1] < dn[i - 1]) else dn[i - 1]
        if st[i - 1] == -1 and cc[i] > up[i]:
            st[i] = 1
        elif st[i - 1] == 1 and cc[i] < dn[i]:
            st[i] = -1
        else:
            st[i] = st[i - 1]
    st[:first] = np.nan
    out[idx] = st
    return out


def panel_states(panel: dict[str, pd.DataFrame], bar: str = C.ST_BAR,
                 lookback: int = C.LOOKBACK, mult: float = C.MULTIPLIER) -> pd.DataFrame:
    """Daily state matrix [session x qid] in {+1, -1, NaN}.
    panel: dict with wide frames 'open', 'high', 'low', 'close' on the session calendar.
    bar='weekly': the rule runs on W-FRI bars; the state known at a Friday close applies from the
    next session (carried through the following week)."""
    o, h, l, c = panel["open"], panel["high"], panel["low"], panel["close"]
    if bar == "daily":
        out = {}
        for col in c.columns:
            out[col] = _state_np(o[col].values, h[col].values, l[col].values, c[col].values, lookback, mult)
        return pd.DataFrame(out, index=c.index)
    if bar == "weekly":
        wk = {}
        for col in c.columns:
            df = pd.DataFrame({"open": o[col], "high": h[col], "low": l[col], "close": c[col]}).dropna(subset=["close"])
            if len(df) < lookback + 1:
                wk[col] = pd.Series(np.nan, index=[])
                continue
            bars = resample_ohlc(df, C.ST_WEEKLY)
            wk[col] = supertrend(bars, lookback, mult)["state"]
        W = pd.DataFrame(wk)
        # known at the Friday close -> applies from the next session
        W.index = W.index + pd.Timedelta(days=1)
        return W.reindex(c.index.union(W.index)).ffill().reindex(c.index)
    raise ValueError(bar)
