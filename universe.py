"""
The point-in-time universe, the liquidity floor, each stock's sector basket, and the
equal-weight benchmarks.

  membership(uni, sh, cal)   -> member [session x qid] (bool), sector [session x qid] (object), rank
                                a month-end snapshot applies from the NEXT session (nothing after t is used)
  liquidity(P, cal)          -> liquid [session x qid]: median daily traded value over LIQ_WINDOW >= LIQ_FLOOR_CR;
                                names with no traded value on record at all pass on market cap alone
  eligible(...)              -> member & liquid & priced & has a sector with >= MIN_PEERS other eligible members
  baskets(r, E, sector)      -> R_ex   [session x qid]  the stock's group, equal-weighted, EXCLUDING itself (its "partner")
                                R_grp  [session x qid]  the same group including itself (what the benchmark holds)
                                R_sec  [session x group] the group returns; R_uni [session] the equal-weight universe;
                                members decided at t-1, so every basket is investable
  ew_benchmark(r, E, cal)    -> the equal-weight universe book: monthly rebalance, costs on turnover, NAV and daily return

Returns are simple daily returns r_t = P_t / P_{t-1} - 1; a hole (no print) is NaN and never filled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def _apply_from_next_session(monthly: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """A snapshot dated d (a month-end close) is known after that close: it applies from the first
    session strictly after d, until the next snapshot."""
    m = monthly.copy()
    m.index = m.index + pd.Timedelta(days=1)
    out = m.reindex(m.index.union(cal)).ffill().reindex(cal)
    return out


def membership(uni: pd.DataFrame, sh: pd.DataFrame, cal: pd.DatetimeIndex, n: int = C.UNIVERSE_N):
    # the snapshot date is the month-end TRADING day; it applies from the session after it
    rank = uni.pivot_table(index="date", columns="qid", values="rank", aggfunc="min")
    member_m = (rank <= n)
    sec = sh.pivot_table(index="date", columns="qid", values="sector", aggfunc="first")
    sec = sec.reindex(index=member_m.index, columns=member_m.columns)
    member = _apply_from_next_session(member_m.astype(float), cal).fillna(0).astype(bool)
    sector = _apply_from_next_session(sec, cal)
    rank_d = _apply_from_next_session(rank, cal)
    return member, sector, rank_d


def liquidity(P: dict[str, pd.DataFrame], cal: pd.DatetimeIndex, floor_cr: float = C.LIQ_FLOOR_CR, window: int = C.LIQ_WINDOW) -> pd.DataFrame:
    tv = P["turnover"]                                     # rupees crore per session (fetch_prices.py)
    med = tv.rolling(window, min_periods=int(window * 0.5)).median()
    liquid = med >= floor_cr
    no_value = tv.notna().sum(axis=0) == 0                 # no traded value on record at all: passes on market cap alone
    liquid.loc[:, no_value[no_value].index] = True
    return liquid.reindex(cal, fill_value=False)


def _sector_codes(sector: pd.DataFrame) -> tuple[np.ndarray, list]:
    cats = pd.Categorical(sector.values.ravel())
    codes = cats.codes.reshape(sector.shape)              # -1 = no sector
    return codes, list(cats.categories)


def eligible(member: pd.DataFrame, liquid: pd.DataFrame, close: pd.DataFrame, sector: pd.DataFrame,
             min_peers: int = C.MIN_PEERS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """E0 = member & liquid & priced & has a sector; E = E0 & (the group has >= min_peers OTHER eligible members).
    Returns E and the peer count matrix."""
    cols = member.columns
    E0 = member & liquid.reindex(columns=cols, fill_value=False) & close.reindex(columns=cols).notna() & sector.reindex(columns=cols).notna()
    codes, cats = _sector_codes(sector.reindex(columns=cols))
    e0 = E0.values
    n = np.zeros(e0.shape, dtype=int)
    k = len(cats)
    for t in range(e0.shape[0]):
        ct = codes[t]
        sel = e0[t] & (ct >= 0)
        if not sel.any():
            continue
        cnt = np.bincount(ct[sel], minlength=k)
        row = np.where(ct >= 0, cnt[np.clip(ct, 0, None)], 0)
        n[t] = row
    peers = pd.DataFrame(n, index=member.index, columns=cols) - E0.astype(int)     # others, not itself
    E = E0 & (peers >= min_peers)
    return E, peers


def baskets(r: pd.DataFrame, E: pd.DataFrame, sector: pd.DataFrame):
    """Equal-weight group returns, members decided at t-1 (E.shift(1)) and having a return at t."""
    cols = r.columns
    M = (E.shift(1, fill_value=False) & r.notna()).values
    rv = np.nan_to_num(r.values, nan=0.0)
    codes, cats = _sector_codes(sector.reindex(columns=cols))
    k = len(cats)
    T, N = rv.shape
    S = np.zeros((T, N)); Nn = np.zeros((T, N))
    R_sec = np.full((T, k), np.nan); n_sec = np.zeros((T, k), dtype=int)
    R_uni = np.full(T, np.nan); n_uni = np.zeros(T, dtype=int)
    for t in range(T):
        ct = codes[t]
        sel = M[t] & (ct >= 0)
        if M[t].any():
            R_uni[t] = rv[t][M[t]].mean(); n_uni[t] = int(M[t].sum())
        if not sel.any():
            continue
        s = np.bincount(ct[sel], weights=rv[t][sel], minlength=k)
        c = np.bincount(ct[sel], minlength=k)
        has = ct >= 0
        S[t][has] = s[ct[has]]; Nn[t][has] = c[ct[has]]
        with np.errstate(invalid="ignore", divide="ignore"):
            R_sec[t] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        n_sec[t] = c
    self_r = rv * M                                       # the stock's own return, if it is a member
    with np.errstate(invalid="ignore", divide="ignore"):
        R_ex = np.where(Nn - M > 0, (S - self_r) / (Nn - M), np.nan)
        R_grp = np.where(Nn > 0, S / np.maximum(Nn, 1), np.nan)
    idx = r.index
    return (pd.DataFrame(R_ex, index=idx, columns=cols), pd.DataFrame(R_grp, index=idx, columns=cols),
            pd.DataFrame(R_sec, index=idx, columns=cats), pd.DataFrame(n_sec, index=idx, columns=cats),
            pd.Series(R_uni, index=idx, name="R_uni"), pd.Series(n_uni, index=idx, name="n_uni"))


def basket_level(R_ex: pd.DataFrame) -> pd.DataFrame:
    """log level of each stock's partner basket: cumulative sum of log(1 + R_ex), a missing day counted as flat."""
    return np.log1p(R_ex.fillna(0.0)).cumsum()


def month_starts(cal: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The first session of each calendar month in cal (where the benchmark rebalances)."""
    s = pd.Series(cal, index=cal)
    return pd.DatetimeIndex(s.groupby([cal.year, cal.month]).first().values)


def ew_benchmark(r: pd.DataFrame, E: pd.DataFrame, cal: pd.DatetimeIndex, cost_bps: float = C.COST_BPS,
                 start: pd.Timestamp | None = None) -> pd.DataFrame:
    """Equal-weight book of the eligible universe (membership known at t-1), bought at the first session's
    close, held, rebalanced to equal weight at the first session of each month, cost_bps per side on the
    turnover. A name with no print on a day earns 0 that day and is dropped at the next rebalance."""
    r = r.reindex(cal); E = E.reindex(cal, fill_value=False)
    start = start or cal[0]
    i0 = cal.get_loc(start)
    rv = np.nan_to_num(r.values, nan=0.0)
    Em = E.shift(1, fill_value=False).values
    rebal = set(cal.get_indexer(month_starts(cal)))
    c = cost_bps / 1e4
    T, N = rv.shape
    w = np.zeros(N)
    nav = np.full(T, np.nan); ret = np.full(T, np.nan); turn = np.zeros(T); cost = np.zeros(T)
    nav_prev = 1.0
    first = True
    for t in range(i0, T):
        if first:
            m = Em[t]
            if not m.any():
                continue
            w = m / m.sum()
            turn[t] = 1.0; cost[t] = c
            nav[t] = nav_prev * (1 - cost[t]) * (1 + (w * rv[t]).sum())
            ret[t] = nav[t] / nav_prev - 1
            w = w * (1 + rv[t]); w = w / w.sum()
            nav_prev = nav[t]; first = False
            continue
        if t in rebal:
            m = Em[t]
            if m.any():
                tgt = m / m.sum()
                tv = np.abs(tgt - w).sum() / 2
                turn[t] = tv; cost[t] = 2 * tv * c
                w = tgt
        day = (w * rv[t]).sum()
        nav[t] = nav_prev * (1 - cost[t]) * (1 + day)
        ret[t] = nav[t] / nav_prev - 1
        w = w * (1 + rv[t]); s = w.sum()
        w = w / s if s > 0 else w
        nav_prev = nav[t]
    return pd.DataFrame({"nav": nav, "ret": ret, "turnover": turn, "cost": cost}, index=cal)
