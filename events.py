"""
Phase 1 - the mechanism, event by event.

The question: after a stock has fallen to z <= Z_ENTRY against its partner basket, does it beat the
basket over the next 5 / 20 / 60 sessions - and does it matter whether its own trend has turned up?

  The cells (the 2 x 2):
    both            a cheap crossing where the stock's supertrend is in BUY at the crossing, or flips to BUY within
                    TURN_WINDOW sessions while z is still <= Z_STILL_CHEAP (and no structural stop has fired).
                    Entry = the session both hold. THE CELL THE STUDY RESTS ON.
    cheap alone     a cheap crossing that never qualifies within TURN_WINDOW. Entry = the crossing (the falling knife).
    turned-up alone a flip to BUY while z > Z_STILL_CHEAP (not cheap). Entry = the flip.
    neither         count-matched random (stock, date) draws: the null.

  Measurement (never the z): the stock's return minus its partner basket's, over the h sessions AFTER the
  execution session t_x = entry + DELAY, net of 2 x COST_BPS on the stock. "Caught up" = net relative return
  at H_MARK > 0. The z path is reported beside it for reference only (it comes back by construction).

  Crossings are counted once per run: after a crossing the stock is "in a run" until z > Z_EXIT or TIME_STOP
  sessions; crossings inside the run are not new events. Events within CA_EXCLUDE sessions of a suspect
  corporate-action jump (|log return| > CA_JUMP) are excluded and listed.

  Null: (a) the same stocks on random eligible dates; (b) random eligible same-group peers on the same dates.
  NULL_DRAWS draws of the cell's mean; the pass mark uses the stricter of the two 95th percentiles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config as C

CELLS = ["both", "cheap alone", "turned-up alone"]


# --------------------------------------------------------------- helpers --
def cum_log(r: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Cumulative log return with a missing day counted as flat (the stock holds its last price)."""
    return np.log1p(r.fillna(0.0)).cumsum()


def hac_t(y: pd.Series, lags: int = C.HAC_LAGS, min_n: int = 20) -> float:
    y = y.dropna()
    if len(y) < min_n:
        return np.nan
    res = sm.OLS(y.values, np.ones((len(y), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.tvalues[0])


def suspect_jumps(close: pd.DataFrame, thr: float = C.CA_JUMP) -> pd.DataFrame:
    lr = np.log(close) - np.log(close.ffill().shift(1))
    return lr.abs() > thr


# --------------------------------------------------------------- events --
def crossings(Z: pd.DataFrame, OK: pd.DataFrame, E: pd.DataFrame, z_entry: float = C.Z_ENTRY, z_exit: float = C.Z_EXIT,
              time_stop: int = C.TIME_STOP) -> list[tuple[int, int]]:
    """(row, col) of accepted cheap crossings: z <= z_entry after z > z_entry the session before, the fit passing
    and the stock eligible; one per run."""
    zv, okv, ev = Z.values, OK.values, E.values
    prev = np.vstack([np.full((1, zv.shape[1]), np.nan), zv[:-1]])
    cross = (zv <= z_entry) & (prev > z_entry) & okv & ev
    out = []
    for j in range(zv.shape[1]):
        idx = np.flatnonzero(cross[:, j])
        if not len(idx):
            continue
        run_end = -1
        for t in idx:
            if t <= run_end:
                continue
            out.append((t, j))
            # the run lasts until z > z_exit or time_stop sessions
            seg = zv[t + 1:t + 1 + time_stop, j]
            back = np.flatnonzero(seg > z_exit)
            run_end = t + (back[0] + 1 if len(back) else time_stop)
    return out


def classify(Z: pd.DataFrame, OK: pd.DataFrame, E: pd.DataFrame, ST: pd.DataFrame, sector: pd.DataFrame, FIT: pd.Series,
             z_entry: float = C.Z_ENTRY, z_exit: float = C.Z_EXIT, z_still: float = C.Z_STILL_CHEAP, z_stop: float = C.Z_STOP,
             turn_window: int = C.TURN_WINDOW, time_stop: int = C.TIME_STOP) -> pd.DataFrame:
    """Every cheap crossing with its cell ('both' / 'cheap alone'), plus every not-cheap BUY flip ('turned-up alone')."""
    cal = Z.index; cols = Z.columns
    zv, stv, okv, ev = Z.values, ST.values, OK.values, E.values
    rows = []
    for t, j in crossings(Z, OK, E, z_entry, z_exit, time_stop):
        st0 = stv[t, j]
        rec = {"cross_date": cal[t], "qid": cols[j], "sector": sector.iat[t, j], "z_cross": zv[t, j], "st_at_cross": st0,
               "fit_date": FIT.iloc[t]}
        entry = None; turned = False; days = np.nan
        if st0 == 1:
            entry, turned, days = t, True, 0
        else:
            for k in range(1, turn_window + 1):
                tt = t + k
                if tt >= len(cal):
                    break
                if zv[tt, j] <= z_stop:                       # structural stop before any turn
                    break
                if stv[tt, j] == 1 and stv[tt - 1, j] == -1:   # the flip
                    if zv[tt, j] <= z_still and okv[tt, j] and ev[tt, j]:
                        entry, turned, days = tt, True, k
                    break                                       # a flip that is no longer cheap does not qualify
        rec.update({"cell": "both" if turned else "cheap alone", "entry_date": cal[entry] if turned else cal[t],
                    "days_to_turn": days, "z_entry": zv[entry, j] if turned else zv[t, j]})
        rows.append(rec)
    # turned-up alone: BUY flips that are not cheap
    prev = np.vstack([np.full((1, zv.shape[1]), np.nan), stv[:-1]])
    flips = (stv == 1) & (prev == -1) & ev & ~np.isnan(zv) & (zv > z_still)
    for t, j in zip(*np.nonzero(flips)):
        rows.append({"cross_date": cal[t], "qid": cols[j], "sector": sector.iat[t, j], "z_cross": zv[t, j], "st_at_cross": -1,
                     "fit_date": FIT.iloc[t], "cell": "turned-up alone", "entry_date": cal[t], "days_to_turn": 0, "z_entry": zv[t, j]})
    cols_out = ["cross_date", "qid", "sector", "z_cross", "st_at_cross", "fit_date", "cell", "entry_date", "days_to_turn", "z_entry"]
    ev_df = pd.DataFrame(rows, columns=cols_out)
    ev_df["entry_date"] = pd.to_datetime(ev_df["entry_date"]); ev_df["cross_date"] = pd.to_datetime(ev_df["cross_date"])
    ev_df["year"] = ev_df["entry_date"].dt.year
    return ev_df


# -------------------------------------------------------- forward returns --
def forward_rel(entries: pd.DataFrame, CLs: pd.DataFrame, CLb: pd.DataFrame, cal: pd.DatetimeIndex, horizons=C.EVENT_H,
                delay: int = C.DELAY, cost_bps: float = C.COST_BPS, Z: pd.DataFrame | None = None, z_exit: float = C.Z_EXIT) -> pd.DataFrame:
    """Relative (stock - basket) log return over t_x+1 .. t_x+h, t_x = entry + delay; net = minus 2 x cost.
    Needs the full horizon inside the calendar (else NaN)."""
    out = entries.copy()
    pos = cal.get_indexer(out["entry_date"]) + delay
    col = CLs.columns.get_indexer(out["qid"])
    T = len(cal)
    Sv, Bv = CLs.values, CLb.values
    rt = 2 * cost_bps / 1e4
    for h in horizons:
        end = pos + h
        okh = (end < T) & (pos >= 0) & (col >= 0)
        rel = np.full(len(out), np.nan)
        p0 = np.clip(pos, 0, T - 1); p1 = np.clip(end, 0, T - 1); cc = np.clip(col, 0, None)
        rel[okh] = (Sv[p1[okh], cc[okh]] - Sv[p0[okh], cc[okh]]) - (Bv[p1[okh], cc[okh]] - Bv[p0[okh], cc[okh]])
        out[f"rel_{h}"] = rel
        out[f"net_{h}"] = rel - rt
        if Z is not None:
            zb = np.full(len(out), np.nan)
            Zv = Z.values
            for i in np.flatnonzero(okh):
                seg = Zv[p0[i] + 1:p1[i] + 1, cc[i]]
                zb[i] = float(np.nanmax(seg) >= z_exit) if np.isfinite(seg).any() else np.nan
            out[f"z_back_{h}"] = zb
    h_mark = C.H_MARK if C.H_MARK in horizons else horizons[-1]
    out["caught_up"] = out[f"net_{h_mark}"] > 0
    return out


def exclude_corporate_actions(ev: pd.DataFrame, J: pd.DataFrame, cal: pd.DatetimeIndex, width: int = C.CA_EXCLUDE) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not len(ev):
        return ev, ev
    Jv = J.values
    pos = cal.get_indexer(ev["entry_date"]); col = J.columns.get_indexer(ev["qid"])
    flag = np.zeros(len(ev), dtype=bool)
    for i, (t, j) in enumerate(zip(pos, col)):
        if t < 0 or j < 0:
            continue
        flag[i] = bool(Jv[max(0, t - width):t + width + 1, j].any())
    return ev[~flag].reset_index(drop=True), ev[flag].reset_index(drop=True)


# ---------------------------------------------------------------- stats --
def cell_stats(ev: pd.DataFrame, horizons=C.EVENT_H, by: str | None = None) -> pd.DataFrame:
    rows = []
    groups = ev.groupby(["cell"] + ([by] if by else []), observed=True)
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        for h in horizons:
            x = g[f"net_{h}"].dropna()
            daily = g.dropna(subset=[f"net_{h}"]).groupby("entry_date")[f"net_{h}"].mean()
            row = {"cell": key[0], "horizon": h, "events": len(x), "dates": len(daily),
                   "mean_rel_gross": g[f"rel_{h}"].mean(), "mean_rel_net": x.mean(), "median_rel_net": x.median(),
                   "hit_net": float((x > 0).mean()) if len(x) else np.nan, "sd": x.std(),
                   "hac_t": hac_t(daily, lags=max(h, 5)),
                   "z_back_share": g[f"z_back_{h}"].mean() if f"z_back_{h}" in g else np.nan}
            if by:
                row[by] = key[1]
            rows.append(row)
    return pd.DataFrame(rows)


def halves(ev: pd.DataFrame, cal: pd.DatetimeIndex, h: int = C.H_MARK) -> pd.DataFrame:
    mid = cal[len(cal) // 2]
    ev = ev.assign(half=np.where(ev["entry_date"] < mid, f"first (to {mid.date()})", f"second (from {mid.date()})"))
    return cell_stats(ev, (h,), by="half")


def paths(ev: pd.DataFrame, CLs: pd.DataFrame, CLb: pd.DataFrame, cal: pd.DatetimeIndex, k_max: int = C.TIME_STOP,
          delay: int = C.DELAY) -> pd.DataFrame:
    """Average cumulative relative return k = 0..k_max sessions after execution, per cell."""
    rows = []
    T = len(cal); Sv, Bv = CLs.values, CLb.values
    for cell, g in ev.groupby("cell", observed=True):
        pos = cal.get_indexer(g["entry_date"]) + delay; col = CLs.columns.get_indexer(g["qid"])
        okk = (pos + k_max < T) & (pos >= 0) & (col >= 0)
        pos, col = pos[okk], col[okk]
        if not len(pos):
            continue
        P = np.zeros((len(pos), k_max + 1))
        for k in range(1, k_max + 1):
            P[:, k] = (Sv[pos + k, col] - Sv[pos, col]) - (Bv[pos + k, col] - Bv[pos, col])
        for k in range(k_max + 1):
            rows.append({"cell": cell, "k": k, "mean_rel": P[:, k].mean(), "median_rel": np.median(P[:, k]),
                         "share_positive": float((P[:, k] > 0).mean()), "n": len(pos)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- null --
def null_draws(ev_both: pd.DataFrame, E: pd.DataFrame, sector: pd.DataFrame, CLs: pd.DataFrame, CLb: pd.DataFrame,
               cal: pd.DatetimeIndex, h: int = C.H_MARK, draws: int = C.NULL_DRAWS, seed: int = C.SEED,
               delay: int = C.DELAY, cost_bps: float = C.COST_BPS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two count-matched nulls for the 'both' cell. Returns (summary, per-draw means)."""
    rng = np.random.default_rng(seed)
    T = len(cal); Sv, Bv = CLs.values, CLb.values
    Ev = E.values
    cols = list(CLs.columns)
    pos = cal.get_indexer(ev_both["entry_date"]); col = CLs.columns.get_indexer(ev_both["qid"])
    keep = (pos >= 0) & (col >= 0) & (pos + delay + h < T)
    pos, col = pos[keep], col[keep]
    secv = sector.values
    rt = 2 * cost_bps / 1e4
    n = len(pos)
    if n == 0:
        return pd.DataFrame(), pd.DataFrame()
    # (a) same stock, random eligible date with a full horizon ahead
    elig_dates = {}
    for j in np.unique(col):
        d = np.flatnonzero(Ev[:, j])
        elig_dates[j] = d[d + delay + h < T]
    # (b) same date, random eligible peer of the same group
    def rel(p, c):
        p0 = p + delay; p1 = p0 + h
        return (Sv[p1, c] - Sv[p0, c]) - (Bv[p1, c] - Bv[p0, c]) - rt
    a_means, b_means = [], []
    actual = rel(pos, col).mean()
    for _ in range(draws):
        rp = np.array([rng.choice(elig_dates[j]) if len(elig_dates[j]) else -1 for j in col])
        m = rp >= 0
        a_means.append(rel(rp[m], col[m]).mean() if m.any() else np.nan)
        rc = np.empty(n, dtype=int)
        for i in range(n):
            t, j = pos[i], col[i]
            s = secv[t, j]
            peers = np.flatnonzero(Ev[t] & (secv[t] == s))
            peers = peers[peers != j]
            rc[i] = rng.choice(peers) if len(peers) else j
        b_means.append(rel(pos, rc).mean())
    a = np.array(a_means); b = np.array(b_means)
    per_draw = pd.DataFrame({"draw": range(draws), "same_stock_random_date": a, "same_date_random_peer": b})
    summ = pd.DataFrame([
        {"null": "same stock, random date", "events": n, "actual_mean_net": actual, "null_mean": np.nanmean(a), "null_sd": np.nanstd(a),
         "p95": np.nanpercentile(a, 95), "p99": np.nanpercentile(a, 99), "actual_percentile": float((a < actual).mean())},
        {"null": "same date, random peer", "events": n, "actual_mean_net": actual, "null_mean": np.nanmean(b), "null_sd": np.nanstd(b),
         "p95": np.nanpercentile(b, 95), "p99": np.nanpercentile(b, 99), "actual_percentile": float((b < actual).mean())},
    ])
    return summ, per_draw


# -------------------------------------------------------------- verdict --
def verdict(stats: pd.DataFrame, hv: pd.DataFrame, null_summ: pd.DataFrame, marks: dict = C.PASS_1, h: int = C.H_MARK) -> pd.DataFrame:
    both = stats[(stats["cell"] == "both") & (stats["horizon"] == h)]
    tu = stats[(stats["cell"] == "turned-up alone") & (stats["horizon"] == h)]
    rows = []
    n = int(both["events"].iloc[0]) if len(both) else 0
    mean = both["mean_rel_net"].iloc[0] if len(both) else np.nan
    hit = both["hit_net"].iloc[0] if len(both) else np.nan
    rows.append({"criterion": f"events in the 'both' cell (fewer = inconclusive)", "value": n, "mark": marks["min_events"], "passed": n >= marks["min_events"]})
    rows.append({"criterion": f"mean relative return over {h} sessions, net of {2 * C.COST_BPS} bps", "value": mean, "mark": marks["mean_rel_net_20"], "passed": bool(mean >= marks["mean_rel_net_20"]) if pd.notna(mean) else False})
    rows.append({"criterion": "caught up (net relative return > 0) share", "value": hit, "mark": marks["caught_up"], "passed": bool(hit >= marks["caught_up"]) if pd.notna(hit) else False})
    hb = hv[hv["cell"] == "both"] if len(hv) else hv
    same_sign = bool(len(hb) == 2 and (hb["mean_rel_net"] > 0).all())
    rows.append({"criterion": "positive on both halves of the sample", "value": float(same_sign), "mark": 1.0, "passed": same_sign})
    if len(null_summ):
        pct = float(null_summ["actual_percentile"].min())
        rows.append({"criterion": "above the 95th percentile of both count-matched nulls", "value": pct, "mark": marks["null_pct"], "passed": pct >= marks["null_pct"]})
    else:
        rows.append({"criterion": "above the 95th percentile of both count-matched nulls", "value": np.nan, "mark": marks["null_pct"], "passed": False})
    if marks.get("both_beats_turned_up"):
        tmean = tu["mean_rel_net"].iloc[0] if len(tu) else np.nan
        rows.append({"criterion": "'both' beats 'turned-up alone' on the mean (cheapness adds to the trend)", "value": mean - tmean if pd.notna(tmean) else np.nan,
                     "mark": 0.0, "passed": bool(pd.notna(tmean) and mean > tmean)})
    v = pd.DataFrame(rows)
    overall = "INCONCLUSIVE" if n < marks["min_events"] else ("PASS" if v["passed"].all() else "FAIL")
    v.attrs["overall"] = overall
    return v
