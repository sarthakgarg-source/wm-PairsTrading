"""
Phase 2 - the long-only book, and everything it is judged against.

  simulate(entries, D, p)  a daily simulation on the session calendar:
      * a signal dated t (the entry_date of an event) is executed at the close of t + DELAY
      * up to N_POS positions, at most SECTOR_CAP per group; candidates ranked by z (most negative first)
      * each new position is bought with NAV / N_POS (as much of it as the idle sleeve has); no rebalancing while held
      * exits are decided on the session before and executed at the next close:
            caught up        z under the ENTRY fit >= Z_EXIT
            turn failed      the stock's supertrend back in SELL
            structural       z under the entry fit <= Z_STOP
            time             held TIME_STOP sessions
            fixed            held `hold` sessions (the count-matched random books)
            delisted         the stock has no further print: the position is closed at its last price
      * COST_BPS per side on every stock trade; the idle sleeve earns the equal-weight universe's return
        (the book is fully invested; what the slots do not hold, the benchmark holds)
      * returns: nav, the daily sector weights (for the sector-matched benchmark) and every trade

  Benchmarks:
      ew        the equal-weight universe book (universe.ew_benchmark) at the same cost
      matched   each day the book's own sector weights on the group returns, the remainder on the universe:
                the "selection-only" yardstick - beating it means picking the right stocks inside each group
      index     the reference price index (NIFTY 500), for reference only

  metrics(ret, bench_ret)   CAGR, vol, Sharpe (rf = 0), worst fall, hit rate; active = book - bench
  breakeven(...)            the one-way cost at which the book's net CAGR equals the sector-matched benchmark's
  random_books(...)         count-matched random books: every actual trade replaced by a random eligible peer of the
                            same group on the same entry date, held the same number of sessions; NULL_DRAWS of them
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

import config as C
import universe as U


@dataclass
class Params:
    n_pos: int = C.N_POS
    sector_cap: int = C.SECTOR_CAP
    cost_bps: float = C.COST_BPS
    delay: int = C.DELAY
    time_stop: int = C.TIME_STOP
    z_exit: float = C.Z_EXIT
    z_stop: float = C.Z_STOP
    use_z_exit: bool = True
    use_st_exit: bool = True
    mirror: bool = False          # the placebo: z sign flipped (rich instead of cheap)


class Data:
    """Everything simulate() needs, as numpy arrays on the session calendar."""

    def __init__(self, cal, close, r, ST, logP, logB, fits, sector, R_uni, E):
        self.cal = cal
        self.cols = list(close.columns)
        self.col_of = {c: i for i, c in enumerate(self.cols)}
        self.close = close.values
        self.r = np.nan_to_num(r.values, nan=0.0)
        self.ST = ST.reindex(columns=close.columns).values
        self.logP = logP.values; self.logB = logB.values
        self.fits = fits
        self.sector = sector.reindex(columns=close.columns).values
        self.R_uni = np.nan_to_num(R_uni.reindex(cal).values, nan=0.0)
        self.E = E.reindex(columns=close.columns).values
        # r is credited on the session a price returns after a hole (run_all: close / close.ffill().shift(1) - 1)
        # the last session with a print, per stock (for the delisting exit)
        has = ~np.isnan(self.close)
        self.last_print = np.where(has.any(axis=0), has.shape[0] - 1 - np.argmax(has[::-1], axis=0), -1)
        self.CLb = None   # set by the caller when trades need the basket's return (events.cum_log of R_ex)

    def z_series(self, fit_date, j, t0, t1) -> np.ndarray:
        f = self.fits[fit_date]
        qid = self.cols[j]
        if qid not in f.index:
            return np.full(t1 - t0, np.nan)
        a, b, s = f.at[qid, "alpha"], f.at[qid, "beta"], f.at[qid, "sd"]
        return (self.logP[t0:t1, j] - a - b * self.logB[t0:t1, j]) / s


def simulate(entries: pd.DataFrame, D: Data, p: Params, start: pd.Timestamp | None = None) -> dict:
    cal = D.cal; T = len(cal)
    c = p.cost_bps / 1e4
    ent = entries.copy()
    ent["t_sig"] = cal.get_indexer(ent["entry_date"])
    ent["j"] = ent["qid"].map(D.col_of)
    ent = ent.dropna(subset=["j"]); ent["j"] = ent["j"].astype(int)
    ent = ent[ent["t_sig"] >= 0]
    has_hold = "hold" in ent.columns
    by_exec = {}
    for row in ent.itertuples(index=False):
        by_exec.setdefault(row.t_sig + p.delay, []).append(row)
    i0 = cal.get_loc(start) if start is not None else 0
    open_pos: dict[int, dict] = {}
    nav = np.full(T, np.nan); ret = np.full(T, np.nan); n_open = np.zeros(T); invested = np.zeros(T); cost_paid = np.zeros(T)
    secw_rows = []
    trades = []
    idle = 1.0; nav_prev = 1.0
    sectors_seen = sorted({s for s in pd.unique(D.sector.ravel()) if isinstance(s, str)})
    sec_idx = {s: i for i, s in enumerate(sectors_seen)}
    secw = np.zeros((T, len(sectors_seen)))
    for t in range(i0, T):
        # the idle sleeve earns the universe
        idle *= (1 + D.R_uni[t])
        day_cost = 0.0
        # 1. exits decided on t-1, executed at close t
        for j in list(open_pos):
            q = open_pos[j]
            held = t - q["t_exec"]                 # sessions held as of today's close; the decision uses the session before
            reason = None
            if held >= 1:
                k1 = t - 1 - q["t_exec"]           # offset of the decision session (t-1) from the entry execution
                z1 = q["z"][k1] if 0 <= k1 < len(q["z"]) else np.nan
                if p.mirror and np.isfinite(z1):
                    z1 = -z1
                if has_hold and pd.notna(q["hold"]) and held >= q["hold"]:
                    reason = "fixed"
                elif p.use_z_exit and np.isfinite(z1) and z1 >= p.z_exit:
                    reason = "caught up"
                elif p.use_z_exit and np.isfinite(z1) and z1 <= p.z_stop:
                    reason = "structural"
                elif p.use_st_exit and D.ST[t - 1, j] == -1:
                    reason = "turn failed"
                elif held >= p.time_stop:
                    reason = "time"
            if t > D.last_print[j]:
                reason = "delisted"
            if reason is None:
                continue
            px = D.close[t, j]
            if np.isnan(px):
                if reason == "delisted":
                    px = q["last_px"]            # closed at its last print, no further return
                else:
                    continue                     # no print today: cannot trade, try again tomorrow
            proceeds = q["shares"] * px * (1 - c)
            day_cost += q["shares"] * px * c
            idle += proceeds
            gross = px / q["px_in"] - 1
            net = (1 - c) * (1 + gross) * (1 - c) - 1
            tr = {"qid": D.cols[j],"sector": q["sector"], "entry_signal": cal[q["t_sig"]], "entry_exec": cal[q["t_exec"]],
                  "exit_signal": cal[t - 1], "exit_exec": cal[t], "reason": reason, "hold": held, "z_entry": q["z_entry"],
                  "z_exit": q["z"][min(t - 1 - q["t_exec"], len(q["z"]) - 1)] if len(q["z"]) else np.nan,
                  "px_in": q["px_in"], "px_out": px, "gross": gross, "net": net, "weight_in": q["w_in"]}
            if D.CLb is not None:
                tr["basket"] = float(np.exp(D.CLb[t, j] - D.CLb[q["t_exec"], j]) - 1)
                tr["rel_net"] = net - tr["basket"]
            trades.append(tr)
            del open_pos[j]
        # 2. entries executed at close t
        cands = by_exec.get(t, [])
        if cands and len(open_pos) < p.n_pos:
            cands = sorted(cands, key=lambda r: (r.z_entry if not p.mirror else -r.z_entry))
            cur_nav = sum(q["shares"] * (D.close[t, q["j"]] if not np.isnan(D.close[t, q["j"]]) else q["last_px"]) for q in open_pos.values()) + idle
            sec_count: dict[str, int] = {}
            for q in open_pos.values():
                sec_count[q["sector"]] = sec_count.get(q["sector"], 0) + 1
            for r_ in cands:
                if len(open_pos) >= p.n_pos:
                    break
                j = r_.j
                if j in open_pos or not D.E[t - 1, j]:
                    continue
                s = r_.sector if isinstance(r_.sector, str) else D.sector[t, j]
                if sec_count.get(s, 0) >= p.sector_cap:
                    continue
                px = D.close[t, j]
                if np.isnan(px) or px <= 0:
                    continue
                cash = min(cur_nav / p.n_pos, idle)
                if cash <= 0:
                    break
                shares = cash * (1 - c) / px
                day_cost += cash * c
                idle -= cash
                z = D.z_series(r_.fit_date, j, t, min(T, t + p.time_stop + 2)) if (r_.fit_date in D.fits) else np.full(p.time_stop + 2, np.nan)
                open_pos[j] = {"j": j, "shares": shares, "px_in": px, "last_px": px, "t_exec": t, "t_sig": r_.t_sig, "sector": s,
                               "z_entry": r_.z_entry, "z": z, "w_in": cash / cur_nav, "hold": getattr(r_, "hold", np.nan)}
                sec_count[s] = sec_count.get(s, 0) + 1
        # 3. mark
        val = 0.0
        for q in open_pos.values():
            px = D.close[t, q["j"]]
            if not np.isnan(px):
                q["last_px"] = px
            val += q["shares"] * q["last_px"]
            if q["sector"] in sec_idx:
                secw[t, sec_idx[q["sector"]]] += q["shares"] * q["last_px"]
        nav[t] = val + idle
        if nav[t] > 0:
            secw[t] /= nav[t]
        ret[t] = nav[t] / nav_prev - 1
        n_open[t] = len(open_pos); invested[t] = val / nav[t] if nav[t] > 0 else 0; cost_paid[t] = day_cost / nav_prev
        nav_prev = nav[t]
    out = pd.DataFrame({"nav": nav, "ret": ret, "n_pos": n_open, "invested": invested, "cost": cost_paid}, index=cal)
    tr = pd.DataFrame(trades)
    return {"nav": out, "trades": tr, "secw": pd.DataFrame(secw, index=cal, columns=sectors_seen)}


# ------------------------------------------------------------ benchmarks --
def matched_benchmark(secw: pd.DataFrame, R_sec: pd.DataFrame, R_uni: pd.Series) -> pd.Series:
    """Each day the book's sector weights of the session before on the group returns, the rest on the universe."""
    w = secw.shift(1).fillna(0.0)
    cols = [c for c in w.columns if c in R_sec.columns]
    Rs = R_sec.reindex(w.index)[cols].fillna(0.0)
    ret = (w[cols] * Rs).sum(axis=1) + (1 - w[cols].sum(axis=1)) * R_uni.reindex(w.index).fillna(0.0)
    return ret.rename("matched")


def index_returns(idx: pd.DataFrame | None, cal: pd.DatetimeIndex, name: str = C.INDEX_REF) -> pd.Series | None:
    if idx is None or name not in idx.columns:
        return None
    s = idx[name].reindex(cal)
    return (s / s.shift(1) - 1).rename("index")


# --------------------------------------------------------------- metrics --
def drawdown(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1


def metrics(ret: pd.Series, bench: pd.Series | None = None, ann: int = C.ANNUALISE) -> dict:
    r = ret.dropna()
    if len(r) < 20:
        return {}
    nav = (1 + r).cumprod()
    yrs = len(r) / ann
    out = {"sessions": len(r), "years": yrs, "total": nav.iloc[-1] - 1, "cagr": nav.iloc[-1] ** (1 / yrs) - 1,
           "vol": r.std() * np.sqrt(ann), "sharpe": (r.mean() / r.std() * np.sqrt(ann)) if r.std() > 0 else np.nan,
           "worst_fall": drawdown(nav).min(), "hit": float((r > 0).mean())}
    out["calmar"] = out["cagr"] / abs(out["worst_fall"]) if out["worst_fall"] < 0 else np.nan
    if bench is not None:
        b = bench.reindex(r.index).fillna(0.0)
        a = r - b
        bn = (1 + b).cumprod()
        out.update({"bench_cagr": bn.iloc[-1] ** (1 / yrs) - 1, "bench_sharpe": (b.mean() / b.std() * np.sqrt(ann)) if b.std() > 0 else np.nan,
                    "bench_worst_fall": drawdown(bn).min(), "active_cagr": out["cagr"] - (bn.iloc[-1] ** (1 / yrs) - 1),
                    "tracking_error": a.std() * np.sqrt(ann), "active_sharpe": (a.mean() / a.std() * np.sqrt(ann)) if a.std() > 0 else np.nan,
                    "share_days_ahead": float((a > 0).mean())})
    return out


def metrics_by(ret: pd.Series, bench: pd.Series | None, groups: pd.Series) -> pd.DataFrame:
    rows = []
    for g, idx in groups.groupby(groups).groups.items():
        m = metrics(ret.reindex(idx), bench.reindex(idx) if bench is not None else None)
        if m:
            rows.append({"group": g, **m})
    return pd.DataFrame(rows).set_index("group") if rows else pd.DataFrame()


def halves_index(cal: pd.DatetimeIndex, start: pd.Timestamp) -> pd.Series:
    sub = cal[cal >= start]
    mid = sub[len(sub) // 2]
    return pd.Series(np.where(cal < mid, f"first (to {mid.date()})", f"second (from {mid.date()})"), index=cal)


def trade_summary(tr: pd.DataFrame) -> pd.DataFrame:
    if not len(tr):
        return pd.DataFrame()
    rows = [{"group": "all", "trades": len(tr), "mean_net": tr["net"].mean(), "median_net": tr["net"].median(), "hit_net": float((tr["net"] > 0).mean()),
             "mean_rel_net": tr["rel_net"].mean() if "rel_net" in tr else np.nan, "hit_rel_net": float((tr["rel_net"] > 0).mean()) if "rel_net" in tr else np.nan,
             "mean_hold": tr["hold"].mean(), "median_hold": tr["hold"].median(), "worst": tr["net"].min(), "best": tr["net"].max()}]
    for reason, g in tr.groupby("reason"):
        rows.append({"group": f"exit: {reason}", "trades": len(g), "share": len(g) / len(tr), "mean_net": g["net"].mean(), "median_net": g["net"].median(),
                     "hit_net": float((g["net"] > 0).mean()), "mean_rel_net": g["rel_net"].mean() if "rel_net" in g else np.nan,
                     "hit_rel_net": float((g["rel_net"] > 0).mean()) if "rel_net" in g else np.nan, "mean_hold": g["hold"].mean(), "median_hold": g["hold"].median()})
    return pd.DataFrame(rows).set_index("group")


def turnover(tr: pd.DataFrame, nav: pd.DataFrame) -> float:
    """One-way turnover per year: the sum of entry weights over the years run."""
    if not len(tr):
        return np.nan
    yrs = nav["ret"].notna().sum() / C.ANNUALISE
    return float(tr["weight_in"].sum() / yrs) if yrs > 0 else np.nan


# ------------------------------------------------------------- break-even --
def breakeven(entries: pd.DataFrame, D: Data, p: Params, R_sec: pd.DataFrame, R_uni: pd.Series, start: pd.Timestamp,
              costs=C.COST_SENS) -> tuple[pd.DataFrame, float]:
    """The one-way cost at which the book's net CAGR equals its sector-matched benchmark's. The benchmark holds the
    same idle sleeve (the universe, cost-free) and blind picks inside the book's groups, so the comparison isolates
    the slots: what the picks earn over their groups, less what they cost. (Against the equal-weight universe book,
    which pays a monthly rebalance, a mostly idle book would 'win' at any cost - the review of 2026-10-28.)"""
    rows = []
    for cb in costs:
        sim = simulate(entries, D, replace(p, cost_bps=cb), start)
        m = metrics(sim["nav"]["ret"], matched_benchmark(sim["secw"], R_sec, R_uni))
        rows.append({"cost_bps": cb, "book_cagr": m.get("cagr"), "bench_cagr": m.get("bench_cagr"), "active_cagr": m.get("active_cagr"),
                     "book_sharpe": m.get("sharpe"), "bench_sharpe": m.get("bench_sharpe"), "worst_fall": m.get("worst_fall")})
    df = pd.DataFrame(rows)
    be = np.nan
    a = df["active_cagr"].values; cc = df["cost_bps"].values
    if np.isfinite(a).all():
        if a[0] <= 0:
            be = 0.0
        elif a[-1] > 0:
            be = np.inf
        else:
            k = int(np.argmax(a <= 0))
            x0, x1, y0, y1 = cc[k - 1], cc[k], a[k - 1], a[k]
            be = float(x0 + (0 - y0) * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x1)
    return df, be


# ------------------------------------------------------- random books --
def random_books(tr: pd.DataFrame, D: Data, p: Params, start: pd.Timestamp, draws: int = C.NULL_DRAWS, seed: int = C.SEED,
                 bench: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each actual trade replaced by a random eligible peer of the same group on the same signal date, held the
    same number of sessions. Returns (per-draw metrics, summary)."""
    rng = np.random.default_rng(seed)
    if not len(tr):
        return pd.DataFrame(), pd.DataFrame()
    cal = D.cal
    tsig = cal.get_indexer(tr["entry_signal"])
    rows = []
    for d in range(draws):
        ent = []
        for i, row in enumerate(tr.itertuples(index=False)):
            t = tsig[i]
            if t < 0:
                continue
            s = row.sector
            peers = np.flatnonzero(D.E[t] & (D.sector[t] == s))
            jself = D.col_of.get(row.qid, -1)
            peers = peers[peers != jself]
            if not len(peers):
                continue
            j = int(rng.choice(peers))
            ent.append({"entry_date": cal[t], "qid": D.cols[j],"sector": s, "z_entry": rng.uniform(-3, -2), "fit_date": pd.NaT, "hold": row.hold})
        if not ent:
            continue
        ent = pd.DataFrame(ent)
        sim_g = simulate(ent, D, replace(p, cost_bps=0.0, use_z_exit=False, use_st_exit=False), start)
        sim_n = simulate(ent, D, replace(p, use_z_exit=False, use_st_exit=False), start)
        mg = metrics(sim_g["nav"]["ret"], bench); mn = metrics(sim_n["nav"]["ret"], bench)
        rows.append({"draw": d, "gross_sharpe": mg.get("sharpe"), "gross_cagr": mg.get("cagr"), "net_sharpe": mn.get("sharpe"), "net_cagr": mn.get("cagr"),
                     "net_worst_fall": mn.get("worst_fall"), "net_active_sharpe": mn.get("active_sharpe"), "trades": len(sim_n["trades"])})
    per = pd.DataFrame(rows)
    summ = pd.DataFrame({k: {"mean": per[k].mean(), "sd": per[k].std(), "p5": per[k].quantile(0.05), "p50": per[k].median(), "p95": per[k].quantile(0.95)}
                         for k in ("gross_sharpe", "gross_cagr", "net_sharpe", "net_cagr", "net_worst_fall", "net_active_sharpe")}).T
    return per, summ


# ---------------------------------------------------------------- verdict --
def verdict(m_matched: dict, m_ew: dict, halves: pd.DataFrame, null_per: pd.DataFrame, be: float, marks: dict = C.PASS_2) -> pd.DataFrame:
    rows = []
    rows.append({"criterion": "net Sharpe above the sector-matched benchmark's", "value": m_matched.get("sharpe", np.nan) - m_matched.get("bench_sharpe", np.nan), "mark": 0.0,
                 "passed": bool(m_matched.get("sharpe", -9) > m_matched.get("bench_sharpe", 9))})
    rows.append({"criterion": "worst fall shallower than the sector-matched benchmark's", "value": m_matched.get("worst_fall", np.nan) - m_matched.get("bench_worst_fall", np.nan), "mark": 0.0,
                 "passed": bool(m_matched.get("worst_fall", -9) > m_matched.get("bench_worst_fall", 9))})
    ok_h = bool(len(halves) == 2 and (halves["sharpe"] > halves["bench_sharpe"]).all())
    rows.append({"criterion": "net Sharpe above the benchmark's on both halves", "value": float(ok_h), "mark": 1.0, "passed": ok_h})
    if len(null_per):
        pct = float((null_per["gross_sharpe"] < m_ew.get("gross_sharpe", np.nan)).mean())
        rows.append({"criterion": "gross Sharpe above the 95th percentile of the count-matched random books", "value": pct, "mark": marks["null_pct"], "passed": pct >= marks["null_pct"]})
    rows.append({"criterion": "break-even one-way cost (bps) vs the sector-matched benchmark", "value": be, "mark": marks["breakeven_bps"], "passed": bool(be >= marks["breakeven_bps"])})
    v = pd.DataFrame(rows)
    v.attrs["overall"] = "PASS" if v["passed"].all() else "FAIL"
    return v
