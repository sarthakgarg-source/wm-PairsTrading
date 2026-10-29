"""
Phase 3 - robustness.

  grid(...)          the parameter grid, one dimension at a time from the base case (GRID_FULL: the cross product).
                     Each row re-runs formation (when the window changes), the supertrend (when the bar changes),
                     the event study and the book, and reports the Phase 1 and Phase 2 headline numbers.
                     Reported whole; nothing is picked from it.
  placebo(...)       the mirror: RICH (z >= +2) and turned up. If a long book of those also beats the benchmark,
                     the edge is the trend, not the cheapness.
  block_bootstrap(.) the book's active daily returns (book - matched benchmark) resampled in blocks of BLOCK_LEN
                     sessions: the interval around the active Sharpe and the share of draws at or below zero.
  effective_n(...)   events share dates: the count of distinct entry dates and of non-overlapping events
                     (entries at least H_MARK sessions apart within a stock) beside the raw count.
  stock_pairs(...)   the stock-to-stock variant: within each group, every pair's Engle-Granger fit on the formation
                     window at each year-end, the SP_TOP best-fitting pairs per group kept for the year that
                     follows, the cheap leg entered on the same cheap + turned-up rule and measured against the
                     other leg (event study only; no book).
"""
from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd

import book as B
import config as C
import events as EV
import formation as F
import supertrend as STm
import universe as U


def _pipeline(ctx: dict, formation_w: int, z_entry: float, z_exit: float, st_bar: str, time_stop: int, n_pos: int,
              cache: dict, label: str) -> dict:
    """One full run (events + book) at one setting. ctx carries the shared panels."""
    cal = ctx["cal"]
    if ("fits", formation_w) not in cache:
        cache[("fits", formation_w)] = F.fit_all(ctx["logP"], ctx["logB"], ctx["valid"], ctx["E"], cal, window=formation_w, verbose=False)
    fits = cache[("fits", formation_w)]
    if ("z", formation_w) not in cache:
        cache[("z", formation_w)] = F.daily_z(fits, ctx["logP"], ctx["logB"], cal)
    Z, OK, FIT = cache[("z", formation_w)]
    if ("st", st_bar) not in cache:
        cache[("st", st_bar)] = STm.panel_states(ctx["P"], bar=st_bar)
    ST = cache[("st", st_bar)]
    ev = EV.classify(Z, OK, ctx["E"], ST, ctx["sector"], FIT, z_entry=z_entry, z_exit=z_exit, z_still=z_entry / 2, time_stop=time_stop)
    ev = EV.forward_rel(ev, ctx["CLs"], ctx["CLb"], cal, Z=Z, z_exit=z_exit)
    ev, _ = EV.exclude_corporate_actions(ev, ctx["J"], cal)
    st = EV.cell_stats(ev)
    both = ev[ev["cell"] == "both"]
    row = {"variant": label, "formation": formation_w, "z_entry": z_entry, "z_exit": z_exit, "st_bar": st_bar, "time_stop": time_stop, "n_pos": n_pos,
           "events_both": len(both)}
    sb = st[(st["cell"] == "both") & (st["horizon"] == C.H_MARK)]
    if len(sb):
        row.update({"p1_mean_rel_net_20": sb["mean_rel_net"].iloc[0], "p1_hit_net_20": sb["hit_net"].iloc[0], "p1_hac_t": sb["hac_t"].iloc[0]})
    D = B.Data(cal, ctx["P"]["close"], ctx["r"], ST, ctx["logP"], ctx["logB"], fits, ctx["sector"], ctx["R_uni"], ctx["E"])
    D.CLb = ctx["CLb"].values
    p = B.Params(n_pos=n_pos, time_stop=time_stop, z_exit=z_exit)
    start = ctx["start"]
    sim = B.simulate(both[["entry_date", "qid", "sector", "z_entry", "fit_date"]], D, p, start)
    mb = B.matched_benchmark(sim["secw"], ctx["R_sec"], ctx["R_uni"])
    m = B.metrics(sim["nav"]["ret"], mb)
    me = B.metrics(sim["nav"]["ret"], ctx["bench_ew"]["ret"])
    row.update({"trades": len(sim["trades"]), "net_cagr": m.get("cagr"), "net_sharpe": m.get("sharpe"), "worst_fall": m.get("worst_fall"),
                "matched_sharpe": m.get("bench_sharpe"), "matched_worst_fall": m.get("bench_worst_fall"), "active_sharpe_vs_matched": m.get("active_sharpe"),
                "ew_sharpe": me.get("bench_sharpe"), "active_cagr_vs_ew": me.get("active_cagr"),
                "beats_matched_sharpe": bool(m.get("sharpe", -9) > m.get("bench_sharpe", 9)), "beats_matched_fall": bool(m.get("worst_fall", -9) > m.get("bench_worst_fall", 9))})
    return row


def grid(ctx: dict, cache: dict, full: bool = C.GRID_FULL, verbose: bool = True) -> pd.DataFrame:
    base = {"formation": C.FORMATION, "z_entry": C.Z_ENTRY, "z_exit": C.Z_EXIT, "st_bar": C.ST_BAR, "time_stop": C.TIME_STOP, "n_pos": C.N_POS}
    settings = []
    if full:
        for combo in itertools.product(*C.GRID.values()):
            settings.append(dict(zip(C.GRID.keys(), combo)))
    else:
        settings.append(dict(base))
        for k, vals in C.GRID.items():
            for v in vals:
                if v != base[k]:
                    s = dict(base); s[k] = v; settings.append(s)
    rows = []
    for s in settings:
        label = "base" if s == base else ", ".join(f"{k}={v}" for k, v in s.items() if v != base[k])
        if verbose:
            print(f"  grid: {label}")
        rows.append(_pipeline(ctx, s["formation"], s["z_entry"], s["z_exit"], s["st_bar"], s["time_stop"], s["n_pos"], cache, label))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- placebo --
def placebo(ctx: dict, Z, OK, FIT, ST, fits) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Rich + turned up: classify on -Z with the same thresholds; the book exits on -z."""
    cal = ctx["cal"]
    ev = EV.classify(-Z, OK, ctx["E"], ST, ctx["sector"], FIT)
    ev = EV.forward_rel(ev, ctx["CLs"], ctx["CLb"], cal, Z=-Z)
    ev, _ = EV.exclude_corporate_actions(ev, ctx["J"], cal)
    ev["cell"] = ev["cell"].map({"both": "rich + turned up", "cheap alone": "rich alone", "turned-up alone": "turned-up alone"})
    st = EV.cell_stats(ev)
    both = ev[ev["cell"] == "rich + turned up"]
    D = B.Data(cal, ctx["P"]["close"], ctx["r"], ST, ctx["logP"], ctx["logB"], fits, ctx["sector"], ctx["R_uni"], ctx["E"])
    D.CLb = ctx["CLb"].values
    sim = B.simulate(both[["entry_date", "qid", "sector", "z_entry", "fit_date"]], D, B.Params(mirror=True), ctx["start"])
    mb = B.matched_benchmark(sim["secw"], ctx["R_sec"], ctx["R_uni"])
    m = B.metrics(sim["nav"]["ret"], mb)
    return ev, st, m, sim


# -------------------------------------------------------- block bootstrap --
def block_bootstrap(active: pd.Series, draws: int = C.BLOCK_BOOT, block: int = C.BLOCK_LEN, seed: int = C.SEED, ann: int = C.ANNUALISE) -> pd.DataFrame:
    a = active.dropna().values
    n = len(a)
    if n < block * 5:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    sh = np.empty(draws); mu = np.empty(draws)
    for d in range(draws):
        starts = rng.integers(0, n - block + 1, size=nb)
        x = np.concatenate([a[s:s + block] for s in starts])[:n]
        sd = x.std()
        sh[d] = x.mean() / sd * np.sqrt(ann) if sd > 0 else np.nan
        mu[d] = x.mean() * ann
    actual = a.mean() / a.std() * np.sqrt(ann) if a.std() > 0 else np.nan
    return pd.DataFrame([{"measure": "active Sharpe", "actual": actual, "mean": np.nanmean(sh), "p5": np.nanpercentile(sh, 5), "p50": np.nanpercentile(sh, 50),
                          "p95": np.nanpercentile(sh, 95), "share_le_0": float(np.nanmean(sh <= 0)), "draws": draws, "block": block},
                         {"measure": "active return (annualised)", "actual": a.mean() * ann, "mean": np.nanmean(mu), "p5": np.nanpercentile(mu, 5), "p50": np.nanpercentile(mu, 50),
                          "p95": np.nanpercentile(mu, 95), "share_le_0": float(np.nanmean(mu <= 0)), "draws": draws, "block": block}])


def effective_n(ev: pd.DataFrame, h: int = C.H_MARK) -> pd.DataFrame:
    rows = []
    for cell, g in ev.groupby("cell", observed=True):
        g = g.sort_values("entry_date")
        non_overlap = 0
        for _, gg in g.groupby("qid"):
            last = None
            for d in gg["entry_date"]:
                if last is None or (d - last).days > h * 1.5:
                    non_overlap += 1; last = d
        rows.append({"cell": cell, "events": len(g), "distinct_dates": g["entry_date"].nunique(), "distinct_stocks": g["qid"].nunique(),
                     "non_overlapping_within_stock": non_overlap, "max_events_one_date": int(g.groupby("entry_date").size().max())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------- stock pairs --
def stock_pairs(ctx: dict, ST: pd.DataFrame, window: int = C.FORMATION, top: int = C.SP_TOP, verbose: bool = True):
    """Engle-Granger within group, yearly refit, the top pairs per group; events on the cheap leg vs the other leg."""
    cal = ctx["cal"]; logP = ctx["logP"]; valid_p = ctx["P"]["close"].notna()
    E = ctx["E"]; sector = ctx["sector"]
    cols = list(logP.columns); Y = logP.values; V = valid_p.values; Ev = E.values; Sv = sector.values
    min_obs = int(C.MIN_FORMATION_SHARE * window)
    s = pd.Series(cal, index=cal)
    ye = pd.DatetimeIndex(s.groupby(cal.year).last().values)
    ye = ye[(ye >= cal[window - 1]) & (ye < cal[-1])]
    fits_rows = []
    Zp = {}; legs = {}
    for d in ye:
        k = cal.get_loc(d)
        rows = slice(k - window + 1, k + 1)
        nxt_k = cal.get_loc(ye[ye.get_loc(d) + 1]) if ye.get_loc(d) + 1 < len(ye) else len(cal) - 1
        elig = np.flatnonzero(Ev[k])
        groups: dict[str, list[int]] = {}
        for j in elig:
            g = Sv[k, j]
            if isinstance(g, str) and V[rows, j].sum() >= min_obs:
                groups.setdefault(g, []).append(j)
        n_tested = 0
        for g, members in groups.items():
            cand = []
            for a, b in itertools.combinations(members, 2):
                m = V[rows, a] & V[rows, b]
                if m.sum() < min_obs:
                    continue
                f = F.fit_one(Y[rows, a][m], Y[rows, b][m])
                n_tested += 1
                if f and f["beta"] > 0 and f["p"] < C.ADF_P_MAX and C.HALFLIFE_MIN <= f["halflife"] <= C.HALFLIFE_MAX:
                    cand.append((f["p"], a, b, f))
            cand.sort(key=lambda x: x[0])
            for p_, a, b, f in cand[:top]:
                fits_rows.append({"refit_date": d, "group": g, "a": cols[a], "b": cols[b], **f})
                key = (cols[a], cols[b])
                # the pair's z over the following year, as two legs: +z for a (a cheap when z <= -2), -z for b
                z = (Y[k + 1:nxt_k + 1, a] - f["alpha"] - f["beta"] * Y[k + 1:nxt_k + 1, b]) / f["sd"]
                for leg, sign, other in ((cols[a], 1.0, cols[b]), (cols[b], -1.0, cols[a])):
                    name = f"{leg}|{other}"
                    if name not in Zp:
                        Zp[name] = np.full(len(cal), np.nan); legs[name] = (leg, other, g)
                    Zp[name][k + 1:nxt_k + 1] = sign * z
        if verbose:
            print(f"  stock pairs {d.date()}: {n_tested:,} pairs tested, {sum(1 for r_ in fits_rows if r_['refit_date'] == d)} kept")
    if not Zp:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    names = list(Zp)
    Z = pd.DataFrame(np.column_stack([Zp[n] for n in names]), index=cal, columns=names)
    leg_q = [legs[n][0] for n in names]; oth_q = [legs[n][1] for n in names]
    ST_l = ST[leg_q].copy(); ST_l.columns = names
    E_l = (E[leg_q].values & E[oth_q].values)
    E_l = pd.DataFrame(E_l, index=cal, columns=names)
    sec_l = pd.DataFrame(np.array([[legs[n][2]] * len(cal) for n in names]).T, index=cal, columns=names)
    OK = Z.notna()
    FIT = pd.Series(pd.NaT, index=cal)
    ev = EV.classify(Z, OK, E_l, ST_l, sec_l, FIT)
    if not len(ev):
        return pd.DataFrame(fits_rows), ev, pd.DataFrame()
    CLs = ctx["CLs"][leg_q].copy(); CLs.columns = names
    CLb = ctx["CLs"][oth_q].copy(); CLb.columns = names
    ev = EV.forward_rel(ev, CLs, CLb, cal, Z=Z)
    ev["leg"] = ev["qid"].map({n: legs[n][0] for n in names}); ev["other"] = ev["qid"].map({n: legs[n][1] for n in names})
    J_l = ctx["J"][leg_q].copy(); J_l.columns = names
    ev, _ = EV.exclude_corporate_actions(ev, J_l, cal)
    st = EV.cell_stats(ev)
    return pd.DataFrame(fits_rows), ev, st
