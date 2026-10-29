"""
Run the study and write every table to outputs/tables/.

    python run_all.py                 # Phase 0 -> 1 -> (2 -> 3 only if Phase 1 passes)
    python run_all.py --force         # run every phase whatever the verdicts (the event table is still the deliverable)
    python run_all.py --fast          # 20 null draws, no grid, no stock pairs: a dry run in a few minutes
    python run_all.py --no-grid       # skip the Phase 3 grid (the slow part)

Stop rules are the plan's: Phase 0 flags a price history that cannot be trusted; Phase 1 must pass its
pre-registered marks before the book is built. Both verdicts are written as tables and into meta.json.
"""
from __future__ import annotations

import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

import book as B
import config as C
import data
import events as EV
import formation as F
import grid as G
import supertrend as STm
import universe as U

warnings.filterwarnings("ignore")


def _w(df, name, **kw):
    C.TABLES.mkdir(parents=True, exist_ok=True)
    kw.setdefault("float_format", "%.6f")
    if isinstance(df, pd.DataFrame) and df.index.name is None and isinstance(df.index, pd.RangeIndex):
        kw.setdefault("index", False)
    df.to_csv(C.TABLES / f"{name}.csv", **kw)


def run(force: bool = False, fast: bool = False, do_grid: bool = True) -> dict:
    t0 = time.time()
    C.TABLES.mkdir(parents=True, exist_ok=True)
    for f in C.TABLES.glob("*.csv"):
        f.unlink()
    null_draws = 20 if fast else C.NULL_DRAWS
    meta = {"run_at": pd.Timestamp.now().isoformat(timespec="seconds"), "fast": fast, "force": force}

    # ------------------------------------------------------------ data --
    P = data.load_prices()
    idx = data.load_index()
    cal = data.calendar(P, idx)
    P = data.on_calendar(P, cal)
    uni = data.load_universe()
    lab = data.labels(uni)
    sh = data.sector_history(uni)
    member, sector, rank = U.membership(uni, sh, cal)
    close = P["close"].reindex(columns=member.columns)
    for k in ("open", "high", "low", "turnover"):
        P[k] = P[k].reindex(columns=member.columns)
    P["close"] = close
    liquid = U.liquidity(P, cal)
    E, peers = U.eligible(member, liquid, close, sector)
    r = close / close.ffill().shift(1) - 1      # across a hole the move is credited on the session the price returns
    R_ex, R_grp, R_sec, n_sec, R_uni, n_uni = U.baskets(r, E, sector)
    logP = np.log(close)
    logB = U.basket_level(R_ex)
    valid = close.notna() & R_ex.notna()
    CLs = EV.cum_log(r); CLb = EV.cum_log(R_ex)
    J = EV.suspect_jumps(close)
    meta.update({"synthetic": data.is_synthetic(), "sessions": len(cal), "first_session": str(cal[0].date()), "last_session": str(cal[-1].date()),
                 "names": int(close.shape[1]), "names_priced": int(close.notna().any().sum())})

    # ------------------------------------------------------- Phase 0 --
    cov = data.coverage(P, cal, lab)
    _w(cov, "0_coverage")
    ms = pd.DataFrame({"members": member.sum(axis=1), "liquid_members": (member & liquid.reindex(columns=member.columns, fill_value=False)).sum(axis=1),
                       "priced_members": (member & close.notna()).sum(axis=1), "with_sector": (member & sector.notna()).sum(axis=1),
                       "eligible": E.sum(axis=1), "groups_in_use": sector.where(E).nunique(axis=1)}, index=cal)
    ms_m = ms.resample("ME").last()
    _w(ms_m, "0_universe_size")
    src = sh.groupby(sh["date"].dt.year)["sector_source"].value_counts(normalize=True).unstack().fillna(0)
    _w(src, "0_sector_source")
    grp = pd.DataFrame({"mean_eligible": n_sec.where(n_sec > 0).mean(), "min_eligible": n_sec.where(n_sec > 0).min(), "max_eligible": n_sec.max(),
                        "share_sessions_ge_min_peers": (n_sec > C.MIN_PEERS).mean()}).sort_values("mean_eligible", ascending=False)
    _w(grp, "0_groups")
    jl = J.stack(); jl = jl[jl]
    lr = np.log(close) - np.log(close.ffill().shift(1))
    jumps = pd.DataFrame({"date": jl.index.get_level_values(0), "qid": jl.index.get_level_values(1)})
    jumps["ticker"] = jumps["qid"].map(lab)
    jumps["log_return"] = [lr.at[d, c] for d, c in zip(jumps["date"], jumps["qid"])]
    jumps["in_universe"] = [bool(member.at[d, c]) for d, c in zip(jumps["date"], jumps["qid"])]
    _w(jumps, "0_suspect_jumps")
    last_print = close.apply(lambda s: s.last_valid_index())
    dl = pd.DataFrame({"ticker": lab.reindex(close.columns), "last_print": last_print})
    dl = dl[dl["last_print"] < cal[-1]]
    dl["was_member"] = [bool(member[c].any()) for c in dl.index]
    _w(dl.rename_axis("qid"), "0_delistings")
    ew_check = pd.DataFrame({"ew_universe": R_uni})
    ir = B.index_returns(idx, cal)
    if ir is not None:
        ew_check["index"] = ir
    _w(ew_check.resample("ME").apply(lambda x: np.exp(np.log1p(x).sum()) - 1), "0_ew_vs_index_monthly")
    M0 = C.PASS_0
    rows0 = []
    priced_share = float((member & close.notna()).sum().sum() / max(member.sum().sum(), 1))
    rows0.append({"check": "share of member-sessions with a price", "value": priced_share, "mark": M0["priced_share"], "passed": priced_share >= M0["priced_share"]})
    sec_share = float((member & sector.notna()).sum().sum() / max(member.sum().sum(), 1))
    rows0.append({"check": "share of member-sessions with a sector", "value": sec_share, "mark": M0["sector_share"], "passed": sec_share >= M0["sector_share"]})
    med_elig = float(ms["eligible"].median())
    mark_el = M0["min_eligible"] if not data.is_synthetic() else 50
    rows0.append({"check": "median eligible names per session", "value": med_elig, "mark": mark_el, "passed": med_elig >= mark_el})
    jrate = len(jumps[jumps["in_universe"]]) / max(member.sum().sum() / 252, 1)
    rows0.append({"check": "suspect jumps per member-year (adjustment artefacts)", "value": jrate, "mark": M0["max_jumps_per_member_year"], "passed": jrate <= M0["max_jumps_per_member_year"]})
    if ir is not None:
        both_ok = ew_check.dropna()
        cc = both_ok.corr().iloc[0, 1]
        rows0.append({"check": "correlation of the equal-weight universe with the reference index (daily)", "value": float(cc), "mark": M0["min_corr_index"], "passed": cc >= M0["min_corr_index"]})
        yrs = len(both_ok) / C.ANNUALISE
        cg = (1 + both_ok).prod() ** (1 / yrs) - 1
        gap = float(cg["ew_universe"] - cg["index"])
        rows0.append({"check": "equal-weight universe CAGR minus the index's (a broken price panel shows here)", "value": gap, "mark": M0["max_cagr_gap"], "passed": abs(gap) <= M0["max_cagr_gap"]})
        meta["ew_cagr"] = float(cg["ew_universe"]); meta["index_cagr"] = float(cg["index"])
    v0 = pd.DataFrame(rows0)
    _w(v0, "0_verdict")
    meta["phase0"] = "PASS" if v0["passed"].all() else "WARN"
    if any(v0["check"].str.startswith("equal-weight universe CAGR") & ~v0["passed"]):
        meta["phase0"] = "FAIL"
        print("0. data audit: FAIL - the equal-weight universe's CAGR is far from the index's: the price panel is not to be trusted. STOP.")
        print(v0.to_string(index=False))
        return _finish(meta, t0)
    print(f"0. data audit: {meta['phase0']} - {int(close.shape[1])} names, eligible median {med_elig:.0f}, {len(jumps)} suspect jumps, {len(dl)} names end before the last session")

    # -------------------------------------------------------- signal --
    print("1. formation: monthly fits on the trailing", C.FORMATION, "sessions")
    fits = F.fit_all(logP, logB, valid, E, cal)
    Z, OK, FIT = F.daily_z(fits, logP, logB, cal)
    fs = F.fit_summary(fits)
    _w(fs, "1_fit_summary")
    _w(F.fits_long(fits), "1_fits")
    ST = STm.panel_states(P, bar=C.ST_BAR)
    _w(pd.DataFrame({"share_in_BUY": (ST == 1).sum(axis=1) / ST.notna().sum(axis=1).replace(0, np.nan)}).resample("ME").last(), "1_breadth")
    zs = Z.where(OK & E).stack()
    _w(pd.DataFrame({"stat": ["sessions x names with a passing fit", "share z <= -2", "share z <= -1", "share |z| <= 1", "share z >= 2"],
                     "value": [len(zs), float((zs <= -2).mean()), float((zs <= -1).mean()), float((zs.abs() <= 1).mean()), float((zs >= 2).mean())]}), "1_z_distribution")
    meta["refits"] = len(fits); meta["mean_pass_share"] = float(fs["pass_share"].mean()) if len(fs) else np.nan

    # ------------------------------------------------------- Phase 1 --
    ev = EV.classify(Z, OK, E, ST, sector, FIT)
    ev = EV.forward_rel(ev, CLs, CLb, cal, Z=Z)
    ev, excl = EV.exclude_corporate_actions(ev, J, cal)
    _w(ev.assign(ticker=ev["qid"].map(lab)), "1_events"); _w(excl.assign(ticker=excl["qid"].map(lab)), "1_excluded")
    cs = EV.cell_stats(ev); _w(cs, "1_cell_stats")
    _w(EV.cell_stats(ev, (C.H_MARK,), by="year"), "1_by_year")
    _w(EV.cell_stats(ev, (C.H_MARK,), by="sector"), "1_by_sector")
    hv = EV.halves(ev, cal); _w(hv, "1_halves")
    _w(EV.paths(ev, CLs, CLb, cal), "1_paths")
    cr = ev[ev["cell"].isin(["both", "cheap alone"])]
    xs = pd.DataFrame({"crossings": [len(cr)], "share_in_SELL_at_crossing": [float((cr["st_at_cross"] == -1).mean()) if len(cr) else np.nan],
                       "share_qualifying_both": [float((cr["cell"] == "both").mean()) if len(cr) else np.nan],
                       "median_days_to_turn_both": [cr.loc[cr["cell"] == "both", "days_to_turn"].median() if len(cr) else np.nan]})
    _w(xs, "1_crossing_state")
    both = ev[ev["cell"] == "both"]
    ns, nd = EV.null_draws(both, E, sector, CLs, CLb, cal, draws=null_draws)
    _w(ns, "1_null_summary"); _w(nd, "1_null_draws")
    _w(G.effective_n(ev), "1_effective_n")
    v1 = EV.verdict(cs, hv, ns)
    _w(v1, "1_verdict")
    meta["phase1"] = v1.attrs["overall"]; meta["events_both"] = int(len(both)); meta["events"] = int(len(ev))
    print(f"   Phase 1: {meta['phase1']} - {len(both)} 'both' events of {len(cr)} crossings; {len(ev[ev['cell'] == 'turned-up alone'])} not-cheap BUY flips")
    for row in v1.itertuples():
        print(f"     {'ok  ' if row.passed else 'FAIL'} {row.criterion}: {row.value:.4f} (mark {row.mark})" if isinstance(row.value, float) else f"     {'ok  ' if row.passed else 'FAIL'} {row.criterion}: {row.value} (mark {row.mark})")

    if meta["phase1"] != "PASS" and not force:
        print("   Phase 1 did not pass: the study stops here (the event table is the deliverable). Use --force to build the book anyway.")
        meta["phase2"] = "not run"; meta["phase3"] = "not run"
        return _finish(meta, t0)

    # ------------------------------------------------------- Phase 2 --
    print("2. the book")
    start = cal[cal.get_loc(min(fits)) + 1] if fits else cal[0]
    D = B.Data(cal, close, r, ST, logP, logB, fits, sector, R_uni, E); D.CLb = CLb.values
    cols_e = ["entry_date", "qid", "sector", "z_entry", "fit_date"]
    sim = B.simulate(both[cols_e], D, B.Params(), start)
    bench = U.ew_benchmark(r, E, cal, C.COST_BPS, start)
    mb = B.matched_benchmark(sim["secw"], R_sec, R_uni)
    nav = pd.DataFrame({"book_nav": sim["nav"]["nav"], "book_ret": sim["nav"]["ret"], "n_pos": sim["nav"]["n_pos"], "invested": sim["nav"]["invested"],
                        "ew_nav": bench["nav"], "ew_ret": bench["ret"], "matched_ret": mb})
    nav["matched_nav"] = (1 + mb.where(nav["book_ret"].notna())).cumprod()
    if ir is not None:
        nav["index_ret"] = ir; nav["index_nav"] = (1 + ir.where(nav["book_ret"].notna())).cumprod()
    _w(nav.loc[start:], "2_nav")
    _w(sim["trades"].assign(ticker=sim["trades"]["qid"].map(lab)) if len(sim["trades"]) else sim["trades"], "2_trades"); _w(B.trade_summary(sim["trades"]), "2_trade_summary")
    _w(sim["secw"].loc[start:].mean().rename("mean_weight").to_frame(), "2_sector_weights")
    mm = B.metrics(sim["nav"]["ret"], mb); me = B.metrics(sim["nav"]["ret"], bench["ret"])
    mrows = [{"book": "book vs sector-matched", **mm}, {"book": "book vs equal-weight universe", **me}]
    if ir is not None:
        mrows.append({"book": "book vs index", **B.metrics(sim["nav"]["ret"], ir)})
    mrows += [{"book": "equal-weight universe", **B.metrics(bench["ret"])}, {"book": "sector-matched", **B.metrics(mb.where(sim["nav"]["ret"].notna()))}]
    if ir is not None:
        mrows.append({"book": "index", **B.metrics(ir.where(sim["nav"]["ret"].notna()))})
    M2 = pd.DataFrame(mrows).set_index("book"); M2.loc["book vs sector-matched", "turnover_per_year"] = B.turnover(sim["trades"], sim["nav"])
    _w(M2, "2_metrics")
    h2 = B.metrics_by(sim["nav"]["ret"], mb, B.halves_index(cal, start)); _w(h2, "2_halves")
    _w(B.metrics_by(sim["nav"]["ret"], mb, pd.Series(cal.year, index=cal)), "2_by_year")
    cg, be = B.breakeven(both[cols_e], D, B.Params(), R_sec, R_uni, start)
    _w(cg, "2_cost_grid")
    rb_per, rb_sum = B.random_books(sim["trades"], D, B.Params(), start, draws=null_draws, bench=mb)
    _w(rb_per, "2_random_books"); _w(rb_sum.rename_axis("measure"), "2_random_summary")
    sim_g = B.simulate(both[cols_e], D, B.Params(cost_bps=0.0), start)
    me["gross_sharpe"] = B.metrics(sim_g["nav"]["ret"])["sharpe"]
    # the 2 x 2 as books
    cells = []
    for name, sub, prm in (("both", both, B.Params()),
                           ("cheap alone", ev[ev["cell"] == "cheap alone"], B.Params(use_st_exit=False)),
                           ("turned-up alone", ev[ev["cell"] == "turned-up alone"], B.Params(use_z_exit=False))):
        s_ = B.simulate(sub[cols_e], D, prm, start)
        m_ = B.metrics(s_["nav"]["ret"], B.matched_benchmark(s_["secw"], R_sec, R_uni))
        cells.append({"cell": name, "entries": len(sub), "trades": len(s_["trades"]), **m_})
    if len(rb_sum):
        cells.append({"cell": "random (median of draws)", "entries": len(both), "trades": int(rb_per["trades"].median()), "cagr": rb_sum.loc["net_cagr", "p50"],
                      "sharpe": rb_sum.loc["net_sharpe", "p50"], "worst_fall": rb_sum.loc["net_worst_fall", "p50"], "active_sharpe": rb_sum.loc["net_active_sharpe", "p50"]})
    _w(pd.DataFrame(cells).set_index("cell"), "2_cells")
    v2 = B.verdict(mm, me, h2, rb_per, be)
    _w(v2, "2_verdict")
    meta["phase2"] = v2.attrs["overall"]; meta["trades"] = int(len(sim["trades"])); meta["breakeven_bps"] = be
    meta["book"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in mm.items()}
    print(f"   Phase 2: {meta['phase2']} - {len(sim['trades'])} trades, net CAGR {mm.get('cagr', np.nan):.1%} vs matched {mm.get('bench_cagr', np.nan):.1%}, break-even {be:.0f} bps")
    for row in v2.itertuples():
        print(f"     {'ok  ' if row.passed else 'FAIL'} {row.criterion}: {row.value:.3f} (mark {row.mark})")

    # ------------------------------------------------------- Phase 3 --
    print("3. robustness")
    ctx = {"cal": cal, "P": P, "r": r, "E": E, "sector": sector, "logP": logP, "logB": logB, "valid": valid, "CLs": CLs, "CLb": CLb, "J": J,
           "R_sec": R_sec, "R_uni": R_uni, "bench_ew": bench, "start": start}
    cache = {("fits", C.FORMATION): fits, ("z", C.FORMATION): (Z, OK, FIT), ("st", C.ST_BAR): ST}
    if C.PLACEBO:
        pev, pst, pm, psim = G.placebo(ctx, Z, OK, FIT, ST, fits)
        _w(pst, "3_placebo_stats"); _w(pd.DataFrame([pm]), "3_placebo_metrics")
        meta["placebo_active_sharpe"] = pm.get("active_sharpe")
    active = (sim["nav"]["ret"] - mb).loc[start:]
    bb = G.block_bootstrap(active); _w(bb, "3_bootstrap")
    if do_grid and not fast:
        gr = G.grid(ctx, cache); _w(gr, "3_grid")
    if C.STOCK_PAIRS and not fast:
        spf, spe, sps = G.stock_pairs(ctx, ST)
        if len(spf):
            spf = spf.assign(a_ticker=spf["a"].map(lab), b_ticker=spf["b"].map(lab))
        if len(spe):
            spe = spe.assign(leg_ticker=spe["leg"].map(lab), other_ticker=spe["other"].map(lab))
        _w(spf, "3_sp_fits"); _w(spe, "3_sp_events"); _w(sps, "3_sp_stats")
        meta["stock_pairs_events_both"] = int((spe["cell"] == "both").sum()) if len(spe) else 0
    meta["phase3"] = "done"
    return _finish(meta, t0)


def _finish(meta: dict, t0: float) -> dict:
    meta["elapsed_s"] = round(time.time() - t0, 1)
    (C.TABLES / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"done in {meta['elapsed_s']:.0f}s -> {C.rel(C.TABLES)}/")
    return meta


if __name__ == "__main__":
    run(force="--force" in sys.argv, fast="--fast" in sys.argv, do_grid="--no-grid" not in sys.argv)
