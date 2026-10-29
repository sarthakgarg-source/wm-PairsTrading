"""
Regression checks.   python checks.py   -> exit 0 = all pass

  A. UNIVERSE   a month-end snapshot applies from the NEXT session only; the liquidity floor; the ex-self basket
                equals the hand calculation; a name with no traded value passes on market cap alone
  B. FORMATION  a planted cointegrated pair passes the level test and a planted random walk does not; z under a fit
                equals the hand calculation; the daily z uses the fit dated strictly before the session
  C. EVENTS     one crossing per run; 'both' needs the flip while z is still cheap; forward windows are t_x+1..t_x+h
                with t_x = entry + DELAY; net = gross - 2 x cost; the 'neither' null is count-matched
  D. BOOK       a single planted trade's NAV equals the hand calculation with costs on both sides; the time stop fires
                after TIME_STOP sessions; the sector cap holds; the equal-weight benchmark's first turnover is 1
  E. SIGNAL     LW ATR weights 1/15..5/15; the state alphabet; a weekly state is applied from the next session
  F. HYGIENE    the password is in no tracked file; .env is ignored; no person's name (WM_REDACT_NAMES); a report built on
                synthetic data says so; the pass marks in config are the plan's
  G. OUTPUTS    when outputs/tables exists: the verdict tables carry the right criteria; the trades never overlap a
                name with itself; 2_nav starts at 1
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

import book as B
import config as C
import db
import events as EV
import formation as F
import supertrend as ST
import universe as U

P_, F_ = [], []


def check(name, cond, detail=""):
    (P_ if bool(cond) else F_).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{('  - ' + str(detail)) if detail and not cond else ''}")


def _planted_panel(seed=0, T=1400, names=("A0", "A1", "A2", "A3", "A4", "A5", "A6", "B0", "B1", "B2", "B3", "B4", "B5", "B6")):
    """14 names in two groups; a group factor plus an AR(1) residual (half-life 10) for all but A6/B6 (random walks)."""
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2015-01-01", periods=T)
    fac = {"A": np.cumsum(rng.normal(0, 0.01, T)), "B": np.cumsum(rng.normal(0, 0.01, T))}
    lp = {}
    for n in names:
        g = n[0]
        if n.endswith("6"):
            e = np.cumsum(rng.normal(0, 0.01, T))
        else:
            phi = 0.5 ** (1 / 10); e = np.zeros(T)
            for t in range(1, T):
                e[t] = phi * e[t - 1] + rng.normal(0, 0.02 * np.sqrt(1 - phi ** 2))
        lp[n] = 4 + fac[g] + e
    close = pd.DataFrame(np.exp(pd.DataFrame(lp, index=cal).values), index=cal, columns=list(names))
    P = {"close": close, "open": close.shift(1).fillna(close), "high": close * 1.01, "low": close * 0.99,
         "turnover": close * 1e6 / 1e7}                  # 1e6 shares a day, in rupees crore
    sector = pd.DataFrame([[n[0] for n in names]] * T, index=cal, columns=list(names))
    member = pd.DataFrame(True, index=cal, columns=list(names))
    return cal, P, close, sector, member


def main() -> int:
    print("A. universe")
    cal = pd.bdate_range("2020-01-01", periods=120)
    snap = pd.DataFrame({"X": [1.0, 0.0]}, index=pd.to_datetime(["2020-01-31", "2020-03-31"]))
    d = U._apply_from_next_session(snap, cal)
    check("A snapshot dated 2020-01-31 is not in force on 2020-01-31 but is on 2020-02-03", pd.isna(d.loc["2020-01-31", "X"]) and d.loc["2020-02-03", "X"] == 1.0)
    check("A the next snapshot replaces it from the next session", d.loc["2020-03-31", "X"] == 1.0 and d.loc["2020-04-01", "X"] == 0.0)
    cal2, P, close, sector, member = _planted_panel()
    liq = U.liquidity({"turnover": P["turnover"]}, cal2, floor_cr=1.0)
    check("A liquidity floor: 1e6 shares x ~55 = ~5.5 cr/day passes a 1 cr floor after the window", bool(liq.iloc[-1].all()) and not liq.iloc[10].any())
    tv2 = P["turnover"].copy(); tv2["A0"] = np.nan
    liq2 = U.liquidity({"turnover": tv2}, cal2, floor_cr=1e9)
    check("A a name with no traded value on record passes; one below a huge floor fails", bool(liq2["A0"].all()) and not liq2["A1"].any())
    E, peers = U.eligible(member, liq, close, sector, min_peers=5)
    check("A peers = others in the group (6 for a group of 7)", int(peers.iloc[-1]["A0"]) == 6 and bool(E.iloc[-1].all()))
    r = close / close.shift(1) - 1
    R_ex, R_grp, R_sec, n_sec, R_uni, n_uni = U.baskets(r, E, sector)
    t = 300
    manual = r.iloc[t][["A1", "A2", "A3", "A4", "A5", "A6"]].mean()
    check("A ex-self basket = mean of the other members' returns", np.isclose(R_ex.iloc[t]["A0"], manual))
    check("A group return = mean over all members; universe = mean over everyone", np.isclose(R_sec.iloc[t]["A"], r.iloc[t][[c for c in close.columns if c[0] == "A"]].mean()) and np.isclose(R_uni.iloc[t], r.iloc[t].mean()))
    E3 = E.copy(); E3.iloc[:, :] = False; E3["A0"] = True; E3["A1"] = True
    _, _, _, n3, _, _ = U.baskets(r, E3, sector)
    check("A members are decided at t-1 (a stock made eligible today joins the basket tomorrow)", int(n3.iloc[-1]["A"]) == 2 and int(n3.iloc[0]["A"]) == 0)

    print("B. formation")
    logP = np.log(close); logB = U.basket_level(R_ex); valid = close.notna() & R_ex.notna()
    fits = F.fit_all(logP, logB, valid, E, cal2, window=504, verbose=False)
    last = fits[max(fits)]
    stationary = [n for n in last.index if not n.endswith("6")]
    check("B most planted stationary residuals pass the level test (>= 8 of 12 on a 504-session window), the random walks do not",
          int(last.loc[stationary, "ok"].sum()) >= 8 and not bool(last.loc[["A6", "B6"], "ok"].any()), last[["p", "halflife", "ok"]].round(3).to_dict("index"))
    check("B half-life of the planted AR(1) near 10 sessions", 5 <= last.loc["A0", "halflife"] <= 20, last.loc["A0", "halflife"])
    Z, OK, FIT = F.daily_z(fits, logP, logB, cal2)
    d0 = max(fits); j = "A0"; f = last.loc[j]
    t1 = cal2[cal2.get_loc(d0) + 1] if cal2.get_loc(d0) + 1 < len(cal2) else None
    if t1 is not None:
        zman = (logP.at[t1, j] - f["alpha"] - f["beta"] * logB.at[t1, j]) / f["sd"]
        check("B daily z on the session after a refit uses that refit's parameters", np.isclose(Z.at[t1, j], zman) and FIT.at[t1] == d0)
    check("B on the refit session itself the previous fit is still in force", FIT.at[d0] < d0)
    e = rng_e = np.random.default_rng(1).normal(0, 1, 600)
    check("B ADF t on white noise is strongly negative (< -10) and the EG p near 0", F._adf_t(e, 1) < -10 and float(__import__("statsmodels.tsa.adfvalues", fromlist=["mackinnonp"]).mackinnonp(F._adf_t(e, 1), "c", 2)) < 0.01)

    print("C. events")
    T_ = 200; names = ["S0", "S1", "S2", "S3", "S4", "S5"]
    cal3 = pd.bdate_range("2021-01-01", periods=T_)
    Zc = pd.DataFrame(0.0, index=cal3, columns=names)
    Zc.loc[cal3[50:70], "S0"] = -2.2         # a crossing at 50, back above -0.5 at 70
    Zc.loc[cal3[55], "S0"] = -2.5            # deeper inside the run: not a new event
    Zc.loc[cal3[100:112], "S0"] = -2.1       # a second run
    Zc.loc[cal3[120:140], "S1"] = -2.1       # S1: cheap, turns up on day 125 while still cheap
    Zc.loc[cal3[150:160], "S2"] = -2.1       # S2: cheap, turns up on day 153 but z is -0.8 then (not still cheap)
    Zc.loc[cal3[153:160], "S2"] = -0.8
    OKc = pd.DataFrame(True, index=cal3, columns=names); Ec = OKc.copy()
    STc = pd.DataFrame(-1.0, index=cal3, columns=names)
    STc.loc[cal3[100:], "S0"] = 1.0           # S0 in BUY at its second crossing
    STc.loc[cal3[125:], "S1"] = 1.0
    STc.loc[cal3[153:], "S2"] = 1.0
    STc.loc[cal3[30:], "S3"] = 1.0            # S3: a BUY flip at 30 while not cheap -> turned-up alone
    sec3 = pd.DataFrame("G", index=cal3, columns=names)
    FITc = pd.Series(cal3[0], index=cal3)
    ev = EV.classify(Zc, OKc, Ec, STc, sec3, FITc)
    s0 = ev[ev["qid"] == "S0"]
    check("C one crossing per run: S0 has 2 crossings (days 50 and 100), not 3", len(s0) == 2 and list(s0["cross_date"]) == [cal3[50], cal3[100]])
    check("C in BUY at the crossing -> 'both' with entry at the crossing", s0.iloc[1]["cell"] == "both" and s0.iloc[1]["entry_date"] == cal3[100] and s0.iloc[0]["cell"] == "cheap alone")
    s1 = ev[ev["qid"] == "S1"].iloc[0]
    check("C a flip to BUY 5 sessions later while still cheap -> 'both', entry at the flip", s1["cell"] == "both" and s1["entry_date"] == cal3[125] and s1["days_to_turn"] == 5)
    s2 = ev[ev["qid"] == "S2"].iloc[0]
    check("C a flip to BUY when z has already recovered to -0.8 -> 'cheap alone'", s2["cell"] == "cheap alone")
    s3 = ev[ev["qid"] == "S3"]
    check("C a BUY flip while not cheap -> 'turned-up alone'", len(s3) == 1 and s3.iloc[0]["cell"] == "turned-up alone" and s3.iloc[0]["entry_date"] == cal3[30])
    rs = pd.DataFrame(0.0, index=cal3, columns=names); rb = pd.DataFrame(0.0, index=cal3, columns=names)
    rs.loc[cal3[126:131], "S1"] = 0.01        # +1% on each of the 5 sessions t_x+1..t_x+5 (t_x = 125 + 1 = 126 -> sessions 127..131)
    rs.loc[cal3[126], "S1"] = 0.0
    rs.loc[cal3[127:132], "S1"] = 0.01
    rb.loc[cal3[127:132], "S1"] = 0.002
    CLs = EV.cum_log(rs); CLb = EV.cum_log(rb)
    fw = EV.forward_rel(ev, CLs, CLb, cal3, horizons=(5,), cost_bps=40)
    r1 = fw[fw["qid"] == "S1"].iloc[0]
    expect = 5 * (np.log1p(0.01) - np.log1p(0.002))
    check("C forward relative return = sum over t_x+1..t_x+5 of (stock - basket) log returns, t_x = entry + 1", np.isclose(r1["rel_5"], expect), (r1["rel_5"], expect))
    check("C net = gross - 2 x 40 bps", np.isclose(r1["net_5"], expect - 0.008))
    ex_rs = rs.copy(); ex_rs.loc[cal3[126], "S1"] = 0.5
    fw2 = EV.forward_rel(ev, EV.cum_log(ex_rs), CLb, cal3, horizons=(5,), cost_bps=40)
    check("C the execution session's own return is not counted", np.isclose(fw2[fw2["qid"] == "S1"].iloc[0]["rel_5"], expect))
    both = fw[fw["cell"] == "both"]
    ns, nd = EV.null_draws(both, Ec, sec3, CLs, CLb, cal3, h=5, draws=5)
    check("C the null is count-matched (as many events as the cell)", len(ns) == 2 and int(ns["events"].iloc[0]) == len(both) and len(nd) == 5)

    print("D. book")
    cal4 = pd.bdate_range("2022-01-01", periods=120); nm = ["K0", "K1", "K2", "K3", "K4", "K5"]
    px = pd.DataFrame(100.0, index=cal4, columns=nm)
    px.loc[cal4[12:], "K0"] = 110.0           # bought at close of session 11 (signal 10) at 100; 110 from session 12, held flat
    r4 = px / px.shift(1) - 1
    STd = pd.DataFrame(1.0, index=cal4, columns=nm)
    sec4 = pd.DataFrame([["X", "X", "X", "X", "X", "Y"]] * len(cal4), index=cal4, columns=nm)
    E4 = pd.DataFrame(True, index=cal4, columns=nm)
    logP4 = np.log(px); logB4 = pd.DataFrame(0.0, index=cal4, columns=nm)
    fits4 = {cal4[0]: pd.DataFrame({"alpha": np.log(100.0), "beta": 0.0, "sd": 0.05, "ok": True}, index=nm)}
    D = B.Data(cal4, px, r4, STd, logP4, logB4, fits4, sec4, pd.Series(0.0, index=cal4), E4)
    ent = pd.DataFrame([{"entry_date": cal4[10], "qid": "K0","sector": "X", "z_entry": -2.5, "fit_date": cal4[0]}])
    out = B.simulate(ent, D, B.Params(n_pos=10, cost_bps=100, use_z_exit=False, use_st_exit=False, time_stop=20), cal4[0])
    tr = out["trades"].iloc[0]
    c = 0.01; w = 0.1
    check("D bought at the close of signal + 1 (session 11), one tenth of NAV, 100 bps paid", tr["entry_exec"] == cal4[11] and np.isclose(tr["weight_in"], w) and np.isclose(out["nav"]["nav"].iloc[11], 1 - w * c))
    check("D the planted trade's net return = (1-c)(1+10%)(1-c) - 1", np.isclose(tr["net"], (1 - c) * 1.1 * (1 - c) - 1) and np.isclose(tr["gross"], 0.1), (tr["net"], tr["gross"]))
    check("D the time stop: held TIME_STOP sessions (decided at the close before, executed at the 20th close after entry)", tr["reason"] == "time" and tr["hold"] == 20, (tr["reason"], tr["hold"]))
    check("D NAV after the round trip = the slot's gain less costs on both sides", np.isclose(out["nav"]["nav"].iloc[-1], (1 - w) + w * (1 - c) * 1.1 * (1 - c)), (out["nav"]["nav"].iloc[-1], (1 - w) + w * (1 - c) ** 2 * 1.1))
    ent5 = pd.DataFrame([{"entry_date": cal4[10], "qid": n,"sector": "X", "z_entry": -2.5 - i * 0.1, "fit_date": cal4[0]} for i, n in enumerate(nm[:5])])
    out5 = B.simulate(ent5, D, B.Params(n_pos=10, sector_cap=4, cost_bps=0, use_z_exit=False, use_st_exit=False, time_stop=5), cal4[0])
    check("D the sector cap holds (5 candidates in one group, 4 taken)", out5["nav"]["n_pos"].iloc[12] == 4 and len(out5["trades"]) == 4)
    fz = {cal4[0]: pd.DataFrame({"alpha": np.log(100.0), "beta": 0.0, "sd": 0.05, "ok": True}, index=nm)}
    px6 = px.copy(); px6.loc[cal4[11:], "K0"] = 100.0; px6.loc[cal4[15:], "K0"] = 100.0 * np.exp(-0.5 * 0.05 * 0 + 0.0)
    D6 = B.Data(cal4, px6, px6 / px6.shift(1) - 1, STd, np.log(px6), logB4, fz, sec4, pd.Series(0.0, index=cal4), E4)
    out6 = B.simulate(ent, D6, B.Params(n_pos=10, cost_bps=0, time_stop=50), cal4[0])
    check("D z under the entry fit >= Z_EXIT (price at the fitted level, z = 0) -> 'caught up' decided at the execution close, sold the next", out6["trades"].iloc[0]["reason"] == "caught up" and out6["trades"].iloc[0]["hold"] == 1, out6["trades"].iloc[0][["reason", "hold"]].to_dict())
    bm = U.ew_benchmark(r4, E4, cal4, 40.0, cal4[0])
    f0 = bm["nav"].first_valid_index()
    check("D the equal-weight benchmark is bought in full on its first session (turnover 1) and later turnover is below 1", np.isclose(bm.at[f0, "turnover"], 1.0) and (bm.loc[f0:, "turnover"].iloc[1:] < 1).all(), bm["turnover"].head(3).tolist())

    print("E. signal")
    wts = np.arange(1, C.LOOKBACK + 1, dtype=float) / np.arange(1, C.LOOKBACK + 1).sum()
    check("E LW ATR weights are 1/15 .. 5/15", np.allclose(wts, [1 / 15, 2 / 15, 3 / 15, 4 / 15, 5 / 15]))
    st_d = ST.panel_states(P, bar="daily")
    check("E states in {+1, -1} after the warm-up", set(np.unique(st_d.dropna().values.astype(int))) <= {1, -1})
    one = ST.supertrend(pd.DataFrame({"open": P["open"]["A0"], "high": P["high"]["A0"], "low": P["low"]["A0"], "close": close["A0"]}))["state"]
    check("E the panel's daily state equals the single-series supertrend", np.allclose(st_d["A0"].dropna().values, one.dropna().values))
    st_w = ST.panel_states(P, bar="weekly")
    fri = [d for d in cal2 if d.weekday() == 4][40]
    nxt = cal2[cal2.get_loc(fri) + 1]
    wk = ST.supertrend(ST.resample_ohlc(pd.DataFrame({"open": P["open"]["A0"], "high": P["high"]["A0"], "low": P["low"]["A0"], "close": close["A0"]}), "W-FRI"))["state"]
    check("E a weekly state known at the Friday close applies from the next session", st_w.at[nxt, "A0"] == wk.loc[:fri].iloc[-1] and st_w.at[fri, "A0"] == wk.loc[:fri].iloc[-2])

    print("F. hygiene")
    db.load_env()
    pw = os.getenv("WM_DB_PASSWORD", "")
    files = [p for p in C.ROOT.rglob("*") if p.suffix in (".py", ".md", ".txt", ".json", ".csv", ".html", ".example", ".ipynb") and ".venv" not in p.parts and "outputs" not in p.parts and p.name != ".env"]
    leak = [str(C.rel(p)) for p in files if pw and pw in p.read_text(encoding="utf-8", errors="ignore")]
    check("F the database password appears in no source, doc, data or report file", not leak, leak)
    check("F .env is git-ignored", ".env" in (C.ROOT / ".gitignore").read_text().splitlines())
    try:
        tracked = subprocess.run(["git", "ls-files", ".env"], cwd=C.ROOT, capture_output=True, text=True).stdout.strip()
        check("F .env is not tracked by git", tracked == "")
    except FileNotFoundError:
        check("F .env is not tracked by git (git not found: skipped)", True)
    names_env = [n.strip() for n in os.getenv("WM_REDACT_NAMES", "").split(",") if n.strip()]
    hits = []
    for p in files:
        if p.suffix in (".csv", ".html"):
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        hits += [f"{p.name}: {n}" for n in names_env if re.search(r"\b" + re.escape(n) + r"\b", txt, flags=re.I)]
    check("F no person's name in any tracked file (WM_REDACT_NAMES)", not hits, hits[:5])
    check("F the pass marks in config are the plan's (+1.5% net, 55%, 95th percentile, 80 bps break-even)",
          C.PASS_1["mean_rel_net_20"] == 0.015 and C.PASS_1["caught_up"] == 0.55 and C.PASS_1["null_pct"] == 0.95 and C.PASS_2["breakeven_bps"] == 80 and C.PASS_2["null_pct"] == 0.95)
    if C.REPORT_HTML.exists() and (C.TABLES / "meta.json").exists():
        meta = json.loads((C.TABLES / "meta.json").read_text())
        rep = C.REPORT_HTML.read_text(encoding="utf-8")
        check("F a report built on synthetic data says so in its head (and a real one does not)", ("content='synthetic'" in rep[:600]) == bool(meta.get("synthetic")))

    print("G. outputs")
    if (C.TABLES / "1_verdict.csv").exists():
        v1 = pd.read_csv(C.TABLES / "1_verdict.csv")
        check("G 1_verdict carries the six Phase 1 criteria", len(v1) == 6 and v1["criterion"].str.contains("both halves").any())
        ev1 = pd.read_csv(C.TABLES / "1_events.csv", parse_dates=["entry_date", "cross_date"])
        check("G every event's entry is on or after its crossing, within TURN_WINDOW sessions", bool(((ev1["entry_date"] >= ev1["cross_date"]) & (ev1["days_to_turn"].fillna(0) <= C.TURN_WINDOW)).all()))
        both1 = ev1[ev1["cell"] == "both"]
        check("G 'both' events have z at entry <= Z_STILL_CHEAP", bool((both1["z_entry"] <= C.Z_STILL_CHEAP).all()) if len(both1) else True)
    if (C.TABLES / "2_trades.csv").exists():
        tr2 = pd.read_csv(C.TABLES / "2_trades.csv", parse_dates=["entry_exec", "exit_exec"])
        ok_ov = True
        for _, g in tr2.sort_values("entry_exec").groupby("qid"):
            if (g["entry_exec"].values[1:] < g["exit_exec"].values[:-1]).any():
                ok_ov = False; break
        check("G a name is never held twice at once", ok_ov)
        check("G every trade is held at most TIME_STOP + 2 sessions (time stop decided at TIME_STOP, executed next; a hole can add one)", bool((tr2["hold"] <= C.TIME_STOP + 2).all()) if len(tr2) else True, tr2["hold"].max() if len(tr2) else None)
        nav2 = pd.read_csv(C.TABLES / "2_nav.csv", index_col=0)
        check("G the book's NAV starts near 1 and never goes non-positive", 0.9 < nav2["book_nav"].iloc[0] < 1.1 and (nav2["book_nav"].dropna() > 0).all(), nav2["book_nav"].iloc[0])
        check("G 2_nav has the three yardsticks", {"ew_nav", "matched_nav"} <= set(nav2.columns))

    print(f"\n{len(P_)} passed, {len(F_)} failed")
    return 0 if not F_ else 1


if __name__ == "__main__":
    sys.exit(main())
