"""
Load the data files into the panels the study runs on. The key everywhere is `qid` (the Refinitiv quote_id):
RIC, ticker and ISIN change over a security's life; quote_id does not.

    P      = load_prices()        # dict of wide frames open/high/low/close/turnover [session x qid]
    uni    = load_universe()      # long: date (month-end trading day), qid, ric, ticker, isin, name, mcap_cr, rank, gics_*
    sh     = sector_history(uni)  # long: date, qid, sector, sector_source
    idx    = load_index()         # wide closes of the reference indices
    cal    = calendar(P, idx)     # the session calendar (NIFTY 50 sessions) from DATA_START
    lab    = labels(uni)          # qid -> ticker, for readable tables

Nothing here opens a database connection: the fetch_*.py scripts write data/, this reads it.
A file whose first line starts with '# SYNTHETIC' (or the data/SYNTHETIC marker) flags the run
as a demo: every output is stamped and checks.py refuses to call it a result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

SYNTHETIC_SEEN: set[str] = set()


def _read_csv(path: Path, **kw) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        head = f.readline()
    skip = 0
    if head.startswith("#"):
        skip = 1
        if "SYNTHETIC" in head.upper():
            SYNTHETIC_SEEN.add(path.name)
    return pd.read_csv(path, skiprows=skip, **kw)


def is_synthetic() -> bool:
    return bool(SYNTHETIC_SEEN) or C.SYNTHETIC_MARKER.exists()


def _qid(s: pd.Series) -> pd.Series:
    """quote_ids as strings (the panel column labels)."""
    return pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)


# ------------------------------------------------------------------ prices --
def load_prices(verbose: bool = True) -> dict[str, pd.DataFrame]:
    if not C.PRICES_FILE.exists():
        raise SystemExit(f"{C.rel(C.PRICES_FILE)} missing - run fetch_prices.py (or make_demo_data.py for a synthetic demo)")
    if C.SYNTHETIC_MARKER.exists():
        SYNTHETIC_SEEN.add(C.PRICES_FILE.name)
    df = pd.read_parquet(C.PRICES_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df["qid"] = _qid(df["qid"])
    df = df[df["date"] >= pd.Timestamp(C.DATA_START) - pd.Timedelta(days=400)]
    df = df[(df["close"] > 0)].drop_duplicates(["date", "qid"], keep="last")
    P = {}
    for col in ("open", "high", "low", "close", "turnover_cr"):
        if col in df.columns:
            P[col] = df.pivot(index="date", columns="qid", values=col).sort_index()
    for col in ("open", "high", "low"):
        P[col] = P[col].where(P[col].notna(), P["close"])
    P["turnover"] = P.pop("turnover_cr") if "turnover_cr" in P else P["close"] * np.nan
    if verbose:
        c = P["close"]
        print(f"prices: {c.shape[1]:,} names x {c.shape[0]:,} sessions, {c.index[0].date()} .. {c.index[-1].date()}"
              + ("  [SYNTHETIC]" if is_synthetic() else ""))
    return P


# ---------------------------------------------------------------- universe --
def load_universe() -> pd.DataFrame:
    u = _read_csv(C.UNIVERSE_CSV, parse_dates=["date"])
    u = u.dropna(subset=["qid"])
    u["qid"] = _qid(u["qid"])
    return u.sort_values(["date", "rank"])


def labels(uni: pd.DataFrame) -> pd.Series:
    """qid -> the latest ticker (falls back to the RIC, then the qid itself)."""
    last = uni.sort_values("date").drop_duplicates("qid", keep="last").set_index("qid")
    lab = last["ticker"] if "ticker" in last.columns else pd.Series(index=last.index, dtype=object)
    if "ric" in last.columns:
        lab = lab.fillna(last["ric"])
    return lab.fillna(pd.Series(last.index, index=last.index)).rename("ticker")


def load_sectors_pit() -> pd.DataFrame:
    if not C.SECTORS_PIT_CSV.exists():
        return pd.DataFrame(columns=["date", "qid", C.SECTOR_LEVEL])
    s = _read_csv(C.SECTORS_PIT_CSV, parse_dates=["date"])
    s["qid"] = _qid(s["qid"])
    return s


def sector_history(uni: pd.DataFrame, level: str = C.SECTOR_LEVEL) -> pd.DataFrame:
    """Point-in-time sector per (month-end, qid) for every universe row.
    The monthly GICS record (scores_value) where it exists (2017-01 ->); before it, the earliest monthly record
    for the qid; failing both, the security master's value (symbology_changes, carried in the universe file).
    The column `sector_source` says which. A blank sector means no basket for that month."""
    pit = load_sectors_pit()
    out = uni[["date", "qid"]].copy()
    out["ym"] = out["date"].dt.to_period("M")
    out["sector"] = np.nan
    out["sector"] = out["sector"].astype(object)
    out["sector_source"] = None
    if len(pit) and level in pit.columns:
        p = pit[["date", "qid", level]].dropna(subset=[level])
        p = p.assign(ym=p["date"].dt.to_period("M")).drop_duplicates(["ym", "qid"], keep="last")
        out = out.drop(columns="sector").merge(p[["ym", "qid", level]].rename(columns={level: "sector"}), on=["ym", "qid"], how="left")
        out.loc[out["sector"].notna(), "sector_source"] = "monthly"
        first = p.sort_values("date").drop_duplicates("qid").set_index("qid")[level]
        fill = out["qid"].map(first)
        back = out["sector"].isna() & fill.notna()
        out.loc[back, "sector"] = fill[back]
        out.loc[back, "sector_source"] = "earliest"
    if level in uni.columns:
        cur = uni.dropna(subset=[level]).drop_duplicates("qid", keep="last").set_index("qid")[level]
        fill = out["qid"].map(cur)
        m = out["sector"].isna() & fill.notna()
        out.loc[m, "sector"] = fill[m]
        out.loc[m, "sector_source"] = "master"
    out["sector"] = out["sector"].replace({"": np.nan})
    return out.drop(columns="ym")


# ------------------------------------------------------------------- index --
def load_index() -> pd.DataFrame | None:
    if not C.INDEX_CSV.exists():
        return None
    ix = _read_csv(C.INDEX_CSV, parse_dates=["date"]).set_index("date").sort_index()
    return ix.apply(pd.to_numeric, errors="coerce")


def calendar(P: dict[str, pd.DataFrame], idx: pd.DataFrame | None) -> pd.DatetimeIndex:
    """The session calendar: NIFTY 50's sessions (as wm-utils), else the reference index's, else every date with
    at least 20 stock closes; from DATA_START, and not past the last stock print."""
    cal = None
    if idx is not None:
        for name in (C.CALENDAR_INDEX, C.INDEX_REF):
            if name in idx.columns and idx[name].notna().sum() > 100:
                cal = idx.index[idx[name].notna()]
                break
    if cal is None:
        n = P["close"].notna().sum(axis=1)
        cal = n.index[n >= 20]
    last = P["close"].index[P["close"].notna().sum(axis=1) >= 20].max()
    return cal[(cal >= pd.Timestamp(C.DATA_START)) & (cal <= last)]


def on_calendar(P: dict[str, pd.DataFrame], cal: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Every panel on the session calendar; a missing session inside a name's history is NOT filled
    (a hole stays a hole; run_all credits the move on the session the price returns), and nothing is
    carried past a name's last print."""
    return {k: (v.reindex(cal) if isinstance(v, pd.DataFrame) else v) for k, v in P.items()}


def coverage(P: dict[str, pd.DataFrame], cal: pd.DatetimeIndex, lab: pd.Series | None = None) -> pd.DataFrame:
    c = P["close"].reindex(cal)
    tv = P["turnover"].reindex(cal)
    rows = []
    for q in c.columns:
        s = c[q].dropna()
        if not len(s):
            continue
        rows.append({"qid": q, "ticker": lab.get(q, "") if lab is not None else "", "first": s.index[0].date(), "last": s.index[-1].date(),
                     "sessions": len(s), "holes": int(c[q].loc[s.index[0]:s.index[-1]].isna().sum()),
                     "share_with_traded_value": float(tv[q].loc[s.index[0]:s.index[-1]].notna().mean())})
    return pd.DataFrame(rows).set_index("qid")


if __name__ == "__main__":
    P = load_prices()
    idx = load_index()
    cal = calendar(P, idx)
    uni = load_universe()
    sh = sector_history(uni)
    print(f"calendar: {len(cal):,} sessions {cal[0].date()} .. {cal[-1].date()}")
    print(f"universe: {uni['date'].nunique()} month-ends, {uni['qid'].nunique():,} names ever; ranks to {uni['rank'].max()}")
    top = uni[uni["rank"] <= C.UNIVERSE_N]
    priced = top["qid"].isin(P["close"].columns).mean()
    print(f"top-{C.UNIVERSE_N} rows whose quote_id has prices: {priced:.1%}")
    print("sector source by year (share of rows):")
    print(sh.groupby(sh['date'].dt.year)['sector_source'].value_counts(normalize=True).unstack().round(2).to_string())
    print(f"rows with no sector at all: {sh['sector'].isna().mean():.1%}")
    if is_synthetic():
        print("\n" + "!" * 78 + "\n!!  SYNTHETIC DEMO DATA - results are NOT the desk's\n" + "!" * 78)
    sys.exit(0)
