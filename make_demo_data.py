"""
Synthetic demo data so the pipeline runs end to end without the database. NEVER a result.

    python make_demo_data.py                       # refuses if data/ already holds the real (fetched) files
    set PAIRS_DATA_DIR=<dir> & set PAIRS_OUT_DIR=<dir> & python make_demo_data.py & python run_all.py --fast
                                                   # the demo beside the real data, in other folders

What is planted: 8 groups x ~20 names; each log price = beta x (its group factor) + a residual that is a
stationary AR(1) (half-life ~ 12 sessions) for 70% of the names and a random walk for the rest; the
supertrend therefore turns on real swings. A few names stop printing (delistings). No edge is planted
beyond the mean reversion of the stationary residuals; whether the rule finds it after costs is the demo.
Files carry the real schema (qid, turnover_cr, the GICS columns) and the SYNTHETIC stamp.
"""
import sys

import numpy as np
import pandas as pd

import config as C

GROUPS = ["Banks", "Capital Goods", "Materials", "Software & Services", "Pharmaceuticals, Biotechnology & Life Sciences",
          "Automobiles & Components", "Consumer Durables & Apparel", "Utilities"]
LEVELS = ["gics_sector_name", "gics_industry_group_name", "gics_industry_name", "gics_sub_industry_name"]


def _write_csv(df: pd.DataFrame, path, **kw) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# SYNTHETIC demo data (make_demo_data.py)\n")
        df.to_csv(f, index=False, lineterminator="\n", **kw)


def main(seed: int = 3) -> int:
    if C.PRICES_FILE.exists() and not C.SYNTHETIC_MARKER.exists() and "--overwrite-real" not in sys.argv:
        raise SystemExit(f"{C.DATA} holds real fetched data; not overwriting it. Set PAIRS_DATA_DIR / PAIRS_OUT_DIR to other folders "
                         f"(see the docstring), or pass --overwrite-real.")
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(pd.Timestamp(C.DATA_START), "2026-10-27")
    T = len(cal)
    names, group_of, qids = [], {}, {}
    for g in GROUPS:
        for k in range(rng.integers(16, 24)):
            n = f"{g[:4].upper().replace(' ', '')}{k:02d}"
            names.append(n); group_of[n] = g; qids[n] = 90_000_000_000 + len(names)
    N = len(names)
    mkt = np.cumsum(rng.normal(0.0004, 0.010, T))
    fac = {g: mkt + np.cumsum(rng.normal(0.0, 0.006, T)) for g in GROUPS}
    logp = np.zeros((T, N))
    for j, n in enumerate(names):
        beta = rng.uniform(0.7, 1.3)
        if rng.uniform() < 0.7:
            phi = 0.5 ** (1 / 12)
            e = np.zeros(T); sd = rng.uniform(0.02, 0.05)
            for t in range(1, T):
                e[t] = phi * e[t - 1] + rng.normal(0, sd * np.sqrt(1 - phi ** 2))
        else:
            e = np.cumsum(rng.normal(0, 0.012, T))
        logp[:, j] = np.log(rng.uniform(50, 3000)) + beta * fac[group_of[n]] + e
    close = np.exp(logp)
    o = close * np.exp(rng.normal(0, 0.004, (T, N)))
    h = np.maximum(o, close) * np.exp(np.abs(rng.normal(0, 0.006, (T, N))))
    l = np.minimum(o, close) * np.exp(-np.abs(rng.normal(0, 0.006, (T, N))))
    tv = close * np.exp(rng.normal(12, 0.8, (T, N))) / 1e7          # rupees crore
    delist = {}
    for j in rng.choice(N, 6, replace=False):
        d = rng.integers(T // 2, T - 100)
        close[d:, j] = np.nan; delist[names[j]] = cal[d - 1]
    rows = []
    for j, n in enumerate(names):
        m = ~np.isnan(close[:, j])
        rows.append(pd.DataFrame({"qid": qids[n], "date": cal[m], "open": o[m, j], "high": h[m, j], "low": l[m, j], "close": close[m, j], "turnover_cr": tv[m, j]}))
    C.DATA.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_parquet(C.PRICES_FILE, index=False)
    C.SYNTHETIC_MARKER.write_text("synthetic demo data - make_demo_data.py\n")
    # universe: month-end trading days, mcap = price x fixed shares, rank; delisted names drop out
    shares = np.exp(rng.normal(18, 1, N))
    me = cal[cal.to_series().groupby([cal.year, cal.month]).transform("max") == cal]
    urows = []
    for d in me:
        t = cal.get_loc(d)
        mc = close[t] * shares / 1e7
        order = [j for j in np.argsort(-np.nan_to_num(mc, nan=-1)) if not np.isnan(mc[j])]
        for rk, j in enumerate(order, 1):
            n = names[j]
            urows.append({"date": d, "qid": qids[n], "ric": f"{n}.NS", "ticker": n, "isin": f"INE{j:06d}01", "name": n, "mcap_cr": mc[j], "rank": rk,
                          **{c: group_of[n] for c in LEVELS}, "delist_date": delist.get(n)})
    uni = pd.DataFrame(urows)
    _write_csv(uni, C.UNIVERSE_CSV, float_format="%.2f")
    pit = uni[uni["date"] >= "2017-01-01"][["date", "qid"] + LEVELS]
    _write_csv(pit, C.SECTORS_PIT_CSV)
    r = pd.DataFrame(close, index=cal).pct_change(fill_method=None).mean(axis=1).fillna(0)
    ix = pd.DataFrame({"date": cal, C.INDEX_REF: 10000 * (1 + r).cumprod(), C.CALENDAR_INDEX: 10000 * (1 + r * 0.9).cumprod()})
    _write_csv(ix, C.INDEX_CSV, float_format="%.4f")
    print(f"wrote synthetic data: {N} names in {len(GROUPS)} groups, {T:,} sessions, {len(delist)} delistings -> {C.DATA} (marker: SYNTHETIC)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
