"""
Refresh data/stock_ohlcv.parquet from the Windmill Postgres: daily prices and traded value for every quote_id
that ever appears in data/universe_monthly.csv (run fetch_universe.py first).

    python fetch_prices.py

  * open / high / low / close - `ohlcv_eikon` by quote_id: corporate-action adjusted, one continuous series per
    security across RIC and ticker changes, delisted names included, from 2014-07-01. The ONLY price source:
    mixing in the legacy table's prices would splice two adjustment bases (a fake jump wherever they differ).
  * turnover_cr - daily traded value in rupees crore, for the liquidity floor only. `ohlcv_eikon.volume` is empty
    and `liquidity_check_raw_data` starts in 2025, so it comes from the legacy `merged_historical_data_with_rics`
    as close x volume within the same legacy row (both on the legacy basis, so the product is the rupee value
    whatever that basis is). The legacy table is keyed by ticker, its older rows carry ric = NULL; rows are
    taken for every ticker the quote_id has carried in `symbology_changes`, where the row's ric is NULL or one
    of the quote_id's RICs (a different non-null RIC is another company).
"""
import sys

import pandas as pd
from sqlalchemy import text

import config as C
import db


def main() -> int:
    uni = pd.read_csv(C.UNIVERSE_CSV)
    qids = sorted(int(q) for q in uni["qid"].dropna().unique())
    start = (pd.Timestamp(C.DATA_START) - pd.DateOffset(years=1)).date()
    eng = db.engine()
    with eng.connect() as conn:
        px = pd.read_sql(text(f"""SELECT quote_id AS qid, date, open, high, low, close FROM {db.PRICES_EIKON}
                                  WHERE quote_id = ANY(:q) AND date >= :s AND close IS NOT NULL AND close > 0"""), conn, params={"q": qids, "s": start})
        print(f"ohlcv_eikon: {px['qid'].nunique():,} quote_ids, {len(px):,} rows")
        sym = pd.read_sql(text(f"""SELECT quote_id AS qid, ticker, new_ticker, ric, new_ric FROM {db.SYMBOLOGY} WHERE quote_id = ANY(:q)"""), conn, params={"q": qids})
        tk = pd.concat([sym[["qid", "ticker"]], sym[["qid", "new_ticker"]].rename(columns={"new_ticker": "ticker"})]).dropna().drop_duplicates()
        rc = pd.concat([sym[["qid", "ric"]], sym[["qid", "new_ric"]].rename(columns={"new_ric": "ric"})]).dropna().drop_duplicates()
        lg = pd.read_sql(text(f"""SELECT date, ticker, ric, close, volume FROM {db.PRICES_LEGACY}
                                  WHERE ticker = ANY(:t) AND date >= :s AND close > 0 AND volume IS NOT NULL"""), conn, params={"t": sorted(tk["ticker"].unique()), "s": start})
        print(f"legacy table (traded value): {lg['ticker'].nunique():,} tickers, {len(lg):,} rows")
    # legacy rows -> quote_id: by ticker, keeping rows whose ric is NULL or one of the quote_id's RICs
    lg = lg.merge(tk, on="ticker", how="inner")
    own = set(map(tuple, rc[["qid", "ric"]].values))
    lg = lg[lg["ric"].isna() | pd.Series([(q, r) in own for q, r in zip(lg["qid"], lg["ric"])], index=lg.index)]
    lg["turnover_cr"] = pd.to_numeric(lg["close"], errors="coerce") * pd.to_numeric(lg["volume"], errors="coerce") / 1e7
    lg["date"] = pd.to_datetime(lg["date"])
    tv = lg.groupby(["qid", "date"], as_index=False)["turnover_cr"].max()     # a renamed ticker can duplicate a day; the larger is the real print
    px["date"] = pd.to_datetime(px["date"])
    for c in ("open", "high", "low", "close"):
        px[c] = pd.to_numeric(px[c], errors="coerce")
    df = px.merge(tv, on=["qid", "date"], how="left").sort_values(["qid", "date"]).reset_index(drop=True)
    df["qid"] = df["qid"].astype("int64")
    missing = sorted(set(qids) - set(df["qid"].unique()))
    C.DATA.mkdir(exist_ok=True)
    df.to_parquet(C.PRICES_FILE, index=False)
    if C.SYNTHETIC_MARKER.exists():
        C.SYNTHETIC_MARKER.unlink()
    print(f"wrote {C.rel(C.PRICES_FILE)}: {len(df):,} rows, {df['qid'].nunique():,} quote_ids, "
          f"{df['date'].min().date()} .. {df['date'].max().date()}; {df['turnover_cr'].notna().mean():.1%} of rows with a traded value")
    out = C.DATA / "prices_missing.txt"
    if missing:
        names = uni.drop_duplicates("qid").set_index("qid").reindex(missing)[["ticker", "name"]]
        out.write_text(names.to_csv())
        print(f"{len(missing)} universe quote_ids have no prices in ohlcv_eikon -> {C.rel(out)}")
    elif out.exists():
        out.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
