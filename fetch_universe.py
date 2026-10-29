"""
Refresh data/universe_monthly.csv from the Windmill Postgres - the rolling market-cap universe, built the way
wm-utils builds it (wm_utils/universe/mcap_historical_universe/get_rolling_mcap_universe.py,
GetRollingMcapUniverse.get_rolling_mcap_universe(rebal_freq='m', top=600, end_date_type='tradingday_end_dates')):

  * month-end TRADING days = the last NIFTY 50 session of each month in `index_price_data`
  * `mcap_based_universe` rows on those days with instrument_class = 'ordinary'
  * ranked by `mcap` descending within each date, method='first'; the top UNIVERSE_FETCH_N kept
  * each quote_id mapped to its latest identifiers and GICS through `symbology_changes`
    (drop_duplicates on quote_id, keep='last'): new_ric -> ric, new_ticker -> ticker, new_isin -> isin

    python fetch_universe.py

Columns: date (the month-end trading day), qid (quote_id), ric, ticker, isin, name, mcap_cr, rank, the GICS
levels, delist_date. The date is kept as the trading day (not rolled to the calendar month-end) because the
study applies a snapshot from the session after it.
"""
import sys

import pandas as pd
from sqlalchemy import text

import config as C
import db

LEVELS = ["gics_sector_name", "gics_industry_group_name", "gics_industry_name", "gics_sub_industry_name"]


def month_end_trading_days(conn, start) -> list:
    td = pd.read_sql(text(f"SELECT DISTINCT date FROM {db.INDEX} WHERE ticker = 'NIFTY 50' AND date >= :s ORDER BY date"), conn, params={"s": start})
    td["date"] = pd.to_datetime(td["date"])
    me = td.groupby(td["date"].dt.to_period("M"))["date"].max()
    return [d.date() for d in me]


def main() -> int:
    eng = db.engine()
    start = (pd.Timestamp(C.DATA_START) - pd.DateOffset(years=1)).date()
    with eng.connect() as conn:
        days = month_end_trading_days(conn, start)
        mc = pd.read_sql(text(f"""SELECT date, quote_id, mcap, mcap_scaled_in_cr FROM {db.MCAP}
                                  WHERE date = ANY(:d) AND instrument_class = 'ordinary'"""), conn, params={"d": days})
        sym = pd.read_sql(text(f"""SELECT id, quote_id, new_ric, new_ticker, new_isin, new_full_name, {', '.join(LEVELS)}, delist_date
                                   FROM {db.SYMBOLOGY} ORDER BY id"""), conn)
    mc["date"] = pd.to_datetime(mc["date"])
    mc["mcap"] = pd.to_numeric(mc["mcap"], errors="coerce")
    mc = mc.dropna(subset=["mcap"])
    mc["rank"] = mc.groupby("date")["mcap"].rank(method="first", ascending=False).astype(int)
    mc = mc[mc["rank"] <= C.UNIVERSE_FETCH_N]
    sym = sym.drop_duplicates(subset=["quote_id"], keep="last").drop(columns="id")
    df = mc.merge(sym, on="quote_id", how="left").rename(columns={
        "quote_id": "qid", "new_ric": "ric", "new_ticker": "ticker", "new_isin": "isin", "new_full_name": "name", "mcap_scaled_in_cr": "mcap_cr"})
    df = df[["date", "qid", "ric", "ticker", "isin", "name", "mcap_cr", "rank"] + LEVELS + ["delist_date"]].sort_values(["date", "rank"])
    short = df.groupby("date").size()
    C.DATA.mkdir(exist_ok=True)
    df.to_csv(C.UNIVERSE_CSV, index=False, float_format="%.2f")
    print(f"wrote {C.rel(C.UNIVERSE_CSV)}: {len(df):,} rows, {df['date'].nunique()} month-end trading days "
          f"{df['date'].min().date()} .. {df['date'].max().date()}, {df['qid'].nunique():,} distinct quote_ids")
    if (short < C.UNIVERSE_FETCH_N).any():
        print(f"  dates with fewer than {C.UNIVERSE_FETCH_N} names: {[d.date() for d in short[short < C.UNIVERSE_FETCH_N].index][:10]}")
    print(f"  without a GICS industry group: {df['gics_industry_group_name'].isna().mean():.1%} of rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
