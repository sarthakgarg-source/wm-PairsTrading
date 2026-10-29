"""
Refresh data/sectors_monthly.csv - point-in-time GICS per (month-end, quote_id) from `scores_value`
(monthly from 2017-01; the weekly rows after 2025-12 are thinned to the last of each month).

    python fetch_sectors.py

The current classification and the delisting date come with the universe file (fetch_universe.py, from
`symbology_changes`, whose GICS columns are fetched once at onboarding and carried forward - so they are the
classification as of onboarding, not a history). data.sector_history uses: the monthly record where it
exists; before 2017-01 the earliest monthly record for the quote_id; failing both, the symbology value.
"""
import sys

import pandas as pd
from sqlalchemy import text

import config as C
import db

LEVELS = ["gics_sector_name", "gics_industry_group_name", "gics_industry_name", "gics_sub_industry_name"]


def main() -> int:
    eng = db.engine()
    with eng.connect() as conn:
        pit = pd.read_sql(text(f"""SELECT rebal_date AS date, quote_id AS qid, {', '.join(LEVELS)} FROM {db.SECTORS_PIT}
                                   WHERE gics_industry_group_name IS NOT NULL AND quote_id IS NOT NULL"""), conn)
    pit["date"] = pd.to_datetime(pit["date"])
    pit["ym"] = pit["date"].dt.to_period("M")
    last = pit.groupby("ym")["date"].transform("max")
    pit = pit[pit["date"] == last].drop(columns="ym").drop_duplicates(["date", "qid"]).sort_values(["date", "qid"])
    C.DATA.mkdir(exist_ok=True)
    pit.to_csv(C.SECTORS_PIT_CSV, index=False)
    print(f"wrote {C.rel(C.SECTORS_PIT_CSV)}: {len(pit):,} rows, {pit['date'].nunique()} month-ends "
          f"{pit['date'].min().date()} .. {pit['date'].max().date()}, {pit['qid'].nunique():,} quote_ids; "
          f"{pit[C.SECTOR_LEVEL].nunique()} groups at {C.SECTOR_LEVEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
