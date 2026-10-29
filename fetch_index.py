"""
Refresh data/index_prices.csv (one close column per reference index) from `index_price_data`.

    python fetch_index.py

Price indices only - the database holds no total-return series (UNDERSTANDING.md, open question).
NIFTY 50's sessions are also the trading calendar, as in wm-utils (GetTimeSeriesData._get_trading_days).
"""
import sys

import pandas as pd
from sqlalchemy import text

import config as C
import db


def main() -> int:
    eng = db.engine()
    q = text(f"""SELECT date, ticker, close FROM {db.INDEX} WHERE ticker = ANY(:t) AND close IS NOT NULL ORDER BY date""")
    with eng.connect() as conn:
        df = pd.read_sql(q, conn, params={"t": C.INDEX_ALIASES})
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    C.DATA.mkdir(exist_ok=True)
    wide.to_csv(C.INDEX_CSV, float_format="%.4f")
    print(f"wrote {C.rel(C.INDEX_CSV)}: {len(wide):,} rows, {wide.index[0].date()} .. {wide.index[-1].date()}; columns {list(wide.columns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
