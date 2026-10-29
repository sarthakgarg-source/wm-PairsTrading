"""
Connection check (read-only). Confirms the credentials work and prints the span of every table this study
reads, plus a split spot check on the price source.

    python db_probe.py
"""
import sys

import pandas as pd
from sqlalchemy import text

import db


def main() -> int:
    eng = db.engine()
    with eng.connect() as conn:
        ro = conn.execute(text("SHOW default_transaction_read_only")).scalar()
        print(f"session read-only: {ro}")
        for name, tbl, dcol in (("prices (Eikon, by quote_id)", db.PRICES_EIKON, "date"), ("traded value (legacy, by ticker)", db.PRICES_LEGACY, "date"),
                                ("market-cap universe", db.MCAP, "date"), ("sectors, point-in-time", db.SECTORS_PIT, "rebal_date"),
                                ("index closes / calendar", db.INDEX, "date")):
            r = conn.execute(text(f"SELECT MIN({dcol}), MAX({dcol}) FROM {tbl}")).fetchone()
            print(f"{name:34s} {tbl:45s} {r[0]} .. {r[1]}")
        n = conn.execute(text(f"SELECT COUNT(DISTINCT quote_id), COUNT(delist_date) FROM {db.SYMBOLOGY}")).fetchone()
        print(f"{'security master':34s} {db.SYMBOLOGY:45s} {n[0]:,} quote_ids, {n[1]:,} rows with a delisting date")
        print("\nsplit spot check on the price source (NESTLEIND 1:10, 2024-01-05: no jump on either side):")
        q = text(f"""SELECT o.date, o.close FROM {db.PRICES_EIKON} o
                     WHERE o.quote_id = (SELECT quote_id FROM {db.SYMBOLOGY} WHERE new_ticker = 'NESTLEIND' LIMIT 1)
                       AND o.date BETWEEN '2024-01-03' AND '2024-01-08' ORDER BY o.date""")
        df = pd.read_sql(q, conn)
        print("  " + "  ".join(f"{r.date}: {float(r.close):.1f}" for r in df.itertuples()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
