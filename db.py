"""
One place that knows how to reach the Windmill Postgres.

Credentials are read from a `.env` file in the repo root (or from real
environment variables, which win). There is deliberately NO default password in
this file or anywhere else in the source tree. The `.env` is git-ignored.

    cp .env.example .env      # then fill in WM_DB_USER / WM_DB_PASSWORD
    python db_probe.py        # confirms the connection and the spans of the tables this study reads

Needs the optional packages in requirements-db.txt (sqlalchemy, psycopg2).
The study itself never touches the database: run_all.py reads only data/*.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = ("WM_DB_HOST", "WM_DB_PORT", "WM_DB_NAME", "WM_DB_USER", "WM_DB_PASSWORD")


def load_env(path: Path | None = None) -> None:
    """Minimal .env reader (KEY=VALUE, # comments). Real env vars take precedence."""
    path = path or (ROOT / ".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def engine(db_name: str | None = None):
    from sqlalchemy import create_engine
    from sqlalchemy.engine import URL

    load_env()
    missing = [k for k in REQUIRED if not os.getenv(k)]
    if missing:
        raise SystemExit(f"missing {missing}. Copy .env.example to .env and fill it in - see LINKS_AND_ACCESS.md")
    url = URL.create(
        "postgresql+psycopg2",
        username=os.environ["WM_DB_USER"],
        password=os.environ["WM_DB_PASSWORD"],
        host=os.environ["WM_DB_HOST"],
        port=int(os.environ["WM_DB_PORT"]),
        database=db_name or os.environ["WM_DB_NAME"],
        query={"sslmode": "require"},
    )
    # the database is read-only for every agent and script (wm-utils docs/tables/README.md): enforced per session
    return create_engine(url, connect_args={"options": "-c default_transaction_read_only=on"})


# The tables this study reads (database `windmill`, schema public). The key everywhere is `quote_id`:
# RIC, ticker and ISIN change over a security's life (wm-utils docs/tables, "Symbology join keys").
PRICES_EIKON = "public.ohlcv_eikon"                   # quote_id, date, open, high, low, close (volume empty) — corp-action adjusted; the price source
PRICES_LEGACY = "public.merged_historical_data_with_rics"   # date, ticker, ric, close, volume — legacy, ticker-keyed; used ONLY for daily traded value
MCAP = "public.mcap_based_universe"                   # date, quote_id, mcap, instrument_class — the daily mcap-ranked universe
SYMBOLOGY = "public.symbology_changes"                # quote_id, ric/ticker (history rows), new_ric/new_ticker/new_isin, gics_*, delist_date
SECTORS_PIT = "public.scores_value"                   # rebal_date, quote_id, gics_* — point-in-time GICS, monthly from 2017-01
INDEX = "public.index_price_data"                     # date, ticker ('NIFTY 50', 'NIFTY 500', ...), close — also the trading calendar
