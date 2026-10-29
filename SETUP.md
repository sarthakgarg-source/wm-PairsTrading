# SETUP
    python -m venv .venv && .venv\Scripts\activate
    pip install -r requirements.txt            # pandas, numpy, scipy, statsmodels, pyarrow, plotly
    pip install -r requirements-db.txt         # only for the fetch scripts (sqlalchemy, psycopg2)
    copy .env.example .env                     # fill in WM_DB_USER / WM_DB_PASSWORD (office network or VPN)

## 1. Data (read-only, the wm-utils way — LINKS_AND_ACCESS.md)
    python db_probe.py          # connection, "session read-only: on", table spans, a split spot check
    python fetch_index.py       # data/index_prices.csv — NIFTY 50 (the calendar) and NIFTY 500 (the reference)
    python fetch_universe.py    # data/universe_monthly.csv — the rolling mcap universe, top-600 at month-end trading days
    python fetch_sectors.py     # data/sectors_monthly.csv — point-in-time GICS
    python fetch_prices.py      # data/stock_ohlcv.parquet — ohlcv_eikon by quote_id + traded value (~3M rows; a few minutes)
    python data.py              # coverage, sector sources by year, no SYNTHETIC banner

Without the database, a synthetic demo runs in other folders (it refuses to overwrite real data in `data/`):

    set PAIRS_DATA_DIR=C:\temp\pairs_demo_data
    set PAIRS_OUT_DIR=C:\temp\pairs_demo_out
    python make_demo_data.py && python run_all.py --fast && python make_report.py

## 2. Run
    python run_all.py           # Phase 0 -> 1, and 2 -> 3 only if Phase 1 passes
    python run_all.py --force   # every phase whatever Phase 1 says (the report marks the book "for inspection only"); ~15 min with the grid
    python run_all.py --fast    # 20 null draws, no grid, no stock pairs
    python run_all.py --no-grid
    python make_report.py       # outputs/report.html
    python checks.py            # "0 failed"

If `import statsmodels` fails with "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x", the SciPy on that interpreter predates NumPy 2: `python -m pip install -U numpy scipy statsmodels pandas` (or pin `numpy<2`).

## 3. Change a parameter
Only in `config.py`. The three fixed choices the review confirmed — partner = the group basket, entry z = −2, "turned up" = the daily LW supertrend — and the pass marks are not grid dimensions. Everything in `config.GRID` is, and Phase 3 reports it whole. Re-run `checks.py` after any code edit.
