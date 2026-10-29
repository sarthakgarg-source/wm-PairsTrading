# Links and access
No secret is written in this file. `.env` (git-ignored) carries the Windmill Postgres login for `db_probe.py` / `fetch_*.py`; `run_all.py`, `make_report.py` and `checks.py` read `data/*` and never open a connection. Host `wm-prod-internal-postgres.cae4omyvwslq.ap-south-1.rds.amazonaws.com`, port 5432, db `windmill`, `sslmode=require`, user and password in `.env`. Delete `.env` before forwarding a zip.

**The database is read-only.** Every connection `db.py` opens sets `default_transaction_read_only = on` (the wm-utils table catalog's rule: no agent or script writes to it, ever). `db_probe.py` prints the session flag.

## How the data is pulled — following wm-utils
The data layer follows `smallcase/wm-utils` (https://github.com/smallcase/wm-utils):
* the table catalog and its conventions — https://github.com/smallcase/wm-utils/tree/master/wm_utils/docs/tables (README: task → table routing, "Symbology join keys", the read-only rule);
* the rolling market-cap universe — https://github.com/smallcase/wm-utils/blob/master/wm_utils/universe/mcap_historical_universe/get_rolling_mcap_universe.py. `fetch_universe.py` reproduces `GetRollingMcapUniverse.get_rolling_mcap_universe(rebal_freq='m', top=600, end_date_type='tradingday_end_dates')`: month-end NIFTY 50 sessions, `instrument_class = 'ordinary'`, rank by `mcap` (method `first`), identifiers and GICS from `symbology_changes` by `quote_id` (`drop_duplicates(keep='last')`).

**The key is `quote_id`** (called `qid` in this repo). RIC, ticker and ISIN change over a security's life; the catalog says to join on `quote_id` and resolve names through `symbology_changes`. Tickers and RICs appear in the tables only as labels.

## Tables read (schema `public`)
| Table | Used for | Notes |
|---|---|---|
| `mcap_based_universe` | month-end market caps → the point-in-time top-500 | daily, from 2005; keyed (`date`, `quote_id`) |
| `symbology_changes` | `quote_id` → latest RIC / ticker / ISIN / name, GICS, delisting date; all historical tickers and RICs of a `quote_id` (for the traded-value join) | one row per change event; GICS fetched at onboarding |
| `ohlcv_eikon` | daily OHLC by `quote_id` — **the only price source** | corporate-action adjusted, one series across RIC changes, delisted names included; from 2014-07-01 (hence `DATA_START`); `volume` is empty |
| `merged_historical_data_with_rics` | daily traded value (close × volume within the same row) for the liquidity floor only | legacy, ticker-keyed; a name's older rows carry `ric` = NULL. Never used for returns: its adjustment basis differs from Eikon's for some names, and splicing the two made fake jumps (UNDERSTANDING.md, 2026-10-28) |
| `scores_value` | point-in-time GICS per `quote_id` | monthly from 2017-01 |
| `index_price_data` | NIFTY 50 sessions (the calendar, as wm-utils) and NIFTY 500 / 100 / MIDCAP 150 / SMLCAP 250 / TOTAL MKT closes | price indices; no TRI in the database |
| `liquidity_check_raw_data` | **not used** | daily turnover by `quote_id`, the catalog's route for liquidity — but it starts in 2025 |
| `index_constituents_monthly` | **not used** | index membership by ISIN, from 2026-04 only, no Nifty 500 |

## Sibling studies
`wm-corrF2` / `wm-BREAK` (Study I: correlation and the pair tests; this repo's `db.py`, `supertrend.py` and report shape come from it), `wm-CORR` (G: structure and cointegration; `coint.py` is the Engle-Granger reference), `wm-VIXcorr*` (H), `wm-IDXrot` / `wm-EQrot` / `wm-EQtilt` (D/E/F, the rotations), `wm-EQrotStressTest` (the stress-test playbook: random selection, placebo, partitions, block bootstrap, effective n — the Phase 3 items here), `wm-STstrat` (A/B/C, the supertrend engine).

## Reference
NSE index methodology and factsheets: https://www.niftyindices.com · statsmodels Engle-Granger and MacKinnon p-values: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html · Claude Code: https://code.claude.com/docs
