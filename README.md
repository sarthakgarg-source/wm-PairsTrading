# wm-PairsTrading — Study L: stock pairs, long-only

**The sentence under test.** Within a sector, a stock that has fallen well below its long-run relationship with its peers — and whose own trend has turned up — outperforms its sector over the next one to three months, net of costs, often enough to run as a long-only selection rule.

Everything here answers that sentence with a yes or a no, and stops early if the answer is no. The plan is `plan_stock_pairs_long_only.md` (2026-10-27, amended at the review of the same date — see UNDERSTANDING.md §1 for what changed and why).

    pip install -r requirements.txt
    python fetch_index.py; python fetch_universe.py; python fetch_sectors.py; python fetch_prices.py   # the real data (SETUP.md)
    python run_all.py          # Phase 0 -> 1 -> (2 -> 3 only if Phase 1 passes); --fast for a dry run; --force to build the book regardless
    python make_report.py      # outputs/report.html — interactive, works offline
    python checks.py           # must say "0 failed"

## The four phases, each with a stop rule

| Phase | What | Stop rule | Tables |
|---|---|---|---|
| 0 — data audit | point-in-time universe, liquidity floor, sector coverage, delistings, corporate-action artefacts, the equal-weight universe against the index | the price history cannot be trusted (`0_verdict`) | `0_*` |
| 1 — the mechanism | every cheap crossing of every stock–basket pair that passed the level test; the 2 × 2 (cheap + turned up / cheap alone / turned-up alone / random); the stock's return against its basket over 5 / 20 / 60 sessions; two count-matched nulls | the pre-registered marks (`config.PASS_1`): +1.5 % net over 20 sessions, 55 % caught up, both halves, above the 95th percentile of the null, at least 150 events, and "both" beats "turned-up alone" | `1_*` |
| 2 — the book | 30 positions, 4 per group, equal-weighted, one-session delay, 40 bps a side, empty slots in the equal-weight universe; judged against the sector-matched benchmark, the equal-weight universe and the index; count-matched random books; the cost grid and break-even; the 2 × 2 as books | `config.PASS_2`: Sharpe and worst fall better than the sector-matched benchmark, on both halves; above the random books' 95th percentile; break-even ≥ 80 bps | `2_*` |
| 3 — robustness | the grid (formation / entry / exit / bar / hold / count, one at a time, reported whole), the placebo (rich + turned up), block bootstrap, effective n, the stock-to-stock pair variant | none — reported | `3_*` |

## Layout
    config.py       every parameter and the pass marks        data.py        loaders; the SYNTHETIC stamp
    universe.py     membership (from the next session), liquidity, baskets ex-self, equal-weight benchmark
    formation.py    monthly fits, the Engle-Granger level test, daily z under the fit in force
    events.py       Phase 1: crossings, the cells, forward relative returns, nulls, the verdict
    book.py         Phase 2: the simulation, benchmarks, metrics, break-even, random books, the verdict
    grid.py         Phase 3: grid, placebo, bootstrap, effective n, stock-to-stock pairs
    supertrend.py   the desk's supertrend (daily here; weekly as the grid variant)
    figs.py / make_report.py / run_all.py / checks.py
    db.py, db_probe.py, fetch_universe.py, fetch_sectors.py, fetch_prices.py, fetch_index.py   Postgres, read-only (credentials in .env only)
    make_demo_data.py   synthetic data for a dry run in other folders (PAIRS_DATA_DIR / PAIRS_OUT_DIR) — never a result

## Data (`data/`, git-ignored)
Pulled read-only from the Windmill Postgres the way wm-utils does it (LINKS_AND_ACCESS.md), keyed on `quote_id` (`qid`):
`universe_monthly.csv` (the rolling market-cap universe — top-600 at each month-end trading day, with RIC / ticker / GICS / delisting date from `symbology_changes`), `sectors_monthly.csv` (point-in-time GICS, monthly from 2017-01), `stock_ohlcv.parquet` (adjusted daily OHLC from `ohlcv_eikon` + daily traded value for the liquidity floor), `index_prices.csv` (NIFTY 50 = the calendar, NIFTY 500 = the reference). `python data.py` prints the coverage.

## Read next
`GLOSSARY.md` — every term, table and column in plain words. `LOGIC.md` — the exact definitions. `UNDERSTANDING.md` — the reading choices, the data's limits, the change log. `LINKS_AND_ACCESS.md` — the database and the sibling studies.
