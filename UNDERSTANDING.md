# UNDERSTANDING

## 1. The plan → the build: what was kept and what the review of 2026-10-27 changed
| Plan item | Built as | Why it changed (if it did) |
|---|---|---|
| Universe = Nifty 500, point-in-time | top-500 by market cap at each month-end trading day (`mcap_based_universe`, the wm-utils rolling universe), applied from the next session | the database has no historical Nifty 500 membership (`index_constituents_monthly` starts 2026-04, no Nifty 500); the market-cap rule is how the index is built, and it is survivorship-free |
| Window 2012 → | **2014-07 →** (`DATA_START`) | delisted names exist only in `ohlcv_eikon`, which starts 2014-07-01; before it the panel would be survivors only. First signals mid-2016 (two-year formation) — the sibling studies' start |
| NSE sector / basic industry | GICS industry group (`SECTOR_LEVEL`), monthly from 2017, earliest-on-record before | that is what the database carries; basic industry would leave many groups below five names. A group needs ≥ 5 other eligible names (`MIN_PEERS`) |
| ADF p < 0.05 on the residual | Engle-Granger p (MacKinnon, N = 2) on the residual, fixed lag 1 | the plain ADF table overstates significance on a fitted residual |
| Half-life 5–60 | **5–30** | two half-lives (z from −2 to −0.5) must fit inside the 60-session time stop |
| "Turned up" = flips to BUY, or is in BUY, within N | in BUY at the crossing, or a flip within 10 sessions **while z is still ≤ −1** and no −3.5 stop has fired; entry = the day both hold | "flips or is in BUY" made the flip clause idle; the still-cheap condition stops a bounce that has already closed the gap from counting |
| Returns from the crossing, split by "turned up within 10" | returns from the session after the entry's execution | counting from the crossing would put the bounce that flipped the supertrend inside the "turned-up" group's return: look-ahead |
| "Caught up ≥ 55 %" | net relative return at 20 sessions > 0 | "z came back" is the illusion the plan itself warns against |
| Cash in empty slots | the equal-weight universe in empty slots | a half-invested book beats the benchmark on worst fall without any skill |
| Benchmark = sector-neutral equal-weight book | the sector-matched book (the book's own group weights on the group returns, the same idle sleeve) for every mark, the break-even included; the equal-weight universe and the index as reference lines | the book's group weights follow the signals; against a fixed-weight benchmark the gap mixes sector bets with stock picking |
| 2 × 2 = cheap / turned-up / both / neither (§2) vs both / cheap / turned-up / random (Phase 2) | one 2 × 2 everywhere: both, cheap alone, turned-up alone, random (= neither) | the two lists differed; "turned-up alone" is defined (a BUY flip with z > −1) |
| Pass marks (Phase 1) | the plan's four, plus **≥ 150 events** (else INCONCLUSIVE) and **both > turned-up alone** | without a minimum n a thin cell could pass by luck; without the sixth mark a momentum rule would pass as a cheapness rule |
| Stress-test playbook | placebo (rich + turned up), block bootstrap, effective n, halves and years, count-matched random books | the items of `wm-EQrotStressTest` that apply to a selection rule |
| Stock-to-stock pairs | yearly refit, 20 per group, event study against the other leg | a monthly refit of every pair in every group is ~2M tests a year; yearly keeps it to minutes; no book, because the exits would need the pair's z |
| Costs 30–50 bps | 40 bps a side; grid 0–120; dividends not in the prices | no total-return series in the database; price returns on both sides, so the comparison is fair |

The three fixed choices the plan asked the desk to confirm — partner = the group basket, entry z = −2, "turned up" = the daily LW supertrend — are the base case and not grid dimensions.

## 2. Reading choices (confirm or correct)
1. Open positions keep the fit they were entered on; a refit that drops the stock from the pass list does not close the position (an exit rule has to fire).
2. The idle sleeve moves in and out of the equal-weight universe without cost; the book pays costs on stock trades only. The benchmark pays costs on its own monthly rebalance.
3. A delisted name is closed at its last print (the Eikon series ends at the delisting; the last print is taken as the exit price). Across a hole inside a name's history the move is credited on the session the price returns.
4. A name with no traded value in the legacy table (about 6 % of price rows lack one) cannot be screened; if it has none at all it passes on market cap, if it has some the median is taken over the sessions that have it. All are top-600 by market cap, so the floor rarely decides alone.
5. The sector before 2017 is the name's earliest classification on record — a mild look-ahead where a company changed group before 2017. `0_sector_source` says how much of the sample it covers.
6. Sharpe uses a zero risk-free rate, as the sibling studies do; the marks compare Sharpes against each other, so the rate cancels.
7. The sixth Phase 1 criterion (both > turned-up alone) compares means, not a test: the two cells overlap in time and the turned-up cell is ten times larger.

## 3. Data issues
* **The key is `quote_id`.** The first build keyed on the RIC and fetched prices from `merged_historical_data_with_rics` by RIC. That table stores most of a large name's older history with `ric` = NULL (HDFCBANK: NULL to 2023-11-24, `HDBK.NS` after), so 8 % of member-sessions — mostly the biggest names before 2023 — had no price. Splicing it with `ohlcv_eikon` closed the gap but mixed two adjustment bases (fake jumps; the equal-weight universe compounded at > 50 % a year). Rebuilt the wm-utils way (the table catalog and `GetRollingMcapUniverse`): `quote_id` everywhere, `ohlcv_eikon` the only price source, the legacy table used only for traded value.
* `ohlcv_eikon` starts 2014-07-01 and its `volume` is empty; `liquidity_check_raw_data` (the catalog's turnover table) starts 2025. Traded value for the liquidity floor therefore comes from the legacy table, as close × volume within its own rows, joined to the `quote_id` through every ticker the security has carried (`symbology_changes`).
* 11 universe quote_ids have no rows in `ohlcv_eikon` (`data/prices_missing.txt`).
* `mcap_based_universe` is daily; the fetch takes the last NIFTY 50 session of each month, as wm-utils does.
* `scores_value` is weekly after 2025-12; thinned to month-ends. `symbology_changes`' GICS is fetched at onboarding and carried forward — a classification, not a history.
* `index_constituents_monthly` does not help (2026-04 →, no Nifty 500).
* Suspect jumps (|one-day move| > 30 %) are listed in `0_suspect_jumps`; events near them are excluded, not corrected.

## 4. Open questions for the desk
1. Is a total-return series for the Nifty 500 (or a dividend table) available anywhere? Without it both the stock and its basket are price returns, which is fair but understates the level of every CAGR.
2. Should the group be the GICS industry group (≈ 25 groups, used) or the sector (11)? The sector variant is one config change (`SECTOR_LEVEL`).
3. Is 40 bps a side the desk's number for this universe, or should the base case be 50 with the smallcase impact?

## 5. Change log
* 2026-10-27 — the plan (`plan_stock_pairs_long_only.md`) and its review the same day; the amendments of §1 agreed in principle.
* 2026-10-28 — data layer rebuilt the wm-utils way (table catalog + `GetRollingMcapUniverse`): keyed on `quote_id`; `ohlcv_eikon` the only price source; traded value from the legacy table for the liquidity floor only; every connection read-only. The first build's RIC-keyed fetch had left 8 % of member-sessions unpriced, and a splice of the two price tables had produced fake jumps. A Phase 0 stop added: the equal-weight universe's CAGR must sit within 10 points of the index's (now −0.6 points). The break-even moved from the equal-weight book to the sector-matched benchmark (against the ew book a mostly idle book "passed" at any cost). Grid fixed (the minimum formation sessions now scale with the window).
* 2026-10-28 — **first real run.** Phase 0 PASS (98 % of member-sessions priced, median 295 eligible names, 24 groups). Level test: 7–10 % of fitted stocks pass a month against 5 % by chance, ~24 a month — stock-to-group cointegration is thin, as Study I found for indices. **Phase 1 INCONCLUSIVE**: 133 cheap + turned-up events (< 150); on them +0.08 % net over 20 sessions (+0.9 % gross), 40 % caught up, negative in the first half; above both nulls (96.5th percentile) and above turned-up alone (+1.5 points) — but the nulls and "turned-up alone" are themselves negative (−1.4 % net: a stock whose daily supertrend flips to BUY trails its group over the next month). Cheap alone −1.8 % (the falling knife). The study stops at Phase 1. **Forced, for inspection** (`outputs/forced/`): the book is level with its sector-matched benchmark (13.7 % vs 13.8 % net CAGR, Sharpe 0.78 vs 0.79), break-even 24 bps, 85th percentile of random books; 0 of 11 grid settings beat the benchmark on Sharpe; the placebo (rich + turned up) is −1.8 %; stock-to-stock pairs −0.7 % on 1,533 events; block bootstrap active Sharpe −0.15 (5th–95th −0.72 to +0.36).
* 2026-10-28 — repo built: config, data layer (universe from the next session, liquidity, baskets ex-self, benchmarks), formation (monthly fits, Engle-Granger with MacKinnon p-values, daily z under the fit in force), Phase 1 (cells, forward relative returns from the session after execution, two count-matched nulls, verdict), Phase 2 (daily simulation with the five exits, three benchmarks, cost grid and break-even, random books, the 2 × 2 as books, verdict), Phase 3 (grid, placebo, block bootstrap, effective n, stock-to-stock pairs), report, 45 checks. Data fetched: 1,284 names, 3,048 sessions 2014-07-01 → 2026-10-27, 96.4 % of universe rows with a group.
