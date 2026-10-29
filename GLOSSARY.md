# GLOSSARY — everything used in this study, in plain words

Read this with `outputs/report.html`. Each entry says what the thing is, how it is made here, and how to read the number. Table prefixes follow the phases (`0_` data, `1_` signal and mechanism, `2_` book, `3_` robustness).

## The pieces everything is built from

**Session** — one trading day on the NSE calendar (the days the Nifty 500 has a close). Everything here is daily; "20 sessions" is about a month, "60" about a quarter.

**qid (quote_id)** — the permanent security identifier the database keys prices, the universe and the sectors on. Used instead of the RIC or the ticker because both change over a company's life (a rename, a merger, a delisting suffix). Every table carries `qid` and, for reading, the latest `ticker`. A "name" in this document is a qid.

**Universe** — at each month-end trading day, the 500 largest ordinary shares by market cap (`mcap_based_universe`, built the way wm-utils builds its rolling universe); that list is used from the next session until the next month-end. Built the way the Nifty 500 is, but not the index itself (no historical membership exists in the database). Names that later delisted are in it while they were large, so nothing is known about the future.

**Liquidity floor** — a name must have traded at least ₹5 crore a day (median over six months) to be used. The traded value comes from the legacy price table (close × volume in the same row) and is used for this floor only; a name with no traded value on record passes on market cap alone.

**Price** — the adjusted daily close from `ohlcv_eikon`, one continuous series per qid across renames and RIC changes. It is the only price source: never mixed with another table's prices.

**Group / sector** — the GICS industry group (Banks, Capital Goods, Materials, Software & Services, …; 24 in the record). Monthly from 2017; before that, each name's earliest monthly classification on record. A name needs at least five other eligible names in its group to have a basket.

**Eligible** — in the universe, liquid, priced, with a group of at least five peers. The `eligible` line in `0_universe_size`.

**Partner basket** — a stock's own group with the stock itself left out, equal-weighted, rebuilt every day from the members decided the session before. "The stock against its sector" always means against this basket. (The same basket *with* the stock is what the benchmark holds.)

**Return** — the day's move as a fraction, P today ÷ P yesterday − 1. Prices are adjusted for splits and bonuses, not for dividends, on both the stock and its basket. A day with no print is left blank; over several days the stock is treated as holding its last price.

**Supertrend / BUY / SELL** — the desk's trend rule on the stock's own daily bars: a band 2 × ATR above and below the mid-price (ATR = a 5-day weighted average of the daily range); BUY after a close above the upper band, SELL after a close below the lower one; the bands only tighten in the trend's direction. "Flip" = the day the state changes. "Turned up" = BUY.

## §2 — the signal (tables `1_fit_summary`, `1_fits`, `1_z_distribution`, `1_breadth`)

**Formation / fit** — at every month-end, for every eligible stock, a straight line of its log price on its basket's log level over the last two years (504 sessions). The slope is the stock's beta to its group; the residual is how far the stock sits from the line.

**z** — the residual divided by its own standard deviation over the formation window. −2 means two standard deviations below the relationship: "cheap against its group". The fit made at month-end m is used for every day of month m+1; nothing after the day itself is ever used. A position keeps the fit it was entered on.

**Level test / "a pair exists"** — the residual must have a long-run level: an Engle-Granger test (an ADF on the residual with the stricter two-variable p-values, because the residual came from a fitted line) with p below 0.05, a **half-life** between 5 and 30 sessions (the time for half of a deviation to fade), and a positive slope. A stock that fails has no signal that month, whatever its z says — the lesson of Study I (correlation is not cointegration). `pass_share` is the share of fitted stocks that pass; 5 % would pass by chance.

**`1_fit_summary` columns** — `fitted`, `passing`, `pass_share`, `chance_share` (0.05), `p_lt_005_share`, `median_p`, `median_beta`, `median_halflife_passing`, `median_sd_passing`, `share_hl_too_long` (> 30), `share_hl_too_short` (< 5). **`1_fits`**: every (refit date, qid): `alpha`, `beta`, `sd`, `adf_t`, `p`, `halflife`, `n`, `ok`.

**`1_z_distribution`** — over all sessions × names with a passing fit: the share of z at or below −2, −1, inside ±1, at or above +2. **`1_breadth`** — the share of names in BUY, month by month.

## §3 — Phase 1, the mechanism (tables `1_events`, `1_cell_stats`, `1_halves`, `1_by_year`, `1_by_sector`, `1_paths`, `1_null_*`, `1_crossing_state`, `1_effective_n`, `1_excluded`, `1_verdict`)

**Cheap crossing** — the day z goes from above −2 to at or below −2, under a passing fit, for an eligible stock. One per *run*: once a stock has crossed it is "in a run" until z is back above −0.5 or 60 sessions have passed; deeper readings inside the run are not new events.

**The cells (the 2 × 2)** —
- **both** — a cheap crossing where the stock is already in BUY, or flips to BUY within 10 sessions while z is still at or below −1 (and no −3.5 stop has fired). Entry = the day both hold. The cell the study rests on.
- **cheap alone** — a cheap crossing that never qualifies within 10 sessions: the falling knife. Entry = the crossing.
- **turned-up alone** — a flip to BUY while the stock is not cheap (z above −1). The trend without the cheapness. Entry = the flip.
- **neither / random** — the count-matched null (below).

**Entry date / execution** — the signal is read at the entry day's close and executed at the next close (one-session delay). Every forward measure starts the session after execution.

**Relative return `rel_h`** — the stock's log return minus its partner basket's over the h sessions after execution (h = 5, 20, 60). **`net_h`** = that minus 80 basis points (40 a side on the stock). **Caught up** = `net_20` > 0. These are the measures — never the z.

**`z_back_h`** — for reference only: did z get back to −0.5 within h sessions? It comes back by construction (the rolling fit); the study never reads it as the stock coming back.

**`1_cell_stats` columns** — per cell and horizon: `events`, `dates` (distinct entry dates), `mean_rel_gross`, `mean_rel_net`, `median_rel_net`, `hit_net` (share positive net), `sd`, `hac_t` (a Newey-West t on one value per date, so clustered events do not inflate it), `z_back_share`. `1_by_year` / `1_by_sector` / `1_halves` are the same at 20 sessions, cut by entry year, group, and the two halves of the calendar.

**`1_paths`** — the average cumulative relative return k = 0 … 60 sessions after execution, per cell (the chart's lines); `share_positive` at each k.

**Count-matched null** — the same number of events as the both cell, drawn at random, 200 times, each giving one mean: (a) the same stocks on random eligible dates; (b) random eligible names of the same group on the same dates. `p95` is the mean that only 5 % of random draws beat; `actual_percentile` is where the real mean sits. The pass mark uses the stricter of the two.

**`1_crossing_state`** — of all cheap crossings: the share where the stock was already in SELL (Study K's finding was 59 %), the share that qualified as both, the typical days to the turn.

**`1_effective_n`** — events share dates and stocks: distinct dates, distinct stocks, events at least 30 sessions apart within a stock, the most events on one date. The honest n is nearer the smaller numbers.

**`1_excluded`** — events within ±5 sessions of a one-day move beyond ±30 % for the stock (an adjustment artefact or a corporate event); listed, not used.

**`1_verdict`** — the six pre-registered criteria with value, mark and ✔/✘; PASS / FAIL / INCONCLUSIVE (fewer than 150 events).

## §4 — Phase 2, the book (tables `2_nav`, `2_trades`, `2_trade_summary`, `2_metrics`, `2_halves`, `2_by_year`, `2_cost_grid`, `2_random_*`, `2_cells`, `2_sector_weights`, `2_verdict`)

**The book** — up to 30 positions, at most 4 per group, each bought with a thirtieth of the book at entry and held without rebalancing until an exit; candidates ranked by z (most negative first). Costs 40 bps a side on every stock trade. Whatever the slots do not hold sits in the equal-weight universe, not cash — so an empty book is the benchmark, and a shallow drawdown cannot come from being out of the market.

**Exits** (decided at one close, executed at the next): **caught up** (z under the entry fit back to −0.5), **turn failed** (supertrend back to SELL), **structural** (z at or below −3.5: the relationship broke), **time** (60 sessions), **delisted** (no further print: closed at the last price). `2_trade_summary` gives the count, share and average result of each.

**Equal-weight universe (`ew`)** — every eligible name at equal weight, rebalanced at the first session of each month, paying the same 40 bps a side on its turnover. A reference line on the NAV chart.

**Sector-matched benchmark (`matched`)** — each day, the book's own group weights of the session before applied to the group returns (the group *with* the stock), the remainder on the equal-weight universe. It has the book's sector bets and none of its stock picks; beating it is stock selection, beating only `ew` could be a sector tilt. The pass marks use it.

**Index** — the NIFTY 500 price index, for reference.

**`2_metrics` columns** — `cagr`, `vol`, `sharpe` (mean ÷ sd × √252, risk-free taken as 0), `worst_fall` (largest peak-to-trough fall), `hit` (share of up days), `calmar`; against a benchmark: `bench_cagr`, `bench_sharpe`, `bench_worst_fall`, `active_cagr`, `tracking_error`, `active_sharpe` (Sharpe of book − benchmark), `share_days_ahead`; `turnover_per_year` (one-way, as a share of the book). `2_halves` and `2_by_year` are the same against the matched benchmark by half and by year. `gross_sharpe` (in the verdict) is the book run at zero cost.

**`2_cost_grid` / break-even** — the book re-run at 0, 20, …, 120 bps a side against its sector-matched benchmark; the **break-even cost** is where the book's net CAGR equals the benchmark's — how much cost the picks' edge over their groups can absorb. 0 means the picks trail their groups even before costs. The mark is 80.

**Random books (`2_random_books`, `2_random_summary`)** — 200 books in which every actual trade is replaced by a random eligible name of the same group on the same date, held as long. Their gross Sharpe says what the discipline alone (dates, groups, holds, caps) earns; the book must beat 95 % of them.

**`2_cells`** — the 2 × 2 run as books: both (the book), cheap alone (no supertrend exit), turned-up alone (no z exits), random (the median draw).

**`2_verdict`** — the five Phase 2 criteria.

## §5 — Phase 3 (tables `3_grid`, `3_placebo_*`, `3_bootstrap`, `3_sp_*`)

**Grid (`3_grid`)** — the base case and each single change: formation 1 / 2 / 3 years, entry z −1.5 / −2 / −2.5, exit −0.5 / 0, daily / weekly supertrend, hold cap 40 / 60 / 90, 20 / 30 / 40 positions. Per row the both cell's 20-session mean, hit and t, and the book's net CAGR, Sharpe, worst fall against the matched benchmark (`beats_matched_sharpe`, `beats_matched_fall`). Reported whole; a robustness check, not a search.

**Placebo (`3_placebo_stats`, `3_placebo_metrics`)** — the mirror rule, *rich* (z at or above +2) and turned up, as events and as a long book. If it does as well as the real rule, the supertrend is doing the work and cheapness adds nothing.

**Block bootstrap (`3_bootstrap`)** — the book's daily active return against the matched benchmark resampled in 20-session blocks, 1000 times: the 5th–95th percentile band around the active Sharpe and the share of draws at or below zero.

**Stock-to-stock pairs (`3_sp_fits`, `3_sp_events`, `3_sp_stats`)** — the secondary variant: within each group at every year-end, every pair of names is fitted the same way; the 20 best-fitting pairs per group are kept for the following year; the cheap leg is entered on the same cheap + turned-up rule and measured against the *other leg* instead of the basket. Event study only (no book); `leg` and `other` name the two.

## §1 — the data audit (tables `0_coverage`, `0_universe_size`, `0_groups`, `0_sector_source`, `0_suspect_jumps`, `0_delistings`, `0_ew_vs_index_monthly`, `0_verdict`)

`0_coverage`: per qid the ticker, first / last print, sessions, holes (sessions without a print inside the name's history), and the share of its sessions with a traded value. `0_universe_size`: month by month, members → liquid → priced → with sector → eligible, and groups in use. `0_groups`: per group the mean / min / max eligible count. `0_sector_source`: by year, the share of universe rows whose group came from the monthly record, the earliest monthly record, or the security master. `0_suspect_jumps`: every one-day move beyond ±30 % (qid, ticker, log return, in the universe or not). `0_delistings`: names whose last print is before the last session. `0_ew_vs_index_monthly`: the equal-weight universe's monthly return beside the Nifty 500's (a sanity check; they should move together). `0_verdict`: the six audit checks — five are a WARN; the sixth (the equal-weight universe's CAGR within 10 points of the index's) STOPS the run, because a broken price panel shows there first.
