# LOGIC — exact definitions

Securities are identified by i = `quote_id` (`qid`) throughout. Daily sessions t = NIFTY 50's sessions (`index_price_data`, as wm-utils) from `DATA_START` = 2014-07-01 to the last session with ≥ 20 stock prints. P_i,t = the adjusted close from `ohlcv_eikon` (splits, bonuses; no dividends) — the only price source. r_i,t = P_i,t / P_i,s − 1 where s is the last session before t with a print (a hole's move is credited on the session the price returns); NaN where P_i,t is missing. Cumulative log returns CL_i,t = Σ_{s≤t} ln(1 + r_i,s) with a missing r counted as 0.

## Universe — `universe.py` (tables 0_)
* Snapshot at each month-end trading day d (the last NIFTY 50 session of the month): `mcap_based_universe` rows on d with `instrument_class = 'ordinary'`, ranked by `mcap` descending (method `first`), the top `UNIVERSE_N` = 500 (fetched to 600) — wm-utils `GetRollingMcapUniverse`. Identifiers and GICS by `quote_id` from `symbology_changes` (`drop_duplicates(keep='last')`). **In force from the first session strictly after d** until the next snapshot (`_apply_from_next_session`). The same rule for the sector snapshot.
* Sector: GICS industry group (`SECTOR_LEVEL`). Monthly from `scores_value` by `quote_id` (2017-01 →); before the monthly record, the earliest monthly record for the `quote_id`; failing that, the security master's value (`symbology_changes`, fetched at onboarding). `sector_source` ∈ {monthly, earliest, master}.
* Liquidity: TV_i,t = close × volume / 10⁷ (₹ crore) from the legacy `merged_historical_data_with_rics` row of i's ticker on t (rows whose `ric` is NULL or one of i's RICs); liquid_i,t = median(TV over t−125..t, ≥ 63 obs) ≥ `LIQ_FLOOR_CR` = 5. An i with no traded value on record at all is liquid. TV feeds this floor only, never a return.
* E0_i,t = member & liquid & P_i,t present & sector present. peers_i,t = number of other E0 names in i's group at t. **E_i,t = E0 & peers ≥ `MIN_PEERS` = 5.**
* Baskets, members M_t = E_t−1 & r_t present (decided the session before):
  R_grp_i,t = mean_{j ∈ group(i,t), M} r_j,t (incl. i); **R_ex_i,t = (Σ_{j∈group, M} r_j,t − r_i,t·1[i∈M]) / (n − 1[i∈M])** — i's partner basket; R_sec_s,t = mean over the group's members; R_uni_t = mean over all members. Basket log level B_i,t = Σ_{s≤t} ln(1 + R_ex_i,s), a missing day flat.
* Equal-weight benchmark (`ew_benchmark`): w = 1/n over M at the first session (turnover 1, cost c); each day w_j ← w_j(1 + r_j)/Σ; at the first session of each month w ← 1/n over M_t, turnover τ = Σ|target − drifted| / 2, cost 2τc; NAV_t = NAV_t−1 (1 − cost_t)(1 + Σ w r_t). c = `COST_BPS` / 10⁴.
* Phase 0 verdict (`0_verdict`, `PASS_0`): share of member-sessions with a price ≥ 0.97; with a sector ≥ 0.95; median eligible per session ≥ 250; suspect jumps (|ln(P_t/P_s)| > `CA_JUMP` = 0.30, in universe) per member-year ≤ 0.05; corr(R_uni, index return) ≥ 0.85 — WARN if any fails. **|CAGR(R_uni) − CAGR(index)| ≤ 0.10** over the sessions both exist — FAIL and STOP if not (a broken price panel shows here first: the spliced panel of 2026-10-28 gave an equal-weight CAGR above 50 %).

## Formation — `formation.py` (tables 1_fit*)
* Refit dates d: the last session of each month from the first with `FORMATION` = 504 sessions of history. For every i with E_i,d and ≥ `MIN_FORMATION_SHARE` × 504 = 403 sessions with both P and R_ex present in d−503..d (the share scales with the window in the grid):
  ln P_i,t = α + β ln B_i,t + e_t (OLS); sd = sd(e, ddof 2).
* Level test (Engle-Granger): Δe_t = c + ρ e_t−1 + φ Δe_t−1 (`ADF_LAG` = 1); t_ρ; p = MacKinnon(t_ρ, constant, N = 2 variables) (`statsmodels.tsa.adfvalues.mackinnonp`). Half-life HL = −ln 2 / ln(1 + ρ₀) from Δe_t = c + ρ₀ e_t−1, ∞ unless −1 < ρ₀ < 0.
  **ok = p < 0.05 & 5 ≤ HL ≤ 30 & β > 0 & sd > 0.**
* Daily: for t in (d, d′] (d′ the next refit) z_i,t = (ln P_i,t − α_d − β_d ln B_i,t) / sd_d; OK_i,t = ok_d; FIT_t = d. On d itself the previous fit is in force.
* `1_fit_summary`: per d, fitted, passing, pass share (vs 0.05 by chance), medians.

## Phase 1 — `events.py` (tables 1_)
* Cheap crossing: z_i,t ≤ `Z_ENTRY` = −2 and z_i,t−1 > −2 and OK & E. One per run: after a crossing at t the run lasts until the first s > t with z_i,s > `Z_EXIT` = −0.5, or t + `TIME_STOP` = 60; crossings inside the run are ignored.
* Cells. At a crossing (t, i) with supertrend state s_i,t (daily, `supertrend.py`):
  - s = +1 → **both**, entry = t, days_to_turn 0.
  - s = −1 → look at t+1 … t+`TURN_WINDOW` (10): stop at the first session with z ≤ `Z_STOP` (−3.5) or the first flip (s_t′ = +1, s_t′−1 = −1); the flip qualifies iff z_i,t′ ≤ `Z_STILL_CHEAP` (−1) & OK & E → **both**, entry = t′. Otherwise **cheap alone**, entry = t.
  - **turned-up alone**: every flip (s_t = +1, s_t−1 = −1) with E, z present and z > −1; entry = t.
* Forward relative return, t_x = entry + `DELAY` (1): rel_h = (CL_i,t_x+h − CL_i,t_x) − (CLB_i,t_x+h − CLB_i,t_x), CLB from R_ex; NaN unless t_x + h is inside the calendar. net_h = rel_h − 2 × `COST_BPS`/10⁴. caught_up = net_20 > 0. z_back_h = 1[max z over t_x+1..t_x+h ≥ −0.5] (reference only).
* Exclusion: an event is dropped if any |ln(P_t/P_t−1)| > 0.30 for the stock within ±`CA_EXCLUDE` = 5 sessions of its entry (`1_excluded`).
* Statistics per cell and h (`1_cell_stats`): n, distinct dates, mean gross, mean / median net, hit = share net > 0, sd, Newey-West t of the per-date means (one value per entry date, lags max(h, 5)). `1_halves` splits at the calendar's middle session. `1_paths`: mean cumulative rel for k = 0..60.
* Nulls (`1_null_summary`, `NULL_DRAWS` = 200, seed 7), for the both cell with a full horizon: (a) same stock, a uniformly random session where E holds and t + 1 + 20 is inside the calendar; (b) same date, a uniformly random other E name of the same group. Each draw: the mean of net_20 over as many draws as events. actual_percentile = share of draws below the actual mean.
* Verdict (`1_verdict`, `PASS_1`): n ≥ 150 else INCONCLUSIVE; mean net_20 ≥ +1.5 %; hit ≥ 0.55; mean net_20 > 0 on both halves; min over the two nulls of actual_percentile ≥ 0.95; mean net_20(both) > mean net_20(turned-up alone). PASS iff all.

## Phase 2 — `book.py` (tables 2_)
* Entries: the both cell's (entry date, qid, z at entry, fit date). Start = the session after the first refit date.
* Each session t (after the idle sleeve earns R_uni_t):
  1. Exits for positions with held = t − t_exec ≥ 1, decided on t−1: z′ = z under the **entry fit** at t−1; in order: fixed hold (random books); z′ ≥ `Z_EXIT` → "caught up"; z′ ≤ `Z_STOP` → "structural"; s_i,t−1 = −1 → "turn failed"; held ≥ `TIME_STOP` → "time". If t is past the stock's last print → "delisted" at the last price. Sold at P_t (1 − c); if P_t missing (a hole) the exit waits for the next print.
  2. Entries: candidates whose entry date is t − `DELAY`, E at t−1, not held, ranked by z ascending; taken while positions < `N_POS` = 30 and the group count < `SECTOR_CAP` = 4; cash = min(NAV/30, idle); shares = cash(1 − c)/P_t.
  3. NAV_t = Σ shares × last price + idle. Sector weights w_s,t = value in group s / NAV.
* Trades: gross = P_out/P_in − 1; net = (1 − c)(1 + gross)(1 − c) − 1; basket = exp(CLB_out − CLB_in) − 1; rel_net = net − basket.
* Benchmarks: **ew** = `ew_benchmark` at the same cost from the same start; **matched**_t = Σ_s w_s,t−1 R_sec_s,t + (1 − Σ_s w_s,t−1) R_uni_t; **index** = NIFTY 500 price return.
* Metrics on daily returns: CAGR, vol = sd × √252, Sharpe = mean/sd × √252 (rf 0), worst fall = min(NAV/cummax − 1), hit; against a benchmark b: active = r − b, active Sharpe, tracking error, share of days ahead. Halves at the middle session from the start; by calendar year.
* Break-even (`2_cost_grid`): the book re-run at each cost in `COST_SENS`, against its own sector-matched benchmark (cost-free, same idle sleeve); be = the cost where (book CAGR − matched CAGR) crosses 0 by linear interpolation; ∞ if positive at 120, 0 if not positive at 0. (Against the ew book, which pays a monthly rebalance while the book's idle sleeve does not, a mostly idle book passes at any cost — changed 2026-10-28.)
* Random books (`2_random_books`): per draw every actual trade → a random other E name of the same group at the same signal date with `hold` = the actual hold; simulated gross (c = 0) and net with fixed-hold exits only; the gross Sharpe distribution.
* The 2 × 2 as books (`2_cells`): both (all exits); cheap alone (no supertrend exit); turned-up alone (no z exits); random = the median draw.
* Verdict (`2_verdict`, `PASS_2`): Sharpe > matched Sharpe; worst fall > matched worst fall; Sharpe > matched Sharpe on both halves; share of random books with gross Sharpe below the book's ≥ 0.95; be ≥ 80.

## Phase 3 — `grid.py` (tables 3_)
* Grid: the base case and each single change in `GRID` (formation 252/756, z_entry −1.5/−2.5 with Z_STILL_CHEAP = z_entry/2, z_exit 0, weekly bar, time stop 40/90, 20/40 positions); each row = Phase 1 both-cell mean/hit/t at 20 sessions and the book's net metrics against the matched benchmark. `GRID_FULL` = the cross product.
* Placebo: classify on −z with the same thresholds (rich ≥ +2, still rich ≥ +1, stop ≥ +3.5); the book exits on −z. Reported as the "rich + turned up" cell and a book.
* Block bootstrap: a_t = book − matched; `BLOCK_BOOT` = 1000 resamples of blocks of `BLOCK_LEN` = 20 sessions to the same length; Sharpe and annualised mean per draw; 5th / 50th / 95th percentiles, share ≤ 0.
* Effective n: per cell, events, distinct dates, distinct stocks, events ≥ 30 sessions apart within a stock, the largest number of events on one date.
* Stock pairs: at each year-end session with 504 sessions of history, in each group, every pair (a, b) of E names with ≥ 403 common sessions: ln P_a = α + β ln P_b + e (the same test as the formation); kept iff ok; the `SP_TOP` = 20 lowest p per group. Over the following year z = (ln P_a − α − β ln P_b)/sd; leg a's column = +z, leg b's = −z; a leg's partner = the other leg; the same cells and forward measure with CLB = the other leg's CL; E for a leg = E of both legs. Event study only.

## Supertrend — `supertrend.py`
Daily bars (the base case): TR = max(H − L, |H − C₋₁|, |L − C₋₁|); ATR = Σ_{k=1..5} (k/15) TR_t−5+k; bands (H + L)/2 ± 2 ATR, ratcheting; state +1 after a close above the upper band from −1, −1 after a close below the lower band from +1. Weekly variant: the rule on W-FRI bars; the state known at the Friday close applies from the next session.
