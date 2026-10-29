# Study L — stock pairs, long-only: the plan

Prepared 2026-10-27 after the review of the same date. Builds on Studies I (correlation and the pair tests), J (the drift as a weight) and K (the drift in the rotation; costs, null, pass marks).

---

## 1. What we are testing, in one sentence

Within a sector, a stock that has fallen well below its long-run relationship with its peers — and whose own trend has turned up — outperforms its sector over the next one to three months, net of costs, often enough to run as a long-only selection rule.

Everything below is built to answer that sentence with a yes or a no, and to stop early if the answer is no.

## 2. What we learned from I, J, K — and how each lesson shapes this study

| Lesson | Where it came from | What it changes here |
|---|---|---|
| Correlation is not cointegration. Two things can move together every week and drift apart forever. | I: 0.8–0.94 correlation, 9 of 10 index ratios with no long-run level | Pairs are formed on a **level** relationship (cointegration test, half-life), never on correlation. A pair that fails the level test is not a pair. |
| The z "comes back" by construction; the asset need not. | I, J, K: 87% of −1 sd crossings came back, the asset did +0.5% | The mechanism is tested on the **stock's later return against its partner**, never on whether z returned. |
| Test the mechanism before the strategy. | J and K built books on a signal that had no forward information | Phase 1 is an **event study** (does the cheap leg catch up?). The book is built only if it passes. |
| The gate sees the fall first; the laggard is usually already in SELL. | K: 59% of crossings with the asset in SELL | "Cheap" alone is a falling knife. The entry requires **cheap + turned up** (the stock's own supertrend back in BUY). The 2 × 2 (cheap alone / turned-up alone / both / neither) is run so we know which part works. |
| A pick is only as good as random picks with the same discipline. | K: Study D's rotation at the 24th percentile of a count-matched null | A **count-matched random null** is in the design from day one: random stocks from the same sector, entered on the same dates, held the same way. |
| Costs and the traded instrument decide, not gross numbers. | K: edges gone at 18–64 bps a side | Costs in from the first table (stocks: 30–50 bps a side incl. impact), adjusted prices, a one-day execution delay as the base case. |
| Pre-register the marks; the grid is a robustness check, not a search. | J, K | One base case fixed in `config.py` before any run; pass marks written before the run; the grid reported in full. |
| Concentration raises return and drawdown; diversification across many independent bets is what a stock book has that an index book does not. | K: top-1 23% CAGR with −21% fall; D top-3 no better than random | The book holds **many positions** (20–40), equal-weighted, sector-capped — the edge, if any, is small per trade and has to be collected across hundreds of trades. |
| Long-only means the "short" leg is an underweight, not a short. | the review of 2026-10-27 | The strategy is a **selection rule inside a sector-neutral long book**: hold the cheap-and-turned-up stocks instead of all the sector's stocks. The benchmark is the sector-neutral equal-weight book, not the index. |

## 3. Universe and data

- **Universe**: Nifty 500 constituents, point-in-time (historical membership, so no survivorship bias), with a liquidity floor (e.g. median 6-month daily traded value ≥ ₹5 cr). Expect 350–450 names in any year.
- **Grouping**: NSE sector / basic-industry classification. A stock's **partner** is its sector basket (equal-weighted peers, excluding itself) — one regression per stock, hundreds of "pairs", few free choices. Stock-to-stock pairs are the secondary variant (see §5).
- **Prices**: daily OHLC adjusted for splits and bonuses; total return (dividends) for the book's P&L; the supertrend on the adjusted OHLC.
- **From the desk database**: stock daily prices, sector mapping, index membership history, traded value. From NSE if missing: historical constituent lists, TRI.
- **Window**: 2012 → today (ten-plus years, three to four sector cycles).

## 4. The signal

- **Formation**: for each stock, on the trailing 2 years (504 sessions), regress log price on log sector-basket level (the regression ratio = the stock's beta to its sector). Residual e; **z = e / sd(e)**. Re-fit monthly on trailing data only (the daily z between re-fits uses the last fit). Nothing after t is ever used.
- **A pair exists** only if the residual passes a level test on the formation window: ADF p < 0.05 and half-life between 5 and 60 sessions. Stocks that fail have no signal that month (this is the cointegration lesson applied).
- **Cheap**: z ≤ −2 (the stock sits 2 sd below its sector relationship).
- **Turned up**: the stock's own desk supertrend (daily LW, 5-bar ATR × 2; weekly as the robustness variant) flips to BUY, or is in BUY, within the last N sessions of the cheap reading.
- **Entry**: the first session both hold. **Exit**: z back to −0.5 (caught up), or the supertrend back to SELL (the turn failed), or 60 sessions (time stop), or z ≤ −3.5 (structural — the relationship broke; a value trap, not a drift).

## 5. Phases, with a stop rule at each

**Phase 0 — data audit (1 day).** Adjusted prices reconcile to a few known corporate actions; constituents point-in-time; sector mapping stable; liquidity floor applied. Output: a coverage table and a list of known gaps. Stop if the price history cannot be trusted.

**Phase 1 — the mechanism, event by event (2 days).** Every z ≤ −2 crossing of every stock-sector pair that passed the level test, 2012 →: the stock's return minus its sector's over the next 5 / 20 / 60 sessions; the share that caught up; split by whether the stock's supertrend was in BUY or SELL at the crossing, and by whether it turned up within 10 sessions. The count-matched null here is: the same stocks' returns on random dates, and random sector peers on the same dates.

*Pass mark (fixed now):* among "cheap + turned up" events, mean relative return over 20 sessions ≥ +1.5% after 40 bps a side, caught up ≥ 55% of the time, the same sign on both halves of the sample, and above the 95th percentile of the null. **If this fails, the study stops here** — with the event table as the deliverable, which is itself worth having (it says whether sector mean reversion exists at the stock level in India).

**Phase 2 — the book (2 days), only if Phase 1 passes.** A long-only portfolio: at each session, the open signals ranked by z (most negative first); up to 30 positions, equal-weighted, at most 4 per sector, cash for empty slots; the benchmark is the sector-neutral equal-weight book of the same universe, and the Nifty 500 TRI for reference. Costs 40 bps a side; one-day execution delay. The 2 × 2 of §2 (cheap alone / turned-up alone / both / random) as four books.

*Pass marks (fixed now):* beat the sector-neutral benchmark on net Sharpe and on worst fall; the net Sharpe above the benchmark's on both halves; gross Sharpe above the 95th percentile of the count-matched random book; break-even cost ≥ 80 bps a side.

**Phase 3 — robustness and the note (1–2 days).** The grid: formation window (1 / 2 / 3 years), entry z (−1.5 / −2 / −2.5), exit (−0.5 / 0), supertrend bar (daily / weekly), hold cap (40 / 60 / 90), position count (20 / 30 / 40); the desk's stress-test playbook; the stock-to-stock pair variant (Engle-Granger within sector, the 20 best-fitting pairs per sector) as a comparison; a decision note in the shape of "Gold, Not Timing".

## 6. What could fool us, and the control for each

| Trap | Control |
|---|---|
| Survivorship (only today's constituents) | Point-in-time membership; delisted names kept until their last print |
| Corporate actions making a "cheap" reading out of a split | Adjusted prices; events within 5 sessions of a corporate action excluded and listed |
| The value trap: cheap because the business broke | The turn-up requirement; the −3.5 structural exit; the event study reports the "cheap alone" cell separately so the size of the trap is visible |
| The z illusion | Returns against the partner, never the z, as the measure |
| Overlapping events inflating n | Events per stock counted once per run; the null uses the same count |
| Multiple testing across sectors and parameters | One base case, pass marks first, the grid reported whole; no sector dropped after the fact |
| Sector regimes (a whole sector cheap vs the market) | The partner is the sector basket, so a sector-wide fall is not a signal; the book is sector-capped |
| Liquidity and impact at the small end | The liquidity floor; costs at 40 bps with 60 as a sensitivity |

## 7. What "it works" and "it does not" will look like

- **Works**: Phase 1 passes on the "both" cell and not on "cheap alone" — mean reversion exists but needs the trend confirmation; Phase 2's book beats the sector-neutral benchmark on both Sharpe and worst fall net of 40 bps, holds on both halves, and sits above the null. Then the question becomes sizing and implementation (turnover, capacity, the smallcase constraints).
- **Does not**: Phase 1's "both" cell is a coin flip, or passes gross and fails at 40 bps — the same shape as I, J and K. The deliverable is the event table and a one-page note that closes stock-level mean reversion for this universe the way K closed it for the indices.

## 8. What is needed from the desk to start

- Database access for stock daily prices, sector mapping and traded value (fetch scripts in the repo, credentials in `.env` as before).
- Historical Nifty 500 membership and TRI (NSE files) if not in the database.
- Confirmation of the three fixed choices above: partner = sector basket (primary), entry z = −2, "turned up" = the daily LW supertrend. Everything else is a grid dimension.

Repo shape as the sibling studies: `config.py` (every parameter, pass marks included), `run_all.py`, `make_report.py`, `checks.py`, GLOSSARY / LOGIC / UNDERSTANDING, no person's name anywhere, `.env` git-ignored.
