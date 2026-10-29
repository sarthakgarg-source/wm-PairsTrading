"""
wm-PairsTrading  ·  Study L  ·  Stock pairs, long-only

  Within a sector, a stock that has fallen well below its long-run relationship with its
  peers - and whose own trend has turned up - outperforms its sector over the next one to
  three months, net of costs, often enough to run as a long-only selection rule.

  Phase 0  data audit                    (stop if the price history cannot be trusted)
  Phase 1  the mechanism, event by event (stop if the "cheap + turned up" cell fails its marks)
  Phase 2  the long-only book            (only if Phase 1 passes)
  Phase 3  robustness, placebo, stock-to-stock pairs

Daily sessions. Every parameter lives here, the pass marks included; they were written before
the first run (plan of 2026-10-27, amended at the review of the same date) and are not tuned.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("PAIRS_DATA_DIR", ROOT / "data"))      # overridable so the synthetic demo can run beside the real data
OUT = Path(os.environ.get("PAIRS_OUT_DIR", ROOT / "outputs"))
TABLES = OUT / "tables"
CHARTS = OUT / "charts"


def rel(p: Path) -> str:
    """A path for printing: relative to the repo when inside it, else as is."""
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)

# ------------------------------------------------------------- data files --
PRICES_FILE = DATA / "stock_ohlcv.parquet"        # qid, date, open, high, low, close, turnover_cr   (fetch_prices.py)
UNIVERSE_CSV = DATA / "universe_monthly.csv"      # date (month-end trading day), qid, ric, ticker, isin, name, mcap_cr, rank, gics_*, delist_date   (fetch_universe.py)
SECTORS_PIT_CSV = DATA / "sectors_monthly.csv"    # date (month-end), qid, gics_*   (fetch_sectors.py; 2017-01 ->)
INDEX_CSV = DATA / "index_prices.csv"             # date, one close column per index   (fetch_index.py)
SYNTHETIC_MARKER = DATA / "SYNTHETIC"             # written by make_demo_data.py only; its presence stamps every output

# --------------------------------------------------------------- universe --
DATA_START = "2014-07-01"        # ohlcv_eikon (the price source, delisted names included) starts here
UNIVERSE_N = 500                 # the largest N ordinary shares by market cap at each month-end = the point-in-time "Nifty 500-like" universe
UNIVERSE_FETCH_N = 600           # fetched with a buffer so a name that drops to 550 and comes back keeps its history
LIQ_FLOOR_CR = 5.0               # rupees crore: median daily traded value over LIQ_WINDOW sessions must be at least this
LIQ_WINDOW = 126                 # sessions (6 months)
SECTOR_LEVEL = "gics_industry_group_name"   # the grouping: GICS industry group (~25 groups); "gics_sector_name" (11) is the coarse variant
MIN_PEERS = 5                    # a stock needs at least this many eligible peers in its group to have a basket
INDEX_REF = "NIFTY 500"          # the reference index (price index; no TRI in the database)
CALENDAR_INDEX = "NIFTY 50"      # the trading calendar, as wm-utils (GetTimeSeriesData._get_trading_days)
INDEX_ALIASES = ["NIFTY 500", "NIFTY 100", "NIFTY 50", "NIFTY MIDCAP 150", "NIFTY SMLCAP 250", "NIFTY TOTAL MKT"]

# ----------------------------------------------------------------- signal --
FORMATION = 504                  # sessions in the formation window (2 years)
MIN_FORMATION_SHARE = 0.8        # a stock needs this share of the window's sessions with a price and a basket to be fitted
REFIT = "ME"                     # re-fit at every month-end; the fit applies from the next session (nothing after t is used)
ADF_LAG = 1                      # lagged differences in the ADF regression on the residual
ADF_P_MAX = 0.05                 # Engle-Granger p-value (MacKinnon, two variables) below which the residual has a level
HALFLIFE_MIN = 5                 # sessions
HALFLIFE_MAX = 30                # sessions: two half-lives (z from -2 to -0.5) must fit inside the time stop
Z_ENTRY = -2.0                   # cheap: the stock sits 2 sd below its sector relationship
Z_STILL_CHEAP = -1.0             # at the entry session (cheap AND turned up) z must still be at or below this
Z_EXIT = -0.5                    # caught up
Z_STOP = -3.5                    # structural: the relationship broke - a value trap, not a drift
TURN_WINDOW = 10                 # sessions after the cheap crossing in which the supertrend may turn to BUY
TIME_STOP = 60                   # sessions: exit anyway (= 2 x HALFLIFE_MAX)

LOOKBACK, MULTIPLIER, ATR_METHOD = 5, 2.0, "lw"      # the desk's supertrend, recovered not tuned
ST_BAR = "daily"                 # the base case; "weekly" (W-FRI bars, state carried daily) is a grid variant
ST_WEEKLY = "W-FRI"

# ---------------------------------------------------- costs and execution --
COST_BPS = 40                    # one-way, per side, impact included; stocks
COST_SENS = (0, 20, 40, 60, 80, 100, 120)   # the cost grid the break-even is read off
DELAY = 1                        # sessions: a signal at close t is traded at close t+1
CA_JUMP = 0.30                   # |log return| above this in one session marks a suspect corporate-action artefact
CA_EXCLUDE = 5                   # events within this many sessions of a suspect jump are excluded and listed

# --------------------------------------------- Phase 1: the event study --
EVENT_H = (5, 20, 60)            # sessions ahead
H_MARK = 20                      # the horizon the pass mark is read at
NULL_DRAWS = 200                 # count-matched random draws
SEED = 7
HAC_LAGS = 20                    # Newey-West lags on the one-value-per-date series
PASS_1 = {                       # fixed before the run
    "mean_rel_net_20": 0.015,    # mean relative return over H_MARK sessions, net of 2 x COST_BPS, at least this
    "caught_up": 0.55,           # share of events whose net relative return at H_MARK is positive, at least this
    "null_pct": 0.95,            # the cell's mean above this percentile of the count-matched null
    "min_events": 150,           # fewer events than this = inconclusive, not a pass
    "both_beats_turned_up": True # the "both" cell must beat "turned-up alone" on the mean, else it is momentum not cheapness
}

# ------------------------------------------------- Phase 2: the book --
N_POS = 30                       # positions
SECTOR_CAP = 4                   # at most this many per group
BENCH_REBAL = "ME"               # the equal-weight universe benchmark rebalances at month-end
PASS_0 = {                       # Phase 0 sanity marks (a WARN, not a stop - except the CAGR gap, which flags a broken price panel)
    "priced_share": 0.97, "sector_share": 0.95, "min_eligible": 250, "max_jumps_per_member_year": 0.05,
    "min_corr_index": 0.85, "max_cagr_gap": 0.10,
}
PASS_2 = {
    "breakeven_bps": 80,         # the one-way cost at which the book's net CAGR equals the benchmark's, at least this
    "null_pct": 0.95,            # gross Sharpe above this percentile of the count-matched random book
    # and: net Sharpe and worst fall better than the sector-matched benchmark; net Sharpe above the benchmark's on both halves
}

# ---------------------------------------------- Phase 3: robustness --
GRID = {                         # one dimension at a time from the base case (GRID_FULL = True runs the cross product; hours)
    "formation": (252, 504, 756),
    "z_entry": (-1.5, -2.0, -2.5),
    "z_exit": (-0.5, 0.0),
    "st_bar": ("daily", "weekly"),
    "time_stop": (40, 60, 90),
    "n_pos": (20, 30, 40),
}
GRID_FULL = False
PLACEBO = True                   # the mirror cell: rich (z >= +2) and turned up - if it also beats the benchmark the edge is the trend, not the cheapness
BLOCK_BOOT = 1000                # block-bootstrap draws of the book's active daily returns
BLOCK_LEN = 20                   # sessions per block
STOCK_PAIRS = True               # the stock-to-stock variant (Engle-Granger within group, SP_TOP pairs per group, yearly refit)
SP_TOP = 20
SP_REFIT = "YE"

# --------------------------------------------------------------- reporting --
REPORT_HTML = OUT / "report.html"
ANNUALISE = 252
