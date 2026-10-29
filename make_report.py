"""
Build outputs/report.html from outputs/tables/ - the clean report.

Shape (the sibling studies' convention): §0 the answer in one screen; then one chapter per phase, each in the
form  the question -> the answer in one line -> the chart (and where to look) -> the table -> the key numbers
-> what it means for the book -> the detail folded away.  Every number in the prose is computed from a table.
Jargon is dotted-underlined with its meaning on hover (TERMS). No person's name anywhere.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import figs

T = C.TABLES
PLOTLY_JS = Path(__import__("plotly").__file__).parent / "package_data" / "plotly.min.js"

TERMS = {
    "partner basket": "the stock's group (GICS industry group), equal-weighted, every eligible member except the stock itself; rebuilt daily from the members decided the session before",
    "level test": "Engle-Granger: an ADF test on the residual of log price on log basket, with MacKinnon's two-variable p-values; p below 0.05, half-life 5–30 sessions, beta above 0",
    "z": "the residual of the stock's log price against its partner basket, divided by the residual's standard deviation over the formation window; −2 = two standard deviations below the relationship",
    "half-life": "sessions for half of a deviation to fade, from an AR(1) on the residual",
    "cheap": "z at or below −2 under a fit that passed the level test",
    "turned up": "the stock's own daily supertrend (5-bar linear-weighted ATR × 2) is in BUY, or flips to BUY within 10 sessions of the cheap crossing while z is still at or below −1",
    "relative return": "the stock's log return minus its partner basket's over the same sessions, starting the session after execution",
    "net": "after 40 basis points a side on the stock (80 round trip)",
    "caught up": "the net relative return at 20 sessions is positive",
    "count-matched null": "the same number of events as the cell, drawn at random: the same stocks on random dates, or random same-group peers on the same dates; 200 draws of the mean",
    "sector-matched benchmark": "each day, the book's own group weights of the session before applied to the group returns, the remainder on the equal-weight universe: what a blind pick inside the same groups would earn",
    "equal-weight universe": "every eligible name at equal weight, rebalanced monthly, paying the same costs on its turnover",
    "Sharpe": "annualised mean daily return divided by annualised standard deviation, risk-free rate taken as zero",
    "worst fall": "the largest peak-to-trough decline of the net NAV",
    "break-even cost": "the one-way cost per side at which the book's net CAGR equals its sector-matched benchmark's (0 = the picks trail their groups even before costs)",
    "placebo": "the mirror rule: rich (z ≥ +2) and turned up, run as a long book; if it also wins, the trend carries the result, not the cheapness",
    "block bootstrap": "the book's daily active returns resampled in blocks of 20 sessions, 1000 times: an interval around the active Sharpe",
}


# ------------------------------------------------------------- helpers --
def _csv(name, **kw):
    p = T / f"{name}.csv"
    return pd.read_csv(p, **kw) if p.exists() else None


def _img(name, **kw):
    try:
        return f"<div class='fig'>{figs.html(name, **kw)}</div>"
    except Exception as e:  # noqa: BLE001 - a missing table must show in the report, not kill it
        return f"<p class='missing'>chart {name} not available: {e}</p>"


def _tbl(df, fmt="{:.3f}", index=True, max_rows=60, pct_cols=(), int_cols=()):
    if df is None or not len(df):
        return "<p class='missing'>(no table)</p>"
    d = df.head(max_rows).copy()
    for c in d.columns:
        if c in pct_cols:
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v * 100:+.1f}%")
        elif c in int_cols:
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{int(v)}")
        elif pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else fmt.format(v))
    return d.to_html(index=index, border=0, classes="t", escape=True, na_rep="")


def _pct(v, d=1):
    return "n/a" if v is None or pd.isna(v) else f"{v * 100:+.{d}f}%"


def _p(v, d=0):
    return "n/a" if v is None or pd.isna(v) else f"{v * 100:.{d}f}%"


def _n(v, d=2):
    return "n/a" if v is None or pd.isna(v) else f"{v:.{d}f}"


def _d(term):
    return f"<abbr title=\"{TERMS.get(term, term)}\">{term}</abbr>"


def _q(txt):
    return f"<p class='q'><b>The question.</b> {txt}</p>"


def _ans(txt):
    return f"<p class='ans'>{txt}</p>"


def _walk(txt):
    return f"<p class='walk'><b>How we know — look at the chart.</b> {txt}</p>"


def _book(txt):
    return f"<div class='book'><b>What it means for the book.</b> {txt}</div>"


def _tiles(items):
    return "<div class='tiles'>" + "".join(f"<div class='tile'><div class='v'>{v}</div><div class='k'>{k}</div></div>" for k, v in items) + "</div>"


def _more(title, inner):
    return f"<details><summary>{title}</summary>{inner}</details>"


def _verdict_tbl(v):
    if v is None:
        return "<p class='missing'>(not run)</p>"
    d = v.copy()
    d["value"] = d["value"].map(lambda x: "" if pd.isna(x) else (f"{x:.4f}" if abs(x) < 10 else f"{x:.0f}"))
    d["passed"] = d["passed"].map(lambda x: "✔" if x else "✘")
    return d.to_html(index=False, border=0, classes="t verdict", escape=True)


# --------------------------------------------------------------- build --
def build() -> str:
    meta = json.loads((T / "meta.json").read_text())
    syn = bool(meta.get("synthetic"))
    v0, v1, v2 = _csv("0_verdict"), _csv("1_verdict"), _csv("2_verdict")
    us = _csv("0_universe_size", index_col=0, parse_dates=True); fs = _csv("1_fit_summary", index_col=0, parse_dates=True)
    cs = _csv("1_cell_stats"); hv = _csv("1_halves"); ns = _csv("1_null_summary"); xs = _csv("1_crossing_state"); en = _csv("1_effective_n")
    zd = _csv("1_z_distribution"); by = _csv("1_by_year"); bs = _csv("1_by_sector"); exc = _csv("1_excluded")
    M2 = _csv("2_metrics", index_col=0); h2 = _csv("2_halves", index_col=0); y2 = _csv("2_by_year", index_col=0); ts = _csv("2_trade_summary", index_col=0)
    cg = _csv("2_cost_grid"); rs = _csv("2_random_summary", index_col=0); cells = _csv("2_cells", index_col=0); sw = _csv("2_sector_weights", index_col=0)
    gr = _csv("3_grid"); pst = _csv("3_placebo_stats"); pm = _csv("3_placebo_metrics"); bb = _csv("3_bootstrap"); sps = _csv("3_sp_stats"); spf = _csv("3_sp_fits")
    grp = _csv("0_groups", index_col=0); dl = _csv("0_delistings"); jumps = _csv("0_suspect_jumps"); src = _csv("0_sector_source", index_col=0)

    def cell(c, h=C.H_MARK):
        if cs is None:
            return {}
        r = cs[(cs["cell"] == c) & (cs["horizon"] == h)]
        return r.iloc[0].to_dict() if len(r) else {}

    both, cheap, turned = cell("both"), cell("cheap alone"), cell("turned-up alone")
    p1 = meta.get("phase1", "not run"); p2 = meta.get("phase2", "not run")
    first, last = meta.get("first_session"), meta.get("last_session")
    bk = M2.loc["book vs sector-matched"] if M2 is not None and "book vs sector-matched" in M2.index else None
    be = meta.get("breakeven_bps")

    # ---- §0
    if p1 != "PASS" and meta.get("force"):
        p2 = f"{p2} (forced — for inspection only)"
    if p1 == "PASS" and p2 == "PASS":
        headline = "Yes, on this history: the cheap-and-turned-up stock beats its group after costs, and a 30-name book of them beats the sector-matched benchmark on both Sharpe and worst fall."
    elif p1 == "PASS":
        headline = "Half: the mechanism is there event by event, but a book built on it does not clear its marks."
    elif p1 == "INCONCLUSIVE":
        failed = [] if v1 is None else [c for c, ok in zip(v1["criterion"], v1["passed"]) if not ok and not c.startswith("events")]
        headline = (f"Inconclusive by the pre-registered rule: only {meta.get('events_both', 0)} 'cheap + turned up' events, fewer than the {C.PASS_1['min_events']} needed for a verdict."
                    + (f" On the events there are, {len(failed)} of the other five marks are missed." if failed else " On the events there are, every other mark is met."))
    elif p1 == "FAIL":
        headline = "No: the cheap-and-turned-up stock does not beat its group by enough, often enough, after costs - the same shape as Studies I, J and K."
    else:
        headline = "Not run."
    s0 = f"""
    <h2 id='s0'>§0 The answer</h2>
    <p class='lead'>{headline}</p>
    {_tiles([("Phase 1 — the mechanism", p1), ("Phase 2 — the book", p2), ("'both' events", f"{meta.get('events_both', 0):,}"),
             (f"mean net relative return, {C.H_MARK} sessions", _pct(both.get('mean_rel_net'))), ("caught up", _p(both.get('hit_net'))),
             ("book net CAGR vs matched", f"{_pct(bk['cagr'])} vs {_pct(bk['bench_cagr'])}" if bk is not None else "—"),
             ("break-even cost", f"{be:.0f} bps" if be is not None and np.isfinite(be) else "—")])}
    <p>The sentence under test: <i>within a sector, a stock that has fallen well below its long-run relationship with its peers — and whose own trend has turned up — outperforms its sector over the next one to three months, net of costs, often enough to run as a long-only selection rule.</i>
    Everything below answers it with a yes or a no, phase by phase, against marks fixed before the run (<code>config.PASS_1</code>, <code>config.PASS_2</code>).</p>
    <table class='t'><tr><th>Phase</th><th>Verdict</th><th>In one line</th></tr>
    <tr><td>0 — data</td><td>{meta.get('phase0', '—')}</td><td>{meta.get('names', 0):,} names ({meta.get('names_priced', 0):,} with prices), {meta.get('sessions', 0):,} sessions {first} → {last}; point-in-time top-500 by market cap, keyed on quote_id, delisted names kept.</td></tr>
    <tr><td>1 — mechanism</td><td>{p1}</td><td>{both.get('events', 0):.0f} cheap-and-turned-up events; {_pct(both.get('mean_rel_net'))} net against the {_d('partner basket')} over {C.H_MARK} sessions, {_p(both.get('hit_net'))} {_d('caught up')}; cheap alone {_pct(cheap.get('mean_rel_net'))}, turned-up alone {_pct(turned.get('mean_rel_net'))}.</td></tr>
    <tr><td>2 — book</td><td>{p2}</td><td>{('net CAGR ' + _pct(bk['cagr']) + ' vs ' + _pct(bk['bench_cagr']) + ' sector-matched; Sharpe ' + _n(bk['sharpe']) + ' vs ' + _n(bk['bench_sharpe']) + '; worst fall ' + _pct(bk['worst_fall']) + ' vs ' + _pct(bk['bench_worst_fall'])) if bk is not None else 'not built (Phase 1 did not pass)'}.</td></tr>
    <tr><td>3 — robustness</td><td>{meta.get('phase3', '—')}</td><td>{(str(int(gr['beats_matched_sharpe'].sum())) + ' of ' + str(len(gr)) + ' grid settings beat the matched benchmark on net Sharpe') if gr is not None else 'not run'}{('; placebo active Sharpe ' + _n(pm['active_sharpe'].iloc[0])) if pm is not None and len(pm) else ''}.</td></tr></table>
    """

    # ---- §1 data
    med_el = us["eligible"].median() if us is not None else np.nan
    s1 = f"""
    <h2 id='s1'>§1 The data, and what it cannot do</h2>
    {_q("Can the price history be trusted, and is the universe free of survivorship?")}
    {_ans(f"Verdict {meta.get('phase0', '—')}. {meta.get('names_priced', 0):,} of the {meta.get('names', 0):,} names that were ever in the universe have prices; a typical month has {med_el:.0f} eligible names in {us['groups_in_use'].median() if us is not None else 0:.0f} groups.")}
    {_img('0_universe')}
    {_walk("the top line is the point-in-time top-500 by market cap; each line below it is a filter (liquidity floor, a sector on record, at least five eligible peers). The gap between the top and bottom lines is what the filters cost; the dotted line (right axis) is how many groups are in play.")}
    {_verdict_tbl(v0)}
    {_book("The universe is the largest 500 ordinary shares by market cap at each month-end trading day (the wm-utils rolling market-cap universe), applied from the next session — not the Nifty 500 itself (no historical membership exists in the database), but built the same way and free of survivorship: every name is keyed on its permanent quote_id, so renames and RIC changes do not break a series, and names that later delisted are kept until their last print. Prices come from one source (ohlcv_eikon), split- and bonus-adjusted; dividends are not in them, so returns are price returns on both the stock and its basket. The sector is GICS industry group: monthly from 2017, the earliest monthly record before that.")}
    {_more("Groups, delistings, suspect jumps, sector sources", _tbl(grp, pct_cols=('share_sessions_ge_min_peers',)) + "<p>Names whose last print is before the last session (kept until that print):</p>" + _tbl(dl, max_rows=40)
           + "<p>Sessions with |log return| above 30% — the corporate-action artefact screen; events within ±5 sessions of one are excluded:</p>" + _tbl(jumps, max_rows=40)
           + "<p>Where the sector came from, by year (share of universe rows):</p>" + _tbl(src, fmt='{:.2f}'))}
    """

    # ---- §2 signal
    s2 = f"""
    <h2 id='s2'>§2 The signal: a level relationship, not a correlation</h2>
    {_q("How many stocks have a long-run level against their group at all — the cointegration lesson of Study I applied stock by stock?")}
    {_ans(f"{_p(fs['pass_share'].mean()) if fs is not None else 'n/a'} of fitted stocks pass the {_d('level test')} on a typical month-end, against {C.ADF_P_MAX:.0%} by chance; the passers' median {_d('half-life')} is {fs['median_halflife_passing'].median() if fs is not None else np.nan:.0f} sessions.")}
    {_img('1_fit_pass')}
    {_walk("the bars are the share of fitted stocks that pass each month; the dashed line is what chance alone gives (the test's own size). A share well above the line means the sector relationship is real for a good part of the universe; a share near it means the passers are noise.")}
    {_tiles([("refits", f"{meta.get('refits', 0)}"), ("mean pass share", _p(fs['pass_share'].mean()) if fs is not None else '—'),
             ("median half-life (passers)", f"{fs['median_halflife_passing'].median():.0f} sessions" if fs is not None else '—'),
             ("share z ≤ −2 (passing fits)", _p(zd.set_index('stat').loc['share z <= -2', 'value'], 1) if zd is not None else '—')])}
    {_book(f"A stock with no level against its group has no signal that month, whatever its z says. The {_d('z')} is measured only where the fit passed; the share of such sessions at or below −2 is the raw supply of 'cheap' readings.")}
    {_more("Fit summary by month", _tbl(fs.reset_index(), pct_cols=('pass_share', 'chance_share', 'p_lt_005_share', 'share_hl_too_long', 'share_hl_too_short'), max_rows=200))}
    """

    # ---- §3 Phase 1
    halves_txt = ""
    if hv is not None and len(hv):
        hb = hv[hv["cell"] == "both"]
        halves_txt = "; ".join(f"{r['half']}: {_pct(r['mean_rel_net'])} on {int(r['events'])} events" for _, r in hb.iterrows())
    null_txt = ""
    if ns is not None and len(ns):
        null_txt = "; ".join(f"{r['null']}: 95th percentile {_pct(r['p95'])}, actual at the {_p(r['actual_percentile'])} point" for _, r in ns.iterrows())
    s3 = f"""
    <h2 id='s3'>§3 Phase 1 — the mechanism, event by event</h2>
    {_q("After a stock has fallen to z ≤ −2 against its partner basket, does it beat the basket over the next 5 / 20 / 60 sessions — and does it matter whether its own trend has turned up?")}
    {_ans(f"<b>{p1}.</b> {both.get('events', 0):.0f} {_d('cheap')} + {_d('turned up')} events: {_d('relative return')} {_pct(both.get('mean_rel_net'))} {_d('net')} over {C.H_MARK} sessions, {_p(both.get('hit_net'))} caught up, Newey-West t {_n(both.get('hac_t'), 1)}. Cheap alone: {_pct(cheap.get('mean_rel_net'))} ({cheap.get('events', 0):.0f} events). Turned-up alone: {_pct(turned.get('mean_rel_net'))} ({turned.get('events', 0):.0f} events).")}
    {_img('1_paths')}
    {_walk(f"each line is the average cumulative return of the stock minus its basket after the execution session, one line per cell. The blue line is the cell the study rests on; it must clear the dotted cost line and sit above the grey triangle (the stricter {_d('count-matched null')}'s 95th percentile) at {C.H_MARK} sessions. The red line is the falling knife: cheap but not turned. The orange line is the trend alone: if blue and orange sit together, cheapness adds nothing.")}
    {_verdict_tbl(v1)}
    {_tiles([("events (both)", f"{both.get('events', 0):.0f}"), ("distinct entry dates", f"{both.get('dates', 0):.0f}"), (f"mean net, {C.H_MARK}s", _pct(both.get('mean_rel_net'))),
             ("median net", _pct(both.get('median_rel_net'))), ("caught up", _p(both.get('hit_net'))), ("z came back (for reference)", _p(both.get('z_back_share')))])}
    <p><b>Halves.</b> {halves_txt or 'n/a'}. <b>Null.</b> {null_txt or 'n/a'}. <b>Crossings in SELL.</b> {_p(xs['share_in_SELL_at_crossing'].iloc[0]) if xs is not None else 'n/a'} of cheap crossings found the stock's supertrend already in SELL (Study K's lesson); {_p(xs['share_qualifying_both'].iloc[0]) if xs is not None else 'n/a'} of crossings qualified as 'both', typically {xs['median_days_to_turn_both'].iloc[0] if xs is not None else 0:.0f} sessions after the crossing.</p>
    {_book("Only the 'both' cell is traded. If it passes while 'cheap alone' does not, mean reversion exists but needs the trend confirmation; if 'turned-up alone' does as well, the rule is a momentum rule in disguise and the sixth criterion says so.")}
    {_img('1_null_hist')}
    {_img('1_by_year')}
    {_more("Every cell at every horizon", _tbl(cs, pct_cols=('mean_rel_gross', 'mean_rel_net', 'median_rel_net', 'hit_net', 'sd', 'z_back_share'), int_cols=('events', 'dates', 'horizon')))}
    {_more("By year, by group, halves, effective n, exclusions", _tbl(by, pct_cols=('mean_rel_gross', 'mean_rel_net', 'median_rel_net', 'hit_net', 'sd', 'z_back_share'), int_cols=('events', 'dates', 'horizon', 'year'), max_rows=100)
           + _tbl(bs, pct_cols=('mean_rel_gross', 'mean_rel_net', 'median_rel_net', 'hit_net', 'sd', 'z_back_share'), int_cols=('events', 'dates', 'horizon'), max_rows=100)
           + _tbl(hv, pct_cols=('mean_rel_gross', 'mean_rel_net', 'median_rel_net', 'hit_net', 'sd', 'z_back_share'), int_cols=('events', 'dates', 'horizon'))
           + _tbl(en) + f"<p>{len(exc) if exc is not None else 0} events excluded for a suspect jump within ±{C.CA_EXCLUDE} sessions.</p>")}
    """

    # ---- §4 Phase 2
    if bk is not None:
        s4_ans = f"<b>{p2}.</b> Net CAGR {_pct(bk['cagr'])} against {_pct(bk['bench_cagr'])} for the {_d('sector-matched benchmark')}; {_d('Sharpe')} {_n(bk['sharpe'])} vs {_n(bk['bench_sharpe'])}; {_d('worst fall')} {_pct(bk['worst_fall'])} vs {_pct(bk['bench_worst_fall'])}; {meta.get('trades', 0):,} trades; {_d('break-even cost')} {be:.0f} bps."
        rtxt = ""
        if rs is not None and "gross_sharpe" in rs.index:
            rtxt = f"The {_d('count-matched null')} as books: median gross Sharpe {_n(rs.loc['gross_sharpe', 'p50'])}, 95th percentile {_n(rs.loc['gross_sharpe', 'p95'])}."
        forced = (f"<div class='syn' style='background:#ff7f0e'>Built with <code>--force</code>: Phase 1 did not pass ({p1}), so by the plan's stop rule this book "
                  f"would not exist. It is shown for inspection only; its verdict does not override Phase 1.</div>") if meta.get("force") and p1 != "PASS" else ""
        s4 = f"""
        <h2 id='s4'>§4 Phase 2 — the book</h2>
        {forced}
        {_q("Does holding the cheap-and-turned-up stocks instead of all of the group's stocks beat the sector-neutral book, after costs, on return and on the worst fall?")}
        {_ans(s4_ans)}
        {_img('2_nav')}
        {_walk("log scale, growth of 1. The thick blue line is the book net of costs; green is the sector-matched benchmark (the same group weights, blind picks inside them); grey the equal-weight universe. The book has to pull away from green, not just from grey: beating grey can be a sector bet, beating green is stock selection.")}
        {_verdict_tbl(v2)}
        {_tiles([("net CAGR", _pct(bk['cagr'])), ("matched benchmark", _pct(bk['bench_cagr'])), ("Sharpe", _n(bk['sharpe'])), ("matched Sharpe", _n(bk['bench_sharpe'])),
                 ("worst fall", _pct(bk['worst_fall'])), ("matched worst fall", _pct(bk['bench_worst_fall'])), ("trades", f"{meta.get('trades', 0):,}"),
                 ("turnover / year (one-way)", _n(bk.get('turnover_per_year'), 1)), ("break-even cost", f"{be:.0f} bps")])}
        <p>{rtxt}</p>
        {_book("The book is a selection rule inside a long-only, sector-capped portfolio: 30 slots, at most 4 per group, equal-weighted at entry, the empty slots holding the universe. Its edge, if any, is small per trade and collected across hundreds of trades; the break-even cost says how much of it survives execution.")}
        {_img('2_drawdown')}
        {_img('2_cost_grid')}
        {_img('2_random_hist')}
        {_img('2_cells')}
        {_more("Metrics, halves, years, trades, sector weights", _tbl(M2.reset_index(), pct_cols=('total', 'cagr', 'vol', 'worst_fall', 'hit', 'bench_cagr', 'bench_worst_fall', 'active_cagr', 'tracking_error', 'share_days_ahead'))
               + _tbl(h2.reset_index(), pct_cols=('total', 'cagr', 'vol', 'worst_fall', 'hit', 'bench_cagr', 'bench_worst_fall', 'active_cagr', 'tracking_error', 'share_days_ahead'))
               + _tbl(y2.reset_index(), pct_cols=('total', 'cagr', 'vol', 'worst_fall', 'hit', 'bench_cagr', 'bench_worst_fall', 'active_cagr', 'tracking_error', 'share_days_ahead'))
               + _tbl(ts.reset_index(), pct_cols=('mean_net', 'median_net', 'hit_net', 'mean_rel_net', 'hit_rel_net', 'worst', 'best', 'share'), int_cols=('trades',))
               + _tbl(cg, pct_cols=('book_cagr', 'bench_cagr', 'active_cagr', 'worst_fall')) + _tbl(cells.reset_index(), pct_cols=('cagr', 'worst_fall', 'bench_cagr', 'bench_worst_fall', 'active_cagr'), int_cols=('entries', 'trades'))
               + _tbl(sw.reset_index(), pct_cols=('mean_weight',)))}
        """
    else:
        s4 = f"<h2 id='s4'>§4 Phase 2 — the book</h2><p class='missing'>Not built: Phase 1 did not pass its marks ({p1}). The event table (§3) is the deliverable. <code>python run_all.py --force</code> builds the book regardless, for inspection.</p>"

    # ---- §5 Phase 3
    s5_parts = [f"<h2 id='s5'>§5 Phase 3 — what could still fool us</h2>"]
    if gr is not None:
        s5_parts.append(_q("Does the result survive the choices that were not pre-registered — formation window, entry and exit lines, the supertrend bar, the hold cap, the position count?"))
        s5_parts.append(_ans(f"{int(gr['beats_matched_sharpe'].sum())} of {len(gr)} settings beat the matched benchmark on net Sharpe; {int(gr['beats_matched_fall'].sum())} on worst fall. The grid is reported whole; nothing was picked from it."))
        s5_parts.append(_img('3_grid'))
        s5_parts.append(_more("The grid", _tbl(gr, pct_cols=('p1_mean_rel_net_20', 'p1_hit_net_20', 'net_cagr', 'worst_fall', 'matched_worst_fall', 'active_cagr_vs_ew'), int_cols=('events_both', 'trades', 'formation', 'time_stop', 'n_pos'))))
    if pst is not None and len(pst):
        pr = pst[(pst["cell"] == "rich + turned up") & (pst["horizon"] == C.H_MARK)]
        s5_parts.append(f"<p><b>{_d('placebo')}.</b> Rich + turned up: {_pct(pr['mean_rel_net'].iloc[0]) if len(pr) else 'n/a'} net over {C.H_MARK} sessions on {int(pr['events'].iloc[0]) if len(pr) else 0} events"
                        + (f"; as a book, active Sharpe {_n(pm['active_sharpe'].iloc[0])} against its matched benchmark" if pm is not None and len(pm) else "") + ". If this is as good as the real rule, the trend is doing the work.</p>")
    if bb is not None and len(bb):
        r = bb.iloc[0]
        s5_parts.append(f"<p><b>{_d('block bootstrap')}.</b> Active Sharpe {_n(r['actual'])}; 5th–95th percentile {_n(r['p5'])} to {_n(r['p95'])}; {_p(r['share_le_0'])} of draws at or below zero.</p>")
    if sps is not None and len(sps):
        sb = sps[(sps["cell"] == "both") & (sps["horizon"] == C.H_MARK)]
        s5_parts.append(f"<p><b>Stock-to-stock pairs.</b> Engle-Granger within each group at every year-end, the {C.SP_TOP} best-fitting pairs per group kept for the year ({len(spf) if spf is not None else 0:,} pair-years); the cheap leg entered on the same rule and measured against the other leg: {_pct(sb['mean_rel_net'].iloc[0]) if len(sb) else 'n/a'} net over {C.H_MARK} sessions on {int(sb['events'].iloc[0]) if len(sb) else 0} events, {_p(sb['hit_net'].iloc[0]) if len(sb) else 'n/a'} caught up.</p>")
        s5_parts.append(_more("Stock pairs: every cell", _tbl(sps, pct_cols=('mean_rel_gross', 'mean_rel_net', 'median_rel_net', 'hit_net', 'sd', 'z_back_share'), int_cols=('events', 'dates', 'horizon'))))
    if len(s5_parts) == 1:
        s5_parts.append("<p class='missing'>Not run.</p>")
    s5 = "\n".join(s5_parts)

    # ---- §6 traps
    s6 = """
    <h2 id='s6'>§6 The traps, and the control for each</h2>
    <table class='t'><tr><th>Trap</th><th>Control in this study</th></tr>
    <tr><td>Survivorship (only today's names)</td><td>Point-in-time top-500 by market cap; every name keyed on its permanent quote_id; delisted names kept until their last print (ohlcv_eikon), from 2014-07 when that table starts.</td></tr>
    <tr><td>A corporate action read as "cheap"</td><td>Adjusted prices; events within ±5 sessions of a |30%| one-session move excluded and listed.</td></tr>
    <tr><td>The value trap</td><td>The turn-up requirement; the −3.5 structural exit; the "cheap alone" cell reported beside "both" so the size of the trap is visible.</td></tr>
    <tr><td>The z illusion</td><td>Every measure is the stock's return against its basket; "z came back" is shown for reference only.</td></tr>
    <tr><td>Look-ahead in the split</td><td>Returns are counted from the session after the entry's execution, never from the crossing; the fit in force is the one dated strictly before the session; snapshots apply from the next session.</td></tr>
    <tr><td>Overlapping events</td><td>One crossing per run; Newey-West t on one value per date; distinct dates and non-overlapping counts reported.</td></tr>
    <tr><td>Multiple testing</td><td>One base case; marks fixed before the run; the grid reported whole; no group dropped after the fact.</td></tr>
    <tr><td>A sector bet dressed as selection</td><td>The partner is the group basket, so a group-wide fall is not a signal; the book is capped per group and judged against a benchmark holding its own group weights.</td></tr>
    <tr><td>Cash flattering the drawdown</td><td>Empty slots hold the equal-weight universe, not cash.</td></tr>
    <tr><td>Momentum in disguise</td><td>The "turned-up alone" cell and the placebo (rich + turned up).</td></tr>
    <tr><td>Liquidity at the small end</td><td>The liquidity floor; 40 bps a side with the cost grid to 120.</td></tr></table>
    """

    glossary = "<h2 id='g'>Glossary</h2><dl>" + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in TERMS.items()) + "</dl>"
    banner = "<div class='syn'>SYNTHETIC DEMO DATA — every number on this page comes from make_demo_data.py, not from the desk's prices. Not a result.</div>" if syn else ""
    css = """
    body{font-family:Georgia,serif;max-width:1180px;margin:30px auto;padding:0 20px;color:#222;line-height:1.5}
    h1{font-size:26px} h2{margin-top:44px;border-bottom:1px solid #ddd;padding-bottom:4px} .lead{font-size:19px;font-weight:bold}
    .q{background:#f5f7fa;padding:8px 12px;border-left:4px solid #1f77b4} .ans{font-size:17px} .walk{color:#444} .book{background:#fff8e6;padding:10px 12px;border-left:4px solid #ff7f0e;margin:12px 0}
    .tiles{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0} .tile{background:#f0f0f0;padding:10px 14px;border-radius:6px;min-width:120px} .tile .v{font-size:20px;font-weight:bold} .tile .k{font-size:12px;color:#555}
    table.t{border-collapse:collapse;font-size:12.5px;margin:8px 0} table.t th,table.t td{padding:4px 8px;border-bottom:1px solid #e5e5e5;text-align:right} table.t th:first-child,table.t td:first-child{text-align:left}
    table.verdict td:last-child{font-size:16px} details{margin:12px 0} summary{cursor:pointer;color:#1f77b4} .missing{color:#a00} abbr{text-decoration:underline dotted;cursor:help}
    .syn{background:#d62728;color:#fff;padding:10px;font-weight:bold;text-align:center} .fig{margin:14px 0} nav a{margin-right:14px} dl dt{font-weight:bold;margin-top:8px}
    """
    js = PLOTLY_JS.read_text(encoding="utf-8") if PLOTLY_JS.exists() else ""
    html = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='data' content='{'synthetic' if syn else 'real'}'><title>{'[SYNTHETIC] ' if syn else ''}Study L — stock pairs, long-only</title><style>{css}</style><script>{js}</script></head><body>
    {banner}
    <h1>Study L — stock pairs, long-only: cheap against the sector, and turned up</h1>
    <p>Built {meta.get('run_at', '')} on sessions {first} → {last}{' (fast run: fewer null draws, no grid)' if meta.get('fast') else ''}. Parameters and pass marks in <code>config.py</code>; definitions in <code>LOGIC.md</code>; every table in <code>outputs/tables/</code>.</p>
    <nav><a href='#s0'>§0 answer</a><a href='#s1'>§1 data</a><a href='#s2'>§2 signal</a><a href='#s3'>§3 mechanism</a><a href='#s4'>§4 book</a><a href='#s5'>§5 robustness</a><a href='#s6'>§6 traps</a><a href='#g'>glossary</a></nav>
    {s0}{s1}{s2}{s3}{s4}{s5}{s6}{glossary}
    </body></html>"""
    return html


def main() -> int:
    C.OUT.mkdir(exist_ok=True)
    html = build()
    C.REPORT_HTML.write_text(html, encoding="utf-8")
    print(f"wrote {C.rel(C.REPORT_HTML)} ({len(html) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
