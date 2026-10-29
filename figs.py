"""
The report's charts (Plotly, embedded in report.html; no internet needed).

Every chart's bold title is its finding with the numbers in it (computed here from the same table the
report reads); the grey line under it is the method. figs.FIGS maps a name to a builder; html(name)
renders it as a div. The library itself is embedded once by make_report.py.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

import config as C

T = C.TABLES
COL = {"both": "#1f77b4", "cheap alone": "#d62728", "turned-up alone": "#ff7f0e", "neither": "#7f7f7f", "random": "#7f7f7f",
       "book": "#1f77b4", "ew": "#7f7f7f", "matched": "#2ca02c", "index": "#9467bd", "rich + turned up": "#8c564b"}


def _csv(name, **kw):
    return pd.read_csv(T / f"{name}.csv", **kw)


def _layout(fig: go.Figure, title: str, subtitle: str = "", height: int = 460, **kw) -> go.Figure:
    fig.update_layout(title={"text": f"<b>{title}</b>" + (f"<br><span style='font-size:12px;color:#666'>{subtitle}</span>" if subtitle else ""), "x": 0.01, "xanchor": "left"},
                      height=height, margin={"l": 50, "r": 20, "t": 90 if subtitle else 60, "b": 50}, template="plotly_white",
                      legend={"orientation": "h", "y": -0.15}, hovermode="x unified", **kw)
    return fig


def _pct(v, d=1):
    return "n/a" if pd.isna(v) else f"{v * 100:+.{d}f}%"


# ---------------------------------------------------------------- charts --
def universe_size():
    df = _csv("0_universe_size", index_col=0, parse_dates=True)
    fig = go.Figure()
    for c, lab in (("members", "in the top-500 by market cap"), ("liquid_members", "... and liquid"), ("with_sector", "... with a sector"), ("eligible", "eligible (group has ≥ 5 peers)")):
        fig.add_trace(go.Scatter(x=df.index, y=df[c], name=lab, mode="lines"))
    fig.add_trace(go.Scatter(x=df.index, y=df["groups_in_use"], name="groups in use", mode="lines", yaxis="y2", line={"dash": "dot"}))
    fig.update_layout(yaxis2={"overlaying": "y", "side": "right", "title": "groups"})
    return _layout(fig, f"{df['eligible'].iloc[-1]:.0f} names eligible at the end, {df['eligible'].median():.0f} on a typical month",
                   "month-end counts: top-500 by market cap, after the liquidity floor, with a sector, and with at least 5 eligible peers in the group")


def fit_pass():
    df = _csv("1_fit_summary", index_col=0, parse_dates=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["pass_share"], name="share passing the level test", marker_color=COL["both"]))
    fig.add_hline(y=C.ADF_P_MAX, line={"dash": "dash", "color": "#d62728"}, annotation_text=f"chance ({C.ADF_P_MAX:.0%})", annotation_position="top left")
    fig.add_trace(go.Scatter(x=df.index, y=df["median_halflife_passing"], name="median half-life of the passers (sessions)", yaxis="y2", mode="lines", line={"color": "#2ca02c"}))
    fig.update_layout(yaxis={"tickformat": ".0%"}, yaxis2={"overlaying": "y", "side": "right", "title": "sessions"})
    return _layout(fig, f"{df['pass_share'].mean():.0%} of the fitted stocks pass the level test on a typical month ({C.ADF_P_MAX:.0%} would pass by chance)",
                   f"each month-end: log price on log partner basket over {C.FORMATION} sessions; Engle-Granger p < {C.ADF_P_MAX}, half-life {C.HALFLIFE_MIN}–{C.HALFLIFE_MAX} sessions, beta > 0")


def paths():
    df = _csv("1_paths")
    ns = _csv("1_null_summary") if (T / "1_null_summary.csv").exists() else pd.DataFrame()
    fig = go.Figure()
    for cell, g in df.groupby("cell"):
        fig.add_trace(go.Scatter(x=g["k"], y=g["mean_rel"], name=f"{cell} (n={int(g['n'].iloc[0])})", mode="lines", line={"color": COL.get(cell, None), "width": 3 if cell == "both" else 1.5}))
    rt = 2 * C.COST_BPS / 1e4
    fig.add_hline(y=rt, line={"dash": "dot", "color": "#999"}, annotation_text=f"round-trip cost {2 * C.COST_BPS} bps", annotation_position="bottom right")
    if len(ns):
        p95 = ns["p95"].max() + rt
        fig.add_trace(go.Scatter(x=[C.H_MARK], y=[p95], mode="markers", marker={"symbol": "triangle-down", "size": 12, "color": "#7f7f7f"}, name="95th percentile of the null (gross)"))
    b = df[(df["cell"] == "both") & (df["k"] == C.H_MARK)]
    v = b["mean_rel"].iloc[0] if len(b) else np.nan
    fig.update_layout(yaxis={"tickformat": "+.1%", "title": "stock minus its partner basket, cumulative"}, xaxis={"title": "sessions after execution"})
    return _layout(fig, f"'Cheap + turned up': {_pct(v)} against the basket after {C.H_MARK} sessions, gross (cost line at {_pct(rt)})",
                   "average cumulative relative return after the execution session, per cell; the triangle is the stricter null's 95th percentile at the mark")


def null_hist():
    nd = _csv("1_null_draws"); ns = _csv("1_null_summary")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=nd["same_stock_random_date"], name="same stock, random date", opacity=0.6, marker_color="#7f7f7f"))
    fig.add_trace(go.Histogram(x=nd["same_date_random_peer"], name="same date, random peer", opacity=0.6, marker_color="#bcbd22"))
    act = ns["actual_mean_net"].iloc[0]
    fig.add_vline(x=act, line={"color": COL["both"], "width": 3}, annotation_text=f"actual {_pct(act)}", annotation_position="top")
    fig.update_layout(barmode="overlay", xaxis={"tickformat": "+.1%", "title": f"mean net relative return over {C.H_MARK} sessions"})
    pct = ns["actual_percentile"].min()
    return _layout(fig, f"The actual mean sits at the {pct:.0%} point of the stricter null ({len(nd)} draws each)",
                   "each draw: the same number of events, random dates for the same stocks / random same-group peers on the same dates")


def by_year():
    df = _csv("1_by_year")
    fig = go.Figure()
    for cell, g in df.groupby("cell"):
        fig.add_trace(go.Bar(x=g["year"], y=g["mean_rel_net"], name=cell, marker_color=COL.get(cell), customdata=g["events"], hovertemplate="%{y:+.1%} (n=%{customdata})"))
    fig.update_layout(barmode="group", yaxis={"tickformat": "+.1%"})
    b = df[df["cell"] == "both"]
    pos = int((b["mean_rel_net"] > 0).sum())
    return _layout(fig, f"'Cheap + turned up' is positive net in {pos} of {len(b)} years", f"mean net relative return over {C.H_MARK} sessions by entry year, per cell")


def nav():
    df = _csv("2_nav", index_col=0, parse_dates=True)
    fig = go.Figure()
    for c, lab in (("book_nav", "the book (net)"), ("matched_nav", "sector-matched benchmark"), ("ew_nav", "equal-weight universe (net)"), ("index_nav", C.INDEX_REF)):
        if c in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], name=lab, mode="lines", line={"color": COL[c.split("_")[0]], "width": 3 if c == "book_nav" else 1.5}))
    m = _csv("2_metrics", index_col=0)
    r = m.loc["book vs sector-matched"]
    fig.update_layout(yaxis={"type": "log", "title": "growth of 1 (log scale)"})
    return _layout(fig, f"Net CAGR {_pct(r['cagr'])} vs {_pct(r['bench_cagr'])} for the sector-matched benchmark; worst fall {_pct(r['worst_fall'])} vs {_pct(r['bench_worst_fall'])}",
                   f"{C.N_POS} positions, {C.SECTOR_CAP} per group, {C.COST_BPS} bps a side, one-session delay; empty slots hold the equal-weight universe")


def drawdown():
    df = _csv("2_nav", index_col=0, parse_dates=True)
    fig = go.Figure()
    for c, lab in (("book_nav", "the book"), ("matched_nav", "sector-matched benchmark"), ("ew_nav", "equal-weight universe")):
        if c in df.columns:
            dd = df[c] / df[c].cummax() - 1
            fig.add_trace(go.Scatter(x=df.index, y=dd, name=lab, mode="lines", line={"color": COL[c.split("_")[0]]}))
    fig.update_layout(yaxis={"tickformat": ".0%"})
    dd = df["book_nav"] / df["book_nav"].cummax() - 1
    return _layout(fig, f"The book's worst fall: {_pct(dd.min())} ({dd.idxmin().date()})", "drawdown from the running high, net of costs")


def cost_grid():
    df = _csv("2_cost_grid")
    meta = json.loads((T / "meta.json").read_text())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["cost_bps"], y=df["active_cagr"], mode="lines+markers", name="book CAGR minus sector-matched benchmark CAGR", line={"color": COL["book"]}))
    fig.add_hline(y=0, line={"color": "#999"})
    fig.add_vline(x=C.PASS_2["breakeven_bps"], line={"dash": "dash", "color": "#d62728"}, annotation_text=f"mark {C.PASS_2['breakeven_bps']} bps")
    be = meta.get("breakeven_bps", np.nan)
    fig.update_layout(yaxis={"tickformat": "+.1%"}, xaxis={"title": "one-way cost, bps"})
    title = (f"Break-even cost {be:.0f} bps a side" if np.isfinite(be) and be > 0 else
             "No edge to absorb any cost: the picks trail their groups even before costs" if np.isfinite(be) else "No break-even inside the cost grid")
    return _layout(fig, title, "the book re-run at each one-way cost; the benchmark holds the same idle sleeve and blind picks in the book's groups, cost-free")


def random_hist():
    per = _csv("2_random_books"); m = _csv("2_metrics", index_col=0)
    gs = json.loads((T / "meta.json").read_text())
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=per["gross_sharpe"], name="random count-matched books (gross Sharpe)", marker_color="#7f7f7f"))
    act = m.loc["book vs equal-weight universe", "gross_sharpe"] if "gross_sharpe" in m.columns else np.nan
    if pd.notna(act):
        fig.add_vline(x=act, line={"color": COL["book"], "width": 3}, annotation_text=f"the book {act:.2f}")
    pct = float((per["gross_sharpe"] < act).mean()) if pd.notna(act) else np.nan
    return _layout(fig, f"The book's gross Sharpe sits at the {pct:.0%} point of {len(per)} random books" if pd.notna(pct) else "random books",
                   "each draw replaces every trade with a random eligible peer of the same group on the same date, held as long")


def cells():
    df = _csv("2_cells", index_col=0)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["sharpe"], name="net Sharpe", marker_color=[COL.get(c, "#7f7f7f") for c in df.index]))
    if "bench_sharpe" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["bench_sharpe"], name="its sector-matched benchmark", mode="markers", marker={"symbol": "line-ew-open", "size": 30, "color": "#000"}))
    return _layout(fig, "The 2 × 2 as books: which part of the rule carries it", "net Sharpe of a book built on each cell's entries; the bar must clear its own benchmark's mark")


def grid_plot():
    df = _csv("3_grid")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["variant"], y=df["net_sharpe"], name="book net Sharpe", marker_color=COL["book"]))
    fig.add_trace(go.Scatter(x=df["variant"], y=df["matched_sharpe"], name="sector-matched benchmark", mode="markers", marker={"symbol": "line-ew-open", "size": 24, "color": "#000"}))
    n = int(df["beats_matched_sharpe"].sum())
    return _layout(fig, f"{n} of {len(df)} grid settings beat the sector-matched benchmark on net Sharpe", "one dimension at a time from the base case; reported whole", height=520)


FIGS = {"0_universe": universe_size, "1_fit_pass": fit_pass, "1_paths": paths, "1_null_hist": null_hist, "1_by_year": by_year,
        "2_nav": nav, "2_drawdown": drawdown, "2_cost_grid": cost_grid, "2_random_hist": random_hist, "2_cells": cells, "3_grid": grid_plot}


def html(name: str, **kw) -> str:
    fig = FIGS[name](**kw)
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id=f"fig_{name}")
