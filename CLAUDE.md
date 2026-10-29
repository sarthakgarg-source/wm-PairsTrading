# wm-PairsTrading — conventions for Claude Code
Study L. Read README.md, GLOSSARY.md, LOGIC.md, UNDERSTANDING.md. `python run_all.py` → `python make_report.py` → `python checks.py` after any edit to a module; checks must say "0 failed".
- Parameters and pass marks in config.py only. The marks were fixed before the first run and are not tuned; a change needs a dated reason in UNDERSTANDING.md first, and the review's agreement.
- Daily sessions. Nothing after t is ever used: snapshots apply from the next session; the fit in force is the one dated strictly before; a signal at close t trades at close t + DELAY; forward returns start the session after execution.
- The measure is always the stock's return against its partner basket (or the book against its benchmark), never the z. "z came back" is a reference column only.
- The key is `qid` (the Refinitiv `quote_id`), never the RIC or ticker — both change over a security's life. Prices come from `ohlcv_eikon` only; never splice in the legacy table's prices. Delisted names stay until their last print.
- The database is read-only (wm-utils docs/tables): `db.py` sets `default_transaction_read_only`; never write to it, never run a wm-utils job that writes. Follow the wm-utils table catalog for which table to use.
- make_demo_data.py refuses to overwrite real data in data/; run the demo with PAIRS_DATA_DIR / PAIRS_OUT_DIR pointing elsewhere.
- Every number in report prose is computed from a table in outputs/tables/. Chart titles are findings with the numbers in them; the method goes in the grey subtitle. Jargon goes through `_d(term)` (hover definition from TERMS) and is said in plain words first.
- A new table or column goes into GLOSSARY.md (and LOGIC.md if it is a definition) in the same commit. A new chart goes into figs.FIGS and is embedded with `_img(name)`.
- Phase order and stop rules are the plan's: Phase 2 runs only if Phase 1 passes (or `--force`, which the report must say).
- Secrets live in `.env` only (git-ignored). Synthetic data (make_demo_data.py) is never read as a result: data/SYNTHETIC stamps every output and checks.py pins the banner.
- No person's name anywhere — not in code comments, docstrings, chart text, docs or the change log (write 'the review of <date>', 'the desk', 'as agreed'). `WM_REDACT_NAMES` in `.env` drives the hygiene check; the list itself never goes into a tracked file.
- Python 3.12: no same-quote nesting inside f-strings — precompute joined strings.
