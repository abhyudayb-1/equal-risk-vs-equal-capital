# CLAUDE.md

Guidance for Claude Code when working in this repository. `README.md` explains what the project
is and what it found. This file covers the rules for changing it.

## Commands

Always use the venv (system Python has no numpy). On Windows the interpreter is
`.\.venv\Scripts\python.exe`; elsewhere it is `.venv/bin/python`.

```bash
python compare_rules.py --no-charts                  # the study, ~6s
python compare_rules.py --no-charts --sensitivities  # + the 17-setting sweep, ~25s
python run_backtest.py --no-charts                   # the engine alone
for t in tests/test_*.py; do python "$t"; done       # all suites; each prints "N checks passed"
python -m src.<module>                               # per-module self-check on synthetic data
```

Run every suite after any change to `src/`. Tests are standalone scripts with no framework: each
file defines `test_*()` functions and a `check(label, condition)` helper, and its `__main__` block
calls them in order. New test files follow the same shape, including the `sys.path.insert`
preamble. **No test or self-check may touch the network**; live data belongs in the two CLIs.

## Rules that protect the results

- **No look-ahead.** At rebalance date `t`, use only data dated `<= t`. Route every estimator
  through `data.asof_slice()`. A decision at the close of `t` earns returns from `t+1`. `overlay.py`
  has two alignment rules that differ by one day; read its notes before touching them.
- **`BacktestResult` (`src/types.py`) is the contract.** Every strategy returns one and calls
  `.validate()` before returning. If it changes, change every call site in the same commit.
- **Financing is always charged** on leverage (`r_p = L·(r_a − r_f) + r_f`). Omitting it is the
  classic way this strategy gets oversold.
- **Failures are surfaced, never swallowed.** A solver that didn't converge (logged, counted in
  `meta`) is a different event from `ERCSolverError` (no feasible portfolio; it raises). Keep them
  distinct. `ERCSolverError` subclasses `ValueError`, so catch it *before* any `ValueError` handler.
- **Every constant lives in `src/config.py`**, with a comment recording how its value was chosen.
- **The overlay's lag and its ~12.5% volatility overshoot are findings, not bugs.** Don't tune them
  away, and never describe the matched column as exactly matched.
- **The oracle in `tests/test_riskparity.py` was written independently of the solver it checks.**
  Keep it independent: don't refactor it to share code with `src/riskparity.py`.

## Reports and the single data date

`README.md` quotes one live run on one data date (2026-11-02). Eight files in `reports/` are
tracked from that run. If you refresh numbers, re-run **both** CLIs and update every figure in the
README together. Mixing two data dates produces a comparison that cannot be right. Module
self-checks write to a temp directory, never to `reports/`. If `git status` shows `reports/`
changed after a self-check, restore it with `git checkout -- reports/`.

## Testing habits

- Never assert on a single simulated path's realised outcome; assert on parameters or mechanisms.
- A test that cannot fail proves nothing. Break the thing (e.g. monkeypatch the guard away) and
  check that the test fails.
- Every bug fix gets a regression check in the matching suite.
- Pass `encoding="utf-8"` when reading or writing text; Windows defaults to cp1252.
