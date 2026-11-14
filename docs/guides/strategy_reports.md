# Strategy research reports

A **report** is a self-contained, reproducible study of one strategy on one
universe and period. It is executed from a marimo notebook and exported to static
HTML. Every report runs the same version-controlled template
(`reports/templates/strategy_report.py`). Report *instances* — results,
conclusions, HTML — live in `reports/<name>/` and are **git-ignored**, because
they are proprietary.

## Workflow

```bash
# 1. Start from an example config (or another report's config.yaml)
algo report new tsmom-sp500 --from reports/templates/configs/tsmom_sp500.yaml

# 2. Edit reports/tsmom-sp500/config.yaml if needed, then build
algo report build tsmom-sp500
#    → reports/tsmom-sp500/report.html and tearsheet.html

# 3. Read it, write conclusions into reports/tsmom-sp500/notes.md, rebuild
algo report build tsmom-sp500    # resumes: only missing trials run

# Interactive
algo report edit tsmom-sp500
```

`build` executes the notebook headlessly (`marimo export html --no-include-code`).
The studies are stored in the report's own `optuna.db` and **resumed**: a rebuild
re-renders using stored trials and runs only what is missing (for example after
raising `n_trials` or `top_k`). Each study records the config's *fingerprint*, a
hash of everything that affects results. A build whose config no longer matches
its stored trials fails instead of silently mixing them. Use a new report name for
each variant. `--refresh-template` replaces a report's notebook with the current
template.

## What a report contains

| Section | Content | Source |
|---|---|---|
| Executive summary | Validated OOS Sharpe, DSR, PBO, return, drawdown, checks | computed |
| 1 Theory | Maths and economics of the strategy | `docs/strategies/<strategy>.md` |
| 2 Implementation | How targets become trades; the strategy's source; signals on one symbol | code |
| 3 Methodology | Universe coverage, fold design, costs, search space, config, commands, environment | config + catalog |
| 4 Screening engine | Runtime, distribution of the objective and fold Sharpes, parameter sensitivity, leaderboard | screening study |
| 5 Nautilus validation | Engine parity on the top-k, validated OOS statistics, equity/drawdown vs benchmark, full-period Nautilus run | validation study |
| 6 Tearsheet | NautilusTrader's tearsheet of the validated OOS returns | `tearsheet.html` |
| 7 Conclusions | Automated checks, analyst notes, standing caveats | `notes.md` |
| 8 Re-running | Which config fields to change for new assets or periods | — |

## The three stages

1. **Screening.** An Optuna walk-forward study (`evaluation.run_study`) on
   `screening.engine`, normally `vector`: the numba kernel in
   `execution/vector.py`, which reproduces Nautilus's fills and returns on bars
   with volume. Each trial backtests every fold's test window, and its objective
   is the mean fold Sharpe.
2. **Validation.** The `validation.top_k` best distinct parameter sets are
   enqueued into a second study and re-run fold for fold on Nautilus. The report
   compares the two engines trial by trial and fold by fold. The validated best
   is summarised with `summarize_study(..., selection=screening_study)`. Its
   **Deflated Sharpe Ratio** is deflated for *every screened trial* (their count
   and the dispersion of their Sharpes), and the **PBO** is computed over them.
   The best of five re-runs was selected from the full search, so the full search
   is what the deflation must account for.
   The `top_k × n_splits` Nautilus fold runs are spread over worker processes
   (`validation.workers`, default half the CPUs). Each worker reloads the
   universe from the catalog (~3 GB for the S&P 500 daily set), so lower it on a
   machine short of memory. It does not change results or the fingerprint.
3. **Full backtest.** The validated parameters on Nautilus over the whole period
   after warm-up, for Nautilus's own statistics, order counts, rejections and
   turnover. This is in-sample for the parameters and labelled as such.

## Reading the numbers

* **Sharpe**: annualised from daily mark-to-market returns (252 days), as
  Nautilus computes `Sharpe Ratio (252 days)`.
* **PSR**: probability that the true Sharpe exceeds 0, given the track record's
  length, skew and kurtosis.
* **DSR**: PSR against the Sharpe the best of N skill-less trials would reach by
  luck. Treat ≥ 0.95 as significant.
* **PBO**: probability that the in-sample best trial ranks below the median out
  of sample, by combinatorially symmetric cross-validation. Below 0.5 is
  necessary; well below is better.
* **Minimum track record**: years of returns needed for 95% confidence that the
  Sharpe is positive. If it exceeds the out-of-sample span, the data cannot yet
  distinguish the strategy from noise.

## Universes with listings and delistings

Index-member universes (e.g. `data/universe/sp500_ever_members.txt`, every
ticker in the S&P 500 at any point since the start date) avoid survivorship bias
but are ragged: names list and delist at different times.

* Folds are cut on the **union** of the universe's bar closes
  (`MarketData.closes(union=True)`).
* Multi-asset strategies see frames aligned on the union of dates. A symbol is
  null before its first and after its last bar, so it drops out of
  cross-sectional rankings automatically.
* A symbol whose data ends before the universe's is **flattened on its last bar**
  (its last close stands in for the delisting price).
* With many names, each position is roughly `1/N` of equity. Keep
  `evaluation.rebalance_band` well below `1/N` or nothing trades. Use a large
  `initial_capital` so whole-share rounding is negligible.
* Zero-volume bars (placeholders some vendors emit) get no fill on Nautilus,
  which reports "No market" rejections. The kernel fills them at the close. The
  data section counts them, and the checks flag any rejections.

## Adding a strategy to the report workflow

1. Implement and register it as described in [Adding a Strategy](adding_strategy.md).
2. Write `docs/strategies/<registry-name>.md`: signal and sizing maths, economic
   rationale, parameters, testable hypotheses and references. It is rendered
   as section 1 of every report on that strategy (headings are demoted one level).
3. Add an example config under `reports/templates/configs/`.

## Changing the template

The template is ordinary marimo. Keep computations in
`snippy_scales.reporting.pipeline`, where they are unit-tested, and keep only
layout in the notebook. Run `uv run marimo check reports/templates/strategy_report.py`
after editing. Existing reports keep their copy until rebuilt with
`--refresh-template`.
