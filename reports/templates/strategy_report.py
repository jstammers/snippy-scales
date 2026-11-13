"""Strategy research report — the version-controlled marimo template.

Do not run this file in place: instances live in ``reports/<name>/`` (git-ignored)
next to their ``config.yaml``::

    algo report new <name> --from reports/templates/configs/<example>.yaml
    algo report build <name>       # executes this notebook → reports/<name>/report.html
    algo report edit <name>        # interactive

The computations are in ``snippy_scales.reporting.pipeline`` (tested); this
notebook only lays them out. See ``docs/guides/strategy_reports.md``.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Strategy research report")


@app.cell(hide_code=True)
def _():
    import inspect
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from snippy_scales.execution import compute_targets
    from snippy_scales.reporting import ReportPaths, load_report_config
    from snippy_scales.reporting import pipeline as rp

    _ = alt.data_transformers.disable_max_rows()
    return (
        Path,
        ReportPaths,
        alt,
        compute_targets,
        inspect,
        load_report_config,
        mo,
        pl,
        rp,
    )


@app.cell(hide_code=True)
def _(Path, ReportPaths, load_report_config, mo):
    _arg = mo.cli_args().get("config")
    config_path = Path(_arg) if _arg else Path(mo.notebook_dir() or ".") / "config.yaml"
    mo.stop(
        not config_path.exists(),
        mo.callout(
            mo.md(
                f"""
                **No `config.yaml` next to this notebook** (looked for `{config_path}`).

                This is the report **template**. Create a report instance and build it:

                ```bash
                algo report new my-report --from reports/templates/configs/tsmom_sp500.yaml
                algo report build my-report
                ```
                """
            ),
            kind="warn",
        ),
    )
    cfg = load_report_config(config_path)
    paths = ReportPaths(config_path.parent)
    return cfg, paths


@app.cell(hide_code=True)
def _(rp):
    env = rp.environment()
    return (env,)


@app.cell(hide_code=True)
def _(cfg, env, mo):
    mo.md(
        f"""
        # {cfg.title}

        Report `{cfg.name}` · strategy `{cfg.strategy}` · generated {env["generated"]}
        from commit `{env["git commit"]}` (`{env["git branch"]}`)

        {f"> **Research question.** {cfg.question}" if cfg.question else ""}
        """
    )
    return


@app.cell(hide_code=True)
def _(bench_oos, checks_df, mo, rp, validation):
    _r = validation.report
    _passed = int(checks_df["passed"].sum())
    _bench = f"{rp.annualised_sharpe(bench_oos):.2f}" if bench_oos is not None else "n/a"
    mo.vstack(
        [
            mo.md("## Executive summary"),
            mo.md(
                "Headline numbers are for the **validated best parameter set**: its "
                "walk-forward out-of-sample (OOS) returns, stitched across folds and run "
                "on Nautilus. The deflated Sharpe ratio accounts for every trial screened. "
                "Details in sections 4–5; the verdict is in section 7."
            ),
            mo.hstack(
                [
                    mo.stat(f"{_r.sharpe:.2f}", label="OOS Sharpe", caption=f"benchmark {_bench}"),
                    mo.stat(
                        f"{_r.dsr:.2f}", label="Deflated Sharpe", caption=f"best of {_r.n_trials}"
                    ),
                    mo.stat(
                        "n/a" if _r.pbo is None else f"{_r.pbo:.2f}",
                        label="PBO",
                        caption="prob. of overfitting",
                    ),
                    mo.stat(f"{_r.total_return:+.1%}", label="OOS return"),
                    mo.stat(f"{_r.max_drawdown:.1%}", label="OOS max drawdown"),
                    mo.stat(f"{_passed}/{checks_df.height}", label="Checks passed"),
                ],
                widths="equal",
                wrap=True,
            ),
            mo.md(f"**Best parameters:** `{_r.best_params}`"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Contents** — 1 Theory · 2 Implementation · 3 Methodology (data, design,
    hyper-parameters, reproduction) · 4 Screening engine · 5 Nautilus validation ·
    6 Tearsheet · 7 Conclusions · 8 Re-running on new assets or periods
    """)
    return


@app.cell(hide_code=True)
def _(cfg, mo, rp):
    _text = rp.theory_markdown(cfg)
    mo.vstack(
        [
            mo.md("## 1. Theory"),
            mo.md(_text)
            if _text
            else mo.callout(
                mo.md(
                    f"No theory write-up at `{cfg.theory_path}`. Add one (see "
                    "`docs/strategies/`) or set `theory:` in the config."
                ),
                kind="warn",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(cfg, inspect, mo, rp):
    strategy_cls = rp.strategy_class(cfg)
    _source = inspect.getsource(strategy_cls)
    _module = strategy_cls.__module__
    mo.vstack(
        [
            mo.md(
                f"""
                ## 2. Implementation

                The strategy is `{_module}.{strategy_cls.__qualname__}`, registered as
                `{cfg.strategy}` in `snippy_scales/cli/_common.py:registry()`. Like every
                research strategy here it is **pure polars/numpy** and **causal**: it maps
                bars to *target positions in units of equity* and knows nothing about
                orders (`tests/test_strategy_causality.py` checks it never looks ahead).

                The same targets drive both backtest engines:

                1. `execution.compute_targets` runs `generate_signals` once on the whole
                   history and keys each target by bar close. For a multi-asset strategy,
                   the frames are first aligned on the **union** of bar closes. A symbol
                   is null before its first bar and after its last, so a universe whose
                   members list and delist at different times keeps every date. A symbol
                   whose data ends before the universe's is flattened on its last bar.
                2. At each bar close the engine holds
                   `contracts = round(target × weight × equity / (close × multiplier))`,
                   with weights `1/N` across symbols, filled at that close (so the
                   position earns the *next* bar's return). Trades smaller than
                   `rebalance_band × equity` are skipped.
                3. **Screening** replays this in a numba kernel
                   (`execution/vector.py`, thousands of times faster); **validation**
                   replays it bar by bar in NautilusTrader (`execution/strategy.py:
                   SignalStrategy`), the reference implementation.
                """
            ),
            mo.accordion(
                {f"Source of `{strategy_cls.__qualname__}`": mo.md(f"```python\n{_source}\n```")}
            ),
        ]
    )
    return (strategy_cls,)


@app.cell(hide_code=True)
def _(cfg, mo, screening, strategy_cls):
    _params = screening.report.best_params
    _args = ", ".join(f"{k}={v!r}" for k, v in _params.items())
    mo.md(
        f"""
        Running the screened best parameters by hand:

        ```python
        from snippy_scales.execution import Costs, load_market_data, run_backtest, vector
        from {strategy_cls.__module__} import {strategy_cls.__qualname__}

        data = load_market_data(symbols, "{cfg.data.schema_}", venue={cfg.data.venue!r})
        strategy = {strategy_cls.__qualname__}({_args})
        fast = vector.run_backtest(strategy, data, start=..., costs=Costs(...))  # screening kernel
        run = run_backtest(strategy, data, start=..., costs=Costs(...))          # Nautilus
        run.sharpe, run.stats, run.equity, run.fills
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(alt, cfg, compute_targets, data, mo, pl, screening, strategy_cls):
    _symbol = cfg.data.example_symbol if cfg.data.example_symbol in data.frames else data.symbols[0]
    _research = strategy_cls(**screening.report.best_params)
    _targets = compute_targets(_research, data.frames)[_symbol]
    _frame = (
        data.frames[_symbol]
        .select("ts_init", "close")
        .with_columns(
            pl.col("ts_init").dt.epoch("ns").replace_strict(_targets, default=None).alias("target")
        )
    )
    _base = alt.Chart(_frame).encode(x=alt.X("ts_init:T", title=None))
    _price = _base.mark_line(strokeWidth=1).encode(y=alt.Y("close:Q", title=f"{_symbol} close"))
    _target = _base.mark_area(opacity=0.6, interpolate="step-after").encode(
        y=alt.Y("target:Q", title="target (units of equity)")
    )
    mo.vstack(
        [
            mo.md(
                f"**Signals on one symbol.** `{_symbol}` close and the target position the "
                "screened best parameters produce for it (before the `1/N` weight)."
            ),
            alt.vconcat(
                _price.properties(height=180, width="container"),
                _target.properties(height=120, width="container"),
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(cfg, mo, rp):
    with mo.status.spinner(title="Loading market data from the catalog…"):
        data = rp.load_universe(cfg)
    missing = rp.missing_symbols(cfg, data)
    return data, missing


@app.cell(hide_code=True)
def _(cfg, data, missing, mo, pl, rp):
    summary_df = rp.data_summary(data)
    calendar = rp.calendar(cfg, data)
    _ended = summary_df.filter(pl.col("last") < summary_df["last"].max()).height
    _started = summary_df.filter(pl.col("first") > summary_df["first"].min()).height
    _d = cfg.data
    _source = (
        f"universe file `data/universe/{_d.universe}.txt`"
        if _d.universe
        else "explicit symbol list"
    )
    mo.vstack(
        [
            mo.md(
                f"""
                ## 3. Methodology

                ### 3.1 Data

                | | |
                |---|---|
                | Universe | {_source}{f" (first {_d.max_symbols})" if _d.max_symbols else ""} |
                | Symbols loaded | **{len(data.frames)}** ({len(missing)} configured symbols have no `{_d.schema_}` bars) |
                | Bars | `{_d.schema_}` at venue `{_d.venue or "auto"}`, {summary_df["bars"].sum():,} in total |
                | History | {summary_df["first"].min():%Y-%m-%d} → {summary_df["last"].max():%Y-%m-%d} |
                | Evaluated period | {calendar[0]:%Y-%m-%d} → {calendar[-1]:%Y-%m-%d} ({calendar.len():,} bar closes) |
                | Listed after the start / ended before the end | {_started} / {_ended} symbols |
                | Benchmark | {f"`{_d.benchmark}` buy-and-hold" if _d.benchmark else "none"} |
                | Zero-volume bars | {summary_df["zero_volume_bars"].sum():,} across {summary_df.filter(pl.col("zero_volume_bars") > 0).height} symbols (Nautilus fills nothing on them; the screening kernel does) |

                Bars come from the NautilusTrader catalog (`data/catalog/`), exactly as
                stored (`ts_init` = bar close). Equity bars from Alpaca are split- and
                dividend-adjusted (`adjustment: all` in the ingest config), so returns
                include dividends. A symbol with no bar on a date simply doesn't trade
                that day. A symbol whose data ends early is closed at its last close.
                That close is a proxy for the delisting price; no delisting return is
                modelled.
                """
            ),
            mo.accordion(
                {
                    "Per-symbol coverage": mo.ui.table(summary_df, page_size=15, selection=None),
                    "Configured symbols without bars": mo.md(", ".join(missing) or "none"),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(alt, data, mo, rp):
    _size = rp.universe_size(data)
    mo.vstack(
        [
            mo.md("**Universe size over time.** Symbols with a bar at each close."),
            alt.Chart(_size)
            .mark_line(strokeWidth=1)
            .encode(x=alt.X("ts:T", title=None), y=alt.Y("symbols:Q", title="symbols with a bar"))
            .properties(height=160, width="container"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(alt, cfg, data, mo, pl, rp):
    fold_list = rp.folds(cfg, data)
    _ff = rp.fold_frame(fold_list)
    _bars = pl.concat(
        [
            _ff.select(
                "fold",
                pl.col("train_start").alias("from"),
                pl.col("train_end").alias("to"),
                pl.lit("signal history (warm-up)").alias("window"),
            ),
            _ff.select(
                "fold",
                pl.col("test_start").alias("from"),
                pl.col("test_end").alias("to"),
                pl.lit("out-of-sample test").alias("window"),
            ),
        ]
    )
    _e = cfg.evaluation
    mo.vstack(
        [
            mo.md(
                f"""
                ### 3.2 Evaluation design

                **Walk-forward.** {_e.n_splits} contiguous, non-overlapping test windows
                at the end of the evaluated period, each {_e.test_size:g} of its bar
                closes (capped so at least {_e.min_train} closes precede the first one).
                Each fold trades **only** its test window. Signals are computed on every
                bar up to the window's end, so the strategy is warm on the first test
                bar and never sees later data. The built-in strategies fit nothing, so
                parameters are *selected* on these folds. That selection is why the
                Deflated Sharpe ratio and the probability of backtest overfitting (PBO)
                are reported, not raw Sharpe alone.

                **Objective.** Each trial's value is its **mean fold Sharpe**
                (annualised, daily mark-to-market, `Sharpe Ratio (252 days)` as
                Nautilus computes it). The reported out-of-sample series concatenates
                ("stitches") the fold test windows.

                **Execution assumptions.**

                | | |
                |---|---|
                | Initial capital | {_e.initial_capital:,.0f} |
                | Commission | {_e.commission_rate:.2%} of notional{f" + {_e.commission_per_contract} per contract" if _e.commission_per_contract else ""} |
                | Slippage | {_e.slippage_prob:.0%} probability of one tick per fill |
                | Rebalance band | {_e.rebalance_band:.4%} of equity per trade (smaller trades skipped) |
                | Leverage (Nautilus margin) | {_e.leverage:g}× |
                | Fills | at the bar close that produced the signal; no market impact |
                """
            ),
            mo.ui.table(_ff, selection=None),
            alt.Chart(_bars)
            .mark_bar(height=14)
            .encode(
                x=alt.X("from:T", title=None),
                x2="to:T",
                y=alt.Y("fold:O"),
                color=alt.Color("window:N", legend=alt.Legend(orient="bottom", title=None)),
            )
            .properties(height=130, width="container"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(cfg, mo, rp):
    _s = cfg.screening
    mo.vstack(
        [
            mo.md(
                f"""
                ### 3.3 Hyper-parameters

                Search: **{_s.sampler}** sampler (seed {_s.seed}),
                {"the full grid" if _s.sampler == "grid" else f"{_s.n_trials} trials"} on the
                **{_s.engine}** engine; then the top **{cfg.validation.top_k}** distinct
                parameter sets re-run on Nautilus. Parameters not searched keep their
                defaults.
                """
            ),
            mo.ui.table(rp.space_table(cfg), selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(cfg, env, mo, paths, rp):
    _env = "\n".join(f"| {k} | `{v}` |" for k, v in env.items())
    mo.md(
        f"""
        ### 3.4 Reproduction

        The report is fully determined by its config, the code at the commit below,
        and the catalog contents. Config fingerprint (what the studies are keyed on):
        `{cfg.fingerprint()}`.

        ```yaml
        {cfg.to_yaml().replace(chr(10), chr(10) + "        ")}
        ```

        ```bash
        {rp.commands(cfg, paths).replace(chr(10), chr(10) + "        ")}
        ```

        | Environment | |
        |---|---|
        {_env.replace(chr(10), chr(10) + "        ")}
        """
    )
    return


@app.cell(hide_code=True)
def _(cfg, data, mo, paths, rp):
    with mo.status.spinner(title=f"Screening on the {cfg.screening.engine} engine…"):
        screening = rp.run_screening(cfg, data, paths)
    return (screening,)


@app.cell(hide_code=True)
def _(cfg, mo, screening):
    _t = screening.trials
    _dur = _t["duration_s"]
    _n_bt = _t.height * cfg.evaluation.n_splits
    _r = screening.report
    mo.vstack(
        [
            mo.md(
                f"""
                ## 4. Screening engine

                {_t.height} trials × {cfg.evaluation.n_splits} folds = **{_n_bt:,} backtests**
                on the `{cfg.screening.engine}` engine ({screening.new_trials} run in this
                build, the rest resumed from `optuna.db`). A trial's time includes signal
                generation for every fold, which for single-asset strategies is usually the
                larger share.
                """
            ),
            mo.hstack(
                [
                    mo.stat(f"{_dur.sum():.0f} s", label="Total trial time"),
                    mo.stat(f"{_dur.median():.2f} s", label="Median per trial"),
                    mo.stat(f"{_dur.sum() / _n_bt * 1000:.0f} ms", label="Per fold backtest"),
                    mo.stat(
                        f"{_t.height / max(_dur.sum(), 1e-9) * 60:.1f}", label="Trials / minute"
                    ),
                ],
                widths="equal",
            ),
            mo.md(
                f"""
                **Best screened trial.** Mean fold Sharpe {_r.mean_fold_sharpe:.3f};
                stitched OOS Sharpe {_r.sharpe:.3f}; DSR {_r.dsr:.3f} (best of {_r.n_trials});
                PBO {"n/a" if _r.pbo is None else f"{_r.pbo:.2f}"}. These are screening-engine
                numbers; section 5 re-runs the leaders on Nautilus.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(alt, mo, pl, screening):
    _t = screening.trials
    _hist = (
        alt.Chart(_t)
        .mark_bar()
        .encode(
            x=alt.X("mean_fold_sharpe:Q", bin=alt.Bin(maxbins=30), title="mean fold Sharpe"),
            y=alt.Y("count():Q", title="trials"),
        )
        .properties(height=180, width=300, title="Objective across trials")
    )
    _dur = (
        alt.Chart(_t)
        .mark_bar()
        .encode(
            x=alt.X("duration_s:Q", bin=alt.Bin(maxbins=30), title="seconds per trial"),
            y=alt.Y("count():Q", title="trials"),
        )
        .properties(height=180, width=300, title="Runtime across trials")
    )
    _folds = [c for c in _t.columns if c.startswith("fold_")]
    _long = _t.unpivot(_folds, index="trial", variable_name="fold", value_name="sharpe")
    _box = (
        alt.Chart(_long)
        .mark_boxplot()
        .encode(x=alt.X("fold:N", title=None), y=alt.Y("sharpe:Q", title="fold Sharpe"))
        .properties(height=180, width=300, title="Fold Sharpe across trials")
    )
    _hist_best = _t.with_columns(pl.col("mean_fold_sharpe").cum_max().alias("best so far"))
    _history = (
        alt.Chart(_hist_best)
        .mark_circle(size=18, opacity=0.6)
        .encode(x=alt.X("trial:Q"), y=alt.Y("mean_fold_sharpe:Q", title="mean fold Sharpe"))
        + alt.Chart(_hist_best).mark_line(color="firebrick").encode(x="trial:Q", y="best so far:Q")
    ).properties(height=180, width=300, title="Optimisation history")
    mo.vstack(
        [
            mo.md("### 4.1 Distribution of results"),
            alt.vconcat(alt.hconcat(_hist, _dur), alt.hconcat(_box, _history)),
        ]
    )
    return


@app.cell(hide_code=True)
def _(alt, mo, screening):
    _t = screening.trials
    _params = [c for c in _t.columns if c.startswith("p_")]
    _charts = [
        alt.Chart(_t)
        .mark_circle(size=30, opacity=0.6)
        .encode(
            x=alt.X(f"{p}:Q", title=p.removeprefix("p_"), scale=alt.Scale(zero=False)),
            y=alt.Y("mean_fold_sharpe:Q", title="mean fold Sharpe"),
            tooltip=["trial", p, "mean_fold_sharpe"],
        )
        .properties(height=160, width=200)
        for p in _params
    ]
    try:
        import optuna

        _imp = optuna.importance.get_param_importances(screening.study)
        _imp_md = " · ".join(f"`{k}` {v:.2f}" for k, v in _imp.items())
    except Exception as _exc:  # noqa: BLE001 — importance needs ≥2 varied parameters
        _imp_md = f"n/a ({_exc.__class__.__name__})"
    mo.vstack(
        [
            mo.md(
                "### 4.2 Parameter sensitivity\n\nEach point is one trial. A robust "
                "strategy shows broad plateaus, not isolated spikes. Parameter importance "
                f"(fANOVA): {_imp_md}"
            ),
            alt.hconcat(*_charts) if _charts else mo.md("_No searched parameters._"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, screening):
    _top = screening.trials.sort("mean_fold_sharpe", descending=True).head(15)
    mo.vstack([mo.md("### 4.3 Leaderboard (screening engine)"), mo.ui.table(_top, selection=None)])
    return


@app.cell(hide_code=True)
def _(cfg, data, mo, paths, rp, screening):
    with mo.status.spinner(title=f"Validating the top {cfg.validation.top_k} on Nautilus…"):
        validation = rp.run_validation(cfg, data, screening, paths)
    comparison = rp.engine_comparison(screening, validation)
    return comparison, validation


@app.cell(hide_code=True)
def _(alt, comparison, mo, pl, rp, screening, validation):
    _fc = rp.fold_comparison(screening, validation)
    _lo = float(min(_fc["screen"].min(), _fc["nautilus"].min())) if _fc.height else -1.0
    _hi = float(max(_fc["screen"].max(), _fc["nautilus"].max())) if _fc.height else 1.0
    _diag = (
        alt.Chart(pl.DataFrame({"x": [_lo, _hi]}))
        .mark_line(color="grey", strokeDash=[4, 4])
        .encode(x="x:Q", y="x:Q")
    )
    _scatter = (
        alt.Chart(_fc)
        .mark_circle(size=40)
        .encode(
            x=alt.X("screen:Q", title="screening engine fold Sharpe"),
            y=alt.Y("nautilus:Q", title="Nautilus fold Sharpe"),
            color=alt.Color("fold:N"),
        )
    )
    _speed = comparison["speedup"].median() if comparison.height else float("nan")
    mo.vstack(
        [
            mo.md(
                f"""
                ## 5. Nautilus validation

                The top {comparison.height} distinct screened parameter sets, re-run fold for
                fold on NautilusTrader (event-driven, with margin checks, liquidity from
                bar volume, and per-order commissions). Points on the dashed line mean the
                engines agree. Nautilus took a median **{_speed:.0f}×** longer per trial.
                """
            ),
            mo.ui.table(comparison, selection=None),
            (_diag + _scatter).properties(height=260, width=320),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, validation):
    _r = validation.report
    _rows = [
        ("Validated best parameters", f"`{_r.best_params}`"),
        ("Mean fold Sharpe (Nautilus)", f"{_r.mean_fold_sharpe:.3f}"),
        ("Fold Sharpes", ", ".join(f"{s:.2f}" for s in _r.fold_sharpes)),
        ("Stitched OOS Sharpe", f"{_r.sharpe:.3f}"),
        ("Stitched OOS return", f"{_r.total_return:+.2%}"),
        ("Stitched OOS max drawdown", f"{_r.max_drawdown:.2%}"),
        ("PSR (true Sharpe > 0)", f"{_r.psr:.3f}"),
        (f"DSR (deflated for {_r.n_trials} screened trials)", f"{_r.dsr:.3f}"),
        ("PBO (over screened trials)", "n/a" if _r.pbo is None else f"{_r.pbo:.2f}"),
        ("Minimum track record (95%)", f"{_r.min_track_record_years:.1f} years"),
    ]
    _stats = "\n".join(f"| {k} | {v:.4g} |" for k, v in sorted(_r.stats_returns.items()))
    mo.vstack(
        [
            mo.md(
                "### 5.1 Validated out-of-sample performance\n\n| | |\n|---|---|\n"
                + "\n".join(f"| {k} | {v} |" for k, v in _rows)
            ),
            mo.accordion(
                {
                    "Nautilus return statistics (stitched OOS)": mo.md(
                        "| statistic | value |\n|---|---|\n" + _stats
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(cfg, rp, validation):
    bench = rp.benchmark_returns(cfg)
    bench_oos = rp.within(bench, validation.report.returns) if bench is not None else None
    return bench, bench_oos


@app.cell(hide_code=True)
def _(alt, bench_oos, cfg, mo, pl, rp, validation):
    _series = [
        rp.drawdown(validation.report.returns).with_columns(
            pl.lit("strategy (OOS)").alias("series")
        )
    ]
    if bench_oos is not None:
        _series.append(
            rp.drawdown(bench_oos).with_columns(
                pl.lit(f"{cfg.data.benchmark} buy-and-hold").alias("series")
            )
        )
    _dd = pl.concat(_series)
    _eq = (
        alt.Chart(_dd)
        .mark_line(strokeWidth=1.2)
        .encode(
            x=alt.X("ts:T", title=None),
            y=alt.Y("equity:Q", title="growth of 1", scale=alt.Scale(type="log")),
            color=alt.Color("series:N", legend=alt.Legend(orient="bottom", title=None)),
        )
        .properties(height=220, width="container")
    )
    _draw = (
        alt.Chart(_dd)
        .mark_area(opacity=0.35)
        .encode(
            x=alt.X("ts:T", title=None),
            y=alt.Y("drawdown:Q", title="drawdown", axis=alt.Axis(format="%")),
            color="series:N",
        )
        .properties(height=130, width="container")
    )
    _years = [
        rp.calendar_year_returns(validation.report.returns).with_columns(
            pl.lit("strategy (OOS)").alias("series")
        )
    ]
    if bench_oos is not None:
        _years.append(
            rp.calendar_year_returns(bench_oos).with_columns(
                pl.lit(str(cfg.data.benchmark)).alias("series")
            )
        )
    _yr = (
        alt.Chart(pl.concat(_years))
        .mark_bar()
        .encode(
            x=alt.X("year:O"),
            xOffset="series:N",
            y=alt.Y("return:Q", axis=alt.Axis(format="%")),
            color="series:N",
        )
        .properties(height=160, width="container")
    )
    mo.vstack(
        [
            mo.md(
                "**Stitched OOS equity, drawdown and calendar-year returns**, with the "
                "benchmark over the same days. Gaps between folds are not traded."
            ),
            alt.vconcat(_eq, _draw, _yr),
        ]
    )
    return


@app.cell(hide_code=True)
def _(cfg, data, mo, rp, validation):
    with mo.status.spinner(title="Full-period Nautilus backtest of the validated best…"):
        full_run = rp.full_backtest(cfg, data, validation.report.best_params)
    return (full_run,)


@app.cell(hide_code=True)
def _(alt, full_run, mo, pl, rp):
    _stats = "\n".join(f"| {k} | {v:,.4g} |" for k, v in sorted(full_run.stats.items()))
    _eq = full_run.equity
    _fills = full_run.fills
    _turnover = (
        float((_fills["last_qty"].astype(float) * _fills["last_px"].astype(float)).sum())
        if len(_fills)
        else 0.0
    )
    _years = max((_eq["ts"][-1] - _eq["ts"][0]).days / 365.25, 1e-9) if _eq.height > 1 else 1.0
    _start_equity = float(_eq["equity"][0]) if _eq.height else 1.0
    mo.vstack(
        [
            mo.md(
                f"""
                ### 5.2 Full-period Nautilus backtest

                The validated parameters over the whole evaluated period after warm-up.
                **This run is in-sample for the parameters**: they were chosen on folds
                inside it. It shows how Nautilus trades the strategy, not an unbiased
                estimate. Use 5.1 for that.

                | | |
                |---|---|
                | Sharpe | {full_run.sharpe:.3f} |
                | Total return | {full_run.total_return:+.2%} |
                | Orders | {full_run.total_orders:,} |
                | Denied / rejected orders | {len(full_run.rejections)} |
                | Annual turnover (traded notional / starting equity) | {_turnover / _start_equity / _years:.1f}× |
                """
            ),
            alt.Chart(
                _eq.with_columns((pl.col("equity") / pl.col("equity").first()).alias("growth"))
            )
            .mark_line(strokeWidth=1.2)
            .encode(x=alt.X("ts:T", title=None), y=alt.Y("growth:Q", title="equity (growth of 1)"))
            .properties(height=200, width="container"),
            mo.accordion(
                {
                    "All Nautilus statistics": mo.md("| statistic | value |\n|---|---|\n" + _stats),
                    "Denied / rejected orders by reason": mo.ui.table(
                        rp.rejection_summary(full_run), selection=None
                    ),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(bench, cfg, mo, paths, validation):
    from snippy_scales.evaluation import write_tearsheet

    write_tearsheet(
        validation.report,
        paths.tearsheet,
        benchmark=bench,
        benchmark_name=cfg.data.benchmark or "Benchmark",
    )
    mo.vstack(
        [
            mo.md(
                f"""
                ## 6. Tearsheet

                NautilusTrader's tearsheet of the validated stitched OOS returns, also
                saved as `{paths.tearsheet.name}` next to this report.
                """
            ),
            mo.iframe(paths.tearsheet.read_text(), height="1600px"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(bench_oos, comparison, full_run, rp, validation):
    checks_df = rp.checks(validation, comparison, bench_oos, full_run)
    return (checks_df,)


@app.cell(hide_code=True)
def _(checks_df, mo, paths, pl):
    _table = checks_df.with_columns(
        pl.when(pl.col("passed")).then(pl.lit("✅")).otherwise(pl.lit("❌")).alias("result"),
        pl.col("value").round(3),
    ).select("result", "check", "value")
    _passed = int(checks_df["passed"].sum())
    _kind = (
        "success"
        if _passed == checks_df.height
        else ("warn" if _passed >= checks_df.height / 2 else "danger")
    )
    _notes = paths.notes.read_text() if paths.notes.exists() else "_No `notes.md`._"
    mo.vstack(
        [
            mo.md("## 7. Conclusions\n\n### 7.1 Automated checks"),
            mo.callout(
                mo.md(
                    f"**{_passed} of {checks_df.height} checks passed.** These are necessary, not sufficient: read them with the caveats below and the analyst's notes."
                ),
                kind=_kind,
            ),
            mo.ui.table(_table, selection=None),
            mo.md("### 7.2 Analyst notes\n\n" + _notes),
            mo.md(
                """
                ### 7.3 Standing caveats

                * **Selection.** Parameters are chosen on the same folds that are
                  reported. The DSR and PBO correct for that statistically, but a
                  genuinely untouched hold-out period is stronger evidence.
                * **Universe.** An "ever members" index universe avoids survivorship bias
                  from dropping delisted names. It is not *point-in-time*: a stock trades
                  before it joined the index, which can flatter momentum-style signals.
                * **Execution.** Fills are at the signal bar's close with no market impact.
                  Costs are a flat commission (plus optional tick slippage). Short-borrow
                  costs and dividends paid on shorts are not modelled.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 8. Re-running on new assets or periods

    Every input is in the report's `config.yaml`. Start a variant with
    `algo report new <name> --from reports/<this report>/config.yaml`, edit it,
    then `algo report build <name>`:

    | To change… | edit |
    |---|---|
    | Asset class / universe | `data.universe` (a `data/universe/<name>.txt` file) or `data.symbols`; `data.venue`, `data.schema` (e.g. `GLBX` futures, `ohlcv-1d`) |
    | Time period | `data.start`, `data.end` (signals still warm up on earlier bars) |
    | Fold design | `evaluation.n_splits`, `test_size`, `min_train` |
    | Costs and sizing | `evaluation.commission_rate`, `commission_per_contract`, `slippage_prob`, `rebalance_band`, `initial_capital` |
    | Search | `screening.space`, `sampler`, `n_trials`; `validation.top_k` |
    | Strategy | `strategy` (a registry name) and `theory` (a markdown file under `docs/strategies/`) |

    A changed config gets a new fingerprint, and the build refuses to mix it with
    trials stored under the old one. Use a new report name for each variant.
    Futures need instrument definitions in the catalog
    (`algo data sync <config> --instruments-only`). For several instruments, prefer
    a parent symbol such as `ES.FUT` over `ES.c.0`, so rolls are traded on real
    outrights.
    """)
    return


if __name__ == "__main__":
    app.run()
