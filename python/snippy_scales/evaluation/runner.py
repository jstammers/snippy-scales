"""High-level orchestrator for walk-forward evaluation with parameter search.

:class:`EvaluationRunner` chains together:

1. :class:`~snippy_scales.evaluation.split.WalkForwardSplit` — generates
   train/test folds.
2. Parameter search iteration (:class:`~snippy_scales.evaluation.sweep.ParameterGrid`,
   :class:`~snippy_scales.evaluation.sweep.RandomSearch`, or
   :class:`~snippy_scales.evaluation.sweep.OptunaSearch`).
3. :class:`~snippy_scales.backtesting.runners.BacktestRunner` /
   :class:`~snippy_scales.backtesting.runners.BasketRunner` — runs the strategy
   on each train and test slice.
4. :class:`~snippy_scales.evaluation.database.SQLiteStore` — persists results
   (optional).
5. :class:`~snippy_scales.evaluation.tearsheet.TearsheetGenerator` — saves
   tearsheet reports (optional).

Usage::

    from snippy_scales.evaluation import EvaluationRunner, ParameterGrid
    from snippy_scales.strategies.trend import TrendFollowing

    runner = EvaluationRunner(
        n_splits=5,
        db_path="results.db",
        tearsheet_dir="reports/",
    )
    grid = ParameterGrid({"fast_period": [10, 20, 40], "slow_period": [40, 60, 120]})
    result = runner.evaluate(TrendFollowing, bars, params=grid, symbol="ES.c.0")

    print(result.best_params)
    print(result.summary_df())
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from snippy_scales.backtesting.runners import BacktestRunner, BasketRunner
from snippy_scales.evaluation.results import EvaluationResult, FoldResult, SweepResult
from snippy_scales.evaluation.split import WalkForwardSplit
from snippy_scales.evaluation.tearsheet import TearsheetGenerator

if TYPE_CHECKING:
    import polars as pl

    from snippy_scales.strategies.momentum_cs import MultiAssetStrategy
    from snippy_scales.strategies.trend import Strategy

logger = logging.getLogger(__name__)

_ParamSearch = Any  # ParameterGrid | RandomSearch | OptunaSearch | dict | None


class EvaluationRunner:
    """Walk-forward evaluation runner with optional parameter search.

    Args:
        initial_capital: Starting portfolio equity (default 1 000 000).
        fees: Per-trade commission fraction (default 10 bps).
        slippage: Round-trip slippage fraction (default 5 bps).
        n_splits: Number of walk-forward folds (default 5).
        test_size: Fraction of total bars reserved for each test window
            (default 0.2).
        gap: Bars skipped between train end and test start (default 0).
        window: ``"expanding"`` or ``"rolling"`` (default ``"expanding"``).
        min_train_size: Minimum bars required in the training window.
        db_path: Path to the SQLite results database.  Pass ``None`` to skip
            persistence (default ``None``).
        tearsheet_dir: Directory to save tearsheet images.  Pass ``None`` to
            skip tearsheet generation (default ``None``).
        tearsheet_fmt: ``"png"``, ``"pdf"``, or ``"both"`` (default ``"both"``).

    Example::

        runner = EvaluationRunner(n_splits=5, db_path="results.db")
        result = runner.evaluate(TrendFollowing, bars, symbol="ES.c.0")
    """

    def __init__(
        self,
        *,
        initial_capital: float = 1_000_000.0,
        fees: float = 0.001,
        slippage: float = 0.0005,
        n_splits: int = 5,
        test_size: float | int = 0.2,
        gap: int = 0,
        window: str = "expanding",
        min_train_size: int | None = None,
        db_path: Path | str | None = None,
        tearsheet_dir: Path | str | None = None,
        tearsheet_fmt: str = "both",
    ) -> None:
        self.initial_capital = initial_capital
        self.fees = fees
        self.slippage = slippage
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.window = window
        self.min_train_size = min_train_size
        self._db_path = Path(db_path) if db_path is not None else None
        self._tearsheet_dir = Path(tearsheet_dir) if tearsheet_dir is not None else None
        self._tearsheet_fmt = tearsheet_fmt

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        strategy_class: type[Strategy],
        bars: pl.DataFrame,
        *,
        params: _ParamSearch = None,
        symbol: str = "UNKNOWN",
        experiment_name: str | None = None,
    ) -> EvaluationResult:
        """Run walk-forward evaluation for a single-asset strategy.

        Args:
            strategy_class: Any class implementing the
                :class:`~snippy_scales.strategies.trend.Strategy` ABC.
            bars: OHLCV bar DataFrame for one instrument.
            params: Parameter search specification.  Accepts:

                * ``None`` — evaluates with the strategy's default parameters.
                * ``dict`` — evaluates with a single fixed parameter set.
                * :class:`~snippy_scales.evaluation.sweep.ParameterGrid` — exhaustive grid.
                * :class:`~snippy_scales.evaluation.sweep.RandomSearch` — random sampling.
                * :class:`~snippy_scales.evaluation.sweep.OptunaSearch` — Bayesian optimisation.

            symbol: Instrument name for labelling (default ``"UNKNOWN"``).
            experiment_name: Human-readable name for the DB entry.  Defaults
                to ``"<StrategyClass>_<symbol>"``.

        Returns:
            :class:`~snippy_scales.evaluation.results.EvaluationResult` containing
            all sweep results and convenience accessors for the best parameters.
        """
        name = experiment_name or f"{strategy_class.__name__}_{symbol}"
        splitter = self._make_splitter()
        folds = splitter.split(bars)

        if not folds:
            raise ValueError(
                f"WalkForwardSplit produced no folds for {len(bars)}-bar DataFrame.  "
                "Try reducing n_splits or test_size."
            )

        sweep_results = self._run_sweep(
            strategy_class=strategy_class,
            params=params,
            folds=folds,
            symbol=symbol,
            is_multi_asset=False,
        )

        result = EvaluationResult(
            experiment_name=name,
            strategy_class=f"{strategy_class.__module__}.{strategy_class.__name__}",
            symbols=[symbol],
            sweep_results=sweep_results,
        )

        self._maybe_persist(result)
        self._maybe_tearsheet(result, name)
        return result

    def evaluate_multi_asset(
        self,
        strategy_class: type[MultiAssetStrategy],
        multi_bars: dict[str, pl.DataFrame],
        *,
        params: _ParamSearch = None,
        experiment_name: str | None = None,
    ) -> EvaluationResult:
        """Run walk-forward evaluation for a multi-asset strategy.

        Args:
            strategy_class: Any class implementing the
                :class:`~snippy_scales.strategies.momentum_cs.MultiAssetStrategy` ABC.
            multi_bars: Mapping of ``symbol → bar DataFrame``.  All DataFrames
                must have the same number of rows.
            params: Parameter search specification (same as :meth:`evaluate`).
            experiment_name: Human-readable name for the DB entry.

        Returns:
            :class:`~snippy_scales.evaluation.results.EvaluationResult`.
        """
        symbols = list(multi_bars.keys())
        name = experiment_name or f"{strategy_class.__name__}_{'_'.join(symbols)}"
        splitter = self._make_splitter()
        multi_folds = splitter.split_multi(multi_bars)

        if not multi_folds:
            raise ValueError(
                "WalkForwardSplit produced no folds for the provided multi-asset data.  "
                "Try reducing n_splits or test_size."
            )

        sweep_results = self._run_sweep_multi(
            strategy_class=strategy_class,
            params=params,
            multi_folds=multi_folds,
        )

        result = EvaluationResult(
            experiment_name=name,
            strategy_class=f"{strategy_class.__module__}.{strategy_class.__name__}",
            symbols=symbols,
            sweep_results=sweep_results,
        )

        self._maybe_persist(result)
        self._maybe_tearsheet(result, name)
        return result

    # ── Internal: sweep orchestration ─────────────────────────────────────────

    def _run_sweep(
        self,
        *,
        strategy_class: type,
        params: _ParamSearch,
        folds: list[Any],
        symbol: str,
        is_multi_asset: bool,
    ) -> list[SweepResult]:
        """Iterate over parameter sets and collect SweepResult objects."""
        param_sets = self._resolve_param_sets(params, strategy_class, folds, symbol)
        sweep_results: list[SweepResult] = []

        for param_set in param_sets:
            logger.debug("Evaluating params: %s", param_set)
            fold_results = self._run_folds(
                strategy_class=strategy_class,
                param_set=param_set,
                folds=folds,
                symbol=symbol,
            )
            sweep_results.append(SweepResult(params=param_set, folds=fold_results))

        return sweep_results

    def _run_sweep_multi(
        self,
        *,
        strategy_class: type,
        params: _ParamSearch,
        multi_folds: list[tuple[dict[str, Any], dict[str, Any]]],
    ) -> list[SweepResult]:
        """Iterate over parameter sets for multi-asset strategies."""
        param_sets = self._resolve_param_sets_simple(params)
        sweep_results: list[SweepResult] = []

        for param_set in param_sets:
            logger.debug("Evaluating params: %s", param_set)
            fold_results: list[FoldResult] = []

            for fold_idx, (train_bars, test_bars) in enumerate(multi_folds):
                try:
                    train_result = self._run_basket(strategy_class, param_set, train_bars)
                    test_result = self._run_basket(strategy_class, param_set, test_bars)
                    fold_results.append(
                        FoldResult(
                            fold_idx=fold_idx,
                            params=param_set,
                            train_metrics=train_result.metrics,
                            test_metrics=test_result.metrics,
                            train_equity_curve=train_result.equity_curve,
                            test_equity_curve=test_result.equity_curve,
                            train_start="",
                            train_end="",
                            test_start="",
                            test_end="",
                        )
                    )
                except Exception:
                    logger.warning(
                        "Fold %d failed for params %s", fold_idx, param_set, exc_info=True
                    )

            sweep_results.append(SweepResult(params=param_set, folds=fold_results))

        return sweep_results

    def _run_folds(
        self,
        *,
        strategy_class: type,
        param_set: dict[str, Any],
        folds: list[Any],
        symbol: str,
    ) -> list[FoldResult]:
        """Run a single param set across all folds."""
        fold_results: list[FoldResult] = []

        for fold in folds:
            try:
                train_result = self._run_single(strategy_class, param_set, fold.train, symbol)
                test_result = self._run_single(strategy_class, param_set, fold.test, symbol)
                fold_results.append(
                    FoldResult(
                        fold_idx=fold.fold_idx,
                        params=param_set,
                        train_metrics=train_result.metrics,
                        test_metrics=test_result.metrics,
                        train_equity_curve=train_result.equity_curve,
                        test_equity_curve=test_result.equity_curve,
                        train_start=fold.train_start,
                        train_end=fold.train_end,
                        test_start=fold.test_start,
                        test_end=fold.test_end,
                    )
                )
            except Exception:
                logger.warning(
                    "Fold %d failed for params %s", fold.fold_idx, param_set, exc_info=True
                )

        return fold_results

    # ── Internal: individual run helpers ──────────────────────────────────────

    def _run_single(
        self,
        strategy_class: type,
        param_set: dict[str, Any],
        bars: pl.DataFrame,
        symbol: str,
    ) -> Any:
        strategy = strategy_class(**param_set)
        runner = BacktestRunner(
            initial_capital=self.initial_capital,
            fees=self.fees,
            slippage=self.slippage,
        )
        return runner.run(strategy, bars, symbol=symbol)

    def _run_basket(
        self,
        strategy_class: type,
        param_set: dict[str, Any],
        multi_bars: dict[str, Any],
    ) -> Any:
        strategy = strategy_class(**param_set)
        runner = BasketRunner(
            initial_capital=self.initial_capital,
            fees=self.fees,
            slippage=self.slippage,
        )
        return runner.run(strategy, multi_bars)

    # ── Internal: parameter resolution ────────────────────────────────────────

    def _resolve_param_sets(
        self,
        params: _ParamSearch,
        strategy_class: type,
        folds: list[Any],
        symbol: str,
    ) -> list[dict[str, Any]]:
        """Handle all param search types, including Optuna's special flow."""
        from snippy_scales.evaluation.sweep import OptunaSearch

        if isinstance(params, OptunaSearch):
            return self._run_optuna(params, strategy_class, folds, symbol)

        return self._resolve_param_sets_simple(params)

    def _resolve_param_sets_simple(self, params: _ParamSearch) -> list[dict[str, Any]]:
        if params is None:
            return [{}]
        if isinstance(params, dict):
            return [params]
        # ParameterGrid or RandomSearch — both are iterables
        return list(params)

    def _run_optuna(
        self,
        search: Any,
        strategy_class: type,
        folds: list[Any],
        symbol: str,
    ) -> list[dict[str, Any]]:
        """Run Optuna optimisation and return the ordered list of param sets tried."""
        import optuna  # type: ignore[import-not-found]

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        tried_params: list[dict[str, Any]] = []

        def objective(trial: Any) -> float:
            param_set = search.suggest(trial)
            tried_params.append(param_set)
            fold_results = self._run_folds(
                strategy_class=strategy_class,
                param_set=param_set,
                folds=folds,
                symbol=symbol,
            )
            if not fold_results:
                return float("-inf")
            sharpes = [f.test_metrics.sharpe_ratio for f in fold_results]
            return float(np.mean(sharpes))

        sampler = optuna.samplers.TPESampler(seed=search.seed)
        study = optuna.create_study(direction=search.direction, sampler=sampler)
        study.optimize(objective, n_trials=search.n_trials, show_progress_bar=False)

        return tried_params

    # ── Internal: splitter factory ─────────────────────────────────────────────

    def _make_splitter(self) -> WalkForwardSplit:
        return WalkForwardSplit(
            n_splits=self.n_splits,
            test_size=self.test_size,
            gap=self.gap,
            window=self.window,
            min_train_size=self.min_train_size,
        )

    # ── Internal: persistence & tearsheet ─────────────────────────────────────

    def _maybe_persist(self, result: EvaluationResult) -> None:
        if self._db_path is None:
            return
        from snippy_scales.evaluation.database import SQLiteStore

        store = SQLiteStore(self._db_path)
        exp_id = store.save_evaluation(
            result,
            config={
                "n_splits": self.n_splits,
                "window": self.window,
                "test_size": self.test_size,
                "gap": self.gap,
                "initial_capital": self.initial_capital,
                "fees": self.fees,
                "slippage": self.slippage,
            },
        )
        logger.info("Saved evaluation to %s (experiment_id=%d)", self._db_path, exp_id)

    def _maybe_tearsheet(self, result: EvaluationResult, name: str) -> None:
        if self._tearsheet_dir is None:
            return
        safe_name = name.replace(" ", "_").replace("/", "-")
        output_path = self._tearsheet_dir / f"{safe_name}_tearsheet"
        gen = TearsheetGenerator()
        try:
            paths = gen.generate_walk_forward(
                result,
                output_path=output_path,
                fmt=self._tearsheet_fmt,
            )
            for p in paths:
                logger.info("Tearsheet saved: %s", p)
        except Exception:
            logger.warning("Tearsheet generation failed", exc_info=True)
