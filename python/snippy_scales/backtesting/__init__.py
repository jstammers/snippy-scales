"""Backtesting layer using raptorbt as the high-performance execution engine.

This package translates strategy signal output (signed-float positions) into
raptorbt's entry/exit format and provides clean, typed result objects for
downstream analysis and reporting.

Typical usage::

    from snippy_scales.backtesting.runner import BacktestRunner, make_config
    from snippy_scales.strategies.momentum_ts import TimeSeriesMomentum

    strategy = TimeSeriesMomentum(lookback=252)
    runner = BacktestRunner(initial_capital=1_000_000.0, fees=0.001)
    result = runner.run(strategy, bars, symbol="ES.c.0")
    print(result.metrics)
"""
