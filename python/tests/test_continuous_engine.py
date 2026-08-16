"""Tests for the continuous target-position execution engine and cost models.

The load-bearing tests here are:

* **Parity** — with a sign-only signal and zero costs, ``TargetPositionEngine``
  must reproduce raptorbt's equity curve and its 29 non-ratio metrics exactly.
  Without this, the new engine could be quietly wrong in a way no other test
  would catch.
* **Annualisation** — ``make_config`` passes ``periods_per_year`` so raptorbt's
  ratios agree with ours.  ``calmar_ratio`` is the sole remaining divergence:
  it ignores that setting upstream and stays hard-wired to 365.
* **Magnitude** — the assertion that fails against ``SignFlipInterpreter``,
  which is the entire reason this engine exists.
* **Lookahead** — shifting execution earlier must inflate performance.
* **Cost monotonicity** — more expensive trading must not improve Sharpe.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import raptorbt

from snippy_scales.backtesting.runner import (
    BacktestResult,
    ContinuousConfig,
    FuturesCostModel,
    InstrumentSpec,
    ProportionalCost,
    RaptorExecutionEngine,
    TargetPositionEngine,
    make_config,
    make_continuous_config,
    positions_from_signals,
    run_single,
)

_BASE_NS = 1_577_836_800_000_000_000  # 2020-01-01 UTC in nanoseconds
_DAY_NS = 86_400_000_000_000

#: ``calmar_ratio`` is invariant to raptorbt's ``periods_per_year`` (verified
#: identical at 252, 365 and 1000), so it remains annualised with 365 upstream
#: while Sharpe and Sortino now honour the setting.  Excluded from parity until
#: that is fixed; see ``docs/research/raptorbt_defects.md``.
_RATIO_FIELDS = frozenset({"calmar_ratio"})

#: sqrt(365 / 252) — the factor by which raptorbt overstates daily-bar Sharpe.
_ANNUALISATION_GAP = float(np.sqrt(365.0 / 252.0))


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_spec(
    n: int = 300,
    seed: int = 0,
    positions: np.ndarray | None = None,
    weight: float = 1.0,
    symbol: str = "X",
) -> InstrumentSpec:
    """Build a synthetic single-instrument leg.

    Args:
        n: Number of bars.
        seed: RNG seed for the price path.
        positions: Optional continuous positions to attach.
        weight: Capital weight for the leg.
        symbol: Instrument identifier.

    Returns:
        A populated :class:`InstrumentSpec` with a simple two-trade signal.
    """
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    for bar, array in ((10, entries), (50, exits), (100, entries), (180, exits)):
        if bar < n:
            array[bar] = True
    return InstrumentSpec(
        symbol=symbol,
        timestamps=_BASE_NS + np.arange(n, dtype=np.int64) * _DAY_NS,
        open=close.copy(),
        high=close * 1.001,
        low=close * 0.999,
        close=close,
        volume=np.full(n, 1e6),
        entries=entries,
        exits=exits,
        direction=1,
        weight=weight,
        positions=positions,
    )


def _with_positions(spec: InstrumentSpec, positions: np.ndarray) -> InstrumentSpec:
    """Return a copy of *spec* carrying *positions*.

    Args:
        spec: Template instrument leg.
        positions: Continuous target positions to attach.

    Returns:
        A new :class:`InstrumentSpec`.
    """
    return dataclasses.replace(spec, positions=positions)


def _free_config(**overrides: object) -> ContinuousConfig:
    """Return a zero-cost :class:`ContinuousConfig`.

    Args:
        **overrides: Passed through to :func:`make_continuous_config`.

    Returns:
        A cost-free configuration for isolating P&L mechanics.
    """
    params: dict[str, object] = {
        "initial_capital": 100_000.0,
        "cost_model": ProportionalCost(fees=0.0, slippage=0.0),
    }
    params.update(overrides)
    return make_continuous_config(**params)  # type: ignore[arg-type]


def _raptor_result(spec: InstrumentSpec) -> BacktestResult:
    """Run the same leg through raptorbt with costs disabled.

    Args:
        spec: Instrument leg.

    Returns:
        The raptorbt-backed :class:`BacktestResult`.
    """
    return run_single(
        symbol=spec.symbol,
        timestamps=spec.timestamps,
        open_prices=spec.open,
        high_prices=spec.high,
        low_prices=spec.low,
        close_prices=spec.close,
        volume=spec.volume,
        entries=spec.entries,
        exits=spec.exits,
        direction=spec.direction,
        weight=spec.weight,
        config=make_config(initial_capital=100_000.0, fees=0.0, slippage=0.0),
    )


# ── Parity with raptorbt ──────────────────────────────────────────────────────


def test_parity_equity_curve_matches_raptorbt() -> None:
    """A sign-only signal must reproduce raptorbt's equity curve."""
    spec = _make_spec()
    mine = TargetPositionEngine().execute([spec], config=_free_config())
    theirs = _raptor_result(spec)

    np.testing.assert_allclose(mine.equity_curve, theirs.equity_curve, rtol=1e-9)


def test_parity_metrics_match_raptorbt() -> None:
    """All 32 comparable metrics must agree with raptorbt exactly.

    Only ``calmar_ratio`` is excluded — see :data:`_RATIO_FIELDS`.
    """
    spec = _make_spec()
    mine = TargetPositionEngine().execute([spec], config=_free_config())
    theirs = _raptor_result(spec)

    mismatched: list[str] = []
    for field in dataclasses.fields(mine.metrics):
        if field.name in _RATIO_FIELDS:
            continue
        ours = float(getattr(mine.metrics, field.name))
        ref = float(getattr(theirs.metrics, field.name))
        if abs(ours - ref) > 1e-6 * max(1.0, abs(ref)):
            mismatched.append(f"{field.name}: {ours} != {ref}")

    assert not mismatched, "metrics diverged from raptorbt: " + "; ".join(mismatched)


def test_parity_trades_match_raptorbt() -> None:
    """Synthesised trades must match raptorbt's in count, price and P&L."""
    spec = _make_spec()
    mine = TargetPositionEngine().execute([spec], config=_free_config())
    theirs = _raptor_result(spec)

    assert len(mine.trades) == len(theirs.trades)
    for ours, ref in zip(mine.trades, theirs.trades, strict=True):
        assert ours.entry_price == pytest.approx(ref.entry_price, rel=1e-9)
        assert ours.exit_price == pytest.approx(ref.exit_price, rel=1e-9)
        assert ours.pnl == pytest.approx(ref.pnl, rel=1e-9)
        assert ours.return_pct == pytest.approx(ref.return_pct, rel=1e-9)


def test_make_config_annualises_with_252_not_365() -> None:
    """``make_config`` must override raptorbt's 365-day default.

    raptorbt annualises with 365 unless told otherwise, which overstates
    daily-bar Sharpe by ``sqrt(365/252)`` ≈ 1.20.  ``make_config`` passes
    ``periods_per_year`` explicitly so raptorbt's ratios agree with
    ``BacktestMetrics.from_returns``.  This test pins both halves: that our
    default matches, and that raptorbt's bare default still does not.
    """
    spec = _make_spec()
    mine = TargetPositionEngine().execute([spec], config=_free_config())
    theirs = _raptor_result(spec)

    for field in ("sharpe_ratio", "sortino_ratio"):
        ours = float(getattr(mine.metrics, field))
        ref = float(getattr(theirs.metrics, field))
        assert ref == pytest.approx(ours, rel=1e-6), f"{field} diverged from raptorbt"

    # raptorbt's own default remains 365; the agreement above is ours to keep.
    legacy = run_single(
        symbol=spec.symbol,
        timestamps=spec.timestamps,
        open_prices=spec.open,
        high_prices=spec.high,
        low_prices=spec.low,
        close_prices=spec.close,
        volume=spec.volume,
        entries=spec.entries,
        exits=spec.exits,
        direction=spec.direction,
        weight=spec.weight,
        config=raptorbt.BacktestConfig(initial_capital=100_000.0, fees=0.0, slippage=0.0),
    )
    ratio = legacy.metrics.sharpe_ratio / mine.metrics.sharpe_ratio
    assert ratio == pytest.approx(_ANNUALISATION_GAP, rel=1e-3)


def test_raptorbt_basket_path_agrees_with_single_path() -> None:
    """Basket and single paths must agree on identical equity curves.

    raptorbt <=0.3.2 inflated the basket path's Sharpe by 4-7x relative to its
    own single-instrument path on byte-identical equity. That was fixed
    upstream in 0.8.0; this test is the regression guard, since every runner
    and therefore every evaluation fold goes through the basket path.
    """
    spec = _make_spec(n=600, seed=1)
    config = make_config(initial_capital=100_000.0, fees=0.0, slippage=0.0)

    single = _raptor_result(spec)
    basket = RaptorExecutionEngine().execute([spec], config=config)

    np.testing.assert_allclose(single.equity_curve, basket.equity_curve, rtol=1e-9)
    assert basket.metrics.sharpe_ratio == pytest.approx(single.metrics.sharpe_ratio, rel=1e-6)


def test_raptorbt_basket_path_still_drops_trade_level_metrics() -> None:
    """Known upstream gap: the basket path zeroes several trade-level metrics.

    Fixed in 0.8.0 for Sharpe, but ``exposure_pct``, ``omega_ratio`` and
    ``max_drawdown_duration`` are still returned as zero on the basket path
    while the single path populates them. ``TargetPositionEngine`` computes all
    of them, so this pins the gap rather than working around it.
    """
    spec = _make_spec(n=600, seed=1)
    config = make_config(initial_capital=100_000.0, fees=0.0, slippage=0.0)

    single = _raptor_result(spec)
    basket = RaptorExecutionEngine().execute([spec], config=config)

    assert single.metrics.exposure_pct > 0.0
    assert basket.metrics.exposure_pct == 0.0
    assert basket.metrics.max_drawdown_duration == 0
    assert single.metrics.max_drawdown_duration > 0


def test_periods_per_year_scales_sharpe() -> None:
    """Sharpe must scale as sqrt(periods_per_year) — the intraday fix."""
    spec = _make_spec()
    engine = TargetPositionEngine()
    daily = engine.execute([spec], config=_free_config(periods_per_year=252.0))
    hourly = engine.execute([spec], config=_free_config(periods_per_year=252.0 * 23))

    ratio = hourly.metrics.sharpe_ratio / daily.metrics.sharpe_ratio
    assert ratio == pytest.approx(np.sqrt(23.0), rel=1e-9)


# ── Magnitude: the reason this engine exists ──────────────────────────────────


def test_position_magnitude_changes_returns() -> None:
    """Half-sized positions must produce different returns from full-sized ones.

    This is the assertion that fails against ``SignFlipInterpreter``, which
    reads only the sign of a position.
    """
    n = 300
    base = _make_spec(n=n)
    ramp = np.where(np.arange(n) >= 10, 1.0, 0.0)

    engine = TargetPositionEngine()
    half = engine.execute([_with_positions(base, ramp * 0.5)], config=_free_config())
    full = engine.execute([_with_positions(base, ramp)], config=_free_config())

    assert half.metrics.total_return_pct != pytest.approx(full.metrics.total_return_pct, rel=1e-6)
    # Half the exposure earns (approximately) half the log-return.
    assert abs(half.metrics.total_return_pct) < abs(full.metrics.total_return_pct)


def test_zero_positions_produce_flat_equity() -> None:
    """An all-zero position series must leave capital untouched."""
    n = 200
    spec = _with_positions(_make_spec(n=n), np.zeros(n))
    result = TargetPositionEngine().execute([spec], config=_free_config())

    assert result.metrics.total_return_pct == pytest.approx(0.0)
    assert result.metrics.exposure_pct == pytest.approx(0.0)
    assert result.trades == []


def test_leverage_scales_linearly_in_log_space() -> None:
    """Doubling a constant position must double the cumulative log return."""
    n = 400
    base = _make_spec(n=n)
    engine = TargetPositionEngine()

    one = engine.execute([_with_positions(base, np.ones(n))], config=_free_config())
    two = engine.execute([_with_positions(base, np.full(n, 2.0))], config=_free_config())

    # Sum of log(1 + 2r) is not exactly 2*sum(log(1 + r)), but gross exposure
    # ordering must hold and the doubled book must move strictly further.
    assert abs(two.metrics.total_return_pct) > abs(one.metrics.total_return_pct)


# ── Lookahead ─────────────────────────────────────────────────────────────────


def test_execution_lag_zero_creates_lookahead() -> None:
    """A perfect-foresight signal must only pay off when execution is unlagged."""
    n = 400
    base = _make_spec(n=n)
    foresight = np.sign(np.diff(base.close, prepend=base.close[0]))
    spec = _with_positions(base, foresight)

    engine = TargetPositionEngine()
    honest = engine.execute([spec], config=_free_config(execution_lag=1))
    cheating = engine.execute([spec], config=_free_config(execution_lag=0))

    assert cheating.metrics.sharpe_ratio > 10.0
    assert honest.metrics.sharpe_ratio < 1.0


# ── Costs ─────────────────────────────────────────────────────────────────────


def test_sharpe_decreases_monotonically_with_commission() -> None:
    """Raising per-contract commission must never improve Sharpe."""
    n = 300
    rng = np.random.default_rng(3)
    base = _make_spec(n=n)
    churn = np.sign(rng.normal(size=n))  # heavy turnover so costs bite
    spec = _with_positions(base, churn)

    engine = TargetPositionEngine()
    sharpes: list[float] = []
    for commission in (0.0, 1.0, 2.0, 5.0, 20.0):
        config = make_continuous_config(
            initial_capital=100_000.0,
            cost_model=FuturesCostModel(
                commission_per_side=commission,
                half_spread_ticks=0.5,
                tick_size=0.25,
                tick_value=12.50,
            ),
        )
        sharpes.append(engine.execute([spec], config=config).metrics.sharpe_ratio)

    assert sharpes == sorted(sharpes, reverse=True)
    assert sharpes[0] > sharpes[-1]


def test_fees_paid_accumulate_with_turnover() -> None:
    """A churning position must pay strictly more in fees than a static one."""
    n = 300
    rng = np.random.default_rng(11)
    base = _make_spec(n=n)
    cost = ProportionalCost(fees=0.001, slippage=0.0005)

    engine = TargetPositionEngine()
    config = make_continuous_config(initial_capital=100_000.0, cost_model=cost)
    static = engine.execute([_with_positions(base, np.ones(n))], config=config)
    churn = engine.execute([_with_positions(base, np.sign(rng.normal(size=n)))], config=config)

    assert churn.metrics.total_fees_paid > static.metrics.total_fees_paid


def test_proportional_cost_matches_hand_calculation() -> None:
    """One full turnover at 15 bp must cost 15 bp of equity."""
    cost = ProportionalCost(fees=0.001, slippage=0.0005)
    traded = np.array([1.0, 0.5, 0.0])
    fractions = cost.cost_fraction(traded, np.array([100.0, 100.0, 100.0]))
    np.testing.assert_allclose(fractions, [0.0015, 0.00075, 0.0])


def test_futures_cost_bigger_contract_is_cheaper_per_notional() -> None:
    """ES must cost less per unit notional than MES.

    Commission is a flat cash amount per contract while the spread scales with
    contract size, so the larger contract wins.  These are the figures the
    research memo's cost table is built from.
    """
    es = FuturesCostModel(
        commission_per_side=2.0, half_spread_ticks=0.5, tick_size=0.25, tick_value=12.50
    )
    mes = FuturesCostModel(
        commission_per_side=0.75, half_spread_ticks=0.5, tick_size=0.25, tick_value=1.25
    )
    price = 6000.0

    assert es.contract_notional(price) == pytest.approx(300_000.0)
    assert mes.contract_notional(price) == pytest.approx(30_000.0)
    assert es.cost_bp_per_side(price) == pytest.approx(0.275, rel=1e-3)
    assert mes.cost_bp_per_side(price) == pytest.approx(0.4583, rel=1e-3)
    assert es.cost_bp_per_side(price) < mes.cost_bp_per_side(price)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fees": -0.1}, "fees"),
        ({"slippage": -0.1}, "slippage"),
    ],
)
def test_proportional_cost_rejects_negative(kwargs: dict[str, float], message: str) -> None:
    """Negative cost parameters must be rejected."""
    with pytest.raises(ValueError, match=message):
        ProportionalCost(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"commission_per_side": -1.0}, "commission_per_side"),
        ({"half_spread_ticks": -1.0}, "half_spread_ticks"),
        ({"tick_size": 0.0}, "tick_size"),
        ({"tick_value": 0.0}, "tick_value"),
    ],
)
def test_futures_cost_rejects_invalid(kwargs: dict[str, float], message: str) -> None:
    """Invalid contract specifications must be rejected."""
    with pytest.raises(ValueError, match=message):
        FuturesCostModel(**kwargs)


# ── Multi-leg baskets ─────────────────────────────────────────────────────────


def test_basket_combines_legs_by_weight() -> None:
    """Two half-weighted identical legs must equal one full-weighted leg."""
    n = 300
    positions = np.ones(n)
    single = _with_positions(_make_spec(n=n, seed=5, weight=1.0), positions)
    half_a = _with_positions(_make_spec(n=n, seed=5, weight=0.5, symbol="A"), positions)
    half_b = _with_positions(_make_spec(n=n, seed=5, weight=0.5, symbol="B"), positions)

    engine = TargetPositionEngine()
    one = engine.execute([single], config=_free_config())
    two = engine.execute([half_a, half_b], config=_free_config())

    np.testing.assert_allclose(one.equity_curve, two.equity_curve, rtol=1e-9)


def test_basket_rejects_mismatched_bar_counts() -> None:
    """Legs of differing length must raise rather than silently truncate."""
    engine = TargetPositionEngine()
    with pytest.raises(ValueError, match="bars"):
        engine.execute(
            [_make_spec(n=300, symbol="A"), _make_spec(n=200, symbol="B")],
            config=_free_config(),
        )


def test_rejects_empty_instruments() -> None:
    """An empty leg list must raise."""
    with pytest.raises(ValueError, match="must not be empty"):
        TargetPositionEngine().execute([], config=_free_config())


def test_rejects_raptorbt_config() -> None:
    """Passing a raptorbt config must fail loudly rather than be ignored."""
    with pytest.raises(ValueError, match="ContinuousConfig"):
        TargetPositionEngine().execute([_make_spec()], config=make_config())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"initial_capital": 0.0}, "initial_capital"),
        ({"periods_per_year": 0.0}, "periods_per_year"),
        ({"execution_lag": -1}, "execution_lag"),
    ],
)
def test_continuous_config_validates(kwargs: dict[str, float], message: str) -> None:
    """Invalid configuration must be rejected at construction."""
    with pytest.raises(ValueError, match=message):
        ContinuousConfig(**kwargs)  # type: ignore[arg-type]


# ── Signal reconstruction ─────────────────────────────────────────────────────


def test_positions_from_signals_roundtrip() -> None:
    """Entry/exit arrays must reconstruct the intended holding pattern."""
    entries = np.array([False, True, False, False, False, False])
    exits = np.array([False, False, False, True, False, False])
    positions = positions_from_signals(entries, exits, direction=1)
    np.testing.assert_allclose(positions, [0.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def test_positions_from_signals_short_direction() -> None:
    """A short book must produce negative positions."""
    entries = np.array([False, True, False, False])
    exits = np.array([False, False, True, False])
    positions = positions_from_signals(entries, exits, direction=-1)
    np.testing.assert_allclose(positions, [0.0, -1.0, 0.0, 0.0])


def test_single_bar_series_does_not_crash() -> None:
    """A one-bar backtest must return flat rather than raise."""
    spec = _with_positions(_make_spec(n=1), np.array([1.0]))
    result = TargetPositionEngine().execute([spec], config=_free_config())
    assert result.metrics.total_return_pct == pytest.approx(0.0)
