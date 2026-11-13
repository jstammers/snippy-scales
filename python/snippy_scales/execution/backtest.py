"""``run_backtest``: one research strategy, one Nautilus ``BacktestEngine`` run.

Market data is loaded once into :class:`MarketData` (from the catalog with
:func:`load_market_data`, or from polars frames with :meth:`MarketData.from_frames`)
and can then back any number of runs over different windows, which is what
walk-forward folds need.

Signals are computed on all history up to ``end``, so indicators are warm on
the first traded bar; only bars with ``start <= ts_init <= end`` are replayed.

Costs map onto Nautilus models: ``commission_per_contract`` ->
``PerContractFeeModel``, ``commission_rate`` (fraction of notional) ->
``MakerTakerFeeModel``, ``slippage_prob`` -> ``DefaultFillModel`` (one tick of
slippage on bar data, applied with that probability).
"""

from __future__ import annotations

import bisect
import math
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import polars as pl
from nautilus_trader.backtest import BacktestEngine, BacktestEngineConfig
from nautilus_trader.common import LoggerConfig, LogLevel
from nautilus_trader.execution import (
    DefaultFillModel,
    FixedFeeModel,
    MakerTakerFeeModel,
    PerContractFeeModel,
)
from nautilus_trader.model import AccountType, BarType, Money, NautilusDataType, OmsType, Venue

from snippy_scales.data import catalog as cat
from snippy_scales.data import futures
from snippy_scales.data.futures import RollRule, RollSchedule
from snippy_scales.execution.strategy import SignalStrategy

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    import pandas as pd
    from nautilus_trader.model import Bar, Currency, FuturesContract

    from snippy_scales.strategies.momentum_cs import MultiAssetStrategy
    from snippy_scales.strategies.trend import Strategy

    TimeBound = str | dt.date | dt.datetime | int | None

#: Seed for the fill model, so slippage is reproducible run to run.
_FILL_SEED = 42

#: Key of the annualised Sharpe in Nautilus' return statistics (daily returns).
SHARPE = "Sharpe Ratio (252 days)"


@dataclass(frozen=True)
class Costs:
    """Transaction costs for a backtest.

    Attributes:
        commission_per_contract: Commission per contract (or share) per fill,
            in the instrument's quote currency.
        commission_rate: Commission as a fraction of fill notional
            (``0.001`` = 10 bp). Mutually exclusive with ``commission_per_contract``.
        slippage_prob: Probability that a market order fills one tick worse.
    """

    commission_per_contract: float = 0.0
    commission_rate: float = 0.0
    slippage_prob: float = 0.0

    def __post_init__(self) -> None:
        if self.commission_per_contract < 0 or self.commission_rate < 0:
            raise ValueError("commissions must be non-negative")
        if self.commission_per_contract and self.commission_rate:
            raise ValueError("set commission_per_contract or commission_rate, not both")
        if not 0.0 <= self.slippage_prob <= 1.0:
            raise ValueError(f"slippage_prob must be in [0, 1], got {self.slippage_prob}")

    def fee_model(self, currency: Currency) -> Any:  # noqa: ANN401 — Nautilus FeeModel
        if self.commission_per_contract:
            return PerContractFeeModel(Money(self.commission_per_contract, currency))
        if self.commission_rate:
            return MakerTakerFeeModel(Decimal(0), Decimal(repr(self.commission_rate)))
        return FixedFeeModel(Money(0, currency))

    def fill_model(self) -> DefaultFillModel | None:
        if not self.slippage_prob:
            return None
        return DefaultFillModel(1.0, self.slippage_prob, _FILL_SEED)


def _ts_column(frame: pl.DataFrame, column: str = "ts_init") -> pl.Series:
    ts = frame[column]
    return ts if ts.dtype == pl.Int64 else ts.dt.cast_time_unit("ns").dt.epoch("ns")


@dataclass(frozen=True)
class MarketData:
    """Bars and instruments for one or more symbols, ready to backtest.

    A *symbol* is what a strategy sees: an equity (``AAPL``), a continuous series
    (``ES.c.0``) or a parent futures symbol (``ES.FUT``) that rolls through real
    outrights. Every symbol has a :class:`~snippy_scales.data.futures.RollSchedule`
    saying which instrument it trades at each bar close; for anything but a
    parent symbol that is a single instrument.

    Attributes:
        frames: ``symbol -> bars`` (polars) for signal computation; back-adjusted
            for parent symbols.
        bars: ``instrument_id -> Nautilus bars`` sorted by ``ts_init``, for replay.
        instruments: ``instrument_id -> Nautilus instrument``, all at one venue.
        bar_types: ``instrument_id -> BarType`` string.
        schedules: ``symbol -> RollSchedule``.
    """

    frames: dict[str, pl.DataFrame]
    bars: dict[str, list[Bar]] = field(repr=False)
    instruments: dict[str, Any] = field(repr=False)
    bar_types: dict[str, str] = field(repr=False)
    schedules: dict[str, RollSchedule]

    @classmethod
    def from_frames(
        cls,
        frames: Mapping[str, pl.DataFrame],
        instruments: Mapping[str, Any],
        schema: str = "ohlcv-1d",
    ) -> MarketData:
        """Build from polars frames (``ts_event`` open, ``ts_init`` close, OHLCV),
        one instrument per symbol.

        Raises:
            ValueError: If symbols differ, or a frame's ``ts_init`` is not
                ``ts_event`` plus the *schema* bar length.
        """
        if set(frames) != set(instruments):
            raise ValueError("frames and instruments must cover the same symbols")
        bars: dict[str, list[Bar]] = {}
        bar_types: dict[str, str] = {}
        by_id: dict[str, Any] = {}
        schedules: dict[str, RollSchedule] = {}
        for symbol, frame in frames.items():
            instrument = instruments[symbol]
            iid = str(instrument.id)
            bar_types[iid] = cat.bar_type(symbol, str(instrument.id.venue), schema)
            bars[iid] = _frame_bars(frame, instrument, bar_types[iid], schema, symbol)
            by_id[iid] = instrument
            schedules[symbol] = RollSchedule.single(symbol, iid)
        return cls(
            frames={s: f.sort("ts_init") for s, f in frames.items()},
            bars=bars,
            instruments=by_id,
            bar_types=bar_types,
            schedules=schedules,
        )

    @classmethod
    def from_outrights(
        cls,
        symbol: str,
        frames: Mapping[str, pl.DataFrame],
        contracts: Sequence[FuturesContract],
        schema: str = "ohlcv-1d",
        rule: RollRule | None = None,
    ) -> MarketData:
        """Build a rolled futures symbol from outright frames (keyed by instrument ID)."""
        by_id = {str(c.id): c for c in contracts}
        schedule = futures.roll_schedule(symbol, contracts, frames, rule)
        bars: dict[str, list[Bar]] = {}
        bar_types: dict[str, str] = {}
        for iid in schedule.instrument_ids:
            contract = by_id[iid]
            bar_types[iid] = cat.bar_type(str(contract.raw_symbol), str(contract.id.venue), schema)
            bars[iid] = _frame_bars(frames[iid], contract, bar_types[iid], schema, iid)
        return cls(
            frames={symbol: futures.back_adjust(schedule, frames)},
            bars=bars,
            instruments={iid: by_id[iid] for iid in schedule.instrument_ids},
            bar_types=bar_types,
            schedules={symbol: schedule},
        )

    @property
    def symbols(self) -> list[str]:
        return list(self.frames)

    def closes(self, *, union: bool = False) -> pl.Series:
        """Bar closes, sorted (``Datetime[ns, UTC]``).

        Args:
            union: Every close any symbol has, instead of only those shared by
                all symbols. Use it for a universe whose members list and delist
                at different times (the shared closes may then be few or none).
        """
        common: pl.DataFrame | None = None
        for frame in self.frames.values():
            ts = frame.select(pl.col("ts_init").cast(pl.Datetime("ns", "UTC")))
            if common is None:
                common = ts
            elif union:
                common = pl.concat([common, ts]).unique()
            else:
                common = common.join(ts, on="ts_init", how="inner")
        assert common is not None  # noqa: S101 — frames is never empty
        return common["ts_init"].sort()

    def merge(self, other: MarketData) -> MarketData:
        """Both datasets together (symbols must not overlap)."""
        if set(self.frames) & set(other.frames):
            raise ValueError("symbols overlap")
        return MarketData(
            {**self.frames, **other.frames},
            {**self.bars, **other.bars},
            {**self.instruments, **other.instruments},
            {**self.bar_types, **other.bar_types},
            {**self.schedules, **other.schedules},
        )

    def select(self, symbols: Sequence[str]) -> MarketData:
        """The subset of this data for *symbols*."""
        ids = [i for s in symbols for i in self.schedules[s].instrument_ids]
        return MarketData(
            {s: self.frames[s] for s in symbols},
            {i: self.bars[i] for i in ids},
            {i: self.instruments[i] for i in ids},
            {i: self.bar_types[i] for i in ids},
            {s: self.schedules[s] for s in symbols},
        )

    def window(self, start: TimeBound, end: TimeBound) -> MarketData:
        """Frames up to *end* (for signals) and bars in ``[start, end]`` (for replay).

        A rolled contract's bars are also limited to its active segment plus the
        close it is rolled out on.
        """
        lo, hi = cat.to_ns(start), cat.to_ns(end)
        frames = (
            self.frames
            if hi is None
            else {s: f.filter(_ts_column(f) <= hi) for s, f in self.frames.items()}
        )
        bars: dict[str, list[Bar]] = {}
        for schedule in self.schedules.values():
            for iid in schedule.instrument_ids:
                first, last = schedule.span(iid)
                if lo is not None:
                    first = max(first, lo)
                if hi is not None:
                    last = min(last, hi)
                series = self.bars[iid]
                keys = [b.ts_init for b in series]
                bars[iid] = series[
                    bisect.bisect_left(keys, first) : bisect.bisect_right(keys, last)
                ]
        return MarketData(frames, bars, self.instruments, self.bar_types, self.schedules)


def _frame_bars(
    frame: pl.DataFrame,
    instrument: Any,  # noqa: ANN401 — any Nautilus instrument
    bar_type: str,
    schema: str,
    label: str,
) -> list[Bar]:
    interval = cat.bar_interval_ns(schema)
    if ((_ts_column(frame) - _ts_column(frame, "ts_event")) != interval).any():
        raise ValueError(f"{label}: ts_init must equal ts_event + the {schema} bar length")
    return cat.frame_to_bars(
        frame,
        bar_type=bar_type,
        price_precision=instrument.price_precision,
        interval_ns=interval,
        size_precision=instrument.size_precision,
    )


def _query_bars(catalog: Any, bar_type: str) -> list[Bar]:  # noqa: ANN401
    return sorted(catalog.query(NautilusDataType.Bar, [bar_type]), key=lambda b: b.ts_init)


def _load_parent(
    catalog: Any,  # noqa: ANN401
    symbol: str,
    schema: str,
    venue: str | None,
    catalog_path: Path | None,
    rule: RollRule | None,
) -> MarketData:
    root = futures.parent_root(symbol)
    contracts = futures.outrights(catalog, root, venue=venue)
    frames: dict[str, pl.DataFrame] = {}
    for c in contracts:
        raw, v = str(c.raw_symbol), str(c.id.venue)
        if catalog.query_last_timestamp(NautilusDataType.Bar, cat.bar_type(raw, v, schema)):
            frames[str(c.id)] = cat.load_bars(raw, schema, venue=v, catalog_path=catalog_path)
    if not frames:
        raise FileNotFoundError(
            f"No {schema} bars for any {root} outright in the catalog. "
            f"Run `algo data sync` with {symbol} in the config first."
        )
    schedule = futures.roll_schedule(symbol, contracts, frames, rule)
    by_id = {str(c.id): c for c in contracts}
    bar_types = {
        iid: cat.bar_type(str(by_id[iid].raw_symbol), str(by_id[iid].id.venue), schema)
        for iid in schedule.instrument_ids
    }
    return MarketData(
        frames={symbol: futures.back_adjust(schedule, frames)},
        bars={iid: _query_bars(catalog, bt) for iid, bt in bar_types.items()},
        instruments={iid: by_id[iid] for iid in schedule.instrument_ids},
        bar_types=bar_types,
        schedules={symbol: schedule},
    )


def load_market_data(
    symbols: str | Sequence[str],
    schema: str = "ohlcv-1d",
    *,
    venue: str | None = None,
    roll: RollRule | None = None,
    catalog_path: Path | None = None,
) -> MarketData:
    """Load bars and instrument definitions for *symbols* from the catalog.

    Bars are replayed exactly as stored (``ts_init`` = bar close). A parent
    futures symbol (``ES.FUT``) loads every stored outright of the root, picks
    the active one per bar with *roll* (default: volume) and back-adjusts the
    signal series (see :mod:`snippy_scales.data.futures`).

    Raises:
        FileNotFoundError: If a series or instrument is missing from the catalog.
    """
    names = [symbols] if isinstance(symbols, str) else list(symbols)
    catalog = cat.open_catalog(catalog_path)
    data: MarketData | None = None
    for symbol in names:
        if futures.is_parent(symbol):
            part = _load_parent(catalog, symbol, schema, venue, catalog_path, roll)
        else:
            bt = cat.resolve_bar_type(symbol, schema, venue=venue, catalog_path=catalog_path)
            iid = str(BarType.from_str(bt).instrument_id)
            found = catalog.instruments(instrument_ids=[iid])
            if not found:
                raise FileNotFoundError(
                    f"No instrument definition for {iid} in the catalog. "
                    "Run `algo data sync <config> --instruments-only` first."
                )
            frame = cat.load_bars(symbol, schema, venue=venue, catalog_path=catalog_path)
            part = MarketData(
                frames={symbol: frame},
                bars={iid: _query_bars(catalog, bt)},
                instruments={iid: found[-1]},
                bar_types={iid: bt},
                schedules={symbol: RollSchedule.single(symbol, iid)},
            )
        data = part if data is None else data.merge(part)
    if data is None:
        raise ValueError("no symbols given")
    return data


@dataclass(frozen=True)
class BacktestRun:
    """Output of one backtest.

    Attributes:
        stats_returns: Nautilus return statistics (e.g. :data:`SHARPE`).
        stats_pnls: Nautilus PnL statistics per currency.
        stats_general: Nautilus general statistics.
        returns: Nautilus daily returns: ``ts`` (UTC), ``return``.
        equity: Account equity at every bar close, before that bar's trades:
            ``ts`` (UTC), ``equity``.
        fills: Nautilus fills report.
        positions: Nautilus positions report.
        rejections: Reasons of orders that were denied or rejected. A non-empty
            tuple means the run did not trade as the strategy intended.
        total_orders: Orders submitted.
        rolls: ``(bar close ns, old instrument, new instrument)`` for every roll
            of a held position.
    """

    stats_returns: dict[str, float]
    stats_pnls: dict[str, dict[str, float]]
    stats_general: dict[str, float]
    returns: pl.DataFrame
    equity: pl.DataFrame
    fills: pd.DataFrame = field(repr=False)
    positions: pd.DataFrame = field(repr=False)
    rejections: tuple[str, ...]
    total_orders: int
    rolls: tuple[tuple[int, str, str], ...] = ()

    @property
    def stats(self) -> dict[str, float]:
        """All non-NaN statistics in one flat dict."""
        merged: dict[str, float] = {**self.stats_returns, **self.stats_general}
        for per_currency in self.stats_pnls.values():
            merged.update(per_currency)
        return {k: float(v) for k, v in merged.items() if not math.isnan(float(v))}

    @property
    def sharpe(self) -> float:
        """Annualised Sharpe of daily returns; ``0.0`` if undefined (no trading)."""
        value = float(self.stats_returns.get(SHARPE, math.nan))
        return 0.0 if math.isnan(value) else value

    @property
    def total_return(self) -> float:
        """Total return over the run, from the equity curve, as a fraction."""
        values = self.equity["equity"]
        return float(values[-1] / values[0] - 1.0) if values.len() > 1 else 0.0


def _series_frame(values: Mapping[int, float], column: str) -> pl.DataFrame:
    return pl.DataFrame(
        {"ts": list(values.keys()), column: list(values.values())},
        schema={"ts": pl.Int64, column: pl.Float64},
    ).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))


def run_backtest(
    research: Strategy | MultiAssetStrategy,
    data: MarketData,
    *,
    start: TimeBound = None,
    end: TimeBound = None,
    initial_capital: float = 1_000_000.0,
    costs: Costs | None = None,
    rebalance_band: float = 0.0,
    weights: Mapping[str, float] | None = None,
    leverage: float = 10.0,
) -> BacktestRun:
    """Backtest *research* on *data* between *start* and *end*.

    Args:
        research: Research strategy producing targets in units of equity.
        data: Bars and instruments (:func:`load_market_data`, :meth:`MarketData.from_frames`).
        start: First bar close to trade (inclusive). Earlier bars warm up signals.
        end: Last bar close (inclusive). Later bars are ignored.
        initial_capital: Starting balance in the instruments' quote currency.
        costs: Transaction costs. Defaults to none.
        rebalance_band: Skip trades below this fraction of equity.
        weights: ``symbol -> capital weight``. Defaults to equal weights.
        leverage: Venue default leverage for margin checks.

    Returns:
        The :class:`BacktestRun`.
    """
    costs = costs if costs is not None else Costs()
    data = data.window(start, end)
    expected = Counter(b.ts_init for series in data.bars.values() for b in series)
    strategy = SignalStrategy(
        research,
        history=data.frames,
        instruments=data.instruments,
        bar_types=data.bar_types,
        schedules=data.schedules,
        expected=expected,
        weights=weights,
        rebalance_band=rebalance_band,
    )
    first = next(iter(data.instruments.values()))
    currency = first.quote_currency
    engine = BacktestEngine(BacktestEngineConfig(logging=LoggerConfig(stdout_level=LogLevel.ERROR)))
    try:
        engine.add_venue(
            Venue(str(first.id.venue)),
            OmsType.NETTING,
            AccountType.MARGIN,
            [Money(initial_capital, currency)],
            base_currency=currency,
            default_leverage=Decimal(repr(leverage)),
            fee_model=costs.fee_model(currency),
            fill_model=costs.fill_model(),
        )
        for instrument in data.instruments.values():
            engine.add_instrument(instrument)
        for series in data.bars.values():
            if series:
                engine.add_data(series)
        engine.add_strategy(strategy)
        engine.run()
        result = engine.get_result()
        return BacktestRun(
            stats_returns=dict(result.stats_returns),
            stats_pnls={k: dict(v) for k, v in result.stats_pnls.items()},
            stats_general=dict(result.stats_general),
            returns=_series_frame(result.returns_series, "return"),
            equity=_series_frame(dict(sorted(strategy.equity.items())), "equity"),
            fills=engine.generate_fills_report(),
            positions=engine.generate_positions_report(),
            rejections=tuple(strategy.rejections),
            total_orders=result.total_orders,
            rolls=tuple(strategy.rolls),
        )
    finally:
        engine.dispose()
