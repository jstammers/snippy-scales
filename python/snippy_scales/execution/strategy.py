"""``SignalStrategy``: run a vectorised research strategy inside Nautilus.

Research strategies (:class:`~snippy_scales.strategies.trend.Strategy`,
:class:`~snippy_scales.strategies.momentum_cs.MultiAssetStrategy`) map a polars
bar frame to a series of target positions in **units of equity**: ``1.0`` is
fully invested long, ``-0.5`` is short half the portfolio. ``SignalStrategy``
turns those targets into Nautilus market orders.

Signals are computed once, before the run, on the whole bar history, and looked
up by bar close (``ts_init``) as bars arrive. Calling
``generate_signals`` bar by bar would be about 25x slower, and gives the same
answer only because every strategy is causal (``test_strategy_causality.py``).

When a bar arrives that the precomputed table doesn't cover (live trading), the
bar is appended to the history and the targets are recomputed on the trailing
window. Backtests and live trading therefore share one code path.

Each symbol trades through a :class:`~snippy_scales.data.futures.RollSchedule`.
An equity or a continuous series has a single instrument. A parent futures
symbol (``ES.FUT``) has one outright per segment: the signals come from the
back-adjusted series, the orders go to the active outright, and on a roll's bar
the old contract is closed and the new one opened at that close (two fills, so
roll costs are paid).

Sizing: on the bar closing at ``t`` the strategy holds

    contracts = target_t * weight * equity_t / (close_t * multiplier)

rounded to the instrument's size increment, where ``equity_t`` is read once all
instruments closing at ``t`` have been marked. The order fills at that bar's close
and earns the next bar's return, the same timing as the old
``TargetPositionEngine`` with ``execution_lag=1``. A ``rebalance_band`` skips
trades smaller than that fraction of equity.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from nautilus_trader.model import BarType, OrderSide, Quantity, Venue
from nautilus_trader.trading import Strategy as NautilusStrategy
from numba import njit

from snippy_scales.strategies.momentum_cs import MultiAssetStrategy
from snippy_scales.strategies.trend import SYMBOL

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nautilus_trader.model import Bar, Currency, OrderDenied, OrderRejected

    from snippy_scales.data.futures import RollSchedule
    from snippy_scales.strategies.trend import Strategy

#: Columns a history frame must carry, as :func:`~snippy_scales.data.catalog.load_bars` returns.
HISTORY_COLUMNS: tuple[str, ...] = ("ts_event", "ts_init", "open", "high", "low", "close", "volume")


def _ts_ns(frame: pl.DataFrame) -> np.ndarray:
    """``ts_init`` of *frame* as int64 nanoseconds."""
    ts = frame["ts_init"]
    if ts.dtype != pl.Int64:
        ts = ts.dt.cast_time_unit("ns").dt.epoch("ns")
    return ts.to_numpy().astype(np.int64)


#: Schema of :meth:`Targets.frame`.
TARGET_SCHEMA = {SYMBOL: pl.Utf8, "ts_init": pl.Int64, "target": pl.Float64}


@njit(cache=True)
def block_searchsorted(ref: np.ndarray, keys: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """``np.searchsorted(ref, keys)`` for *keys* sorted within each block of *starts*.

    One merge pass per block: over 10x faster than a binary search per key on a
    universe-sized panel.
    """
    out = np.empty(keys.size, np.int64)
    for b in range(starts.size - 1):
        j = 0
        for i in range(starts[b], starts[b + 1]):
            k = keys[i]
            while j < ref.size and ref[j] < k:
                j += 1
            out[i] = j
    return out


def _block_rows(starts: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Block offsets after keeping the rows *keep* (a bool mask over all rows)."""
    kept = np.concatenate([[0], np.cumsum(keep)])
    return kept[starts]


@dataclass(frozen=True)
class Panel:
    """Bars of many symbols stacked into one long frame (built by :func:`stack_frames`).

    Attributes:
        bars: Every column of the frames, :data:`SYMBOL` set to the symbol;
            each symbol's rows contiguous and sorted by ``ts_init``.
        symbols: The symbols, in block order.
        starts: ``int64 [N + 1]`` row offset of each symbol's block (the last
            entry is the row count); a symbol without bars has an empty block.
        ts: ``int64`` ``ts_init`` (ns) of every row.
        columns: The frames' own columns (*bars* adds :data:`SYMBOL` if they lack it).
    """

    bars: pl.DataFrame
    symbols: tuple[str, ...]
    starts: np.ndarray
    ts: np.ndarray
    columns: tuple[str, ...]

    def until(self, end_ns: int | None) -> Panel:
        """The rows with ``ts_init <= end_ns`` (every symbol kept, perhaps empty)."""
        if end_ns is None:
            return self
        keep = self.ts <= end_ns
        if keep.all():
            return self
        return Panel(
            self.bars.filter(pl.Series(keep)),
            self.symbols,
            _block_rows(self.starts, keep),
            self.ts[keep],
            self.columns,
        )


def stack_frames(frames: Mapping[str, pl.DataFrame]) -> Panel:
    """Stack ``symbol -> bars`` into a :class:`Panel`, each symbol sorted by ``ts_init``.

    Build it once per dataset and window it with :meth:`Panel.until`: one
    filter, instead of one per symbol.
    """
    if not frames:
        raise ValueError("frames must not be empty")
    parts = [
        f.sort("ts_init").with_columns(pl.lit(s, pl.Utf8).alias(SYMBOL)) for s, f in frames.items()
    ]
    bars = pl.concat(parts, how="diagonal_relaxed")
    lengths = np.array([p.height for p in parts], np.int64)
    columns = tuple(dict.fromkeys(c for f in frames.values() for c in f.columns))
    return Panel(
        bars, tuple(frames), np.concatenate([[0], np.cumsum(lengths)]), _ts_ns(bars), columns
    )


@dataclass(frozen=True)
class Targets:
    """Target positions of many symbols, one per real bar, as flat arrays.

    Laid out as :class:`Panel`: symbol *k*'s rows are
    ``starts[k]:starts[k + 1]``, sorted by ``ts_init``.

    Attributes:
        symbols: The symbols, in block order.
        starts: ``int64 [N + 1]`` block offsets.
        ts: ``int64`` bar close (ns) of every row.
        target: ``float64`` target in units of equity (null/NaN already read as flat).
    """

    symbols: tuple[str, ...]
    starts: np.ndarray
    ts: np.ndarray
    target: np.ndarray

    @property
    def block(self) -> np.ndarray:
        """Symbol index of every row."""
        return np.repeat(np.arange(len(self.symbols)), np.diff(self.starts))

    def until(self, end_ns: int | None) -> Targets:
        """The rows with ``ts_init <= end_ns``."""
        if end_ns is None:
            return self
        keep = self.ts <= end_ns
        if keep.all():
            return self
        return Targets(
            self.symbols, _block_rows(self.starts, keep), self.ts[keep], self.target[keep]
        )

    def exit_delisted(self) -> Targets:
        """Flatten a symbol on its last bar when its data ends before the universe's.

        A symbol whose bars stop early (delisted, merged, dropped from the data
        source) would otherwise keep its position, marked at a stale price, for
        the rest of the run. It is closed at its last close instead.
        """
        if len(self.symbols) < 2:  # noqa: PLR2004
            return self
        last = self.starts[1:][np.diff(self.starts) > 0] - 1
        if last.size == 0:
            return self
        ends = self.ts[last]
        early = last[ends < ends.max()]
        if early.size == 0:
            return self
        target = self.target.copy()
        target[early] = 0.0
        return Targets(self.symbols, self.starts, self.ts, target)

    def frame(self) -> pl.DataFrame:
        """:data:`TARGET_SCHEMA`: ``symbol``, ``ts_init`` (ns), ``target``."""
        symbols = pl.Series(SYMBOL, self.symbols, pl.Utf8)
        return pl.DataFrame(
            {
                SYMBOL: symbols.gather(self.block),
                "ts_init": self.ts,
                "target": self.target,
            },
            schema=TARGET_SCHEMA,
        )

    def dicts(self) -> dict[str, dict[int, float]]:
        """``symbol -> {ts_init ns: target}``, the per-bar lookup :class:`SignalStrategy` uses."""
        ts, target = self.ts.tolist(), self.target.tolist()
        return {
            s: dict(zip(ts[lo:hi], target[lo:hi], strict=True))
            for s, lo, hi in zip(self.symbols, self.starts[:-1], self.starts[1:], strict=True)
        }


def _finite(values: pl.Series) -> np.ndarray:
    """Signal values as float64 with null/NaN/inf read as flat."""
    arr = values.cast(pl.Float64).fill_null(0.0).to_numpy()
    return np.where(np.isfinite(arr), arr, 0.0)


def _align(panel: Panel) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Align every symbol of *panel* on the **union** of bar closes.

    A symbol's rows between its first and last bar that it has no bar for are
    forward-filled (``volume`` 0); rows before its first bar or after its last
    are null. A universe whose members list and delist at different times (an
    S&P 500 "ever members" universe) therefore keeps every date, and a strategy
    never sees a price from before a listing or after a delisting.

    The grid is built from row indices (one ``gather``), not one join per symbol.

    Returns:
        ``(grid, real, calendar)``: the aligned bars of every symbol stacked
        symbol-major (``len(symbols) * len(calendar)`` rows), a bool mask of the
        grid rows that are real bars, and the calendar (``int64`` ns).
    """
    n_sym = len(panel.symbols)
    # polars: numpy 2's hash-based np.unique takes seconds on a large panel's closes.
    calendar = pl.Series(panel.ts).unique().sort().to_numpy()
    t_n = calendar.size
    lengths = np.diff(panel.starts)
    block = np.repeat(np.arange(n_sym), lengths)
    pos = block_searchsorted(calendar, panel.ts, panel.starts)
    src = np.full((n_sym, t_n), -1, np.int64)
    src[block, pos] = np.arange(panel.ts.size)
    real = src >= 0
    filled = np.maximum.accumulate(src, axis=1) if t_n else src  # last bar at or before
    last = np.full(n_sym, -1, np.int64)
    listed = lengths > 0
    last[listed] = pos[panel.starts[1:][listed] - 1]
    alive = (filled >= 0) & (np.arange(t_n)[None, :] <= last[:, None])
    idx = pl.Series(np.where(alive, filled, -1).ravel())
    idx = pl.select(pl.when(idx >= 0).then(idx)).to_series()
    ts_dtype = panel.bars.schema["ts_init"]
    calendar_col = pl.Series("ts_init", np.tile(calendar, n_sym))
    if ts_dtype != pl.Int64:
        calendar_col = calendar_col.cast(pl.Datetime("ns", "UTC")).cast(ts_dtype)
    real_col, alive_col = pl.Series(real.ravel()), pl.Series(alive.ravel())
    grid = panel.bars.select(pl.col(c).gather(idx) for c in panel.columns)
    grid = grid.with_columns(calendar_col)
    if "volume" in grid.columns:
        grid = grid.with_columns(
            pl.when(real_col).then(pl.col("volume")).when(alive_col).then(0.0).alias("volume")
        )
    return grid, real.ravel(), calendar


def panel_targets(
    research: Strategy | MultiAssetStrategy, panel: Panel, *, exits: bool = True
) -> Targets:
    """Run *research* on every symbol of *panel* and keep a target per real bar.

    A :class:`MultiAssetStrategy` sees the symbols aligned on the union of their
    bar closes (see :func:`_align`); targets are kept only on each symbol's real
    bars. A single-asset strategy sees each symbol's bars alone, through
    :meth:`~snippy_scales.strategies.trend.Strategy.generate_signals_panel` (one
    query for the whole universe when the strategy overrides it). With several
    symbols, one whose bars end before the others' is flattened on its last bar
    (unless *exits* is false).
    """
    if isinstance(research, MultiAssetStrategy):
        grid, real, calendar = _align(panel)
        values = research.generate_signals_panel(grid, panel.symbols)
        if values.len() != grid.height:
            raise ValueError(
                f"generate_signals_panel returned {values.len()} values for {grid.height} rows"
            )
        ts = np.tile(calendar, len(panel.symbols))[real]
        counts = real.reshape(len(panel.symbols), calendar.size).sum(axis=1)
        starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        out = Targets(panel.symbols, starts, ts, _finite(values)[real])
    else:
        values = research.generate_signals_panel(panel.bars)
        if values.len() != panel.bars.height:
            raise ValueError(
                f"generate_signals_panel returned {values.len()} values "
                f"for {panel.bars.height} bars"
            )
        out = Targets(panel.symbols, panel.starts, panel.ts, _finite(values))
    return out.exit_delisted() if exits else out


def target_frame(
    research: Strategy | MultiAssetStrategy, frames: Mapping[str, pl.DataFrame]
) -> pl.DataFrame:
    """:func:`panel_targets` on *frames*, as a :data:`TARGET_SCHEMA` frame."""
    return panel_targets(research, stack_frames(frames)).frame()


def compute_targets(
    research: Strategy | MultiAssetStrategy, frames: Mapping[str, pl.DataFrame]
) -> dict[str, dict[int, float]]:
    """:func:`panel_targets` keyed by bar close: ``symbol -> {ts_init ns: target}``.

    The per-bar lookup :class:`SignalStrategy` trades from; array engines use
    :func:`panel_targets` directly and skip building these dicts.
    """
    return panel_targets(research, stack_frames(frames)).dicts()


class SignalStrategy(NautilusStrategy):
    """Nautilus strategy that trades the target positions of a research strategy.

    Args:
        research: Research strategy producing targets in units of equity.
        history: ``symbol -> bars`` used to precompute targets (back-adjusted for
            rolled futures). It should cover the run plus any warm-up; bars after
            it trigger a recompute.
        instruments: ``instrument_id -> Nautilus instrument``; all at one venue.
        bar_types: ``instrument_id -> BarType`` string to subscribe to.
        schedules: ``symbol -> RollSchedule`` over those instruments.
        expected: ``bar close -> number of bars`` fed to the engine at that close,
            so a timestamp's batch is known to be complete. Defaults to counting
            *history*; closes it doesn't cover expect one bar per symbol.
        weights: ``symbol -> capital weight``. Defaults to ``1 / n_symbols``.
        rebalance_band: Skip trades whose notional is below this fraction of
            equity. ``0.0`` rebalances on every bar. Rolls are never skipped.
        history_window: Bars kept when recomputing live; ``None`` keeps all.
    """

    def __new__(cls, *_args: object, **_kwargs: object) -> SignalStrategy:
        # The Nautilus base is a PyO3 class whose __new__ takes only `config`.
        return super().__new__(cls)

    def __init__(
        self,
        research: Strategy | MultiAssetStrategy,
        *,
        history: Mapping[str, pl.DataFrame],
        instruments: Mapping[str, Any],
        bar_types: Mapping[str, str],
        schedules: Mapping[str, RollSchedule],
        expected: Mapping[int, int] | None = None,
        weights: Mapping[str, float] | None = None,
        rebalance_band: float = 0.0,
        history_window: int | None = None,
    ) -> None:
        super().__init__()
        symbols = list(schedules)
        if set(history) != set(symbols):
            raise ValueError("history and schedules must cover the same symbols")
        traded = {iid for sched in schedules.values() for iid in sched.instrument_ids}
        if traded - set(instruments) or traded - set(bar_types):
            raise ValueError(f"no instrument or bar type for {sorted(traded - set(instruments))}")
        venues = {str(instruments[i].id.venue) for i in traded}
        if len(venues) != 1:
            raise ValueError(f"all instruments must share one venue, got {sorted(venues)}")
        if rebalance_band < 0.0:
            raise ValueError(f"rebalance_band must be non-negative, got {rebalance_band}")
        currencies = {str(instruments[i].quote_currency) for i in traded}
        if len(currencies) != 1:
            raise ValueError(f"all instruments must share one currency, got {sorted(currencies)}")

        self._research = research
        self._history = {s: history[s].select(HISTORY_COLUMNS).sort("ts_init") for s in symbols}
        self._schedules = dict(schedules)
        self._instruments = {i: instruments[i] for i in traded}
        self._bar_types = {i: BarType.from_str(bar_types[i]) for i in traded}
        self._iid_of = {str(bt): i for i, bt in self._bar_types.items()}
        self._weights = (
            dict(weights) if weights is not None else dict.fromkeys(symbols, 1.0 / len(symbols))
        )
        self._band = rebalance_band
        self._window = history_window
        self._venue = Venue(venues.pop())
        self._currency: Currency = self._instruments[next(iter(traded))].quote_currency
        self._targets = compute_targets(research, self._history)
        self._ends = {s: self._last_close(f) for s, f in self._history.items()}
        if expected is None:
            counts = (
                pl.concat([f.select("ts_init") for f in self._history.values()])
                .group_by("ts_init")
                .len()
            )
            expected = dict(zip(_ts_ns(counts).tolist(), counts["len"].to_list(), strict=True))
        self._expected = dict(expected)
        self._holding: dict[str, str | None] = dict.fromkeys(symbols)
        self._pending: dict[str, Bar] = {}
        self._pending_ts: int | None = None
        self.equity: dict[int, float] = {}
        self.rejections: list[str] = []
        self.rolls: list[tuple[int, str, str]] = []

    # -- lifecycle -----------------------------------------------------------

    def on_start(self) -> None:
        for bar_type in self._bar_types.values():
            self.subscribe_bars(bar_type)

    def on_bar(self, bar: Bar) -> None:
        """Collect the bars closing at one timestamp, then rebalance them together.

        Sizing every instrument after all of them are marked to this close
        uses one consistent equity figure, and a roll sees both contracts'
        closes. A bar with a later close flushes an incomplete batch (e.g. a
        symbol that didn't trade that day).
        """
        if self._pending_ts is not None and bar.ts_init != self._pending_ts:
            self._flush()
        self._pending_ts = bar.ts_init
        self._pending[self._iid_of[str(bar.bar_type)]] = bar
        if len(self._pending) >= self._expected.get(bar.ts_init, len(self._schedules)):
            self._flush()

    def on_order_denied(self, event: OrderDenied) -> None:
        self.rejections.append(f"denied: {event.reason}")
        self.log.warning(f"Order denied: {event.reason}")

    def on_order_rejected(self, event: OrderRejected) -> None:
        self.rejections.append(f"rejected: {event.reason}")
        self.log.warning(f"Order rejected: {event.reason}")

    # -- internals -----------------------------------------------------------

    def _flush(self) -> None:
        """Roll and rebalance every symbol in the pending batch at one equity figure."""
        batch, ts = self._pending, self._pending_ts
        self._pending, self._pending_ts = {}, None
        if not batch or ts is None:
            return
        equity = float(self.portfolio.equity(self._venue)[self._currency])
        self.equity[ts] = equity
        active = {s: sched.active(ts) for s, sched in self._schedules.items()}
        live = {s: batch[i] for s, i in active.items() if i in batch}
        if any(ts > self._ends[s] for s in live):
            self._extend_history(live)
        for symbol, bar in live.items():
            iid = active[symbol]
            held = self._holding[symbol]
            rolled = False
            if held is not None and held != iid:
                rolled = True
                self._close(held)
                self.rolls.append((ts, held, iid))
                self._holding[symbol] = None
            target = self._targets[symbol].get(ts)
            if target is not None:
                self._rebalance(symbol, iid, target, float(bar.close), equity, force=rolled)

    @staticmethod
    def _last_close(frame: pl.DataFrame) -> int:
        """Close of the last bar in *frame* (``-1`` if empty)."""
        return int(_ts_ns(frame.tail(1))[0]) if frame.height else -1

    def _extend_history(self, live: Mapping[str, Bar]) -> None:
        """Append live bars to the history and recompute targets (the live path).

        Beyond the history the active contract is a schedule's last one, the
        unadjusted anchor of a back-adjusted series, so raw prices append as is.
        """
        for symbol, bar in live.items():
            if bar.ts_init <= self._ends[symbol]:
                continue
            frame = self._history[symbol]
            row = pl.DataFrame(
                {
                    "ts_event": [bar.ts_event],
                    "ts_init": [bar.ts_init],
                    "open": [float(bar.open)],
                    "high": [float(bar.high)],
                    "low": [float(bar.low)],
                    "close": [float(bar.close)],
                    "volume": [float(bar.volume)],
                }
            ).cast(frame.schema)
            frame = pl.concat([frame, row])
            if self._window is not None:
                frame = frame.tail(self._window)
            self._history[symbol] = frame
            self._ends[symbol] = bar.ts_init
        self._targets = compute_targets(self._research, self._history)

    def _net_position(self, instrument: Any) -> Decimal:  # noqa: ANN401 — any Nautilus instrument
        """Net position rounded to the instrument's size precision.

        Nautilus reports it with float noise (``Decimal('3928347.000000001')``):
        unrounded, an unchanged target leaves a delta of ~1e-9, which becomes a
        zero ``Quantity`` and raises, aborting the rest of that bar's batch.
        """
        position = Decimal(str(self.portfolio.net_position(instrument.id)))
        return position.quantize(Decimal(1).scaleb(-instrument.size_precision))

    def _close(self, instrument_id: str) -> None:
        """Flatten the position in *instrument_id* (the old contract of a roll)."""
        instrument = self._instruments[instrument_id]
        position = self._net_position(instrument)
        if position == 0:
            return
        side = OrderSide.SELL if position > 0 else OrderSide.BUY
        quantity = Quantity(float(abs(position)), instrument.size_precision)
        self.submit_order(self.order_factory.market(instrument.id, side, quantity))

    def _rebalance(
        self,
        symbol: str,
        instrument_id: str,
        target: float,
        price: float,
        equity: float,
        *,
        force: bool = False,
    ) -> None:
        """Trade *instrument_id* towards *target* units of equity at *price*.

        *force* skips the rebalance band (re-entering after a roll).
        """
        instrument = self._instruments[instrument_id]
        if price <= 0.0 or equity <= 0.0:
            return
        multiplier = float(instrument.multiplier)
        increment = Decimal(str(instrument.size_increment))
        wanted = target * self._weights[symbol] * equity / (price * multiplier)
        lots = (Decimal(repr(wanted)) / increment).to_integral_value(ROUND_HALF_EVEN)
        delta = lots * increment - self._net_position(instrument)
        if delta == 0:
            return
        if not force and float(abs(delta)) * price * multiplier < self._band * equity:
            return
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        quantity = Quantity(float(abs(delta)), instrument.size_precision)
        self.submit_order(self.order_factory.market(instrument.id, side, quantity))
        self._holding[symbol] = instrument_id
