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

from decimal import ROUND_HALF_EVEN, Decimal
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from nautilus_trader.model import BarType, OrderSide, Quantity, Venue
from nautilus_trader.trading import Strategy as NautilusStrategy

from snippy_scales.strategies.momentum_cs import MultiAssetStrategy

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


def _align(
    frames: Mapping[str, pl.DataFrame],
) -> tuple[dict[str, pl.DataFrame], dict[str, np.ndarray]]:
    """Align every frame on the **union** of bar closes, sorted by ``ts_init``.

    A symbol's rows between its first and last bar that it has no bar for are
    forward-filled (``volume`` 0); rows before its first bar or after its last
    are null. A universe whose members list and delist at different times (an
    S&P 500 "ever members" universe) therefore keeps every date, and a strategy
    never sees a price from before a listing or after a delisting.

    Returns:
        ``(aligned frames, symbol -> bool mask of the rows that are real bars)``.
    """
    calendar = (
        pl.concat([f.select(pl.col("ts_init")) for f in frames.values()]).unique().sort("ts_init")
    )
    aligned: dict[str, pl.DataFrame] = {}
    real: dict[str, np.ndarray] = {}
    for symbol, frame in frames.items():
        frame = frame.sort("ts_init")
        values = [c for c in frame.columns if c != "ts_init"]
        joined = calendar.join(
            frame.with_columns(pl.lit(True).alias("__bar")), on="ts_init", how="left"
        )
        has_bar = joined["__bar"].fill_null(False)  # noqa: FBT003
        if frame.height:
            first, last = frame["ts_init"][0], frame["ts_init"][-1]
            alive = pl.col("ts_init").is_between(first, last)
            fills = [
                pl.when(alive)
                .then(pl.lit(0.0) if c == "volume" else pl.col(c).forward_fill())
                .otherwise(None)
                .alias(c)
                for c in values
            ]
            # Real bars keep their own values (volume included); gaps are filled.
            joined = joined.with_columns(
                pl.when(pl.col("__bar")).then(pl.col(c)).otherwise(f).alias(c)
                for c, f in zip(values, fills, strict=True)
            )
        aligned[symbol] = joined.select(frame.columns)
        real[symbol] = has_bar.to_numpy()
    return aligned, real


def _exit_delisted(
    frames: Mapping[str, pl.DataFrame], targets: dict[str, dict[int, float]]
) -> None:
    """Flatten a symbol on its last bar when its data ends before the universe's.

    A symbol whose bars stop early (delisted, merged, dropped from the data
    source) would otherwise keep its position, marked at a stale price, for the
    rest of the run. It is closed at its last close instead.
    """
    if len(frames) < 2:  # noqa: PLR2004
        return
    ends = {s: int(_ts_ns(f).max()) for s, f in frames.items() if f.height}
    if not ends:
        return
    horizon = max(ends.values())
    for symbol, end in ends.items():
        if end < horizon and end in targets[symbol]:
            targets[symbol][end] = 0.0


def _clean(values: pl.Series, n: int, symbol: str) -> np.ndarray:
    """Signal values as float64 with null/NaN read as flat; checks the length."""
    if values.len() != n:
        raise ValueError(
            f"generate_signals returned {values.len()} values for {n} bars of {symbol}"
        )
    arr = values.cast(pl.Float64).fill_null(0.0).to_numpy()
    return np.where(np.isfinite(arr), arr, 0.0)


def compute_targets(
    research: Strategy | MultiAssetStrategy, frames: Mapping[str, pl.DataFrame]
) -> dict[str, dict[int, float]]:
    """Run *research* on *frames* and key each target position by bar close.

    A :class:`MultiAssetStrategy` sees the frames aligned on the union of their
    bar closes (see :func:`_align`: null before a symbol's first bar and after
    its last); targets are kept only on each symbol's real bars. A single-asset
    strategy runs on each frame independently. With several symbols, one whose
    bars end before the others' is flattened on its last bar.

    Args:
        research: The research strategy.
        frames: ``symbol -> bars`` with at least ``ts_init`` and ``close``.

    Returns:
        ``symbol -> {ts_init ns: target position in units of equity}``.
    """
    if not frames:
        raise ValueError("frames must not be empty")
    targets: dict[str, dict[int, float]] = {}
    if isinstance(research, MultiAssetStrategy):
        aligned, real = _align(frames)
        signals = research.generate_signals(aligned)
        for symbol, frame in aligned.items():
            values = _clean(signals[symbol], frame.height, symbol)
            mask = real[symbol]
            targets[symbol] = dict(
                zip(_ts_ns(frame)[mask].tolist(), values[mask].tolist(), strict=True)
            )
    else:
        for symbol, frame in frames.items():
            frame = frame.sort("ts_init")
            values = _clean(research.generate_signals(frame), frame.height, symbol)
            targets[symbol] = dict(zip(_ts_ns(frame).tolist(), values.tolist(), strict=True))
    _exit_delisted(frames, targets)
    return targets


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

    def _close(self, instrument_id: str) -> None:
        """Flatten the position in *instrument_id* (the old contract of a roll)."""
        instrument = self._instruments[instrument_id]
        position = self.portfolio.net_position(instrument.id)
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
        delta = lots * increment - self.portfolio.net_position(instrument.id)
        if delta == 0:
            return
        if not force and float(abs(delta)) * price * multiplier < self._band * equity:
            return
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        quantity = Quantity(float(abs(delta)), instrument.size_precision)
        self.submit_order(self.order_factory.market(instrument.id, side, quantity))
        self._holding[symbol] = instrument_id
