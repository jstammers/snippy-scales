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

    from snippy_scales.strategies.trend import Strategy

#: Columns a history frame must carry, as :func:`~snippy_scales.data.catalog.load_bars` returns.
HISTORY_COLUMNS: tuple[str, ...] = ("ts_event", "ts_init", "open", "high", "low", "close", "volume")


def _ts_ns(frame: pl.DataFrame) -> np.ndarray:
    """``ts_init`` of *frame* as int64 nanoseconds."""
    ts = frame["ts_init"]
    if ts.dtype != pl.Int64:
        ts = ts.dt.cast_time_unit("ns").dt.epoch("ns")
    return ts.to_numpy().astype(np.int64)


def _align(frames: Mapping[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Restrict every frame to the bar closes they all share, sorted by ``ts_init``."""
    common = None
    for frame in frames.values():
        ts = frame.select("ts_init")
        common = ts if common is None else common.join(ts, on="ts_init", how="inner")
    assert common is not None  # noqa: S101 — callers pass at least one frame
    return {
        symbol: frame.join(common, on="ts_init", how="semi").sort("ts_init")
        for symbol, frame in frames.items()
    }


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

    A :class:`MultiAssetStrategy` sees the frames aligned on their shared bar
    closes; a single-asset strategy runs on each frame independently.

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
        aligned = _align(frames)
        signals = research.generate_signals(aligned)
        for symbol, frame in aligned.items():
            values = _clean(signals[symbol], frame.height, symbol)
            targets[symbol] = dict(zip(_ts_ns(frame).tolist(), values.tolist(), strict=True))
        return targets
    for symbol, frame in frames.items():
        frame = frame.sort("ts_init")
        values = _clean(research.generate_signals(frame), frame.height, symbol)
        targets[symbol] = dict(zip(_ts_ns(frame).tolist(), values.tolist(), strict=True))
    return targets


class SignalStrategy(NautilusStrategy):
    """Nautilus strategy that trades the target positions of a research strategy.

    Args:
        research: Research strategy producing targets in units of equity.
        history: ``symbol -> bars`` used to precompute targets. It should cover
            the run plus any warm-up; bars after it trigger a recompute.
        instruments: ``symbol -> Nautilus instrument``; all at one venue.
        bar_types: ``symbol -> BarType`` string to subscribe to.
        weights: ``symbol -> capital weight``. Defaults to ``1 / n_symbols``.
        rebalance_band: Skip trades whose notional is below this fraction of
            equity. ``0.0`` rebalances on every bar.
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
        weights: Mapping[str, float] | None = None,
        rebalance_band: float = 0.0,
        history_window: int | None = None,
    ) -> None:
        super().__init__()
        symbols = list(instruments)
        if set(history) != set(symbols) or set(bar_types) != set(symbols):
            raise ValueError("history, instruments and bar_types must cover the same symbols")
        venues = {str(i.id.venue) for i in instruments.values()}
        if len(venues) != 1:
            raise ValueError(f"all instruments must share one venue, got {sorted(venues)}")
        if rebalance_band < 0.0:
            raise ValueError(f"rebalance_band must be non-negative, got {rebalance_band}")
        currencies = {str(i.quote_currency) for i in instruments.values()}
        if len(currencies) != 1:
            raise ValueError(f"all instruments must share one currency, got {sorted(currencies)}")

        self._research = research
        self._history = {s: history[s].select(HISTORY_COLUMNS).sort("ts_init") for s in symbols}
        self._instruments = dict(instruments)
        self._bar_types = {s: BarType.from_str(bar_types[s]) for s in symbols}
        self._symbol_of = {str(bt): s for s, bt in self._bar_types.items()}
        self._weights = (
            dict(weights) if weights is not None else dict.fromkeys(symbols, 1.0 / len(symbols))
        )
        self._band = rebalance_band
        self._window = history_window
        self._venue = Venue(venues.pop())
        self._currency: Currency = next(iter(instruments.values())).quote_currency
        self._targets = compute_targets(research, self._history)
        # How many symbols have a bar at each close, so a timestamp's batch is
        # known to be complete. Closes beyond the history expect every symbol.
        counts = (
            pl.concat([f.select("ts_init") for f in self._history.values()])
            .group_by("ts_init")
            .len()
        )
        self._expected: dict[int, int] = dict(
            zip(_ts_ns(counts).tolist(), counts["len"].to_list(), strict=True)
        )
        self._pending: dict[str, Bar] = {}
        self._pending_ts: int | None = None
        self.equity: dict[int, float] = {}
        self.rejections: list[str] = []

    # -- lifecycle -----------------------------------------------------------

    def on_start(self) -> None:
        for bar_type in self._bar_types.values():
            self.subscribe_bars(bar_type)

    def on_bar(self, bar: Bar) -> None:
        """Collect the bars closing at one timestamp, then rebalance them together.

        Sizing every instrument after all of them are marked to this close
        uses one consistent equity figure. A bar with a later close flushes an
        incomplete batch (e.g. a symbol that didn't trade that day).
        """
        if self._pending_ts is not None and bar.ts_init != self._pending_ts:
            self._flush()
        self._pending_ts = bar.ts_init
        self._pending[self._symbol_of[str(bar.bar_type)]] = bar
        if len(self._pending) >= self._expected.get(bar.ts_init, len(self._instruments)):
            self._flush()

    def on_order_denied(self, event: OrderDenied) -> None:
        self.rejections.append(f"denied: {event.reason}")
        self.log.warning(f"Order denied: {event.reason}")

    def on_order_rejected(self, event: OrderRejected) -> None:
        self.rejections.append(f"rejected: {event.reason}")
        self.log.warning(f"Order rejected: {event.reason}")

    # -- internals -----------------------------------------------------------

    def _flush(self) -> None:
        """Rebalance every instrument in the pending batch at one equity figure."""
        batch, ts = self._pending, self._pending_ts
        self._pending, self._pending_ts = {}, None
        if not batch or ts is None:
            return
        equity = float(self.portfolio.equity(self._venue)[self._currency])
        self.equity[ts] = equity
        if any(ts > self._history_end(s) for s in batch):
            self._extend_history(batch)
        for symbol, bar in batch.items():
            target = self._targets[symbol].get(ts)
            if target is not None:
                self._rebalance(symbol, target, float(bar.close), equity)

    def _history_end(self, symbol: str) -> int:
        """Close of the last bar in *symbol*'s history (``-1`` if empty)."""
        frame = self._history[symbol]
        return int(_ts_ns(frame.tail(1))[0]) if frame.height else -1

    def _extend_history(self, batch: Mapping[str, Bar]) -> None:
        """Append live bars to the history and recompute targets (the live path)."""
        for symbol, bar in batch.items():
            if bar.ts_init <= self._history_end(symbol):
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
        self._targets = compute_targets(self._research, self._history)

    def _rebalance(self, symbol: str, target: float, price: float, equity: float) -> None:
        """Trade *symbol* towards *target* units of equity at *price*."""
        instrument = self._instruments[symbol]
        if price <= 0.0 or equity <= 0.0:
            return
        multiplier = float(instrument.multiplier)
        increment = Decimal(str(instrument.size_increment))
        wanted = target * self._weights[symbol] * equity / (price * multiplier)
        lots = (Decimal(repr(wanted)) / increment).to_integral_value(ROUND_HALF_EVEN)
        delta = lots * increment - self.portfolio.net_position(instrument.id)
        if delta == 0:
            return
        if float(abs(delta)) * price * multiplier < self._band * equity:
            return
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        quantity = Quantity(float(abs(delta)), instrument.size_precision)
        self.submit_order(self.order_factory.market(instrument.id, side, quantity))
