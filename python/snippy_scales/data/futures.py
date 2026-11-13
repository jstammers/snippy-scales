"""Real futures contracts: roll schedules and back-adjusted signal series.

A **parent symbol** (``ES.FUT``) stands for every outright contract of a root.
Databento parent syncs store each outright as its own catalog series
(``ESH5.GLBX-1-DAY-LAST-EXTERNAL``) with its own definition (expiry, multiplier).

:func:`roll_schedule` decides which outright is *active* (traded) on each bar
close:

* ``RollRule(method="volume")`` (default, like Databento's ``.v.0``): roll to a
  later contract on the first close at which its daily volume exceeds the
  active contract's. Never rolls back.
* ``RollRule(method="calendar", days_before_expiry=5)``: roll on the first close
  at least that many days before the active contract's expiry.

``months`` (e.g. ``"HMUZ"``) restricts both rules to a contract cycle.

A roll at close ``T`` means: the position in the old contract is closed and the
new one opened at ``T``'s close, so ``T`` needs a bar in both contracts.

:func:`back_adjust` splices the active contracts into one series for signal
computation using **backward ratio** adjustment: every bar before a roll at ``T``
is multiplied by ``close_new(T) / close_old(T)``. Each bar-to-bar return of the
spliced series is then exactly the return of the contract held over that bar,
including across rolls. A new roll rescales all earlier bars by one constant,
which leaves returns, return-based volatility and moving-average signs unchanged,
so signals computed before and after a roll agree (spread adjustment would not
have that property).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
from nautilus_trader.model import FuturesContract

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from nautilus_trader.persistence import ParquetDataCatalog

#: CME month codes, January to December.
MONTH_CODES = "FGHJKMNQUVXZ"
_DAY_NS = 86_400_000_000_000


def is_parent(symbol: str) -> bool:
    """Whether *symbol* is Databento parent symbology for futures (``ES.FUT``)."""
    return symbol.endswith(".FUT")


def parent_root(symbol: str) -> str:
    """``ES.FUT`` -> ``ES``."""
    return symbol.removesuffix(".FUT")


def month_code(contract: FuturesContract, root: str) -> str | None:
    """The contract month letter of an outright (``ESH5`` -> ``H``)."""
    raw = str(contract.raw_symbol)
    code = raw[len(root)] if raw.startswith(root) and len(raw) > len(root) else ""
    return code if code in MONTH_CODES else None


def outrights(
    catalog: ParquetDataCatalog, root: str, *, venue: str | None = None
) -> list[FuturesContract]:
    """Outright ``FuturesContract`` definitions of *root*, sorted by expiry.

    Spreads, options and Databento continuous IDs (``ES.c.0``) are excluded.
    """
    found: dict[str, FuturesContract] = {}
    for inst in catalog.instruments():
        if not isinstance(inst, FuturesContract) or inst.underlying != root:
            continue
        if venue is not None and str(inst.id.venue) != venue:
            continue
        if month_code(inst, root) is None:  # continuous IDs, spreads
            continue
        found[str(inst.id)] = inst
    return sorted(found.values(), key=lambda c: c.expiration_ns)


@dataclass(frozen=True)
class RollRule:
    """How to choose the active contract.

    Attributes:
        method: ``"volume"`` or ``"calendar"``.
        days_before_expiry: Calendar rule: roll this many days before expiry.
        months: Contract months to trade (e.g. ``"HMUZ"``); ``None`` allows all.
    """

    method: Literal["volume", "calendar"] = "volume"
    days_before_expiry: int = 5
    months: str | None = None

    def __post_init__(self) -> None:
        if self.method not in ("volume", "calendar"):
            raise ValueError(f"method must be 'volume' or 'calendar', got {self.method!r}")
        if self.days_before_expiry < 0:
            raise ValueError("days_before_expiry must be non-negative")
        if self.months is not None and set(self.months) - set(MONTH_CODES):
            raise ValueError(f"months must use codes from {MONTH_CODES!r}")


@dataclass(frozen=True)
class RollSchedule:
    """Which instrument is active from which bar close.

    ``instrument_ids[i]`` is active for closes in ``[starts[i], starts[i+1])``;
    ``starts[0]`` is the earliest possible time. A roll at ``starts[i]`` (i > 0)
    closes ``instrument_ids[i-1]`` and opens ``instrument_ids[i]`` at that close.
    """

    symbol: str
    instrument_ids: tuple[str, ...]
    starts: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.instrument_ids or len(self.instrument_ids) != len(self.starts):
            raise ValueError("a schedule needs one start per instrument")
        if any(b <= a for a, b in zip(self.starts, self.starts[1:], strict=False)):
            raise ValueError("roll times must be strictly increasing")

    @classmethod
    def single(cls, symbol: str, instrument_id: str) -> RollSchedule:
        """A schedule that never rolls."""
        return cls(symbol, (instrument_id,), (-(2**63),))

    @property
    def rolls(self) -> list[tuple[int, str, str]]:
        """``(roll close, old instrument, new instrument)`` per roll."""
        return [
            (self.starts[i], self.instrument_ids[i - 1], self.instrument_ids[i])
            for i in range(1, len(self.starts))
        ]

    def active(self, ts: int) -> str:
        """The instrument active at bar close *ts*."""
        i = bisect.bisect_right(self.starts, ts) - 1
        return self.instrument_ids[max(i, 0)]

    def span(self, instrument_id: str) -> tuple[int, int]:
        """Closes ``[first, last]`` for which *instrument_id*'s bars are needed.

        That is its active segment plus the close of the roll out of it.
        """
        i = self.instrument_ids.index(instrument_id)
        last = self.starts[i + 1] if i + 1 < len(self.starts) else 2**63 - 1
        return self.starts[i], last


def _ts_ns(frame: pl.DataFrame) -> np.ndarray:
    ts = frame["ts_init"]
    if ts.dtype != pl.Int64:
        ts = ts.dt.cast_time_unit("ns").dt.epoch("ns")
    return ts.to_numpy().astype(np.int64)


def roll_schedule(
    symbol: str,
    contracts: Sequence[FuturesContract],
    frames: Mapping[str, pl.DataFrame],
    rule: RollRule | None = None,
) -> RollSchedule:
    """Choose the active outright at every bar close.

    Args:
        symbol: Name for the schedule (e.g. ``"ES.FUT"``).
        contracts: Outright definitions (any order).
        frames: ``instrument_id -> bars`` (``ts_init``, ``close``, ``volume``).
            Contracts without bars are ignored.
        rule: Roll rule. Defaults to volume.

    Raises:
        ValueError: If no contract has bars.
    """
    rule = rule or RollRule()
    root = parent_root(symbol)
    eligible = sorted(
        (
            c
            for c in contracts
            if str(c.id) in frames
            and frames[str(c.id)].height
            and (rule.months is None or (month_code(c, root) or "?") in rule.months)
        ),
        key=lambda c: c.expiration_ns,
    )
    if not eligible:
        raise ValueError(f"no contract of {symbol} has bars")
    ids = [str(c.id) for c in eligible]
    expiry = {str(c.id): c.expiration_ns for c in eligible}
    order = {iid: i for i, iid in enumerate(ids)}
    volume: dict[str, dict[int, float]] = {
        iid: dict(zip(_ts_ns(frames[iid]).tolist(), frames[iid]["volume"].to_list(), strict=True))
        for iid in ids
    }
    closes = sorted({t for v in volume.values() for t in v})

    active: str | None = None
    out_ids: list[str] = []
    out_starts: list[int] = []
    for ts in closes:
        trading = [iid for iid in ids if ts in volume[iid]]
        if active is None:
            first = max(trading, key=lambda i: volume[i][ts])
            active = first
            out_ids.append(first)
            out_starts.append(-(2**63))
            continue
        later = [iid for iid in trading if order[iid] > order[active]]
        if not later:
            continue
        if rule.method == "volume":
            best = max(later, key=lambda i: volume[i][ts])
            due = volume[best][ts] > volume[active].get(ts, -1.0)
        else:
            best = later[0]
            due = ts >= expiry[active] - rule.days_before_expiry * _DAY_NS
        # A roll must close the old contract at this bar's price, so the old
        # contract needs a bar here unless it has stopped trading for good.
        stopped = ts > max(volume[active])
        if due and (ts in volume[active] or stopped):
            active = best
            out_ids.append(best)
            out_starts.append(ts)
    return RollSchedule(symbol, tuple(out_ids), tuple(out_starts))


def back_adjust(
    schedule: RollSchedule, frames: Mapping[str, pl.DataFrame], symbol: str | None = None
) -> pl.DataFrame:
    """Splice the active contracts into one backward-ratio-adjusted bar frame.

    Args:
        schedule: Roll schedule.
        frames: ``instrument_id -> bars`` with ``ts_event``, ``ts_init``, OHLCV.
        symbol: Value of the ``symbol`` column. Defaults to the schedule's.

    Returns:
        The spliced bars (columns of the inputs, plus ``instrument_id`` of the
        contract each bar came from), prices multiplied by the cumulative ratio
        of later rolls, so the last segment is unadjusted.
    """
    columns = ("ts_event", "ts_init", "open", "high", "low", "close", "volume")
    pieces: list[pl.DataFrame] = []
    ts_col = pl.col("ts_init").cast(pl.Datetime("ns", "UTC")).dt.epoch("ns")
    for iid, start, nxt in zip(
        schedule.instrument_ids,
        schedule.starts,
        [*schedule.starts[1:], 2**63 - 1],
        strict=True,
    ):
        segment = frames[iid].filter((ts_col >= start) & (ts_col < nxt)).select(columns)
        pieces.append(segment.with_columns(pl.lit(iid).alias("instrument_id")))

    # factors[i]: ratio at roll i; piece j is scaled by the product over rolls j..end.
    factors = np.ones(len(pieces))
    for i, (ts, old, new) in enumerate(schedule.rolls):
        factors[i] = _close_at(frames[new], ts) / _close_at(frames[old], ts)
    scales = np.cumprod(factors[::-1])[::-1]
    adjusted = [
        p.with_columns(pl.col("open", "high", "low", "close") * float(scale))
        for p, scale in zip(pieces, scales, strict=True)
    ]
    out = pl.concat(adjusted).sort("ts_init")
    return out.with_columns(pl.lit(symbol or schedule.symbol).alias("symbol"))


def _close_at(frame: pl.DataFrame, ts: int) -> float:
    """Close at bar close *ts*, or the last close before it (a contract that stopped)."""
    stamps = _ts_ns(frame)
    i = int(np.searchsorted(stamps, ts, side="right")) - 1
    if i < 0:
        raise ValueError(f"no bar at or before {ts} to roll from")
    return float(frame["close"][i])


def tradable_symbols(
    catalog: ParquetDataCatalog, series: Sequence[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Replace outright contract series with one parent symbol per root.

    Args:
        catalog: Catalog holding the instrument definitions.
        series: ``(symbol, venue)`` bar series, e.g. from
            :func:`~snippy_scales.data.catalog.list_bar_series`.

    Returns:
        ``(symbol, venue)`` with every outright of a root collapsed into
        ``("<ROOT>.FUT", venue)``, order preserved.
    """
    outright_root: dict[tuple[str, str], str] = {}
    for inst in catalog.instruments():
        if isinstance(inst, FuturesContract) and month_code(inst, inst.underlying) is not None:
            outright_root[(str(inst.raw_symbol), str(inst.id.venue))] = inst.underlying
    out: list[tuple[str, str]] = []
    for symbol, venue in series:
        root = outright_root.get((symbol, venue))
        entry = (f"{root}.FUT", venue) if root is not None else (symbol, venue)
        if entry not in out:
            out.append(entry)
    return out
