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
import datetime as dt
import re
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


def is_rolled(symbol: str) -> bool:
    """Whether *symbol* is a front-month roll stitched from ``.c.0``/``.c.1`` (``ES.ROLL``).

    ``<ROOT>.<VENUE>.ROLL`` (``BRN.IFEU.ROLL``) names the venue the bars come from.
    """
    return symbol.endswith(".ROLL")


def rolled_root(symbol: str) -> str:
    """``ES.ROLL`` -> ``ES``; ``BRN.IFEU.ROLL`` -> ``BRN``."""
    return symbol.removesuffix(".ROLL").split(".")[0]


def rolled_venue(symbol: str) -> str | None:
    """``BRN.IFEU.ROLL`` -> ``IFEU``; ``None`` if *symbol* names no venue (``ES.ROLL``)."""
    parts = symbol.removesuffix(".ROLL").split(".")
    return parts[1] if len(parts) > 1 else None


def drop_scale_glitches(frame: pl.DataFrame, factor: float = 3.0) -> pl.DataFrame:
    """*frame* without bars whose close is off by a price-scale error.

    A bar is dropped when its close is more than *factor* times, or less than
    1/*factor* of, both the previous and the next close (positive prices only).
    A one-day move of that size that reverses the next day is a decimal error in
    the feed, not a market move: ICE's MSCI futures print a tenth of their price
    on 2021-08-13 in Databento's data.
    """
    close = pl.col("close")
    up, down = close / close.shift(1), close / close.shift(-1)
    glitch = (close > 0) & (
        ((up > factor) & (down > factor))
        | ((up > 0) & (up < 1 / factor) & (down > 0) & (down < 1 / factor))
    )
    return frame.sort("ts_init").filter(~glitch.fill_null(value=False))


#: CME FX futures (``<root>.c.0`` on GLBX) quoted in USD per unit of each currency.
FX_FUTURES = {
    "AUD": "6A",
    "GBP": "6B",
    "CAD": "6C",
    "EUR": "6E",
    "JPY": "6J",
    "NZD": "6N",
    "CHF": "6S",
}
FX_VENUE = "GLBX"


def to_usd(frame: pl.DataFrame, fx: pl.DataFrame) -> pl.DataFrame:
    """One contract's bars, priced in a foreign currency, as a USD-quoted contract.

    A futures position has no notional to convert: it is settled daily, and each
    day's variation margin, ``(close_t - close_{t-1}) * multiplier``, is paid in the
    contract's currency. A USD account converts it at that day's rate ``x_t``. The
    USD price is built so its daily change is exactly that converted settlement:

        usd_t = usd_{t-1} + (close_t - close_{t-1}) * x_t,   usd_0 = close_0 * x_0,

    so holding it for one day earns the local return times ``x_t / x_{t-1}``, not
    the local return plus the currency's move (which an unhedged notional would).
    Open, high and low are moved from the previous USD close the same way.

    Args:
        frame: Bars with ``ts_event`` and OHLC in the contract's currency.
        fx: ``ts_event``, ``close``: USD per unit of that currency; the last rate
            on or before each bar's date is used.

    Raises:
        ValueError: If there is no rate before the first bar, or a USD price is not
            positive.
    """
    rates = (
        fx.select(
            pl.col("ts_event").cast(pl.Datetime("ns", "UTC")).dt.date().alias("__date"),
            pl.col("close").cast(pl.Float64).alias("__x"),
        )
        .unique("__date", keep="last")
        .sort("__date")
    )
    dated = frame.sort("ts_event").with_columns(
        pl.col("ts_event").cast(pl.Datetime("ns", "UTC")).dt.date().alias("__date")
    )
    out = dated.join_asof(rates, on="__date", strategy="backward")
    if out["__x"].null_count():
        raise ValueError("no FX rate on or before the first bar")
    prev_local = pl.col("close").shift(1)
    usd_close = (
        pl.when(prev_local.is_null())
        .then(pl.col("close") * pl.col("__x"))
        .otherwise((pl.col("close") - prev_local) * pl.col("__x"))
        .cum_sum()
    )
    out = out.with_columns(__usd=usd_close).with_columns(
        [
            pl.when(prev_local.is_null())
            .then(pl.col(c) * pl.col("__x"))
            .otherwise(pl.col("__usd").shift(1) + (pl.col(c) - prev_local) * pl.col("__x"))
            .alias(c)
            for c in ("open", "high", "low")
        ]
    )
    out = out.with_columns(close=pl.col("__usd")).drop("__date", "__x", "__usd")
    if (out.select(pl.min_horizontal("open", "high", "low", "close").min()).item() or 0) <= 0:
        raise ValueError("a USD price is not positive")
    return out


def _tag_contracts(frame: pl.DataFrame, spans: Sequence[tuple[str, str, str]]) -> pl.DataFrame:
    """*frame* plus a ``contract`` column from date spans ``(d0, d1 exclusive, raw)``."""
    if not spans:
        return frame.head(0).with_columns(contract=pl.lit(None, pl.String))
    mapping = pl.DataFrame(
        {
            "d0": [dt.date.fromisoformat(s[0]) for s in spans],
            "d1": [dt.date.fromisoformat(s[1]) for s in spans],
            "contract": [s[2] for s in spans],
        }
    ).sort("d0")
    dated = frame.with_columns(
        __date=pl.col("ts_event").cast(pl.Datetime("ns", "UTC")).dt.date()
    ).sort("__date")
    joined = dated.join_asof(mapping, left_on="__date", right_on="d0", strategy="backward")
    return joined.filter(pl.col("__date") < pl.col("d1")).drop("__date", "d0", "d1")


_CME_MONTH = re.compile(rf"([{MONTH_CODES}])(\d{{1,2}})$")  # ESZ6, CLF27
_ICE_MONTH = re.compile(rf"FM([{MONTH_CODES}])00(\d{{2}})!$")  # "BRN FMG0019!"


def contract_month(raw: str) -> tuple[int, int, int] | None:
    """Delivery month of a raw contract symbol: ``(month 1-12, year, year modulus)``.

    The year is as written in the symbol: one digit on CME (``ESZ6``, modulus 10),
    two on ICE (``BRN FMG0019!``, modulus 100). ``None`` if the symbol has neither form.
    """
    match = _ICE_MONTH.search(raw)
    if match is not None:
        return MONTH_CODES.index(match.group(1)) + 1, int(match.group(2)), 100
    match = _CME_MONTH.search(raw)
    if match is None:
        return None
    return MONTH_CODES.index(match.group(1)) + 1, int(match.group(2)), 10 ** len(match.group(2))


def months_between(near: str, far: str) -> int | None:
    """Whole months from the delivery month of *near* to that of *far* (raw symbols).

    ``None`` if either symbol can't be parsed or *far* does not deliver later.
    """
    a, b = contract_month(near), contract_month(far)
    if a is None or b is None or a[2] != b[2]:
        return None
    months = ((b[1] - a[1]) % a[2]) * 12 + b[0] - a[0]
    return months if months > 0 else None


def roll_yield(
    front: pl.DataFrame,
    second: pl.DataFrame,
    front_spans: Sequence[tuple[str, str, str]],
    second_spans: Sequence[tuple[str, str, str]],
) -> pl.DataFrame:
    """Annualised roll yield of a futures curve from its front and second month.

    ``carry = (F1 / F2 - 1) * 12 / months``, with ``F1``, ``F2`` the closes of the
    front and second contracts and ``months`` the gap between their delivery months:
    the return a long position earns if the curve does not move (Koijen, Moskowitz,
    Pedersen & Vrugt, 2018). Positive in backwardation, negative in contango. It is a
    ratio of two prices in one currency, so it needs no currency conversion.

    Only closes at which both contracts traded (``volume > 0``) are used: a stale
    second-month price would otherwise read as a move in the curve.

    Args:
        front: Bars of the front-month continuous series (``ES.c.0``).
        second: Bars of the second-month series (``ES.c.1``).
        front_spans: Contract mapping of *front* (``(d0, d1 exclusive, raw symbol)``).
        second_spans: Contract mapping of *second*.

    Returns:
        ``ts_init``, ``carry`` (Float64), sorted, one row per usable close.
    """
    near = _tag_contracts(front.filter(pl.col("volume") > 0), front_spans)
    far = _tag_contracts(second.filter(pl.col("volume") > 0), second_spans)
    joined = near.select("ts_init", "close", "contract").join(
        far.select("ts_init", pl.col("close").alias("far"), pl.col("contract").alias("far_c")),
        on="ts_init",
        how="inner",
    )
    gaps = {
        pair: months_between(*pair)
        for pair in set(zip(joined["contract"].to_list(), joined["far_c"].to_list(), strict=True))
    }
    months = pl.Series(
        "months",
        [
            gaps[pair]
            for pair in zip(joined["contract"].to_list(), joined["far_c"].to_list(), strict=True)
        ],
        dtype=pl.Float64,
    )
    return (
        joined.with_columns(months)
        .filter(pl.col("months").is_not_null() & (pl.col("far") > 0) & (pl.col("close") > 0))
        .select(
            "ts_init",
            ((pl.col("close") / pl.col("far") - 1.0) * 12.0 / pl.col("months")).alias("carry"),
        )
        .unique("ts_init", keep="first")
        .sort("ts_init")
    )


#: A raw symbol that reappears after this long is a different contract (``ESZ6``
#: is Dec 2016 and Dec 2026: CME symbols carry one year digit).
_SYMBOL_REUSE_DAYS = 3 * 365


def _unique_contracts(
    *span_lists: Sequence[tuple[str, str, str]],
) -> list[list[tuple[str, str, str]]]:
    """Rename contracts so a reused raw symbol gets a distinct name per listing.

    Each contract becomes ``<raw>_<year of its first span>`` (``ESZ6_2016``).
    """
    events = sorted(
        (dt.date.fromisoformat(d0), dt.date.fromisoformat(d1), raw, i, j)
        for i, spans in enumerate(span_lists)
        for j, (d0, d1, raw) in enumerate(spans)
    )
    current: dict[str, tuple[str, dt.date]] = {}  # raw -> (name, last end date)
    names: dict[tuple[int, int], str] = {}
    for d0, d1, raw, i, j in events:
        seen = current.get(raw)
        if seen is None or (d0 - seen[1]).days > _SYMBOL_REUSE_DAYS:
            seen = (f"{raw}_{d0.year}", d1)
        current[raw] = (seen[0], max(seen[1], d1))
        names[(i, j)] = seen[0]
    return [
        [(d0, d1, names[(i, j)]) for j, (d0, d1, _) in enumerate(spans)]
        for i, spans in enumerate(span_lists)
    ]


def calendar_roll(
    symbol: str,
    front: pl.DataFrame,
    second: pl.DataFrame,
    front_spans: Sequence[tuple[str, str, str]],
    second_spans: Sequence[tuple[str, str, str]],
    venue: str,
    *,
    days_before: int = 5,
) -> tuple[dict[str, pl.DataFrame], RollSchedule]:
    """Split front/second-month continuous bars into contracts and schedule the rolls.

    Databento's calendar continuous series (``ES.c.0``, ``ES.c.1``) switch
    contract at expiry; *front_spans* / *second_spans* say which contract each
    one maps to on each date (:func:`snippy_scales.data.catalog.read_symbology`).
    A contract's bars come from both series: while it is the second month, then
    while it is the front month. The schedule rolls from the front contract to
    the next one at the ``days_before``-th last close, before the front series
    switches, at which both traded (a bar with volume; the earliest such close if
    there are fewer), so no position is held into expiry.

    Returns:
        ``instrument_id -> bars`` per contract (``<raw symbol>.<venue>``) and the
        roll schedule.

    The series starts at the first contract that can be rolled into the next one:
    leading contracts whose roll can't be priced (the second month has no bars yet
    when the data begins) are dropped.

    Raises:
        ValueError: If a contract never shares a close with the next one before
            the switch (the roll can't be priced) after the series has started, if
            fewer than half the contracts are left, or there are no bars.
    """
    if days_before < 1:
        raise ValueError("days_before must be at least 1")
    front_spans, second_spans = _unique_contracts(front_spans, second_spans)
    tagged = pl.concat(
        [_tag_contracts(front, front_spans), _tag_contracts(second, second_spans)],
        how="vertical_relaxed",
    )
    if tagged.is_empty():
        raise ValueError(f"no bars for {symbol}")
    frames = {
        f"{raw}.{venue}": part.drop("contract")
        .unique("ts_init", keep="first")
        .sort("ts_init")
        .with_columns(pl.lit(raw).alias("symbol"))
        for (raw,), part in tagged.partition_by("contract", as_dict=True).items()
    }
    # Contracts in the order they are the front month, with the date each starts.
    front_tagged = _tag_contracts(front, front_spans).sort("ts_init")
    order: list[str] = []
    switch: dict[str, int] = {}
    for raw, ts in zip(
        front_tagged["contract"].to_list(), _ts_ns(front_tagged).tolist(), strict=True
    ):
        if raw not in switch:  # never roll back to a contract already held
            order.append(raw)
            switch[raw] = ts
    ids = [f"{raw}.{venue}" for raw in order]
    kept, starts = ids[:1], [-(2**63)]

    def traded(frame: pl.DataFrame) -> np.ndarray:  # Nautilus can't fill on a 0-volume bar
        return _ts_ns(frame.filter(pl.col("volume") > 0))

    for old, new, new_raw in zip(ids, ids[1:], order[1:], strict=False):
        common = np.intersect1d(traded(frames[old]), traded(frames[new]))
        common = common[(common < switch[new_raw]) & (common > starts[-1])]
        if common.size == 0:
            if len(kept) == 1:  # the series starts before its second month has bars:
                kept = [new]  # begin at the first contract that can be rolled into
                continue
            raise ValueError(f"{symbol}: {old} and {new} share no close before the roll")
        kept.append(new)
        starts.append(int(common[max(common.size - days_before, 0)]))
    if 2 * len(kept) < len(ids):
        raise ValueError(
            f"{symbol}: only the last {len(kept)} of {len(ids)} contracts can be rolled "
            "(the second month is too sparse)"
        )
    return frames, RollSchedule(symbol, tuple(kept), tuple(starts))


def synthetic_outright(
    source: FuturesContract,
    raw_symbol: str,
    venue: str,
    frame: pl.DataFrame,
    *,
    currency: str | None = None,
) -> FuturesContract:
    """A definition for one contract of a rolled series, cloned from *source*.

    Only the continuous series' definition is stored, so tick size, multiplier
    and currency come from it (they are the same for every contract of a root);
    activation and expiry span the contract's bars. A *currency* other than the
    source's re-quotes the contract (prices from :func:`to_usd`): its tick becomes
    ``0.01``, since converted prices are off the exchange's tick grid.
    """
    ts = _ts_ns(frame)
    spec = source.to_dict()
    spec.update(
        id=f"{raw_symbol}.{venue}",
        raw_symbol=raw_symbol,
        activation_ns=int(ts[0]) - _DAY_NS,
        expiration_ns=int(ts[-1]) + _DAY_NS,
    )
    if currency is not None and currency != spec["currency"]:
        spec.update(currency=currency, price_precision=2, price_increment="0.01")
    return FuturesContract.from_dict(spec)


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
