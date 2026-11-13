"""Phase 4: roll schedules, back-adjustment, and trading real outrights with rolls.

Synthetic market: a spot random walk; quarterly ES outrights priced at
``spot * exp(carry * days_to_expiry)`` (contango), each listed 200 days before
expiry. Volume is high while a contract is the front (more than 7 days to expiry),
lower before that and lowest in its last 7 days, so a volume roll happens exactly
when the front reaches 7 days to expiry. Volumes are large enough that test
orders never exceed a bar's volume (which would make fills walk the book).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from nautilus_trader.model import (
    AssetClass,
    Currency,
    FuturesContract,
    InstrumentId,
    Price,
    Quantity,
    Symbol,
)

from snippy_scales.data import catalog as cat
from snippy_scales.data import futures
from snippy_scales.data.futures import RollRule, back_adjust, roll_schedule
from snippy_scales.execution import (
    Costs,
    MarketData,
    compute_targets,
    load_market_data,
    run_backtest,
)
from snippy_scales.strategies import Strategy, TimeSeriesMomentum, TrendFollowing

_DAY = 86_400_000_000_000
_BASE = dt.datetime(2020, 1, 1, tzinfo=dt.UTC)
_BASE_NS = int(_BASE.timestamp()) * 1_000_000_000
_N = 700
_EXPIRY_DAYS = [75, 166, 257, 348, 439, 530, 621, 712, 803]  # ~quarterly
_CODES = ["H0", "M0", "U0", "Z0", "H1", "M1", "U1", "Z1", "H2"]
_MULT = 50


def _contract(code: str, expiry_day: int, venue: str = "GLBX") -> FuturesContract:
    expiry = _BASE_NS + expiry_day * _DAY
    return FuturesContract(
        InstrumentId.from_str(f"ES{code}.{venue}"),
        Symbol(f"ES{code}"),
        AssetClass.INDEX,
        "ES",
        expiry - 200 * _DAY,
        expiry,
        Currency.from_str("USD"),
        2,
        Price.from_str("0.01"),
        Quantity.from_int(_MULT),
        Quantity.from_int(1),
        0,
        0,
    )


def _market(
    seed: int = 0, carry: float = 0.0002
) -> tuple[list[FuturesContract], dict[str, pl.DataFrame]]:
    rng = np.random.default_rng(seed)
    spot = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, _N)))
    contracts = [_contract(c, d) for c, d in zip(_CODES, _EXPIRY_DAYS, strict=True)]
    frames: dict[str, pl.DataFrame] = {}
    for contract, expiry_day in zip(contracts, _EXPIRY_DAYS, strict=True):
        days = np.arange(_N)
        live = (days >= expiry_day - 200) & (days < expiry_day)
        dte = expiry_day - days[live]
        price = np.round(spot[live] * np.exp(carry * dte), 2)
        volume = np.where(dte > 97, 2e5, np.where(dte > 7, 1e6, 1e5))
        ts_open = _BASE_NS + days[live].astype(np.int64) * _DAY
        frames[str(contract.id)] = pl.DataFrame(
            {
                "ts_event": ts_open,
                "ts_init": ts_open + _DAY,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume,
            }
        ).with_columns(pl.col("ts_event", "ts_init").cast(pl.Datetime("ns", "UTC")))
    return contracts, frames


def _close_ns(day: int) -> int:
    """Close of the bar opened on *day* (days since 2020-01-01)."""
    return _BASE_NS + (day + 1) * _DAY


class _Fixed(Strategy):
    def __init__(self, value: float) -> None:
        self.value = value

    def generate_signals(self, bars: pl.DataFrame) -> pl.Series:
        return pl.Series("position", np.full(bars.height, self.value))


# ── Schedules ─────────────────────────────────────────────────────────────────


def test_volume_roll_when_front_reaches_seven_days() -> None:
    contracts, frames = _market()
    schedule = roll_schedule("ES.FUT", contracts, frames)
    # The first bar's busiest contract is H0; then one roll per expiry.
    assert schedule.instrument_ids[0] == "ESH0.GLBX"
    expected = [_close_ns(d - 7) for d in _EXPIRY_DAYS[: len(schedule.rolls)]]
    assert [r[0] for r in schedule.rolls] == expected
    assert [r[2] for r in schedule.rolls] == [f"ES{c}.GLBX" for c in _CODES[1 : len(expected) + 1]]


def test_calendar_roll_days_before_expiry() -> None:
    contracts, frames = _market()
    schedule = roll_schedule(
        "ES.FUT", contracts, frames, RollRule("calendar", days_before_expiry=5)
    )
    for ts, old, _new in schedule.rolls:
        expiry = next(c.expiration_ns for c in contracts if str(c.id) == old)
        assert ts >= expiry - 5 * _DAY
        assert ts - _DAY < expiry - 5 * _DAY  # the first such close


def test_months_filter_skips_contracts() -> None:
    contracts, frames = _market()
    schedule = roll_schedule("ES.FUT", contracts, frames, RollRule("calendar", months="HZ"))
    codes = {i.removeprefix("ES")[0] for i in schedule.instrument_ids}
    assert codes <= {"H", "Z"}


def test_schedule_active_and_span() -> None:
    contracts, frames = _market()
    schedule = roll_schedule("ES.FUT", contracts, frames)
    ts, old, new = schedule.rolls[0]
    assert schedule.active(ts - 1) == old
    assert schedule.active(ts) == new
    assert schedule.span(old)[1] == ts  # the old contract's bar at the roll is needed


def test_rule_validation() -> None:
    with pytest.raises(ValueError, match="method"):
        RollRule("oi")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="months"):
        RollRule(months="HA")


def test_outrights_excludes_continuous_and_other_roots(tmp_path: Path) -> None:
    catalog = cat.open_catalog(tmp_path)
    contracts, _ = _market()
    other = FuturesContract.from_dict(
        {**contracts[0].to_dict(), "id": "NQH0.GLBX", "raw_symbol": "NQH0", "underlying": "NQ"}
    )
    catalog.write_instruments(
        [*contracts, cat.continuous_instrument(contracts[0], "ES.c.0"), other]
    )
    assert [str(c.id) for c in futures.outrights(catalog, "ES")] == [f"ES{c}.GLBX" for c in _CODES]


# ── Back-adjustment ───────────────────────────────────────────────────────────


def _held_returns(schedule: futures.RollSchedule, frames: dict[str, pl.DataFrame]) -> np.ndarray:
    """Bar returns of the contract held over each bar (decided at the previous close)."""
    adjusted = back_adjust(schedule, frames)
    ts = adjusted["ts_init"].dt.epoch("ns").to_list()
    closes = {
        iid: dict(zip(f["ts_init"].dt.epoch("ns").to_list(), f["close"].to_list(), strict=True))
        for iid, f in frames.items()
    }
    out = []
    for prev, cur in zip(ts, ts[1:], strict=False):
        held = schedule.active(prev)
        out.append(closes[held][cur] / closes[held][prev] - 1.0)
    return np.array(out)


def test_back_adjusted_returns_equal_held_contract_returns() -> None:
    contracts, frames = _market(carry=0.001)  # large carry: big roll gaps
    schedule = roll_schedule("ES.FUT", contracts, frames)
    adjusted = back_adjust(schedule, frames)["close"].to_numpy()
    np.testing.assert_allclose(
        adjusted[1:] / adjusted[:-1] - 1.0, _held_returns(schedule, frames), rtol=1e-10
    )
    # The last segment is unadjusted.
    last = schedule.instrument_ids[-1]
    assert adjusted[-1] == frames[last]["close"][-1]


def test_unadjusted_splice_has_roll_gaps() -> None:
    """Sanity check: without adjustment, roll days show the contango gap."""
    contracts, frames = _market(carry=0.001)
    schedule = roll_schedule("ES.FUT", contracts, frames)
    spliced = back_adjust(schedule, frames)
    raw = (
        pl.concat(
            frames[iid].join(
                spliced.filter(pl.col("instrument_id") == iid).select("ts_init"), on="ts_init"
            )
            for iid in schedule.instrument_ids
        )
        .sort("ts_init")["close"]
        .to_numpy()
    )
    raw_ret = raw[1:] / raw[:-1] - 1.0
    assert np.max(np.abs(raw_ret - _held_returns(schedule, frames))) > 0.05


@pytest.mark.parametrize(
    "research",
    [
        TrendFollowing(fast_period=10, slow_period=40),
        TimeSeriesMomentum(lookback=60, vol_lookback=20),
    ],
    ids=lambda r: type(r).__name__,
)
def test_signals_unchanged_by_later_rolls(research: Strategy) -> None:
    """Ratio adjustment keeps precomputed signals causal across new rolls."""
    contracts, frames = _market(carry=0.001)
    full = MarketData.from_outrights("ES.FUT", frames, contracts)
    cutoff = full.schedules["ES.FUT"].rolls[2][0] - 10 * _DAY  # before the 3rd roll
    early_frames = {
        iid: f.filter(pl.col("ts_init").dt.epoch("ns") <= cutoff) for iid, f in frames.items()
    }
    early = MarketData.from_outrights(
        "ES.FUT", {k: v for k, v in early_frames.items() if v.height}, contracts
    )
    a = compute_targets(research, early.frames)["ES.FUT"]
    b = compute_targets(research, full.frames)["ES.FUT"]
    common = sorted(a)
    np.testing.assert_allclose(
        [a[t] for t in common], [b[t] for t in common], rtol=1e-9, atol=1e-12
    )


# ── Trading rolls ─────────────────────────────────────────────────────────────


def test_backtest_rolls_and_matches_adjusted_series() -> None:
    contracts, frames = _market(carry=0.001)
    data = MarketData.from_outrights("ES.FUT", frames, contracts)
    schedule = data.schedules["ES.FUT"]
    run = run_backtest(_Fixed(1.0), data, initial_capital=1e8, leverage=100.0)
    assert run.rejections == ()
    assert [r[0] for r in run.rolls] == [r[0] for r in schedule.rolls]
    # Every position but the last contract's is flat at the end.
    positions = run.positions
    open_ids = set(
        positions.loc[positions["side"].astype(str) != "FLAT", "instrument_id"].astype(str)
    )
    assert open_ids == {schedule.instrument_ids[-1]}
    # Equity follows target * adjusted returns (whole-contract rounding aside).
    adjusted = data.frames["ES.FUT"]["close"].to_numpy()
    expected = 1e8 * np.concatenate(([1.0], np.cumprod(adjusted[1:] / adjusted[:-1])))
    np.testing.assert_allclose(run.equity["equity"].to_numpy(), expected, rtol=3e-5)


def test_roll_bar_trades_both_contracts_and_pays_costs() -> None:
    contracts, frames = _market(carry=0.001)
    data = MarketData.from_outrights("ES.FUT", frames, contracts)
    run = run_backtest(
        _Fixed(1.0), data, costs=Costs(commission_per_contract=2.0), rebalance_band=0.5
    )
    fills = run.fills
    fills_ts = fills["ts_event"].astype("int64")
    for ts, old, new in run.rolls:
        at_roll = fills[fills_ts == ts]
        assert set(at_roll["instrument_id"].astype(str)) == {old, new}
    # With a 50% band the only trades are the initial entry and two per roll.
    assert run.total_orders == 1 + 2 * len(run.rolls)
    commission = sum(float(str(c).split()[0]) for c in fills["commission"])
    contracts_traded = fills["last_qty"].astype(float).sum()
    assert commission == pytest.approx(2.0 * contracts_traded)


def test_window_inside_later_segment() -> None:
    contracts, frames = _market()
    data = MarketData.from_outrights("ES.FUT", frames, contracts)
    schedule = data.schedules["ES.FUT"]
    start = schedule.rolls[1][0] + 5 * _DAY
    run = run_backtest(_Fixed(-0.5), data, start=start, initial_capital=1e8)
    first = run.fills.sort_values("ts_event").iloc[0]
    assert str(first["instrument_id"]) == schedule.active(start)
    assert run.rejections == ()


def test_load_parent_symbol_from_catalog(tmp_path: Path) -> None:
    contracts, frames = _market()
    catalog = cat.open_catalog(tmp_path)
    catalog.write_instruments(contracts)
    for contract in contracts:
        cat.write_bars(
            catalog,
            cat.frame_to_bars(
                frames[str(contract.id)],
                bar_type=cat.bar_type(str(contract.raw_symbol), "GLBX", "ohlcv-1d"),
                price_precision=2,
                interval_ns=_DAY,
            ),
        )
    data = load_market_data("ES.FUT", catalog_path=tmp_path)
    in_memory = MarketData.from_outrights("ES.FUT", frames, contracts)
    assert data.schedules == in_memory.schedules
    np.testing.assert_allclose(
        data.frames["ES.FUT"]["close"].to_numpy(), in_memory.frames["ES.FUT"]["close"].to_numpy()
    )
    run = run_backtest(TrendFollowing(fast_period=10, slow_period=40), data)
    assert run.rejections == ()
    with pytest.raises(FileNotFoundError, match="NQ outright"):
        load_market_data("NQ.FUT", catalog_path=tmp_path)
