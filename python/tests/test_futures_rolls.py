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


# ── Rolled continuous series (ES.ROLL from ES.c.0 / ES.c.1) ───────────────────


def _day(d: int) -> str:
    return (_BASE + dt.timedelta(days=d)).date().isoformat()


def _continuous(
    frames: dict[str, pl.DataFrame], rank: int
) -> tuple[pl.DataFrame, list[tuple[str, str, str]]]:
    """A calendar continuous series (rank 0 = front) and its contract mapping.

    Like Databento's ``.c.N``: on day *d* it holds the contract with the
    ``rank``-th nearest expiry still ahead, switching at expiry.
    """
    spans, pieces = [], []
    for i, expiry in enumerate(_EXPIRY_DAYS):
        j = i + rank
        lo = 0 if i == 0 else _EXPIRY_DAYS[i - 1]
        if j >= len(_CODES) or lo >= _N:
            break
        spans.append((_day(lo), _day(expiry), f"ES{_CODES[j]}"))
        f = frames[f"ES{_CODES[j]}.GLBX"]
        day = (pl.col("ts_event").dt.epoch("ns") - _BASE_NS) // _DAY
        pieces.append(f.filter((day >= lo) & (day < expiry)))
    return pl.concat(pieces).sort("ts_init"), spans


def test_calendar_roll_recovers_contracts_and_rolls_before_the_switch() -> None:
    _, frames = _market(carry=0.001)
    front, front_spans = _continuous(frames, 0)
    second, second_spans = _continuous(frames, 1)
    out, schedule = futures.calendar_roll(
        "ES.ROLL", front, second, front_spans, second_spans, "GLBX", days_before=5
    )
    assert schedule.instrument_ids[:3] == ("ESH0_2020.GLBX", "ESM0_2020.GLBX", "ESU0_2020.GLBX")
    # Each roll is the 5th last close the two contracts share before the front switches.
    for (ts, _old, _new), expiry in zip(schedule.rolls, _EXPIRY_DAYS, strict=False):
        assert ts == _close_ns(expiry - 5)
    # Bars of a contract are its own prices, from both series.
    m0 = out["ESM0_2020.GLBX"]
    src = frames["ESM0.GLBX"].join(m0.select("ts_init", "close"), on="ts_init")
    assert src.height == m0.height
    np.testing.assert_array_equal(src["close"], src["close_right"])
    # The back-adjusted series earns exactly the held contract's returns.
    adjusted = back_adjust(schedule, out)["close"].to_numpy()
    np.testing.assert_allclose(
        adjusted[1:] / adjusted[:-1] - 1.0, _held_returns(schedule, out), rtol=1e-10
    )


def test_calendar_roll_names_reused_symbols_apart() -> None:
    a, b = futures._unique_contracts(
        [("2016-09-01", "2016-12-16", "ESZ6"), ("2026-09-01", "2026-12-18", "ESZ6")],
        [("2016-06-01", "2016-09-01", "ESZ6")],
    )
    assert [s[2] for s in a] == ["ESZ6_2016", "ESZ6_2026"]
    assert b[0][2] == "ESZ6_2016"


def _expiry_period(k: int) -> pl.Expr:
    """Bars while the k-th contract is the front month."""
    day = (pl.col("ts_event").dt.epoch("ns") - _BASE_NS) // _DAY
    lo = 0 if k == 0 else _EXPIRY_DAYS[k - 1]
    return (day >= lo) & (day < _EXPIRY_DAYS[k])


def test_calendar_roll_needs_a_shared_close() -> None:
    _, frames = _market()
    front, front_spans = _continuous(frames, 0)
    second, second_spans = _continuous(frames, 1)
    with pytest.raises(ValueError, match="share no close"):
        futures.calendar_roll(  # the second month is missing while M0 is the front
            "ES.ROLL", front, second.filter(~_expiry_period(1)), front_spans, second_spans, "GLBX"
        )


def test_calendar_roll_starts_at_the_first_contract_it_can_roll_into() -> None:
    _, frames = _market()
    front, front_spans = _continuous(frames, 0)
    second, second_spans = _continuous(frames, 1)
    # The data begins before the second month has bars: H0 can't be rolled out of.
    _, schedule = futures.calendar_roll(
        "ES.ROLL", front, second.filter(~_expiry_period(0)), front_spans, second_spans, "GLBX"
    )
    assert schedule.instrument_ids[0] == f"ES{_CODES[1]}_{_BASE.year}.GLBX"
    assert len(schedule.instrument_ids) == len(front_spans) - 1


def test_calendar_roll_refuses_a_series_left_with_few_contracts() -> None:
    _, frames = _market()
    front, front_spans = _continuous(frames, 0)
    second, second_spans = _continuous(frames, 1)
    sparse = second.filter(
        ~pl.any_horizontal([_expiry_period(k) for k in range(len(front_spans) - 1)])
    )
    with pytest.raises(ValueError, match="too sparse"):
        futures.calendar_roll("ES.ROLL", front, sparse, front_spans, second_spans, "GLBX")


def test_load_rolled_symbol_from_catalog(tmp_path: Path) -> None:
    contracts, frames = _market(carry=0.001)
    catalog = cat.open_catalog(tmp_path)
    for rank in (0, 1):
        sym = f"ES.c.{rank}"
        series, spans = _continuous(frames, rank)
        cat.write_bars(
            catalog,
            cat.frame_to_bars(
                series,
                bar_type=cat.bar_type(sym, "GLBX", "ohlcv-1d"),
                price_precision=2,
                interval_ns=_DAY,
            ),
        )
        cat.write_symbology(sym, "GLBX", spans, tmp_path)
    cat.write_instruments(catalog, [cat.continuous_instrument(contracts[0], "ES.c.0")])

    data = load_market_data("ES.ROLL", catalog_path=tmp_path)
    schedule = data.schedules["ES.ROLL"]
    adjusted = data.frames["ES.ROLL"]
    # Weekend bars are dropped (one bar per trading day for every root).
    assert set(adjusted["ts_event"].dt.weekday().to_list()) <= {1, 2, 3, 4, 5}
    assert schedule.instrument_ids[0] == "ESH0_2020.GLBX"
    held = data.instruments["ESM0_2020.GLBX"]
    assert float(held.multiplier) == _MULT
    run = run_backtest(_Fixed(1.0), data, initial_capital=1e8, leverage=100.0)
    assert run.rejections == ()
    assert [r[0] for r in run.rolls] == [r[0] for r in schedule.rolls]


def test_rolled_symbol_can_name_its_venue() -> None:
    assert futures.rolled_root("BRN.IFEU.ROLL") == "BRN"
    assert futures.rolled_venue("BRN.IFEU.ROLL") == "IFEU"
    assert futures.rolled_root("ES.ROLL") == "ES"
    assert futures.rolled_venue("ES.ROLL") is None


def test_to_usd_converts_each_days_settlement_at_that_days_rate() -> None:
    days = [_BASE + dt.timedelta(days=d) for d in range(3)]
    local = pl.DataFrame(
        {
            "ts_event": days,
            "open": [99.0, 104.0, 108.0],
            "high": [101.0, 111.0, 110.0],
            "low": [98.0, 103.0, 104.0],
            "close": [100.0, 110.0, 105.0],
        }
    )
    fx = pl.DataFrame({"ts_event": days[:2], "close": [1.0, 2.0]})  # day 2 reuses day 1's
    usd = futures.to_usd(local, fx)
    # 100 * 1, then + (110 - 100) * 2, then + (105 - 110) * 2: no move on the notional.
    assert usd["close"].to_list() == [100.0, 120.0, 110.0]
    assert usd["open"].to_list() == [99.0, 100.0 + 4 * 2, 120.0 - 2 * 2]
    assert usd["high"].to_list()[1:] == [122.0, 120.0]
    assert usd["low"].to_list()[1:] == [106.0, 108.0]
    with pytest.raises(ValueError, match="no FX rate"):
        futures.to_usd(local, fx.tail(1))


def test_drop_scale_glitches_removes_one_day_decimal_errors_only() -> None:
    days = [_BASE + dt.timedelta(days=d) for d in range(7)]
    closes = [100.0, 101.0, 10.1, 102.0, -2.0, 40.0, 41.0]  # a tenth; a negative print
    frame = pl.DataFrame(
        {"ts_event": days, "ts_init": [d + dt.timedelta(days=1) for d in days], "close": closes}
    )
    kept = futures.drop_scale_glitches(frame)["close"].to_list()
    # 10.1 reverts next day: an error. -2.0 is negative and 40 doesn't revert: real moves.
    assert kept == [100.0, 101.0, 102.0, -2.0, 40.0, 41.0]


def test_load_rolled_symbol_from_another_venue_in_usd(tmp_path: Path) -> None:
    """A GBP contract on IFLL is listed on the USD account's venue, converted."""
    contracts, frames = _market(carry=0.001)
    catalog = cat.open_catalog(tmp_path)
    rate = 1.25
    for rank in (0, 1):
        series, spans = _continuous(frames, rank)
        sym = f"Z.c.{rank}"
        cat.write_bars(
            catalog,
            cat.frame_to_bars(
                series.with_columns(pl.lit(sym).alias("symbol")),
                bar_type=cat.bar_type(sym, "IFLL", "ohlcv-1d"),
                price_precision=2,
                interval_ns=_DAY,
            ),
        )
        ice_spans = [(d0, d1, raw.replace("ES", "Z   FM") + "!") for d0, d1, raw in spans]
        cat.write_symbology(sym, "IFLL", ice_spans, tmp_path)
        if rank == 0:  # GBP/USD at a constant rate, from the CME future
            cat.write_bars(
                catalog,
                cat.frame_to_bars(
                    series.with_columns(
                        pl.lit(rate).alias(c) for c in ("open", "high", "low", "close")
                    ),
                    bar_type=cat.bar_type("6B.c.0", "GLBX", "ohlcv-1d"),
                    price_precision=4,
                    interval_ns=_DAY,
                ),
            )
    spec = cat.continuous_instrument(contracts[0], "Z.c.0").to_dict()
    spec.update(id="Z.c.0.IFLL", currency="GBP", exchange="IFLL")
    cat.write_instruments(catalog, [FuturesContract.from_dict(spec)])

    local = load_market_data("Z.IFLL.ROLL", catalog_path=tmp_path)
    data = load_market_data("Z.IFLL.ROLL", venue="GLBX", catalog_path=tmp_path)
    first = data.schedules["Z.IFLL.ROLL"].instrument_ids[0]
    assert first == "IFLL_Z_FMH0_2020.GLBX"
    held = data.instruments[first]
    assert str(held.quote_currency) == "USD"
    assert str(local.instruments["Z   FMH0!_2020.IFLL"].quote_currency) == "GBP"
    # At a constant rate the USD price is the GBP price times the rate.
    np.testing.assert_allclose(
        data.frames["Z.IFLL.ROLL"]["close"].to_numpy(),
        rate * local.frames["Z.IFLL.ROLL"]["close"].to_numpy(),
        rtol=1e-6,
    )
    run = run_backtest(_Fixed(1.0), data, initial_capital=1e8, leverage=100.0)
    assert run.rejections == ()
    assert len(run.rolls) == len(data.schedules["Z.IFLL.ROLL"].rolls)


def test_missing_mapping_says_how_to_get_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="algo data sync"):
        cat.read_symbology("ES.c.0", "GLBX", tmp_path)


def test_report_loads_rolled_symbols_and_benchmark(tmp_path: Path) -> None:
    from snippy_scales.reporting import ReportConfig  # noqa: PLC0415
    from snippy_scales.reporting import pipeline as rp  # noqa: PLC0415

    contracts, frames = _market(carry=0.001)
    catalog = cat.open_catalog(tmp_path)
    for rank in (0, 1):
        series, spans = _continuous(frames, rank)
        sym = f"ES.c.{rank}"
        bars = cat.frame_to_bars(
            series,
            bar_type=cat.bar_type(sym, "GLBX", "ohlcv-1d"),
            price_precision=2,
            interval_ns=_DAY,
        )
        cat.write_bars(catalog, bars)
        cat.write_symbology(sym, "GLBX", spans, tmp_path)
    cat.write_instruments(catalog, [cat.continuous_instrument(contracts[0], "ES.c.0")])
    cfg = ReportConfig.model_validate(
        {
            "name": "unit",
            "title": "t",
            "strategy": "momentum-ts",
            "data": {"symbols": ["ES.ROLL", "NQ.ROLL"], "venue": "GLBX", "benchmark": "ES.ROLL"},
        }
    )
    data = rp.load_universe(cfg, tmp_path)
    assert list(data.frames) == ["ES.ROLL"]  # NQ has no continuous series
    assert rp.missing_symbols(cfg, data) == ["NQ.ROLL"]
    bench = rp.benchmark_bars(cfg, tmp_path)
    assert bench is not None
    assert bench["close"].to_list() == data.frames["ES.ROLL"]["close"].to_list()
    # A rolled symbol that names its venue is available whatever the account venue.
    other = cfg.model_copy(
        update={"data": cfg.data.model_copy(update={"symbols": ["ES.GLBX.ROLL"], "venue": "SIM"})}
    )
    moved = rp.load_universe(other, tmp_path)
    assert moved.schedules["ES.GLBX.ROLL"].instrument_ids[0] == "GLBX_ESH0_2020.SIM"


def test_calendar_roll_skips_closes_where_either_contract_did_not_trade() -> None:
    _, frames = _market()
    front, front_spans = _continuous(frames, 0)
    second, second_spans = _continuous(frames, 1)
    quiet = _close_ns(_EXPIRY_DAYS[0] - 5)  # the default roll close for H0 -> M0
    second = second.with_columns(
        volume=pl.when(pl.col("ts_init").dt.epoch("ns") == quiet)
        .then(0.0)
        .otherwise(pl.col("volume"))
    )
    _, schedule = futures.calendar_roll(
        "ES.ROLL", front, second, front_spans, second_spans, "GLBX", days_before=5
    )
    assert schedule.rolls[0][0] == _close_ns(_EXPIRY_DAYS[0] - 6)


# ── Roll yield (carry) ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("near", "far", "months"),
    [
        ("ESZ6", "ESH7", 3),
        ("CLZ9", "CLF0", 1),  # one-digit years wrap at the decade
        ("BRN FMG0019!", "BRN FMH0019!", 1),
        ("Z   FMZ0026!", "Z   FMH0027!", 3),
        ("FGBL SI 20260908 PS", "FGBL SI 20261208 PS", 3),
        ("ESH7", "ESZ6", None),  # a wrapped year digit: the far contract delivers first
        ("ESZ6", "ESZ6", None),
        ("ESZ6", "BRN FMG0019!", None),
        ("AAPL", "ESZ6", None),
    ],
)
def test_months_between(near: str, far: str, months: int | None) -> None:
    assert futures.months_between(near, far) == months


def test_roll_yield_is_the_annualised_front_to_second_ratio() -> None:
    _, frames = _market(carry=0.001)
    front, front_spans = _continuous(frames, 0)
    second, second_spans = _continuous(frames, 1)
    carry = futures.roll_yield(front, second, front_spans, second_spans)

    assert carry.columns == ["ts_init", "carry"]
    assert carry["ts_init"].is_sorted()
    joined = (
        carry.join(front.select("ts_init", "close"), on="ts_init")
        .join(second.select("ts_init", pl.col("close").alias("far")), on="ts_init")
        .with_columns(expected=(pl.col("close") / pl.col("far") - 1.0) * 12.0 / 3.0)
    )
    assert joined.height == carry.height > 0
    np.testing.assert_allclose(joined["carry"].to_numpy(), joined["expected"].to_numpy())


def test_roll_yield_skips_closes_where_a_contract_did_not_trade() -> None:
    _, frames = _market(carry=0.001)
    front, front_spans = _continuous(frames, 0)
    second, second_spans = _continuous(frames, 1)
    stale = second["ts_init"][5]
    second = second.with_columns(
        volume=pl.when(pl.col("ts_init") == stale).then(0.0).otherwise(pl.col("volume"))
    )
    carry = futures.roll_yield(front, second, front_spans, second_spans)
    assert stale not in set(carry["ts_init"].to_list())


def test_rolled_symbol_frame_has_carry(tmp_path: Path) -> None:
    contracts, frames = _market(carry=0.001)
    catalog = cat.open_catalog(tmp_path)
    for rank in (0, 1):
        sym = f"ES.c.{rank}"
        series, spans = _continuous(frames, rank)
        cat.write_bars(
            catalog,
            cat.frame_to_bars(
                series,
                bar_type=cat.bar_type(sym, "GLBX", "ohlcv-1d"),
                price_precision=2,
                interval_ns=_DAY,
            ),
        )
        cat.write_symbology(sym, "GLBX", spans, tmp_path)
    cat.write_instruments(catalog, [cat.continuous_instrument(contracts[0], "ES.c.0")])

    frame = load_market_data("ES.ROLL", catalog_path=tmp_path).frames["ES.ROLL"]
    assert "carry" in frame.columns
    assert frame["carry"].drop_nulls().len() > frame.height // 2


def test_volume_rolled_symbols_name_the_volume_series() -> None:
    assert futures.is_rolled("SB.IFUS.VROLL") and futures.is_rolled("ES.ROLL")
    assert not futures.is_rolled("ES.c.0")
    assert futures.rolled_rule("SB.IFUS.VROLL") == "v" and futures.rolled_rule("ES.ROLL") == "c"
    assert futures.rolled_root("SB.IFUS.VROLL") == "SB"
    assert futures.rolled_venue("SB.IFUS.VROLL") == "IFUS"
    assert futures.rolled_venue("SB.VROLL") is None


def test_repair_prices_fixes_zero_lows_and_drops_zero_closes() -> None:
    frame = pl.DataFrame(
        {
            "open": [10.0, 0.0, 10.0],
            "high": [11.0, 9.0, 11.0],
            "low": [0.0, 0.0, 9.0],
            "close": [10.5, 9.5, 0.0],
            "volume": [1.0, 1.0, 1.0],
        }
    )
    out = futures.repair_prices(frame)
    assert out.height == 2  # the zero close is dropped
    assert out["low"].to_list() == [10.0, 9.5]  # the lower of open and close
    assert out["open"].to_list() == [10.0, 9.5]  # a zero open becomes the close
    assert out["high"].to_list() == [11.0, 9.5]


def test_forward_only_confirms_switches_and_never_goes_back() -> None:
    h, k, n = "SB  FMH0025!_2024", "SB  FMK0025!_2024", "SB  FMN0025!_2025"
    #        a far contract for one day, a flip back, then a confirmed switch
    seen = [h, n, h, k, h, k, k, k, h, k]
    assert futures._forward_only(seen, confirm=3) == [h, h, h, h, h, h, h, k, k, k]  # noqa: SLF001
    assert futures._forward_only([], confirm=3) == []  # noqa: SLF001
    # One-digit CME years are completed from the listing year (Z9 then F0: a decade on).
    assert futures._delivery("CLF0_2019") > futures._delivery("CLZ9_2019")  # noqa: SLF001


def test_second_nearby_return_follows_one_contract_across_an_expiry() -> None:
    _, frames = _market(carry=0.001)  # a steep curve: contracts 90 days apart differ by ~9%
    front, front_spans = _continuous(frames, 0)
    second, second_spans = _continuous(frames, 1)
    out = futures.second_nearby_return(front, second, front_spans, second_spans)

    assert out.columns == ["ts_init", "second_return"] and out["ts_init"].is_sorted()
    assert out.height > second.height * 0.9
    # The series itself jumps when it switches contract; the held contract does not.
    jump = second.sort("ts_init")["close"].pct_change().abs().max()
    assert jump is not None and jump > 0.07  # noqa: PLR2004
    assert out["second_return"].abs().max() < 0.06  # noqa: PLR2004
    # On an ordinary day it is the second series' own return.
    plain = second.sort("ts_init").select("ts_init", pl.col("close").pct_change().alias("r"))
    both = out.join(plain, on="ts_init").filter(
        (pl.col("r") - pl.col("second_return")).abs() < 1e-12
    )
    assert both.height > out.height * 0.9
