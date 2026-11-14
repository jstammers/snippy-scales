"""Datasets for the spike. Real data is read-only (``SNIPPY_DATA_ROOT``)."""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

from common import future
from nautilus_trader.model import Bar, Quantity

from snippy_scales.data import catalog as cat
from snippy_scales.data.futures import RollRule, RollSchedule
from snippy_scales.execution import MarketData, load_market_data

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "tests"))

#: Point value per 1.0 of the quoted price (grains are quoted in cents).
MULTIPLIERS = {
    "6A": 100_000, "6B": 62_500, "6E": 125_000, "6J": 12_500_000, "CL": 1_000,
    "ES": 50, "GC": 100, "NG": 10_000, "NQ": 20, "RTY": 50, "SI": 5_000, "YM": 5,
    "ZB": 1_000, "ZC": 50, "ZF": 1_000, "ZN": 1_000, "ZS": 50, "ZT": 2_000, "ZW": 50,
}  # fmt: skip

EQUITIES = ["AAPL", "AMZN", "AMD", "ADBE", "AMGN", "ABT", "AEP", "AIG"]


def aapl() -> MarketData:
    return load_market_data("AAPL")


def equities() -> MarketData:
    return load_market_data(EQUITIES)


def glbx_futures() -> MarketData:
    """The 19 GLBX ``*.c.0`` daily series with stand-in contracts (real multipliers)."""
    catalog = cat.open_catalog()
    syms = [s for s, v in cat.list_bar_series("ohlcv-1d") if v == "GLBX"]
    frames, bars, instruments, bar_types, schedules = {}, {}, {}, {}, {}
    for s in syms:
        bt = cat.bar_type(s, "GLBX", "ohlcv-1d")
        prec = cat.stored_precision(catalog, bt)
        pp = prec[0] if prec else 2
        inst = future(s, "GLBX", MULTIPLIERS[s.split(".")[0]], pp, f"{Decimal(1).scaleb(-pp):f}")
        iid = str(inst.id)
        frames[s] = cat.load_bars(s, "ohlcv-1d")
        bars[iid] = sorted(catalog.query(cat.NautilusDataType.Bar, [bt]), key=lambda b: b.ts_init)
        instruments[iid] = inst
        bar_types[iid] = bt
        schedules[s] = RollSchedule.single(s, iid)
    return MarketData(frames, bars, instruments, bar_types, schedules)


def ample_volume(data: MarketData, volume: int = 10**9) -> MarketData:
    """*data* with every bar's volume replaced by *volume*.

    Nautilus treats a bar's volume as the liquidity at its O/H/L/C, so an order
    bigger than a thin real bar walks to worse prices. The vectorised engines
    fill everything at the close; this isolates engine accuracy from that.
    """
    bars = {}
    for iid, series in data.bars.items():
        prec = data.instruments[iid].size_precision
        q = Quantity(volume, prec)
        bars[iid] = [
            Bar(b.bar_type, b.open, b.high, b.low, b.close, q, b.ts_event, b.ts_init)
            for b in series
        ]
    return MarketData(data.frames, bars, data.instruments, data.bar_types, data.schedules)


def rolled_es() -> MarketData:
    """The synthetic ES outright market of ``tests/test_futures_rolls.py``, volume roll."""
    from test_futures_rolls import _market  # noqa: PLC0415

    contracts, frames = _market()
    return MarketData.from_outrights("ES.FUT", frames, contracts, rule=RollRule("volume"))
