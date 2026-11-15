"""Independent replication of Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum*.

Re-implements the paper's methodology from scratch (pandas/numpy only, no
``snippy_scales`` strategy or engine code), so its results can be compared with
``TimeSeriesMomentum`` run through ``execution``:

* daily excess returns per futures root, from the held contract: the front month,
  rolled 5 trading days before Databento's calendar ``.c.0`` switches contract
  (contracts recovered from ``.c.0``/``.c.1`` bars and their stored mapping by
  ``snippy_scales.data.futures.calendar_roll``; data access only). Returns are in
  each contract's own currency, as in the paper;
* ex-ante volatility: EWMA of squared demeaned daily returns, centre of mass
  60 days, annualised with 261 (paper eq. 1);
* TSMOM(k, h): sign of the past k-month return, position ``40% / sigma`` set at
  month end, held h months, overlapping portfolios averaged (Jegadeesh-Titman),
  equal weight across the instruments available that month (paper eq. 5);
* the pooled panel regression of Fig. 1 Panel B with month-clustered t-stats.

Usage::

    SNIPPY_DATA_ROOT=/path/to/data uv run python scripts/replicate_tsmom.py [--out DIR]

Read-only on ``data/``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

DATA = Path(os.environ.get("SNIPPY_DATA_ROOT", "data"))

#: (venue, root) -> asset class: the 34 CME roots of the tsmom-futures report plus
#: the ICE roots whose front month rolls cleanly (data from 2019). Excluded: sparse
#: front months (CME TN 6M PL PA; ICE T R SO3 and all IFUS) and short-rate futures
#: (GE SR3 I), which the paper did not use.
ASSET_CLASS = {
    **{("GLBX", r): "equity" for r in ("ES", "NQ", "YM", "RTY", "NKD", "EMD")},
    **{("GLBX", r): "bond" for r in ("ZT", "ZF", "ZN", "ZB", "UB")},
    **{("GLBX", r): "fx" for r in ("6A", "6B", "6C", "6E", "6J", "6S", "6N")},
    **{
        ("GLBX", r): "commodity"
        for r in ("CL", "HO", "RB", "NG", "GC", "SI", "HG")
        + ("ZC", "ZS", "ZW", "KE", "ZL", "ZM", "LE", "HE", "GF")
    },
    **{("IFEU", r): "commodity" for r in ("BRN", "G", "C", "RC", "W")},
    ("IFLL", "Z"): "equity",
}
ROLL_DAYS_BEFORE = 5

TARGET_VOL = 0.40
COM = 60  # EWMA centre of mass, days
ANNUAL_DAYS = 261


# ── data ───────────────────────────────────────────────────────────────────────


def rolled_returns(root: str, venue: str) -> tuple[pd.Series, int]:
    """Daily returns of the held contract and the number of rolls."""
    from snippy_scales.data import catalog as cat  # noqa: PLC0415 — data access only
    from snippy_scales.data import futures  # noqa: PLC0415

    def weekdays(sym: str) -> pl.DataFrame:
        f = cat.load_bars(sym, "ohlcv-1d", venue=venue)
        return f.filter(pl.col("ts_event").dt.weekday() <= 5)  # noqa: PLR2004

    frames, schedule = futures.calendar_roll(
        f"{root}.ROLL",
        weekdays(f"{root}.c.0"),
        weekdays(f"{root}.c.1"),
        cat.read_symbology(f"{root}.c.0", venue),
        cat.read_symbology(f"{root}.c.1", venue),
        venue,
        days_before=ROLL_DAYS_BEFORE,
    )
    adj = futures.back_adjust(schedule, frames)
    s = adj.select(pl.col("ts_event").dt.date().alias("d"), "close").to_pandas().set_index("d")
    s.index = pd.DatetimeIndex(s.index)
    return s["close"].pct_change(), len(schedule.rolls)


def futures_returns(venues: tuple[str, ...]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Daily returns (date x root) and roll counts for the roots on *venues*."""
    cols, rolls = {}, {}
    for venue, root in ASSET_CLASS:
        if venue in venues:
            cols[root], rolls[root] = rolled_returns(root, venue)
    return pd.DataFrame(cols).sort_index(), rolls


def stock_returns(start: str = "2015-01-01") -> pd.DataFrame:
    """Daily returns of every S&P 500 ever-member in the catalog (split/div adjusted)."""
    from snippy_scales.data.catalog import load_bars  # noqa: PLC0415 — data access only

    names = [
        s.strip()
        for s in (DATA / "universe" / "sp500_ever_members.txt").read_text().splitlines()
        if s.strip() and not s.startswith("#")
    ]
    cols = {}
    for name in names:
        try:
            b = load_bars(name, "ohlcv-1d")
        except Exception:  # noqa: BLE001 — symbol missing from the catalog
            continue
        if b.height < 300:
            continue
        b = b.sort("ts_init").filter(pl.col("volume") > 0)
        s = (
            b.select(pl.col("ts_event").dt.date().alias("date"), "close")
            .to_pandas()
            .set_index("date")["close"]
        )
        s.index = pd.DatetimeIndex(s.index)
        s = s[~s.index.duplicated()]
        cols[name] = s.pct_change()
    r = pd.DataFrame(cols).sort_index()
    return r[r.index >= start]


def spy_returns() -> pd.Series:
    from snippy_scales.data.catalog import load_bars  # noqa: PLC0415

    b = load_bars("SPY", "ohlcv-1d").sort("ts_init")
    s = b.select(pl.col("ts_event").dt.date().alias("d"), "close").to_pandas().set_index("d")
    s.index = pd.DatetimeIndex(s.index)
    return s["close"].pct_change()


# ── paper methodology ─────────────────────────────────────────────────────────


def ex_ante_vol(daily: pd.DataFrame) -> pd.DataFrame:
    """Paper eq. 1: annualised EWMA vol, centre of mass 60 days (NaN returns skipped)."""
    var = daily.ewm(com=COM, min_periods=COM, ignore_na=True).var(bias=True)
    return (ANNUAL_DAYS * var).pow(0.5)


def monthly(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly compounded returns and month-end ex-ante vol (NaN where no data)."""
    month = pd.DatetimeIndex(daily.index).to_period("M")
    traded = daily.notna().groupby(month).sum() > 5
    r_m = (1 + daily.fillna(0)).groupby(month).prod() - 1
    r_m = r_m.where(traded)
    vol_m = ex_ante_vol(daily).groupby(month).last().where(traded)
    return r_m, vol_m


def past_return(r_m: pd.DataFrame, k: int) -> pd.DataFrame:
    """Return over months t-k+1..t (known at the end of month t)."""
    log_r = pd.DataFrame(np.log1p(r_m.to_numpy()), index=r_m.index, columns=r_m.columns)
    return log_r.rolling(k, min_periods=k).sum().transform("exp") - 1


def tsmom(
    r_m: pd.DataFrame, vol_m: pd.DataFrame, k: int = 12, h: int = 1, *, scale: bool = True
) -> tuple[pd.Series, pd.DataFrame]:
    """Diversified TSMOM(k, h) monthly returns and the per-instrument (h=1) returns."""
    sig = _sign(past_return(r_m, k))
    lev = (
        TARGET_VOL / vol_m if scale else pd.DataFrame(1.0, index=vol_m.index, columns=vol_m.columns)
    )
    pos = (sig * lev).where(sig.notna() & vol_m.notna())
    legs = [pos.shift(j) * r_m for j in range(1, h + 1)]  # portfolio formed j months ago
    per_inst = sum(legs) / h if h > 1 else legs[0]
    per_inst = per_inst.where(legs[0].notna())
    return per_inst.mean(axis=1).dropna(), per_inst


def _sign(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.sign(df.to_numpy()), index=df.index, columns=df.columns)


def sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(12)) if len(x) > 2 else float("nan")


def stats(x: pd.Series) -> dict[str, float | str]:
    x = x.dropna()
    eq = (1 + x).cumprod()
    return {
        "start": str(x.index[0]),
        "end": str(x.index[-1]),
        "months": len(x),
        "ann_return": float(x.mean() * 12),
        "ann_vol": float(x.std() * np.sqrt(12)),
        "sharpe": sharpe(x),
        "t_stat": float(x.mean() / x.std() * np.sqrt(len(x))),
        "max_dd": float((eq / eq.cummax() - 1).min()),
    }


def pooled_sign_regression(r_m: pd.DataFrame, vol_m: pd.DataFrame, lags: int = 24) -> list[float]:
    """Fig. 1 Panel B: r_t/sigma_{t-1} = a + b_h sign(r_{t-h}); t(b_h) clustered by month."""
    y_all = r_m / vol_m.shift(1)
    out = []
    for h in range(1, lags + 1):
        x_all = _sign(r_m.shift(h))
        df = pd.DataFrame({"y": y_all.stack(), "x": x_all.stack()}).dropna()
        X = np.column_stack([np.ones(len(df)), df["x"].to_numpy()])
        y = df["y"].to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        u = y - X @ beta
        bread = np.linalg.inv(X.T @ X)
        groups = df.index.get_level_values(0)
        meat = np.zeros((2, 2))
        for g in pd.unique(groups):
            m = groups == g
            s = X[m].T @ u[m]
            meat += np.outer(s, s)
        n_g = len(pd.unique(groups))
        cov = bread @ meat @ bread * n_g / (n_g - 1)
        out.append(float(beta[1] / np.sqrt(cov[1, 1])))
    return out


def avg_pairwise_corr(per_inst: pd.DataFrame, min_obs: int = 36) -> float:
    c = per_inst.corr(min_periods=min_obs).to_numpy()
    iu = np.triu_indices_from(c, 1)
    return float(np.nanmean(c[iu]))


# ── main ──────────────────────────────────────────────────────────────────────


def run_universe(name: str, daily: pd.DataFrame, classes: dict[str, str] | None) -> dict:
    r_m, vol_m = monthly(daily)
    port, per_inst = tsmom(r_m, vol_m, 12, 1)
    res: dict = {"universe": name, "n_instruments": int(daily.shape[1])}
    res["tsmom_12_1"] = stats(port)
    res["per_instrument_sharpe"] = {c: sharpe(per_inst[c]) for c in per_inst.columns}
    pis = pd.Series(res["per_instrument_sharpe"]).dropna()
    res["per_instrument_sharpe_summary"] = {
        "median": float(pis.median()),
        "mean": float(pis.mean()),
        "share_positive": float((pis > 0).mean()),
    }
    res["avg_pairwise_corr_tsmom_legs"] = avg_pairwise_corr(per_inst)
    res["avg_pairwise_corr_raw_monthly"] = avg_pairwise_corr(r_m)
    if classes:
        res["by_class"] = {}
        for cls in sorted(set(classes.values())):
            members = [c for c in per_inst.columns if classes.get(c) == cls]
            res["by_class"][cls] = stats(per_inst[members].mean(axis=1).dropna())
    # no vol scaling (Kim, Tse & Wald 2016): sign x 1/N
    res["tsmom_12_1_unscaled_sharpe"] = sharpe(tsmom(r_m, vol_m, 12, 1, scale=False)[0])
    # passive long at the same ex-ante risk (paper Fig. 3 comparison)
    passive = ((TARGET_VOL / vol_m).shift(1) * r_m).mean(axis=1).dropna()
    res["passive_long_vol_scaled"] = stats(passive)
    res["kh_grid_sharpe"] = {
        f"k{k}_h{h}": sharpe(tsmom(r_m, vol_m, k, h)[0])
        for k in (1, 3, 6, 9, 12)
        for h in (1, 3, 6, 12)
    }
    res["pooled_sign_tstats_lag1_24"] = pooled_sign_regression(r_m, vol_m)
    res["_series"] = {"tsmom": port, "passive": passive}
    return res


def main() -> None:
    p = argparse.ArgumentParser(description="Replicate Moskowitz, Ooi & Pedersen (2012).")
    p.add_argument("--out", type=Path, default=Path("tsmom_replication"))
    p.add_argument("--skip-stocks", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    classes = {root: cls for (_, root), cls in ASSET_CLASS.items()}
    cme, rolls = futures_returns(("GLBX",))
    results: dict[str, Any] = {"rolls_per_root": rolls}
    results["futures"] = run_universe("futures (CME)", cme, classes)
    s = results["futures"]["_series"]["tsmom"]
    results["futures_subperiods"] = {
        "2011-2015": stats(s[str(s.index[0]) : "2015-12"]),
        "2016-2020": stats(s["2016-01":"2020-12"]),
        "2021-2026": stats(s["2021-01":]),
        "2016-on (stock window)": stats(s["2016-01":]),
        "2020-on (ICE window)": stats(s["2020-01":]),
    }
    both, ice_rolls = futures_returns(("GLBX", "IFEU", "IFLL"))
    results["rolls_per_root"] |= ice_rolls
    results["futures_cme_ice"] = run_universe("futures (CME + ICE)", both, classes)
    t = results["futures_cme_ice"]["_series"]["tsmom"]
    results["futures_cme_ice"]["tsmom_12_1_2020_on"] = stats(t["2020-01":])
    if not args.skip_stocks:
        stk = stock_returns()
        results["stocks"] = run_universe("stocks", stk, None)
        spy = spy_returns().to_frame("SPY")
        results["spy"] = run_universe("spy", spy[spy.index >= "2015-01-01"], None)

    series = {
        f"{u}_{k}": v
        for u in ("futures", "futures_cme_ice", "stocks", "spy")
        if u in results
        for k, v in results[u].pop("_series").items()
    }
    pd.DataFrame(series).to_csv(args.out / "monthly_returns.csv")
    (args.out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
