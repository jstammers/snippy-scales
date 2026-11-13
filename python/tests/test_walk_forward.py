"""Tests for evaluation/walk_forward.py: folds, search space, studies, reports."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna
import polars as pl
import pytest

from snippy_scales.evaluation import (
    Fold,
    WalkForward,
    load_strategy_class,
    make_folds,
    returns_statistics,
    run_study,
    stitch_returns,
    suggest,
    summarize_study,
    write_tearsheet,
)
from snippy_scales.execution import Costs
from snippy_scales.strategies import CrossSectionalMomentum, TrendFollowing
from tests.conftest import synthetic_data


def _closes(n: int, seed: int, drift: float = 0.0004) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))


@pytest.fixture
def wf() -> WalkForward:
    data = synthetic_data({"X": _closes(600, seed=1)})
    return WalkForward(
        data,
        make_folds(data.closes(), n_splits=4, min_train=100),
        costs=Costs(commission_rate=1e-4),
    )


# ── Folds ─────────────────────────────────────────────────────────────────────


def test_folds_are_contiguous_and_end_at_last_bar() -> None:
    closes = synthetic_data({"X": _closes(500, seed=0)}).closes()
    folds = make_folds(closes, n_splits=5, test_size=0.1, min_train=50)
    assert len(folds) == 5
    assert folds[-1].test_end == closes[-1]
    for a, b in zip(folds, folds[1:], strict=False):
        assert b.test_start > a.test_end
        assert (b.test_start - a.test_end) == dt.timedelta(days=1)
    assert all(f.train_end < f.test_start for f in folds)
    assert folds[0].train_start == closes[0]


def test_test_size_capped_by_min_train() -> None:
    closes = synthetic_data({"X": _closes(300, seed=0)}).closes()
    folds = make_folds(closes, n_splits=5, test_size=0.2, min_train=200)
    test_bars = (folds[0].test_end - folds[0].test_start).days + 1
    assert test_bars == 20  # (300 - 200) // 5, not 0.2 * 300
    with pytest.raises(ValueError, match="too few"):
        make_folds(closes, n_splits=5, min_train=299)


def test_fold_json_round_trip() -> None:
    fold = make_folds(synthetic_data({"X": _closes(400, seed=0)}).closes(), min_train=50)[2]
    assert Fold.from_json(fold.to_json()) == fold


# ── Search space ──────────────────────────────────────────────────────────────


def test_suggest_handles_each_spec_kind() -> None:
    trial = optuna.trial.FixedTrial({"a": 20, "b": 7, "c": 0.25})
    params = suggest(
        trial, {"a": [10, 20], "b": {"low": 5, "high": 9}, "c": {"low": 0.1, "high": 0.5}, "d": 3}
    )
    assert params == {"a": 20, "b": 7, "c": 0.25, "d": 3}


def test_stitch_compounds_shared_days() -> None:
    ts = [dt.datetime(2020, 1, d, tzinfo=dt.UTC) for d in (1, 2, 2, 3)]
    frame = pl.DataFrame({"ts": ts, "return": [0.01, 0.1, 0.1, -0.02]}).with_columns(
        pl.col("ts").cast(pl.Datetime("ns", "UTC"))
    )

    runs = [SimpleNamespace(returns=frame.head(2)), SimpleNamespace(returns=frame.tail(2))]
    out = stitch_returns(runs)  # ty: ignore[invalid-argument-type]
    assert out["return"].to_list() == pytest.approx([0.01, 1.1 * 1.1 - 1, -0.02])


# ── Studies ───────────────────────────────────────────────────────────────────


def test_grid_study_runs_every_combination(wf: WalkForward, tmp_path: Path) -> None:
    space = {"fast_period": [5, 10], "slow_period": [30, 60]}
    study = run_study(
        TrendFollowing, wf, space=space, sampler="grid", storage=None, returns_dir=tmp_path
    )
    trials = study.trials
    assert len(trials) == 4
    assert {(t.params["fast_period"], t.params["slow_period"]) for t in trials} == {
        (5, 30),
        (5, 60),
        (10, 30),
        (10, 60),
    }
    for t in trials:
        folds = t.user_attrs["folds"]
        assert len(folds) == 4
        assert t.value == pytest.approx(np.mean([f["sharpe"] for f in folds]))
        assert Path(t.user_attrs["returns_path"]).is_file()
        assert all(f["rejections"] == 0 for f in folds)
    assert study.user_attrs["symbols"] == ["X"]
    assert load_strategy_class(study.user_attrs["strategy"]) is TrendFollowing


def test_report_deflates_for_the_search(wf: WalkForward, tmp_path: Path) -> None:
    space = {"fast_period": [5, 10, 20], "slow_period": [30, 60]}
    study = run_study(
        TrendFollowing, wf, space=space, sampler="grid", storage=None, returns_dir=tmp_path
    )
    report = summarize_study(study)
    assert report.n_trials == 6
    assert report.mean_fold_sharpe == pytest.approx(study.best_value)
    assert report.best_params == study.best_params
    assert report.pbo is not None and 0.0 <= report.pbo <= 1.0
    assert 0.0 <= report.dsr <= report.psr <= 1.0
    assert report.max_drawdown <= 0.0
    # Stitched OOS covers the four test windows: one daily return per bar after
    # each fold's first (on which the position is only opened).
    assert report.returns.height == sum((f.test_end - f.test_start).days for f in wf.folds)
    assert report.stats_returns["Sharpe Ratio (252 days)"] == pytest.approx(report.sharpe)


def test_default_params_single_trial(wf: WalkForward, tmp_path: Path) -> None:
    study = run_study(TrendFollowing, wf, storage=None, returns_dir=tmp_path)
    report = summarize_study(study)
    assert report.n_trials == 1
    assert report.best_params == {}
    assert report.pbo is None
    assert report.dsr == pytest.approx(report.psr)


def test_tpe_respects_n_trials(wf: WalkForward, tmp_path: Path) -> None:
    study = run_study(
        TrendFollowing,
        wf,
        space={"fast_period": {"low": 5, "high": 20}, "slow_period": {"low": 30, "high": 80}},
        n_trials=3,
        storage=None,
        returns_dir=tmp_path,
    )
    assert len(study.trials) == 3


def test_sqlite_storage_round_trip(wf: WalkForward, tmp_path: Path) -> None:
    storage = f"sqlite:///{tmp_path / 'db' / 'optuna.db'}"
    run_study(
        TrendFollowing,
        wf,
        space={"fast_period": [5, 10]},
        sampler="grid",
        study_name="s1",
        storage=storage,
        returns_dir=tmp_path,
    )
    loaded = optuna.load_study(study_name="s1", storage=storage)
    report = summarize_study(loaded)
    assert report.study_name == "s1"
    assert report.n_trials == 2
    assert [Fold.from_json(f) for f in loaded.user_attrs["folds"]] == list(wf.folds)


def test_multi_asset_study(tmp_path: Path) -> None:
    data = synthetic_data({f"A{i}": _closes(500, seed=10 + i, drift=0.0003 * i) for i in range(4)})
    wf = WalkForward(data, make_folds(data.closes(), n_splits=3, min_train=150))
    study = run_study(
        CrossSectionalMomentum,
        wf,
        space={"lookback": [60, 120], "skip_recent": 5, "rebal_freq": 10, "vol_lookback": 20},
        sampler="grid",
        storage=None,
        returns_dir=tmp_path,
    )
    report = summarize_study(study)
    assert report.symbols == ["A0", "A1", "A2", "A3"]
    assert report.n_trials == 2
    assert np.isfinite(report.sharpe)


def test_grid_rejects_ranges(wf: WalkForward, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="grid search"):
        run_study(
            TrendFollowing,
            wf,
            space={"fast_period": {"low": 5, "high": 20}},
            sampler="grid",
            storage=None,
            returns_dir=tmp_path,
        )


def test_tearsheet_written(wf: WalkForward, tmp_path: Path) -> None:
    report = summarize_study(run_study(TrendFollowing, wf, storage=None, returns_dir=tmp_path))
    out = write_tearsheet(report, tmp_path / "reports" / "t.html")
    assert out.is_file() and out.stat().st_size > 10_000


def test_returns_statistics_matches_numpy() -> None:
    r = np.random.default_rng(3).normal(0.0005, 0.01, 300)
    ts = [dt.datetime(2020, 1, 1, tzinfo=dt.UTC) + dt.timedelta(days=i) for i in range(300)]
    frame = pl.DataFrame({"ts": ts, "return": r}).with_columns(
        pl.col("ts").cast(pl.Datetime("ns", "UTC"))
    )
    stats = returns_statistics(frame)
    assert stats["Sharpe Ratio (252 days)"] == pytest.approx(
        r.mean() / r.std(ddof=1) * np.sqrt(252), rel=1e-9
    )
