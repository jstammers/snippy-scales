"""Strategy reports: ragged universes, config, scaffolding, and the report pipeline."""

from __future__ import annotations

import datetime as dt
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from snippy_scales.execution import MarketData, compute_targets, run_backtest, vector
from snippy_scales.reporting import (
    REPO_ROOT,
    TEMPLATE,
    ReportConfig,
    ReportPaths,
    load_report_config,
    new_report,
)
from snippy_scales.reporting import config as report_config
from snippy_scales.reporting import pipeline as rp
from snippy_scales.strategies import CrossSectionalMomentum, MultiAssetStrategy, Strategy
from tests.conftest import daily_bars, synthetic_future

EXAMPLES = sorted((REPO_ROOT / "reports" / "templates" / "configs").glob("*.yaml"))


def _closes(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))


def _ragged(n: int = 500) -> MarketData:
    """A lists from the start; B lists 100 bars late; C delists 150 bars early; D, E full."""
    frames = {}
    for i, symbol in enumerate("ABCDE"):
        frame = daily_bars(_closes(n, 40 + i))
        if symbol == "B":
            frame = frame.slice(100)
        elif symbol == "C":
            frame = frame.slice(0, n - 150)
        frames[symbol] = frame
    return MarketData.from_frames(frames, {s: synthetic_future(s) for s in frames})


class _Recorder(MultiAssetStrategy):
    def __init__(self) -> None:
        self.seen: dict[str, pl.DataFrame] = {}

    def generate_signals(self, multi_bars: dict[str, pl.DataFrame]) -> dict[str, pl.Series]:
        self.seen = multi_bars
        return {s: pl.Series(np.ones(f.height)) for s, f in multi_bars.items()}


class _Long(Strategy):
    def generate_signals(self, bars: pl.DataFrame) -> pl.Series:
        return pl.Series("position", np.ones(bars.height))


# ── Ragged universes ──────────────────────────────────────────────────────────


def test_multi_asset_frames_align_on_the_union_of_dates() -> None:
    data = _ragged()
    recorder = _Recorder()
    targets = compute_targets(recorder, data.frames)
    union = data.closes(union=True)
    assert union.len() == 500
    assert data.closes().len() == 500 - 100 - 150  # shared dates only
    for symbol, frame in recorder.seen.items():
        assert frame.height == union.len()
        assert frame.columns == data.frames[symbol].columns
    b = recorder.seen["B"]["close"]
    assert b[:100].null_count() == 100 and b[100:].null_count() == 0
    c = recorder.seen["C"]["close"]
    assert c[-150:].null_count() == 150 and c[:-150].null_count() == 0
    # Targets only on each symbol's real bars.
    for symbol, frame in data.frames.items():
        assert len(targets[symbol]) == frame.height


def test_gaps_inside_a_symbols_life_are_forward_filled() -> None:
    data = _ragged()
    frames = dict(data.frames)
    frames["A"] = frames["A"].filter(pl.int_range(pl.len()) != 250)
    recorder = _Recorder()
    targets = compute_targets(recorder, frames)
    a = recorder.seen["A"]
    assert a["close"].null_count() == 0
    assert a["close"][250] == a["close"][249]
    assert a["volume"][250] == 0.0
    assert len(targets["A"]) == frames["A"].height


def test_cross_sectional_momentum_trades_a_ragged_universe() -> None:
    data = _ragged()
    run = vector.run_backtest(
        CrossSectionalMomentum(lookback=60, skip_recent=5, rebal_freq=5), data
    )
    traded = set(run.fills["instrument_id"])
    assert {"A.SIM", "B.SIM", "C.SIM"} <= traded
    late = run.fills[run.fills["instrument_id"] == "B.SIM"]["ts_event"].min()
    assert late >= data.frames["B"]["ts_init"][0]


def test_delisted_symbol_is_flattened_on_its_last_bar() -> None:
    data = _ragged()
    targets = compute_targets(_Long(), data.frames)
    last_c = int(data.frames["C"]["ts_init"].dt.epoch("ns")[-1])
    assert targets["C"][last_c] == 0.0
    last_a = int(data.frames["A"]["ts_init"].dt.epoch("ns")[-1])
    assert targets["A"][last_a] == 1.0  # still listed at the end: untouched

    run = run_backtest(_Long(), data, initial_capital=1e8)
    fills_c = run.fills[run.fills["instrument_id"] == "C.SIM"]
    assert fills_c.iloc[-1]["order_side"].endswith("SELL")


def test_single_symbol_is_never_flattened() -> None:
    frame = daily_bars(_closes(100, 1))
    targets = compute_targets(_Long(), {"X": frame})
    assert set(targets["X"].values()) == {1.0}


# ── Config ────────────────────────────────────────────────────────────────────


def _cfg(**overrides: object) -> ReportConfig:
    raw: dict[str, object] = {
        "name": "unit",
        "title": "Unit test report",
        "strategy": "momentum-ts",
        "data": {"symbols": ["A", "B", "C", "D", "E"]},
        "evaluation": {"n_splits": 3, "test_size": 0.2, "min_train": 100, "commission_rate": 5e-4},
        "screening": {
            "n_trials": 3,
            "space": {"lookback": [20, 40, 60], "vol_lookback": [20], "rebal_freq": [1, 5]},
        },
        "validation": {"top_k": 2},
    }
    raw.update(overrides)
    return ReportConfig.model_validate(raw)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_example_configs_validate(path: Path) -> None:
    cfg = load_report_config(path)
    rp.strategy_class(cfg)  # registered
    assert cfg.theory_path.exists(), f"missing theory write-up {cfg.theory_path}"


def test_fingerprint_tracks_only_what_changes_results() -> None:
    base = _cfg()
    assert base.fingerprint() == _cfg(title="Renamed", question="Why?").fingerprint()
    assert base.fingerprint() != _cfg(strategy="trend-following").fingerprint()
    changed = _cfg(screening={"n_trials": 4, "space": {"lookback": [20]}})
    assert base.fingerprint() != changed.fingerprint()


def test_config_requires_symbols_and_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="symbols"):
        _cfg(data={})
    with pytest.raises(ValueError, match="Extra inputs"):
        _cfg(data={"symbols": ["A"], "simbols": ["B"]})


def test_universe_file_skips_comments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "idx.txt").write_text("# header — 3 tickers\nAAA\n\nBBB  # renamed\nCCC\nAAA\n")
    monkeypatch.setattr(report_config, "UNIVERSE_DIR", tmp_path)
    data = report_config.DataConfig.model_validate({"universe": "idx", "symbols": ["ZZZ"]})
    assert data.resolve_symbols() == ["ZZZ", "AAA", "BBB", "CCC"]
    assert data.resolve_symbols({"BBB", "CCC"}) == ["BBB", "CCC"]
    capped = report_config.DataConfig.model_validate({"universe": "idx", "max_symbols": 2})
    assert capped.resolve_symbols() == ["AAA", "BBB"]


def test_config_yaml_round_trips(tmp_path: Path) -> None:
    cfg = _cfg()
    path = tmp_path / "config.yaml"
    path.write_text(cfg.to_yaml())
    assert load_report_config(path) == cfg


# ── Scaffolding and CLI ───────────────────────────────────────────────────────


def test_new_report_copies_template_and_keeps_notes(tmp_path: Path) -> None:
    paths = new_report("tsmom-x", EXAMPLES[0], reports_dir=tmp_path)
    assert paths.notebook.read_text() == TEMPLATE.read_text()
    assert load_report_config(paths.config).name == "tsmom-x"
    paths.notes.write_text("my conclusions")
    with pytest.raises(FileExistsError):
        new_report("tsmom-x", EXAMPLES[0], reports_dir=tmp_path)
    new_report("tsmom-x", EXAMPLES[0], reports_dir=tmp_path, force=True)
    assert paths.notes.read_text() == "my conclusions"


def test_cli_new_and_list(tmp_path: Path) -> None:
    from snippy_scales.cli.main import app

    runner = CliRunner()
    result = runner.invoke(
        app, ["report", "new", "r1", "--from", str(EXAMPLES[0]), "--reports-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["report", "list", "--reports-dir", str(tmp_path)])
    assert result.exit_code == 0 and "r1" in result.output
    result = runner.invoke(app, ["report", "build", "nope", "--reports-dir", str(tmp_path)])
    assert result.exit_code == 1


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_report_instances_are_git_ignored() -> None:
    def ignored(path: str) -> bool:
        out = subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO_ROOT, check=False)
        return out.returncode == 0

    assert ignored("reports/some-report/report.html")
    assert ignored("reports/some-report/notes.md")
    assert not ignored("reports/templates/strategy_report.py")
    assert not ignored("reports/README.md")


def test_template_passes_marimo_check() -> None:
    pytest.importorskip("marimo")
    out = subprocess.run(
        [sys.executable, "-m", "marimo", "check", str(TEMPLATE)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stdout + out.stderr


# ── Pipeline ──────────────────────────────────────────────────────────────────


def test_pipeline_screens_validates_and_resumes(tmp_path: Path) -> None:
    cfg = _cfg()
    data = _ragged()
    paths = ReportPaths(tmp_path / "unit")
    paths.root.mkdir()

    screening = rp.run_screening(cfg, data, paths)
    assert screening.trials.height == 3 and screening.new_trials == 3
    assert {"duration_s", "mean_fold_sharpe", "fold_0", "fold_2", "p_lookback"} <= set(
        screening.trials.columns
    )

    validation = rp.run_validation(cfg, data, screening, paths)
    assert validation.trials.height == 2
    assert validation.report.n_trials == 3  # deflated for every screened trial
    comparison = rp.engine_comparison(screening, validation)
    assert comparison.height == 2
    assert np.abs(comparison["sharpe_diff"].to_numpy()).max() < 1e-3  # engines agree
    folds = rp.fold_comparison(screening, validation)
    assert folds.height == 2 * cfg.evaluation.n_splits

    # Re-running resumes: nothing new is run.
    again = rp.run_screening(cfg, data, paths)
    assert again.new_trials == 0 and again.trials.height == 3
    assert rp.run_validation(cfg, data, again, paths).new_trials == 0

    # A changed config refuses to reuse the stored trials.
    with pytest.raises(ValueError, match="different config"):
        rp.run_screening(_cfg(evaluation={"n_splits": 4}), data, paths)

    checks = rp.checks(validation, comparison, rp.daily_returns(data.frames["A"]))
    assert checks.schema["passed"] == pl.Boolean and checks.height == 7

    run = rp.full_backtest(cfg, data, validation.report.best_params)
    assert run.equity.height > 0 and run.rejections == ()


def test_calendar_respects_start_and_end() -> None:
    data = _ragged()
    cfg = _cfg(data={"symbols": ["A"], "start": "2020-03-01", "end": "2020-12-31"})
    cal = rp.calendar(cfg, data)
    first, last = cal.to_list()[0], cal.to_list()[-1]
    assert first >= dt.datetime(2020, 3, 1, tzinfo=dt.UTC)
    assert last <= dt.datetime(2020, 12, 31, tzinfo=dt.UTC)
    assert cal.len() < data.closes(union=True).len()


def test_summaries() -> None:
    data = _ragged()
    summary = rp.data_summary(data)
    assert summary.height == 5 and summary["zero_volume_bars"].sum() == 0
    assert rp.universe_size(data)["symbols"].max() == 5
    returns = rp.daily_returns(data.frames["A"])
    dd = rp.drawdown(returns)
    assert dd["drawdown"].to_numpy().max() <= 0.0
    years = rp.calendar_year_returns(returns)
    assert years["year"].to_list() == sorted(years["year"].to_list())
    assert rp.annualised_sharpe(returns) == pytest.approx(
        returns["return"].mean() / returns["return"].std() * np.sqrt(252)  # type: ignore[operator]
    )


def test_theory_and_space_tables() -> None:
    cfg = _cfg()
    theory = rp.theory_markdown(cfg)
    assert theory is not None and not theory.startswith("# ")
    lines = theory.splitlines()
    assert "### Signal" in lines and "## Signal" not in lines
    table = rp.space_table(cfg)
    rows = {r["parameter"]: r for r in table.iter_rows(named=True)}
    assert rows["lookback"]["search"] == "categorical"
    assert rows["max_leverage"]["search"] == "default"
    assert rp.theory_markdown(_cfg(theory="docs/strategies/missing.md")) is None


def test_environment_and_commands() -> None:
    env = rp.environment()
    assert {"git commit", "python", "nautilus_trader"} <= set(env)
    text = rp.commands(_cfg(), ReportPaths(Path("reports/unit")))
    assert "algo report build unit" in text
