"""End-to-end tests for ``algo eval`` and ``algo backtest`` on a temporary catalog."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from snippy_scales.cli import backtest as backtest_cli
from snippy_scales.cli import eval as eval_cli
from snippy_scales.data import catalog as cat
from tests.conftest import daily_bars, synthetic_future

_runner = CliRunner()


def _write_symbol(root: Path, symbol: str, seed: int, n: int = 500) -> None:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    catalog = cat.open_catalog(root)
    catalog.write_instruments([synthetic_future(symbol, venue="GLBX")])
    cat.write_bars(
        catalog,
        cat.frame_to_bars(
            daily_bars(close),
            bar_type=cat.bar_type(symbol, "GLBX", "ohlcv-1d"),
            price_precision=2,
            interval_ns=cat.bar_interval_ns("ohlcv-1d"),
        ),
    )


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Catalog with two symbols; studies and returns kept under tmp_path."""
    monkeypatch.setattr(eval_cli, "EXPERIMENTS_DIR", tmp_path / "experiments")
    for i, symbol in enumerate(("ES.c.0", "NQ.c.0")):
        _write_symbol(tmp_path / "catalog", symbol, seed=i)
    return {
        "catalog": str(tmp_path / "catalog"),
        "storage": f"sqlite:///{tmp_path / 'experiments' / 'optuna.db'}",
        "root": str(tmp_path),
    }


def _invoke(app: object, args: list[str]) -> str:
    result = _runner.invoke(app, args)  # ty: ignore[invalid-argument-type]
    assert result.exit_code == 0, result.output
    return result.output


def test_run_sweep_list_report_tearsheet(env: dict[str, str]) -> None:
    common = ["--catalog", env["catalog"], "--storage", env["storage"], "--min-train", "100"]
    out = _invoke(
        eval_cli.app, ["run", "trend-following", "-s", "ES.c.0", "--study", "base", *common]
    )
    assert "DSR" in out and "defaults" in out

    params = '{"fast_period": [5, 10], "slow_period": [40, 80]}'
    out = _invoke(
        eval_cli.app,
        [
            "sweep",
            "trend-following",
            "-s",
            "ES.c.0",
            "--params",
            params,
            "--search",
            "grid",
            "--study",
            "grid",
            *common,
        ],
    )
    assert "best of 4" in out
    assert "PBO" in out

    out = _invoke(eval_cli.app, ["list", "--storage", env["storage"]])
    assert "base" in out and "grid" in out

    out = _invoke(eval_cli.app, ["report", "grid", "--storage", env["storage"]])
    assert "fast_period" in out

    html = Path(env["root"]) / "t.html"
    _invoke(
        eval_cli.app,
        [
            "tearsheet",
            "grid",
            "--output",
            str(html),
            "--benchmark",
            "NQ.c.0",
            "--catalog",
            env["catalog"],
            "--storage",
            env["storage"],
        ],
    )
    assert html.stat().st_size > 10_000


def test_run_all_covers_single_and_multi_asset(env: dict[str, str]) -> None:
    out = _invoke(
        eval_cli.app,
        ["run-all", "--catalog", env["catalog"], "--storage", "memory", "--min-train", "100"],
    )
    assert "Summary" in out
    for name in ("trend-following", "momentum-ts", "mean-reversion", "momentum-cs"):
        assert name in out
    assert "2 symbols" in out


def test_run_all_without_data_fails(tmp_path: Path) -> None:
    result = _runner.invoke(eval_cli.app, ["run-all", "--catalog", str(tmp_path)])
    assert result.exit_code == 1
    assert "No 'ohlcv-1d' bars" in result.output


def test_unknown_strategy_and_bad_params(env: dict[str, str]) -> None:
    result = _runner.invoke(
        eval_cli.app, ["run", "nope", "-s", "ES.c.0", "--catalog", env["catalog"]]
    )
    assert result.exit_code == 1 and "Unknown strategy" in result.output
    result = _runner.invoke(
        eval_cli.app,
        [
            "run",
            "trend-following",
            "-s",
            "ES.c.0",
            "--params",
            '{"fast_period": [1, 2]}',
            "--catalog",
            env["catalog"],
            "--storage",
            "memory",
        ],
    )
    assert result.exit_code == 1 and "use `sweep`" in result.output


def test_backtest_run(env: dict[str, str]) -> None:
    out = _invoke(
        backtest_cli.app,
        [
            "-s",
            "ES.c.0",
            "-s",
            "NQ.c.0",
            "--catalog",
            env["catalog"],
            "--start",
            "2020-06-01",
            "--fees",
            "0.0005",
            "momentum-ts",
            "--params",
            '{"lookback": 60}',
        ],
    )
    assert "Sharpe Ratio (252 days)" in out
    assert "total return" in out


def test_vector_engine_from_the_cli(env: dict[str, str]) -> None:
    common = ["-s", "ES.c.0", "--catalog", env["catalog"], "--engine", "vector"]
    out = _invoke(backtest_cli.app, [*common, "trend-following"])
    assert "Sharpe Ratio (252 days)" in out
    out = _invoke(
        eval_cli.app,
        ["run", "trend-following", *common, "--storage", env["storage"], "--min-train", "100"],
    )
    assert "DSR" in out
    bad = _runner.invoke(backtest_cli.app, ["-s", "ES.c.0", "--engine", "gpu", "trend-following"])
    assert bad.exit_code != 0


def test_run_all_rolls_outrights_and_skips_undefined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Outrights become one ES.FUT symbol; series without definitions are skipped."""
    from tests.test_futures_rolls import _market

    monkeypatch.setattr(eval_cli, "EXPERIMENTS_DIR", tmp_path / "experiments")
    root = tmp_path / "catalog"
    catalog = cat.open_catalog(root)
    contracts, frames = _market()
    catalog.write_instruments(contracts)
    for contract in contracts:
        cat.write_bars(
            catalog,
            cat.frame_to_bars(
                frames[str(contract.id)],
                bar_type=cat.bar_type(str(contract.raw_symbol), "GLBX", "ohlcv-1d"),
                price_precision=2,
                interval_ns=cat.bar_interval_ns("ohlcv-1d"),
            ),
        )
    _write_symbol(root, "NQ.c.0", seed=3, n=700)
    # A continuous series with bars but no instrument definition.
    cat.write_bars(
        catalog,
        cat.frame_to_bars(
            daily_bars(np.linspace(50, 60, 300)),
            bar_type=cat.bar_type("CL.c.0", "GLBX", "ohlcv-1d"),
            price_precision=2,
            interval_ns=cat.bar_interval_ns("ohlcv-1d"),
        ),
    )
    out = _invoke(
        eval_cli.app,
        ["run-all", "--catalog", str(root), "--storage", "memory", "--min-train", "100"],
    )
    assert "Discovered 3 symbol(s)" in out
    assert "skip CL.c.0" in out
    assert "ES.FUT" in out and "ESH0" not in out
    assert "2 symbols" in out  # momentum-cs on ES.FUT + NQ.c.0


def test_backtest_parent_symbol(tmp_path: Path) -> None:
    from tests.test_futures_rolls import _market

    root = tmp_path / "catalog"
    catalog = cat.open_catalog(root)
    contracts, frames = _market()
    catalog.write_instruments(contracts)
    for contract in contracts:
        cat.write_bars(
            catalog,
            cat.frame_to_bars(
                frames[str(contract.id)],
                bar_type=cat.bar_type(str(contract.raw_symbol), "GLBX", "ohlcv-1d"),
                price_precision=2,
                interval_ns=cat.bar_interval_ns("ohlcv-1d"),
            ),
        )
    out = _invoke(
        backtest_cli.app,
        [
            "-s",
            "ES.FUT",
            "--catalog",
            str(root),
            "--per-contract",
            "2.25",
            "--roll",
            "calendar",
            "--roll-months",
            "HMUZ",
            "trend-following",
        ],
    )
    assert "rolls" in out
