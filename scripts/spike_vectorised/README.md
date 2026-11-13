# Vectorised-engine spike (research code)

Throwaway code behind [`docs/research/vectorised_engines_spike.md`](../../docs/research/vectorised_engines_spike.md).
It is not part of the package, is not tested, and needs libraries the project does not
depend on, so it runs from a **scratch virtualenv**, never the project's `.venv`:

```bash
uv venv --python 3.12 /tmp/spike-venv
VIRTUAL_ENV=/tmp/spike-venv uv pip install -e . "vectorbt[rust]==1.2.0" "raptorbt==0.13.4" pytest
cd scripts/spike_vectorised
export PYTHONPATH=. SNIPPY_DATA_ROOT=/path/to/algotrading/data   # read-only
/tmp/spike-venv/bin/python accuracy.py accuracy.json
NUMBA_CACHE_DIR=$(mktemp -d) /tmp/spike-venv/bin/python speed.py speed.json
```

| File | What |
|---|---|
| `common.py` | `Prices`/`Panel` (dense closes and targets that replay `SignalStrategy`'s decisions), comparison and timing helpers |
| `engines.py` | Adapters: in-house numba `kernel`, vectorbt `from_orders` / `from_order_func`, raptorbt `PortfolioSession` |
| `fixtures.py` | AAPL, 8 equities, the 19 GLBX `*.c.0` series with stand-in contracts, synthetic rolled ES; `ample_volume` |
| `accuracy.py` | The accuracy matrix vs `execution.run_backtest` |
| `speed.py` | Single runs, 100-trial × 5-fold grid, batched variants, first-call JIT cost |
| `signals.py` | Target generation (signals, dict packaging, panel) vs kernel time, up to 400 equities |

Excluded from `ty` (`pyproject.toml`) because `vectorbt`, `raptorbt` and `numba` are not
project dependencies.
