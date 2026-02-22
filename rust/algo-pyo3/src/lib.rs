//! PyO3 bindings — exposes algo-core to Python via maturin.

use pyo3::prelude::*;

mod backtest;
mod stats;

/// Top-level module registered as `snippy_scales._algo_core`
#[pymodule]
fn algo_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    backtest::register(m)?;
    stats::register(m)?;
    Ok(())
}
